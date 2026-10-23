"""SAT-based passive learner of separating deterministic finite automata.

All labelled words are put in an observation tree up front; words without a label are
don't-care. :meth:`ObservationTreeSquare.learn` then finds a smallest DFA into which the
tree maps:

1. Promote nodes that are apart from the whole basis, breadth-first, until none is left.
   The basis nodes are pairwise apart, so they need distinct states, and the size of the basis
   is a lower bound on the number of states.
2. Restrict, for every informative tree node, the hypothesis states it may map to (its
   *mapping domain*), using basis compatibility, known outputs and frontier/basis apartness.
3. Encode "a DFA with ``size`` states into which the tree maps" as CNF
   (:class:`~.DfaEncoding.DfaEncoding`) and solve it with CaDiCaL. If the CNF is
   unsatisfiable the state bound is increased and the search repeats.

``frontier_to_basis_dict`` maps non-basis nodes to the basis nodes they are not apart from.
Basis nodes never have an entry.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from aalpy.automata import Dfa, DfaState
from pysat.solvers import Cadical195

from Apartness import Apartness
from DCNode import DCNode
from DCValue import DCValue
from DfaEncoding import DfaEncoding, Edge, HypothesisMapping

logger = logging.getLogger(__name__)

# CaDiCaL options for the hypothesis search: run in stable mode only (target phases and
# reluctant restarts), which favours finding satisfying assignments.
_SOLVER_OPTIONS = {"stabilizeonly": 1}


class ObservationTreeSquare:
    """Learn a smallest separating DFA from a set of labelled words.

    Args:
        alphabet: Input symbols of the words; assumed to contain at least two symbols.
        words: Input words paired with their output: ``True`` for accepted and ``False`` for
            rejected. All other words are don't-care.

    Raises:
        ValueError: If a word occurs with two different outputs.

    Attributes:
        alphabet: Input alphabet used by the learner.
        solver_time: Accumulated time spent building and solving SAT instances.
        root: Root node of the observation tree.
        size: Current DFA state bound.
        guaranteed_basis: Basis nodes, which are pairwise apart. Basis node ``i`` is
            hypothesis state ``i``.
        frontier_to_basis_dict: Basis nodes each non-basis observation-tree node is not apart from.
    """

    def __init__(self, alphabet: Sequence[Any], words: Iterable[tuple[Sequence[Any], bool]]) -> None:
        self.alphabet = list(alphabet)
        self.solver_time = 0.0
        DCNode._id_counter = 0
        self.root = DCNode()
        self.root.set_output(DCValue.DC)
        self.size = 1
        self.guaranteed_basis = [self.root]
        self._basis_set = {self.root}
        self.frontier_to_basis_dict: dict[DCNode, set[DCNode]] = {}
        self._alphabet_index = {letter: index for index, letter in enumerate(self.alphabet)}

        for word, accepts in words:
            self._add_word(word, DCValue.TRUE if accepts else DCValue.FALSE)
        self._init_frontier_to_basis_dict()

        logger.debug("Initialized learner: alphabet_size=%d, nodes=%d", len(self.alphabet),
                     sum(1 for _ in self._iter_nodes()))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def learn(self) -> Dfa:
        """Return a DFA with the fewest states that agrees with all labelled words."""
        self._promote_all()
        while True:
            hypothesis_mapping = self._find_hypothesis()
            if hypothesis_mapping is not None:
                break
            self.size += 1
            logger.debug("No hypothesis at current bound; increasing state_bound to %d", self.size)

        transition_mapping, output_mapping = hypothesis_mapping
        hypothesis = self._construct_hypothesis(transition_mapping, output_mapping)
        logger.debug("Hypothesis constructed: states=%d, accepting_states=%d", len(output_mapping),
                     sum(output_mapping))
        return hypothesis

    # ------------------------------------------------------------------
    # Observation tree primitives
    # ------------------------------------------------------------------

    def _add_word(self, word: Sequence[Any], output: DCValue) -> None:
        """Add ``word`` with ``output`` to the tree; prefixes without a label of their own stay don't-care."""
        node = self.root
        for input_value in word:
            node = node.extend_and_get(input_value, DCValue.DC)
        if node.has_known_output() and node.output is not output:
            raise ValueError(f"Word {tuple(word)} has conflicting outputs")
        node.set_output(output)

    def _iter_nodes(self) -> Iterator[DCNode]:
        """Yield all observation-tree nodes in breadth-first order."""
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            queue.extend(node.successors.values())
            yield node

    # ------------------------------------------------------------------
    # Basis
    # ------------------------------------------------------------------

    def _promote_all(self) -> None:
        """Promote the first node apart from the whole basis until there is none."""
        while (node := self._find_next_basis_node()) is not None:
            self._promote_node_to_basis(node)
        logger.debug("Basis ready: basis_size=%d, state_bound=%d", len(self.guaranteed_basis), self.size)

    def _find_next_basis_node(self) -> DCNode | None:
        """Find the first observation-tree node, breadth-first, that is apart from the current basis."""
        for candidate in self._iter_nodes():
            if (candidate not in self._basis_set and candidate.leads_to_known
                    and all(Apartness.states_are_apart(candidate, basis_node, self)
                            for basis_node in self.guaranteed_basis)):
                return candidate
        return None

    def _promote_node_to_basis(self, node: DCNode) -> None:
        """Add ``node`` to the basis and drop it from the other nodes' candidates."""
        self.guaranteed_basis.append(node)
        self._basis_set.add(node)
        self.size = max(self.size, len(self.guaranteed_basis))
        del self.frontier_to_basis_dict[node]
        for other, candidates in self.frontier_to_basis_dict.items():
            if not Apartness.states_are_apart(other, node, self):
                candidates.add(node)

        logger.debug("Promoted basis node: basis_size=%d, state_bound=%d, access_sequence=%s",
                     len(self.guaranteed_basis), self.size, node.access_sequence)

    def _init_frontier_to_basis_dict(self) -> None:
        """Give every non-root node the root as candidate unless it is apart from it."""
        self.frontier_to_basis_dict = {
            node: ({self.root} if not Apartness.states_are_apart(node, self.root, self) else set())
            for node in self._iter_nodes() if node is not self.root}

    # ------------------------------------------------------------------
    # Hypothesis construction
    # ------------------------------------------------------------------

    def _find_hypothesis(self) -> HypothesisMapping | None:
        """Search for a DFA with ``self.size`` states consistent with the tree; ``None`` if there is none."""
        start_time = time.perf_counter()
        try:
            nodes, edges = self._flatten_observation_tree()
            logger.debug("Building SAT hypothesis: state_bound=%d, basis_size=%d, observation_nodes=%d, edges=%d",
                         self.size, len(self.guaranteed_basis), len(nodes), len(edges))

            mapping_domains = self._compute_mapping_domains(nodes)
            if not all(mapping_domains):
                logger.debug("SAT setup rejected: observation nodes have empty mapping domains")
                return None
            if not self._prepare_sat_domains(nodes, mapping_domains):
                logger.debug("SAT setup rejected while applying output/apartness constraints")
                return None

            encoding = DfaEncoding(self.size, len(self.guaranteed_basis), len(self.alphabet))
            clauses = encoding.encode_observations(nodes, edges, mapping_domains)
            logger.debug("SAT instance ready: variables=%d, clauses=%d", encoding.num_vars, len(clauses))
            solver = Cadical195()
            try:
                solver.configure(_SOLVER_OPTIONS)  # Options must be set before any clause is added.
                solver.append_formula(clauses)
                result = solver.solve()
                if result is not True and result is not False:
                    raise RuntimeError(f"Unexpected CaDiCaL result: {result!r}.")
                logger.debug("SAT solver: %s (elapsed=%.3fs)", "SAT" if result else "UNSAT",
                             time.perf_counter() - start_time)
                return encoding.decode(solver.get_model()) if result else None
            finally:
                solver.delete()
        finally:
            elapsed = time.perf_counter() - start_time
            self.solver_time += elapsed
            logger.debug("Hypothesis search finished: elapsed=%.3fs, cumulative_solver_time=%.3fs", elapsed,
                         self.solver_time)

    def _flatten_observation_tree(self) -> tuple[list[DCNode], list[Edge]]:
        """Index the root and all informative nodes breadth-first, with the edges between them."""
        nodes = [self.root]
        edges: list[Edge] = []
        for source_index, node in enumerate(nodes):  # ``nodes`` grows while iterating: a BFS queue.
            for input_value, successor in node.successors.items():
                if successor.leads_to_known:
                    edges.append((source_index, len(nodes), self._alphabet_index[input_value]))
                    nodes.append(successor)
        return nodes, edges

    def _compute_mapping_domains(self, nodes: Sequence[DCNode]) -> list[set[int]]:
        """States each node may map to: its own state for basis nodes, else compatible basis or free states."""
        basis_index = {basis_node: state for state, basis_node in enumerate(self.guaranteed_basis)}
        free_states = range(len(self.guaranteed_basis), self.size)
        domains: list[set[int]] = []
        for node in nodes:
            if node in basis_index:
                domains.append({basis_index[node]})
                continue
            domains.append({basis_index[candidate] for candidate in self.frontier_to_basis_dict[node]}
                           .union(free_states))
        return domains

    def _prepare_sat_domains(self, nodes: Sequence[DCNode], mapping_domains: list[set[int]]) -> bool:
        """Prune domains by forced state outputs and frontier/basis apartness; ``False`` if one becomes empty."""
        known_nodes = [(index, node.output is DCValue.TRUE) for index, node in enumerate(nodes)
                       if node.has_known_output()]

        # Known nodes with a singleton domain fix the output of their state.
        forced_output: list[bool | None] = [None] * self.size
        for node_index, accepts in known_nodes:
            domain = mapping_domains[node_index]
            if len(domain) != 1:
                continue
            state = next(iter(domain))
            if forced_output[state] is not None and forced_output[state] != accepts:
                return False
            forced_output[state] = accepts

        for node_index, accepts in known_nodes:
            domain = mapping_domains[node_index]
            domain.difference_update([state for state in domain
                                      if forced_output[state] is not None and forced_output[state] != accepts])
            if not domain:
                return False

        # Only apartness between frontier nodes and basis nodes is needed here.
        # Basis nodes are already mutually apart, and arbitrary observation-node
        # pairs needlessly create an O(n^2) Cartesian product. A basis node's
        # domain is the singleton holding its own state.
        node_index = {node: index for index, node in enumerate(nodes)}
        basis_states = [(state, basis_node) for state, basis_node in enumerate(self.guaranteed_basis)
                        if basis_node in node_index]
        for index, node in enumerate(nodes):
            if node in self._basis_set:
                continue
            frontier_domain = mapping_domains[index]
            for state, basis_node in basis_states:
                if state in frontier_domain and Apartness.states_are_apart(node, basis_node, self):
                    frontier_domain.discard(state)
                    if not frontier_domain:
                        return False
        return True

    def _construct_hypothesis(self, transition_mapping: Sequence[Sequence[int]],
                              output_mapping: Sequence[bool]) -> Dfa:
        """Build an AALpy DFA with states ``s0 .. s{size-1}``, starting in ``s0``."""
        states = [DfaState(f"s{index}") for index in range(self.size)]
        for state, accepting, targets in zip(states, output_mapping, transition_mapping):
            state.is_accepting = accepting
            for input_value, target_index in zip(self.alphabet, targets):
                state.transitions[input_value] = states[target_index]
        hypothesis = Dfa(states[0], states)
        hypothesis.compute_prefixes()
        hypothesis.characterization_set = hypothesis.compute_characterization_set(raise_warning=False)
        return hypothesis

"""SAT-based observation-tree learner for separating deterministic finite automata.

Each call to :meth:`ObservationTreeSquare.build_hypothesis` runs one round:

1. Make the observation tree adequate: expand the frontier below every basis
   node, promote nodes that are apart from the whole basis, and query
   distinguishing suffixes until frontier nodes are identified.
2. Restrict, for every informative tree node, the hypothesis states it may
   map to (its *mapping domain*), using basis compatibility, known outputs
   and frontier/basis apartness.
3. Encode "a DFA with ``size`` states into which the tree maps" as CNF
   (:class:`~.DfaEncoding.DfaEncoding`) and solve it with CaDiCaL, steering the
   search towards the previous hypothesis while the basis is unchanged. The
   encoding and solver are kept while the state bound and basis are unchanged,
   so each round only adds the clauses for new observations. If the CNF is
   unsatisfiable the state bound is increased for the next round.

``frontier_to_basis_dict`` maps non-basis nodes to the basis nodes they have
not yet been shown to be apart from. Since apartness is monotone in the tree,
each entry is a superset of the basis nodes the node is truly compatible with.
Basis nodes never have an entry, so they are never treated as frontier nodes.
"""

from __future__ import annotations

import itertools
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

# Frontier expansion queries all words of the largest length whose count stays
# below this bound after each basis node; see ``ObservationTreeSquare._expand_frontier``.
_FRONTIER_EXPANSION_WORDS = 1000

# CaDiCaL options for the hypothesis search: run in stable mode only (target phases and
# reluctant restarts), which favours finding satisfying assignments.
_SOLVER_OPTIONS = {"stabilizeonly": 1}


class ObservationTreeSquare:
    """Learn a separating DFA from an incomplete observation tree.

    The learner incrementally queries a system under learning (SUL), maintains
    an apartness-aware basis, and encodes the resulting DFA construction as a
    SAT problem.

    Args:
        alphabet: Input symbols understood by the SUL; assumed to contain at
            least two symbols.
        sul: System-under-learning object exposing a ``query`` method.
        assume_prefix_closed: Whether an unknown output implies that longer
            prefixes are also unknown.

    Attributes:
        automaton_type: The type of automaton produced, always ``"dfa"``.
        alphabet: Input alphabet used by the learner.
        sul: System under learning queried for observations.
        assume_prefix_closed: Whether unknown outputs are treated as
            prefix-closed.
        solver_time: Accumulated time spent building and solving SAT
            instances.
        root: Root node of the observation tree.
        size: Current DFA state bound.
        guaranteed_basis: Current basis nodes that must remain represented.
            Basis node ``i`` is hypothesis state ``i``. Only grows through
            promotion, which keeps it in sync with an internal set.
        frontier_to_basis_dict: Compatible basis candidates for each
            non-basis observation-tree node.
    """

    def __init__(self, alphabet: Sequence[Any], sul: Any, assume_prefix_closed: bool = True) -> None:
        self.automaton_type = "dfa"
        self.alphabet = list(alphabet)
        self.sul = sul
        self.assume_prefix_closed = assume_prefix_closed
        self.solver_time = 0.0
        DCNode._id_counter = 0
        self.root = DCNode()
        self.root.set_output(self.sul.query(()))
        self.size = 1
        self.guaranteed_basis = [self.root]
        self._basis_set = {self.root}
        self.frontier_to_basis_dict: dict[DCNode, set[DCNode]] = {}
        self._alphabet_index = {letter: index for index, letter in enumerate(self.alphabet)}
        # Frontier expansion is only needed after the state bound changes or
        # after a new basis node is promoted.
        self._last_expanded_state: tuple[int, int] | None = None
        # The last hypothesis found and the basis size it was found for. While the basis is
        # unchanged it provides the solver's decision phases, as the next hypothesis mostly
        # differs from it only where the new observations contradict it.
        self._last_hypothesis: tuple[int, HypothesisMapping] | None = None
        # The encoding and solver of the current (state bound, basis size), reused across rounds.
        self._sat_session: tuple[tuple[int, int], DfaEncoding, Cadical195] | None = None

        logger.debug("Initialized learner: alphabet_size=%d, initial_output=%s, state_bound=%d",
                     len(self.alphabet), self.root.output, self.size)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_hypothesis(self) -> Dfa | None:
        """Build a DFA hypothesis from the current observations.

        Returns:
            A DFA consistent with the current observation tree, or ``None`` if
            the current state bound is insufficient. When no hypothesis exists
            at the current bound, the state bound is increased for the next
            call.
        """
        logger.debug("=== Building hypothesis: state_bound=%d, basis_size=%d ===", self.size,
                     len(self.guaranteed_basis))

        self._find_adequate_observation_tree()
        hypothesis_mapping = self._find_hypothesis()
        if hypothesis_mapping is None:
            self.size += 1
            logger.debug("No hypothesis at current bound; increasing state_bound to %d", self.size)
            return None

        transition_mapping, output_mapping = hypothesis_mapping
        hypothesis = self._construct_hypothesis(transition_mapping, output_mapping)
        logger.debug("Hypothesis constructed: states=%d, accepting_states=%d", len(output_mapping),
                     sum(output_mapping))
        return hypothesis

    def process_counter_example(self, counterexample: Sequence[Any], output: bool | DCValue) -> None:
        """Add a counterexample and its observed output to the tree.

        Args:
            counterexample: Input sequence that distinguishes the current
                hypothesis from the SUL.
            output: Observed output for the counterexample, given as a boolean
                or as a ``DCValue``.
        """
        logger.debug("Processing counterexample: length=%d, reported_output=%s", len(counterexample), output)

        node, _ = self._observe(counterexample)
        if isinstance(output, bool):
            output = DCValue.TRUE if output else DCValue.FALSE
        node.set_output(output)
        self._update_frontier_to_basis_dict()

        logger.debug("Counterexample incorporated: final_output=%s, basis_size=%d, state_bound=%d", output,
                     len(self.guaranteed_basis), self.size)

    # ------------------------------------------------------------------
    # Observation tree primitives
    # ------------------------------------------------------------------

    def _observe(self, inputs: Sequence[Any]) -> tuple[DCNode, bool]:
        """Add ``inputs`` to the tree, querying the SUL for every missing prefix.

        Every non-basis node on the path that has no candidate entry yet gets
        the full basis as its candidates.

        Returns:
            The node reached by ``inputs`` and whether any newly queried
            output was known. With ``assume_prefix_closed``, an unknown output
            marks all missing longer prefixes as unknown without querying them.
        """
        node = self.root
        queried_known_output = False
        rest_unknown = False
        for index, input_value in enumerate(inputs):
            successor = node.successors.get(input_value)
            if successor is None:
                if rest_unknown:
                    output = DCValue.DC
                else:
                    output = self.sul.query(tuple(inputs[:index + 1]))
                    if output.is_known():
                        queried_known_output = True
                    elif self.assume_prefix_closed:
                        rest_unknown = True
                successor = node.extend_and_get(input_value, output)
            elif self.assume_prefix_closed and successor.output is DCValue.DC:
                rest_unknown = True
            node = successor
            if node not in self.frontier_to_basis_dict and node not in self._basis_set:
                self.frontier_to_basis_dict[node] = set(self.guaranteed_basis)
        return node, queried_known_output

    def _get_successor(self, inputs: Sequence[Any], start_node: DCNode | None = None) -> DCNode | None:
        """Follow ``inputs`` from ``start_node`` (default: the root) without extending the tree."""
        node = self.root if start_node is None else start_node
        for input_value in inputs:
            node = node.successors.get(input_value)
            if node is None:
                return None
        return node

    def _iter_nodes(self) -> Iterator[DCNode]:
        """Yield all observation-tree nodes in breadth-first order."""
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            queue.extend(node.successors.values())
            yield node

    def _count_candidate_nodes(self) -> int:
        """Count basis and informative nodes; only used for debug logging."""
        return sum(1 for node in self._iter_nodes() if node in self._basis_set or node.leads_to_known)

    # ------------------------------------------------------------------
    # Adequate observation tree
    # ------------------------------------------------------------------

    def _find_adequate_observation_tree(self) -> None:
        """Expand, promote and identify until no frontier node can be identified further."""
        logger.debug("Searching for adequate observation tree: state_bound=%d, basis_size=%d", self.size,
                     len(self.guaranteed_basis))

        if (self.size, len(self.guaranteed_basis)) != self._last_expanded_state:
            self._expand_frontier()
        self._update_frontier_to_basis_dict()

        promotions = self._promote_and_expand()
        identification_rounds = 0
        while self._make_frontiers_identified():
            identification_rounds += 1
            self._update_frontier_to_basis_dict()
            promotions += self._promote_and_expand()

        logger.debug("Observation tree ready: basis_size=%d, promotions=%d, identification_rounds=%d",
                     len(self.guaranteed_basis), promotions, identification_rounds)

    def _expand_frontier(self) -> None:
        """Query every word of a fixed length after each basis access sequence.

        The length is the smallest ``n`` with ``|alphabet|^(n + 1) >= 1000``.
        """
        length = 0
        while len(self.alphabet) ** (length + 1) < _FRONTIER_EXPANSION_WORDS:
            length += 1
        debug = logger.isEnabledFor(logging.DEBUG)
        nodes_before = self._count_candidate_nodes() if debug else 0

        basis_access_sequences = [node.access_sequence for node in self.guaranteed_basis]
        for word in itertools.product(self.alphabet, repeat=length):
            word = list(word)
            for access_sequence in basis_access_sequences:
                self._observe(access_sequence + word)

        self._last_expanded_state = (self.size, len(self.guaranteed_basis))
        if debug:
            logger.debug("Expanded observation frontier: depth=%d, basis_size=%d, candidate_nodes=%d -> %d", length,
                         len(self.guaranteed_basis), nodes_before, self._count_candidate_nodes())

    def _promote_and_expand(self) -> int:
        """Promote basis nodes until none is apart from the basis, expanding after each promotion.

        Returns:
            The number of promoted nodes.
        """
        promotions = 0
        while self._promote_node_to_basis():
            promotions += 1
            # A newly promoted basis node changes the set of access sequences,
            # so expand immediately after that basis change.
            self._expand_frontier()
            self._update_frontier_to_basis_dict()
        return promotions

    def _promote_node_to_basis(self) -> bool:
        """Add the first node apart from the whole basis to it; ``False`` if there is none."""
        next_basis_node = self._find_next_basis_node()
        if next_basis_node is None:
            return False

        self.guaranteed_basis.append(next_basis_node)
        self._basis_set.add(next_basis_node)
        self.size = max(self.size, len(self.guaranteed_basis))
        self._rebuild_frontier_to_basis_dict()

        logger.debug("Promoted basis node: basis_size=%d, state_bound=%d, access_sequence=%s",
                     len(self.guaranteed_basis), self.size, next_basis_node.access_sequence)
        return True

    def _find_next_basis_node(self) -> DCNode | None:
        """Find the first observation-tree node, breadth-first, that is apart from the current basis."""
        for candidate in self._iter_nodes():
            if (candidate not in self._basis_set and candidate.leads_to_known
                    and all(Apartness.states_are_apart(candidate, basis_node, self)
                            for basis_node in self.guaranteed_basis)):
                return candidate
        return None

    def _make_frontiers_identified(self) -> bool:
        """Try to identify every non-basis successor of every basis node; ``True`` if the tree was extended."""
        extended = False
        identified_frontiers = 0
        for basis_node in self.guaranteed_basis:
            for input_value in self.alphabet:
                frontier_node = basis_node.successors.get(input_value)
                if frontier_node is None or frontier_node in self._basis_set:
                    continue
                while self._identify_frontier(frontier_node):
                    extended = True
                    identified_frontiers += 1
                    self._update_basis_candidates(frontier_node)

        if identified_frontiers:
            logger.debug("Identified frontier nodes: extensions=%d, basis_size=%d", identified_frontiers,
                         len(self.guaranteed_basis))
        return extended

    def _identify_frontier(self, frontier_node: DCNode) -> bool:
        """Query distinguishing suffixes of the frontier's candidates until one yields a known output."""
        candidates = self.frontier_to_basis_dict.get(frontier_node)
        if not candidates:
            return False
        # The generator walks the tree lazily while it is being extended below.
        for witness_sequence in Apartness.get_distinguishing_sequences(list(candidates), self):
            target = self._get_successor(witness_sequence, start_node=frontier_node)
            if target is not None and target.output is not None:
                continue
            _, extended = self._observe(frontier_node.access_sequence + witness_sequence)
            if extended:
                return True
        return False

    # ------------------------------------------------------------------
    # Basis candidates
    # ------------------------------------------------------------------

    def _non_apart_basis_nodes(self, node: DCNode, basis: Iterable[DCNode]) -> set[DCNode]:
        return {basis_node for basis_node in basis if not Apartness.states_are_apart(node, basis_node, self)}

    def _update_basis_candidates(self, frontier_node: DCNode) -> None:
        """Drop the candidates of ``frontier_node`` that it is now apart from."""
        candidates = self.frontier_to_basis_dict.get(frontier_node)
        if candidates is not None:
            self.frontier_to_basis_dict[frontier_node] = self._non_apart_basis_nodes(frontier_node, candidates)

    def _update_frontier_to_basis_dict(self) -> None:
        """Refresh candidates below the root, skipping subtrees of identified or uninformative nodes."""
        stack = [self.root]
        while stack:
            node = stack.pop()
            if not node.leads_to_known:
                continue
            if node not in self._basis_set:
                self._update_basis_candidates(node)
                if not self.frontier_to_basis_dict.get(node):
                    continue
            stack.extend(node.successors.values())

    def _rebuild_frontier_to_basis_dict(self) -> None:
        """Recompute the candidates of every non-basis node against the whole basis."""
        self.frontier_to_basis_dict = {node: self._non_apart_basis_nodes(node, self.guaranteed_basis)
                                       for node in self._iter_nodes() if node not in self._basis_set}

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
            empty_mapping_domains = sum(not domain for domain in mapping_domains)
            if empty_mapping_domains:
                logger.debug("SAT setup rejected: %d observation nodes have empty mapping domains",
                             empty_mapping_domains)
                self._close_sat_session()
                return None
            if not self._prepare_sat_domains(nodes, mapping_domains):
                logger.debug("SAT setup rejected while applying output/apartness constraints")
                self._close_sat_session()
                return None

            encoding, solver = self._get_sat_session()
            new_clauses = encoding.add_observations(nodes, edges, mapping_domains)
            solver.append_formula(new_clauses)
            logger.debug("SAT instance ready: variables=%d, new_clauses=%d", encoding.num_vars, len(new_clauses))
            if self._last_hypothesis is not None and self._last_hypothesis[0] == len(self.guaranteed_basis):
                # Phases only steer the search, so the outcome is the same with or without them.
                solver.set_phases(encoding.phases(self._last_hypothesis[1]))
            result = solver.solve()
            if result is False:
                logger.debug("SAT solver: UNSAT (elapsed=%.3fs)", time.perf_counter() - start_time)
                self._close_sat_session()
                return None
            if result is not True:
                raise RuntimeError(f"Unexpected CaDiCaL result: {result!r}.")

            logger.debug("SAT solver: SAT (elapsed=%.3fs)", time.perf_counter() - start_time)
            hypothesis_mapping = encoding.decode(solver.get_model())
            self._last_hypothesis = (len(self.guaranteed_basis), hypothesis_mapping)
            return hypothesis_mapping
        finally:
            elapsed = time.perf_counter() - start_time
            self.solver_time += elapsed
            logger.debug("Hypothesis search finished: elapsed=%.3fs, cumulative_solver_time=%.3fs", elapsed,
                         self.solver_time)

    def _get_sat_session(self) -> tuple[DfaEncoding, Cadical195]:
        """Return the encoding and solver for the current state bound and basis, creating them if needed."""
        key = (self.size, len(self.guaranteed_basis))
        if self._sat_session is None or self._sat_session[0] != key:
            self._close_sat_session()
            solver = Cadical195()
            solver.configure(_SOLVER_OPTIONS)  # Options must be set before any clause is added.
            self._sat_session = (key, DfaEncoding(self.size, len(self.guaranteed_basis), len(self.alphabet)), solver)
        return self._sat_session[1], self._sat_session[2]

    def _close_sat_session(self) -> None:
        """Release the solver of the current session; the next search starts a new one."""
        if self._sat_session is not None:
            self._sat_session[2].delete()
            self._sat_session = None

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
        all_states = range(self.size)
        free_states = range(len(self.guaranteed_basis), self.size)
        domains: list[set[int]] = []
        for node in nodes:
            if node in basis_index:
                domains.append({basis_index[node]})
                continue
            candidates = self.frontier_to_basis_dict.get(node)
            if candidates is None:
                domains.append(set(all_states))
            else:
                domains.append({basis_index[candidate] for candidate in candidates}.union(free_states))
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
            if node not in self.frontier_to_basis_dict:  # Basis nodes have no entry.
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

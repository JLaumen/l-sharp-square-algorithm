"""Incremental CNF encoding of "a DFA with a given number of states into which the observation tree maps"."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence

from DCNode import DCNode
from DCValue import DCValue

# (source node index, target node index, letter index) of an observation-tree edge.
Edge = tuple[int, int, int]
# Target state of each (state, letter), and whether each state is accepting.
HypothesisMapping = tuple[list[list[int]], list[bool]]

# Up to this many literals, at-most-one is encoded pairwise, which propagates better than a
# sequential counter and keeps hypothesis searches considerably faster at typical state bounds.
_PAIRWISE_AT_MOST_ONE_LIMIT = 32


class CnfBuilder:
    """Collects clauses and hands out fresh SAT variables."""

    def __init__(self) -> None:
        self.clauses: list[list[int]] = []
        self.num_vars = 0

    def new_var(self) -> int:
        self.num_vars += 1
        return self.num_vars

    def new_vars(self, count: int) -> list[int]:
        start = self.num_vars + 1
        self.num_vars += count
        return list(range(start, start + count))

    def take_clauses(self) -> list[list[int]]:
        """Return the clauses collected so far and start collecting anew."""
        clauses, self.clauses = self.clauses, []
        return clauses

    def exactly_one(self, literals: Sequence[int]) -> None:
        """At least one literal, plus pairwise or (for many literals) sequential-counter at-most-one."""
        clauses = self.clauses
        clauses.append(list(literals))
        count = len(literals)
        if count <= 1:
            return
        if count <= _PAIRWISE_AT_MOST_ONE_LIMIT:
            clauses.extend([-literals[left], -literals[right]]
                           for left in range(count) for right in range(left + 1, count))
            return
        # Sinz's sequential counter: ``prefix[i]`` holds if one of ``literals[0..i]`` does.
        prefix = self.new_vars(count - 1)
        clauses.append([-literals[0], prefix[0]])
        for index in range(1, count - 1):
            clauses.append([-prefix[index - 1], prefix[index]])
            clauses.append([-literals[index], -prefix[index - 1]])
            clauses.append([-literals[index], prefix[index]])
        clauses.append([-literals[-1], -prefix[-1]])


class DfaEncoding:
    """Incremental CNF for a DFA with ``number_of_states`` states into which the observation tree maps.

    One encoding serves every hypothesis search for a fixed state bound and basis. Each call of
    :meth:`add_observations` returns only the clauses for what is new since the previous call, so a
    solver that keeps all returned clauses also keeps what it learned. This is sound because the
    tree only grows: known outputs never change and mapping domains only shrink.

    Variables:

    * ``map(u, s)``: tree node ``u`` maps to state ``s``, for ``s`` in the mapping domain of ``u``
      when ``u`` was added;
    * ``delta(s, a, t)``: the ``a``-transition of ``s`` goes to ``t``, for every state ``t``;
    * ``accepting(s)``: state ``s`` is accepting;
    * auxiliary variables of the at-most-one, output-signature and symmetry-breaking constraints.

    Basis state ``i`` is fixed by the basis node ``i`` mapping to it. The other states are numbered
    by value precedence: a node may only map to such a state ``s`` if an earlier node maps to
    ``s - 1``. Any consistent DFA can be renumbered this way, by numbering its non-basis states in
    the order of the first node mapping to them and putting states no node maps to last, so
    satisfiability is unchanged. Nodes are ordered as they were added; the first call adds them by
    decreasing number of nodes they are apart from, since the earliest nodes restrict the search
    the most.

    Args:
        number_of_states: Number of hypothesis states.
        basis_size: Number of basis states, which come first and are fixed.
        alphabet_size: Number of input letters.
    """

    def __init__(self, number_of_states: int, basis_size: int, alphabet_size: int) -> None:
        self._cnf = CnfBuilder()
        self.number_of_states = number_of_states
        self.basis_size = basis_size
        self.alphabet_size = alphabet_size
        all_states = range(number_of_states)
        self.transition_variables = [[{target: self._cnf.new_var() for target in all_states}
                                      for _ in range(alphabet_size)] for _ in all_states]
        self.output_variables = self._cnf.new_vars(number_of_states)
        for row in self.transition_variables:
            for transition in row:
                self._cnf.exactly_one(list(transition.values()))

        # Encoded nodes in precedence order and with parents before children, with their variables,
        # the states excluded since, and their parent as (node, letter index).
        self._nodes: list[DCNode] = []
        self._tree_order: list[DCNode] = []
        self.mapping_variables: dict[DCNode, dict[int, int]] = {}
        self._excluded: dict[DCNode, set[int]] = {}
        self._parents: dict[DCNode, tuple[DCNode, int]] = {}
        self._known_nodes: set[DCNode] = set()

        # Suffixes form a trie of reversed words: suffix ``i`` is ``letter + suffixes[rest]``, with
        # ``rest == -1`` for the empty word. Per suffix: (ancestor, sign) for every known node
        # ``ancestor + suffix``, the masks of states that accepting and rejecting such ancestors may
        # map to, and the signature variable of each state.
        self._suffix_index: dict[tuple[int, int], int] = {}
        self._suffixes: list[tuple[int, int]] = []
        self._occurrences: list[list[tuple[DCNode, int]]] = []
        self._sign_masks: list[list[int]] = []
        self._signatures: list[dict[int, int]] = []

        # Per non-basis state: the literal for "some node added so far maps to it", and every such
        # literal as (variable, state, position of the node that extended it).
        self._used: dict[int, int] = {}
        self._used_variables: list[tuple[int, int, int]] = []

    @property
    def num_vars(self) -> int:
        return self._cnf.num_vars

    def add_observations(self, nodes: Sequence[DCNode], edges: Sequence[Edge],
                         mapping_domains: Sequence[set[int]]) -> list[list[int]]:
        """Encode the informative tree and return the clauses not returned before.

        Args:
            nodes: Informative tree nodes in breadth-first order; node ``0`` is the root. They must
                include every node added before.
            edges: Tree edges between ``nodes``, each listed after the edge into its source.
            mapping_domains: Allowed states for each node. For a node added before, states outside
                its domain at that time are ignored.

        Returns:
            The new clauses.
        """
        parents: list[tuple[int, int] | None] = [None] * len(nodes)
        for source_index, target_index, letter_index in edges:
            parents[target_index] = (source_index, letter_index)

        new_indices: list[int] = []
        for node_index, (node, domain) in enumerate(zip(nodes, mapping_domains)):
            mapping = self.mapping_variables.get(node)
            if mapping is None:
                new_indices.append(node_index)
                parent = parents[node_index]
                if parent is not None:
                    self._parents[node] = (nodes[parent[0]], parent[1])
            else:
                excluded = self._excluded[node]
                for state in mapping.keys() - domain - excluded:
                    excluded.add(state)
                    self._cnf.clauses.append([-mapping[state]])

        self._tree_order.extend(nodes[node_index] for node_index in new_indices)
        if not self._nodes:
            degrees = _apartness_degrees(nodes, parents)
            new_indices.sort(key=lambda node_index: -degrees[node_index])  # Stable: ties stay breadth-first.
        for node_index in new_indices:
            self._add_node(nodes[node_index], mapping_domains[node_index])
        for node_index in new_indices:
            self._encode_functional_simulation(nodes[node_index])

        new_known: list[tuple[DCNode, int]] = []
        for node in nodes:
            if node not in self._known_nodes and node.has_known_output():
                self._known_nodes.add(node)
                new_known.append((node, 1 if node.output is DCValue.TRUE else -1))
        output_variables = self.output_variables
        for node, sign in new_known:
            self._cnf.clauses.extend([-mapping_literal, sign * output_variables[state]]
                                     for state, mapping_literal in self.mapping_variables[node].items())
        self._encode_output_signatures(new_known)
        return self._cnf.take_clauses()

    def _add_node(self, node: DCNode, domain: set[int]) -> None:
        """Create the mapping variables of ``node`` with their unique choice and value precedence."""
        cnf = self._cnf
        clauses = cnf.clauses
        mapping = {state: cnf.new_var() for state in sorted(domain)}
        self.mapping_variables[node] = mapping
        self._excluded[node] = set()
        cnf.exactly_one(list(mapping.values()))

        # Value precedence: map(u, s) -> some earlier node maps to s - 1, for non-basis states.
        position = len(self._nodes)
        self._nodes.append(node)
        used = self._used
        for state, literal in mapping.items():
            if state <= self.basis_size:
                continue
            previous = used.get(state - 1)
            clauses.append([-literal] if previous is None else [-literal, previous])
        for state, literal in mapping.items():
            if state < self.basis_size:
                continue
            previous = used.get(state)
            if previous is None:
                used[state] = literal
            else:
                current = used[state] = cnf.new_var()
                self._used_variables.append((current, state, position))
                clauses.append([-literal, current])
                clauses.append([-previous, current])
                clauses.append([-current, previous, literal])

    def _encode_functional_simulation(self, node: DCNode) -> None:
        """map(u, s) & delta(s, a, t) -> map(ua, t) for the edge into ``node``.

        Without a map(ua, t) variable the transition is excluded.
        """
        parent = self._parents.get(node)
        if parent is None:
            return
        parent_node, letter_index = parent
        mapping = self.mapping_variables[node]
        for source_state, source_literal in self.mapping_variables[parent_node].items():
            self._cnf.clauses.extend([-source_literal, -transition_literal, mapping[target_state]]
                                     if target_state in mapping else [-source_literal, -transition_literal]
                                     for target_state, transition_literal
                                     in self.transition_variables[source_state][letter_index].items())

    def _encode_output_signatures(self, new_known: Iterable[tuple[DCNode, int]]) -> None:
        """Redundant: map(u, s) -> sig(s, w) == output(uw) for every node ``uw`` with a known output.

        ``sig(s, w)`` is the output of state ``s`` after the non-empty suffix ``w``; ``accepting(s)``
        plays this role for the empty suffix. Two nodes whose outputs after some suffix differ thus
        cannot share a state, which unit propagation sees in two steps instead of only once all
        transitions along the suffix are fixed. Every DFA satisfies these clauses with ``sig`` set to
        its actual outputs, so satisfiability is unchanged.

        ``sig(s, w)`` is only created once nodes with both outputs after ``w`` may map to ``s``: with
        a single output the variable would be pure and its clauses could not cause any propagation.
        """
        cnf = self._cnf
        clauses = cnf.clauses
        touched: set[int] = set()
        for node, sign in new_known:
            suffix = -1  # The empty suffix.
            parent = self._parents.get(node)
            while parent is not None:
                ancestor, letter = parent
                key = (letter, suffix)
                next_suffix = self._suffix_index.get(key)
                if next_suffix is None:
                    next_suffix = self._suffix_index[key] = len(self._suffixes)
                    self._suffixes.append(key)
                    self._occurrences.append([])
                    self._sign_masks.append([0, 0])
                    self._signatures.append({})
                suffix = next_suffix
                mapping = self.mapping_variables[ancestor]
                self._occurrences[suffix].append((ancestor, sign))
                self._sign_masks[suffix][sign > 0] |= sum(1 << state for state in mapping)
                signature = self._signatures[suffix]
                clauses.extend([-mapping_literal, sign * signature[state]]
                               for state, mapping_literal in mapping.items() if state in signature)
                touched.add(suffix)
                parent = self._parents.get(ancestor)

        for suffix in touched:
            rejecting_mask, accepting_mask = self._sign_masks[suffix]
            signature = self._signatures[suffix]
            new_states = [state for state in range(self.number_of_states)
                          if (accepting_mask & rejecting_mask) >> state & 1 and state not in signature]
            for state in new_states:
                signature[state] = cnf.new_var()
            for ancestor, sign in self._occurrences[suffix]:
                mapping = self.mapping_variables[ancestor]
                clauses.extend([-mapping[state], sign * signature[state]] for state in new_states if state in mapping)

    def phases(self, hint: HypothesisMapping) -> list[int]:
        """Literals describing ``hint``, a DFA over the first ``len(hint[0])`` states, as solver phases.

        The non-basis states of ``hint`` are first renumbered by value precedence. Mapping,
        signature and precedence variables then follow the run of ``hint`` on the tree; variables of
        states beyond the hint, and transitions into them, are set to false.
        """
        transition_mapping, output_mapping = hint
        hinted_states = len(transition_mapping)
        run: dict[DCNode, int] = {}
        for node in self._tree_order:
            parent = self._parents.get(node)
            run[node] = 0 if parent is None else transition_mapping[run[parent[0]]][parent[1]]

        renumbering = list(range(hinted_states))
        next_state = self.basis_size
        renumbered = set(range(self.basis_size))
        for state in itertools.chain((run[node] for node in self._nodes), range(hinted_states)):
            if state not in renumbered:
                renumbered.add(state)
                renumbering[state] = next_state
                next_state += 1
        transitions: list[list[int]] = [[]] * hinted_states
        outputs: list[bool] = [False] * hinted_states
        for state in range(hinted_states):
            transitions[renumbering[state]] = [renumbering[target] for target in transition_mapping[state]]
            outputs[renumbering[state]] = output_mapping[state]

        literals: list[int] = []
        for state, row in enumerate(self.transition_variables):
            for letter, transition in enumerate(row):
                target = transitions[state][letter] if state < hinted_states else None
                literals.extend(literal if successor == target else -literal
                                for successor, literal in transition.items())
        literals.extend(literal if state < hinted_states and outputs[state] else -literal
                        for state, literal in enumerate(self.output_variables))

        first_use: dict[int, int] = {}
        for position, node in enumerate(self._nodes):
            state = renumbering[run[node]]
            first_use.setdefault(state, position)
            literals.extend(literal if candidate == state else -literal
                            for candidate, literal in self.mapping_variables[node].items())
        literals.extend(variable if first_use.get(state, position + 1) <= position else -variable
                        for variable, state, position in self._used_variables)

        # Output of every hinted state after every suffix, in suffix order (rest suffixes come first).
        suffix_outputs: list[list[bool]] = []
        for (letter, rest), signature in zip(self._suffixes, self._signatures):
            rest_outputs = outputs if rest < 0 else suffix_outputs[rest]
            suffix_outputs.append([rest_outputs[transitions[state][letter]] for state in range(hinted_states)])
            literals.extend(variable if suffix_outputs[-1][state] else -variable
                            for state, variable in signature.items() if state < hinted_states)
        return literals

    def decode(self, model: Iterable[int]) -> HypothesisMapping:
        """Read the transition targets and accepting states from a satisfying assignment."""
        true_literals = set(model)
        transition_mapping: list[list[int]] = []
        for state, row in enumerate(self.transition_variables):
            targets = []
            for letter, transition in enumerate(row):
                target = next((target for target, literal in transition.items() if literal in true_literals), None)
                if target is None:
                    raise RuntimeError(f"No transition selected for delta({state}, {letter}).")
                targets.append(target)
            transition_mapping.append(targets)
        return transition_mapping, [literal in true_literals for literal in self.output_variables]


def _apartness_degrees(nodes: Sequence[DCNode], parents: Sequence[tuple[int, int] | None]) -> list[int]:
    """Count for every node how many nodes are apart from it."""
    # Suffixes are numbered through a trie of reversed words, as for the output signatures, with
    # ``None`` as the empty suffix. Per suffix: the masks of nodes rejecting and accepting after it.
    suffix_index: dict[tuple[int, int] | None, int] = {}
    masks: list[list[int]] = []
    occurrences: list[tuple[int, int, bool]] = []
    for node_index, node in enumerate(nodes):
        if not node.has_known_output():
            continue
        accepting = node.output is DCValue.TRUE
        key: tuple[int, int] | None = None
        ancestor = node_index
        while True:
            suffix = suffix_index.setdefault(key, len(suffix_index))
            if suffix == len(masks):
                masks.append([0, 0])
            masks[suffix][accepting] |= 1 << ancestor
            occurrences.append((ancestor, suffix, accepting))
            parent = parents[ancestor]
            if parent is None:
                break
            ancestor, letter = parent
            key = (letter, suffix)
    apart = [0] * len(nodes)
    for node_index, suffix, accepting in occurrences:
        apart[node_index] |= masks[suffix][not accepting]
    return [mask.bit_count() for mask in apart]

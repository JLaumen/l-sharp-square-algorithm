"""CNF encoding of "a DFA with at most a given number of states into which the observation tree maps"."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import combinations

# A clause; tuples are cheaper to create than lists, and the solver accepts both.
Clause = tuple[int, ...]
# Target state of each (state, letter), and whether each state is accepting.
HypothesisMapping = tuple[list[list[int]], list[bool]]

# Up to this many literals, at-most-one is encoded pairwise, which propagates better than a
# sequential counter and keeps hypothesis searches considerably faster at typical state bounds.
_PAIRWISE_AT_MOST_ONE_LIMIT = 32


class DfaEncoding:
    """CNF for a DFA with at most ``number_of_states`` states into which the observation tree maps.

    The tree is given by its informative nodes, numbered breadth-first with the root as node 0.

    Variables:

    * ``map(u, s)``: node ``u`` maps to state ``s``, for ``s`` in the mapping domain of ``u``;
    * ``delta(s, a, t)``: the ``a``-transition of ``s`` goes to ``t``;
    * ``accepting(s)``: state ``s`` is accepting;
    * ``sig(s, w)``: the output of state ``s`` after the non-empty suffix ``w`` (redundant, see below);
    * ``off(t)`` for non-basis states ``t``: states ``t, t+1, ...`` are unused. Solving under the
      assumption ``off(k)`` searches for a DFA with ``k`` states, so one encoding serves all bounds
      up to ``number_of_states``;
    * auxiliary variables of the at-most-one and symmetry-breaking constraints.

    Basis state ``i`` is fixed by the basis node ``i`` mapping to it. The other states are numbered
    by value precedence: a node may only map to such a state ``s`` if an earlier node (breadth-first)
    maps to ``s - 1``. Any consistent DFA can be renumbered this way, by numbering its non-basis
    states in the order of the first node mapping to them and putting states no node maps to last,
    so satisfiability is unchanged. (Ordering the nodes by the number of nodes they are apart from
    instead made the hardest benchmark instances 20-150 times slower to solve.)

    The output signatures make two nodes whose outputs after some suffix differ unable to share a
    state as soon as both are mapped, rather than only once all transitions along the suffix are
    fixed. Every DFA satisfies them with ``sig`` set to its actual outputs. ``sig(s, w)`` is only
    created for states that nodes with both outputs after ``w`` may map to: with one output the
    variable would be pure and its clauses could not cause any propagation.

    Args:
        number_of_states: Maximum number of states.
        basis_size: Number of basis states, which come first.
        alphabet_size: Number of input letters.
        parents: Per node its parent and the letter from the parent to it; ``None`` for the root.
        mapping_domains: Per node the sorted states it may map to.
        outputs: Per node ``1`` if accepting, ``-1`` if rejecting and ``0`` if unknown.
        suffix_occurrences: Per suffix the pairs ``(u, output of uw)`` of all nodes ``u`` for which
            ``uw`` has a known output; suffix ``0`` is the empty word.
    """

    def __init__(self, number_of_states: int, basis_size: int, alphabet_size: int,
                 parents: Sequence[tuple[int, int] | None], mapping_domains: Sequence[Sequence[int]],
                 outputs: Sequence[int], suffix_occurrences: Sequence[Sequence[tuple[int, int]]]) -> None:
        self.number_of_states = number_of_states
        self.num_vars = 0
        self.clauses: list[Clause] = []
        self.transition_variables = [[self._new_vars(number_of_states) for _ in range(alphabet_size)]
                                     for _ in range(number_of_states)]
        self.output_variables = self._new_vars(number_of_states)
        for row in self.transition_variables:
            for transition in row:
                self._exactly_one(transition)
        # Per node: map(u, s) for every state (0 outside the domain) and its (state, map(u, s)) pairs.
        self._mapping: list[list[int]] = []
        self._mapping_items: list[list[tuple[int, int]]] = []
        self._encode_mapping(mapping_domains, basis_size)
        self._encode_functional_simulation(parents)
        self._encode_outputs(outputs)
        self._encode_output_signatures(suffix_occurrences[1:])
        self.off = self._encode_state_activation(basis_size)

    def _new_vars(self, count: int) -> list[int]:
        start = self.num_vars + 1
        self.num_vars += count
        return list(range(start, start + count))

    def _exactly_one(self, literals: Sequence[int]) -> None:
        """At least one literal, plus pairwise or (for many literals) sequential-counter at-most-one."""
        clauses = self.clauses
        clauses.append(tuple(literals))
        count = len(literals)
        if count <= _PAIRWISE_AT_MOST_ONE_LIMIT:
            clauses.extend(combinations([-literal for literal in literals], 2))
            return
        # Sinz's sequential counter: ``prefix[i]`` holds if one of ``literals[0..i]`` does.
        prefix = self._new_vars(count - 1)
        clauses.append((-literals[0], prefix[0]))
        for index in range(1, count - 1):
            clauses.append((-prefix[index - 1], prefix[index]))
            clauses.append((-literals[index], -prefix[index - 1]))
            clauses.append((-literals[index], prefix[index]))
        clauses.append((-literals[-1], -prefix[-1]))

    def _encode_mapping(self, mapping_domains: Sequence[Sequence[int]], basis_size: int) -> None:
        """Mapping variables with their unique choice and value precedence, in node order."""
        clauses = self.clauses
        used = [0] * self.number_of_states  # Per state: literal for "some node so far maps to it".
        for domain in mapping_domains:
            items = list(zip(domain, self._new_vars(len(domain))))
            mapping = [0] * self.number_of_states
            for state, literal in items:
                mapping[state] = literal
            self._mapping.append(mapping)
            self._mapping_items.append(items)
            self._exactly_one([literal for _, literal in items])
            for state, literal in items:
                if state > basis_size:  # map(u, s) -> an earlier node maps to s - 1.
                    previous = used[state - 1]
                    clauses.append((-literal, previous) if previous else (-literal,))
            for state, literal in items:
                if state < basis_size:
                    continue
                previous = used[state]
                if previous:
                    current = used[state] = self._new_vars(1)[0]
                    clauses.extend([(-literal, current), (-previous, current), (-current, previous, literal)])
                else:
                    used[state] = literal

    def _encode_functional_simulation(self, parents: Sequence[tuple[int, int] | None]) -> None:
        """map(u, s) & delta(s, a, t) -> map(ua, t); without a map(ua, t) variable the transition is excluded."""
        clauses = self.clauses
        for node, parent in enumerate(parents):
            if parent is None:
                continue
            parent_node, letter = parent
            mapping = self._mapping[node]
            for source_state, source_literal in self._mapping_items[parent_node]:
                transitions = self.transition_variables[source_state][letter]
                clauses.extend([(-source_literal, -transition, target) if target else (-source_literal, -transition)
                                for transition, target in zip(transitions, mapping)])

    def _encode_outputs(self, outputs: Sequence[int]) -> None:
        """map(u, s) -> accepting(s) has the known output of ``u``."""
        for node, sign in enumerate(outputs):
            if sign:
                self.clauses.extend([(-literal, sign * self.output_variables[state])
                                     for state, literal in self._mapping_items[node]])

    def _encode_output_signatures(self, suffix_occurrences: Iterable[Sequence[tuple[int, int]]]) -> None:
        """map(u, s) -> sig(s, w) has the output of ``uw``, for every non-empty suffix ``w``."""
        mapping_items = self._mapping_items
        domain_masks = [sum(1 << state for state, _ in items) for items in mapping_items]
        for occurrences in suffix_occurrences:
            masks = {1: 0, -1: 0}  # States that nodes with each output after the suffix may map to.
            for node, sign in occurrences:
                masks[sign] |= domain_masks[node]
            both = masks[1] & masks[-1]
            if not both:
                continue
            states = [state for state in range(self.number_of_states) if both >> state & 1]
            signature = dict(zip(states, self._new_vars(len(states))))
            for node, sign in occurrences:
                self.clauses.extend([(-literal, sign * signature[state])
                                     for state, literal in mapping_items[node] if both >> state & 1])

    def _encode_state_activation(self, basis_size: int) -> dict[int, int]:
        """off(t) -> off(t + 1), and off(t) excludes all mappings and transitions to ``t``."""
        clauses = self.clauses
        states = range(basis_size, self.number_of_states)
        off = dict(zip(states, self._new_vars(len(states))))
        for state in states:
            if state + 1 in off:
                clauses.append((-off[state], off[state + 1]))
            clauses.extend([(-off[state], -mapping[state]) for mapping in self._mapping if mapping[state]])
            clauses.extend([(-off[state], -transition[state])
                            for row in self.transition_variables for transition in row])
        return off

    def decode(self, model: Iterable[int], number_of_states: int) -> HypothesisMapping:
        """Transitions and accepting states of the first ``number_of_states`` states of a model."""
        true_literals = set(model)
        transitions = [[next(target for target, literal in enumerate(transition) if literal in true_literals)
                        for transition in row] for row in self.transition_variables[:number_of_states]]
        return transitions, [literal in true_literals for literal in self.output_variables[:number_of_states]]

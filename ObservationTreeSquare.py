"""SAT-based observation-tree learner for separating deterministic finite automata."""

from __future__ import annotations

import itertools
import logging
import time
from collections import deque
from collections.abc import Iterable, Sequence
from typing import Any

from aalpy.automata import Dfa, DfaState
from pysat.card import CardEnc, EncType
from pysat.solvers import Cadical153

from Apartness import Apartness
from DCNode import DCNode
from DCValue import DCValue

logger = logging.getLogger(__name__)


class _SatVariableAllocator:
    def __init__(self, first_var: int = 1) -> None:
        self.next_var = first_var

    def new(self) -> int:
        variable = self.next_var
        self.next_var += 1
        return variable

    def advance_to(self, next_var: int) -> None:
        self.next_var = max(self.next_var, next_var)


def _add_exactly_one(
    clauses: list[list[int]],
    literals: Iterable[int],
    allocator: _SatVariableAllocator,
) -> None:
    literals = list(literals)
    if not literals:
        clauses.append([])
        return

    clauses.append(literals)
    if len(literals) == 1:
        return

    if len(literals) <= 5:
        clauses.extend(
            [-left, -right] for left, right in itertools.combinations(literals, 2)
        )
        return

    encoding = CardEnc.atmost(
        lits=literals,
        bound=1,
        top_id=allocator.next_var - 1,
        encoding=EncType.seqcounter,
    )
    clauses.extend(encoding.clauses)
    allocator.advance_to(encoding.nv + 1)


def _add_guarded_exactly_one(
    clauses: list[list[int]],
    literals: Iterable[int],
    guard: int,
    allocator: _SatVariableAllocator,
) -> None:
    """Add guard -> exactly_one(literals)."""
    literals = list(literals)
    if not literals:
        clauses.append([-guard])
        return

    clauses.append([-guard, *literals])
    if len(literals) == 1:
        return

    if len(literals) <= 5:
        clauses.extend(
            [-guard, -left, -right]
            for left, right in itertools.combinations(literals, 2)
        )
        return

    encoding = CardEnc.atmost(
        lits=literals,
        bound=1,
        top_id=allocator.next_var - 1,
        encoding=EncType.seqcounter,
    )
    clauses.extend([-guard, *clause] for clause in encoding.clauses)
    allocator.advance_to(encoding.nv + 1)


class ObservationTreeSquare:
    """Learn a separating DFA from an incomplete observation tree."""

    def __init__(
        self,
        alphabet: Sequence[Any],
        sul: Any,
        replace_basis: bool = True,
        assume_prefix_closed: bool = True,
    ) -> None:
        self.automaton_type = "dfa"
        self.alphabet = list(alphabet)
        self.sul = sul
        self.replace_basis = replace_basis
        self.assume_prefix_closed = assume_prefix_closed
        self.solver_time = 0.0

        DCNode._id_counter = 0
        self.root = DCNode()
        self.root.set_output(self.sul.query(()))

        self.size = 1
        self.guaranteed_basis = [self.root]
        self.frontier_to_basis_dict: dict[DCNode, set[DCNode]] = {}

    def _insert_observation_sequence(
        self,
        inputs: Sequence[Any],
        outputs: Sequence[DCValue | None],
    ) -> None:
        node = self.root
        for input_value, output in zip(inputs, outputs):
            node = node.extend_and_get(
                input_value,
                output if output is not None else DCValue.DC,
            )
            if output is None:
                node.output = None
            if node not in self.frontier_to_basis_dict:
                self.frontier_to_basis_dict[node] = set(self.guaranteed_basis)

    def _get_successor(
        self,
        inputs: Sequence[Any],
        start_node: DCNode | None = None,
    ) -> DCNode | None:
        node = self.root if start_node is None else start_node
        for input_value in inputs:
            node = node.get_successor(input_value)
            if node is None:
                return None
        return node

    @staticmethod
    def _get_transfer_sequence(
        start_node: DCNode,
        end_node: DCNode,
    ) -> list[Any] | None:
        transfer_sequence: list[Any] = []
        node = end_node
        while node is not start_node:
            if node.parent is None:
                return None
            transfer_sequence.append(node.input_to_parent)
            node = node.parent
        transfer_sequence.reverse()
        return transfer_sequence

    def _get_access_sequence(self, node: DCNode) -> list[Any]:
        access_sequence = self._get_transfer_sequence(self.root, node)
        if access_sequence is None:
            raise ValueError("Node is not part of the observation tree.")
        return access_sequence

    @staticmethod
    def _is_known(node: DCNode) -> bool:
        return node.output is not None and node.output.is_known()

    def _update_basis_candidates(self, frontier_node: DCNode) -> None:
        candidates = self.frontier_to_basis_dict.get(frontier_node)
        if candidates is None:
            return
        self.frontier_to_basis_dict[frontier_node] = {
            candidate
            for candidate in candidates
            if not Apartness.states_are_apart(frontier_node, candidate, self)
        }

    def _update_frontier_to_basis_dict(self) -> None:
        self._update_frontier_to_basis_dict_dfs(self.root)

    def _update_frontier_to_basis_dict_dfs(self, node: DCNode) -> None:
        if not node.leads_to_known:
            return

        if node not in self.guaranteed_basis:
            self._update_basis_candidates(node)
            if not self.frontier_to_basis_dict.get(node):
                return

        for successor in node.successors.values():
            self._update_frontier_to_basis_dict_dfs(successor)

    def _find_first_allowed_basis_node(self) -> DCNode | None:
        """Return the first non-basis node apart from every basis node."""
        basis = tuple(self.guaranteed_basis)
        basis_set = set(basis)
        queue = deque([self.root])
        seen: set[DCNode] = set()

        while queue:
            node = queue.popleft()
            if node in seen:
                continue

            seen.add(node)
            queue.extend(node.successors.values())

            if node in basis_set or not node.leads_to_known:
                continue

            if all(
                Apartness.states_are_apart(node, basis_node, self)
                for basis_node in basis
            ):
                return node

        return None

    def _rebuild_frontier_to_basis_dict(self) -> None:
        basis = tuple(self.guaranteed_basis)
        basis_set = set(basis)
        frontier: dict[DCNode, set[DCNode]] = {}
        queue = deque([self.root])
        seen: set[DCNode] = set()

        while queue:
            node = queue.popleft()
            if node in seen:
                continue

            seen.add(node)
            queue.extend(node.successors.values())

            if node in basis_set:
                continue

            frontier[node] = {
                basis_node
                for basis_node in basis
                if not Apartness.states_are_apart(node, basis_node, self)
            }

        self.frontier_to_basis_dict = frontier

    def _promote_node_to_basis(self) -> bool:
        node = self._find_first_allowed_basis_node()
        if node is None:
            return False

        self.guaranteed_basis.append(node)
        self.size = max(self.size, len(self.guaranteed_basis))
        self._rebuild_frontier_to_basis_dict()
        return True

    def _make_frontiers_identified(self) -> bool:
        extended = False

        for basis_node in self.guaranteed_basis:
            for input_value in self.alphabet:
                frontier_node = basis_node.get_successor(input_value)
                if frontier_node is None:
                    continue

                while self._identify_frontier(frontier_node):
                    extended = True
                    self._update_basis_candidates(frontier_node)

        return extended

    def _identify_frontier(self, frontier_node: DCNode) -> bool:
        candidates = self.frontier_to_basis_dict.get(frontier_node)
        if not candidates:
            return False

        inputs_to_frontier = self._get_access_sequence(frontier_node)

        for witness_sequence in self._get_identification_witnesses(frontier_node):
            inputs = inputs_to_frontier + witness_sequence
            outputs, extended = self._get_output_sequence(inputs)
            self._insert_observation_sequence(inputs, outputs)
            if extended:
                return True

        return False

    def _get_identification_witnesses(
        self,
        frontier_node: DCNode,
    ) -> Iterable[list[Any]]:
        candidates = list(self.frontier_to_basis_dict.get(frontier_node, ()))
        yield from self._filter_unused_witnesses(
            frontier_node,
            Apartness.get_distinguishing_sequences(candidates, self),
        )

    def _filter_unused_witnesses(
        self,
        frontier_node: DCNode,
        witnesses: Iterable[list[Any]],
    ) -> Iterable[list[Any]]:
        for witness_sequence in witnesses:
            target = self._get_successor(
                witness_sequence,
                start_node=frontier_node,
            )
            if target is None or target.output is None:
                yield witness_sequence

    def _compute_mapping_domains(
        self,
        nodes: Sequence[DCNode],
        basis_index: dict[DCNode, int],
        basis_size: int,
        number_of_states: int,
    ) -> list[set[int]]:
        all_states = set(range(number_of_states))
        free_states = set(range(basis_size, number_of_states))
        domains: list[set[int]] = []

        for node in nodes:
            if node in basis_index:
                domains.append({basis_index[node]})
                continue

            candidates = self.frontier_to_basis_dict.get(node)
            if candidates is None:
                domains.append(set(all_states))
                continue

            compatible_basis_states = {
                basis_index[candidate] for candidate in candidates
            }
            domains.append(compatible_basis_states | free_states)

        return domains

    def _prepare_sat_domains(
        self,
        nodes: Sequence[DCNode],
        mapping_domains: list[set[int]],
    ) -> bool:
        number_of_states = self.size
        forced_output: list[bool | None] = [None] * number_of_states

        for node_index, node in enumerate(nodes):
            if not self._is_known(node):
                continue

            domain = mapping_domains[node_index]
            if len(domain) != 1:
                continue

            state = next(iter(domain))
            accepts = node.output is DCValue.TRUE

            if forced_output[state] is not None and forced_output[state] != accepts:
                return False

            forced_output[state] = accepts

        for node_index, node in enumerate(nodes):
            if not self._is_known(node):
                continue

            accepts = node.output is DCValue.TRUE
            domain = mapping_domains[node_index]

            domain.difference_update(
                state
                for state in tuple(domain)
                if (
                    forced_output[state] is not None and forced_output[state] != accepts
                )
            )

            if not domain:
                return False

        basis_indices = {
            basis_node: index for index, basis_node in enumerate(self.guaranteed_basis)
        }
        basis_set = set(self.guaranteed_basis)

        for node_index, node in enumerate(nodes):
            if node in basis_set:
                continue

            domain = mapping_domains[node_index]
            if not domain:
                return False

            for basis_node in self.guaranteed_basis:
                basis_state = basis_indices[basis_node]

                if basis_state not in domain:
                    continue

                if Apartness.states_are_apart(node, basis_node, self):
                    domain.remove(basis_state)

            if not domain:
                return False

        return True

    @staticmethod
    def _compute_transition_domains(
        edges: Sequence[tuple[int, int, int]],
        mapping_domains: Sequence[set[int]],
        number_of_states: int,
        alphabet_size: int,
    ) -> list[list[set[int]]]:
        all_states = set(range(number_of_states))
        transition_domains = [
            [set() for _ in range(alphabet_size)] for _ in range(number_of_states)
        ]
        constrained = [
            [False for _ in range(alphabet_size)] for _ in range(number_of_states)
        ]

        for source_index, target_index, letter_index in edges:
            target_domain = mapping_domains[target_index]

            for source_state in mapping_domains[source_index]:
                constrained[source_state][letter_index] = True
                transition_domains[source_state][letter_index].update(target_domain)

        for state in range(number_of_states):
            for letter in range(alphabet_size):
                if not constrained[state][letter]:
                    transition_domains[state][letter] = set(all_states)

        return transition_domains

    @staticmethod
    def _add_accepting_state_constraints(
        clauses: list[list[int]],
        allocator: _SatVariableAllocator,
        transition_variables: Sequence[Sequence[dict[int, int]]],
        output_variables: Sequence[int],
        number_of_states: int,
        alphabet_size: int,
    ) -> None:
        # Exactly one accepting state.
        _add_exactly_one(
            clauses,
            output_variables,
            allocator,
        )

        for accepting_state in range(number_of_states):
            accepting_literal = output_variables[accepting_state]

            # Every transition from the accepting state is a self-loop.
            for letter in range(alphabet_size):
                transition_literal = transition_variables[accepting_state][letter].get(
                    accepting_state
                )

                if transition_literal is None:
                    clauses.append([-accepting_literal])
                else:
                    clauses.append(
                        [
                            -accepting_literal,
                            transition_literal,
                        ]
                    )

            # Exactly one non-self transition enters the accepting state.
            incoming_non_self: list[int] = []

            for source_state in range(number_of_states):
                if source_state == accepting_state:
                    continue

                for letter in range(alphabet_size):
                    transition_literal = transition_variables[source_state][letter].get(
                        accepting_state
                    )

                    if transition_literal is not None:
                        incoming_non_self.append(transition_literal)

            _add_guarded_exactly_one(
                clauses,
                incoming_non_self,
                accepting_literal,
                allocator,
            )

    def _find_hypothesis(self) -> tuple[list[list[int]], list[bool]] | None:
        total_start_time = time.perf_counter()
        solver: Cadical153 | None = None
        clauses: list[list[int]] = []
        allocator = _SatVariableAllocator()

        try:
            number_of_states = self.size
            alphabet_size = len(self.alphabet)

            logger.debug(
                "find_hypothesis started: states=%d, alphabet_size=%d, basis_size=%d.",
                number_of_states,
                alphabet_size,
                len(self.guaranteed_basis),
            )

            if number_of_states <= 0:
                return None

            nodes, edges = self._flatten_observation_tree()
            basis_size = len(self.guaranteed_basis)

            if basis_size > number_of_states:
                return None

            basis_index = {
                basis_node: state
                for state, basis_node in enumerate(self.guaranteed_basis)
            }

            mapping_domains = self._compute_mapping_domains(
                nodes,
                basis_index,
                basis_size,
                number_of_states,
            )

            if any(not domain for domain in mapping_domains):
                return None

            if not self._prepare_sat_domains(
                nodes,
                mapping_domains,
            ):
                return None

            transition_domains = self._compute_transition_domains(
                edges,
                mapping_domains,
                number_of_states,
                alphabet_size,
            )

            if any(
                not transition_domains[state][letter]
                for state in range(number_of_states)
                for letter in range(alphabet_size)
            ):
                return None

            mapping_variables: list[dict[int, int]] = [
                {state: allocator.new() for state in sorted(domain)}
                for domain in mapping_domains
            ]

            transition_variables: list[list[dict[int, int]]] = [
                [
                    {
                        target: allocator.new()
                        for target in sorted(transition_domains[state][letter])
                    }
                    for letter in range(alphabet_size)
                ]
                for state in range(number_of_states)
            ]

            output_variables = [allocator.new() for _ in range(number_of_states)]

            for mapping in mapping_variables:
                _add_exactly_one(
                    clauses,
                    mapping.values(),
                    allocator,
                )

            for state in range(number_of_states):
                for letter in range(alphabet_size):
                    _add_exactly_one(
                        clauses,
                        transition_variables[state][letter].values(),
                        allocator,
                    )

            self._add_functional_simulation(
                clauses,
                edges,
                mapping_variables,
                transition_variables,
            )

            self._add_output_constraints(
                clauses,
                nodes,
                mapping_variables,
                output_variables,
            )

            self._add_accepting_state_constraints(
                clauses,
                allocator,
                transition_variables,
                output_variables,
                number_of_states,
                alphabet_size,
            )

            self._add_bfs_symmetry_breaking(
                clauses,
                allocator,
                transition_variables,
                basis_size,
                number_of_states,
                alphabet_size,
            )

            solver = Cadical153(bootstrap_with=clauses)
            result = solver.solve()

            if result is False:
                return None

            if result is not True:
                raise RuntimeError(f"Unexpected CaDiCaL result: {result!r}.")

            return self._extract_sat_model(
                solver,
                transition_variables,
                output_variables,
                number_of_states,
                alphabet_size,
            )

        finally:
            if solver is not None:
                solver.delete()

            elapsed = time.perf_counter() - total_start_time
            self.solver_time += elapsed

            logger.debug(
                "find_hypothesis finished: time=%.3fs, states=%d, variables=%d, clauses=%d.",
                elapsed,
                self.size,
                allocator.next_var - 1,
                len(clauses),
            )

    def _flatten_observation_tree(
        self,
    ) -> tuple[list[DCNode], list[tuple[int, int, int]]]:
        queue = deque([self.root])
        nodes = [self.root]
        node_index = {self.root: 0}
        alphabet_index = {letter: index for index, letter in enumerate(self.alphabet)}
        edges: list[tuple[int, int, int]] = []

        while queue:
            node = queue.popleft()
            source_index = node_index[node]

            for input_value, successor in node.successors.items():
                if not successor.leads_to_known:
                    continue

                if successor not in node_index:
                    node_index[successor] = len(nodes)
                    nodes.append(successor)
                    queue.append(successor)

                edges.append(
                    (
                        source_index,
                        node_index[successor],
                        alphabet_index[input_value],
                    )
                )

        return nodes, edges

    @staticmethod
    def _add_functional_simulation(
        clauses: list[list[int]],
        edges: Sequence[tuple[int, int, int]],
        mapping_variables: Sequence[dict[int, int]],
        transition_variables: Sequence[Sequence[dict[int, int]]],
    ) -> None:
        for source_index, target_index, letter_index in edges:
            source_mapping = mapping_variables[source_index]
            target_mapping = mapping_variables[target_index]

            for source_state, source_literal in source_mapping.items():
                transition_row = transition_variables[source_state][letter_index]

                for target_state, transition_literal in transition_row.items():
                    target_literal = target_mapping.get(target_state)

                    if target_literal is None:
                        clauses.append(
                            [
                                -source_literal,
                                -transition_literal,
                            ]
                        )
                    else:
                        clauses.append(
                            [
                                -source_literal,
                                -transition_literal,
                                target_literal,
                            ]
                        )

    @staticmethod
    def _add_output_constraints(
        clauses: list[list[int]],
        nodes: Sequence[DCNode],
        mapping_variables: Sequence[dict[int, int]],
        output_variables: Sequence[int],
    ) -> None:
        for node_index, node in enumerate(nodes):
            if not ObservationTreeSquare._is_known(node):
                continue

            accepts = node.output is DCValue.TRUE

            for state, mapping_literal in mapping_variables[node_index].items():
                clauses.append(
                    [
                        -mapping_literal,
                        (
                            output_variables[state]
                            if accepts
                            else -output_variables[state]
                        ),
                    ]
                )

    @staticmethod
    def _add_bfs_symmetry_breaking(
        clauses: list[list[int]],
        allocator: _SatVariableAllocator,
        transition_variables: Sequence[Sequence[dict[int, int]]],
        basis_size: int,
        number_of_states: int,
        alphabet_size: int,
    ) -> None:
        if basis_size >= number_of_states or alphabet_size == 0:
            return

        first: dict[int, list[list[int]]] = {}

        for state in range(basis_size, number_of_states):
            first[state] = [
                [allocator.new() for _ in range(alphabet_size)] for _ in range(state)
            ]

            markers = [
                first[state][parent][letter]
                for parent in range(state)
                for letter in range(alphabet_size)
            ]
            clauses.append(markers)

            for parent in range(state):
                for letter in range(alphabet_size):
                    marker = first[state][parent][letter]
                    transition_literal = transition_variables[parent][letter].get(state)

                    if transition_literal is None:
                        clauses.append([-marker])
                    else:
                        clauses.append([-marker, transition_literal])

                    for smaller_parent in range(parent):
                        for smaller_letter in range(alphabet_size):
                            earlier_literal = transition_variables[smaller_parent][
                                smaller_letter
                            ].get(state)

                            if earlier_literal is not None:
                                clauses.append(
                                    [
                                        -marker,
                                        -earlier_literal,
                                    ]
                                )

                    for smaller_letter in range(letter):
                        earlier_literal = transition_variables[parent][
                            smaller_letter
                        ].get(state)

                        if earlier_literal is not None:
                            clauses.append(
                                [
                                    -marker,
                                    -earlier_literal,
                                ]
                            )

        for state in range(basis_size, number_of_states - 1):
            descriptor_count = state * alphabet_size
            if descriptor_count == 0:
                continue

            prefix = [allocator.new() for _ in range(descriptor_count)]

            for rank in range(descriptor_count):
                parent = rank // alphabet_size
                letter = rank % alphabet_size
                current = prefix[rank]
                marker = first[state][parent][letter]

                if rank == 0:
                    clauses.extend(
                        [
                            [-current, marker],
                            [-marker, current],
                        ]
                    )
                else:
                    previous = prefix[rank - 1]
                    clauses.append(
                        [
                            -current,
                            previous,
                            marker,
                        ]
                    )
                    clauses.append(
                        [
                            -previous,
                            current,
                        ]
                    )
                    clauses.append(
                        [
                            -marker,
                            current,
                        ]
                    )

            for rank in range(descriptor_count):
                parent = rank // alphabet_size
                letter = rank % alphabet_size
                marker_next = first[state + 1][parent][letter]

                if rank == 0:
                    clauses.append([-marker_next])
                else:
                    clauses.append(
                        [
                            -marker_next,
                            prefix[rank - 1],
                        ]
                    )

    @staticmethod
    def _extract_sat_model(
        solver: Cadical153,
        transition_variables: Sequence[Sequence[dict[int, int]]],
        output_variables: Sequence[int],
        number_of_states: int,
        alphabet_size: int,
    ) -> tuple[list[list[int]], list[bool]]:
        model = set(solver.get_model())

        transition_mapping = [
            [0 for _ in range(alphabet_size)] for _ in range(number_of_states)
        ]

        for state in range(number_of_states):
            for letter in range(alphabet_size):
                selected_target = next(
                    (
                        target
                        for target, literal in transition_variables[state][
                            letter
                        ].items()
                        if literal in model
                    ),
                    None,
                )

                if selected_target is None:
                    raise RuntimeError(
                        f"No transition selected for delta({state}, {letter})."
                    )

                transition_mapping[state][letter] = selected_target

        output_mapping = [literal in model for literal in output_variables]

        return transition_mapping, output_mapping

    def _construct_hypothesis(
        self,
        transition_mapping: Sequence[Sequence[int]],
        output_mapping: Sequence[bool],
    ) -> Dfa:
        states = [DfaState(f"s{index}") for index in range(self.size)]

        for state_index, state in enumerate(states):
            state.is_accepting = output_mapping[state_index]

            for letter_index, input_value in enumerate(self.alphabet):
                target_index = transition_mapping[state_index][letter_index]
                state.transitions[input_value] = states[target_index]

        hypothesis = Dfa(states[0], states)
        hypothesis.compute_prefixes()
        hypothesis.characterization_set = hypothesis.compute_characterization_set(
            raise_warning=False
        )
        return hypothesis

    def _expand_frontier(self) -> None:
        length = 2

        basis_access_sequences = [
            self._get_access_sequence(node) for node in self.guaranteed_basis
        ]

        for word in itertools.product(self.alphabet, repeat=length):
            word = list(word)

            for access_sequence in basis_access_sequences:
                inputs = access_sequence + word
                outputs, _ = self._get_output_sequence(inputs)
                self._insert_observation_sequence(inputs, outputs)

    def _find_adequate_observation_tree(self) -> None:
        self._expand_frontier()
        self._update_frontier_to_basis_dict()

        while self._promote_node_to_basis():
            self._expand_frontier()
            self._update_frontier_to_basis_dict()

        while self._make_frontiers_identified():
            self._update_frontier_to_basis_dict()

            while self._promote_node_to_basis():
                self._expand_frontier()
                self._update_frontier_to_basis_dict()

    def _get_output_sequence(
        self,
        inputs: Sequence[Any],
    ) -> tuple[list[DCValue | None], bool]:
        outputs: list[DCValue | None] = []
        queried_known_output = False
        current_node: DCNode | None = self.root

        for index, input_value in enumerate(inputs):
            if current_node is not None:
                current_node = current_node.get_successor(input_value)

            if current_node is None or current_node.output is None:
                new_output = self.sul.query(tuple(inputs[: index + 1]))
                outputs.append(new_output)

                if new_output.is_known():
                    queried_known_output = True

                if self.assume_prefix_closed and new_output is DCValue.DC:
                    outputs.extend([DCValue.DC] * (len(inputs) - index - 1))
                    break

                if new_output is DCValue.TRUE:
                    outputs.extend([DCValue.DC] * (len(inputs) - index - 1))
                    break

                continue

            outputs.append(current_node.output)

            if self.assume_prefix_closed and current_node.output is DCValue.DC:
                outputs.extend([DCValue.DC] * (len(inputs) - index - 1))
                break

        return outputs, queried_known_output

    def build_hypothesis(self) -> Dfa | None:
        self._find_adequate_observation_tree()
        hypothesis_mapping = self._find_hypothesis()

        if hypothesis_mapping is None:
            self.size += 1
            return None

        transition_mapping, output_mapping = hypothesis_mapping
        return self._construct_hypothesis(
            transition_mapping,
            output_mapping,
        )

    def process_counter_example(
        self,
        counterexample: Sequence[Any],
        output: bool | DCValue,
    ) -> None:
        counterexample_outputs, _ = self._get_output_sequence(counterexample)
        self._insert_observation_sequence(
            counterexample,
            counterexample_outputs,
        )

        node = self._get_successor(counterexample)
        if node is None:
            raise RuntimeError(
                "Counterexample was not inserted into the observation tree."
            )

        if isinstance(output, bool):
            output = DCValue.TRUE if output else DCValue.FALSE

        node.set_output(output)
        self._update_frontier_to_basis_dict()

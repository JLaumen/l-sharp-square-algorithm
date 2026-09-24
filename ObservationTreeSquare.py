import itertools
import logging
import time
from collections import deque

from aalpy.automata import Dfa, DfaState
from pysat.card import CardEnc, EncType
from pysat.solvers import Cadical153

from Apartness import Apartness
from MooreNode import MooreNode

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S", )
logger = logging.getLogger(__name__)


class _SatVariableAllocator:
    """Allocate positive SAT variable IDs sequentially."""

    def __init__(self, first_var=1):
        self.next_var = first_var

    def new(self):
        variable = self.next_var
        self.next_var += 1
        return variable

    def advance_to(self, next_var):
        self.next_var = max(self.next_var, next_var)


def _add_exactly_one(clauses, literals, allocator):
    """Add an exactly-one constraint using a sequential encoding."""
    literals = list(literals)
    clauses.append(literals)

    if len(literals) <= 1:
        return

    cnf = CardEnc.atmost(lits=literals, bound=1, top_id=allocator.next_var - 1, encoding=EncType.seqcounter, )
    clauses.extend(cnf.clauses)
    allocator.advance_to(cnf.nv + 1)


class ObservationTreeSquare:
    def __init__(self, alphabet, sul, replace_basis, use_compatibility, assume_prefix_closed):
        """Initialize the observation tree with a root node."""
        self.automaton_type = "dfa"
        self.replace_basis = replace_basis
        self.use_compatibility = use_compatibility
        self.assume_prefix_closed = assume_prefix_closed

        self.smt_time = 0
        MooreNode._id_counter = 0

        self.alphabet = alphabet
        self.sul = sul
        self.outputAlphabet = [True, False, "unknown"]
        self.states_list = []

        self.root = MooreNode()
        self.root.set_output(self.sul.query([]))

        self.size = 1
        self.guaranteed_basis = [self.root]
        self.frontier_to_basis_dict = {}

    # ------------------------------------------------------------------
    # Observation tree
    # ------------------------------------------------------------------

    def insert_observation(self, inputs, output):
        """Insert an observation into the tree."""
        node = self.root

        for inp in inputs:
            node = node.extend_and_get(inp, None)

        node.set_output(output)

    def insert_observation_sequence(self, inputs, outputs):
        """Insert an input/output sequence into the observation tree."""
        node = self.root

        for inp, output in zip(inputs, outputs):
            node = node.extend_and_get(inp, output)
            node.set_output(output)

            if node not in self.frontier_to_basis_dict:
                self.frontier_to_basis_dict[node] = set(self.guaranteed_basis)

    def experiment(self, inputs):
        """Perform an experiment and update the observation tree."""
        outputs, _ = self._get_output_sequence(inputs, query_mode="final")
        self.insert_observation_sequence(inputs, outputs)
        return outputs[-1]

    def get_successor(self, inputs, start_node=None):
        """Return the node reached by an input sequence."""
        node = self.root if start_node is None else start_node

        for inp in inputs:
            node = node.get_successor(inp)
            if node is None:
                return None

        return node

    @staticmethod
    def get_transfer_sequence(start_node, end_node):
        """Return the input sequence from start_node to end_node."""
        transfer_sequence = []
        node = end_node

        while node != start_node:
            if node.parent is None:
                return None

            transfer_sequence.append(node.input_to_parent)
            node = node.parent

        transfer_sequence.reverse()
        return transfer_sequence

    def get_access_sequence(self, target_node):
        """Return the access sequence from the root to target_node."""
        return self.get_transfer_sequence(self.root, target_node)

    def get_size(self):
        """Return the number of nodes in the observation tree."""
        return self.root.id_counter

    @staticmethod
    def is_known(node):
        """Return whether a node has known output information."""
        return node.output is not None and node.output != "unknown"

    def count_informative_nodes(self):
        """Count nodes with informative output information."""
        queue = deque([self.root])
        count = 0

        while queue:
            node = queue.popleft()

            if node.output != "unknown":
                count += 1

            queue.extend(node.successors.values())

        return count

    # ------------------------------------------------------------------
    # Basis and frontier handling
    # ------------------------------------------------------------------

    def update_basis_candidates(self, frontier_node):
        """Remove incompatible basis states from a frontier node's domain."""
        candidates = self.frontier_to_basis_dict[frontier_node]

        self.frontier_to_basis_dict[frontier_node] = {candidate for candidate in candidates if
                                                      not Apartness.states_are_incompatible(frontier_node, candidate,
                                                                                            self)}

    def update_frontier_to_basis_dict(self):
        """Update basis candidates for all frontier nodes."""
        self.update_frontier_to_basis_dict_dfs(self.root)

    def update_frontier_to_basis_dict_dfs(self, node):
        if not node.leads_to_known:
            return

        if node not in self.guaranteed_basis:
            self.update_basis_candidates(node)

            if not self.frontier_to_basis_dict[node]:
                return

        for successor in node.successors.values():
            self.update_frontier_to_basis_dict_dfs(successor)

    def _collect_basis_clique_candidates(self):
        """Return nodes that can participate in the basis clique."""
        candidates = []
        seen = set()
        queue = deque([self.root])

        while queue:
            node = queue.popleft()

            if node in seen:
                continue

            seen.add(node)

            if node in self.guaranteed_basis or node.leads_to_known:
                candidates.append(node)

            queue.extend(node.successors.values())

        return candidates

    @staticmethod
    def _maximum_clique(adjacency_masks, required_indices=()):
        """Find an exact maximum clique using branch-and-bound."""
        n_vertices = len(adjacency_masks)

        if n_vertices == 0:
            return []

        required_indices = tuple(required_indices)

        # Required vertices must already form a clique.
        for i, vertex in enumerate(required_indices):
            for other in required_indices[i + 1:]:
                if not (adjacency_masks[vertex] & (1 << other)):
                    raise ValueError("Required basis nodes do not form a clique")

        required_mask = 0
        for vertex in required_indices:
            required_mask |= 1 << vertex

        all_vertices = (1 << n_vertices) - 1
        candidate_mask = all_vertices & ~required_mask

        for vertex in required_indices:
            candidate_mask &= adjacency_masks[vertex]

        best = list(required_indices)
        best_size = len(best)

        def color_sort(vertices):
            """Greedily color the induced graph to obtain upper bounds."""
            order = []
            bounds = []
            remaining = vertices
            color = 0

            while remaining:
                color += 1
                available = remaining

                while available:
                    bit = available & -available
                    vertex = bit.bit_length() - 1

                    order.append(vertex)
                    bounds.append(color)

                    available &= ~bit
                    available &= ~adjacency_masks[vertex]
                    remaining &= ~bit

            return order, bounds

        def expand(vertices, clique):
            nonlocal best, best_size

            if not vertices:
                if len(clique) > best_size:
                    best = clique.copy()
                    best_size = len(best)
                return

            order, bounds = color_sort(vertices)

            for index in range(len(order) - 1, -1, -1):
                if len(clique) + bounds[index] <= best_size:
                    return

                vertex = order[index]
                bit = 1 << vertex

                if not (vertices & bit):
                    continue

                new_vertices = vertices & adjacency_masks[vertex]
                clique.append(vertex)

                if not new_vertices:
                    if len(clique) > best_size:
                        best = clique.copy()
                        best_size = len(best)
                else:
                    expand(new_vertices, clique)

                clique.pop()
                vertices &= ~bit

        expand(candidate_mask, list(required_indices))
        return best

    def _find_maximum_basis_clique(self):
        """Find the largest pairwise-incompatible basis clique."""
        candidates = self._collect_basis_clique_candidates()

        if not candidates:
            return [self.root]

        node_index = {node: index for index, node in enumerate(candidates)}

        adjacency_masks = [0] * len(candidates)

        for left_index, left_node in enumerate(candidates):
            for right_index in range(left_index):
                right_node = candidates[right_index]

                if Apartness.states_are_incompatible(left_node, right_node, self):
                    adjacency_masks[left_index] |= 1 << right_index
                    adjacency_masks[right_index] |= 1 << left_index

        if self.replace_basis:
            # State 0 must remain the root.
            required_nodes = [self.root]
        else:
            # Keep the current basis fixed.
            required_nodes = list(self.guaranteed_basis)

        required_indices = []

        for node in required_nodes:
            index = node_index.get(node)

            if index is None:
                raise RuntimeError("A required basis node is missing from the clique "
                                   "candidate set")

            required_indices.append(index)

        clique_indices = self._maximum_clique(adjacency_masks, required_indices, )

        clique = [candidates[index] for index in clique_indices]

        # Keep the root first and use a deterministic order for the
        # remaining basis states.
        clique.sort(key=lambda node: (node is not self.root, len(self.get_access_sequence(node) or ()),
                                      tuple(self.get_access_sequence(node) or ()),))

        return clique

    def _rebuild_frontier_to_basis_dict(self):
        """Recompute all frontier domains after changing the basis."""
        basis = tuple(self.guaranteed_basis)
        basis_set = set(basis)

        new_frontier = {}
        queue = deque([self.root])
        seen = set()

        while queue:
            node = queue.popleft()

            if node in seen:
                continue

            seen.add(node)
            queue.extend(node.successors.values())

            if node in basis_set:
                continue

            new_frontier[node] = {basis_node for basis_node in basis if
                                  not Apartness.states_are_incompatible(node, basis_node, self)}

        self.frontier_to_basis_dict = new_frontier

    def promote_node_to_basis(self):
        """Replace the current basis by a larger maximum clique."""
        old_basis = tuple(self.guaranteed_basis)
        new_basis = self._find_maximum_basis_clique()

        if len(new_basis) <= len(old_basis):
            return False

        self.guaranteed_basis = new_basis
        self.size = max(self.size, len(new_basis))
        self._rebuild_frontier_to_basis_dict()

        logger.debug("Increasing basis size from %d to %d using maximum clique", len(old_basis), len(new_basis), )

        return True

    def make_frontiers_identified(self):
        """Identify all frontier nodes as far as possible."""
        extended = False

        for basis_node in self.guaranteed_basis:
            for letter in self.alphabet:
                frontier_node = basis_node.get_successor(letter)

                while self.identify_frontier(frontier_node):
                    extended = True
                    self.update_basis_candidates(frontier_node)

        return extended

    def identify_frontier(self, frontier_node):
        """Try to identify a specific frontier node."""
        if not self.frontier_to_basis_dict[frontier_node]:
            return False

        inputs_to_frontier = self.get_transfer_sequence(self.root, frontier_node, )

        witnesses = self._get_witnesses_bfs(frontier_node)

        for witness_sequence in witnesses:
            inputs = inputs_to_frontier + witness_sequence
            outputs, extended = self._get_output_sequence(inputs, query_mode="final", )

            self.insert_observation_sequence(inputs, outputs)

            if extended:
                return True

        return False

    def _get_witnesses_bfs(self, frontier_node):
        """Generate distinguishing sequences for a frontier node."""
        basis_candidates = self.frontier_to_basis_dict.get(frontier_node)
        witnesses = Apartness.get_distinguishing_sequences(basis_candidates, self, )

        for witness_sequence in witnesses:
            target = self.get_successor(witness_sequence, start_node=frontier_node, )

            if target is None or target.output is None:
                yield witness_sequence

    # ------------------------------------------------------------------
    # Hypothesis construction
    # ------------------------------------------------------------------

    def construct_hypothesis_states(self, output_mapping=None):
        """Construct the DFA states from the basis."""
        self.states_list = [DfaState(f"s{i}") for i in range(self.size)]

        for i, state in enumerate(self.states_list):
            state.is_accepting = output_mapping[i]

    def construct_hypothesis_transitions(self, transition_mapping=None):
        """Construct the DFA transitions."""
        for i, state in enumerate(self.states_list):
            for j, letter in enumerate(self.alphabet):
                state.transitions[letter] = (self.states_list[transition_mapping[i][j]])

    def construct_hypothesis(self, transition_mapping=None, output_mapping=None, ):
        """Construct the hypothesis DFA."""
        self.construct_hypothesis_states(output_mapping)
        self.construct_hypothesis_transitions(transition_mapping)

        hypothesis = Dfa(self.states_list[0], self.states_list)
        hypothesis.compute_prefixes()
        hypothesis.characterization_set = (hypothesis.compute_characterization_set(raise_warning=False))

        return hypothesis

    # ------------------------------------------------------------------
    # SAT encoding
    # ------------------------------------------------------------------

    def find_hypothesis(self):
        """Find a hypothesis using the pure one-hot SAT encoding."""
        start_time = time.time()
        n_states = self.size
        alphabet_size = len(self.alphabet)

        logger.debug("Trying to build hypothesis of size %d", n_states)
        logger.debug("Basis size: %d, Frontier size: %d", len(self.guaranteed_basis),
                     len(self.frontier_to_basis_dict), )

        if n_states <= 0:
            self.smt_time += time.time() - start_time
            return None, None

        nodes, node_index, edges = self._flatten_observation_tree()
        n_nodes = len(nodes)

        logger.debug("SAT encoding: %d nodes, %d relevant edges, %d states, "
                     "%d alphabet symbols", n_nodes, len(edges), n_states, alphabet_size, )

        basis_size = len(self.guaranteed_basis)

        if basis_size > n_states:
            self.smt_time += time.time() - start_time
            return None, None

        basis_index = {basis_node: state for state, basis_node in enumerate(self.guaranteed_basis)}

        mapping_domains = self._compute_mapping_domains(nodes, basis_index, basis_size, n_states, )

        allocator = _SatVariableAllocator()

        mapping_var = [{state: allocator.new() for state in domain} for domain in mapping_domains]

        transition_var = [[[allocator.new() for _ in range(n_states)] for _ in range(alphabet_size)] for _ in
                          range(n_states)]

        output_var = [allocator.new() for _ in range(n_states)]

        clauses = []

        # Every observation-tree node maps to exactly one DFA state.
        for node_mapping in mapping_var:
            _add_exactly_one(clauses, node_mapping.values(), allocator, )

        # Every DFA transition is deterministic.
        for state in transition_var:
            for literals in state:
                _add_exactly_one(clauses, literals, allocator, )

        self._add_functional_simulation(clauses, edges, mapping_var, transition_var, )

        self._add_basis_constraints(clauses, mapping_var, node_index, basis_index, )

        self._add_output_constraints(clauses, nodes, mapping_var, output_var, )

        self._add_bfs_symmetry_breaking(clauses, allocator, transition_var, basis_size, n_states, alphabet_size, )

        logger.debug("SAT formula: %d variables, %d clauses", allocator.next_var - 1, len(clauses), )

        solver = None

        try:
            solver = Cadical153(bootstrap_with=clauses)
            result = solver.solve()

            if result is False:
                logger.debug("UNSAT at hypothesis size %d", self.size, )
                self.smt_time += time.time() - start_time
                return None, None

            if result is True:
                transition_mapping, output_mapping = (
                    self._extract_sat_model(solver, transition_var, output_var, n_states, alphabet_size, ))

                self.smt_time += time.time() - start_time
                return transition_mapping, output_mapping

            logger.error("Unexpected CaDiCaL result: %r", result, )

            self.smt_time += time.time() - start_time
            return None, None

        except Exception:
            self.smt_time += time.time() - start_time
            logger.exception("CaDiCaL exception while finding hypothesis")
            return None, None

        finally:
            if solver is not None:
                solver.delete()

    def _flatten_observation_tree(self):
        """Return relevant observation-tree nodes and edges."""
        queue = deque([self.root])
        nodes = [self.root]
        node_index = {self.root: 0}
        alphabet_index = {letter: index for index, letter in enumerate(self.alphabet)}

        edges = []

        while queue:
            node = queue.popleft()
            source_index = node_index[node]

            for letter, successor in node.successors.items():
                if not successor.leads_to_known:
                    continue

                if successor not in node_index:
                    node_index[successor] = len(nodes)
                    nodes.append(successor)
                    queue.append(successor)

                edges.append((source_index, node_index[successor], alphabet_index[letter],))

        return nodes, node_index, edges

    @staticmethod
    def _add_functional_simulation(clauses, edges, mapping_var, transition_var, ):
        """Encode deterministic simulation using domain-pruned variables."""
        for source_index, target_index, letter_index in edges:
            source_mapping = mapping_var[source_index]
            target_mapping = mapping_var[target_index]

            for state, source_literal in source_mapping.items():
                transition_row = transition_var[state][letter_index]

                for target_state, transition_literal in enumerate(transition_row):
                    target_literal = target_mapping.get(target_state)

                    if target_literal is None:
                        # This target state is impossible for the
                        # successor observation-tree node.
                        clauses.append([-source_literal, -transition_literal])
                    else:
                        clauses.append([-source_literal, -transition_literal, target_literal, ])

    def _compute_mapping_domains(self, nodes, basis_index, basis_size, n_states, ):
        """Compute the currently possible DFA states for each tree node."""
        all_states = tuple(range(n_states))
        free_states = tuple(range(basis_size, n_states))
        domains = []

        for node in nodes:
            if node in basis_index:
                domains.append((basis_index[node],))
                continue

            candidates = self.frontier_to_basis_dict.get(node)

            if candidates is None:
                domains.append(all_states)
                continue

            compatible_basis_states = tuple(
                basis_index[candidate] for candidate in sorted(candidates, key=basis_index.__getitem__, ))

            domains.append(compatible_basis_states + free_states)

        logger.debug("Morphism domains: %d nodes, %d total possible mappings", len(domains),
                     sum(len(domain) for domain in domains), )

        return domains

    @staticmethod
    def _add_basis_constraints(clauses, mapping_var, node_index, basis_index, ):
        """Fix guaranteed basis nodes to their corresponding states."""
        for basis_node, state in basis_index.items():
            node_idx = node_index[basis_node]
            clauses.append([mapping_var[node_idx][state]])

    @staticmethod
    def _add_output_constraints(clauses, nodes, mapping_var, output_var, ):
        """Connect known observation outputs to DFA state outputs."""
        for node_idx, node in enumerate(nodes):
            if not ObservationTreeSquare.is_known(node):
                continue

            accepts = bool(node.output)

            for state, mapping_literal in mapping_var[node_idx].items():
                clauses.append([-mapping_literal, (output_var[state] if accepts else -output_var[state]), ])

    @staticmethod
    def _add_bfs_symmetry_breaking(clauses, allocator, transition_var, basis_size, n_states, alphabet_size, ):
        """Add basis-aware BFS symmetry-breaking constraints."""
        if basis_size >= n_states or alphabet_size == 0:
            return

        first = {}

        for state in range(basis_size, n_states):
            first[state] = [[allocator.new() for _ in range(alphabet_size)] for _ in range(state)]

            clauses.append([first[state][parent][letter] for parent in range(state) for letter in range(alphabet_size)])

            for parent in range(state):
                for letter in range(alphabet_size):
                    marker = first[state][parent][letter]

                    clauses.append([-marker, transition_var[parent][letter][state], ])

                    for smaller_parent in range(parent):
                        for smaller_letter in range(alphabet_size):
                            clauses.append([-marker, -transition_var[smaller_parent][smaller_letter][state], ])

                    for smaller_letter in range(letter):
                        clauses.append([-marker, -transition_var[parent][smaller_letter][state], ])

        # Require consecutive free states to be discovered in order.
        for state in range(basis_size, n_states - 1):
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
                    clauses.extend([[-current, marker], [-marker, current], ])
                else:
                    previous = prefix[rank - 1]

                    clauses.append([-current, previous, marker])
                    clauses.append([-previous, current])
                    clauses.append([-marker, current])

            # State state + 1 must be discovered strictly later.
            for rank in range(descriptor_count):
                parent = rank // alphabet_size
                letter = rank % alphabet_size
                marker_next = first[state + 1][parent][letter]

                if rank == 0:
                    clauses.append([-marker_next])
                else:
                    clauses.append([-marker_next, prefix[rank - 1], ])

    @staticmethod
    def _extract_sat_model(solver, transition_var, output_var, n_states, alphabet_size, ):
        """Extract transition and output tables from a SAT model."""
        model = set(solver.get_model())

        transition_mapping = [[0 for _ in range(alphabet_size)] for _ in range(n_states)]

        for state in range(n_states):
            for letter in range(alphabet_size):
                selected_target = next(
                    (target for target, literal in enumerate(transition_var[state][letter]) if literal in model),
                    None, )

                if selected_target is None:
                    raise RuntimeError(f"No transition selected for "
                                       f"delta({state}, {letter})")

                transition_mapping[state][letter] = selected_target

        output_mapping = [literal in model for literal in output_var]

        return transition_mapping, output_mapping

    def build_hypothesis(self):
        """Build a hypothesis DFA when the current size is satisfiable."""
        self.find_adequate_observation_tree()

        transition_mapping, output_mapping = self.find_hypothesis()

        if transition_mapping is None:
            self.size += 1
            return None

        return self.construct_hypothesis(transition_mapping=transition_mapping, output_mapping=output_mapping, )

    # ------------------------------------------------------------------
    # Frontier expansion and refinement
    # ------------------------------------------------------------------

    def expand_frontier(self):
        """
        Extend the frontier by self.size - len(self.guaranteed_basis) + 3.
        """
        length = (self.size - len(self.guaranteed_basis) + 3)

        for word in itertools.product(self.alphabet, repeat=length, ):
            for node in self.guaranteed_basis:
                access = self.get_access_sequence(node)
                inputs = access + list(word)

                outputs, _ = self._get_output_sequence(inputs, query_mode="full", )

                self.insert_observation_sequence(inputs, outputs, )

    def update_frontier(self):
        """Update frontier-to-basis candidate sets."""
        self.update_frontier_to_basis_dict()

    def find_adequate_observation_tree(self):
        """
        Find an observation tree in which frontier states are identified
        as much as possible.
        """
        self.expand_frontier()
        self.update_frontier_to_basis_dict()

        while self.promote_node_to_basis():
            self.expand_frontier()
            self.update_frontier_to_basis_dict()

        while self.make_frontiers_identified():
            self.update_frontier_to_basis_dict()

            while self.promote_node_to_basis():
                self.expand_frontier()
                self.update_frontier_to_basis_dict()

    def process_counter_example(self, cex_inputs, output):
        """
        Insert a counterexample and update the observation tree.
        """
        cex_outputs, _ = self._get_output_sequence(cex_inputs, query_mode="full", )

        self.insert_observation_sequence(cex_inputs, cex_outputs, )

        self.get_successor(cex_inputs).set_output(output)
        self.update_frontier_to_basis_dict()

    # ------------------------------------------------------------------
    # SUL interaction
    # ------------------------------------------------------------------

    def _get_output_sequence(self, inputs, query_mode="full"):
        """
        Return outputs for an input sequence.

        query_mode:
            "full"  - query every missing output,
            "none"  - never query,
            "final" - only query the final missing output.
        """
        assert query_mode in {"full", "none", "final"}

        outputs = []
        queried = False
        current_node = self.root

        for index, inp in enumerate(inputs):
            if current_node is not None:
                current_node = current_node.get_successor(inp)

            query_output = (query_mode == "full" or (query_mode == "final" and index == len(inputs) - 1))

            if current_node is None:
                if query_output:
                    new_output = self.sul.query(inputs[:index + 1])
                    outputs.append(new_output)

                    if new_output != "unknown":
                        queried = True
                else:
                    outputs.append(None)

                continue

            if current_node.output is None and query_output:
                new_output = self.sul.query(inputs[:index + 1])
                outputs.append(new_output)

                if new_output != "unknown":
                    queried = True
            else:
                outputs.append(current_node.output)

            # If the last output is unknown, and we assume prefix-closedness, we can stop querying further.
            # Fill the rest of the outputs with "unknown" to avoid unnecessary queries.
            if self.assume_prefix_closed and current_node.output == "unknown":
                outputs.extend(["unknown"] * (len(inputs) - index - 1))
                break

        return outputs, queried

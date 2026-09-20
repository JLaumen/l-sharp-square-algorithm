
import itertools
import logging
import time
from collections import deque

from pysat.card import CardEnc, EncType
from pysat.solvers import Cadical153

from aalpy.automata import Dfa, DfaState

from Apartness import Apartness
from MooreNode import MooreNode





logger = logging.getLogger(__name__)

PAIRWISE_THRESHOLD = 25


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


class _AcyclicObservationNode:
    """Node in the minimized acyclic observation automaton."""

    __slots__ = ("output", "successors", "members")

    def __init__(self, output, successors=None):
        self.output = output
        self.successors = successors or {}
        self.members = []


def _add_exactly_one(clauses, literals, allocator):
    """Add an exactly-one constraint using pairwise or sequential encoding."""
    clauses.append(list(literals))

    if len(literals) <= 1:
        return

    if len(literals) <= PAIRWISE_THRESHOLD:
        for left in range(len(literals)):
            for right in range(left + 1, len(literals)):
                clauses.append([-literals[left], -literals[right]])
        return

    cnf = CardEnc.atmost(
        lits=literals,
        bound=1,
        top_id=allocator.next_var - 1,
        encoding=EncType.seqcounter,
    )
    clauses.extend(cnf.clauses)
    allocator.advance_to(cnf.nv + 1)


class ObservationTreeSquare:
    def __init__(
        self,
        alphabet,
        sul,
        solver_timeout,
        replace_basis,
        use_compatibility,
        minimize_observation_tree=True,
    ):
        """
        Initializes the observation tree with a root node.
        """
        self.automaton_type = "dfa"
        self.solver_timeout = solver_timeout * 1000
        self.replace_basis = replace_basis
        self.use_compatibility = use_compatibility
        # Temporarily quotient the relevant observation tree before building
        # the SAT instance. The original observation tree is never modified.
        self.minimize_observation_tree = minimize_observation_tree

        # Logger information
        self.smt_time = 0
        MooreNode._id_counter = 0

        # Initialize tree
        self.alphabet = alphabet
        self.sul = sul
        self.outputAlphabet = [True, False, "unknown"]
        self.states_list = []

        self.root = MooreNode()
        self.root.set_output(self.sul.query([]))

        self.size = 1
        self.guaranteed_basis = [self.root]
        self.frontier_to_basis_dict = dict()

    def insert_observation(self, inputs, output):
        """
        Insert an observation into the tree using a sequence of inputs and the corresponding output.
        """
        node = self.root
        for inp in inputs:
            node = node.extend_and_get(inp, None)
        node.set_output(output)

    def insert_observation_sequence(self, inputs, outputs):
        """
        Insert an observation into the tree using a sequence of inputs and their corresponding outputs.
        """
        node = self.root
        for inp, output in zip(inputs, outputs):
            node = node.extend_and_get(inp, output)
            node.set_output(output)
            if not node in self.frontier_to_basis_dict:
                candidates = {candidate for candidate in self.guaranteed_basis}
                self.frontier_to_basis_dict[node] = candidates

    def experiment(self, inputs):
        """
        Perform an experiment by querying the SUL if necessary and updating the tree.
        """
        outputs, extended = self._get_output_sequence(inputs, query_mode='final')
        self.insert_observation_sequence(inputs, outputs)
        return outputs[-1]

    def get_successor(self, inputs, start_node=None):
        """
        Retrieve the node corresponding to the given input sequence
        """
        if start_node is None:
            node = self.root
        else:
            node = start_node
        for input_val in inputs:
            successor_node = node.get_successor(input_val)
            if successor_node is None:
                return None
            node = successor_node
        return node

    @staticmethod
    def get_transfer_sequence(start_node, end_node):
        """
        Get the sequence of inputs that moves from the start node to the end node.
        """
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
        """
        Get the sequence of inputs that moves from the root node to the target node.
        """
        transfer_sequence = []
        node = target_node

        while node != self.root:
            if node.parent is None:
                return None
            transfer_sequence.append(node.input_to_parent)
            node = node.parent

        transfer_sequence.reverse()
        return transfer_sequence

    def get_size(self):
        """
        Get the number of nodes in the observation tree.
        """
        return self.root.id_counter

    @staticmethod
    def is_known(node):
        """
        Check if the output of a node is known.
        """
        return node.output is not None and node.output != "unknown"

    def count_informative_nodes(self):
        """
        counts how many nodes have informative information
        """
        queue = deque()
        queue.append(self.root)
        count = 0
        while queue:
            node = queue.popleft()
            if node.output != "unknown":
                count += 1
            for successor in node.successors.values():
                queue.append(successor)
        return count

    def update_basis_candidates(self, frontier_node):
        """
        Update the basis candidates for a specific frontier node.
        """
        candidates = self.frontier_to_basis_dict[frontier_node]
        new_candidates = {node for node in candidates if
                          not Apartness.states_are_incompatible(frontier_node, node, self)}
        self.frontier_to_basis_dict[frontier_node] = new_candidates

    def update_frontier_to_basis_dict(self):
        """
        Update the basis candidates for all frontier nodes.
        """
        self.update_frontier_to_basis_dict_dfs(self.root)

    def update_frontier_to_basis_dict_dfs(self, node):
        if not node.leads_to_known:
            return
        if not node in self.guaranteed_basis:
            self.update_basis_candidates(node)
            if len(self.frontier_to_basis_dict[node]) == 0:
                return
        for successor in node.successors.values():
            self.update_frontier_to_basis_dict_dfs(successor)

    def promote_node_to_basis(self):
        """
        If an isolated frontier node is found, reset the queue and restart from the guaranteed basis plus the isolated node.
        """
        queue = deque([self.root])
        while queue:
            iso_frontier_node = queue.popleft()
            for successor in iso_frontier_node.successors.values():
                queue.append(successor)
            if iso_frontier_node in self.guaranteed_basis:
                continue
            basis_list = self.frontier_to_basis_dict[iso_frontier_node]
            if not basis_list:
                self.guaranteed_basis.append(iso_frontier_node)
                # Update the candidates
                del self.frontier_to_basis_dict[iso_frontier_node]
                for node, candidates in self.frontier_to_basis_dict.items():
                    candidates.add(iso_frontier_node)
                logger.debug(f"Increasing basis size to {len(self.guaranteed_basis)}")
                self.size = max(self.size, len(self.guaranteed_basis))
                return True

        if not self.replace_basis:
            return False

        queue = deque([self.root])
        while queue:
            iso_frontier_node = queue.popleft()
            for successor in iso_frontier_node.successors.values():
                queue.append(successor)
            if iso_frontier_node in self.guaranteed_basis:
                continue
            basis_list = self.frontier_to_basis_dict[iso_frontier_node]
            if len(basis_list) == 1:
                candidate = next(iter(self.frontier_to_basis_dict[iso_frontier_node]))
                if len(self.get_access_sequence(candidate)) <= len(self.get_access_sequence(iso_frontier_node)):
                    continue
                self.guaranteed_basis.remove(candidate)
                self.guaranteed_basis.append(iso_frontier_node)
                # Update the candidates
                del self.frontier_to_basis_dict[iso_frontier_node]
                for node, candidates in self.frontier_to_basis_dict.items():
                    if candidate in candidates:
                        candidates.remove(candidate)
                    candidates.add(iso_frontier_node)
                self.frontier_to_basis_dict[candidate] = {node for node in self.guaranteed_basis}
                return True
        return False

    def make_frontiers_identified(self):
        """
        Loop over all frontier nodes to identify them
        """
        extended = False
        for basis_node in self.guaranteed_basis:
            for letter in self.alphabet:
                frontier_node = basis_node.get_successor(letter)
                while self.identify_frontier(frontier_node):
                    extended = True
                    self.update_basis_candidates(frontier_node)
        return extended

    def identify_frontier(self, frontier_node):
        """
        Identify a specific frontier node
        """
        if len(self.frontier_to_basis_dict[frontier_node]) == 0:
            return False

        inputs_to_frontier = self.get_transfer_sequence(self.root, frontier_node)

        witnesses = self._get_witnesses_bfs(frontier_node)
        for witness_seq in witnesses:
            inputs = inputs_to_frontier + witness_seq
            outputs, extended = self._get_output_sequence(inputs, query_mode='final')
            self.insert_observation_sequence(inputs, outputs)
            if extended:
                return True
        return False

    def _get_witnesses_bfs(self, frontier_node):
        """
        Specifically identify frontier nodes using separating sequences
        """
        basis_candidates = self.frontier_to_basis_dict.get(frontier_node)
        witnesses = Apartness.get_distinguishing_sequences(basis_candidates, self)

        for witness_seq in witnesses:
            leads_to_node = self.get_successor(witness_seq, start_node=frontier_node)
            if leads_to_node is None or leads_to_node.output is None:
                yield witness_seq

    def construct_hypothesis_states(self, output_mapping=None):
        """
        Construct the hypothesis states from the basis
        """
        self.states_list = [DfaState(f's{i}') for i in range(self.size)]
        for i, dfa_state in enumerate(self.states_list):
            dfa_state.is_accepting = output_mapping[i]

    def construct_hypothesis_transitions(self, transition_mapping=None):
        """
        Construct the hypothesis transitions using the transition_mapping and output_mapping.
        """
        for i, dfa_state in enumerate(self.states_list):
            for j, letter in enumerate(self.alphabet):
                dfa_state.transitions[letter] = self.states_list[transition_mapping[i][j]]

    def construct_hypothesis(self, transition_mapping=None, output_mapping=None):
        """
        Constructs the hypothesis DFA from the transition and output mappings.
        """
        self.construct_hypothesis_states(output_mapping=output_mapping)
        self.construct_hypothesis_transitions(transition_mapping=transition_mapping)

        hypothesis = Dfa(self.states_list[0], self.states_list)
        hypothesis.compute_prefixes()
        hypothesis.characterization_set = hypothesis.compute_characterization_set(raise_warning=False)

        return hypothesis

    def find_hypothesis(self):
        """
        Find a DFA of the current size by:

        1. Minimizing the relevant observation tree exactly as DFAMiner
           minimizes an acyclic 3DFA: two states are merged when they have
           the same three-valued output and the same labelled successor
           structure.
        2. Finding a functional simulation (morphism) from that minimized
           acyclic automaton into the candidate DFA using a small one-hot SAT
           encoding.

        This minimized mode deliberately does not add Apartness constraints,
        BFS symmetry breaking, or product/reachability variables. It is the
        simple baseline requested for validating the minimization itself.

        The original observation tree is never modified.
        """
        start_time = time.time()
        n_states = self.size
        alphabet_size = len(self.alphabet)

        logger.debug(
            "Trying hypothesis size %d (simple DFAMiner-style minimization)",
            n_states,
        )

        if n_states <= 0:
            self.smt_time += time.time() - start_time
            return None, None

        (
            nodes,
            _node_index,
            edges,
        ) = self._build_dfaminer_minimized_graph()

        n_nodes = len(nodes)
        logger.debug(
            "Minimized 3DFA: %d states, %d transitions; target DFA: %d states",
            n_nodes,
            len(edges),
            n_states,
        )

        allocator = _SatVariableAllocator()
        clauses = []

        # ------------------------------------------------------------------
        # Target DFA transitions: e[q,a,r]
        # ------------------------------------------------------------------
        transition_var = [
            [
                [allocator.new() for _ in range(n_states)]
                for _ in range(alphabet_size)
            ]
            for _ in range(n_states)
        ]

        for state in transition_var:
            for literals in state:
                _add_exactly_one(clauses, literals, allocator)

        # ------------------------------------------------------------------
        # Target DFA outputs: out[q]
        # ------------------------------------------------------------------
        output_var = [allocator.new() for _ in range(n_states)]

        # ------------------------------------------------------------------
        # Morphism: m[p,q]
        #
        # m[p,q] means minimized 3DFA state p is mapped to target DFA state q.
        # Every minimized state has exactly one target state.
        # ------------------------------------------------------------------
        mapping_var = [
            [allocator.new() for _ in range(n_states)]
            for _ in range(n_nodes)
        ]

        for literals in mapping_var:
            _add_exactly_one(clauses, literals, allocator)

        # The root of the minimized 3DFA is mapped to the initial DFA state 0.
        clauses.append([mapping_var[0][0]])

        # ------------------------------------------------------------------
        # Functional simulation.
        #
        # For p -a-> p' and every q,r:
        #     m[p,q] & e[q,a,r] -> m[p',r]
        # ------------------------------------------------------------------
        self._add_functional_simulation(
            clauses,
            edges,
            mapping_var,
            transition_var,
        )

        # ------------------------------------------------------------------
        # Three-valued output constraints.
        # ------------------------------------------------------------------
        for node_index, node in enumerate(nodes):
            if not self.is_known(node):
                continue

            if bool(node.output):
                for state in range(n_states):
                    clauses.append([
                        -mapping_var[node_index][state],
                        output_var[state],
                    ])
            else:
                for state in range(n_states):
                    clauses.append([
                        -mapping_var[node_index][state],
                        -output_var[state],
                    ])

        logger.debug(
            "Simple minimized SAT formula: %d variables, %d clauses",
            allocator.next_var - 1,
            len(clauses),
        )

        solver = None
        try:
            solver = Cadical153(bootstrap_with=clauses)
            result = solver.solve()

            if result is False:
                logger.debug("UNSAT at hypothesis size %d", n_states)
                self.smt_time += time.time() - start_time
                return None, None

            if result is True:
                transition_mapping, output_mapping = self._extract_sat_model(
                    solver,
                    transition_var,
                    output_var,
                    n_states,
                    alphabet_size,
                )
                self.smt_time += time.time() - start_time
                return transition_mapping, output_mapping

            logger.error("Unexpected CaDiCaL result: %r", result)
            self.smt_time += time.time() - start_time
            return None, None

        except Exception:
            self.smt_time += time.time() - start_time
            logger.exception("CaDiCaL exception while finding hypothesis")
            return None, None
        finally:
            if solver is not None:
                solver.delete()

    def _build_dfaminer_minimized_graph(self):
        """
        Build the relevant acyclic 3DFA and minimize it exactly in the style
        of DFAMiner's acyclic minimizer.

        The relevant observation structure is obtained by keeping only nodes
        that are reachable from the root and can reach a known output. This is
        the same useful-state reduction performed by DFAMiner before its
        bottom-up acyclic minimization.

        A state signature consists only of:

            (three-valued output,
             defined transition label -> minimized successor class)

        No Apartness, basis identity, or SAT-specific information is included
        in the signature. This is intentional: the goal here is to reproduce
        the plain DFAMiner-style 3DFA minimization as closely as possible.
        """
        alphabet_index = {
            letter: index
            for index, letter in enumerate(self.alphabet)
        }

        # --------------------------------------------------------------
        # Build the useful part of the observation tree.
        # --------------------------------------------------------------
        queue = deque([self.root])
        original_nodes = [self.root]
        seen = {self.root}

        while queue:
            node = queue.popleft()
            for successor in node.successors.values():
                if not successor.leads_to_known:
                    continue
                if successor in seen:
                    continue
                seen.add(successor)
                original_nodes.append(successor)
                queue.append(successor)

        # --------------------------------------------------------------
        # DFAMiner-style bottom-up minimization.
        #
        # For an acyclic deterministic automaton, process leaves first.
        # A state is identified by its three-valued output and the classes
        # reached by each of its defined transitions.
        # --------------------------------------------------------------
        representative = {}
        signature_to_class = {}
        minimized_nodes = []

        def output_type(node):
            if node.output is True:
                return "accept"
            if node.output is False:
                return "reject"
            return "dontcare"

        for node in reversed(original_nodes):
            successor_signature = tuple(
                (
                    alphabet_index[letter],
                    representative[successor],
                )
                for letter, successor in sorted(
                    node.successors.items(),
                    key=lambda item: alphabet_index[item[0]],
                )
                if successor.leads_to_known
            )

            signature = (
                output_type(node),
                successor_signature,
            )

            quotient_state = signature_to_class.get(signature)
            if quotient_state is None:
                quotient_state = len(minimized_nodes)
                signature_to_class[signature] = quotient_state

                minimized_nodes.append(
                    _AcyclicObservationNode(node.output)
                )

                minimized_nodes[quotient_state].successors = {
                    letter: representative[successor]
                    for letter, successor in sorted(
                        node.successors.items(),
                        key=lambda item: alphabet_index[item[0]],
                    )
                    if successor.leads_to_known
                }

            representative[node] = quotient_state

        # --------------------------------------------------------------
        # Re-index from the root. DFAMiner's resulting states are numbered by
        # the order in which the minimized automaton is traversed; using BFS
        # here gives deterministic compact numbering and ensures state 0 is
        # the initial state for the SAT encoding.
        # --------------------------------------------------------------
        root_old = representative[self.root]
        old_to_new = {root_old: 0}
        ordered_old_states = [root_old]
        queue = deque([root_old])

        while queue:
            old_state = queue.popleft()
            node = minimized_nodes[old_state]

            for successor in node.successors.values():
                if successor not in old_to_new:
                    old_to_new[successor] = len(ordered_old_states)
                    ordered_old_states.append(successor)
                    queue.append(successor)

        nodes = [
            _AcyclicObservationNode(
                minimized_nodes[old_state].output
            )
            for old_state in ordered_old_states
        ]

        for old_state, new_state in old_to_new.items():
            nodes[new_state].successors = {
                letter: old_to_new[successor]
                for letter, successor in minimized_nodes[old_state].successors.items()
            }

        edges = []
        for source, node in enumerate(nodes):
            for letter, target in sorted(
                node.successors.items(),
                key=lambda item: alphabet_index[item[0]],
            ):
                edges.append((source, target, alphabet_index[letter]))

        logger.debug(
            "DFAMiner-style 3DFA minimization: %d -> %d states",
            len(original_nodes),
            len(nodes),
        )

        return nodes, {index: index for index in range(len(nodes))}, edges

    @staticmethod
    def _add_functional_simulation(clauses, edges, mapping_var, transition_var):
        """Encode a functional simulation from the source graph to the DFA."""
        for source_idx, target_idx, letter_idx in edges:
            source_mapping = mapping_var[source_idx]
            target_mapping = mapping_var[target_idx]

            for source_state in range(len(transition_var)):
                source_literal = source_mapping[source_state]
                transition_row = transition_var[source_state][letter_idx]

                for target_state, transition_literal in enumerate(transition_row):
                    clauses.append([
                        -source_literal,
                        -transition_literal,
                        target_mapping[target_state],
                    ])

    @staticmethod
    def _extract_sat_model(
        solver,
        transition_var,
        output_var,
        n_states,
        alphabet_size,
    ):
        """Extract the target DFA transition and output tables from a SAT model."""
        model = solver.get_model()
        model_set = set(model)

        transition_mapping = [
            [0 for _ in range(alphabet_size)]
            for _ in range(n_states)
        ]

        for state in range(n_states):
            for letter in range(alphabet_size):
                selected_target = None

                for target in range(n_states):
                    if transition_var[state][letter][target] in model_set:
                        selected_target = target
                        break

                if selected_target is None:
                    raise RuntimeError(
                        f"No transition selected for delta({state}, {letter})"
                    )

                transition_mapping[state][letter] = selected_target

        output_mapping = [
            output_var[state] in model_set
            for state in range(n_states)
        ]

        return transition_mapping, output_mapping

    def build_hypothesis(self):
        """
        Builds the hypothesis which will be sent to the SUL and checks consistency
        """
        while True:
            self.find_adequate_observation_tree()
            transition_mapping, output_mapping = self.find_hypothesis()
            if transition_mapping is not None:
                hypothesis = self.construct_hypothesis(transition_mapping=transition_mapping,
                                                       output_mapping=output_mapping)
                return hypothesis
            else:
                self.size += 1
                return None

    def expand_frontier(self):
        """
        Extend the frontier self.size - len(self.guaranteed_basis) steps from the guaranteed basis
        """
        length = self.size - len(self.guaranteed_basis) + 3
        # length = 2
        # Loop over words of length 'length'
        for word in itertools.product(self.alphabet, repeat=length):
            for node in self.guaranteed_basis:
                access = self.get_access_sequence(node)
                inputs = access + list(word)
                outputs, _ = self._get_output_sequence(inputs, query_mode="full")
                self.insert_observation_sequence(inputs, outputs)

    def update_frontier(self):
        self.update_frontier_to_basis_dict()

    def find_adequate_observation_tree(self):
        """
        Tries to find an observation tree,
        for which each frontier state is identified as much as possible.
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
        Inserts the counter example into the observation tree and searches for the
        input-output sequence which is different
        """
        cex_outputs, _ = self._get_output_sequence(cex_inputs, query_mode="full")
        self.insert_observation_sequence(cex_inputs, cex_outputs)
        self.get_successor(cex_inputs).set_output(output)
        self.update_frontier_to_basis_dict()
        return

    def _get_output_sequence(self, inputs, query_mode="full"):
        """
        Returns the sequence of outputs corresponding to the input path.
        The knowledge is obtained from the observation tree or if not available via querying the sul.
        There are 3 query_modes: full, none and final. They allow you to restrict the querying to your needs
        """
        assert query_mode in ["full", "none", "final"]

        outputs = []
        queried = False
        current_node = self.root
        for inp_num in range(len(inputs)):
            inp = inputs[inp_num]
            if current_node is not None:
                current_node = current_node.get_successor(inp)
            if current_node is None:
                if query_mode == "full" or (inp_num == len(inputs) - 1 and query_mode == "final"):
                    new_output = self.sul.query(inputs[:inp_num + 1])
                    outputs.append(new_output)
                    if new_output != "unknown":
                        queried = True
                else:
                    outputs.append(None)
            else:
                if current_node.output is None and (
                        query_mode == "full" or (inp_num == len(inputs) - 1 and query_mode == "final")):
                    new_output = self.sul.query(inputs[:inp_num + 1])
                    outputs.append(new_output)
                    if new_output != "unknown":
                        queried = True
                else:
                    outputs.append(current_node.output)
        return outputs, queried

# Put this outside the ObservationTreeSquare class.

def _run_cadical_worker(clauses, result_queue):
    """
    Run CaDiCaL in a separate process for an optional wall-clock timeout.

    This helper is intentionally unused by ``find_hypothesis`` at present.
    """
    from pysat.solvers import Cadical153

    solver = None

    try:
        solver = Cadical153(
            bootstrap_with=clauses
        )

        result = solver.solve()

        if result:
            result_queue.put(
                ("sat", solver.get_model())
            )
        else:
            result_queue.put(
                ("unsat", None)
            )

    except Exception as exc:
        result_queue.put(
            ("error", repr(exc))
        )

    finally:
        if solver is not None:
            solver.delete()


def _run_cadical_with_timeout(clauses, timeout_ms):
    """
    Run CaDiCaL with a wall-clock timeout.

    This helper is intentionally dormant; ``find_hypothesis`` currently calls
    ``Cadical153.solve()`` directly.

    Returns:
        ("sat", model)
        ("unsat", None)
        ("timeout", None)
        ("error", message)

    When fork is available, the CNF is inherited by the child rather
    than serialized through multiprocessing, which keeps the overhead
    much lower on Linux.
    """
    import multiprocessing as mp

    # --------------------------------------------------------------
    # No timeout requested.
    # --------------------------------------------------------------

    if timeout_ms is None or timeout_ms <= 0:

        from pysat.solvers import Cadical153

        solver = None

        try:
            solver = Cadical153(
                bootstrap_with=clauses
            )

            if solver.solve():
                return "sat", solver.get_model()

            return "unsat", None

        except Exception as exc:
            return "error", repr(exc)

        finally:
            if solver is not None:
                solver.delete()

    # --------------------------------------------------------------
    # Prefer fork when available.
    # --------------------------------------------------------------

    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context("spawn")

    result_queue = ctx.Queue(
        maxsize=1
    )

    process = ctx.Process(
        target=_run_cadical_worker,
        args=(clauses, result_queue)
    )

    process.daemon = True

    try:
        process.start()

        process.join(
            timeout_ms / 1000.0
        )

        # ----------------------------------------------------------
        # Timeout.
        # ----------------------------------------------------------

        if process.is_alive():

            process.terminate()
            process.join()

            return "timeout", None

        # ----------------------------------------------------------
        # Child finished.
        # ----------------------------------------------------------

        try:
            return result_queue.get_nowait()

        except Exception:
            return (
                "error",
                "CaDiCaL process terminated without returning "
                "a result."
            )

    finally:

        if process.is_alive():
            process.terminate()
            process.join()

        result_queue.close()
        result_queue.join_thread()

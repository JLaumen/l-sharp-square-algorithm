import itertools
import logging
import time
from collections import deque

from aalpy.automata import Dfa, DfaState
from pysmt.exceptions import SolverReturnedUnknownResultError
from pysmt.shortcuts import (Solver, Symbol, Function, Int, Bool, Or, GE, LT)
from pysmt.typing import INT, BOOL, FunctionType

from Apartness import Apartness
from MooreNode import MooreNode

test_cases_path = "Benchmarking/incomplete_dfa_benchmark/test_cases/"
logging.basicConfig(level=logging.INFO, format=f"%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")


class ObservationTreeSquare:
    def __init__(self, alphabet, sul, solver_timeout, replace_basis, use_compatibility):
        """
        Initializes the observation tree with a root node.
        """
        self.automaton_type = "dfa"
        self.solver_timeout = solver_timeout * 1000
        self.replace_basis = replace_basis
        self.use_compatibility = use_compatibility

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
                logging.debug(f"Increasing basis size to {len(self.guaranteed_basis)}")
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
        Find a hypothesis using a SAT encoding.

        Encoding
        --------
        x[v][q]:
            observation-tree node v is mapped to DFA state q.

        state_bits[v][k]:
            binary representation of the state of observation node v.

        delta_bits[q][a][k]:
            k-th bit of delta(q, a).

        output_var[q]:
            DFA state q is accepting.

        The observation-tree state representation is one-hot. This is
        useful because the Apartness/frontier restrictions can be encoded
        directly as large SAT clauses.

        The transition function is represented in binary, avoiding the
        O(n^2) transition-target encoding.

        The SAT solver is run in a separate process so that
        self.solver_timeout is enforced as a real wall-clock timeout.
        """

        import logging
        import math
        import multiprocessing as mp
        import time

        from pysat.card import CardEnc, EncType

        logging.debug(f"Trying to build hypothesis of size {self.size}")
        logging.debug(
            f"Basis size: {len(self.guaranteed_basis)}, "
            f"Frontier size: {len(self.frontier_to_basis_dict)}"
        )

        start_sat_time = time.time()

        n_states = self.size
        alphabet_size = len(self.alphabet)

        # ==============================================================
        # Special case: one-state automaton
        # ==============================================================

        if n_states <= 0:
            self.smt_time += time.time() - start_sat_time
            return None, None

        # ==============================================================
        # Number of bits needed to represent DFA states
        # ==============================================================

        state_bits_count = max(
            1,
            math.ceil(math.log2(max(2, n_states)))
        )

        # ==============================================================
        # Flatten observation tree
        # ==============================================================

        queue = deque([self.root])

        nodes = [self.root]

        node_index = {
            self.root: 0
        }

        alphabet_index = {
            letter: i
            for i, letter in enumerate(self.alphabet)
        }

        # Relevant observation-tree edges:
        #
        #     source --letter--> target
        #
        edges = []

        while queue:

            node = queue.popleft()
            source_idx = node_index[node]

            for letter, successor in node.successors.items():

                # Preserve the exact restriction from the original
                # SMT encoding.
                if not successor.leads_to_known:
                    continue

                if successor not in node_index:

                    target_idx = len(nodes)

                    nodes.append(successor)
                    node_index[successor] = target_idx

                    queue.append(successor)

                else:

                    target_idx = node_index[successor]

                edges.append(
                    (
                        source_idx,
                        target_idx,
                        alphabet_index[letter]
                    )
                )

        n_nodes = len(nodes)

        logging.debug(
            f"SAT encoding: "
            f"{n_nodes} observation nodes, "
            f"{n_states} states, "
            f"{alphabet_size} alphabet symbols, "
            f"{state_bits_count} state bits"
        )

        # ==============================================================
        # SAT variable allocation
        # ==============================================================

        next_var = 1

        # --------------------------------------------------------------
        # x[v][q]
        #
        # True iff observation-tree node v represents DFA state q.
        # --------------------------------------------------------------

        state_var = [
            [0 for _ in range(n_states)]
            for _ in range(n_nodes)
        ]

        for v in range(n_nodes):

            for q in range(n_states):

                state_var[v][q] = next_var
                next_var += 1

        # --------------------------------------------------------------
        # Binary representation of observation-tree states.
        #
        # state_bits[v][k] is bit k of state(v).
        # --------------------------------------------------------------

        state_bits = [
            [0 for _ in range(state_bits_count)]
            for _ in range(n_nodes)
        ]

        for v in range(n_nodes):

            for k in range(state_bits_count):

                state_bits[v][k] = next_var
                next_var += 1

        # --------------------------------------------------------------
        # Binary transition representation.
        #
        # delta_bits[q][a][k] is bit k of delta(q,a).
        # --------------------------------------------------------------

        delta_bits = [
            [
                [0 for _ in range(state_bits_count)]
                for _ in range(alphabet_size)
            ]
            for _ in range(n_states)
        ]

        for q in range(n_states):

            for a in range(alphabet_size):

                for k in range(state_bits_count):

                    delta_bits[q][a][k] = next_var
                    next_var += 1

        # --------------------------------------------------------------
        # DFA output variables.
        # --------------------------------------------------------------

        output_var = [
            0
            for _ in range(n_states)
        ]

        for q in range(n_states):

            output_var[q] = next_var
            next_var += 1

        # ==============================================================
        # CNF
        # ==============================================================

        clauses = []

        # ==============================================================
        # Exactly-one state per observation-tree node
        # ==============================================================

        # For small n, pairwise encoding avoids auxiliary variables.
        # For larger n, use a sequential counter.

        PAIRWISE_THRESHOLD = 25

        for v in range(n_nodes):

            lits = [
                state_var[v][q]
                for q in range(n_states)
            ]

            # At least one.
            clauses.append(lits)

            # At most one.
            if n_states > 1:

                if n_states <= PAIRWISE_THRESHOLD:

                    for q1 in range(n_states):

                        for q2 in range(q1 + 1, n_states):

                            clauses.append(
                                [
                                    -state_var[v][q1],
                                    -state_var[v][q2]
                                ]
                            )

                else:

                    cnf = CardEnc.atmost(
                        lits=lits,
                        bound=1,
                        top_id=next_var - 1,
                        encoding=EncType.seqcounter
                    )

                    clauses.extend(cnf.clauses)

                    next_var = max(
                        next_var,
                        cnf.nv + 1
                    )

        # ==============================================================
        # Channel one-hot state -> binary state
        # ==============================================================

        # x[v,q] -> binary(state_bits[v]) == q
        #
        # Since exactly one x[v,q] is true, this uniquely determines
        # all state_bits[v].

        for v in range(n_nodes):

            for q in range(n_states):

                x = state_var[v][q]

                for k in range(state_bits_count):

                    bit = state_bits[v][k]

                    if (q >> k) & 1:

                        # x -> bit
                        clauses.append(
                            [
                                -x,
                                bit
                            ]
                        )

                    else:

                        # x -> !bit
                        clauses.append(
                            [
                                -x,
                                -bit
                            ]
                        )

        # ==============================================================
        # Observation-tree transitions
        # ==============================================================

        # For an edge:
        #
        #     v --a--> u
        #
        # if v == q then u == delta(q,a).
        #
        # For every state bit:
        #
        #     x[v,q] -> (state_bits[u][k] == delta_bits[q][a][k])
        #
        # This requires two ternary clauses.

        for source_idx, target_idx, letter_idx in edges:

            for q in range(n_states):

                source_literal = state_var[source_idx][q]

                for k in range(state_bits_count):

                    d = delta_bits[q][letter_idx][k]
                    b = state_bits[target_idx][k]

                    # x AND d -> b
                    clauses.append(
                        [
                            -source_literal,
                            -d,
                            b
                        ]
                    )

                    # x AND b -> d
                    clauses.append(
                        [
                            -source_literal,
                            d,
                            -b
                        ]
                    )

        # ==============================================================
        # Basis nodes
        # ==============================================================

        # Basis node i must be state i.

        for i, basis_node in enumerate(self.guaranteed_basis):

            v = node_index[basis_node]

            clauses.append(
                [
                    state_var[v][i]
                ]
            )

        # ==============================================================
        # Known outputs
        # ==============================================================

        for v, node in enumerate(nodes):

            if not self.is_known(node):
                continue

            if node.output is True:

                # state(v) = q -> output(q)
                for q in range(n_states):

                    clauses.append(
                        [
                            -state_var[v][q],
                            output_var[q]
                        ]
                    )

            elif node.output is False:

                # state(v) = q -> !output(q)
                for q in range(n_states):

                    clauses.append(
                        [
                            -state_var[v][q],
                            -output_var[q]
                        ]
                    )

        # ==============================================================
        # Apartness / frontier constraints
        # ==============================================================

        # This deliberately preserves the redundant information from the
        # original SMT encoding.
        #
        # For a frontier node v:
        #
        #   state(v) must be one of
        #
        #       compatible basis states
        #
        #   or
        #
        #       any state not yet represented by the guaranteed basis.
        #
        # Since x[v,q] is one-hot, this becomes one large SAT clause.

        basis_size = len(self.guaranteed_basis)

        for node, candidates in self.frontier_to_basis_dict.items():

            if node not in node_index:
                continue

            v = node_index[node]

            allowed_literals = []

            # Compatible basis states.
            for candidate in candidates:

                candidate_idx = self.guaranteed_basis.index(
                    candidate
                )

                allowed_literals.append(
                    state_var[v][candidate_idx]
                )

            # States not currently represented by the basis.
            for q in range(basis_size, n_states):

                allowed_literals.append(
                    state_var[v][q]
                )

            if not allowed_literals:

                # Explicit contradiction.
                clauses.append([])

            else:

                clauses.append(
                    allowed_literals
                )

        # ==============================================================
        # Logging formula size
        # ==============================================================

        logging.debug(
            f"SAT formula: "
            f"{next_var - 1} variables, "
            f"{len(clauses)} clauses"
        )

        # ==============================================================
        # Run CaDiCaL in a separate process
        # ==============================================================

        #
        # fork is preferable here on Linux because the CNF does not need
        # to be serialized and copied through a multiprocessing Queue.
        #
        # Your environment is Linux, so this should be available.
        #

        try:
            ctx = mp.get_context("fork")
        except ValueError:
            # Fallback for platforms where fork is unavailable.
            ctx = mp.get_context()

        result_queue = ctx.Queue(
            maxsize=1
        )

        process = ctx.Process(
            target=_solve_cadical_process,
            args=(
                clauses,
                result_queue
            )
        )

        try:

            process.start()

            timeout_seconds = (
                self.solver_timeout / 1000.0
            )

            # ----------------------------------------------------------
            # Wait for the SAT process.
            # ----------------------------------------------------------

            process.join(
                timeout_seconds
            )

            # ----------------------------------------------------------
            # Timeout.
            # ----------------------------------------------------------

            if process.is_alive():

                logging.debug(
                    f"SAT TIMEOUT after "
                    f"{timeout_seconds:.3f}s"
                )

                process.terminate()

                process.join()

                self.smt_time += (
                    time.time() - start_sat_time
                )

                return None, None

            # ----------------------------------------------------------
            # Process finished.
            # ----------------------------------------------------------

            try:

                status, model = result_queue.get_nowait()

            except Exception:

                logging.error(
                    "SAT process terminated without returning a result"
                )

                self.smt_time += (
                    time.time() - start_sat_time
                )

                return None, None

            # ----------------------------------------------------------
            # SAT solver error.
            # ----------------------------------------------------------

            if status == "error":

                logging.error(
                    f"SAT solver error: {model}"
                )

                self.smt_time += (
                    time.time() - start_sat_time
                )

                return None, None

            # ----------------------------------------------------------
            # UNSAT.
            # ----------------------------------------------------------

            if status == "unsat":

                logging.debug("UNSAT")
                logging.debug(
                    f"No hypothesis of size {self.size} exists"
                )

                self.smt_time += (
                    time.time() - start_sat_time
                )

                return None, None

            # ----------------------------------------------------------
            # SAT.
            # ----------------------------------------------------------

            if status != "sat":

                logging.error(
                    f"Unexpected SAT process status: {status}"
                )

                self.smt_time += (
                    time.time() - start_sat_time
                )

                return None, None

            logging.debug("SAT")

            # ==========================================================
            # Convert model to a set for O(1) lookup.
            # ==========================================================

            model_set = set(model)

            # ==========================================================
            # Extract transition mapping
            # ==========================================================

            transition_mapping = [
                [0 for _ in range(alphabet_size)]
                for _ in range(n_states)
            ]

            for q in range(n_states):

                for a in range(alphabet_size):

                    value = 0

                    for k in range(state_bits_count):

                        variable = delta_bits[q][a][k]

                        if variable in model_set:

                            value |= (
                                1 << k
                            )

                    # If this transition never occurs in the
                    # observation tree, its binary representation may
                    # represent a value >= n_states because we did not
                    # needlessly constrain unused transitions.
                    #
                    # Complete such transitions arbitrarily.
                    if value >= n_states:

                        value = 0

                    transition_mapping[q][a] = value

            # ==========================================================
            # Extract output mapping
            # ==========================================================

            output_mapping = [
                False
                for _ in range(n_states)
            ]

            for q in range(n_states):

                output_mapping[q] = (
                    output_var[q] in model_set
                )

            self.smt_time += (
                time.time() - start_sat_time
            )

            return (
                transition_mapping,
                output_mapping
            )

        finally:

            # Make absolutely sure that the child is gone.
            if process.is_alive():

                process.terminate()
                process.join()

            result_queue.close()
            result_queue.join_thread()

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

def _solve_cadical_process(clauses, result_queue):
    """
    Run CaDiCaL in a separate process.

    The parent process can terminate this process if the wall-clock
    timeout is exceeded.
    """
    from pysat.solvers import Solver

    solver = None

    try:
        solver = Solver(
            name="cadical153",
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

    except Exception as e:
        result_queue.put(
            ("error", repr(e))
        )

    finally:
        if solver is not None:
            solver.delete()

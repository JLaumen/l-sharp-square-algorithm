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
        SAT-based hypothesis construction using:

        * Heule-Verwer style state/transition variables
        * compact exactly-one encodings
        * basis-aware compact BFS symmetry breaking
        * Apartness/domain constraints
        * CaDiCaL 1.5.3

        State variables:

            m[v][q]

        meaning observation-tree node v is represented by DFA state q.

        Transition variables:

            e[q][a][r]

        meaning delta(q,a) = r.

        Output variables:

            out[q]

        meaning q is accepting.

        BFS symmetry breaking is applied only to states which are not
        already fixed by guaranteed_basis.

        This is important because guaranteed_basis may not itself be
        globally BFS ordered.
        """

        import logging
        import math
        import multiprocessing as mp
        import time

        from pysat.card import CardEnc, EncType

        logging.debug(
            f"Trying to build hypothesis of size {self.size}"
        )

        logging.debug(
            f"Basis size: {len(self.guaranteed_basis)}, "
            f"Frontier size: {len(self.frontier_to_basis_dict)}"
        )

        start_time = time.time()

        n_states = self.size
        alphabet_size = len(self.alphabet)

        if n_states <= 0:
            self.smt_time += time.time() - start_time
            return None, None

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

        edges = []

        while queue:

            node = queue.popleft()

            source_idx = node_index[node]

            for letter, successor in node.successors.items():

                # Exactly the same restriction as your original encoding.
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
            f"{n_nodes} nodes, "
            f"{n_states} states, "
            f"{alphabet_size} alphabet symbols"
        )

        # ==============================================================
        # SAT variable allocation
        # ==============================================================

        next_var = 1

        def new_var():
            nonlocal next_var

            v = next_var
            next_var += 1

            return v

        # --------------------------------------------------------------
        # m[v][q]
        #
        # Observation node v -> DFA state q
        # --------------------------------------------------------------

        mapping_var = [
            [
                new_var()
                for _ in range(n_states)
            ]
            for _ in range(n_nodes)
        ]

        # --------------------------------------------------------------
        # e[q][a][r]
        #
        # delta(q,a) = r
        # --------------------------------------------------------------

        transition_var = [
            [
                [
                    new_var()
                    for _ in range(n_states)
                ]
                for _ in range(alphabet_size)
            ]
            for _ in range(n_states)
        ]

        # --------------------------------------------------------------
        # out[q]
        # --------------------------------------------------------------

        output_var = [
            new_var()
            for _ in range(n_states)
        ]

        # ==============================================================
        # CNF
        # ==============================================================

        clauses = []

        # ==============================================================
        # Exactly-one helper
        # ==============================================================

        #
        # The 2019 compact SAT encoding points out that the original
        # pairwise AtMost1 encoding can be replaced by a more compact
        # cardinality encoding.
        #
        # In your previous experiments, however, pairwise encoding has
        # been very competitive. Therefore we use pairwise for small
        # domains and a sequential counter for larger ones.
        #

        PAIRWISE_THRESHOLD = 25

        def add_exactly_one(literals):

            nonlocal next_var

            # At least one.
            clauses.append(literals)

            if len(literals) <= 1:
                return

            if len(literals) <= PAIRWISE_THRESHOLD:

                for i in range(len(literals)):

                    for j in range(i + 1, len(literals)):

                        clauses.append(
                            [
                                -literals[i],
                                -literals[j]
                            ]
                        )

            else:

                cnf = CardEnc.atmost(
                    lits=literals,
                    bound=1,
                    top_id=next_var - 1,
                    encoding=EncType.seqcounter
                )

                clauses.extend(
                    cnf.clauses
                )

                next_var = max(
                    next_var,
                    cnf.nv + 1
                )

        # ==============================================================
        # Every observation-tree node maps to exactly one DFA state
        # ==============================================================

        for v in range(n_nodes):

            add_exactly_one(
                mapping_var[v]
            )

        # ==============================================================
        # Every DFA (state, letter) has exactly one successor
        # ==============================================================

        for q in range(n_states):

            for a in range(alphabet_size):

                add_exactly_one(
                    transition_var[q][a]
                )

        # ==============================================================
        # Functional simulation
        # ==============================================================

        #
        # For:
        #
        #     v --a--> u
        #
        # if:
        #
        #     m[v,q]
        #
        # and:
        #
        #     e[q,a,r]
        #
        # then:
        #
        #     m[u,r].
        #
        # This is the standard compact functional-simulation constraint:
        #
        #     !m[v,q] OR !e[q,a,r] OR m[u,r]
        #

        for source_idx, target_idx, letter_idx in edges:

            source_mapping = mapping_var[source_idx]

            target_mapping = mapping_var[target_idx]

            transition_row = [
                transition_var[q][letter_idx]
                for q in range(n_states)
            ]

            for q in range(n_states):

                m_source = source_mapping[q]

                for r in range(n_states):

                    clauses.append(
                        [
                            -m_source,
                            -transition_row[q][r],
                            target_mapping[r]
                        ]
                    )

        # ==============================================================
        # Basis states
        # ==============================================================

        #
        # Your existing learning algorithm has already fixed the identity
        # of these states:
        #
        #     basis[0] -> state 0
        #     basis[1] -> state 1
        #     ...
        #

        for q, basis_node in enumerate(
            self.guaranteed_basis
        ):

            if q >= n_states:
                raise RuntimeError(
                    "Basis is larger than requested hypothesis."
                )

            v = node_index[basis_node]

            clauses.append(
                [
                    mapping_var[v][q]
                ]
            )

        # ==============================================================
        # Known outputs
        # ==============================================================

        for v, node in enumerate(nodes):

            if not self.is_known(node):
                continue

            for q in range(n_states):

                if bool(node.output):

                    clauses.append(
                        [
                            -mapping_var[v][q],
                            output_var[q]
                        ]
                    )

                else:

                    clauses.append(
                        [
                            -mapping_var[v][q],
                            -output_var[q]
                        ]
                    )

        # ==============================================================
        # Apartness
        # ==============================================================

        #
        # Keep BOTH forms of the Apartness information:
        #
        # 1. Positive domain clause
        #
        #       m[v,c1] OR ... OR m[v,free_1] ...
        #
        # 2. Explicit negative clauses for incompatible basis states.
        #
        # The second form gives unit propagation; the first gives a useful
        # redundant long clause.
        #

        basis_size = len(self.guaranteed_basis)

        basis_index = {
            basis_node: i
            for i, basis_node in enumerate(
                self.guaranteed_basis
            )
        }

        for node, candidates in (
            self.frontier_to_basis_dict.items()
        ):

            if node not in node_index:
                continue

            v = node_index[node]

            candidate_set = set(candidates)

            allowed_literals = []

            # ----------------------------------------------------------
            # Compatible basis states
            # ----------------------------------------------------------

            for candidate in candidates:

                q = basis_index[candidate]

                allowed_literals.append(
                    mapping_var[v][q]
                )

            # ----------------------------------------------------------
            # All currently unused state IDs
            # ----------------------------------------------------------

            for q in range(
                basis_size,
                n_states
            ):

                allowed_literals.append(
                    mapping_var[v][q]
                )

            if not allowed_literals:

                clauses.append([])

            else:

                # Your original redundant Apartness clause.
                clauses.append(
                    allowed_literals
                )

            # ----------------------------------------------------------
            # Explicit incompatibility clauses
            # ----------------------------------------------------------

            for q in range(basis_size):

                if q not in {
                    basis_index[c]
                    for c in candidates
                }:

                    clauses.append(
                        [
                            -mapping_var[v][q]
                        ]
                    )

        # ==============================================================
        # Compact BFS symmetry breaking
        # ==============================================================

        #
        # IMPORTANT:
        #
        # The standard published BFS encoding assumes that ALL target DFA
        # states are numbered according to a BFS traversal from state 0.
        #
        # Your basis states are already named by the learning algorithm.
        #
        # Therefore we apply the same idea only to the states
        #
        #     basis_size, ..., n_states-1.
        #
        # These states are the states which remain symmetric after the
        # guaranteed basis has fixed its own names.
        #
        # For every free state q we identify its canonical first
        # discovery edge:
        #
        #     p --a--> q
        #
        # where p < q.
        #
        # We then force the discovery descriptors of q and q+1 to be
        # lexicographically increasing.
        #
        # This is a compact form of BFS symmetry breaking. The published
        # tight encoding uses analogous parent/transition/first-symbol
        # variables and reduces the basic BFS symmetry-breaking encoding
        # to quadratic-in-M times alphabet size.
        #

        free_states = range(
            basis_size,
            n_states
        )

        # --------------------------------------------------------------
        # no_source[q][p]
        #
        # True iff no state s <= p has any transition to q.
        #
        # q is a free state.
        # --------------------------------------------------------------

        no_source = {}

        # --------------------------------------------------------------
        # no_label[p][q][a]
        #
        # True iff state p has no transition to q on any symbol < a.
        #
        # We store only a >= 1.
        # --------------------------------------------------------------

        no_label = {}

        # --------------------------------------------------------------
        # first[q][p][a]
        #
        # True iff p --a--> q is the first discovery edge of q.
        # --------------------------------------------------------------

        first = {}

        for q in free_states:

            # A free state must have an incoming edge from a smaller
            # numbered state.
            #
            # We encode its first-discovery relation.

            no_source[q] = []

            # ==========================================================
            # no_source
            # ==========================================================

            for p in range(q):

                ns = new_var()

                no_source[q].append(ns)

                incoming_literals = [
                    transition_var[p][a][q]
                    for a in range(alphabet_size)
                ]

                if p == 0:

                    # ns <-> NOT(any transition 0 -> q)

                    for e in incoming_literals:

                        clauses.append(
                            [
                                -ns,
                                -e
                            ]
                        )

                    clauses.append(
                        incoming_literals + [ns]
                    )

                else:

                    previous = no_source[q][p - 1]

                    # ns -> previous

                    clauses.append(
                        [
                            -ns,
                            previous
                        ]
                    )

                    # ns -> no transition p -> q

                    for e in incoming_literals:

                        clauses.append(
                            [
                                -ns,
                                -e
                            ]
                        )

                    # previous AND no transition p->q -> ns

                    clauses.append(
                        [
                            -previous
                        ]
                        + incoming_literals
                        + [ns]
                    )

            # ==========================================================
            # first variables
            # ==========================================================

            first[q] = [
                [
                    new_var()
                    for _ in range(alphabet_size)
                ]
                for _ in range(q)
            ]

            # Every free state must have exactly one canonical discovery
            # edge. At-least-one is enough here because the definition of
            # "first" makes two different descriptors mutually exclusive.
            #

            clauses.append(
                [
                    first[q][p][a]
                    for p in range(q)
                    for a in range(alphabet_size)
                ]
            )

            # ==========================================================
            # no_label
            # ==========================================================

            for p in range(q):

                no_label[p] = (
                    no_label.get(p, {})
                )

                no_label[p][q] = [
                    None
                    for _ in range(alphabet_size)
                ]

                for a in range(1, alphabet_size):

                    nl = new_var()

                    no_label[p][q][a] = nl

                    current = transition_var[p][a - 1][q]

                    if a == 1:

                        # nl <-> !e[p,0,q]

                        clauses.append(
                            [
                                -nl,
                                -current
                            ]
                        )

                        clauses.append(
                            [
                                current,
                                nl
                            ]
                        )

                    else:

                        previous = no_label[p][q][a - 1]

                        # nl -> previous

                        clauses.append(
                            [
                                -nl,
                                previous
                            ]
                        )

                        # nl -> !current

                        clauses.append(
                            [
                                -nl,
                                -current
                            ]
                        )

                        # previous AND !current -> nl

                        clauses.append(
                            [
                                -previous,
                                current,
                                nl
                            ]
                        )

            # ==========================================================
            # Define first[q,p,a]
            # ==========================================================

            for p in range(q):

                for a in range(alphabet_size):

                    f = first[q][p][a]

                    e = transition_var[p][a][q]

                    # first -> actual transition

                    clauses.append(
                        [
                            -f,
                            e
                        ]
                    )

                    # --------------------------------------------------
                    # No earlier source may reach q.
                    # --------------------------------------------------

                    if p > 0:

                        clauses.append(
                            [
                                -f,
                                no_source[q][p - 1]
                            ]
                        )

                    # --------------------------------------------------
                    # No earlier alphabet symbol from p may reach q.
                    # --------------------------------------------------

                    if a > 0:

                        clauses.append(
                            [
                                -f,
                                no_label[p][q][a]
                            ]
                        )

                    # --------------------------------------------------
                    # Reverse implication:
                    #
                    # actual transition + no earlier source + no earlier
                    # symbol -> first.
                    # --------------------------------------------------

                    clause = [
                        -e
                    ]

                    if p > 0:

                        clause.append(
                            -no_source[q][p - 1]
                        )

                    if a > 0:

                        clause.append(
                            -no_label[p][q][a]
                        )

                    clause.append(f)

                    clauses.append(
                        clause
                    )

        # ==============================================================
        # Order first-discovery descriptors
        # ==============================================================

        #
        # Descriptor order is lexicographic on:
        #
        #     (parent state, alphabet symbol)
        #
        # We construct prefix variables:
        #
        #     prefix[q][k]
        #
        # meaning:
        #
        #     the first-discovery descriptor of q has rank <= k.
        #
        # Then:
        #
        #     discovery(q) < discovery(q+1)
        #
        # is enforced by:
        #
        #     first[q+1,k] -> prefix[q][k-1].
        #
        # This avoids the O(M^3 L^2) pairwise comparison.
        #

        for q in range(
            basis_size,
            n_states - 1
        ):

            # q has q*alphabet_size possible descriptors:
            #
            #   (0,0), (0,1), ..., (1,0), ..., (q-1,L-1)
            #

            num_descriptors = q * alphabet_size

            if num_descriptors == 0:
                continue

            prefix = [
                new_var()
                for _ in range(num_descriptors)
            ]

            for rank in range(num_descriptors):

                p = rank // alphabet_size
                a = rank % alphabet_size

                current = prefix[rank]

                f = first[q][p][a]

                if rank == 0:

                    # prefix[0] <-> first[q,0,0]

                    clauses.append(
                        [
                            -current,
                            f
                        ]
                    )

                    clauses.append(
                        [
                            -f,
                            current
                        ]
                    )

                else:

                    previous = prefix[rank - 1]

                    # current -> previous OR f

                    clauses.append(
                        [
                            -current,
                            previous,
                            f
                        ]
                    )

                    # previous -> current

                    clauses.append(
                        [
                            -previous,
                            current
                        ]
                    )

                    # f -> current

                    clauses.append(
                        [
                            -f,
                            current
                        ]
                    )

            # ----------------------------------------------------------
            # q+1 must be discovered after q.
            # ----------------------------------------------------------

            next_state = q + 1

            #
            # If next_state uses descriptor rank k, q must have a descriptor
            # of rank < k.
            #

            for rank in range(
                min(
                    num_descriptors,
                    next_state * alphabet_size
                )
            ):

                p = rank // alphabet_size
                a = rank % alphabet_size

                f_next = first[next_state][p][a]

                if rank == 0:

                    # Nothing is smaller than descriptor 0.

                    clauses.append(
                        [
                            -f_next
                        ]
                    )

                else:

                    clauses.append(
                        [
                            -f_next,
                            prefix[rank - 1]
                        ]
                    )

        # ==============================================================
        # Additional simple BFS shape constraints
        # ==============================================================

        #
        # A DFA state has only |Sigma| outgoing transitions, so in the
        # BFS tree it can have at most |Sigma| children.
        #
        # The first-discovery representation already implies this, but
        # making it explicit can improve propagation.
        #
        # For each parent p, at most alphabet_size free states may have
        # p as their canonical parent.
        #

        if alphabet_size > 0:

            for p in range(
                n_states
            ):

                children = []

                for q in range(
                    max(
                        basis_size,
                        p + 1
                    ),
                    n_states
                ):

                    if q not in first:
                        continue

                    children.extend(
                        first[q][p]
                    )

                #
                # The descriptors first[q][p][a] are mutually exclusive
                # for each q, and at most one such descriptor for p can
                # correspond to a given alphabet symbol.
                #
                # We therefore need to count states discovered from p,
                # not individual symbols.
                #
                # Construct a small child variable for each q.
                #

                child_vars = []

                for q in range(
                    max(
                        basis_size,
                        p + 1
                    ),
                    n_states
                ):

                    if q not in first:
                        continue

                    child = new_var()

                    child_vars.append(child)

                    first_from_p = first[q][p]

                    # child <-> OR_a first[q,p,a]

                    for f in first_from_p:

                        clauses.append(
                            [
                                -f,
                                child
                            ]
                        )

                    clauses.append(
                        [
                            -child
                        ]
                        + first_from_p
                    )

                if child_vars:

                    cnf = CardEnc.atmost(
                        lits=child_vars,
                        bound=alphabet_size,
                        top_id=next_var - 1,
                        encoding=EncType.seqcounter
                    )

                    clauses.extend(
                        cnf.clauses
                    )

                    next_var = max(
                        next_var,
                        cnf.nv + 1
                    )

        # ==============================================================
        # Formula statistics
        # ==============================================================

        logging.debug(
            f"SAT formula: "
            f"{next_var - 1} variables, "
            f"{len(clauses)} clauses"
        )

        # ==============================================================
        # Solve with CaDiCaL
        # ==============================================================

        try:
            ctx = mp.get_context("fork")
        except ValueError:
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

            logging.debug(
                f"Solving with timeout "
                f"{timeout_seconds:.3f}s..."
            )

            process.join(
                timeout_seconds
            )

            # ----------------------------------------------------------
            # Timeout
            # ----------------------------------------------------------

            if process.is_alive():

                logging.debug(
                    f"SAT TIMEOUT after "
                    f"{timeout_seconds:.3f}s"
                )

                process.terminate()
                process.join()

                self.smt_time += (
                    time.time() - start_time
                )

                return None, None

            # ----------------------------------------------------------
            # Get result
            # ----------------------------------------------------------

            try:

                status, model = (
                    result_queue.get_nowait()
                )

            except Exception:

                logging.error(
                    "SAT process terminated without a result"
                )

                self.smt_time += (
                    time.time() - start_time
                )

                return None, None

            # ----------------------------------------------------------
            # Error
            # ----------------------------------------------------------

            if status == "error":

                logging.error(
                    f"SAT solver error: {model}"
                )

                self.smt_time += (
                    time.time() - start_time
                )

                return None, None

            # ----------------------------------------------------------
            # UNSAT
            # ----------------------------------------------------------

            if status == "unsat":

                logging.debug("UNSAT")

                logging.debug(
                    f"No hypothesis of size "
                    f"{self.size} exists"
                )

                self.smt_time += (
                    time.time() - start_time
                )

                return None, None

            # ----------------------------------------------------------
            # SAT
            # ----------------------------------------------------------

            if status != "sat":

                logging.error(
                    f"Unexpected SAT status: {status}"
                )

                self.smt_time += (
                    time.time() - start_time
                )

                return None, None

            logging.debug("SAT")

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

                    chosen = None

                    for r in range(n_states):

                        if (
                            transition_var[q][a][r]
                            in model_set
                        ):

                            chosen = r
                            break

                    if chosen is None:

                        raise RuntimeError(
                            f"No target found for "
                            f"delta({q},{a})"
                        )

                    transition_mapping[q][a] = chosen

            # ==========================================================
            # Extract output mapping
            # ==========================================================

            output_mapping = [
                output_var[q] in model_set
                for q in range(n_states)
            ]

            self.smt_time += (
                time.time() - start_time
            )

            return (
                transition_mapping,
                output_mapping
            )

        finally:

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

def _solve_cadical_process(clauses, result_queue):
    """
    Run CaDiCaL in a separate process.

    The parent process uses this to enforce a wall-clock timeout.
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

    except Exception as e:
        result_queue.put(
            ("error", repr(e))
        )

    finally:
        if solver is not None:
            solver.delete()

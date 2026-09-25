from collections import deque
from random import shuffle, choice, randint

from aalpy.base.Oracle import Oracle
from aalpy.base.SUL import SUL

from DCValue import DCValue


def _outputs_differ(hypothesis_output: bool, sul_output: DCValue) -> bool:
    """Return whether a known SUL output disagrees with the hypothesis."""
    if sul_output is DCValue.DC:
        return False

    expected = DCValue.TRUE if hypothesis_output else DCValue.FALSE
    return sul_output is not expected


class RandomWMethodEqOracle(Oracle):
    """
    Randomized equivalence oracle for separating automata.

    For every hypothesis state that has a DC-free access sequence, the oracle
    first tries every non-empty subsequence of the known bug trace. It then
    performs the usual random walks from that state.
    """

    def __init__(self, alphabet: list, sul: SUL, traces, bug_trace=None, walks_per_state=25, walk_len=12, ):
        """
        Args:
            alphabet: input alphabet
            sul: system under learning
            traces: known labeled traces
            bug_trace: trace known to lead to the bug. If omitted, the first
                '+' trace is used.
            walks_per_state: number of random walks that should start from
                each state
            walk_len: length of random walk
        """
        super().__init__(alphabet, sul)

        self.walks_per_state = walks_per_state
        self.random_walk_len = walk_len
        self.freq_dict = dict()
        self.traces = traces

        if bug_trace is None:
            bug_trace = next(trace for label, trace in traces if label == "+")

        self.bug_trace = tuple(bug_trace)

    def _find_dc_free_access_sequences(self, hypothesis):
        """
        Find a shortest DC-free access sequence for each hypothesis state.

        BFS is performed over the hypothesis. A candidate path is only
        accepted if every output produced by the SUL along that path is
        non-DC.
        """
        initial_state = hypothesis.initial_state

        access_sequences = {initial_state: ()}

        queue = deque([initial_state])

        while queue:
            state = queue.popleft()
            prefix = access_sequences[state]

            for letter in self.alphabet:
                next_state = state.transitions[letter]

                if next_state in access_sequences:
                    continue

                candidate = prefix + (letter,)

                self.sul.pre()

                dc_free = True

                for candidate_letter in candidate:
                    output_sul = self.sul.step(candidate_letter)
                    self.num_steps += 1

                    if output_sul is DCValue.DC:
                        dc_free = False
                        break

                self.sul.post()

                if not dc_free:
                    continue

                access_sequences[next_state] = candidate
                queue.append(next_state)

        return access_sequences

    def _get_bug_subsequences(self):
        """
        Return all non-empty subsequences of the known bug trace.

        Subsequence means that elements remain in their original order,
        but do not need to be contiguous.

        Duplicate input sequences are removed.
        """
        subsequences = set()
        n = len(self.bug_trace)

        for mask in range(1, 1 << n):
            subsequence = tuple(self.bug_trace[index] for index in range(n) if mask & (1 << index))
            subsequences.add(subsequence)

        return list(subsequences)

    def _check_test_case(self, hypothesis, test_case):
        """
        Execute a test case once and return a counterexample if the
        hypothesis and SUL disagree at any point.

        Returns:
            The shortest observed counterexample prefix, or None.
        """
        self.reset_hyp_and_sul(hypothesis)

        for ind, i in enumerate(test_case):
            output_hyp = hypothesis.step(i)
            output_sul = self.sul.step(i)
            self.num_steps += 1

            if output_sul is DCValue.DC:
                break

            if _outputs_differ(output_hyp, output_sul):
                return test_case[:ind + 1]

        return None

    def find_cex(self, hypothesis):
        cexs = []

        # Check the known labeled traces symbol-by-symbol.
        for label, trace in self.traces:
            if label == "?":
                continue

            cex = self._check_test_case(hypothesis, tuple(trace), )

            if cex is not None:
                return [cex]

        # Find shortest access sequences that are DC-free in the SUL.
        access_sequences = self._find_dc_free_access_sequences(hypothesis)

        # Generate every non-empty subsequence of the known bug trace.
        bug_subsequences = self._get_bug_subsequences()

        states_to_cover = []

        for state in hypothesis.states:
            if state not in access_sequences:
                # No DC-free access sequence to this state.
                continue

            prefix = access_sequences[state]

            if prefix not in self.freq_dict:
                self.freq_dict[prefix] = 0

            remaining_walks = (self.walks_per_state - self.freq_dict[prefix])

            if remaining_walks > 0:
                states_to_cover.extend([state] * remaining_walks)

        shuffle(states_to_cover)

        for state in states_to_cover:
            prefix = access_sequences[state]

            self.freq_dict[prefix] += 1

            # ----------------------------------------------------------
            # First: try every subsequence of the known bug trace.
            # ----------------------------------------------------------
            for subsequence in bug_subsequences:
                test_case = prefix + subsequence

                cex = self._check_test_case(hypothesis, test_case, )

                if cex is not None:
                    cexs.append(cex)

                    if len(cexs) >= 100:
                        self.sul.post()
                        return cexs

            # ----------------------------------------------------------
            # Then: perform the usual random walk.
            # ----------------------------------------------------------
            self.reset_hyp_and_sul(hypothesis)

            random_walk = tuple(choice(self.alphabet) for _ in range(randint(1, self.random_walk_len)))

            test_case = prefix + random_walk

            for ind, i in enumerate(test_case):
                output_hyp = hypothesis.step(i)
                output_sul = self.sul.step(i)
                self.num_steps += 1

                if output_sul is DCValue.DC:
                    break

                if _outputs_differ(output_hyp, output_sul):
                    cexs.append(test_case[:ind + 1])

                    if len(cexs) >= 100:
                        self.sul.post()
                        return cexs

        return cexs

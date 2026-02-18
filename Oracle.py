from itertools import product
from random import shuffle, choice, randint

from aalpy.base.Oracle import Oracle
from aalpy.base.SUL import SUL


class RandomWMethodEqOracle(Oracle):
    """
    Randomized version of the W-Method equivalence oracle.
    Random walks stem from fixed prefix (path to the state). At the end of the random
    walk an element from the characterization set is added to the test case.
    """

    def __init__(self, alphabet: list, sul: SUL, traces, walks_per_state=25, walk_len=12, positive=None):
        """
        Args:

            alphabet: input alphabet

            sul: system under learning

            walks_per_state: number of random walks that should start from each state

            walk_len: length of random walk
        """

        super().__init__(alphabet, sul)
        self.walks_per_state = walks_per_state
        self.random_walk_len = walk_len
        self.freq_dict = dict()
        self.traces = traces
        self.positive = positive

    def find_cex(self, hypothesis):
        cexs = []
        for label, trace in self.traces:
            if label == "?":
                continue
            label = True if label == "+" else False
            if hypothesis.execute_sequence(hypothesis.initial_state, trace)[-1] != label:
                print(trace, label)
                return trace

        if not hypothesis.characterization_set:
            hypothesis.characterization_set = hypothesis.compute_characterization_set()
            # fix for non-minimal intermediate hypothesis that can occur in KV
            if not hypothesis.characterization_set:
                hypothesis.characterization_set = [(a,) for a in hypothesis.get_input_alphabet()]

        states_to_cover = []
        for state in hypothesis.states:
            if state.prefix is None:
                state.prefix = hypothesis.get_shortest_path(hypothesis.initial_state, state)
            if state.prefix not in self.freq_dict.keys():
                self.freq_dict[state.prefix] = 0

            states_to_cover.extend([state] * (self.walks_per_state - self.freq_dict[state.prefix]))

        shuffle(states_to_cover)

        for state in states_to_cover:
            self.freq_dict[state.prefix] = self.freq_dict[state.prefix] + 1

            self.reset_hyp_and_sul(hypothesis)

            prefix = state.prefix
            random_walk = tuple(choice(self.alphabet) for _ in range(randint(1, self.random_walk_len)))

            test_case = prefix + random_walk + choice(hypothesis.characterization_set)

            for ind, i in enumerate(test_case):
                output_hyp = hypothesis.step(i)
                output_sul = self.sul.step(i)
                self.num_steps += 1

                if output_sul == "unknown":
                    if self.positive is None:
                        break
                    if self.positive:
                        output_sul = True
                    else:
                        output_sul = False

                if output_sul != output_hyp and output_sul != "unknown":
                    cexs.append(test_case[: ind + 1])
                    print(cexs[0], output_sul, output_hyp)
                    return cexs[0]

        return None
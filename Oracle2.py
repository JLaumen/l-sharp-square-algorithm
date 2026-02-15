import random
from itertools import chain, product

from aalpy.base.Oracle import Oracle
from aalpy.base.SUL import SUL


def state_characterization_set(hypothesis, alphabet, state):
    """
    Return a list of sequences that distinguish the given state from all other states in the hypothesis.
    Args:
        hypothesis: hypothesis automaton
        alphabet: input alphabet
        state: state for which to find distinguishing sequences
    """
    result = []
    for i in range(len(hypothesis.states)):
        if hypothesis.states[i] == state:
            continue
        seq = hypothesis.find_distinguishing_seq(state, hypothesis.states[i])
        if seq:
            result.append(tuple(seq))
    return result


def first_phase_it(alphabet, state_cover, depth, char_set, sul):
    """
    Return an iterator that generates all possible sequences for the first phase of the Wp-method.
    Args:
        alphabet: input alphabet
        state_cover: list of states to cover
        depth: maximum length of middle part
        char_set: characterization set
    """
    char_set = char_set or [()]
    for d in range(depth):
        middle = product(alphabet, repeat=d)
        for s in state_cover:
            if sul.query(s) == "unknown":
                continue
            for m in middle:
                if sul.query(s + m) == "unknown":
                    continue
                for c in char_set:
                    yield s + m + c


def second_phase_it(hyp, alphabet, difference, depth, sul):
    """
    Return an iterator that generates all possible sequences for the second phase of the Wp-method.
    Args:
        hyp: hypothesis automaton
        alphabet: input alphabet
        difference: set of sequences that are in the transition cover but not in the state cover
        depth: maximum length of middle part
    """
    state_mapping = {}
    for d in range(depth):
        middle = product(alphabet, repeat=d)
        for t in difference:
            if sul.query(t) == "unknown":
                continue
            for mid in middle:
                if sul.query(t + mid) == "unknown":
                    continue
                _ = hyp.execute_sequence(hyp.initial_state, t + mid)
                state = hyp.current_state
                if state not in state_mapping:
                    state_mapping[state] = state_characterization_set(hyp, alphabet, state)

                for sm in state_mapping[state]:
                    yield t + mid + sm


class WpMethodEqOracle(Oracle):
    """
    Implements the Wp-method equivalence oracle.
    """

    def __init__(self, alphabet: list, sul: SUL, max_number_of_states=4, traces=[]):
        super().__init__(alphabet, sul)
        self.m = max_number_of_states
        # self.m = 6 if len(alphabet) < 20 else 5
        self.cache = set()
        self.traces = traces
        self.start = None

    def find_cex(self, hypothesis):
        for label, trace in self.traces:
            if label == "?":
                continue
            label = True if label == "+" else False
            if hypothesis.execute_sequence(hypothesis.initial_state, trace)[-1] != label:
                return [trace]

        steps = 0
        if self.start is None:
            length = 3
        else:
            length = self.start
        while steps < 100_000 and length < 8:
           cexs, steps = self.test_rec(hypothesis, [], length, 0)
           length += 1
           self.start = length - 1
           print(f"\n{steps}, {length}")
           if cexs:
               return cexs

        return cexs

        if not hypothesis.characterization_set:
            hypothesis.characterization_set = hypothesis.compute_characterization_set()

        transition_cover = set(
            state.prefix + (letter,)
            for state in hypothesis.states
            for letter in self.alphabet
        )

        state_cover = set(state.prefix for state in hypothesis.states)
        difference = transition_cover.difference(state_cover)
        depth = self.m + 1 - len(hypothesis.states)
        # first phase State Cover * Middle * Characterization Set
        first_phase = first_phase_it(self.alphabet, state_cover, depth, hypothesis.characterization_set, self.sul)

        # second phase (Transition Cover - State Cover) * Middle * Characterization Set
        # of the state that the prefix leads to
        second_phase = second_phase_it(hypothesis, self.alphabet, difference, depth, self.sul)
        test_suite = set(chain(first_phase, second_phase))
        test_size = len(test_suite)
        cexs = []
        for i, seq in enumerate(test_suite):
            print(f"{i}/{test_size} ({len(cexs)})", end="\r")
            if seq not in self.cache:
                self.reset_hyp_and_sul(hypothesis)

                for ind, letter in enumerate(seq):
                    out_hyp = hypothesis.step(letter)
                    out_sul = self.sul.step(letter)
                    self.num_steps += 1

                    if out_hyp != out_sul and out_sul != "unknown":
                        # self.sul.post()
                        cexs.append(seq[: ind + 1])
                        if len(cexs) >= 100:
                            self.sul.post()
                            return cexs
                self.cache.add(seq)

        cex = self.test_rec(hypothesis, [], 8)
        if cex is not None:
            cexs.append(cex)

        self.sul.post()
        return cexs

    def test_rec(self, hypothesis, seq, length, steps):
        if length == 3:
            print(f"{seq[:4]}" + " "*10, end="\r")
        if length == 0:
            return [], steps
        self.reset_hyp_and_sul(hypothesis)
        out_sul = False
        for ind, letter in enumerate(seq):
            out_hyp = hypothesis.step(letter)
            out_sul = self.sul.step(letter)
            self.num_steps += 1
            steps += 1

            if out_hyp != out_sul and out_sul != "unknown":
                return [seq[: ind + 1]], steps

        if out_sul == "unknown" or out_sul == True:
            return [], steps

        cexs = []

        for letter in self.alphabet:
            cex, steps = self.test_rec(hypothesis, seq + [letter], length - 1, steps)
            cexs += cex
        return cexs, steps

class RandomWpMethodEqOracle(Oracle):
    """
    Implements the Random Wp-Method as described in "Complementing Model
    Learning with Mutation-Based Fuzzing" by Rick Smetsers, Joshua Moerman,
    Mark Janssen, Sicco Verwer.
        1) sample uniformly from the states for a prefix
        2) sample geometrically a random word
        3) sample a word from the set of suffixes / state identifiers
    """

    def __init__(
            self, alphabet: list, sul: SUL, min_length=1, expected_length=10, num_tests=1000, ):
        super().__init__(alphabet, sul)
        self.min_length = min_length
        self.expected_length = expected_length
        self.bound = num_tests

    def find_cex(self, hypothesis):
        # fix for non-minimal intermediate hypothesis that can occur in KV
        hypothesis.characterization_set = hypothesis.compute_characterization_set()
        if not hypothesis.characterization_set:
            hypothesis.characterization_set = [(a,) for a in hypothesis.get_input_alphabet()]

        state_mapping = {s: state_characterization_set(hypothesis, self.alphabet, s) for s in hypothesis.states}

        for _ in range(self.bound):
            state = random.choice(hypothesis.states)
            input = state.prefix
            limit = self.min_length
            while limit > 0 or random.random() > 1 / (self.expected_length + 1):
                letter = random.choice(self.alphabet)
                input += (letter,)
                limit -= 1
            if random.random() > 0.5:
                # global suffix with characterization_set
                input += random.choice(hypothesis.characterization_set)
            else:
                # local suffix
                _ = hypothesis.execute_sequence(hypothesis.initial_state, input)
                if state_mapping[hypothesis.current_state]:
                    input += random.choice(state_mapping[hypothesis.current_state])
                else:
                    continue

            self.reset_hyp_and_sul(hypothesis)
            for ind, letter in enumerate(input):
                out_hyp = hypothesis.step(letter)
                out_sul = self.sul.step(letter)
                self.num_steps += 1

                if out_hyp != out_sul:
                    self.sul.post()
                    return input[: ind + 1]
        return None

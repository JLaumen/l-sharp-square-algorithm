import random

from aalpy.base.Oracle import Oracle
from aalpy.base.SUL import SUL


class WpMethodEqOracle(Oracle):
    """
    Implements the Wp-method equivalence oracle.
    """

    def __init__(self, alphabet: list, sul: SUL, max_number_of_states=4, traces=[]):
        super().__init__(alphabet, sul)
        self.cache = dict()
        self.traces = traces
        self.start = None

    def find_cex(self, hypothesis):
        for label, trace in self.traces:
            if label == "?":
                continue
            label = True if label == "+" else False
            if hypothesis.execute_sequence(hypothesis.initial_state, trace)[-1] != label:
                return [trace]

        cexs = set()
        # Start building a random walk
        # To extend the sequence, first shuffle the alphabet,
        # and keep trying a letter until a non-unknown or True output is found.
        alphabet = self.alphabet[:]
        steps = 0
        max_steps = 250 * len(hypothesis.states) * len(self.alphabet)
        while steps < max_steps:
            seq = []
            length = random.randint(1, 30)
            while len(seq) < length:
                print(f"{steps}/{max_steps} ({len(cexs)})", end="\r")
                random.shuffle(alphabet)
                cont = False
                for letter in alphabet:
                    cont = False
                    # Perform current sequence
                    self.reset_hyp_and_sul(hypothesis)
                    for l in seq:
                        hypothesis.step(l)
                        if tuple(seq + [letter]) not in self.cache:
                            self.sul.step(l)
                        self.num_steps += 1
                        steps += 1
                    # Try to extend the sequence
                    out_hyp = hypothesis.step(letter)
                    if tuple(seq + [letter]) not in self.cache:
                        out_sul = self.sul.step(letter)
                        self.cache[tuple(seq + [letter])] = out_sul
                    else:
                        out_sul = self.cache[tuple(seq + [letter])]
                    if out_hyp != out_sul and out_sul != "unknown":
                        cexs.add(tuple(seq + [letter]))
                        break
                    elif out_sul != "unknown" and out_sul != True:
                        seq.append(letter)
                        cont = True
                        break
                if not cont:
                    break
        return cexs

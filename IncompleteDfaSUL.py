from aalpy.automata import Dfa
from aalpy.base.SUL import SUL

from CacheTree import CacheTree


class DfaSUL(SUL):
    def __init__(self, automaton: Dfa):
        super().__init__()
        self.last_output = None
        self.automaton: Dfa = automaton
        self.num_successful_queries = 0

    def pre(self):
        self.automaton.reset_to_initial()
        self.last_output = self.automaton.step(None)

    def step(self, letter=None):
        self.last_output = self.automaton.step(letter)

    def post(self):
        return self.last_output

    def query(self, word: tuple) -> list:
        """
        Performs a membership query on the SUL. Before the query, pre() method is called and after the query post()
        method is called. Each letter in the word (input in the input sequence) is executed using the step method.

        Args:

            word: membership query (word consisting of letters/inputs)

        Returns:

            final output

        """
        self.pre()
        for letter in word:
            self.step(letter)
        out = self.post()
        self.num_queries += 1
        self.num_successful_queries += 1
        self.num_steps += len(word)
        if out == "unknown":
            self.num_successful_queries -= 1
        return out


class IncompleteDfaSUL(DfaSUL):
    def __init__(self, words, automaton: Dfa = None):
        super().__init__(automaton)
        self.input_walk = None
        self.cache = CacheTree()

        for word, output in words:
            self.add_word(word, output)
        for word, output in words:
            if self.word_known(word) != output:
                raise Exception(f"Word {word} with output {output} inconsistent with cache")

    def add_word(self, word, output):
        self.cache.reset()
        for index in range(len(word)):
            input_val = word[index]
            self.cache.step_in_cache(input_val, None)
        self.cache.step_in_cache(None, output)

    def word_known(self, word):
        outputs = self.cache.in_cache(word)
        if type(outputs) != list and type(outputs) != tuple:
            return outputs
        else:
            return outputs[-1]

    def pre(self):
        self.input_walk = []
        if self.automaton is None:
            self.last_output = "unknown"
        else:
            self.automaton.reset_to_initial()
            self.last_output = self.automaton.step(None)

    def step(self, letter=None):
        self.input_walk.append(letter)
        if self.automaton is None:
            self.last_output = "unknown"
        else:
            self.last_output = self.automaton.step(letter)

    def post(self):
        saved_output = self.word_known(self.input_walk)
        if saved_output is None:
            if not self.automaton is None:
                self.add_word(self.input_walk, self.last_output)
                return self.last_output
            else:
                self.add_word(self.input_walk, "unknown")
                return "unknown"
        else:
            return saved_output

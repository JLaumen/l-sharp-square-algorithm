from aalpy.base import Automaton
from aalpy.base.Oracle import Oracle


class ValidityDataOracle(Oracle):
    def __init__(self, data):
        """
        Give data in format: [(["a", "b"], True), (["b", "a", "a"], False)]
        """
        super().__init__(None, None)
        self.data = data
        self.num_queries = 0
        self.num_steps = 0

    def find_cex(self, hypothesis: Automaton):
        for inputs, output in self.data:
            hypothesis.reset_to_initial()
            for input_val in inputs:
                hypothesis.step(input_val)
                self.num_steps += 1
            hyp_output = hypothesis.step(None)
            self.num_queries += 1
            if hyp_output != output:
                return inputs
        return None

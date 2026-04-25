from aalpy.base.SUL import SUL


class IncompleteDfaSUL(SUL):
    def __init__(self, words):
        super().__init__()
        self.cache = dict()
        for word, output in words:
            self.cache[tuple(word)] = output
        self.num_queries = 0
        self.num_successful_queries = 0
        self.num_steps = 0
        self.num_cached_queries = 0

    def pre(self):
        raise NotImplementedError("Pre method is not implemented for IncompleteDfaSUL")

    def step(self, letter=None):
        raise NotImplementedError("Step method is not implemented for IncompleteDfaSUL")

    def post(self):
        raise NotImplementedError("Post method is not implemented for IncompleteDfaSUL")

    def query(self, word):
        self.num_queries += 1
        self.num_steps += len(word)
        if tuple(word) in self.cache:
            self.num_cached_queries += 1
            self.num_successful_queries += 1
            return self.cache[tuple(word)]
        else:
            return "unknown"

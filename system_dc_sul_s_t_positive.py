from aalpy.base import SUL


class SystemDCSULST_Positive(SUL):
    def __init__(self, T, system_sul):
        super().__init__()
        self.T = T
        self.system_sul = system_sul
        self.membership_queries = 0
        self.system_queries = 0
        self.label_mapper = {
            True: True,
            False: False,
            None: True
        }
        self.words = set()

    def query(self, word):
        self.pre()
        self.words.add(tuple(word))
        out = []
        system_out = False
        if not word:
            return [system_out]
        for letter in word:
            t_out = self.T.step(letter)
            system_out = self.label_mapper[self.system_sul.step(letter)]
            if not t_out:
                out.append(True)
                self.post()
                return out
            out.append(system_out)
        self.post()
        # print(word, out)
        return out

    def pre(self):
        self.system_sul.pre()
        self.T.reset_to_initial()

    def post(self):
        self.system_sul.post()
        self.T.reset_to_initial()

    def step(self, letter):
        t_out = self.T.step(letter)
        system_out = self.label_mapper[self.system_sul.step(letter)]
        if not t_out:
            return True
        return system_out


if __name__ == '__main__':
    from aalpy.utils import load_automaton_from_file
    from aalpy.base.SUL import CacheSUL
    from rers_sul_s_t import RERSSULST

    system_sul = CacheSUL(
        RERSSULST(benchmark="m199", t_type="3", for_T=False, is_prefix_closed=False, is_suffix_closed=False))
    T = load_automaton_from_file(f'data/m199/T3.dot', automaton_type='dfa')
    sul = SystemDCSULST(T, system_sul)
    word = ('ai1_ce2', 'usr2_ai2_re6', 'ai1_ce3', 'usr2_ai4_re2', 'assert', 'usr1_ai1_re1')
    print(sul.query(word))
    sul.pre()
    for c in word:
        print(sul.step(c))
    sul.post()
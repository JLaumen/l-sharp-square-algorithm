import random
import time
from copy import deepcopy

from aalpy.base.SUL import CacheSUL
from aalpy.utils import make_input_complete, load_automaton_from_file
from algorithm import learn
from complete_dfa_oracle import CompleteDFAOracle
from data.counter_examples import counter_examples_dict
from dfa3_encoder import DFA3Encoder
from fa_learner import FALearner
from rc_lstar import run_Lstar
from rers_sul_s_t import RERSSULST
from rpni_learner import RPNILearner
from system_dc_oracle_s_t import SystemDCOracleST

from LSharpSquare import run_lsharp_square
from Oracle2 import WpMethodEqOracle
from system_dc_sul_s_t import SystemDCSULST
from system_dc_sul_s_t_negative import SystemDCSULST_Negative
from system_dc_sul_s_t_positive import SystemDCSULST_Positive
from aalpy.learning_algs import run_Lsharp
from aalpy.oracles import RandomWMethodEqOracle


# from RandomWordEqOracle import RandomWordEqOracle


def run(example, t_type):
    # T2 update
    M = load_automaton_from_file(f'data/{example}/T{t_type}.dot', automaton_type='dfa')  #
    make_input_complete(M, missing_transition_go_to="sink_state")

    system_sul = CacheSUL(
        RERSSULST(benchmark=example, t_type=t_type, for_T=False, is_prefix_closed=False, is_suffix_closed=False))
    alphabet = system_sul.sul.alphabet
    sul = SystemDCSULST(M, system_sul)
    sul_positive = SystemDCSULST_Positive(M, system_sul)
    oracle_positive = RandomWMethodEqOracle(alphabet=alphabet, sul=sul_positive, walks_per_state=300, walk_len=30)
    sul_negative = SystemDCSULST_Negative(M, system_sul)
    oracle_negative = RandomWMethodEqOracle(alphabet=alphabet, sul=sul_negative, walks_per_state=300, walk_len=30)
    oracle = WpMethodEqOracle(alphabet, sul, max_number_of_states=6, traces=counter_examples_dict[example][t_type])
    # Learn the positive and negative DFA with L#
    start_time = int(time.time() * 1000) / 1000
    dfa = run_Lsharp(alphabet=alphabet, sul=sul_positive, eq_oracle=oracle_positive, automaton_type='dfa')
    print("Positive DFA:")
    print(dfa)
    dfa = run_Lsharp(alphabet=alphabet, sul=sul_negative, eq_oracle=oracle_negative, automaton_type='dfa')
    print("Negative DFA:")
    print(dfa)

    words = sul_positive.words.union(sul_negative.words)

    dfa3, data = run_lsharp_square(alphabet,
                                   sul,
                                   oracle,
                                   words=words,
                                   return_data=True)
    print(dfa3)
    # print(int(time.time() * 1000) / 1000 - start_time)
    print(data)
    exit()

    return dfa3, data, M, []


def main_single(benchmark, t_type, method, description_type, early_detection):
    random.seed(0)
    example = benchmark

    _ = run(example, t_type)
    
    return None

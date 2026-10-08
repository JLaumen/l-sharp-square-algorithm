import logging
import random

from aalpy.base.SUL import CacheSUL
from aalpy.utils import make_input_complete, load_automaton_from_file
from data.counter_examples import counter_examples_dict
from rers_sul_s_t import RERSSULST

from LSharpSquare import run_lsharp_square
from Oracle import RandomWMethodEqOracle
from system_dc_sul_s_t import SystemDCSULST
from three_valued_oracle import ThreeValuedEqOracle

logging.basicConfig(level=logging.DEBUG, format=f"%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")


def run(example, t_type, single):
    # T2 update
    M = load_automaton_from_file(f'data/{example}/T{t_type}.dot', automaton_type='dfa')  #
    make_input_complete(M, missing_transition_go_to="sink_state")

    system_sul = CacheSUL(
        RERSSULST(benchmark=example, t_type=t_type, for_T=False, is_prefix_closed=False, is_suffix_closed=False))
    alphabet = system_sul.sul.alphabet
    sul = SystemDCSULST(M, system_sul)
    model = f"models/{example}.dot"
    from mealy_to_3dfa import construct_3dfa_from_file
    dfa3 = construct_3dfa_from_file(model, sul, alphabet)
    print(dfa3)
    oracle = ThreeValuedEqOracle(alphabet, dfa3)
    dfa, data = run_lsharp_square(alphabet, sul, oracle, return_data=True)

    if not single:
        print(f"{example},{data['automaton_size']},{data['total_time']}")
    else:
        print("Learned DFA:")
        print(dfa)
        print(f"Total time: {data['total_time']}")

    return


def main_single(benchmark, single):
    random.seed(0)
    t_type = "1"
    example = benchmark

    run(example, t_type, single)

    return None

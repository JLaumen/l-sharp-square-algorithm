import argparse
from main_single import main_single
from stopit import SignalTimeout as Timeout
import random
import warnings

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', '-b', type=str, default="m24")
args = parser.parse_args()

random.seed(0)

if args.benchmark == "all":
    print("benchmark,automaton_size,total_time")
    benchmarks = ["m24", "m45", "m54", "m55", "m76", "m95", "m135", "m158", "m159", "m164", "m172", "m181", "m183",
                  "m185", "m201", "m22", "m27", "m41", "m106", "m131", "m132", "m167", "m173", "m182", "m189", "m196",
                  "m199"]
    single = False
else:
    print(f"Running benchmark {args.benchmark}\n")
    benchmarks = [args.benchmark]
    single = True

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for benchmark in benchmarks:
        with Timeout(60.0*60*10) as timeout_ctx:
            results = main_single(benchmark, single)
        if timeout_ctx.state == timeout_ctx.TIMED_OUT:
            results = {"benchmark": benchmark, "L* time": 0}

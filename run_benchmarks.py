import argparse
import concurrent.futures
import logging
import multiprocessing
import os
from itertools import repeat
from multiprocessing.spawn import freeze_support
from typing import Any

from FiniteDataOracle import FiniteDataOracle
from IncompleteSUL import IncompleteSUL
from LSharpSquare import run_lsharp_square

test_cases_path = "benchmarking/benchmarks/"
results_path = "benchmarking/results/"
logging.basicConfig(level=logging.INFO, format=f"%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")


def is_simple_input(inp: str) -> bool:
    return all(c in ["0", "1", "X"] for c in inp)


def get_possible_words(prefix: str, suffix: str, alphabet: list) -> list:
    words = []
    if suffix:
        if suffix[0] == "X":
            for letter in alphabet:
                words.extend(get_possible_words(prefix + letter, suffix[1:], alphabet))
        else:
            letter = suffix[0]
            words.extend(get_possible_words(prefix + letter, suffix[1:], alphabet))
        return words
    else:
        return [prefix]


def parse_file(filename: str, alphabet: list, horizon: int | None = None) -> tuple[list, list]:
    with open(test_cases_path + filename, 'r') as f:
        known_words = []
        observed_alphabet = []
        for l in f:
            split_index = l.strip().rfind(',')
            inp = l.strip()[:split_index]
            out = l.strip()[split_index + 1:]
            out = out.strip() == "+"
            if is_simple_input(inp):
                inputs = get_possible_words("", inp, alphabet)
                for word in inputs:
                    for letter in word:
                        if not letter in observed_alphabet:
                            observed_alphabet.append(letter)
                    known_words.append((tuple(word), out))
            else:
                word = inp.split(";")
                for letter in word:
                    if not letter in observed_alphabet:
                        observed_alphabet.append(letter)
                if horizon is None or len(word) <= horizon:
                    known_words.append((tuple(word), out))

        return known_words, observed_alphabet


def run_test_case(filename: str, horizon: int | None = None, example: bool = False) -> dict[str, Any]:
    alphabet = [True, False]
    data, alphabet = parse_file(filename, alphabet, horizon)
    sul = IncompleteSUL(data)
    eq_oracle = FiniteDataOracle(alphabet, sul, data)

    learned_dfa, info = run_lsharp_square(alphabet, sul, eq_oracle, return_data=True)

    if example:
        print("Learned DFA:")
        print(learned_dfa)

    successful = learned_dfa is not None and eq_oracle.find_cex(learned_dfa) is None
    info["successful"] = successful
    return info


def process_file(file_name: str, target_folder: str) -> str:
    logging.info(f"Testing {file_name}")
    info = run_test_case(f"{target_folder}/{file_name}")
    row = ','.join([f"{target_folder}/{file_name}", str(info['successful']), str(info['learning_rounds']),
                    str(info['automaton_size']), str(info['learning_time']), str(info['solver_time']),
                    str(info['eq_oracle_time']), str(info['total_time']), str(info['membership_queries']),
                    str(info['validity_query']), str(info['nodes']), str(info['informative_nodes']),
                    str(info['sul_steps']), str(info['queries_eq_oracle']), str(info['steps_eq_oracle'])]) + "\n"
    logging.debug(f"Finished testing {file_name}")
    logging.debug(f"Time: {info['total_time']}")
    logging.debug(f"Queries: {info['membership_queries']}")
    logging.debug(f"Validity: {info['validity_query']}")
    logging.debug(f"Size: {info['automaton_size']}")
    return row


def _process_file_worker(connection, file_name: str, target_folder: str) -> None:
    connection.send(process_file(file_name, target_folder))
    connection.close()


def process_file_with_timeout(file_name: str, target_folder: str, timeout: int) -> str:
    """Run process_file in a separate process, which is killed once it takes longer than timeout seconds."""
    receiver, sender = multiprocessing.Pipe(duplex=False)
    process = multiprocessing.Process(target=_process_file_worker, args=(sender, file_name, target_folder))
    process.start()
    sender.close()
    try:
        if receiver.poll(timeout):
            return receiver.recv()
        logging.info(f"Timeout on {file_name} after {timeout} seconds")
        process.kill()
        return f"{target_folder}/{file_name},False,,0,,,,{timeout},,,,,,,\n"
    except EOFError:
        raise RuntimeError(f"Benchmark {target_folder}/{file_name} failed") from None
    finally:
        process.join()
        receiver.close()


def run_test_cases_pool(file: str, extension: str, timeout: int, jobs: int | None = None) -> None:
    os.makedirs(results_path, exist_ok=True)
    with open(f"{results_path}benchmark{extension}_{file}.csv", "w") as f:
        f.write("file name,succeeded,learning_rounds,automaton_size,learning_time,"
                "smt_time,eq_oracle_time,total_time,queries_learning,validity_query,nodes,"
                "informative_nodes,sul_steps,queries_eq_oracle,steps_eq_oracle\n")
        oliveira = test_cases_path
        target_folder = file
        folder_path = os.path.join(oliveira, target_folder)
        file_names = sorted(os.listdir(folder_path))
        logging.info(f"Running benchmarks on {len(file_names)} files in {folder_path}")
        logging.info(f"Timeout: {timeout} seconds, jobs: {jobs}")

        # Filter out randm22 and randm23 files
        # file_names = [f for f in file_names if not ("randm22" in f or "randm23" in f)]

        # If jobs is None, do not run in parallel
        if jobs is None:
            for file_name in file_names:
                row = process_file_with_timeout(file_name, target_folder, timeout)
                f.write(row)
        else:
            # Every test case runs in its own process; the threads only wait for those processes.
            with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
                for row in executor.map(process_file_with_timeout, file_names, repeat(target_folder),
                                        repeat(timeout)):
                    f.write(row)

        logging.info("Benchmarks complete")


def main(timeout: int = 200, jobs: int | None = None, example: int = None) -> None:
    if example:
        if not 4 <= example <= 23:
            print("Example must be between 4 and 23")
            return
        print(f"Running example test case randm{'0' if example < 10 else ''}{example}.02.02.18.020_0030.04.aba.beta\n")
        info = run_test_case(f"all/randm{'0' if example < 10 else ''}{example}.02.02.18.020_0030.04.aba.beta",
                             example=True)
        print("Total time:", info['total_time'])
        print("Membership queries:", info['membership_queries'])
        print("Validity queries:", info['validity_query'])
        print("Input symbols:", info['sul_steps'])
    else:
        run_test_cases_pool("all", f"_t{timeout}", timeout, jobs)


if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser(description="Run benchmarks in run_benchmarks.py")
    parser.add_argument("-t", "--timeout", type=int, default=200,
                        help="Timeout per benchmark in seconds. Default: 200")
    parser.add_argument("-j", "--jobs", type=int, default=None,
                        help="Number of parallel processes. Default: run sequentially")
    parser.add_argument("-e", "--example", type=int, default=None,
                        help="Run example test case instead of benchmarks. Must be between 4 and 23. Default: None (run benchmarks)")
    args = parser.parse_args()

    main(timeout=args.timeout, jobs=args.jobs, example=args.example)

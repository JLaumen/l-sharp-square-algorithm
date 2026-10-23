import argparse
import concurrent.futures
import logging
import multiprocessing
import os
import subprocess
from itertools import repeat
from multiprocessing.spawn import freeze_support
from typing import Any

from aalpy.automata import Dfa

from DfaMinerRunner import run_dfaminer
from LSharpSquare import run_lsharp_square

test_cases_path = "benchmarking/benchmarks/"
results_path = "benchmarking/results/"
default_dfaminer_path = "../DFAMiner"
algorithms = ["lsharp", "dfaminer"]
# Folder in test_cases_path of each benchmark suite.
suites = {"oliveira": "all", "dfaminer": "dfaminer"}
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


def parse_abbadingo_file(filename: str) -> tuple[list, list]:
    """Read a file in Abbadingo format, as used by the DFAMiner benchmarks; labels other than 0 and 1 are skipped."""
    with open(test_cases_path + filename, 'r') as f:
        _, alphabet_size = (int(value) for value in f.readline().split())
        known_words = []
        for line in f:
            values = [int(value) for value in line.split()]
            if len(values) < 2 or values[0] not in (0, 1):
                continue
            known_words.append((tuple(values[2:2 + values[1]]), values[0] == 1))
    return known_words, list(range(alphabet_size))


def read_test_case(filename: str, horizon: int | None = None) -> tuple[list, list]:
    if filename.startswith(suites["dfaminer"] + "/"):
        return parse_abbadingo_file(filename)
    return parse_file(filename, [True, False], horizon)


def agrees_with(dfa: Dfa, data: list) -> bool:
    """Check that the DFA gives the expected output for every word in the data."""
    for word, expected in data:
        dfa.reset_to_initial()
        for symbol in word:
            dfa.step(symbol)
        # Stepping with ``None`` returns the output of the current state without moving.
        if dfa.step(None) != expected:
            return False
    return True


def run_test_case(filename: str, horizon: int | None = None, example: bool = False, algorithm: str = "lsharp",
                  dfaminer_path: str = default_dfaminer_path, timeout: float | None = None) -> dict[str, Any]:
    data, alphabet = read_test_case(filename, horizon)
    if algorithm == "dfaminer":
        # DFAMiner verifies its own result and fails if it is wrong.
        info = run_dfaminer(dfaminer_path, alphabet, data, timeout)
        info["successful"] = True
        return info

    learned_dfa, info = run_lsharp_square(alphabet, data, return_data=True)

    if example:
        print("Learned DFA:")
        print(learned_dfa)

    successful = agrees_with(learned_dfa, data)
    info["successful"] = successful
    return info


def timeout_row(file_name: str, target_folder: str, timeout: int) -> str:
    return f"{target_folder}/{file_name},False,0,,,{timeout},,\n"


def process_file(file_name: str, target_folder: str, algorithm: str = "lsharp",
                 dfaminer_path: str = default_dfaminer_path, timeout: float | None = None) -> str:
    logging.info(f"Testing {file_name}")
    info = run_test_case(f"{target_folder}/{file_name}", algorithm=algorithm, dfaminer_path=dfaminer_path,
                         timeout=timeout)
    columns = [info[key] for key in ('successful', 'automaton_size', 'learning_time', 'solver_time', 'total_time',
                                     'nodes', 'informative_nodes')]
    # Statistics an algorithm does not report are left empty.
    row = ','.join([f"{target_folder}/{file_name}"] + ['' if column is None else str(column) for column in columns]) + "\n"
    logging.debug(f"Finished testing {file_name}")
    logging.debug(f"Time: {info['total_time']}")
    logging.debug(f"Size: {info['automaton_size']}")
    return row


def _process_file_worker(connection, file_name: str, target_folder: str) -> None:
    connection.send(process_file(file_name, target_folder))
    connection.close()


def process_file_with_timeout(file_name: str, target_folder: str, timeout: int, algorithm: str = "lsharp",
                              dfaminer_path: str = default_dfaminer_path) -> str:
    """Run process_file, giving up once it takes longer than timeout seconds.

    L#-square runs in a separate process, which is killed on timeout. DFAMiner already runs in its own
    process, which ``subprocess`` kills on timeout, so no extra process is needed.
    """
    if algorithm == "dfaminer":
        try:
            return process_file(file_name, target_folder, algorithm, dfaminer_path, timeout)
        except subprocess.TimeoutExpired:
            logging.info(f"Timeout on {file_name} after {timeout} seconds")
            return timeout_row(file_name, target_folder, timeout)

    receiver, sender = multiprocessing.Pipe(duplex=False)
    process = multiprocessing.Process(target=_process_file_worker, args=(sender, file_name, target_folder))
    process.start()
    sender.close()
    try:
        if receiver.poll(timeout):
            return receiver.recv()
        logging.info(f"Timeout on {file_name} after {timeout} seconds")
        process.kill()
        return timeout_row(file_name, target_folder, timeout)
    except EOFError:
        raise RuntimeError(f"Benchmark {target_folder}/{file_name} failed") from None
    finally:
        process.join()
        receiver.close()


def run_test_cases_pool(file: str, extension: str, timeout: int, jobs: int | None = None, algorithm: str = "lsharp",
                        dfaminer_path: str = default_dfaminer_path) -> None:
    os.makedirs(results_path, exist_ok=True)
    with open(f"{results_path}benchmark{extension}_{file}.csv", "w") as f:
        f.write("file name,succeeded,automaton_size,learning_time,solver_time,total_time,nodes,"
                "informative_nodes\n")
        oliveira = test_cases_path
        target_folder = file
        folder_path = os.path.join(oliveira, target_folder)
        file_names = sorted(os.listdir(folder_path))
        logging.info(f"Running benchmarks on {len(file_names)} files in {folder_path}")
        logging.info(f"Algorithm: {algorithm}, timeout: {timeout} seconds, jobs: {jobs}")

        # Filter out randm22 and randm23 files
        # file_names = [f for f in file_names if not ("randm22" in f or "randm23" in f)]

        # If jobs is None, do not run in parallel
        if jobs is None:
            for file_name in file_names:
                row = process_file_with_timeout(file_name, target_folder, timeout, algorithm, dfaminer_path)
                f.write(row)
        else:
            # Every test case runs in its own process; the threads only wait for those processes.
            with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
                for row in executor.map(process_file_with_timeout, file_names, repeat(target_folder),
                                        repeat(timeout), repeat(algorithm), repeat(dfaminer_path)):
                    f.write(row)

        logging.info("Benchmarks complete")


def main(timeout: int = 200, jobs: int | None = None, example: int = None, algorithm: str = "lsharp",
         dfaminer_path: str = default_dfaminer_path, suite: str = "oliveira") -> None:
    if example:
        if not 4 <= example <= 23:
            print("Example must be between 4 and 23")
            return
        print(f"Running example test case randm{'0' if example < 10 else ''}{example}.02.02.18.020_0030.04.aba.beta\n")
        info = run_test_case(f"all/randm{'0' if example < 10 else ''}{example}.02.02.18.020_0030.04.aba.beta",
                             example=True, algorithm=algorithm, dfaminer_path=dfaminer_path)
        print("Total time:", info['total_time'])
        print("Automaton size:", info['automaton_size'])
    else:
        prefix = "" if algorithm == "lsharp" else f"_{algorithm}"
        run_test_cases_pool(suites[suite], f"{prefix}_t{timeout}", timeout, jobs, algorithm, dfaminer_path)


if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser(description="Run benchmarks in run_benchmarks.py")
    parser.add_argument("-t", "--timeout", type=int, default=200,
                        help="Timeout per benchmark in seconds. Default: 200")
    parser.add_argument("-j", "--jobs", type=int, default=None,
                        help="Number of parallel processes. Default: run sequentially")
    parser.add_argument("-e", "--example", type=int, default=None,
                        help="Run example test case instead of benchmarks. Must be between 4 and 23. Default: None (run benchmarks)")
    parser.add_argument("-a", "--algorithm", choices=algorithms, default="lsharp",
                        help="Algorithm to run. Default: lsharp")
    parser.add_argument("--dfaminer-path", default=default_dfaminer_path,
                        help=f"Directory of the DFAMiner repository. Default: {default_dfaminer_path}")
    parser.add_argument("-b", "--benchmarks", choices=list(suites), default="oliveira",
                        help="Benchmark suite to run. Default: oliveira")
    args = parser.parse_args()

    main(timeout=args.timeout, jobs=args.jobs, example=args.example, algorithm=args.algorithm,
         dfaminer_path=args.dfaminer_path, suite=args.benchmarks)

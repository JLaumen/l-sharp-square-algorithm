import argparse
import concurrent.futures
import logging
import os
from typing import Any

from IncompleteDfaSUL import IncompleteDfaSUL
from LSharpSquare import run_lsharp_square
from ValidityDataOracle import ValidityDataOracle

test_cases_path = "benchmarking/benchmarks/"
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
                    known_words.append((word, out))
            else:
                word = inp.split(";")
                for letter in word:
                    if not letter in observed_alphabet:
                        observed_alphabet.append(letter)
                if horizon is None or len(word) <= horizon:
                    known_words.append((word, out))

        return known_words, observed_alphabet


def run_test_case(filename: str, solver_timeout, replace_basis, use_compatibility, horizon: int | None = None) -> dict[
    str, Any]:
    alphabet = [True, False]
    data, alphabet = parse_file(filename, alphabet, horizon)
    sul = IncompleteDfaSUL(data.copy())
    eq_oracle = ValidityDataOracle(data.copy())

    learned_dfa, info = run_lsharp_square(alphabet, sul, eq_oracle, return_data=True, solver_timeout=solver_timeout,
                                          replace_basis=replace_basis, use_compatibility=use_compatibility)

    successful = learned_dfa is not None and eq_oracle.find_cex(learned_dfa) is None
    info["successful"] = successful
    return info


def process_file(file_name: str, target_folder: str, solver_timeout, replace_basis, use_compatibility) -> str:
    logging.info(f"Testing {file_name}")
    info = run_test_case(f"{target_folder}/{file_name}", solver_timeout, replace_basis, use_compatibility)
    row = ','.join([f"{target_folder}/{file_name}", str(info['successful']), str(info['learning_rounds']),
                    str(info['automaton_size']), str(info['learning_time']), str(info['smt_time']),
                    str(info['eq_oracle_time']), str(info['total_time']), str(info['queries_learning']),
                    str(info['validity_query']), str(info['nodes']), str(info['informative_nodes']),
                    str(info['sul_steps']), str(info['queries_eq_oracle']), str(info['steps_eq_oracle'])]) + "\n"
    logging.debug(f"Finished testing {file_name}")
    logging.debug(f"Time: {info['total_time']}")
    logging.debug(f"Queries: {info['queries_learning']}")
    logging.debug(f"Validity: {info['validity_query']}")
    logging.debug(f"Size: {info['automaton_size']}")
    return row


def run_test_cases_pool(file: str, extension: str,
                        solver_timeout, replace_basis,
                        use_compatibility,
                        jobs: int | None = None) -> None:
    with open(f"benchmarking/results/benchmark{extension}_{file}.csv", "w") as f:
        f.write("file name,succeeded,learning_rounds,automaton_size,learning_time,"
                "smt_time,eq_oracle_time,total_time,queries_learning,validity_query,nodes,"
                "informative_nodes,sul_steps,queries_eq_oracle,steps_eq_oracle\n")
        oliveira = test_cases_path
        target_folder = file
        folder_path = os.path.join(oliveira, target_folder)
        file_names = sorted(os.listdir(folder_path))
        logging.info(f"Running benchmarks on {len(file_names)} files in {folder_path}")

        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
            results = list(executor.map(process_file, file_names, [target_folder] * len(file_names),
                                        [solver_timeout] * len(file_names), [replace_basis] * len(file_names),
                                        [use_compatibility] * len(file_names)))
            for row in results:
                f.write(row)

        logging.info("Benchmarks complete")


def main(benchmark: str, solver_timeout: int = 3600,
         replace_basis: bool = False,
         use_compatibility: bool = False,
         jobs: int | None = None) -> None:

    if benchmark == "oliveira":
        run_test_cases_pool(
            "all",
            f"_t{solver_timeout}_r{replace_basis}_c{use_compatibility}",
            solver_timeout,
            replace_basis,
            use_compatibility,
            jobs
        )
    else:
        logging.error(f"Unknown benchmark type: {benchmark}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmarks in run_benchmarks.py")
    parser.add_argument("-b", "--benchmark", type=str, choices=["oliveira"], required=True,
                        help="Benchmark to run: \"oliveira\"")
    parser.add_argument("-t", "--timeout", type=int, default=3600, help="Solver timeout in seconds. Default: 3600")
    parser.add_argument("-r", "--replace-basis", action="store_true", dest="replace_basis",
                        help="Set replace_basis to True")
    parser.add_argument("-c", "--compatibility", action="store_true", dest="use_compatibility",
                        help="Set use_compatibility to True")
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=None,
        help="Number of parallel processes. Default: use all cores. Use 1 for single-core"
    )
    args = parser.parse_args()

    main(
        benchmark=args.benchmark,
        solver_timeout=args.timeout,
        replace_basis=args.replace_basis,
        use_compatibility=args.use_compatibility,
        jobs=args.jobs
    )

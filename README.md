# Repository for the CAV paper "An L# Based Algorithm for Active Learning of Minimal Separating Automata"

**Authors:**  
Jasper Laumen  
Leonne Snel  
Frits Vaandrager

---

# OLIVEIRA BENCHMARKS

## Setup

The Oliveira benchmarks run from the main branch and require Python 3.10 or newer.
Create a virtual environment in the repository and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running the Benchmarks

To run the complete benchmark as in the paper:

```bash
python3 run_benchmarks.py
```

The results will be saved in:

```
benchmarking/results
```

Note: this may take a long time (up to a full day).

---

## Analysing the Results

To summarise a results file and plot the results:

```bash
python3 analyze_results.py benchmarking/results/benchmark_t200_all.csv
```

Without an argument, the newest file in `benchmarking/results` is used.
This prints the number of solved test cases, statistics per number of states of the target DFA, and the unsolved test
cases. Plots of the solved test cases, running time, membership queries and validity queries are saved as PNG files in
`benchmarking/plots/<results file name>/`.

---

## Technical Review Examples

For quick testing, small examples are provided:

```bash
python3 run_benchmarks.py -e [EXAMPLE]
```

Where `[EXAMPLE]` is any number between **4 and 23**.

These examples should finish within approximately one minute.

---

## Additional Options

You can use the following optional arguments:

- `-t [SECONDS]`  
  Timeout per test case (default: 200 seconds).

- `-j [THREADS]`  
  Run multiple test cases in parallel using multiple threads.

- `-a {lsharp,dfaminer}`  
  Algorithm to run (default: `lsharp`). `dfaminer` runs [DFAMiner](https://github.com/liyong31/DFAMiner) on the same
  test cases, each in its own process that is killed on timeout, so it also works with `-j`. Results are written to
  `benchmark_dfaminer_t<timeout>_all.csv`; the node counts, which DFAMiner does not report, are left empty.

- `--dfaminer-path [DIRECTORY]`  
  Directory of the DFAMiner repository (default: `../DFAMiner`). Its requirements (`python-sat`) must be installed in
  the Python environment running the benchmarks.

Using `-j` can significantly reduce the total runtime of the full benchmark, but may affect the runtime of individual test cases depending on available system resources.

---

# RERS BENCHMARKS

## Setup

To run the RERS benchmarks, first build the Docker image (from the rers branch):

```bash
docker build -t l-sharp-square-rers .
```

Alternatively, pull the Docker image from Docker Hub:

```bash
docker pull jlaumen/l-sharp-square-rers
```

Run the Docker image:

```bash
docker run -it --rm l-sharp-square-rers
```

---

## Running the Benchmarks

To run the complete benchmark as in the paper:

```bash
python3 main.py -b all > results.csv
```

The results will be saved to:

```
results.csv
```

This should finish within a few hours.

---

## Running Individual Test Cases

To run a single test case:

```bash
python3 main.py -b m24
```

You can replace `m24` with any other available test case.

This example should finish within approximately one minute.

# Repository for the CAV paper "An L# Based Algorithm for Active Learning of Minimal Separating Automata"

**Authors:**  
Jasper Laumen  
Leonne Snel  
Frits Vaandrager

---

# OLIVEIRA BENCHMARKS

## Setup

To run the Oliveira benchmarks, first build the Docker image (from the main branch):

```bash
docker build -t l-sharp-square-oliveira .
```

Alternatively, pull the Docker image from Docker Hub:

```bash
docker pull jlaumen/l-sharp-square-oliveira
```

Run the Docker image:

```bash
docker run -it --rm l-sharp-square-oliveira
```

---

## Running the Benchmarks

To run the complete benchmark as in the paper:

```bash
python3 run_benchmarks.py -r
```

The results will be saved in:

```
benchmarking/results
```

Note: this may take a long time (up to a full day).

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

- `-c`  
  Replace the apartness check with compatibility.

- `-j [THREADS]`  
  Run multiple test cases in parallel using multiple threads.

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

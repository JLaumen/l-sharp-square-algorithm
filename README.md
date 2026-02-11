# l-sharp-square-algorithm
Python (AALpy) implementation of the LSharpSquare algorithm for the active learning of minimal separating automata by Jasper Laumen, Leonne Snel, and Frits Vaandrager.

# Usage
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the main script:
   ```bash
   python run_benchmark.py -b "oliveira" [-t <timeout>] [-c] [-r]
    ```
    - `-b ""`: Specify the benchmark type, currently only "oliveira".
    - `-t <timeout>`: (Optional) Set a timeout value for the benchmark in seconds.
    - `-c`: (Optional) Use compatibility instead of apartness.
    - `-r`: (Optional) Use basis replacement.
    - `-j <threads>`: (Optional) The number of CPU threads to use.
   
   For example, to run the Oliveira benchmarks with a timeout of 200 seconds on 4 threads with basis replacement, use:
   ```bash
   python run_benchmark.py -b "oliveira" -t 200 -j 4 -r
   ```
    Note that the benchmarks will use all available CPU threads by default, so ensure your system can handle the load. The total
    expected time for the Oliveira benchmarks is approximately 10 CPU hours when using the default timeout of 200
    seconds, which can be divided by the number of available cores to estimate wall-clock time. In the case of hyperthreading,
    individual benchmarks might perform a bit worse, so keep this in mind when comparing different runs (make sure to use the same
    amount of threads for both runs).
3. View the results:
    - The results of the Oliveira benchmarks will be saved to `benchmarking/results`.

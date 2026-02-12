# l-sharp-square-algorithm
Python (AALpy) implementation of the LSharpSquare algorithm for the active learning of minimal separating automata by Jasper Laumen, Leonne Snel, and Frits Vaandrager.

# RERS benchmarks
1. We use the Docker container from the author of the benchmarks: https://github.com/bThink-BGU/Papers-2025-MODELS-Automata-Bug-Description
   ```bash
   docker run --rm -it tomyaacov/automata-bug-description-docker
   ```
2. Clone our repository:
   ```bash
   git clone https://github.com/JLaumen/l-sharp-square-algorithm.git -b "rers"
   ```
3. Move the contents of the repository:
   ```bash
   mv -f ./l-sharp-square-algorithm/* ./
   ```
4. Install pySMT
   ```bash
   pip install pysmt
   ```
5. To run a benchmark, use something like:
   ```bash
   python main.py -B m164 -T 1
   ```
   where "m164" can be replaced with the desired benchmark. The results are printed to the console.

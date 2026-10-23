"""Run DFAMiner (https://github.com/liyong31/DFAMiner) on labelled words in a separate process."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from typing import Any


class DfaMinerFailed(Exception):
    """DFAMiner exited with an error, for instance because its verification of the result failed."""


def _write_abbadingo(path: str, alphabet: list, data: list[tuple[tuple[Any, ...], bool]]) -> None:
    """Write the words in Abbadingo format, with letters replaced by their index in ``alphabet``."""
    letter_index = {letter: index for index, letter in enumerate(alphabet)}
    with open(path, "w") as f:
        f.write(f"{len(data)} {len(alphabet)}\n")
        for word, accepts in data:
            letters = " ".join(str(letter_index[letter]) for letter in word)
            f.write(f"{int(accepts)} {len(word)}{' ' if letters else ''}{letters}\n")


def _parse_seconds(output: str, name: str) -> float:
    match = re.search(rf"Elapsed time in {name}: ([0-9.]+) secs", output)
    if match is None:
        raise DfaMinerFailed(f"No '{name}' time in the output of DFAMiner")
    return float(match.group(1))


def run_dfaminer(dfaminer_path: str, alphabet: list, data: list[tuple[tuple[Any, ...], bool]],
                 timeout: float | None = None) -> dict[str, Any]:
    """Learn a minimal DFA with DFAMiner and verify it against the words.

    DFAMiner is run in its own directory, since it imports its modules from there. The process is
    killed when it takes longer than ``timeout`` seconds.

    Returns:
        Statistics with the keys of :func:`LSharpSquare.run_lsharp_square`; those DFAMiner does not
        report are ``None``. The times are those reported by DFAMiner, without starting Python.

    Raises:
        subprocess.TimeoutExpired: If DFAMiner takes longer than ``timeout`` seconds.
        DfaMinerFailed: If DFAMiner fails or its result disagrees with the words.
    """
    dfaminer_path = os.path.abspath(dfaminer_path)
    if not os.path.isfile(os.path.join(dfaminer_path, "dfaminer.py")):
        raise FileNotFoundError(f"dfaminer.py not found in {dfaminer_path}")

    with tempfile.TemporaryDirectory() as directory:
        samples_path = os.path.join(directory, "samples.txt")
        output_path = os.path.join(directory, "dfa.txt")
        _write_abbadingo(samples_path, alphabet, data)
        completed = subprocess.run([sys.executable, "dfaminer.py", "--file", samples_path, "--out", output_path,
                                    "--verify"], cwd=dfaminer_path, capture_output=True, text=True, timeout=timeout)
        if completed.returncode != 0 or "DFA verification passed" not in completed.stdout:
            raise DfaMinerFailed(f"DFAMiner failed ({completed.returncode}): "
                                 f"{(completed.stdout + completed.stderr)[-500:]}")
        with open(output_path) as f:
            states = int(f.readline().split()[0])

    total_time = _parse_seconds(completed.stdout, "miner")
    solver_time = _parse_seconds(completed.stdout, "minimiser")
    return {"automaton_size": states, "learning_time": total_time - solver_time, "solver_time": solver_time,
            "total_time": total_time, "nodes": None, "informative_nodes": None}

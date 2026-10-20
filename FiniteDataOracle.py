"""Equivalence oracle that checks a hypothesis against a finite labelled dataset."""

from __future__ import annotations

from typing import Any

from aalpy.automata import Dfa
from aalpy.base import Oracle


class FiniteDataOracle(Oracle):
    """Equivalence oracle based on a finite collection of input/output observations.

    The oracle checks whether a hypothesis DFA agrees with every word in a
    finite dataset. The first word for which the hypothesis produces a
    different output is returned as a counterexample.

    Attributes:
        data: Input words together with their expected acceptance results.
        num_queries: Number of calls to :meth:`find_cex`.
        num_tests: Number of individual words tested.
        num_steps: Total number of input symbols processed while testing words.
    """

    def __init__(self, alphabet: list[Any], sul: Any, data: list[tuple[tuple[Any, ...], bool]]) -> None:
        """Initialize the finite-data equivalence oracle.

        Args:
            alphabet: The input alphabet of the system under learning.
            sul: The system under learning (SUL).
            data: Input words paired with their expected acceptance result.
        """
        super().__init__(alphabet, sul)
        self.data = data
        self.num_queries = 0
        self.num_tests = 0
        self.num_steps = 0

    def find_cex(self, hypothesis: Dfa) -> tuple[Any, ...] | None:
        """Find a counterexample to the given hypothesis.

        The hypothesis is tested against each word in the finite dataset, in
        order. Testing stops at the first word whose output differs from the
        expected result.

        Args:
            hypothesis: The hypothesis DFA to test against the finite data.

        Returns:
            The first input word for which the hypothesis disagrees with the
            expected result, or ``None`` if the hypothesis agrees with all
            observations.
        """
        self.num_queries += 1

        for input_word, expected_output in self.data:
            self.num_tests += 1
            self.num_steps += len(input_word)

            hypothesis.reset_to_initial()

            for input_symbol in input_word:
                hypothesis.step(input_symbol)

            # Stepping with ``None`` returns the output of the current state without moving.
            actual_output = hypothesis.step(None)

            if actual_output != expected_output:
                return input_word

        return None

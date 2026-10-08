"""System under learning wrapper for a three-valued DFA."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from DCValue import DCValue


class ThreeDFASUL:
    """SUL wrapper around a deterministic three-valued DFA.

    The wrapped 3DFA is expected to have:

        - ``initial_state``: the initial state.
        - ``initial_state.transitions``: transition dictionary.
        - each state has an ``output`` containing a ``DCValue``.

    Unlike AALpy's standard SUL interface, ``query`` returns the output of the
    complete input word rather than a list of outputs for every prefix. This
    matches the interface expected by ``ObservationTreeSquare``.
    """

    def __init__(self, automaton: Any) -> None:
        self.automaton = automaton
        self.current_state = automaton.initial_state

        self.num_queries = 0
        self.num_steps = 0

    def pre(self) -> None:
        """Reset the SUL to the initial state."""
        self.current_state = self.automaton.initial_state

    def post(self) -> None:
        """No cleanup is required for a DFA."""
        pass

    def step(self, input_value: Any) -> DCValue:
        """Execute one input and return the resulting state's output."""
        if input_value is None:
            return self.current_state.output

        try:
            self.current_state = self.current_state.transitions[input_value]
        except KeyError as exc:
            raise ValueError(
                f'Input {input_value!r} is not defined from the current '
                f'state {self.current_state!r}.'
            ) from exc

        self.num_steps += 1
        return self.current_state.output

    def query(self, word: Sequence[Any]) -> DCValue:
        """Execute a complete word and return its final three-valued output."""
        self.num_queries += 1
        self.pre()

        if not word:
            output = self.current_state.output
        else:
            output = DCValue.DC
            for input_value in word:
                output = self.step(input_value)

        self.post()
        return output

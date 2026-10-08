"""AALpy equivalence oracle for a three-valued Moore machine.

The oracle compares a candidate AALpy ``Dfa`` against the three-valued
``MooreMachine`` produced by ``mealy_to_3dfa.py``.

For every input word:

* ``DCValue.TRUE``  means the candidate DFA must accept (output ``True``).
* ``DCValue.FALSE`` means the candidate DFA must reject (output ``False``).
* ``DCValue.DC``    means either candidate result is allowed.

``find_cex`` performs an exact breadth-first search over the product of the
3DFA and the candidate DFA. Therefore, when a counterexample exists among the
reachable states, the returned word is a shortest counterexample.
"""

from __future__ import annotations

from collections import deque
from typing import Hashable, Iterable

from aalpy.automata import Dfa, MooreMachine
from aalpy.base import Oracle
from DCValue import DCValue


TRUE = DCValue.TRUE
FALSE = DCValue.FALSE
DC = DCValue.DC


class ThreeValuedEqOracle(Oracle):
    """Exact equivalence oracle for a three-valued DFA specification.

    Parameters
    ----------
    alphabet:
        Input alphabet used when exploring both automata.
    three_dfa:
        The AALpy ``MooreMachine`` representing the three-valued DFA.

    Notes
    -----
    Unlike AALpy's usual equivalence oracles, this oracle does not query a
    SUL. The three-valued Moore machine itself is the specification against
    which the hypothesis is checked.
    """

    def __init__(
        self,
        alphabet: Iterable[Hashable],
        three_dfa: MooreMachine,
    ) -> None:
        if not isinstance(three_dfa, MooreMachine):
            raise TypeError(
                "three_dfa must be an instance of aalpy.automata.MooreMachine"
            )

        # Oracle requires a SUL argument, but this oracle does not need one.
        super().__init__(list(alphabet), None)
        self.three_dfa = three_dfa

        valid_outputs = {TRUE, FALSE, DC}
        invalid_states = [
            state
            for state in three_dfa.states
            if state.output not in valid_outputs
        ]
        if invalid_states:
            raise ValueError(
                "The three-valued Moore machine contains states with invalid "
                f"outputs: {[(s.state_id, s.output) for s in invalid_states]!r}"
            )

    @staticmethod
    def _is_violation(spec_output: object, dfa_output: bool) -> bool:
        """Check whether the DFA output violates the 3DFA classification."""
        if spec_output is TRUE:
            return dfa_output is not True

        if spec_output is FALSE:
            return dfa_output is not False

        if spec_output is DC:
            return False

        raise ValueError(f"Unexpected three-valued output: {spec_output!r}")

    def find_cex(self, hypothesis: Dfa) -> tuple[Hashable, ...] | None:
        """Return a shortest counterexample, or ``None`` if none exists.

        A counterexample is a word for which:

        * the 3DFA outputs ``DCValue.TRUE`` and the DFA outputs ``False``; or
        * the 3DFA outputs ``DCValue.FALSE`` and the DFA outputs ``True``.

        Words for which the 3DFA outputs ``DCValue.DC`` are ignored.
        """
        if not isinstance(hypothesis, Dfa):
            raise TypeError(
                "hypothesis must be an instance of aalpy.automata.Dfa"
            )

        initial_spec = self.three_dfa.initial_state
        initial_hyp = hypothesis.initial_state

        # DFAState.output is the boolean acceptance value in AALpy's Dfa.
        # See AALpy's current DfaState implementation.
        initial_output = initial_hyp.is_accepting
        if not isinstance(initial_output, bool):
            raise TypeError(
                "The candidate DFA must have boolean state outputs; "
                f"got {initial_output!r} from state {initial_hyp.state_id!r}."
            )

        # The empty word may itself be a counterexample.
        if self._is_violation(initial_spec.output, initial_output):
            return tuple()

        # State pairs are sufficient for visited-state tracking. The first
        # time BFS reaches a pair is through the shortest corresponding word.
        visited = {(initial_spec, initial_hyp)}
        queue = deque([(initial_spec, initial_hyp, tuple())])

        while queue:
            spec_state, hyp_state, word = queue.popleft()

            for symbol in self.alphabet:
                # The 3DFA constructed by the previous script can be partial
                # when the source Mealy machine has no transition for a symbol.
                # Such a branch carries no constraint, so skip it.
                if symbol not in spec_state.transitions:
                    continue

                if symbol not in hyp_state.transitions:
                    raise ValueError(
                        "Candidate DFA is missing a transition for input "
                        f"{symbol!r} from state {hyp_state.state_id!r}."
                    )

                next_spec = spec_state.transitions[symbol]
                next_hyp = hyp_state.transitions[symbol]
                next_word = word + (symbol,)

                next_output = next_hyp.is_accepting
                if not isinstance(next_output, bool):
                    raise TypeError(
                        "The candidate DFA must have boolean state outputs; "
                        f"got {next_output!r} from state {next_hyp.state_id!r}."
                    )

                if self._is_violation(next_spec.output, next_output):
                    print(f"Counterexample found: {next_word}, {next_spec.output} vs {next_output}")
                    return next_word

                product_state = (next_spec, next_hyp)
                if product_state not in visited:
                    visited.add(product_state)
                    queue.append((next_spec, next_hyp, next_word))

        return None

    def find_counterexample(
        self, hypothesis: Dfa
    ) -> tuple[Hashable, ...] | None:
        """Alias for :meth:`find_cex` with the more descriptive name."""
        return self.find_cex(hypothesis)


__all__ = [
    "TRUE",
    "FALSE",
    "DC",
    "ThreeValuedEqOracle",
]

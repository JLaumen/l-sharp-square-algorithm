"""Construct a three-valued DFA using AALpy's built-in MooreMachine.

The construction starts at the Mealy machine's initial state, represented by a
rejecting ('-') Moore state, and recursively explores reachable Mealy states.
For every input symbol from the explicitly supplied input alphabet:

* If the Mealy transition output is not ``error``, the transition is copied to
  the Moore machine. The destination state has output ``'-'`` and corresponds
  to the Mealy destination state.
* If the Mealy transition output is ``error``, the SUL is queried with the
  complete input word leading to that transition. The transition then goes
  directly to one of two shared sinks:
    * ``'+'`` -> the shared accepting sink;
    * ``'?'`` -> the shared unknown sink.

An error transition never recurses into its Mealy destination.

The resulting object is an AALpy ``MooreMachine`` whose state outputs are one
of ``'+'``, ``'-'``, or ``'?'``. In other words, the Moore state output gives
the three-valued classification of a word.

Example
-------

    from my_sul import MySul
    from mealy_to_3dfa import construct_3dfa_from_file
    from aalpy.utils import save_automaton_to_file

    sul = MySul()
    moore_3dfa = construct_3dfa_from_file(
        'model.dot',
        sul,
        ['a', 'b']
    )

    save_automaton_to_file(moore_3dfa, 'model_3dfa.dot', 'dot')
    print(moore_3dfa.step('a'))
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Hashable, Iterable, List, Tuple

from aalpy.automata import MealyMachine, MooreMachine, MooreState
from aalpy.base import SUL
from aalpy.utils import load_automaton_from_file
from DCValue import DCValue


PLUS = DCValue.TRUE
REJECT = DCValue.FALSE
UNKNOWN = DCValue.DC
ERROR_OUTPUT = "error"
PLUS_SINK_ID = "PLUS_SINK"
UNKNOWN_SINK_ID = "UNKNOWN_SINK"


def _get_sul_result(sul: SUL, word: Tuple[Hashable, ...]) -> Any:
    """Run a full membership query and return its final output.

    AALpy's ``SUL.query`` returns one output per input symbol, so for a
    non-empty word the final element is the output associated with the complete
    queried word. A scalar return value is also accepted for convenience with
    custom SUL implementations.
    """
    result = sul.query(word)

    if isinstance(result, (list, tuple)):
        if not result:
            raise ValueError(
                f"The SUL returned no outputs for non-empty query {word!r}."
            )
        return result[-1]

    return result


def construct_3dfa(
    mealy_machine: MealyMachine,
    sul: SUL,
    input_alphabet: Iterable[Hashable],
    *,
    error_output: Any = ERROR_OUTPUT,
    plus_output: Any = PLUS,
    unknown_output: Any = UNKNOWN,
) -> MooreMachine:
    """Construct a three-valued DFA represented by an AALpy MooreMachine.

    Parameters
    ----------
    mealy_machine:
        The AALpy ``MealyMachine`` to traverse.
    sul:
        The AALpy SUL used to classify error transitions.
    input_alphabet:
        The input symbols to use for the construction. Mealy transitions with
        symbols outside this iterable are ignored.
    error_output:
        Mealy transition output that triggers an SUL query. Defaults to
        ``'error'``.
    plus_output, unknown_output:
        Values returned by the SUL corresponding to the ``'+'`` and ``'?'``
        sinks.

    Returns
    -------
    MooreMachine
        An AALpy Moore machine whose state outputs are ``'+'``, ``'-'`` or
        ``'?'``. Every Mealy-derived state has output ``'-'``.
    """
    if not isinstance(mealy_machine, MealyMachine):
        raise TypeError(
            "mealy_machine must be an instance of aalpy.automata.MealyMachine"
        )

    if not isinstance(sul, SUL):
        raise TypeError("sul must be an instance of aalpy.base.SUL")

    print(f"Constructing 3DFA from Mealy machine with {len(mealy_machine.states)} states...")

    # Materialize once so that generators can safely be passed by callers.
    input_alphabet = list(input_alphabet)

    # A one-to-one correspondence between reachable Mealy states and '-' Moore
    # states. The dictionary also acts as the visited-state map.
    state_map: dict[Any, MooreState] = {}
    moore_states: List[MooreState] = []

    def get_or_create_state(mealy_state: Any) -> MooreState:
        if mealy_state in state_map:
            return state_map[mealy_state]

        moore_state = MooreState(str(mealy_state.state_id), REJECT)
        state_map[mealy_state] = moore_state
        moore_states.append(moore_state)
        return moore_state

    # Shared absorbing sinks for all error transitions.
    plus_sink = MooreState(PLUS_SINK_ID, PLUS)
    unknown_sink = MooreState(UNKNOWN_SINK_ID, UNKNOWN)
    moore_states.extend([plus_sink, unknown_sink])

    for symbol in input_alphabet:
        plus_sink.transitions[symbol] = plus_sink
        unknown_sink.transitions[symbol] = unknown_sink

    initial_state = get_or_create_state(mealy_machine.initial_state)
    visited = {mealy_machine.initial_state}
    queried_errors: dict[Tuple[Hashable, ...], Any] = {}

    def visit(mealy_state: Any, current_word: Tuple[Hashable, ...]) -> None:
        """Recursively explore one reachable Mealy state."""
        moore_state = get_or_create_state(mealy_state)

        for symbol in input_alphabet:
            # The requested alphabet is authoritative: absent transitions are
            # simply ignored, as are transitions on other input symbols.
            if symbol not in mealy_state.transitions:
                continue

            destination_mealy_state = mealy_state.transitions[symbol]
            output = mealy_state.output_fun[symbol]
            next_word = current_word + (symbol,)

            if output == error_output:
                # Error transitions terminate this branch. The Mealy
                # destination is deliberately NOT explored.
                if next_word not in queried_errors:
                    queried_errors[next_word] = _get_sul_result(sul, next_word)

                sul_result = queried_errors[next_word]
                if sul_result is plus_output:
                    moore_state.transitions[symbol] = plus_sink
                    continue

                if sul_result is unknown_output:
                    moore_state.transitions[symbol] = unknown_sink
                    continue

                if sul_result is REJECT:
                    # print(f"Warning: SUL rejected error-prefix {next_word!r}. ")
                    # The SUL classifies this error-prefix as rejecting. In
                    # that case the error transition is converted back into a
                    # normal transition to the corresponding '-' state, and
                    # we continue recursive exploration from the Mealy
                    # destination.
                    destination_moore_state = get_or_create_state(
                        destination_mealy_state
                    )
                    moore_state.transitions[symbol] = destination_moore_state

                    if destination_mealy_state not in visited:
                        visited.add(destination_mealy_state)
                        visit(destination_mealy_state, next_word)
                    continue

                raise ValueError(
                    f"Unexpected SUL output {sul_result!r} for word {next_word!r}. "
                    f"Expected {plus_output!r}, {unknown_output!r}, or {REJECT!r}."
                )

            # Non-error transition: create/reuse the corresponding '-' state
            # and recurse only when that Mealy state has not been visited yet.
            destination_moore_state = get_or_create_state(destination_mealy_state)
            moore_state.transitions[symbol] = destination_moore_state

            if destination_mealy_state not in visited:
                visited.add(destination_mealy_state)
                visit(destination_mealy_state, next_word)

    visit(mealy_machine.initial_state, tuple())

    # AALpy's built-in MooreMachine stores outputs on states, which is exactly
    # what is needed for the + / - / ? classification of the constructed 3DFA.
    moore_machine = MooreMachine(initial_state, moore_states)

    # Populate AALpy's optional shortest access sequences. This also makes the
    # returned model immediately useful with AALpy functionality that expects
    # state.prefix to be available.
    moore_machine.compute_prefixes()

    return moore_machine


def construct_3dfa_from_file(
    mealy_path: str | Path,
    sul: SUL,
    input_alphabet: Iterable[Hashable],
    **kwargs: Any,
) -> MooreMachine:
    """Load a Mealy machine from a file and construct the 3DFA MooreMachine."""
    print(f"Loading Mealy machine from {mealy_path}...")
    mealy_machine = load_automaton_from_file(
        mealy_path,
        automaton_type="mealy",
    )
    return construct_3dfa(mealy_machine, sul, input_alphabet, **kwargs)


__all__ = [
    "PLUS",
    "REJECT",
    "UNKNOWN",
    "ERROR_OUTPUT",
    "PLUS_SINK_ID",
    "UNKNOWN_SINK_ID",
    "construct_3dfa",
    "construct_3dfa_from_file",
]

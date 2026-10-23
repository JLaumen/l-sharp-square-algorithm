"""Entry point of the passive L#-square learner."""

from __future__ import annotations

import time
from typing import Any

from aalpy.automata import Dfa

from DCNode import DCNode
from ObservationTreeSquare import ObservationTreeSquare


def _count_nodes(root: DCNode) -> tuple[int, int]:
    """Count all nodes and the nodes leading to a known observation in the tree rooted at ``root``."""
    nodes = informative_nodes = 0
    stack = [root]
    while stack:
        node = stack.pop()
        nodes += 1
        informative_nodes += node.leads_to_known
        stack.extend(node.successors.values())
    return nodes, informative_nodes


def run_lsharp_square(alphabet: list, words: list[tuple[tuple[Any, ...], bool]],
                      return_data: bool = False) -> Dfa | tuple[Dfa, dict[str, Any]]:
    """Learn a smallest DFA that agrees with a set of labelled words.

    The words are put in an observation tree and a DFA with the fewest states
    into which the tree maps is found with a SAT solver. Words that are not
    given are don't-care. There is no interaction with a teacher.

    Args:
        alphabet: Finite input alphabet of the words, with at least two symbols.
        words: Input words as tuples, paired with their output: ``True`` for
            accepted and ``False`` for rejected.
        return_data: Whether to return learning statistics together with the
            learned DFA. If ``True``, the return value is a ``(dfa, info)``
            tuple; otherwise, only the DFA is returned.

    Returns:
        The learned DFA. If ``return_data`` is ``True``, returns a tuple
        containing the DFA and a dictionary with statistics: DFA size, learning,
        solver and total times, and observation-tree node counts.

    Raises:
        ValueError: If a word is given with two different outputs.
    """
    start_time = time.time()
    observation_tree = ObservationTreeSquare(alphabet, words)
    dfa = observation_tree.learn()

    total_time = time.time() - start_time
    solver_time = observation_tree.solver_time
    nodes, informative_nodes = _count_nodes(observation_tree.root)

    if return_data:
        info = {
            "automaton_size": dfa.size,
            "learning_time": total_time - solver_time,
            "solver_time": solver_time,
            "total_time": total_time,
            "nodes": nodes,
            "informative_nodes": informative_nodes,
        }
        return dfa, info

    return dfa

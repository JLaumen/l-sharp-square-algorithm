"""Entry point of the L#-square learner for incomplete teachers."""

from __future__ import annotations

import time
from typing import Any

from aalpy.automata import Dfa
from aalpy.base import Oracle

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


def run_lsharp_square(alphabet: list, sul: Any, eq_oracle: Oracle, return_data: bool = False,
                      assume_prefix_closed: bool = True) -> Dfa | tuple[Dfa, dict[str, Any]]:
    """Learn a DFA with the L#-square algorithm.

    The algorithm incrementally builds an observation tree from the incomplete
    system under learning, constructs a DFA hypothesis consistent with the
    observations, and submits that hypothesis to the equivalence oracle. If a
    counterexample is returned, the corresponding observation is added to the
    tree and learning continues until the equivalence oracle accepts the
    hypothesis.

    The system under learning must provide a ``query`` method accepting an
    input word as a tuple and returning the observed output for that word. It
    must also expose ``num_queries`` and ``num_steps`` counters for learning
    statistics. The equivalence oracle must implement ``find_cex`` and expose
    ``num_queries`` and ``num_steps`` counters.

    Args:
        alphabet: Finite input alphabet used by the system under learning,
            with at least two symbols.
        sul: Incomplete system under learning. Its ``query`` method is used to
            obtain observations for counterexamples and by the observation
            tree during hypothesis construction.
        eq_oracle: Equivalence oracle used to check each candidate hypothesis.
            It must return an input word as a counterexample, or ``None`` when
            the hypothesis is considered correct.
        return_data: Whether to return learning statistics together with the
            final hypothesis. If ``True``, the return value is a
            ``(hypothesis, info)`` tuple; otherwise, only the hypothesis is
            returned.
        assume_prefix_closed: Whether an unknown (don't-care) observation for
            a prefix should also be treated as unknown for all of its
            extensions.

    Returns:
        The learned DFA hypothesis. If ``return_data`` is ``True``, returns a
        tuple containing the hypothesis and a dictionary with statistics:
        learning rounds, hypothesis size, learning, solver, equivalence-oracle
        and total times, membership-query counts, observation-tree node
        counts, and equivalence-oracle query counts.
    """
    observation_tree = ObservationTreeSquare(alphabet, sul, assume_prefix_closed)

    start_time = time.time()
    learning_rounds = 0
    eq_query_time = 0
    validity_queries = 0

    while True:
        learning_rounds += 1

        # Build a hypothesis from the current observation tree; ``None``
        # means the state bound was too small and has been increased.
        hypothesis = observation_tree.build_hypothesis()
        if hypothesis is None:
            continue

        # Ask the equivalence oracle for a counterexample.
        eq_query_start = time.time()
        counterexample = eq_oracle.find_cex(hypothesis)
        eq_query_time += time.time() - eq_query_start
        validity_queries += 1

        if counterexample is None:
            break

        # Add the counterexample to the observation tree.
        observation_tree.process_counter_example(counterexample, sul.query(tuple(counterexample)))

    total_time = time.time() - start_time
    solver_time = observation_tree.solver_time
    learning_time = total_time - eq_query_time - solver_time
    nodes, informative_nodes = _count_nodes(observation_tree.root)

    info = {
        # Learning algorithm
        "learning_rounds": learning_rounds,
        "automaton_size": hypothesis.size if hypothesis else 0,
        "learning_time": learning_time,
        "solver_time": solver_time,
        "eq_oracle_time": eq_query_time,
        "total_time": total_time,
        "membership_queries": sul.num_queries,
        "validity_query": validity_queries,
        # Observation tree
        "nodes": nodes,
        "informative_nodes": informative_nodes,
        # System under learning
        "sul_steps": sul.num_steps,
        # Equivalence oracle
        "queries_eq_oracle": eq_oracle.num_queries,
        "steps_eq_oracle": eq_oracle.num_steps,
    }

    if return_data:
        return hypothesis, info

    return hypothesis

import time
from typing import Any

from aalpy.base import Oracle

from ObservationTreeSquare import ObservationTreeSquare


def _count_nodes(node: Any) -> int:
    """Count nodes in the observation tree rooted at ``node``."""
    return 1 + sum(_count_nodes(successor) for successor in node.successors.values())


def _count_informative_nodes(node: Any) -> int:
    """Count observation-tree nodes that lead to a known observation."""
    return (1 if node.leads_to_known else 0) + sum(
        _count_informative_nodes(successor) for successor in node.successors.values())


def run_lsharp_square(alphabet: list, sul: Any, eq_oracle: Oracle, return_data: bool = False,
                      replace_basis: bool = True, assume_prefix_closed: bool = True, ) -> tuple | None:
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
        alphabet: Finite input alphabet used by the system under learning.
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
        replace_basis: Whether the observation tree may replace its current
            basis with a larger maximum clique of pairwise-apart nodes.
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

    observation_tree = ObservationTreeSquare(alphabet, sul, replace_basis, assume_prefix_closed, )

    start_time = time.time()
    learning_rounds = 0
    eq_query_time = 0
    validity_queries = 0

    while True:
        learning_rounds += 1

        # Build a hypothesis from the current observation tree.
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
        observation_tree.process_counter_example(counterexample, sul.query(tuple(counterexample)), )

    total_time = time.time() - start_time
    solver_time = observation_tree.solver_time
    learning_time = total_time - eq_query_time - solver_time

    info = {  # Learning algorithm
        "learning_rounds": learning_rounds, "automaton_size": hypothesis.size if hypothesis else 0,
        "learning_time": learning_time, "solver_time": solver_time, "eq_oracle_time": eq_query_time,
        "total_time": total_time, "membership_queries": sul.num_queries, "validity_query": validity_queries,

        # Observation tree
        "nodes": _count_nodes(observation_tree.root),
        "informative_nodes": _count_informative_nodes(observation_tree.root),

        # System under learning
        "sul_steps": sul.num_steps,

        # Equivalence oracle
        "queries_eq_oracle": eq_oracle.num_queries, "steps_eq_oracle": eq_oracle.num_steps, }

    if return_data:
        return hypothesis, info

    return hypothesis

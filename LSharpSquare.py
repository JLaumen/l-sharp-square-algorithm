import time

from aalpy.base import Oracle, SUL

from ObservationTreeSquare import ObservationTreeSquare


def run_lsharp_square(alphabet: list, sul: SUL, eq_oracle: Oracle, return_data: bool = False,
                      solver_timeout: int = 3600, replace_basis: bool = True, use_compatibility: bool = False,
                      assume_prefix_closed: bool = True, ) -> tuple | None:
    """Run the L#-square learning algorithm."""

    observation_tree = ObservationTreeSquare(alphabet, sul, replace_basis, use_compatibility, assume_prefix_closed, )

    start_time = time.time()
    learning_rounds = 0
    eq_query_time = 0
    validity_queries = 0
    hypothesis = None

    while True:
        learning_rounds += 1

        if time.time() - start_time > solver_timeout:
            break

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

        # Determine the expected output for the counterexample.
        hypothesis.reset_to_initial()
        last_output = hypothesis.step(None)

        for letter in counterexample:
            last_output = hypothesis.step(letter)

        # Add the counterexample to the observation tree.
        observation_tree.process_counter_example(counterexample, not last_output, )

    total_time = time.time() - start_time
    smt_time = observation_tree.smt_time
    learning_time = total_time - eq_query_time - smt_time

    info = {  # Learning algorithm
        "learning_rounds": learning_rounds, "automaton_size": hypothesis.size if hypothesis else 0,
        "learning_time": learning_time, "smt_time": smt_time, "eq_oracle_time": eq_query_time, "total_time": total_time,
        "queries_learning": sul.num_queries, "successful_queries_learning": sul.num_successful_queries,
        "validity_query": validity_queries,

        # Observation tree
        "nodes": observation_tree.get_size(), "informative_nodes": observation_tree.count_informative_nodes(),

        # System under learning
        "sul_steps": sul.num_steps, "cache_saved": sul.num_cached_queries,

        # Equivalence oracle
        "queries_eq_oracle": eq_oracle.num_queries, "steps_eq_oracle": eq_oracle.num_steps, }

    if return_data:
        return hypothesis, info

    return hypothesis

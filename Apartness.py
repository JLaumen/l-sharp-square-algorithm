from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Any

from DCValue import DCValue


class Apartness:
    """Utilities for checking and witnessing apartness in an observation tree.

    Two observation-tree nodes are apart when there is a suffix for which
    both nodes have known outputs and those outputs differ.
    """

    @staticmethod
    def _outputs_are_apart(first_output: DCValue | None, second_output: DCValue | None, ) -> bool:
        """Return whether two stored outputs constitute a distinction."""
        if first_output is None or second_output is None:
            return False

        if not first_output.is_known() or not second_output.is_known():
            return False

        return first_output != second_output

    @staticmethod
    def compute_witness(state1: Any, state2: Any, observation_tree: Any, ) -> list[Any] | None:
        """Find a distinguishing sequence between two observation-tree nodes.

        Args:
            state1: First observation-tree node.
            state2: Second observation-tree node.
            observation_tree: Observation tree containing the states.

        Returns:
            A distinguishing input sequence, or ``None`` if the states are
            not apart.
        """
        destination = Apartness._show_states_are_apart_moore(state1, state2, observation_tree.alphabet, )

        if destination is None:
            return None

        return observation_tree.get_transfer_sequence(state1, destination, )

    @staticmethod
    def states_are_apart(state1: Any, state2: Any, observation_tree: Any, ) -> bool:
        """Check whether two observation-tree nodes are apart.

        Args:
            state1: First observation-tree node.
            state2: Second observation-tree node.
            observation_tree: Observation tree containing the states.

        Returns:
            ``True`` if a known output difference can be observed from the
            two states, otherwise ``False``.
        """
        return (Apartness._show_states_are_apart_moore(state1, state2, observation_tree.alphabet, ) is not None)

    @staticmethod
    def _show_states_are_apart_moore(first: Any, second: Any, alphabet: list[Any], ) -> Any | None:
        """Find a node witnessing apartness between two Moore states."""
        pairs = deque([(first, second)])

        while pairs:
            first_node, second_node = pairs.popleft()

            if first_node is None or second_node is None:
                continue

            if Apartness._outputs_are_apart(first_node.output, second_node.output, ):
                return first_node

            for input_value in alphabet:
                pairs.append((first_node.get_successor(input_value), second_node.get_successor(input_value),))

        return None

    @staticmethod
    def get_successors(node: Any, input_sequence: Iterable[Any], ) -> Any | None:
        """Follow an input sequence from an observation-tree node.

        Args:
            node: Starting observation-tree node.
            input_sequence: Input sequence to follow.

        Returns:
            The reached node, or ``None`` if the sequence is not present.
        """
        for input_value in input_sequence:
            if node is None:
                return None
            node = node.get_successor(input_value)

        return node

    @staticmethod
    def get_distinguishing_sequences(group: list[Any], observation_tree: Any, ) -> Iterable[list[Any]]:
        """Generate suffixes that distinguish members of a node group.

        A suffix is yielded when at least two nodes in the group have known
        and different outputs after applying that suffix.

        Args:
            group: Observation-tree nodes to distinguish.
            observation_tree: Observation tree containing the nodes.

        Yields:
            Input sequences that distinguish at least two nodes.
        """
        yield from Apartness._get_distinguishing_sequences_moore(group, observation_tree.alphabet, )

    @staticmethod
    def _get_distinguishing_sequences_moore(group: list[Any], alphabet: list[Any], ) -> Iterable[list[Any]]:
        """Generate distinguishing sequences for Moore/DFA nodes."""
        groups = deque([([], group)])

        while groups:
            access_sequence, current_group = groups.popleft()

            valid_group = [node for node in current_group if node is not None and node.leads_to_known]

            if len(valid_group) >= 2:
                has_true = any(node.output is DCValue.TRUE for node in valid_group)
                has_false = any(node.output is DCValue.FALSE for node in valid_group)

                if has_true and has_false:
                    yield access_sequence

                for input_value in alphabet:
                    groups.append(
                        (access_sequence + [input_value], [node.get_successor(input_value) for node in valid_group],))

    @staticmethod
    def compute_witness_in_tree_and_hypothesis_states(observation_tree: Any, observation_tree_state: Any,
            hypothesis_state: Any, ) -> list[Any] | None:
        """Find a distinguishing sequence between a tree and DFA state.

        Args:
            observation_tree: Observation tree.
            observation_tree_state: Starting observation-tree node.
            hypothesis_state: Hypothesis DFA state.

        Returns:
            A distinguishing input sequence, or ``None`` if no distinction
            is currently known.
        """
        return Apartness.compute_witness_in_tree_and_hypothesis_states_moore(observation_tree, observation_tree_state,
            hypothesis_state, )

    @staticmethod
    def compute_witness_in_tree_and_hypothesis_states_moore(observation_tree: Any, observation_tree_state: Any,
            hypothesis_state: Any, ) -> list[Any] | None:
        """Compare an incomplete observation tree with a DFA hypothesis."""
        pairs = deque([(observation_tree_state, hypothesis_state)])

        while pairs:
            tree_state, hyp_state = pairs.popleft()

            if tree_state is None or hyp_state is None:
                continue

            tree_output = tree_state.output

            # Unknown/unobserved tree outputs cannot distinguish a hypothesis.
            if (tree_output is not None and tree_output.is_known()):
                hypothesis_output = hyp_state.is_accepting
                tree_accepts = tree_output is DCValue.TRUE

                if tree_accepts != hypothesis_output:
                    return observation_tree.get_transfer_sequence(observation_tree_state, tree_state, )

            for input_value in observation_tree.alphabet:
                if input_value in hyp_state.transitions:
                    pairs.append((tree_state.get_successor(input_value), hyp_state.transitions[input_value],))

        return None

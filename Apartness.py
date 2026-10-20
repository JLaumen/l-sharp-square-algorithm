"""Apartness of observation-tree nodes with three-valued (don't-care) outputs.

AALpy's own ``Apartness`` is not used: it has no distinguishing-sequence
generator, and its checks rely on ``DCValue``'s non-standard equality instead
of treating don't-care outputs explicitly.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from DCValue import DCValue

if TYPE_CHECKING:
    from DCNode import DCNode


class Apartness:
    """Apartness checks and distinguishing suffixes for an observation tree.

    Two observation-tree nodes are apart when there is a suffix for which
    both nodes have known outputs and those outputs differ. Because the tree
    only grows and known outputs never change, nodes that are apart stay
    apart.

    The ``observation_tree`` arguments only need an ``alphabet`` attribute.
    """

    @staticmethod
    def states_are_apart(state1: DCNode | None, state2: DCNode | None, observation_tree: Any) -> bool:
        """Check whether two observation-tree nodes are apart.

        The common subtrees are searched breadth-first. Pairs in which either
        node has no known output in its subtree are pruned, since neither they
        nor their descendants can witness apartness.

        Args:
            state1: First observation-tree node.
            state2: Second observation-tree node.
            observation_tree: Observation tree containing the states.

        Returns:
            ``True`` if a known output difference can be observed from the
            two states, otherwise ``False``.
        """
        if state1 is None or state2 is None or not state1.leads_to_known or not state2.leads_to_known:
            return False
        alphabet = observation_tree.alphabet
        pairs = deque([(state1, state2)])

        while pairs:
            first_node, second_node = pairs.popleft()

            # Known outputs are the singletons TRUE and FALSE, so they differ exactly when exactly one is TRUE.
            first_output, second_output = first_node.output, second_node.output
            if ((first_output is DCValue.TRUE and second_output is DCValue.FALSE)
                    or (first_output is DCValue.FALSE and second_output is DCValue.TRUE)):
                return True

            first_successors = first_node.successors
            second_successors = second_node.successors
            for input_value in alphabet:
                first_successor = first_successors.get(input_value)
                if first_successor is None or not first_successor.leads_to_known:
                    continue
                second_successor = second_successors.get(input_value)
                if second_successor is None or not second_successor.leads_to_known:
                    continue
                pairs.append((first_successor, second_successor))

        return False

    @staticmethod
    def get_distinguishing_sequences(group: list[DCNode], observation_tree: Any) -> Iterator[list[Any]]:
        """Generate suffixes that distinguish members of a node group, shortest first.

        A suffix is yielded when at least two nodes in the group have known
        and different outputs after applying that suffix.

        The generator is lazy and callers extend the tree between yields, so
        successors are fixed when a group is enqueued while ``leads_to_known``
        is only evaluated when the group is processed.

        Args:
            group: Observation-tree nodes to distinguish.
            observation_tree: Observation tree containing the nodes.

        Yields:
            Input sequences that distinguish at least two nodes.
        """
        alphabet = observation_tree.alphabet
        groups = deque([([], [node for node in group if node is not None])])

        while groups:
            access_sequence, current_group = groups.popleft()

            valid_group = [node for node in current_group if node.leads_to_known]

            if len(valid_group) >= 2:
                has_true = any(node.output is DCValue.TRUE for node in valid_group)
                has_false = any(node.output is DCValue.FALSE for node in valid_group)

                if has_true and has_false:
                    yield access_sequence

                for input_value in alphabet:
                    successors = [successor for node in valid_group
                                  if (successor := node.successors.get(input_value)) is not None]
                    groups.append((access_sequence + [input_value], successors))

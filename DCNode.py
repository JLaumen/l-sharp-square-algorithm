"""Nodes of an observation tree with three-valued (don't-care) outputs."""

from __future__ import annotations

from typing import Any

from DCValue import DCValue


class DCNode:
    """A node of an observation tree for an incomplete teacher.

    Each node corresponds to the input word on the path from the root to it
    and stores the output observed for that word.

    Attributes:
        id: Creation number; also the node's hash, which keeps the iteration
            order of node sets and dicts deterministic across runs.
        output: Observed output, or ``None`` before one has been set.
        successors: Child node for each input symbol observed so far.
        parent: Parent node, or ``None`` for the root.
        input_to_parent: Input symbol on the edge from ``parent`` to this node.
        access_sequence: Input word from the root to this node.
        leads_to_known: Whether this node or a descendant has a known
            output. Once set it is never cleared.
    """

    __slots__ = ("id", "output", "successors", "parent", "input_to_parent", "access_sequence", "leads_to_known")

    _id_counter = 0

    # Render a node whose output was never set and that has a single successor
    # as that successor, which keeps long counterexample chains readable.
    COMPACT_COUNTER_EXAMPLES = True

    def __init__(self, parent: DCNode | None = None) -> None:
        DCNode._id_counter += 1
        self.id = DCNode._id_counter
        self.output: DCValue | None = None
        self.successors: dict[Any, DCNode] = {}
        self.parent = parent
        self.input_to_parent: Any = None
        self.access_sequence: list[Any] = []
        self.leads_to_known = False

    def __hash__(self) -> int:
        # Equal to ``hash(self.id)`` for positive ids, so set and dict orderings are unchanged.
        return self.id

    def set_leads_to_known(self) -> None:
        """Mark this node and all its ancestors as leading to a known output."""
        node = self
        while node is not None and not node.leads_to_known:
            node.leads_to_known = True
            node = node.parent

    def has_known_output(self) -> bool:
        """Return whether this node's output is set and not don't-care."""
        return self.output is not None and self.output.is_known()

    def set_output(self, output: DCValue) -> None:
        """Store ``output`` for this node, propagating ``leads_to_known`` if it is known."""
        self.output = output

        if output.is_known():
            self.set_leads_to_known()

    def add_successor(self, input_val: Any, output_val: DCValue, successor_node: DCNode) -> None:
        """Attach ``successor_node`` below this node via ``input_val`` and give it ``output_val``."""
        self.successors[input_val] = successor_node
        successor_node.parent = self
        successor_node.input_to_parent = input_val
        successor_node.set_output(output_val)
        successor_node.access_sequence = self.access_sequence + [input_val]

    def get_successor(self, input_val: Any) -> DCNode | None:
        """Return the successor for ``input_val``, or ``None`` if it was not observed."""
        return self.successors.get(input_val)

    def extend_and_get(self, inp: Any, output: DCValue) -> DCNode:
        """Return the successor for ``inp``, creating it with ``output`` if it does not exist.

        The output of an existing successor is left unchanged.
        """
        successor_node = self.successors.get(inp)
        if successor_node is None:
            successor_node = DCNode(parent=self)
            self.add_successor(inp, output, successor_node)
        return successor_node

    def __str__(self) -> str:
        if self.COMPACT_COUNTER_EXAMPLES and self.output is None and len(self.successors) == 1:
            successor = next(iter(self.successors.values()))
            return str(successor)

        result = "node " + str(self.access_sequence) + " / " + str(self.output)

        for input_val, successor in self.successors.items():
            result += "\n" + str(input_val) + ":\n"
            result += "\t" + str(successor).replace("\n", "\n\t")

        return result

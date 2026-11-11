"""SAT-based passive learner of separating deterministic finite automata.

All labelled words are put in an observation tree; words without a label are don't-care.
:meth:`ObservationTreeSquare.learn` then finds a smallest DFA into which the tree maps:

1. Promote nodes that are apart from the whole basis, breadth-first, until none is left. Two
   nodes are apart if some suffix leads from both to known, different outputs. The basis nodes
   are pairwise apart, so they need distinct states, and the size of the basis is a lower bound
   on the number of states.
2. Encode "a DFA with at most ``cap`` states into which the tree maps" as CNF
   (:class:`~.DfaEncoding.DfaEncoding`), where every node may map to the basis states it is not
   apart from and to all non-basis states. Solve it with CaDiCaL for the bounds ``lower, ...,
   cap`` under assumptions, fewest states first, so one encoding and one solver serve several
   bounds. If all are unsatisfiable, repeat with the bounds above ``cap``.

Only *informative* nodes, which have a known output in their subtree, matter. They are numbered
breadth-first, so a parent always comes before its children.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable, Sequence
from itertools import repeat
from typing import Any

from aalpy.automata import Dfa, DfaState
from pysat.solvers import Cadical195
from pysolvers import cadical195_add_cl

from DfaEncoding import DfaEncoding, HypothesisMapping

# CaDiCaL options for the hypothesis search: run in stable mode only (target phases and
# reluctant restarts), which favours finding satisfying assignments.
_SOLVER_OPTIONS = {"stabilizeonly": 1}

# Each encoding allows this many states above the lowest bound it is used for. Larger values build
# fewer but larger encodings; 4 was the fastest of 2, 4 and 8 on the DFAMiner benchmarks.
_EXTRA_STATES_PER_ENCODING = 4


class ObservationTreeSquare:
    """Learn a smallest separating DFA from a set of labelled words.

    Args:
        alphabet: Input symbols of the words; assumed to contain at least two symbols.
        words: Input words paired with their output: ``True`` for accepted and ``False`` for
            rejected. All other words are don't-care.

    Raises:
        ValueError: If a word occurs with two different outputs.

    Attributes:
        alphabet: Input alphabet used by the learner.
        solver_time: Accumulated time spent building and solving SAT instances.
        number_of_nodes: Number of nodes of the observation tree.
        number_of_informative_nodes: Number of nodes with a known output in their subtree.
    """

    def __init__(self, alphabet: Sequence[Any], words: Iterable[tuple[Sequence[Any], bool]]) -> None:
        self.alphabet = list(alphabet)
        self.solver_time = 0.0
        letters = {symbol: letter for letter, symbol in enumerate(self.alphabet)}
        children: list[dict[int, int]] = [{}]  # Per tree node: child per letter, in insertion order.
        outputs = [0]  # Per tree node: 1 accepting, -1 rejecting, 0 unknown.
        for word, accepts in words:
            node = 0
            for symbol in word:
                child = children[node].get(letters[symbol])
                if child is None:
                    child = children[node][letters[symbol]] = len(children)
                    children.append({})
                    outputs.append(0)
                node = child
            sign = 1 if accepts else -1
            if outputs[node] == -sign:
                raise ValueError(f"Word {tuple(word)} has conflicting outputs")
            outputs[node] = sign

        informative = [sign != 0 for sign in outputs]
        for node in reversed(range(len(children))):  # Children are numbered after their parents.
            informative[node] = informative[node] or any(informative[child] for child in children[node].values())
        self.number_of_nodes = len(children)
        self.number_of_informative_nodes = sum(informative)

        tree_nodes = [0]
        self._parents: list[tuple[int, int] | None] = [None]  # Per informative node: (parent, letter).
        for node, tree_node in enumerate(tree_nodes):  # ``tree_nodes`` grows while iterating: a BFS queue.
            for letter, child in children[tree_node].items():
                if informative[child]:
                    self._parents.append((node, letter))
                    tree_nodes.append(child)
        self._outputs = [outputs[tree_node] for tree_node in tree_nodes]
        self._index_suffixes()

    def _index_suffixes(self) -> None:
        """Index, per suffix ``w``, the nodes ``u`` for which ``uw`` has a known output.

        Suffixes are numbered through a trie of reversed words, with suffix ``0`` the empty word.
        Sets ``_suffix_occurrences`` (per suffix: ``(u, output of uw)``), ``_below`` (per node:
        ``(w, output of uw)``) and ``_output_masks`` (per suffix and output: bitmask of the nodes).
        """
        suffix_numbers: dict[tuple[int, int], int] = {}
        occurrences: list[list[tuple[int, int]]] = [[]]
        below: list[list[tuple[int, int]]] = [[] for _ in self._outputs]
        for node, sign in enumerate(self._outputs):
            if not sign:
                continue
            suffix, ancestor = 0, node
            while True:
                occurrences[suffix].append((ancestor, sign))
                below[ancestor].append((suffix, sign))
                parent = self._parents[ancestor]
                if parent is None:
                    break
                ancestor, letter = parent
                suffix = suffix_numbers.setdefault((letter, suffix), len(occurrences))
                if suffix == len(occurrences):
                    occurrences.append([])
        self._suffix_occurrences, self._below = occurrences, below
        self._output_masks = []
        for suffix_occurrences in occurrences:
            masks = {1: 0, -1: 0}
            for node, sign in suffix_occurrences:
                masks[sign] |= 1 << node
            self._output_masks.append(masks)

    def _apart_mask(self, node: int) -> int:
        """Bitmask of the nodes apart from ``node``."""
        mask = 0
        for suffix, sign in self._below[node]:
            mask |= self._output_masks[suffix][-sign]
        return mask

    def learn(self) -> Dfa:
        """Return a DFA with the fewest states that agrees with all labelled words."""
        # Basis: repeatedly the first node, breadth-first, that is apart from all basis nodes so far.
        # Per basis state, the nodes that may map to it are those not apart from its basis node.
        everything = (1 << len(self._outputs)) - 1
        basis: list[int] = []
        compatible: list[int] = []
        candidates = everything  # Nodes apart from all basis nodes so far; the root comes first.
        while candidates:
            node = (candidates & -candidates).bit_length() - 1
            apart = self._apart_mask(node)
            basis.append(node)
            compatible.append(everything & ~apart)
            candidates &= apart

        lower = len(basis)
        while True:
            cap = lower + _EXTRA_STATES_PER_ENCODING
            hypothesis = self._find_hypothesis(basis, compatible, lower, cap)
            if hypothesis is not None:
                return self._construct_hypothesis(*hypothesis)
            lower = cap + 1

    def _find_hypothesis(self, basis: Sequence[int], compatible: Sequence[int], lower: int,
                         cap: int) -> HypothesisMapping | None:
        """Find a DFA with the fewest states in ``lower .. cap``; ``None`` if there is none."""
        start_time = time.perf_counter()
        # Basis nodes are pairwise apart, so a basis node is only compatible with its own state;
        # other nodes may also map to all non-basis states. Known outputs need no pruning: a node
        # is apart from every basis node with the opposite known output.
        domains: list[list[int]] = [[] for _ in self._outputs]
        for state, mask in enumerate(compatible):
            while mask:
                low = mask & -mask
                domains[low.bit_length() - 1].append(state)
                mask ^= low
        free_states = list(range(len(basis), cap))
        for node in set(range(len(domains))) - set(basis):
            domains[node].extend(free_states)

        encoding = DfaEncoding(cap, len(basis), len(self.alphabet), self._parents, domains, self._outputs,
                               self._suffix_occurrences)
        solver = Cadical195()
        try:
            solver.configure(_SOLVER_OPTIONS)  # Options must be set before any clause is added.
            # Adding clauses through pysat's C function skips the per-clause checks of append_formula;
            # map() runs the loop in C and deque() consumes it.
            deque(map(cadical195_add_cl, repeat(solver.cadical), encoding.clauses), maxlen=0)
            for bound in range(lower, cap + 1):
                if solver.solve(assumptions=[encoding.off[bound]] if bound < cap else []):
                    return encoding.decode(solver.get_model(), bound)
            return None
        finally:
            solver.delete()
            self.solver_time += time.perf_counter() - start_time

    def _construct_hypothesis(self, transitions: Sequence[Sequence[int]], accepting: Sequence[bool]) -> Dfa:
        """Build an AALpy DFA with states ``s0, s1, ...``, starting in ``s0``."""
        states = [DfaState(f"s{index}") for index in range(len(accepting))]
        for state, is_accepting, targets in zip(states, accepting, transitions):
            state.is_accepting = is_accepting
            state.transitions = {symbol: states[target] for symbol, target in zip(self.alphabet, targets)}
        return Dfa(states[0], states)

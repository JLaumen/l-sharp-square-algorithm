from collections import deque


class Apartness:
    @staticmethod
    def incompatible_output(output1, output2):
        return output1 != output2 and output1 is not None and output2 is not None and output1 != "unknown" and output2 != "unknown"

    @staticmethod
    def compute_witness(state1, state2, ob_tree):
        # Finds a distinguishing sequence between two states if they are apart based on the observation tree
        if ob_tree.automaton_type == 'mealy':
            state1_destination = Apartness._show_states_are_apart_mealy(state1, state2, ob_tree.alphabet)
        else:
            state1_destination = Apartness._show_states_are_apart_moore(state1, state2, ob_tree.alphabet)
        if not state1_destination:
            return None
        return ob_tree.get_transfer_sequence(state1, state1_destination)

    @staticmethod
    def states_are_apart(state1, state2, ob_tree):
        # Checks if two states are apart by checking any output difference in the observation tree
        if ob_tree.automaton_type == 'mealy':
            return Apartness._show_states_are_apart_mealy(state1, state2, ob_tree.alphabet) is not None
        else:
            return Apartness._show_states_are_apart_moore(state1, state2, ob_tree.alphabet) is not None

    @staticmethod
    def _show_states_are_apart_mealy(first, second, alphabet):
        # Identifies if two states can be distinguished by any input-output pair in the provided alphabet
        pairs = deque([(first, second)])

        while pairs:
            first_node, second_node = pairs.popleft()
            for input_val in alphabet:
                first_output = first_node.get_output(input_val)
                second_output = second_node.get_output(input_val)

                if first_output is not None and second_output is not None:
                    if first_output != second_output and (
                            first_output not in ["unknown", None] and second_output not in ["unknown", None]):
                        return first_node.get_successor(input_val)

                    pairs.append((first_node.get_successor(input_val), second_node.get_successor(input_val)))

        return None

    @staticmethod
    def _show_states_are_apart_moore(first, second, alphabet):
        # Identifies if two states can be distinguished by any input-output pair in the provided alphabet
        pairs = deque([(first, second)])
        while pairs:
            first_node, second_node = pairs.popleft()
            if first_node is not None and second_node is not None:
                first_output = first_node.output
                second_output = second_node.output
                if first_output != second_output and (
                        first_output not in ["unknown", None] and second_output not in ["unknown", None]):
                    return first_node

                for input_val in alphabet:
                    pairs.append((first_node.get_successor(input_val), second_node.get_successor(input_val)))

        return None

    @staticmethod
    def get_successors(node, input_val):
        for inp in input_val:
            if node is None:
                return None
            node = node.get_successor(inp)
        return node

    @staticmethod
    def states_are_incompatible(first, second, ob_tree, experiment=True):
        if not first.leads_to_known or not second.leads_to_known:
            return False
        if not ob_tree.use_compatibility:
            return Apartness.states_are_apart(first, second, ob_tree)

        # Assumes that a node cannot be a descendant of a node with a higher id
        if second.id < first.id:
            first, second = second, first

        # This should not happen, but just in case
        if first.id == second.id:
            return False

        # Checking apartness is easier than checking incompatibility,
        # so we check that first
        if Apartness.states_are_apart(first, second, ob_tree):
            return True

        # Incompatibility can now only occur if the second node is a descendant of the first node.
        parent = second.parent
        while parent is not None and parent.id != first.id:
            parent = parent.parent
        if parent is None:
            return False

        # Try merging the two nodes, and see if there is a conflict.
        # In case of a conflict, we get the access sequences to the nodes causing the conflict
        conflicts = Apartness.merge(first, second, ob_tree)
        if not experiment:
            return conflicts != []

        for first_access, second_access in conflicts:
            # Incompatible!

            # Construct possible candidates that can prove apartness.
            transfer_sequence = ob_tree.get_transfer_sequence(first, second)
            suffix = transfer_sequence + first_access[len(first.access_sequence):]
            candidate = first.access_sequence + suffix
            candidates = []

            while candidate != second_access:
                candidates.append(candidate)
                suffix = transfer_sequence + suffix
                candidate = first.access_sequence + suffix

            for candidate in candidates:
                _ = ob_tree.experiment(candidate)

        return conflicts != []

    @staticmethod
    def merge(first, second, ob_tree):
        """
        Merge the second node into the first node.
        :param first: Node to merge into
        :param second: Node to merge from
        :param ob_tree: Observation tree
        :return: Whether there was a conflict during the merge
        """
        # Obtain the transfer sequence from first to second
        transfer_sequence = ob_tree.get_transfer_sequence(first, second)
        sequence = transfer_sequence.copy()
        conflicts = []
        while True:
            first_node = first
            second_node = Apartness.get_successors(second, sequence)
            if second_node is None:
                break
            witnesses = Apartness.get_distinguishing_sequences([first_node, second_node], ob_tree)
            for witness in witnesses:
                first_conflict = first_node.access_sequence + witness
                second_conflict = second_node.access_sequence + witness
                conflicts.append((first_conflict, second_conflict))
            sequence += transfer_sequence
        return conflicts

    @staticmethod
    def get_distinguishing_sequences(group, ob_tree):
        if ob_tree.automaton_type == "mealy":
            return Apartness._get_distinguishing_sequences_mealy(group, ob_tree.alphabet)
        else:
            return Apartness._get_distinguishing_sequences_moore(group, ob_tree.alphabet)

    @staticmethod
    def _get_distinguishing_sequences_mealy(group, alphabet):
        # Identifies all distinguishing input-output pairs in the provided alphabet of the n states
        groups = deque([([], group)])

        while groups:
            access_seq, group = groups.popleft()
            for input_val in alphabet:
                valid_group = [node for node in group if node.get_output(input_val) is not None]

                if len(valid_group) >= 2:
                    outputs = set([node.get_output(input_val) for node in valid_group])
                    if "unknown" in outputs:
                        outputs.remove("unknown")
                    if None in outputs:
                        outputs.remove(None)
                    if len(outputs) >= 2:
                        yield access_seq + [input_val]

                    groups.append((access_seq + [input_val], [node.get_successor(input_val) for node in valid_group]))

    @staticmethod
    def _get_distinguishing_sequences_moore(group, alphabet):
        # Identifies if two states can be distinguished by any input-output pair in the provided alphabet
        groups = deque([([], group)])
        while groups:
            access_seq, group = groups.popleft()
            valid_group = [node for node in group if node is not None and node.leads_to_known]
            if len(valid_group) >= 2:
                outputs = set([node.output for node in valid_group])
                if "unknown" in outputs:
                    outputs.remove("unknown")
                if None in outputs:
                    outputs.remove(None)
                if len(outputs) >= 2:
                    yield access_seq

                for input_val in alphabet:
                    groups.append((access_seq + [input_val], [node.get_successor(input_val) for node in valid_group]))

    @staticmethod
    def compute_witness_in_tree_and_hypothesis_states(ob_tree, ob_tree_state, hyp_state):
        """
        Determines if the observation tree and the hypothesis are distinguishable based on their state outputs
        """
        if ob_tree.automaton_type == 'mealy':
            return Apartness.compute_witness_in_tree_and_hypothesis_states_mealy(ob_tree, ob_tree_state, hyp_state)
        else:
            return Apartness.compute_witness_in_tree_and_hypothesis_states_moore(ob_tree, ob_tree_state, hyp_state)

    @staticmethod
    def compute_witness_in_tree_and_hypothesis_states_mealy(ob_tree, ob_tree_state, hyp_state):
        """
        Determines if the observation tree and the hypothesis are distinguishable based on their state outputs
        """
        pairs = deque([(ob_tree_state, hyp_state)])

        while pairs:
            tree_state, hyp_state = pairs.popleft()

            for input_val in ob_tree.alphabet:
                tree_output = tree_state.get_output(input_val)

                if tree_output is not None and input_val in hyp_state.output_fun:
                    hyp_output = hyp_state.output_fun[input_val]
                    if tree_output != hyp_output and tree_output not in ["unknown", None]:
                        tree_dest = tree_state.get_successor(input_val)
                        return ob_tree.get_transfer_sequence(ob_tree_state, tree_dest)

                    pairs.append((tree_state.get_successor(input_val), hyp_state.transitions[input_val]))

        return None

    @staticmethod
    def compute_witness_in_tree_and_hypothesis_states_moore(ob_tree, ob_tree_state, hyp_state):
        """
        Determines if the observation tree and the hypothesis are distinguishable based on their state outputs
        """
        pairs = deque([(ob_tree_state, hyp_state)])

        while pairs:
            tree_state, hyp_state = pairs.popleft()
            if (tree_state is not None) and (hyp_state is not None):
                tree_output = tree_state.output
                if ob_tree.automaton_type == 'dfa':
                    hyp_output = hyp_state.is_accepting
                else:
                    hyp_output = hyp_state.output

                if tree_output != hyp_output and tree_output not in ["unknown", None]:
                    return ob_tree.get_transfer_sequence(ob_tree_state, tree_state)

                for input_val in ob_tree.alphabet:
                    if input_val in hyp_state.transitions:
                        pairs.append((tree_state.get_successor(input_val), hyp_state.transitions[input_val]))

        return None

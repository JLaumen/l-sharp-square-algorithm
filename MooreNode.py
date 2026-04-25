class MooreNode:
    _id_counter = 0
    __slots__ = ['id', 'output', 'successors', 'parent', 'input_to_parent', 'access_sequence', 'leads_to_known']

    def __init__(self, parent=None):
        MooreNode._id_counter += 1
        self.id = MooreNode._id_counter
        self.output = None
        self.successors = {}
        self.parent = parent
        self.input_to_parent = None
        self.access_sequence = []
        self.leads_to_known = False

    def __hash__(self):
        return hash(self.id)

    def set_output(self, output):
        self.output = output
        if output is True or output is False:
            self.leads_to_known = True
            node = self
            while node.parent is not None and not node.parent.leads_to_known:
                node = node.parent
                node.leads_to_known = True

    def add_successor(self, input_val, output_val, successor_node):
        """ Adds a successor node to the current node based on input """
        self.successors[input_val] = successor_node
        self.successors[input_val].parent = self
        self.successors[input_val].input_to_parent = input_val
        self.successors[input_val].set_output(output_val)
        self.successors[input_val].access_sequence = self.access_sequence + [input_val]

    def get_successor(self, input_val):
        """ Returns the successor node for the given input """
        if input_val in self.successors:
            return self.successors[input_val]
        return None

    def extend_and_get(self, inp, output):
        """ Extend the node with a new successor and return the successor node """
        if inp in self.successors:
            return self.successors[inp]
        successor_node = MooreNode(parent=self)
        self.add_successor(inp, output, successor_node)
        successor_node.input_to_parent = inp
        return successor_node

    @property
    def id_counter(self):
        return self._id_counter

    def __str__(self):
        compact_counter_examples = True
        if compact_counter_examples and self.output is None and len(self.successors) == 1:
            # skip printing this node and print the child instead.
            successor = list(self.successors.values())[0]
            result = str(successor)
            return result
        else:
            inputs = []
            current_node = self
            while not current_node.parent is None:
                inputs.insert(0, current_node.input_to_parent)
                current_node = current_node.parent

            result = "node " + str(inputs) + " / " + str(self.output)
            for input_val, successor in self.successors.items():
                result += "\n" + str(input_val) + ":\n"
                result += "\t" + str(successor).replace("\n", "\n\t")
            return result

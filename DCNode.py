from DCValue import DCValue


class DCNode:
    _id_counter = 0

    def __init__(self, parent=None):
        DCNode._id_counter += 1
        self.id = DCNode._id_counter
        self.output = None
        self.successors = {}
        self.parent = parent
        self.input_to_parent = None
        self.access_sequence = []
        self.leads_to_known = False

    def __hash__(self):
        return hash(self.id)

    def set_leads_to_known(self):
        if self.leads_to_known:
            return

        self.leads_to_known = True

        if self.parent is not None:
            self.parent.set_leads_to_known()

    def set_output(self, output: DCValue):
        self.output = output

        if output.is_known():
            self.set_leads_to_known()

    def add_successor(self, input_val, output_val, successor_node):
        self.successors[input_val] = successor_node
        successor_node.parent = self
        successor_node.input_to_parent = input_val
        successor_node.set_output(output_val)
        successor_node.access_sequence = self.access_sequence + [input_val]

    def get_successor(self, input_val):
        return self.successors.get(input_val)

    def extend_and_get(self, inp, output):
        if inp in self.successors:
            return self.successors[inp]

        successor_node = DCNode(parent=self)
        self.add_successor(inp, output, successor_node)
        successor_node.input_to_parent = inp
        return successor_node

    def __str__(self):
        compact_counter_examples = True

        if (compact_counter_examples and self.output is None and len(self.successors) == 1):
            successor = next(iter(self.successors.values()))
            return str(successor)

        inputs = []
        current_node = self

        while current_node.parent is not None:
            inputs.insert(0, current_node.input_to_parent)
            current_node = current_node.parent

        result = "node " + str(inputs) + " / " + str(self.output)

        for input_val, successor in self.successors.items():
            result += "\n" + str(input_val) + ":\n"
            result += "\t" + str(successor).replace("\n", "\n\t")

        return result

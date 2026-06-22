class DiGraph:
    def __init__(self):
        self._nodes = {}
        self._succ = {}

    @property
    def nodes(self):
        return self._nodes

    def add_node(self, node, **attrs):
        if node not in self._nodes:
            self._nodes[node] = {}
            self._succ[node] = set()
        self._nodes[node].update(attrs)

    def add_edge(self, src, dst):
        self.add_node(src)
        self.add_node(dst)
        self._succ[src].add(dst)

    def has_edge(self, src, dst):
        return src in self._succ and dst in self._succ[src]

    def __contains__(self, node):
        return node in self._nodes


def topological_sort(graph):
    indegree = {node: 0 for node in graph._nodes}
    for src in graph._succ:
        for dst in graph._succ[src]:
            indegree[dst] += 1

    ready = [node for node, degree in indegree.items() if degree == 0]
    order = []

    while ready:
        node = ready.pop()
        order.append(node)
        for dst in graph._succ[node]:
            indegree[dst] -= 1
            if indegree[dst] == 0:
                ready.append(dst)

    if len(order) != len(indegree):
        raise ValueError("cycle detected in expression graph")

    return order

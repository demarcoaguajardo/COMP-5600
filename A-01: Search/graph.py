class Graph:

    # Constructor
    def __init__(self):
        self.graph = {}
        self.nodeCoords = {}

    # Adds an edge to the graph.
    def addEdge(self, node1, node2, w=1):
        # If node1 and/or node2 not in graph, add them.
        if node1 not in self.graph:
            self.graph[node1] = []
        if node2 not in self.graph:
            self.graph[node2] = []
        # Add edge between node1 and node2.
        self.graph[node1].append((node2, w))
        self.graph[node2].append((node1, w))

    # Adds a node to the graph.
    def addNode(self, node, x, y):
        self.nodeCoords[node] = (x, y)
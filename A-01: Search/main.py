# Import Libraries
import time
from graph import Graph
from searchAlgorithms import bfs, dfs, aStar, eucDistance, manDistance, chebDistance
from utils import readEdgeList, readNodeCoords

# Main Function
def main():
    # Prompt user for test case names
    testCase = input("Enter test case name (e.g. TestCase_01): ")
    print()

    # Gets the file from user input
    edgeListFile = f"{testCase}_EdgeList.txt"
    nodeCoordsFile = f"{testCase}_NodeID.csv"

    # Read edge list and node coords.
    edges = readEdgeList(edgeListFile)
    nodes = readNodeCoords(nodeCoordsFile)

    # Create graph, adding edges and nodes.
    graph = Graph()
    for node, x, y in nodes:
        graph.addNode(node, x, y)
    for n1, n2, w in edges:
        graph.addEdge(n1, n2, w)

    # Run the algorithms.
    startNode = nodes[0][0] # First node in NodeID file.
    goalNode = nodes[-1][0] # Last node in NodeID file.

    # Get node positions.
    nodePositions = {node: (x, y) for node, x, y in nodes}

    # Run and Time BFS algorithm.
    startTime = time.time()
    bfsVisitedNodes = bfs(graph, startNode, goalNode)
    bfsTime = time.time() - startTime
    # Run and Time DFS algorithm.
    startTime = time.time()
    dfsVisitedNodes = dfs(graph, startNode, goalNode)
    dfsTime = time.time() - startTime
    # Run and Time A* algorithm with Euclidean distance heuristic.
    startTime = time.time()
    aStarVisitedNodesEuc = aStar(graph, startNode, goalNode, nodePositions, eucDistance)
    aStarEucTime = time.time() - startTime
    # Run and Time A* algorithm with Manhattan distance heuristic.
    startTime = time.time()
    aStarVisitedNodesMan = aStar(graph, startNode, goalNode, nodePositions, manDistance)
    aStarManTime = time.time() - startTime
    # Run and Time A* algorithm with Chebyshev distance heuristic.
    startTime = time.time()
    aStarVisitedNodesCheb = aStar(graph, startNode, goalNode, nodePositions, chebDistance)
    aStarChebTime = time.time() - startTime

    # Print the list of states visited by the algorithms.
    print("BFS:", bfsVisitedNodes)
    print("DFS:", dfsVisitedNodes)
    print("A* with Euclidean Distance:", aStarVisitedNodesEuc)
    print("A* with Manhattan Distance:", aStarVisitedNodesMan)
    print("A* with Chebyshev Distance:", aStarVisitedNodesCheb)

    # Print the time taken by each algorithm
    print("\nTime taken by each algorithm:")
    print(f"BFS: {bfsTime:.6f} seconds")
    print(f"DFS: {dfsTime:.6f} seconds")
    print(f"A* with Euclidean Distance: {aStarEucTime:.6f} seconds")
    print(f"A* with Manhattan Distance: {aStarManTime:.6f} seconds")
    print(f"A* with Chebyshev Distance: {aStarChebTime:.6f} seconds")

if __name__ == "__main__":
    main()
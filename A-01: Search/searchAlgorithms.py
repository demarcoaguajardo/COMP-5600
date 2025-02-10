# Importing Libraries
from collections import deque
import heapq
import math

# BFS Algorithm
def bfs(graph, start, goal):
    # Start queue with start node.
    queue = deque([start])
    # List that holds visited nodes.
    visited = []

    # While queue is not empty...
    while queue:
        # Dequeue node
        node = queue.popleft()
        # Skip if node was already visited.
        if node in visited:
            continue
        # Add node to visited list.
        visited.append(node)

        # If the goal node is reached, return the visited nodes.
        if node == goal:
            return visited

        # Gets neighbors of node
        for neighbor, cost in graph.graph.get(node, []):
            # Add neighbor to queue if not visited.
            if neighbor not in visited:
                queue.append(neighbor)
    # Return visited nodes if goal not found.
    return visited

# DFS Algorithm
def dfs(graph, start, goal):
    # Start stack with start node.
    stack = [start]
    # List that holds visited nodes.
    visited = []

    # While stack is not empty...
    while stack:
        # Pop node from stack
        node = stack.pop()
        # Skip if node was already visited.
        if node in visited:
            continue
        # Add node to visited list.
        visited.append(node)

        # If the goal node is reached, return the visited nodes.
        if node == goal:
            return visited

        # Gets neighbors of node
        for neighbor, cost in graph.graph.get(node, []):
            # Add neighbor to stack if not visited.
            if neighbor not in visited:
                stack.append(neighbor)
    # Return visited nodes if goal not found.
    return visited

# A* Search Algorithm and Heuristics

# Euclidean Distance Heuristic
def eucDistance(node, goal, nodePositions):
    x1, y1 = nodePositions[node]
    x2, y2 = nodePositions[goal]
    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    # ---------- COMMENT THIS WHEN DONE TESTING ----------
    # print(f"Euclidean Distance from {node} to {goal}: {distance}")

    return distance

# Manhattan Distance Heuristic
def manDistance(node, goal, nodePositions):
    x1, y1 = nodePositions[node]
    x2, y2 = nodePositions[goal]
    distance = abs(x1 - x2) + abs(y1 - y2)

    # ---------- COMMENT THIS WHEN DONE TESTING ----------
    # print(f"Manhattan Distance from {node} to {goal}: {distance}")

    return distance

# Chebyshev Distance Heuristic
def chebDistance(node, goal, nodePositions):
    x1, y1 = nodePositions[node]
    x2, y2 = nodePositions[goal]
    distance = max(abs(x1 - x2), abs(y1 - y2))

    # ---------- COMMENT THIS WHEN DONE TESTING ----------
    # print(f"Chebyshev Distance from {node} to {goal}: {distance}")

    return distance

# A* Search
def aStar(graph, start, goal, nodePositions, heuristic):
    # Priority queue for starting node.
    frontier = []
    heapq.heappush(frontier, (0, start))

    # Cost to reach each node
    costSoFar = {start: 0}

    # Path taken to reach each node
    cameFrom = {start: None}

    # While frontier is not empty...
    while frontier: 
        # Get node with lowest cost + heuristic
        currentCost, currentNode = heapq.heappop(frontier)

        # ---------- COMMENT THIS WHEN DONE TESTING ----------
        # print(f"Current Node: {currentNode}, Cost: {currentCost}")

        # If goal is reached, return path
        if currentNode == goal:
            path = []
            while currentNode:
                path.append(currentNode)
                currentNode = cameFrom[currentNode]
            path.reverse()
            return path

        # Go through neighbors
        for neighbor, cost in graph.graph.get(currentNode, []):

            # Calculate new cost
            newCost = costSoFar[currentNode] + cost

            if neighbor not in costSoFar or newCost < costSoFar[neighbor]:
                costSoFar[neighbor] = newCost
                priority = newCost + heuristic(neighbor, goal, nodePositions)
                heapq.heappush(frontier, (priority, neighbor))
                cameFrom[neighbor] = currentNode

                # ---------- COMMENT THIS WHEN DONE TESTING ----------
                # print(f"Next Node: {neighbor}, New Cost: {newCost}, Priority: {priority}")

    return list(cameFrom.keys())
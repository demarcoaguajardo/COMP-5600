# Importing Libraries
import csv

# Read .txt file and return list of edges.
def readEdgeList(filename):
    edges = []
    with open(filename, 'r') as file:
        for line in file:
            n1, n2, w = line.strip().split(',')
            edges.append((n1, n2, float(w)))
    return edges

# Read .csv file and return node coordinates.
def readNodeCoords(filename):
    nodes = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            n1, x, y = row
            nodes.append((n1, float(x), float(y)))
    return nodes
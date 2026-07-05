def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])

def union(parent, rank, x, y):
    rootX = find(parent, x)
    rootY = find(parent, y)

    if rank[rootX] > rank[rootY]:
        parent[rootY] = rootX
    elif rank[rootX] < rank[rootY]:
        parent[rootX] = rootY
    else:
        parent[rootY] = rootX
        rank[rootX] += 1

def kruskal(graph, vertices):
    result = []
    i = 0
    e = 0

    graph = sorted(graph, key=lambda item: item[2])

    parent = []
    rank = []

    for node in range(vertices):
        parent.append(node)
        rank.append(0)

    while e < vertices - 1:
        u, v, w = graph[i]
        i += 1
        x = find(parent, u)
        y = find(parent, v)

        if x != y:
            e += 1
            result.append([u, v, w])
            union(parent, rank, x, y)

    return result

# Örnek graf
graph = [
    (0, 1, 10),
    (0, 2, 6),
    (0, 3, 5),
    (1, 3, 15),
    (2, 3, 4)
]

vertices = 4

minimum_spanning_tree = kruskal(graph, vertices)

print("Minimum spanning tree:")
for u, v, weight in minimum_spanning_tree:
    print(f"{u} - {v} : {weight}")
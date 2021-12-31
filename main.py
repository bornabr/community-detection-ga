from problem import Problem

f = open('sample dataset.txt', 'r')
lines = f.readlines()

n = int(lines[0])
lines = lines[1:]

adj = [[] for _ in range(n)]

for edge in lines:
	edge = edge.split()
	
	adj[int(edge[0]) - 1].append(int(edge[1]) - 1)
	adj[int(edge[1]) - 1].append(int(edge[0]) -1)

problem = Problem(n, adj, 6)

problem.initial_population()

print(problem.population)

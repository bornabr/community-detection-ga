import random
import copy
from itertools import combinations

ALPHABET = "abcdefghijklmnopqrstuvwxyz"

class Problem:
	def __init__(self, n, adj, default_n_community):
		self.n = n
		self.adj = adj
		self.m = int(sum([len(x) for x in self.adj])/2)
		self.default_n_community = default_n_community
	
	def initial_population(self):
		self.population = []
		for _ in range(self.population_size):
			vertices = list(range(self.n))
			random.shuffle(vertices)
			individual_map = [None for _ in range(self.n)]
			
			counter = 0
			
			while len(vertices) != 0:
				if len(vertices) < int(self.n / self.default_n_community):
					selected = vertices[:]
					for i in selected:
						individual_map[i] = counter
					counter += 1
					break
				else:
					selected = random.choices(vertices, k=int(self.n / self.default_n_community))
					vertices = [e for e in vertices if e not in selected]
					for i in selected:
						individual_map[i] = counter
					counter += 1
			self.population.append([individual_map,None])
			
		self.population = sorted(self.population, key=lambda agent: self.fitness(agent), reverse=False)
	
	def fitness(self, individual):
		Q = 0
		for i in range(self.n):
			for j in range(self.n):
				Q += (int(j in self.adj[i]) - ((len(self.adj[i]) * len(self.adj[j])) / (2 * self.m))) * int(individual[0][i] == individual[0][j])
		Q /= (2 * self.m)
		
		individual[1] = Q
		return Q
		
	def selection(self):
		random.shuffle(self.population)
		self.parents = copy.deepcopy(self.population[:self.tournament_size])
		self.parents = sorted(self.parents, key=lambda agent: self.fitness(
			agent), reverse=True)[:self.parents_size]
	
	def breed(self, parent1, parent2):
		child1 = [None, None]
		child2 = [None, None]
		
		pivot = random.choice(range(1, self.n-1))
		
		child1[0] = copy.deepcopy(parent1[0][:pivot] + parent2[0][pivot:])
		child2[0] = copy.deepcopy(parent2[0][:pivot] + parent1[0][pivot:])
		self.fitness(child1)
		self.fitness(child2)
		return child1, child2

	def breed_offsprings(self):
		self.children = []
		
		for _ in range(self.breed_rate):
			random.shuffle(self.parents)
			for i in range(int(len(self.parents)/2)):
				child1, child2 = self.breed(self.parents[i], self.parents[len(self.parents)-i-1])
				self.children.append(child1)
				self.children.append(child2)
		
	def mutate(self, individual):
		individual_copy = [copy.deepcopy(individual[0][:]), None]
		if random.random() < self.mutation_rate:
			gene = random.choice(range(self.n))
			cluster = random.choice(list(set(individual[0])))
			while cluster == individual[0][gene]:
				cluster = random.choice(list(set(individual[0])))
			individual_copy[0][gene] = cluster
		self.fitness(individual_copy)
		return individual_copy
		
	def mutate_offsprings(self):
		self.mutated_children = []
		for individual in self.children:
			self.mutated_children.append(self.mutate(individual))
		return self.mutated_children
	
	def replacement(self):
		self.mutated_children = sorted(
			self.mutated_children, key=lambda agent: agent[1], reverse=True)
		self.population = sorted(
			self.population, key=lambda agent: agent[1], reverse=True)

		self.population = self.mutated_children[:-
                                          self.elite_size] + self.population[:self.elite_size]
		self.population = sorted(
                    self.population, key=lambda agent: self.fitness(agent), reverse=True)
	
	def evaluate(self):
		pop_fitness = [agent[1] for agent in self.population]
	
		return sum(pop_fitness), min(pop_fitness)

	# def GA(self, population_size, tournament_size, parents_size, mutation_rate, elite_size, n_generations):
	def GA(self, population_size, tournament_size, parents_size, mutation_rate, elite_size, n_generations):
		self.population_size = population_size
		self.tournament_size = tournament_size
		self.parents_size = parents_size
		self.breed_rate = int(self.population_size/self.parents_size)
		self.mutation_rate = mutation_rate
		self.elite_size = elite_size
		self.n_generations = n_generations

		self.initial_population()
		
		for epoch in range(self.n_generations):
			self.selection()
			self.breed_offsprings()
			self.mutate_offsprings()
			
			self.replacement()
			eval_ = self.evaluate()
			
			print("Epoch", epoch, ":\tPopulation total fitness:",
                            eval_[0], "\tBest fitness:", eval_[1])
		
		print(self.population[0][1], self.population[0][0])

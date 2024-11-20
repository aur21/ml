from SoftRobot import *

import random
import argparse
import copy
import json
import itertools
import numpy as np


# Constants defining tissue types
MUSCLE_A = -1
MUSCLE_B = 1
BONE = 0
LIGAMENT = 2
NONE = 3
TYPES = [MUSCLE_A, MUSCLE_B, BONE, LIGAMENT, NONE]

def random_genotype():
    """
    Creates and returns a random genotype (your choice of randomness).
    The genotype *must* be a 64-element list of BONE, LIGAMENT, MUSCLE_A, MUSCLE_B, and NONE
    """
    genotype = []
    for i in range(64):
        genotype.append(random.choice(TYPES))
    return genotype

def mutation(genotype):
    """
    Creates and returns a *new* genotype by mutating the argument genotype.
    The argument genotype must remain unchanged.
    """
    mutated_genotype = genotype.copy()
    for i in range(20):
        mutated_genotype[random.randint(0, 63)] = random.choice(TYPES)
    return mutated_genotype

def crossover(genotype1, genotype2):
    """
    Creates and returns TWO *new* genotypes by applying crossover to the argument genotypes.
    The argument genotypes must remain unchanged.
    """
    crossover_genotype = genotype1.copy()
    for i in range(64):
        if crossover_genotype[i] != genotype2[i]:
            rand = random.randint(0, 4) 
            if rand >= 0 and rand <= 2:
                crossover_genotype[i] = genotype2[i]
            elif rand == 3:
                crossover_genotype[i] = random.choice(TYPES)
    return crossover_genotype

def selection_and_offspring(population, fitness):
    """
    Arguments
        population: list of genotypes
        fitness: list of fitness values (higher is better) corresponding to each
                 of the genotypes in population

    Returns
       A new population (list of genotypes) that is the SAME LENGTH as the 
       argument population, created by selecting genotypes from the argument
       population by their fitness and applying genetic operators (mutation,
       crossover, elitism) 
    """ 
    apex_fitnesses = []
    apex_genotypes = []
    apex_fitnesses.append(0)
    new_population = population.copy()
    fitness_np = np.array(fitness)
    avg = np.average(fitness_np)
    var = np.var(fitness_np)
    for i in range(3):
        apex_fitnesses.append(np.max(fitness_np))
        fitness_np = np.delete(fitness_np, np.where(fitness_np == np.max(fitness_np)))

    max_idx = 0

    for i in range(len(new_population)):
        if fitness[i] >= np.min(apex_fitnesses):
            apex_genotypes.append(i)
        elif fitness[i] < np.min(apex_fitnesses) and fitness[i] > avg:
            if len(apex_genotypes) > 0:
                new_population[i] = crossover(new_population[i], new_population[random.choice(apex_genotypes)])
            else:
                new_population[i] = mutation(new_population[i])
        elif fitness[i] <= avg and fitness[i] >= (avg - (avg - var)):
            new_population[i] = mutation(new_population[i])
        else: 
            new_population[i] = random_genotype()
    return new_population

def evolve(num_generations, pop_size, fps, timesteps):
    """
    Runs the evolutionary algorithm. You do NOT need to modify this function, 
    but you should understand what it does.
    """
    for g in range(num_generations):

        if g == 0:
            population = [random_genotype() for p in range(pop_size)]
        else:
            population = selection_and_offspring(population, fitness)

        fitness = []
        for i,p in enumerate(population):
            simulation = SoftRobot(p, f=fps)
            render = True if g%10 == 0 else False
            fit = simulation.Run(timesteps, render)
            fitness.append(fit)
            print(f"Individual {i} has fitness {fit}")

        print(f"\tGeneration {g+1}: Highest Fitness {max(fitness)}")

        
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run soft robot evolution")
    parser.add_argument("num_generations", type=int, help="Number of generations (int)")
    parser.add_argument("-p", "--pop_size", type=int, default=10, help="Population size (int)")
    parser.add_argument("-t", "--timesteps", type=int, default=300, help="Timesteps per simulation (int)")
    parser.add_argument("-f", "--fps", type=int, default=60, help="Frames per second for simulation (int)")
    args = parser.parse_args()

    # Run evolution
    evolve(args.num_generations, args.pop_size, args.fps, args.timesteps)


if __name__ == "__main__":
    main()

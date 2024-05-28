from ga import *

class SumGA(GeneticAlgorithm):
    """
    An example of using the GeneticAlgorithm class to solve a particular
    problem, in this case finding strings with the maximum number of 1's.
    """
    def fitness(self, chromosome):
        """
        Fitness is the sum of the bits.
        """
        return sum(chromosome)

    def isDone(self):
        """
        Stop when the fitness of the the best member of the current
        population is equal to the maximum fitness.
        """
        return self.fitness(self.bestEver) == self.length


def main():
    # use this main program to incrementally test the GeneticAlgorithm
    # class as you implement it
    """
    ga = SumGA(10,20)
    ga.initializePopulation()
    ga.evaluatePopulation()
    print("testing", ga.popSize)
    for i in range(ga.popSize):
        print("population member:", ga.population[i],"    Fitness:", ga.scores[i]) 
    print("self.totalFitness avg: ", ga.totalFitness/ga.popSize)
    #print("best fitness:", ga.bestEverScore)
    #for j in range(len(ga.bestList)):
    #    print("best candidate:", ga.bestList[j])
    print("----------------------------newPop----------------------------")
    ga.oneGeneration()
    totalFit = 0
    ga.evaluatePopulation()
    print("new fitness:", ga.totalFitness)
    print("newGen Fitness avg: ", ga.totalFitness/ga.popSize)
    print("----------------------------SELECTION----------------------------")
    hello = []
    oof = 0
    for j in range(ga.popSize):
        oof += ga.fitness(ga.selection())
    print("ogfit: ", totalFit, "\nfitness new: ", oof)

    print("----------------------------CROSSOVER----------------------------")
    print("parent1: ", ga.population[0])
    print("parent2: ", ga.population[1])
    print("children: ", ga.crossover(ga.population[0],ga.population[1]))
    print("----------------------------MUTATION----------------------------")
    print("Child pre-mutation: ", ga.population[2])
    ga.mutation(ga.population[2])
    print("Child post-mutation: ", ga.population[2])
"""
    # Chromosomes of length 20, population of size 50
    ga = SumGA(20, 50)
    # Evolve for 100 generations
    # High prob of crossover, low prob of mutation
    bestFound = ga.evolve(100, 0.6, 0.01)
    print(bestFound)
    ga.plotStats("Sum GA")
    
if __name__ == '__main__':
    main()

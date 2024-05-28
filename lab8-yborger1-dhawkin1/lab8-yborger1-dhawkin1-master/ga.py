from random import random, randrange
import pylab

class GeneticAlgorithm(object):
    """
    A genetic algorithm is a model of biological evolution.  It
    maintains a population of chromosomes.  Each chromosome is
    represented as a list of 0's and 1's.  A fitness function must be
    defined to score each chromosome.  Initially, a random population
    is created. Then a series of generations are executed.  Each
    generation, parents are selected from the population based on
    their fitness.  More highly fit chromosomes are more likely to be
    selected to create children.  With some probability crossover will
    be done to model sexual reproduction.  With some very small
    probability mutations will occur.  A generation is complete once
    all of the original parents have been replaced by children.  This
    process continues until the maximum generation is reached or when
    the isDone method returns True.
    """
    def __init__(self, length, popSize, verbose=False):
        self.verbose = verbose      # Set to True to see more info displayed
        self.length = length        # Length of the chromosome
        self.popSize = popSize      # Size of the population
        self.bestEver = None        # Best member ever in this evolution
        self.bestEverScore = 0      # Fitness of best member ever
        self.population = None      # Population is a list of chromosomes
        self.scores = None          # List of fitness of all members of pop
        self.totalFitness = None    # Total fitness in entire population
        self.pCrossover = 0.6       # Probability of crossover
        self.pMutation = 0.1        # Probability of mutation (per bit)
        self.bestList = []          # List of best fitness per generation
        self.avgList = []           # List of avg fitness per generation
        print("Executing genetic algorithm")
        print("Chromosome length:", self.length)
        print("Population size:", self.popSize)

    def initializePopulation(self):
        """
        Initialize each chromosome in the population with a random
        series of 1's and 0's. 

        Returns: None
        Result: Initializes self.population
        """
        self.population = []
        for i in range(self.popSize):
            indiv = []
            for j in range(self.length):
                indiv.append(randrange(0,2))    
            self.population.append(indiv)




    
    def evaluatePopulation(self):
        """
        Computes the fitness of every chromosome in population.  Saves the
        fitness values to the list self.scores.  Checks whether the
        best fitness in the current population is better than
        self.bestEverScore. If so, prints a message that a new best
        was found and its score, updates this variable and saves the
        chromosome to self.bestEver.  Computes the total fitness of
        the population and saves it in self.totalFitness. Appends the
        current bestEverScore to the self.bestList, and the current
        average score of the population to the self.avgList.

        Returns: None
        """
        #self.scores = []
        magical = []
        for chrome in self.population:
            magical.append(self.fitness(chrome))
        self.scores = magical 
        for i in range(self.popSize):
            if self.scores[i] > self.bestEverScore:
                self.bestEverScore = self.scores[i]
                self.bestEver = self.population[i]
                print("new best found! score:", self.bestEverScore)
        self.totalFitness = sum(self.scores)
        self.bestList.append(self.bestEverScore)
        self.avgList.append(sum(self.scores)/self.popSize)
           


    def selection(self):
        """
        Each chromosome's chance of being selected for reproduction is
        based on its fitness.  The higher the fitness the more likely
        it will be selected.  Uses the roulette wheel strategy on 
        self.scores. 

        Returns: A COPY of the selected chromosome. You can make a copy
        of a python list by taking a full slice of it.  For example
        x = [1, 2, 3, 4]
        y = x[:] # y is a copy of x
        """
        spin = random() * self.totalFitness
        partialFitness = 0
        for i in range(self.popSize):
            partialFitness += self.scores[i]
            if partialFitness >= spin:
                break
        indiv = self.population[i][:]
        return indiv

    def crossover(self, parent1, parent2):
        """
        With probability self.pCrossover, recombine the genetic
        material of the given parents at a random location between
        1 and the length-1 of the chromosomes. If no crossover is
        performed, then return the original parents.

        When self.verbose is True, and crossover is done, prints
        the crossover point, and the two children.  Otherwise prints
        that no crossover was done.

        Returns: Two children
        """

        if(random() < self.pCrossover):
            spinjr = randrange(1,self.length-1)
            child1 = parent1[:spinjr] + parent2[spinjr:]
            child2 = parent2[:spinjr] + parent1[spinjr:]
            if(self.verbose):
                print("Crossover point: ", spinjr)
                print("Children: ", child1, child2)

        else:
            if(self.verbose):
                print("no crossover done")
            return parent1, parent2
        return child1, child2



    def mutation(self, chromosome):
        """
        With probability self.pMutation tested at each position in the
        chromosome, flip value.

        When self.verbose is True, if mutation is done prints the
        position of the string being mutated. When no mutations are
        done prints this at the end. 

        Returns: None
        """

        mutation = False
        for gene in range(len(chromosome)):
            if(random() < self.pMutation):
                mutation = True
                if(chromosome[gene] == 0):
                    chromosome[gene] = 1
                else:
                    chromosome[gene] = 0
            if(self.verbose):
                print("Position: ", gene)
        if not mutation and self.verbose:
            print("No mutation")


    def oneGeneration(self):
        """
        Execute one generation of the evolution. Each generation,
        repeatedly select two parents, call crossover to generate two
        children.  Call mutate on each child.  Finally add both
        children to the new population.  Continue until the new
        population is full. Replaces self.pop with a new population.

        When self.verbose is True, prints the parents that were
        selected and their children after crossover and mutation
        have been completed. 

        Returns: None
        """

        newPopulation = []
        while(len(newPopulation) < self.popSize):
            parent1 = self.selection()
            parent2 = self.selection()
            child1, child2 = self.crossover(parent1, parent2)
            self.mutation(child1)
            self.mutation(child2)
            newPopulation.append(child1)
            newPopulation.append(child2)
        if(len(newPopulation) > self.popSize):
            newPopulation = newPopulation[:self.popSize]
        self.population = newPopulation

    def evolve(self, maxGen, pCrossover=0.7, pMutation=0.001):
        """
        Stores the probabilites in appropriate class variables.
        
        Runs a series of generations until maxGen is reached or
        self.isDone() returns True.

        Returns the best chromosome ever found over the course of
        the evolution, which is stored in self.bestEver.
        """
        self.pCrossover = pCrossover
        self.pMutation = pMutation
        self.initializePopulation()
        self.evaluatePopulation()
        for gen in range(maxGen):
            self.oneGeneration()
            self.evaluatePopulation()
            if self.isDone():
                break
        return self.bestEver




    def plotStats(self, title=""):
        """
        Plots a summary of the GA's progress over the generations.
        Adds the given title to the plot. 
        """
        gens = range(len(self.bestList))
        pylab.plot(gens, self.bestList, label="Best")
        pylab.plot(gens, self.avgList, label="Average")
        pylab.legend(loc="upper left")
        pylab.xlabel("Generations")
        pylab.ylabel("Fitness")
        if len(title) != 0:
            pylab.title(title)
        pylab.savefig("FitnessByGeneration.png")
        pylab.show()

    def fitness(self, chromosome):
        """
        The fitness function will change for each problem.  Therefore
        it is not defined here.  To use this class to solve a
        particular problem, inherit from this class and define this
        method.
        """
        # Do not implement, this will be overridden
        pass
    
    def isDone(self):
        """
        The stopping critera will change for each problem.  Therefore
        it is not defined here.  To use this class to solve a
        particular problem, inherit from this class and define this
        method.
        """
        # Do not implement, this will be overridden
        pass


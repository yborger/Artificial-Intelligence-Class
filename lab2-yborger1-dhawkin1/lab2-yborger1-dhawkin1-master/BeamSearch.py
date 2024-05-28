########################################
# CS63: Artificial Intelligence, Lab 2
# Fall 2022, Swarthmore College
########################################

from math import exp
from numpy.random import choice, multinomial 

def stochastic_beam_search(problem, pop_size, steps, init_temp,
                           temp_decay, max_neighbors):
    """Implementes the stochastic beam search local search algorithm.
    Inputs:
        - problem: A TSP instance.
        - pop_size: Number of candidates tracked.
        - steps: The number of moves to make in a given run.
        - init_temp: Initial temperature. Note that temperature has a
                slightly different interpretation here than in simulated
                annealing.
        - temp_decay: Multiplicative factor by which temperature is reduced
                on each step. Temperature parameters should be chosen such
                that e^(-cost / temp) never reaches 0.
        - max_neighbors: Number of neighbors generated each round for each
                candidate in the population.
    Returns: best_candidate, best_cost
        The best candidate identified by the search and its cost.

    NOTE: In this case, you should always call random_neighbor(), rather
          than best_neighbor().
    """
    best_state = []
    best_cost = float("inf")
    pop = []
    best_neigh_cost = float("inf")
    best_neigh_state = []
    ran_neigh_cost = float("inf")
    ran_neigh_state = []
    arr_neigh = []
    arr_neigh_cost = []
    for k in range(pop_size):
        pop.append(problem.random_candidate()) 
    temp = init_temp
    for i in range(steps): #our steps
        for poprun in range(pop_size): #go through each possible "pop" location    
            for j in range(max_neighbors): #get a bunch of random neighbors of each pop location
                ran_neigh_state, ran_neigh_cost  = problem.random_neighbor(pop[j]) #j
                arr_neigh.append(ran_neigh_state)
                arr_neigh_cost.append(ran_neigh_cost)
                if ran_neigh_cost < best_neigh_cost:
                    best_neigh_cost, best_neigh_state = ran_neigh_cost, ran_neigh_state
            if(best_neigh_cost < best_cost): #is this particular beam better than the best
                best_state = best_neigh_state
                best_cost = best_neigh_cost
                print("new best cost:", best_cost)
        pop = probability_helper(temp, pop_size, arr_neigh, arr_neigh_cost)
        temp *= temp_decay
    
    return best_state, best_cost





def probability_helper (temp, pop_size, unknown, cost):
        """Probability_helper explained:
                temp: the temperature passed in changes with each step
                pop_size: the pop_size is how big the beam is, and we need those probabilities
                unknown: a list of all of the random neighbors we collect to get probabilities for them
                cost: a list of the costs of the random neighbors so we do not need to re-calculate the cost each time
        """
        probs = []
        norm = []
        sum = 0.0
        hold = 0.0 
        choices = []
        newNeigh = []
        indices = list(range(len(unknown)))
        for item in cost: #changed to cost so that i is the cost parallel with the unknown we want
            hold = exp(-item/temp)
            probs.append(hold)
            sum += hold
        for item in probs:
            if sum == 0:
                norm.append(0)
            else:
                norm.append(item/sum)
        choices = choice(indices, pop_size, p=norm) #unknown is not 1D
        for i in range(len(choices)):
            newNeigh.append(unknown[choices[i]])
        return newNeigh

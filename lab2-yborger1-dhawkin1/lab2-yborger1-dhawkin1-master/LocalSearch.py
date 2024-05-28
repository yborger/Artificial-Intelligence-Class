#! /usr/bin/env python3
########################################
# CS63: Artificial Intelligence, Lab 2
# Fall 2022, Swarthmore College
########################################
# NOTE: you should not need to modify this file.
########################################

import json
from argparse import ArgumentParser

from TSP import TSP
from HillClimbing import hill_climbing
from SimulatedAnnealing import simulated_annealing
from BeamSearch import stochastic_beam_search

def parse_input():
    parser = ArgumentParser()
    parser.add_argument("search", choices=["HC","SA","BS"],
                        help="Local search algorithm to use: HC, SA, or BS")
    parser.add_argument("coordinates", type=str,\
                        help="JSON file with city coordinates.")
    parser.add_argument("-config", type=str, default="default_config.json",\
                        help="JSON file with search parameters.")
    parser.add_argument("-plot", type=str, default="map.pdf",\
                        help="Filename for map output.")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    args.search_args = config[args.search]
    return args

def main():
    args = parse_input()
    problem = TSP(args.coordinates)
    if args.search == "HC":
        search_alg = hill_climbing
    if args.search == "SA":
        search_alg = simulated_annealing
    if args.search == "BS":
        search_alg = stochastic_beam_search
    solution, cost = search_alg(problem, **args.search_args)
    print("route:")
    print(solution)
    print("cost:", cost)
    problem.plot(solution, args.plot)

if __name__ == "__main__":
    main()

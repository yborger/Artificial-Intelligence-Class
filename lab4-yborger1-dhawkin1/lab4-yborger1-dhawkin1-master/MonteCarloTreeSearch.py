########################################
# CS63: Artificial Intelligence, Lab 4
# Fall 2022, Swarthmore College
########################################

#NOTE: You will probably want to use these imports. Feel free to add more.
from curses import KEY_B2
from lib2to3.pgen2.token import ISTERMINAL
from math import log, sqrt
from pickle import NONE
import random


"""
    Hi! We had to use a late day so we could both make sure to have properly studied for the exam, 
    and we put a lot of effort into our code. Unfortauntely it does not seem to work (getting stuck
    in selection we believe), but we genuinely do not know why it is getting stuck. From our debugging,
    we found that it tries to access a "None" child, but we don't know how it got that child or how it 
    stayed in the loop with that case. We tried many different techniques for hours on end but we 
    unfortunately could not figure out how to fix this error. We would be super happy to hear any 
    ideas you have, or how to fix it if you figure it out super quickly. If there is anything we can
    do as like a side assignment or an additional thing to this program to make up for the fact that
    it was not fully functional at the end, please let us know!
    Have a nice fall break!

    From,
    -Yael, Delaney
"""


class Node(object):
    """Node used in MCTS"""
    def __init__(self, state):
        self.state = state
        self.children = {} # maps moves to Nodes
        self.visits = 0    # number of times node was in select/expand path
        self.wins = 0      # number of wins for player +1
        self.losses = 0    # number of losses for player +1
        self.value = 0     # value (from player +1's perspective)
        self.untried_moves = list(state.availableMoves) # moves to try

    def updateValue(self, outcome):
        """
        Increments self.visits.
        Updates self.wins or self.losses based on the outcome, and then
        updates self.value.

        This function will be called during the backpropagation phase
        on each node along the path traversed in the selection and
        expansion phases.

        outcome: Who won the game.
                 +1 for a 1st player win
                 -1 for a 2nd player win
                  0 for a draw
        """
        self.visits += 1
        if outcome == 1:
            self.wins += 1
        elif outcome == -1:
            self.losses += 1
        self.value = 1 +(self.wins-self.losses)/self.visits
        return


    def UCBWeight(self, UCB_const, parent_visits, parent_turn):
        """
        Weight from the UCB formula used by parent to select a child.

        This function calculates the weight for JUST THIS NODE. The
        selection phase, implemented by the MCTSPlayer, is responsible
        for looping through the parent Node's children and calling
        UCBWeight on each.
  File "/home/dhawkin1/cs63/lab4-yborger1-dhawo be
           converted to -1's perspective.
        returns the UCB weight calculated
        """
        value = self.value #just to be sure we have it :)
        if parent_turn == -1:
            value = 2 - value
        UCB_weight = value + UCB_const * sqrt(log(parent_visits)/self.visits)

        return UCB_weight


class MCTSPlayer(object):
    """Selects moves using Monte Carlo tree search."""
    def __init__(self, num_rollouts=1000, UCB_const=1.0):
        self.name = "MCTS"
        self.num_rollouts = int(num_rollouts)
        self.UCB_const = UCB_const
        self.nodes = {} # dictionary that maps states to their nodes

    def getMove(self, game_state):
        """Returns best move from the game_state after applying MCTS"""
        #TODO: find existing node in tree or create a node for game_state
        #      and add it to the tree
        #TODO: call MCTS to perform rollouts
        #TODO: return the best move from the current player's perspective
        key = str(game_state)
        if key in self.nodes.keys():
            curr_node = self.nodes.get(key)
        else:
            curr_node = Node(game_state)
            hold = {key: curr_node}
            self.nodes.update(hold)
        self.MCTS(curr_node)
        best_val = -float("inf") 
        best_move = None
        for move, child_node in curr_node.children.items():
            if curr_node.state.turn == 1:
                value = child_node.value
            else:
                value = 2 - child_node.value
            if value > best_val:
                best_val = value
                best_move = child_node
        return best_move





    def status(self, node):
        """
        This method is used solely for debugging purposes. Given a
        node in the MCTS tree, reports on the node's data (wins, losses,
        visits, values), as well as the data of all of its immediate
        children. Helps to verify that MCTS is working properly.
        Returns: None
        """

        print("node wins:", node.wins, "   losses:", node.losses, "   visits:",
            node.visits, "   value:", node.value, "   move:", self.getMove(node.state))

    def MCTS(self, current_node):
        """
        Plays out random games from the current node to a terminal state.
        Each rollout consists of four phases:
        1. Selection: Nodes are selected based on the max UCB weight.
                      Ends when a node is reached where not all children
                      have been expanded.
        2. Expansion: A new node is created for a random unexpanded child.
        3. Simulation: Uniform random moves are played until end of game.
        4. Backpropagation: Values and visits are updated for each node
                     on the path traversed during selection and expansion.
        Returns: None
        """

        for i in range(self.num_rollouts):
            path = self.selection(current_node)
            select = path[-1]
            #this is the notation for a list, this is a list of nodes
            if select.state.isTerminal:
                outcome = select.state.winner
            else:
                next_node = self.expansion(select)
                path.append(next_node)
                outcome = self.simulation(next_node.state)
            self.backpropagation(path,outcome)
        #self.status(current_node)


    def selection(self, curr_node):
        """
            choose a root R (curr_node) and keep choosing the BEST children until
            leaf L (end of path, not necessarily terminal) is reached
            returns: a path
        """
        print("selection")
        bestPath = []
        bestPath.append(curr_node)
        bestChild = curr_node
        maximizing_play = -float("inf") #compare for greatest
        minimizing_play = float("inf") #compare for lowest
        while bestChild.state.isTerminal == False and len(bestChild.untried_moves) == 0:
            parent_turn = bestChild.state.turn #oh parent turn is just the turn of the node
            for move in bestChild.children.keys(): #self is the player using MCTS, curr_node is the "root"
                print("HERE", bestChild.children)
                print("move", move)
                bestChild.visits +=1
                childNode = bestChild.children.get(move)
                print("childnode", childNode)
                holdUCB = childNode.UCBWeight(self.UCB_const, bestChild.visits, parent_turn)
                if holdUCB > maximizing_play and parent_turn == 1:
                        maximizing_play = holdUCB
                        bestChild = childNode
                if holdUCB < minimizing_play and parent_turn == -1:
                        minimizing_play = holdUCB
                        bestChild = childNode
                        ##go to the best node found? then loop to end and return
                    #add child to the path
                print("bestChild", bestChild)
            bestPath.append(bestChild)
        return bestPath




    def expansion(self, curr_node):
        """
            if curr_node is not terminal, go to its child (make a child!) :D
            returns: next_node
        """
        print("expansion")
        holdRandom = random.choice(curr_node.untried_moves)
        made_move = curr_node.state.makeMove(holdRandom)
        curr_node.untried_moves.remove(holdRandom)
        key = str(made_move)
        magicMove = str(holdRandom)
        next_node = Node(made_move)
        holdPair = {magicMove: next_node}
        curr_node.children.update(holdPair)

        hold2 = {key: next_node}
        self.nodes.update(hold2)
        
        return next_node

    def simulation(self, game_state):
        """
            rollout from point C (whatever is returned in expansion), until a terminal point
            returns: outcome
        """
        print("simulation")
        while not game_state.isTerminal:
            move = random.choice(game_state.availableMoves)
            next_state = game_state.makeMove(move)
            game_state = next_state

        return game_state.winner

    def backpropagation(self, path, outcome):
        """
            given a path and outcome, updates stats along the treeeeeeeeeeeeeeeeeeeee
            returns: nothing, it's an updating function
        """
        print("backpropagation")
        for node in path:
            node.updateValue(outcome)
        

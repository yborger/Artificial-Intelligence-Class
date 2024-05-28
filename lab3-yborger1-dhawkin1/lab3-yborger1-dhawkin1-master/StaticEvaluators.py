########################################
# CS63: Artificial Intelligence, Lab 3
# Fall 2022, Swarthmore College
########################################

import numpy as np
from random import choice
from Mancala import Mancala

def mancalaBasicEval(mancala_game):
    """Difference between the scores for each player.
    Returns +(max possible score) if player +1 has won.
    Returns -(max possible score) if player -1 has won.

    Otherwise returns (player +1's score) - (player -1's score).

    Remember that the number of houses and seeds may vary."""

    if(mancala_game.isTerminal):
        if mancala_game.winner == 1:
            return 9999

        elif mancala_game.winner == -1:
            return -9999
    maxPossibleScore = mancala_game.scores[0]-mancala_game.scores[1]
    return maxPossibleScore #returns a number

def breakthroughBasicEval(breakthrough_game):
    """Measures how far each player's pieces have advanced
    and returns the difference.

    Returns +(max possible advancement) if player +1 has won.
    Returns -(max possible advancement) if player -1 has won.

    Otherwise finds the rank of each piece (number of rows onto the board it
    has advanced), sums these ranks for each player, and
    returns (player +1's sum of ranks) - (player -1's sum of ranks).

    An example on a 5x3 board:
    ------------
    |  0  1  1 |  <-- player +1 has two pieces on rank 1
    |  1 -1  1 |  <-- +1 has two pieces on rank 2; -1 has one piece on rank 4
    |  0  1 -1 |  <-- +1 has (1 piece * rank 3); -1 has (1 piece * rank 3)
    | -1  0  0 |  <-- -1 has (1*2)
    | -1 -1 -1 |  <-- -1 has (3*1)
    ------------
    sum of +1's piece ranks = 1 + 1 + 2 + 2 + 3 = 9
    sum of -1's piece ranks = 1 + 1 + 1 + 2 + 3 + 4 = 12
    state value = 9 - 12 = -3

    Remember that the height and width of the board may vary."""

    if breakthrough_game.isTerminal:
        if breakthrough_game.winner == 1:
            return 9999

        elif breakthrough_game.winner == -1:
            return -9999
    us, enemy = 0, 0
    for row in range(len(breakthrough_game.board)):
        for col in range(len(breakthrough_game.board[row])):
            #print("row: ", row, "col: ", col)
            if(breakthrough_game.board[row][col] > 0):
                us += row
            elif(breakthrough_game.board[row][col] < 0):
                enemy += (len(breakthrough_game.board)) - row 

    return us - enemy #returns a number


def breakthroughBetterEval(breakthrough_game):
    """A heuristic that generally wins against breakthroughBasicEval.
    This must be a static evaluation function (no search allowed).

    This heuristic uses the last heuristic as a frame work but also takes into account if a piece is
    in danger of being 'killed' the next round, and insead of counting it as +the row its in, it counts
    as +/- 0 because the piece is not a threat to the other player

    """

    if breakthrough_game.isTerminal:
        if breakthrough_game.winner == 1:
            return 9999

        elif breakthrough_game.winner == -1:
            return -9999
    us, enemy, total = 0, 0, 0
    for row in range(len(breakthrough_game.board)):
        for col in range(len(breakthrough_game.board[row])):
            if(breakthrough_game.board[row][col] > 0) and indanger(breakthrough_game, row, col) == False:
                enemy += (len(breakthrough_game.board)) - row
            elif(breakthrough_game.board[row][col] < 0):
                enemy += (len(breakthrough_game.board)) - row 
            

    return us - enemy #returns a number


def indanger(breakthrough_game, r, c):
    """
    This function takes the game board current state and the row and column of the piece we are currently looking
    at and returns true if the piece has an enemy piece diagonal(and in front) of it, meaning the piece can be
    'killed' the next round, false otherwise
    """
    
    danger = False 
    if((breakthrough_game.board[r+1][c-1] == -1) or (breakthrough_game.board[r+1][c-1] == -1)):
        danger = True 

    return danger 
                



if __name__ == '__main__':
    """
    Create a game of Mancala.  Try 10 random moves and check that the
    heuristic is working properly. 
    """
    print("\nTESTING MANCALA HEURISTIC")
    print("-"*50)
    game1 = Mancala()
    print(game1)
    for i in range(10):
        move = choice(game1.availableMoves)
        print("\nmaking move", move)
        game1 = game1.makeMove(move)
        print(game1)
        score = mancalaBasicEval(game1)
        print("basicEval score", score)

    # Add more testing for the Breakthrough
    for i in range(1):
        move = choice(game1.availableMoves)
        game1 = game1.makeMove(move)
        print(game1)
        score = breakthroughBasicEval(game1)
        print("basicEval score", score)

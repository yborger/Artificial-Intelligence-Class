########################################
# CS63: Artificial Intelligence, Lab 3
# Fall 2022, Swarthmore College
########################################

class MinMaxPlayer:
    """Gets moves by depth-limited minimax search."""
    def __init__(self, boardEval, depthBound):
        self.name = "MinMax"
        self.boardEval = boardEval   # static evaluation function
        self.depthBound = depthBound # limit of search
        self.bestMove = None         # best move from root

    def getMove(self, game_state):
        """Create a recursive helper function to implement Minimax, and
        call that helper from here. Initialize bestMove to None before
        the call to helper and then return bestMove found."""
        self.bestMove = None
        goodValue = self.minimaxHelper(game_state, 0) #the self means we call function that is in the class
        #self.bestMove should now have the best move!!!
        return self.bestMove
    def minimaxHelper(self, state, depth):
        if (depth == self.depthBound) or state.isTerminal: 
            return self.boardEval(state) #boardEval still has not been done for this state!
        bestValue = state.turn * -float("inf") #state.turn is a 1 or -1
        for move in state.availableMoves:
            next_state = state.makeMove(move)
            value = self.minimaxHelper(next_state, depth+1) #recursive call woo 
            if state.turn == 1:
                if value > bestValue: 
                    bestValue = value
                    if depth == 0:
                        self.bestMove = move
            else:
                if value < bestValue:
                    bestValue = value
                    if depth == 0:
                        self.bestMove = move
        return bestValue
    


class PruningPlayer:
    """Gets moves by depth-limited minimax search with alpha-beta pruning."""
    def __init__(self, boardEval, depthBound):
        self.name = "Pruning"
        self.boardEval = boardEval   # static evaluation function
        self.depthBound = depthBound # limit of search
        self.bestMove = None         # best move from root
        
    def getMove(self, game_state):
        """Create a recursive helper function to implement AlphaBeta pruning
        and call that helper from here. Initialize bestMove to None before
        the call to helper and then return bestMove found."""
        self.bestMove = None #nothing decided yet!
        goodValueP = self.pruningAB(game_state, 0, -float("inf"), float("inf"))
        #alpha is set to lowest so it can be compared to the maximizer
        #beta is set to highest so it can be compared to the minimizer
        return self.bestMove

    def pruningAB(self, state, depth, alpha, beta): 
        if depth == self.depthBound or state.isTerminal:
            return self.boardEval(state)
        bestValue = state.turn * -float("inf")
        for move in state.availableMoves:
            #print("move: ", move)
            next_state = state.makeMove(move)
            #print("check! alpha: ", alpha, " beta: ", beta)
            value = self.pruningAB(next_state, depth+1, alpha, beta)
            if state.turn == 1:
                if value > bestValue:
                    bestValue = value
                    if depth == 0:
                        self.bestMove = move #updating the class's variable
                alpha = max(value, alpha)
            else: #minimizer
                if value < bestValue:
                    bestValue = value
                    if depth == 0:
                        self.bestMove = move
                beta = min(value, beta)
            if alpha >= beta:
                #print("breaking! alpha: ", alpha, "  beta: ", beta)
                break #no need to keep executing the function
        return bestValue



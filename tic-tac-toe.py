#We will take the epsilon greedy strategy here
boardLength = 3
winningLength = 3
import numpy as np
class Environment:
    
    def __init__(self):
        self.board = np.zeros((boardLength, boardLength))
        self.x = 1
        self.o = -1
        self.winner = None
        self.ended = False
        self.numberOfStates = 3 ** (boardLength*boardLength)
        
        
    def isEmpty(self, i, j):
        return self.board[i,j] == 0.0
    
    def getReward(self, symbol):
        if not self.gameOver():
            return 0
        elif self.winner == symbol:
            return 1
        else:
            return 0
        
    def getState(self):
        #Returns the state of the board represented as a decimal number
        k = 0
        stateDecimal = 0
        for i in range(boardLength):
            for j in range(boardLength):
                if self.board[i,j] == 0:
                    num = 0
                elif self.board[i,j] == self.x:
                    num = 1
                else:
                    num = 2
                stateDecimal += (3**k)*num
                k += 1
        return stateDecimal
                
        
        
    def gameOver(self):
        
        #check for rows
        for i in range(boardLength):
            for player in (self.x, self.o):
                if self.board[i].sum() == player*winningLength:
                    self.winner = player
                    self.ended = True
                    #print("rows matched")
                    #print(player)
                    #print(self.board)
                    return True
        
        #check for columns
        for i in range(boardLength):
            for player in (self.x, self.o):
                if self.board[:,i].sum() == player*winningLength:
                    self.winner = player
                    self.ended = True
                    #print("columns matched")
                    #print(player)
                    #print(self.board)
                    return True
        
        #check for diagonals
        for player in (self.x, self.o):
            if self.board.trace() == player*winningLength:
                self.winner = player
                self.ended = True
                #print("1st diagonal matched")
                #print(player)
                #print(self.board)
                return True
            
            if np.fliplr(self.board).trace() == player*winningLength:
                self.winner = player
                self.ended = True
                #print("2nd diagonal matched")
                #print(player)
                #print(self.board)
                return True
            
        k = 0
        for i in range(boardLength):
            for j in range(boardLength):
                if not self.isEmpty(i,j):
                    k += 1
        
        if (k == boardLength*boardLength):
            #print("draw")
            #print(self.board)
            return True
        
        return False
            
    def drawBoard(self):
        for i in range(boardLength):
            print ("-------------")
            for j in range(boardLength):
                print (" ")
                if self.board[i,j] == self.x:
                  print ("x")
                elif self.board[i,j] == self.o:
                  print ("o")
                else:
                  print (" ")
            print ("\n")
        print ("-------------")
    
    def drawBoard1(self):
        print (self.board)
        
    def drawBoard2(self, p):
        print ("inside drawboard2 ")
        print (p.verbose)
        if p.verbose:
            print (self.board)
            k = self.getState()
            print(p.value[k])
        

class Agent:
    
    #s
    def __init__(self, eps=0.3, alpha=0.5): 
        self.eps = eps
        self.alpha = alpha
        self.verbose = False
        self.stateHistory = []

    def setEpsilon(self, epsilon):
        self.eps = epsilon
        
    def setValue(self, value):
        self.value = value
        
    def setSymbol(self, symbol):
        self.symbol = symbol
        
    def setVerbose(self, verbose):
        self.verbose = verbose
        
    def resetHistory(self):
        self.stateHistory = []
        
    def takeAction(self, env):
        move = []
        r = np.random.random()
        if (r < self.eps):
            possibleMoves = []
            for i in range(boardLength):
                for j in range(boardLength):
                    if env.isEmpty(i,j):
                        possibleMoves.append((i,j))
            #print(possibleMoves)
            if self.verbose:
                print("Hola")
            r = np.random.choice(len(possibleMoves))
            move = possibleMoves[r]
        else:
            x = -1
            y = -1
            maxValue = -1
            valueTable = np.zeros((3,3))
            for i in range(boardLength):
                for j in range(boardLength):
                    if env.isEmpty(i,j):
                        env.board[i,j] = self.symbol
                        state = env.getState()
                        valueTable[i,j] = self.value[state]
                        env.board[i,j] = 0
                        if self.value[state] > maxValue:
                            x = i
                            y = j
                            maxValue = self.value[state]
                    else:
                        valueTable[i,j] = env.board[i,j]
            move = (x,y)
            if self.verbose:
                print(valueTable)
        
        env.board[move[0], move[1]] = self.symbol
        #env.drawBoard1()
        #print (self.symbol)
        #print (env.board[move[0], move[1]])
        
    def updateStateHistory(self, s):
        self.stateHistory.append(s)
        
    def update(self, env):
        #this is the core part of the program which represents the AI
        #the update equation for the value is V(s) <- v(s) + alpha(V(s') - V(s))
        reward = env.getReward(self.symbol)
        for i in reversed(self.stateHistory):
            self.value[i] = self.value[i] + self.alpha * (reward - self.value[i])
            reward = self.value[i]
        self.resetHistory()
        
        
class Human:
    
    
    def __init__(self):
        pass
        
    def setSymbol(self, symbol):
        self.symbol = symbol
        
    def takeAction(self, env):
        while True:
          # break if we make a legal move
          move = input("Enter coordinates i,j for your next move (i,j=0..2): ")
          i, j = move.split(',')
          i = int(i)
          j = int(j)
          if env.isEmpty(i, j):
            env.board[i,j] = self.symbol
            break

    def update(self, env):
        pass

    def updateStateHistory(self, s):
        pass


def getAllStatesAndWinners(env, x=0, y=0):

    results = []
    for i in (0, env.x, env.o):

        env.board[x,y] = i
        if y == 2:

            if x == 2:

                #have reached the last point
                state = env.getState()
                isGameOver = env.gameOver()
                winner = env.winner
                results.append((state, winner, isGameOver))

            else:
                results += getAllStatesAndWinners(env, x+1, 0)
        else:
            results += getAllStatesAndWinners(env, x,y+1)

    return results


def initialV_x(env, state_winner_triples):
    # initialize state values as follows
    # if x wins, V(s) = 1
    # if x loses or draw, V(s) = 0
    # otherwise, V(s) = 0.5
    V = np.zeros(env.numberOfStates)
    for state, winner, ended in state_winner_triples:
        if ended:
            if winner == env.x:
                v = 1
            else:
                v = 0
        else:
            v = 0.5
        V[state] = v
    return V


def initialV_o(env, state_winner_triples):
  # this is (almost) the opposite of initial V for player x
  # since everywhere where x wins (1), o loses (0)
  # but a draw is still 0 for o
    V = np.zeros(env.numberOfStates)
    for state, winner, ended in state_winner_triples:
        if ended:
            if winner == env.o:
                v = 1
            else:
                v = 0
        else:
            v = 0.5
        V[state] = v
    return V

def playGame(p1, p2, board, draw=False):
    
    currentPlayer = None
    while not board.gameOver():
        if currentPlayer == p1:
            currentPlayer = p2
        else:
            currentPlayer = p1
        
        if draw:
            if draw == 1 and currentPlayer == p1:
                board.drawBoard()
            if draw == 2 and currentPlayer == p2:
                board.drawBoard2(p1)
            
        currentPlayer.takeAction(board)
        
        state = board.getState()
        #update the state in which the board is for both the players
        p1.updateStateHistory(state)
        p2.updateStateHistory(state)
        
        #if draw:
        #    board.drawBoard()
        
    #update the value functions
    p1.update(board)
    p2.update(board)
        
        
if __name__ == '__main__':
    
    p1 = Agent()
    p2 = Agent()
    
    env = Environment()
    state_winner_triples = getAllStatesAndWinners(env)
    p1.setSymbol(env.x)
    Vx = initialV_x(env, state_winner_triples)
    Vo = initialV_o(env, state_winner_triples)
    print(Vx)
    p2.setSymbol(env.o)
    p1.setVerbose(False)
    p2.setVerbose(False)
    p1.setValue(Vx)
    p2.setValue(Vo)
    
    for i in range(10000):
        playGame(p1, p2, Environment())
        #print ("-------------")
        
    # play human vs. agent
    # do you think the agent learned to play the game well?
    human = Human()
    human.setSymbol(env.o)
    while True:
        p1.setVerbose(True)
        p1.setEpsilon(-2)
        playGame(p1, human, Environment(), draw=2)
        #print (env.board)
        # I made the agent player 1 because I wanted to see if it would
        # select the center as its starting move. If you want the agent
        # to go second you can switch the human and AI.
        answer = input("Play again? [Y/n]: ")
        if answer and answer.lower()[0] == 'n':
            break

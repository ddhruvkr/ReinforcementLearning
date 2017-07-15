import numpy as np
from grid_world import standardGrid, negativeGrid
from iterative_policy_evaluation import printValues, printPolicy

smallEnough = 10e-4
gamma = 0.9
alpha = 0.1
allActions = ('U', 'D', 'L', 'R')

# the td(0) method.
# the update of valueF happens within each episode itself

def randomAction(a, eps=0.1):
	p = np.random.random()
	if p < (1 - eps):
		return a
	else:
		return np.random.choice(allActions)

def playGame(grid, policy):
	s = (2,0)
	r = 0
	grid.setState(s)
	statesRewardsList.append((s,0))
	#traverse the grid till we reach the terminal state
	while not grid.isGameOver():
		oldS = s
		oldR = r
		a = randomAction(policy[s])
		r = grid.move(a)
		s = grid.getCurrentState()
		valueF[oldS] += alpha*(r + gamma*valueF[s] - valueF[oldS])
	return valueF

if __name__ == '__main__':
	
	grid = negativeGrid()
	valueF = {}
	policy = {}
	for s in grid.allStates():
		valueF[s] = 0
	# state -> action
	policy = {
		(2, 0): 'U',
		(1, 0): 'U',
		(0, 0): 'R',
		(0, 1): 'R',
		(0, 2): 'R',
		(1, 2): 'R',
		(2, 1): 'R',
		(2, 2): 'R',
		(2, 3): 'U',
	}
	statesRewardsList = []
	printPolicy(policy, grid)
	for n in range(2000):
		valueF = playGame(grid, policy)

	print ("final values:")
	printValues(valueF, grid)
	printPolicy(policy, grid)
	print ("\n\n")
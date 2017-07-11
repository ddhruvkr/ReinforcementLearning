import numpy as np
from grid_world import standardGrid, negativeGrid
from iterative_policy_evaluation import printValues, printPolicy

smallEnough = 10e-4
gamma = 0.9
#this is deterministic
#so p(s',a|s,r) =1 or 0, so no stochastic state transition
allActions = ('U', 'D', 'L', 'R')

def playGame(grid, policy):
	
	states = list(grid.actions.keys())
	startIndex = np.random.choice(len(states))
	s = states[startIndex]
	grid.setState(s)
	statesAndRewardsList = []
	statesAndRewardsList.append((s,0))
	#traverse the grid till we reach the terminal state
	while not grid.isGameOver():
		a = policy[s]
		r = grid.move(a)
		s = grid.getCurrentState()
		statesAndRewardsList.append((s,r))

	# now convert the stateAndRewardsMap into stateAndReturnMap
	statesAndReturnsList = []
	returns = 0
	first  = True
	#start at the terminal state and traverse backwards
	for s,r in reversed(statesAndRewardsList):
		if first:
			first = False
		else:
			statesAndReturnsList.append((s, returns))
		returns = r + gamma * returns
	return reversed(statesAndReturnsList)


if __name__ == '__main__':
	
	grid = negativeGrid()
	valueF = {}
	policy = {}
	allReturns={}
	states = grid.actions.keys()
	for s in states:
		if s in grid.actions:
			allReturns[s] = []
		else:
			valueF[s] = 0
	stateAndReturnsList = []
	# initial policy
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

	for n in range(100):
		seenStates = set()
		statesAndReturnsList = playGame(grid, policy)
		for s,r in statesAndReturnsList:
			if s not in seenStates:
				#first visit policy
				seenStates.add(s)
				if s in allReturns:
					allReturns[s].append(float(r))
				else:
					allReturns[s] = r
				valueF[s] = np.mean(allReturns[s])

	printPolicy(policy, grid)
	printValues(valueF, grid)
	print ("\n\n")
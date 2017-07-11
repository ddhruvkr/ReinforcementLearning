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
	#print (states)
	startIndex = np.random.choice(len(states))
	s = states[startIndex]
	#print (s)
	grid.setState(s)
	statesAndRewardsList = []
	statesAndRewardsList.append((s,0))
	while not grid.isGameOver():
		#print ("hola")
		a = policy[s]
		r = grid.move(a)
		s = grid.getCurrentState()
		statesAndRewardsList.append((s,r))
	#print (statesAndRewardsList)

	# now convert the stateAndRewardsMap into stateAndReturnMap
	statesAndReturnsList = []
	returns = 0
	first  = True
	for s,r in reversed(statesAndRewardsList):
		if first:
			first = False
		else:
			statesAndReturnsList.append((s, returns))
		returns = r + gamma * returns
	#print (statesAndReturnsList)
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
		#print (n)
		statesAndReturnsList = playGame(grid, policy)
		#print (statesAndReturnsList)
		for s,r in statesAndReturnsList:
			if s not in seenStates:
				seenStates.add(s)
				#print(allReturn)
				if s in allReturns:

					#temp = allReturns.get(s)
					#temp.append(r)
					#print(allReturns[s])
					allReturns[s].append(float(r))
				else:
					allReturns[s] = r
				#allReturns[s].append(r)
				#print(allReturns[s])
				valueF[s] = np.mean(allReturns[s])

	printPolicy(policy, grid)
	printValues(valueF, grid)
	print ("\n\n")
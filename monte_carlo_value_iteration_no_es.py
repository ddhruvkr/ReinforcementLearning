import numpy as np
from grid_world import standardGrid, negativeGrid
from iterative_policy_evaluation import printValues, printPolicy

smallEnough = 10e-4
gamma = 0.9
#this is deterministic
#so p(s',a|s,r) =1 or 0, so no stochastic state transition
allActions = ('U', 'D', 'L', 'R')

def randomAction(a, eps=0.1):
	p = np.random.random()
	if p < (1 - eps):
		return a
	else:
		return np.random.choice(allActions)

def playGame(grid, policy):
	
	s = (2,0)
	statesActionsRewardsList = []
	grid.setState(s)
	a = randomAction(policy[s])
			
	# each triple is s(t), a(t), r(t)
	# but r(t) results from taking action a(t-1) from s(t-1) and landing in s(t)
	# so any triplet would mean that i am at state s, at this state i should take an action a, and by coming to this state i got a reward r
	statesActionsRewardsList.append((s,a,0))
	#traverse the grid till we reach the terminal state
	while True:
		r = grid.move(a)
		s = grid.getCurrentState()
		if grid.isGameOver():
			statesActionsRewardsList.append((s, None, r))
			break
		else:
			randomNumber = np.random.random()
			a = randomAction(policy[s])
			statesActionsRewardsList.append((s, a, r))
	# now convert the stateAndRewardsMap into stateAndReturnMap
	#print (statesActionsRewardsList)
	statesActionsReturnsList = []
	returns = 0
	first  = True
	#start at the terminal state and traverse backwards
	for s,a,r in reversed(statesActionsRewardsList):
		if first:
			first = False
		else:
			statesActionsReturnsList.append((s, a, returns))
		returns = r + gamma * returns

	return reversed(statesActionsReturnsList)

def getMaximumFromDict(d):
	maxKey = None
	maxValue = float('-inf')
	for k,v in d.items():
		if v > maxValue:
			maxKey = k
			maxValue = v
	return maxKey,maxValue



if __name__ == '__main__':
	
	grid = negativeGrid()
	valueF = {}
	policy = {}
	for s in grid.actions.keys():
		policy[s] = np.random.choice(allActions)
	allReturns={}
	states = grid.actions.keys()
	for s in states:
		if s in grid.actions:
			valueF[s] = {}
			for a in allActions:
				allReturns[(s,a)] = []
				valueF[s][a] = 0
	stateActionsReturnsList = []
	printPolicy(policy, grid)
	for n in range(2000):
		#print(n)
		seenStates = set()
		stateActionsReturnsList = playGame(grid, policy)
		for s,a,r in stateActionsReturnsList:
			if (s,a) not in seenStates:
				#first visit policy
				seenStates.add((s,a))
				if (s,a) in allReturns:
					allReturns[(s,a)].append(r)
				else:
					allReturns[(s,a)] = r
				valueF[s][a] = np.mean(allReturns[(s,a)])
		#print (policy)
		#print(valueF)
		for s in states:
			#print (valueF[s])
			#print (getMaximumFromDict(valueF[s])[0])
			policy[s] = getMaximumFromDict(valueF[s])[0]

	V = {}
	for s in policy.keys():
		V[s] = getMaximumFromDict(valueF[s])[1]

	print ("final values:")
	printValues(V, grid)
	#print(valueF)
	printPolicy(policy, grid)
	print ("\n\n")
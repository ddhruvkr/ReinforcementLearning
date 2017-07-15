import numpy as np
from grid_world import standardGrid, negativeGrid
from iterative_policy_evaluation import printValues, printPolicy
from monte_carlo_value_iteration_es import getMaximumFromDict

smallEnough = 10e-4
gamma = 0.9
alpha = 0.1
allActions = ('U', 'D', 'L', 'R')

def randomAction(a, eps=0.1):
	p = np.random.random()
	if p < (1 - eps):
		return a
	else:
		return np.random.choice(allActions)

def playGame(grid, policyQ, t, count):
	s = (2,0)
	r = 0
	a = randomAction(getMaximumFromDict(policyQ[s])[0], eps=0.5/t)
	grid.setState(s)
	#traverse the grid till we reach the terminal state
	while not grid.isGameOver():
		oldS = s
		oldR = r
		oldA = a
		r = grid.move(a)
		s = grid.getCurrentState()
		# so the only difference from sarsa is that we did not had to calculate the action/policy for this new state before updating the Q for the state
		# in which we entered.
		q2 = getMaximumFromDict(policyQ[s])[1]
		# adaptive learning rate
		policyQ[oldS][oldA] += (alpha/count[s][a])*(r + gamma*q2 - policyQ[oldS][oldA])
		a = randomAction(getMaximumFromDict(policyQ[s])[0], eps=0.5/t)
		count[s][a] += 0.000005
	return policyQ

if __name__ == '__main__':
	
	grid = negativeGrid()
	valueF = {}
	policyQ = {}
	policy = {}
	allReturns={}
	for s in grid.actions.keys():
		policy[s] = np.random.choice(allActions)
	states = grid.allStates()
	for s in states:
		valueF[s] = {}
		policyQ[s] = {}
		for a in allActions:
			policyQ[s][a] = 0
			valueF[s][a] = 0
	t = 1.0
	count = {}
	for s in states:
		count[s] = {}
		for a in allActions:
			count[s][a] = 1.0
	for n in range(5000):
		if n % 100 == 0:
			t += 10e-3
		policyQ = playGame(grid, policyQ, t, count)

	for s, Qs in policyQ.items():
		valueF[s] = getMaximumFromDict(policyQ[s])[1]

	for s in policy.keys():
		if s in grid.actions.keys():
			policy[s] = getMaximumFromDict(policyQ[s])[0]
	print ("final values:")
	printValues(valueF, grid)
	print ("final policy:")
	printPolicy(policy, grid)
import numpy as np
from grid_world import standardGrid, negativeGrid
from iterative_policy_evaluation import printValues, printPolicy
from monte_carlo_value_iteration_es import getMaximumFromDict
from monte_carlo_random import playGame, randomAction

smallEnough = 10e-4
gamma = 0.9
learningRate = 0.001
allActions = ('U', 'D', 'L', 'R')

if __name__ == '__main__':
	
	grid = negativeGrid()
	valueF = {}
	policyQ = {}
	policy = {
		(2, 0): 'U',
		(1, 0): 'U',
		(0, 0): 'R',
		(0, 1): 'R',
		(0, 2): 'R',
		(1, 2): 'U',
		(2, 1): 'L',
		(2, 2): 'U',
		(2, 3): 'L',
	}
	# x = s2x as a function of s
	# our function is v_h = theta*x
	# after we differentiate error wrt theta
	# error = (v(s) - v_h(s))^2
	# since v_h is a function of theta
	# therefore for gradient descent we use only x
	theta = np.random.randn(4) / 2
	def s2x(s):
		return np.array([s[0] - 1, s[1] - 1.5, s[0]*s[1] - 3, 1])
	
	t = 1.0
	for n in range(2000):
		if n % 100 == 0:
			t += 0.01
		alpha = learningRate/t
		seenStates = set()
		statesAndReturnsList = playGame(grid, policy)
		for s,r in statesAndReturnsList:
			if s not in seenStates:
				x = s2x(s)
				V_hat = theta.dot(x)
				theta += alpha * (r - V_hat)*x
			seenStates.add(s)

	states = grid.allStates()
	for s in states:
		if s in grid.actions:
			valueF[s] = theta.dot(s2x(s))
		else:
			# terminal state
			valueF[s] = 0

	print ("final values:")
	printValues(valueF, grid)
	print ("final policy:")
	printPolicy(policy, grid)
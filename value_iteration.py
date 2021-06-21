import numpy as np
from grid_world import standardGrid, negativeGrid
from iterative_policy_evaluation import printValues, printPolicy

if __name__ == '__main__':

	smallEnough = 10e-4
	gamma = 0.9
	#this is deterministic
	#so p(s',a|s,r) =1 or 0, so no stochastic state transition
	print('Hola')
	grid = negativeGrid()
	states = grid.allStates()
	allActions = ('U', 'D', 'L', 'R')
	valueF = {}
	policy = {}
	for s in states:
		valueF[s] = 0

	# initial policy
	for a in grid.actions.keys():
		policy[a] = np.random.choice(allActions)



	'''for s in states:
		# V[s] = 0
		if s in grid.actions:
			valueF[s] = np.random.random()
		else:
			# terminal state
			valueF[s] = 0'''
		#policy evaluation step

	# so basically what we do different here than in policy iteration
	# we first update the valueF as in policy iteration, but we do one thing different.
	# we also keep experimenting with the states or policy. we were doing this step as step 3 in PI.
	# once we get the optimum valueF, we then use this valueF once and for all like we were doing in PI to get the policy
	# we only need to do this step once as compared to multiple times in PI, as we taking differnt combinations of policy in step 2 itself
	while True:
		
		largest = 0
		for s in states:
			if s in policy:
				biggestValue = float('-inf')
				a = policy[s]
				oldStateValue = valueF[s]
				pA = 1
				for a in allActions:
					newValue = 0
					grid.setState(s)
					# moves and returns a reward which we get on moving
					r = grid.move(a)
					newValue = pA * (r + gamma*(valueF[grid.getCurrentState()]))
					if newValue > biggestValue:
						biggestValue = newValue
				valueF[s] = biggestValue
				largest = max(largest, np.abs(valueF[s] - oldStateValue))

		if largest < smallEnough:
			break
		#print ("values for uniformly random actions:")
		printPolicy(policy, grid)
		printValues(valueF, grid)
		print ("\n\n")

	#policy improvement step
	# do policy improvement only once compared to policy iteration where this is also done within the outer loop
	isPolicyChanged = False
	for s in states:

		if s in policy:
			newValue = 0
			oldPolicy = policy[s]
			newPolicy = None 
			bestValue = float('-inf')
			pA = 1
			for a in allActions:
				grid.setState(s)
				# moves and returns a reward which we get on moving
				r = grid.move(a)
				newValue = pA * (r + gamma*(valueF[grid.getCurrentState()]))
				if newValue > bestValue:
					bestValue = newValue
					newPolicy = a	

			policy[s] = newPolicy

	print ("values for uniformly random actions:")
	printValues(valueF, grid)
	print ("\n\n")

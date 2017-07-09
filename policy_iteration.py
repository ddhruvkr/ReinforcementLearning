import numpy as np
from grid_world import standardGrid, negativeGrid
from iterative_policy_evaluation import printValues, printPolicy

if __name__ == '__main__':

	smallEnough = 10e-4
	gamma = 0.9
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
	while True:
		#policy evaluation step
		while True:
			
			largest = 0
			for s in states:
				if s in policy:
					newValue = 0
					a = policy[s]
					oldStateValue = valueF[s]
					pA = 1
					#print (pA)
					
					grid.setState(s)
					# moves and returns a reward which we get on moving
					r = grid.move(a)
					#print(s)
					#print(a)
					#print (grid.getCurrentState())
					newValue += pA * (r + gamma*(valueF[grid.getCurrentState()]))
					valueF[s] = newValue
					largest = max(largest, np.abs(valueF[s] - oldStateValue))

			if largest < smallEnough:
				break
		#print ("values for uniformly random actions:")
		printPolicy(policy, grid)
		printValues(valueF, grid)
		print ("\n\n")

		#policy improvement step
		isPolicyChanged = False
		for s in states:

			if s in policy:
				newValue = 0
				oldPolicy = policy[s]
				newPolicy = None
				oldStateValue = valueF[s]
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
				if oldPolicy != newPolicy:
					isPolicyChanged = True
		if isPolicyChanged == False:
			break

	print ("values for uniformly random actions:")
	printValues(valueF, grid)
	print ("\n\n")
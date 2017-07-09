import numpy as np
from grid_world import standardGrid, negativeGrid
from iterative_policy_evaluation import printValues, printPolicy
#this basically refers to the windy grid problem, where lets assume a wind is blowing in the cells and tberefore we are not sure that
#our agent would move for sure where it wants to
#the case taken here is that if our agent wants to move to say direction L, it could do so only with a probability of 0.5
#and it could move to any other direction with an equal probability

if __name__ == '__main__':

	smallEnough = 10e-4
	gamma = 0.9
	print('Hola')
	grid = negativeGrid(stepCost=-1.0)
	states = grid.allStates()
	allActions = ('U', 'D', 'L', 'R')
	valueF = {}
	policy = {}
	for s in states:
		valueF[s] = 0

	# initial policy
	for a in grid.actions.keys():
		policy[a] = np.random.choice(allActions)

	while True:
		#policy evaluation step
		while True:
			largest = 0
			for s in states:
				if s in policy:
					newValue = 0
					a = policy[s]
					oldStateValue = valueF[s]
					#we need to calculate the probability for every state as it could now go onto any state
					#so the value in the value function would be the sum of all these cases
					for a2 in allActions:
						if a2 == a:
							pA = 0.5
						else:
							pA = 0.5/3
					
						grid.setState(s)
						# moves and returns a reward which we get on moving
						r = grid.move(a2)
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
				
				oldPolicy = policy[s]
				newPolicy = None
				oldStateValue = valueF[s]
				bestValue = float('-inf')
				for a in allActions:
					newValue = 0
					#we need to calculate the probability for every state as it could now go onto any state after selecting the state a
					#so the value in the value function would be the sum of all these cases
					for a2 in allActions:
						if a2 == a:
							pA = 0.5
						else:
							pA = 0.5/3
						grid.setState(s)
						# moves and returns a reward which we get on moving
						r = grid.move(a2)
						newValue += pA * (r + gamma*(valueF[grid.getCurrentState()]))
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
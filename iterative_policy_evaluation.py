import numpy as np
from grid_world import standardGrid

smallEnough = 10e-4
def printValues(valueF, grid):
  for i in range(grid.width):
    print ("---------------------------")
    for j in range(grid.height):
      v = valueF.get((i,j), 0)
      if v >= 0:
        print (" %.2f|" % v, end="")
      else:
        print ("%.2f|" % v, end="") # -ve sign takes up an extra space
    print ("")


def printPolicy(policyF, grid):
  for i in range(grid.width):
    print ("---------------------------")
    for j in range(grid.height):
      a = policyF.get((i,j), ' ')
      print ("  %s  |" % a, end="")
    print ("")

if __name__ == '__main__':

	print('Hola')
	grid = standardGrid()
	states = grid.allStates()

	valueF = {}
	for s in states:
		valueF[s] = 0

	while True:
		gamma = 1
		largest = 0
		for s in states:
			if s in grid.actions:
				newValue = 0
				oldStateValue = valueF[s]
				pA = 1/len(grid.actions.get(s))
				#print (pA)
				
				for a in grid.actions[s]:
					grid.setState(s)
					# movees and returns a reward which we get on moving
					r = grid.move(a)
					#print(s)
					#print(a)
					#print (grid.getCurrentState())
					newValue += pA * (r + gamma*(valueF[grid.getCurrentState()]))
				valueF[s] = newValue
				largest = max(largest, np.abs(valueF[s] - oldStateValue))

		if largest < smallEnough:
			break

	print ("values for uniformly random actions:")
	printValues(valueF, grid)
	print ("\n\n")


	### fixed policy ###
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
	printPolicy(policy, grid)
				
	valueF = {}
	for s in states:
		valueF[s] = 0

	while True:
		gamma = 0.9
		largest = 0
		for s in states:
			if s in policy:
				newValue = 0
				a = policy[s]
				oldStateValue = valueF[s]
				pA = 1
				#print (pA)
				
				grid.setState(s)
				# movees and returns a reward which we get on moving
				r = grid.move(a)
				#print(s)
				#print(a)
				#print (grid.getCurrentState())
				newValue += pA * (r + gamma*(valueF[grid.getCurrentState()]))
				valueF[s] = newValue
				largest = max(largest, np.abs(valueF[s] - oldStateValue))

		if largest < smallEnough:
			break

	print ("values for uniformly random actions:")
	printValues(valueF, grid)
	print ("\n\n")

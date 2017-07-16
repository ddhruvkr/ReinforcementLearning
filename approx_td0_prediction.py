import numpy as np
from grid_world import standardGrid, negativeGrid
from iterative_policy_evaluation import printValues, printPolicy
from td0_prediction import randomAction

smallEnough = 10e-4
gamma = 0.9
learningRate = 0.001
allActions = ('U', 'D', 'L', 'R')

class Model:

	def __init__(self):
		self.theta = np.random.randn(4) / 2

	def s2x(self,s):
		return np.array([s[0] - 1, s[1] - 1.5, s[0]*s[1] - 3, 1])

	def predict(self, s):
		return self.theta.dot(self.s2x(s))

	def grad(self, s):
		return self.s2x(s)

def playGame(grid, policy):
	s = (2,0)
	r = 0
	grid.setState(s)
	statesRewardsList = []
	statesRewardsList.append((s,0))
	#traverse the grid till we reach the terminal state
	while not grid.isGameOver():
		oldS = s
		oldR = r
		a = randomAction(policy[s])
		r = grid.move(a)
		s = grid.getCurrentState()
		statesRewardsList.append((s,r))
		#valueF[oldS] += alpha*(r + gamma*valueF[s] - valueF[oldS])
	return statesRewardsList

if __name__ == '__main__':
	
	grid = negativeGrid()
	valueF = {}
	policyQ = {}
	model = Model()
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
	
	t = 1.0
	for n in range(20000):
		if n % 10 == 0:
			t += 0.01
		alpha = learningRate/t
		statesRewardsList = playGame(grid, policy)
		for i in range(len(statesRewardsList)-1):
			s,r = statesRewardsList[i]
			s2,r2 = statesRewardsList[i+1]
			
			x2 = model.grad(s2)
			V_hat = model.predict(s2)
			x = model.grad(s)
			V = model.predict(s)
			if grid.isTerminalState(s2):
				target = r2
			else:
				target = r2 + gamma*V_hat
			# here we dont use the actual return, but estimate it from our next state observation
			# so we are using the output for the next stage given by the model itself, to set as a target for the training of the model
			# this is strange but it worked
			# theta += alpha(r + gamma*v_hat(s',theta) - v(s,theta))diff(v(s,theta))
			# theta += alpha(target - prediction)diff(prediction)
			# as we target is not the fu
			model.theta += alpha * (target - V)*x

	states = grid.allStates()
	for s in states:
		if s in grid.actions:
			valueF[s] = model.predict(s)
		else:
			# terminal states
			valueF[s] = 0

	print ("final values:")
	printValues(valueF, grid)
	print ("final policy:")
	printPolicy(policy, grid)
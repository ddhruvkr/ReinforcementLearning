import numpy as np

class Grid:

	def __init__(self, width, height, start):

		self.width = width
		self.height = height
		self.x = start[0]
		self.y = start[1]

	def setRewardsActions(self, rewards, actions):

		# (x,y) --> reward
		self.rewards = rewards

		# (x,y) --> list of actions which are allowed
		self.actions = actions

	def setState(self, x, y):
		self.x = x
		self.y = y

	def getCurrentState(self):
		return (self.x, self.y)

	def isTerminalState(self, state):
		# state as in (x,y)
		# as for a terminal state there would be no actions associated with it
		return state not in self.actions

	def allStates(self):
		#states would either have an action or a rewards associated with it
		return set(self.actions.keys() + self.rewards.keys())

	def isGameOver(self):
		#since a terminal state would have no action
		return (self.x, self.y) not in self.actions

	def move(self, action):
		if action in self.actions[(self.x, self.y)]:
			if action == 'U':
				self.y += 1
			elif action == 'D':
				self.y -= 1
			elif action == 'L':
				self.x -= 1
			elif action == 'R':
				self.x += 1
		return self.rewards.get((self.x, self.y), 0)

	def undoMove(self, action):
		if action == 'U':
			self.y -= 1
		elif action == 'D':
			self.y += 1
		elif action == 'L':
			self.x += 1
		elif action == 'R':
			self.x -= 1
		assert(self.getCurrentState() in self.allStates())

	def standardGrid():
		# define a grid that describes the reward for arriving at each state
		# and possible actions at each state
		# the grid looks like this
		# x means you can't go there
		# s means start position
		# number means reward at that state
		# .  .  .  1
		# .  x  . -1
		# s  .  .  .
		g = Grid(3, 4, (2, 0))
		rewards = {(0, 3): 1, (1, 3): -1}
		actions = {
			(0, 0): ('D', 'R'),
			(0, 1): ('L', 'R'),
			(0, 2): ('L', 'D', 'R'),
			(1, 0): ('U', 'D'),
			(1, 2): ('U', 'D', 'R'),
			(2, 0): ('U', 'R'),
			(2, 1): ('L', 'R'),
			(2, 2): ('L', 'R', 'U'),
			(2, 3): ('L', 'U'),
			}
		g.set(rewards, actions)
		return g


	def negativeGrid(stepCost=-0.1):
		# in this game we want to try to minimize the number of moves
		# so we will penalize every move
		g = standard_grid()
		g.rewards.update({
			(0, 0): step_cost,
			(0, 1): step_cost,
			(0, 2): step_cost,
			(1, 0): step_cost,
			(1, 2): step_cost,
			(2, 0): step_cost,
			(2, 1): step_cost,
			(2, 2): step_cost,
			(2, 3): step_cost,
			})
		return g

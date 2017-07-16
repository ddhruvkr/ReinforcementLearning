import numpy as np
from grid_world import standardGrid, negativeGrid
from iterative_policy_evaluation import printValues, printPolicy
from monte_carlo_value_iteration_no_es import getMaximumFromDict
from sarsa import randomAction

smallEnough = 10e-4
gamma = 0.9
learningRate = 0.001
allActions = ('U', 'D', 'L', 'R')

class Model:

  def __init__(self):
    self.theta = np.random.randn(25) / np.sqrt(25)

  def sa2x(self, s, a):
    # using just (r, c, r*c, u, d, l, r, 1) is not expressive enough
    return np.array([
      s[0] - 1              if a == 'U' else 0,
      s[1] - 1.5            if a == 'U' else 0,
      (s[0]*s[1] - 3)/3     if a == 'U' else 0,
      (s[0]*s[0] - 2)/2     if a == 'U' else 0,
      (s[1]*s[1] - 4.5)/4.5 if a == 'U' else 0,
      1                     if a == 'U' else 0,
      s[0] - 1              if a == 'D' else 0,
      s[1] - 1.5            if a == 'D' else 0,
      (s[0]*s[1] - 3)/3     if a == 'D' else 0,
      (s[0]*s[0] - 2)/2     if a == 'D' else 0,
      (s[1]*s[1] - 4.5)/4.5 if a == 'D' else 0,
      1                     if a == 'D' else 0,
      s[0] - 1              if a == 'L' else 0,
      s[1] - 1.5            if a == 'L' else 0,
      (s[0]*s[1] - 3)/3     if a == 'L' else 0,
      (s[0]*s[0] - 2)/2     if a == 'L' else 0,
      (s[1]*s[1] - 4.5)/4.5 if a == 'L' else 0,
      1                     if a == 'L' else 0,
      s[0] - 1              if a == 'R' else 0,
      s[1] - 1.5            if a == 'R' else 0,
      (s[0]*s[1] - 3)/3     if a == 'R' else 0,
      (s[0]*s[0] - 2)/2     if a == 'R' else 0,
      (s[1]*s[1] - 4.5)/4.5 if a == 'R' else 0,
      1                     if a == 'R' else 0,
      1
    ])

  def predict(self, s, a):
    x = self.sa2x(s, a)
    return self.theta.dot(x)

  def grad(self, s, a):
    return self.sa2x(s, a)


def getPolicyQ(model, s):
  # we need Q(s,a) to choose an action
  # i.e. a = argmax[a]{ Q(s,a) }
  policyQ = {}
  for a in allActions:
    q = model.predict(s, a)
    policyQ[a] = q
  return policyQ


if __name__ == '__main__':

  grid = negativeGrid(stepCost=-0.1)

  # initialize model
  model = Model()

  # repeat until convergence
  t = 1.0
  t2 = 1.0
  deltas = []
  for i in range(20000):
    if i % 100 == 0:
      t += 10e-3
      t2 += 0.01
    alpha = learningRate / t2
    s = (2, 0)
    grid.setState(s)
    policyQ = getPolicyQ(model, s)
    a = getMaximumFromDict(policyQ)[0]
    a = randomAction(a, eps=0.5/t)
    r = 0
    while not grid.isGameOver():
      oldS = s
      oldR = r
      oldA = a
      r = grid.move(a)
      s = grid.getCurrentState()
      # if it's a terminal state, all Q are 0
      if grid.isTerminalState(s):
        model.theta += alpha*(r - model.predict(oldS, oldA))*model.grad(oldS, oldA)
      else:
        policyQ2 = getPolicyQ(model, s)
        a = getMaximumFromDict(policyQ2)[0]
        a = randomAction(a, eps=0.5/t)
        model.theta += alpha*(r + gamma*model.predict(s, a) - model.predict(oldS, oldA))*model.grad(oldS, oldA)

  policy = {}
  V = {}
  Q = {}
  for s in grid.actions.keys():
    q = getPolicyQ(model, s)
    Q[s] = q
    a, q = getMaximumFromDict(q)
    policy[s] = a
    V[s] = q

  print ("values:")
  printValues(V, grid)
  print ("policy:")
  printPolicy(policy, grid)
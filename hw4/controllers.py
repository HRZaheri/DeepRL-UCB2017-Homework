import numpy as np
from cost_functions import trajectory_cost_fn
import time

class Controller():
	def __init__(self):
		pass

	# Get the appropriate action(s) for this state(s)
	def get_action(self, state):
		pass


class RandomController(Controller):
	def __init__(self, env):
		self.ac = env.action_space

	def get_action(self, state):
		""" YOUR CODE HERE """
		""" Your code should randomly sample an action uniformly from the action space """
		return self.ac.sample()


class MPCcontroller(Controller):
	""" Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
	def __init__(self, 
				 env, 
				 dyn_model, 
				 horizon=5, 
				 cost_fn=None, 
				 num_simulated_paths=10,
				 ):
		self.env = env
		self.dyn_model = dyn_model
		self.horizon = horizon
		self.cost_fn = cost_fn
		self.num_simulated_paths = num_simulated_paths

	def get_action(self, state):
		""" YOUR CODE HERE """
		""" Note: be careful to batch your simulations through the model for speed """
		# 1. Sample
		sampled_acts = [[self.env.action_space.sample() for i in range(self.horizon)] for j in range(self.num_simulated_paths)]
		states = [state for j in range(num_simulated_paths)]
		scores = np.zero(num_simulated_paths)

		# 2. Imagine rollout
		for i in range(self.horizon):
			nstates = self.dyn_model.predict(states, sampled_acts[:, i])
			scores += np.array([self.cost_fn(states[j], sampled_acts[j, i], nstates[j]) for j in range(self.num_simulated_paths)])
			states = nstates

		# 3. Evaluate
		maxi = 0
		for i in range(1, self.num_simulated_paths):
			if scores[i] > scores[maxi]: maxi = i

		# 4. Return
		return sampled_acts[maxi, 0]


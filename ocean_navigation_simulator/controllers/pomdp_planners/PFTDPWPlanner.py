import numpy as np
from dataclasses import dataclass
from typing import Any

from ocean_navigation_simulator.controllers.pomdp_planners.ParticleBelief import (
    ParticleBelief,
)
from ocean_navigation_simulator.controllers.pomdp_planners.GenerativeParticleFilter import (
    GenerativeParticleFilter,
)

@dataclass
class PFTTree:
	# Belief node
	belief_id_to_action_id: dict # belief_id -> [action_id]
	belief_id_to_action_set: dict # belief_id -> set(actions)
	belief_id_num_visits: list # number of visits to belief
	belief_id_to_belief: list # id -> belief

	# Action node
	action_id_num_visits: list # number of visits to action
	action_id_to_q_values: list # Q value function 
	action_id_to_action: list # id -> action (action_ids are unique, even though actions may not be)
	action_id_to_belief_id_reward: dict # action_id -> [(belief_id, reward)] (action_id determines the (b, a) pair)


class PFTDPWPlanner():
	def __init__(
		self, 
		generative_particle_filter: GenerativeParticleFilter, 
		reward_function,
		rollout_policy,
		rollout_value,
		specific_settings: dict
	) -> None:
		# Initialize tree
		self._initialize_tree()

		# Set up helper functions
		self.generative_particle_filter = generative_particle_filter
		self.reward_function = reward_function
		self.rollout_policy = rollout_policy
		self.rollout_value = rollout_value

		# Set up parameters
		self.specific_settings = {
            "num_mcts_simulate": 100,
			"num_particles": 100,
			"max_depth": 100,
			"ucb_factor": 10.0,
			"dpw_k_observations": 4.0,
			"dpw_alpha_observations": 0.25,
			"dpw_k_actions": 3.0,
			"dpw_alpha_actions": 0.25,
			"discount": 0.99,
			"action_space_cardinality": 8,
        } | specific_settings

	def _initialize_tree(self) -> None:
		self.tree = PFTTree(
			belief_id_to_action_id={}, 
			belief_id_to_action_set={},
			belief_id_num_visits=[], 
			belief_id_to_belief=[], 
			action_id_num_visits=[], 
			action_id_to_q_values=[], 
			action_id_to_action=[], 
			action_id_to_belief_id_reward={}
		)

	def _insert_belief_node(self, new_belief: ParticleBelief) -> int:
		# Insert a belief node into the DPW tree
		new_belief_id = len(self.tree.belief_id_to_belief)
		self.tree.belief_id_num_visits.append(0)
		self.tree.belief_id_to_belief.append(new_belief)
		self.tree.belief_id_to_action_id[new_belief_id] = []
		self.tree.belief_id_to_action_set[new_belief_id] = set()

		return new_belief_id

	def _insert_action_node(
		self, 
		belief_id: int, 
		new_action
	) -> int:
		# Insert an action node stemming from a belief node into the DPW tree
		new_action_id = len(self.tree.action_id_to_action)
		self.tree.action_id_num_visits.append(0)
		self.tree.action_id_to_q_values.append(0)
		self.tree.action_id_to_action.append(new_action)
		self.tree.belief_id_to_action_id[belief_id].append(new_action_id)
		self.tree.belief_id_to_action_set[belief_id].add(new_action)

		return new_action_id

	def _insert_transition(
		self, 
		action_id: int, 
		next_belief_id: int, 
		reward: float
	) -> None:
		# Insert a new transition action_id -> next_belief_id (Basically inserting (b, a) -> b', r)
		if action_id not in self.tree.action_id_to_belief_id_reward:
			self.tree.action_id_to_belief_id_reward[action_id] = []
		self.tree.action_id_to_belief_id_reward[action_id].append((next_belief_id, reward))

	def _sample_new_belief_action(self, belief_id: int) -> Any:
		# Generate an action not yet explored for that belief
		return self.generative_particle_filter.sample_new_action(
			self.tree.belief_id_to_action_set[belief_id]
		)

	def _is_terminal(self, belief: ParticleBelief) -> np.array:
		_, terminal_checks = self.reward_function(belief)
		return all(terminal_checks)
	
	def _generate_transition(
		self, 
		belief: ParticleBelief, 
		action
	) -> tuple[ParticleBelief, np.array]:
		# Generate b', r from T(b, a)
		next_belief = self.generative_particle_filter.generate_next_belief(belief, action)
		rewards, _ = self.reward_function(next_belief)
		reward = rewards * next_belief.weights / np.sum(next_belief.weights)

		return next_belief, reward

	def _rollout_action(self, belief: ParticleBelief) -> Any:
		# Sample a random state and get a heuristic action from that
		sampled_state = self.generative_particle_filter.sample_random_state(belief)
		return self.rollout_policy.get_action(sampled_state)
	
	def _rollout_simulation(self, belief_id: int) -> float:
		# Initialize rollout simulation
		current_belief = self.tree.belief_id_to_belief[belief_id].copy()
		steps = 0
		is_terminal = False
		cumulative_reward = 0.0
		discount = 1.0

		# Run rollout simulation until either terminal state or max steps reached
		while steps < self.specific_settings["max_depth"] and not is_terminal:
			# Dynamics forward with rollout action
			action = self._rollout_action(current_belief)
			current_belief, reward = self._generate_transition(current_belief, action)
			cumulative_reward += discount * reward
			
			# Increment quantities
			discount *= self.specific_settings["discount"]
			steps += 1
			is_terminal = self._is_terminal(current_belief)

		return cumulative_reward

	def _estimate_rollout_value(self, belief_id: int) -> float:
		# Return rollout value from function call
		if self.rollout_value is not None:
			raise Exception("DPW Planner: Rollout value not implemented yet")

		# Run rollout simulation starting from belief b
		if self.rollout_policy is not None:
			return self._rollout_simulation(belief_id)

		raise Exception("DPW Planner: No Rollout policy nor value given")
	
	def _get_default_action(self, belief_id: int) -> float:
		# Run rollout policy to get default action
		if self.rollout_policy is None:
			return self._sample_new_belief_action(belief_id)
	
		# Otherwise return rollout action
		return self._rollout_action(self.tree.belief_id_to_belief[belief_id])
	
	def _action_progressive_widening(self, belief_id: int) -> int:
		# Get current_action set and DPW threshold
		current_action_ids = self.tree.belief_id_to_action_id[belief_id]
		action_dpw_threshold = (
			self.specific_settings["dpw_k_actions"] * 
			(self.tree.belief_id_num_visits[belief_id] ** self.specific_settings["dpw_alpha_actions"])
		)

		# If no actions, use heuristic action
		if len(current_action_ids) == 0:
			action = self._get_default_action(belief_id)
			action_id = self._insert_action_node(belief_id, action)
			return action_id

		# If PW condition met, do PW
		if (
			(len(current_action_ids) <= action_dpw_threshold) and 
			(len(current_action_ids) < self.specific_settings["action_space_cardinality"])
		):
			action = self._sample_new_belief_action(belief_id)
			action_id = self._insert_action_node(belief_id, action)
			return action_id
		
		# Otherwise perform UCB among existing actions
		best_ucb_value = -np.inf
		action_id = None
		assert self.tree.belief_id_num_visits[belief_id] > 0
		log_belief_id_num_visits = np.log(self.tree.belief_id_num_visits[belief_id])

		for action_id_candidate in current_action_ids:
			# Get number of visits for action candidate
			num_action_visits = self.tree.action_id_num_visits[action_id_candidate]
			assert num_action_visits > 0
			
			# Calculate UCB criterion value
			q_value = self.tree.action_id_to_q_values[action_id_candidate]
			ucb_value = q_value
			if (self.specific_settings["ucb_factor"] != 0.0):
				ucb_value += self.specific_settings["ucb_factor"] * np.sqrt(
						log_belief_id_num_visits/num_action_visits)

			# Update best UCB value
			if ucb_value > best_ucb_value:
				best_ucb_value = ucb_value
				action_id = action_id_candidate

		# Make sure we get a valid action id
		assert action_id is not None
		action = self.tree.action_id_to_action[action_id]
		
		return action_id
	
	def _state_progressive_widening(
		self, 
		belief_id: int, 
		action_id: int
	) -> tuple[int, float, bool]:
		# Get all transitions from this (belief, action) pair
		current_transitions = self.tree.action_id_to_belief_id_reward.get(action_id, [])
		state_dpw_threshold = (
			self.specific_settings["dpw_k_observations"] * 
			(self.tree.action_id_num_visits[action_id] ** self.specific_settings["dpw_alpha_observations"])
		)
		new_node = False

		# If no state present or PW condition met, do PW
		if len(current_transitions) <= state_dpw_threshold:	
			belief = self.tree.belief_id_to_belief[belief_id]
			action = self.tree.action_id_to_action[action_id]
			next_belief, reward = self._generate_transition(belief, action)
			next_belief_id = self._insert_belief_node(next_belief)
			self._insert_transition(action_id, next_belief_id, reward)
			new_node = True
		
		# Otherwise pick a belief node at random
		else:
			next_belief_id, reward = current_transitions[np.random.choice(range(len(current_transitions)))]
		
		return next_belief_id, reward, new_node
	
	def get_best_action(self, belief: ParticleBelief) -> Any:
		# Builds a DPW tree and returns the best next action
		# Construct the DPW tree
		self._initialize_tree()
		initial_belief_id = self._insert_belief_node(belief)

		# Plan with the tree by querying the tree for num_mcts_simulate number of times
		for _ in range(self.specific_settings["num_mcts_simulate"]):
			self.mcts_simulate(initial_belief_id, self.specific_settings["max_depth"])

		# Find the best action from the root node
		best_q_value = -np.inf
		best_action_id = None
		for action_id in self.tree.belief_id_to_action_id[initial_belief_id]:
			if self.tree.action_id_to_q_values[action_id] > best_q_value:
				best_q_value = self.tree.action_id_to_q_values[action_id]
				best_action_id = action_id
		
		assert best_action_id is not None
		return self.tree.action_id_to_action[best_action_id]

	def mcts_simulate(self, belief_id: int, current_depth: int) -> float:
		# Check if current_depth == 0 (full depth) or belief is terminal
		belief = self.tree.belief_id_to_belief[belief_id]
		if current_depth == 0:
			return self._estimate_rollout_value(belief_id)
		elif self._is_terminal(belief):
			return 0.0

		# Double Progressive Widening
		action_id = self._action_progressive_widening(belief_id)
		next_belief_id, reward, new_node = self._state_progressive_widening(belief_id, action_id)

		# Simulate recursively
		if new_node:
			q_value = reward + self.discount * self._estimate_rollout_value(next_belief_id)
		else:
			q_value = reward + self.discount * self.mcts_simulate(next_belief_id, current_depth - 1)

		# Update the counters & quantities
		self.tree.belief_id_num_visits[belief_id] += 1
		self.tree.action_id_num_visits[action_id] += 1
		self.tree.action_id_to_q_values[action_id] += (
			(q_value - self.tree.action_id_to_q_values[action_id]) / 
			self.tree.action_id_num_visits[action_id]
		)

		return q_value
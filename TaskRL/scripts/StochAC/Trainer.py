#!/usr/bin/env python
from headers import *
from Memory import ReplayMemory
from Policy import ActorCriticModel
from Transitions import Transition

class Trainer():

	def __init__(self, policy=None, environment=None, memory=None, args=None):

		self.policy = policy
		self.environment = environment
		self.memory = memory
		self.args = args

		self.max_timesteps = 500
		
		self.initial_epsilon = 0.5
		self.final_epislon = 0.05
		self.anneal_epochs = 100
		self.epsilon_anneal_rate = (self.initial_epsilon-self.final_epislon)/self.anneal_epochs
	
	def initialize_memory(self):

		# Number of initial transitions needs to be less than memory size. 
		self.initial_transitions = 5000        
		# transition must have: obs, action taken, terminal?, reward, success, next_state 

		# While memory isn't full:
		#while self.memory.check_full()==0:

		# While number of transitions is less than initial_transitions.
		while self.memory.memory_len<self.initial_transitions:
			
			# Start a new episode. 
			counter=0
			state = self.environment.reset()
			terminal = False

			while counter<self.max_timesteps and self.memory.memory_len<self.initial_transitions and not(terminal):

				# Put in new transitions. 
				action = self.environment.action_space.sample()

				# Take a step in the environment. 
				next_state, onestep_reward, terminal, success = self.environment.step(action)

				# Store in instance of transition class. 
				new_transition = Transition(state,action,next_state,onestep_reward,terminal,success)

				# Append new transition to memory. 
				self.memory.append_to_memory(new_transition)

				# Copy next state into state. 
				state = copy.deepcopy(next_state)

				# Increment counter. 
				counter+=1

	def meta_training(self):

		# Interacting with the environment: 
		# For initialize_memory, just randomly sample actions from the action space, use env.action_space.sample()










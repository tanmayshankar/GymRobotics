#!/usr/bin/env python
from headers import *
from Memory import ReplayMemory
from Policy import DAggerPolicy
from Transitions import Transition

class Trainer():

	def __init__(self, sess=None, policy=None, environment=None, memory=None, args=None):

		# self.policy = policy
		self.PolicyModel = policy
		self.environment = environment
		self.memory = memory
		self.sess = sess
		self.args = args

		self.max_timesteps = 2000
		
		self.initial_epsilon = 0.5
		self.final_epsilon = 0.05
		self.test_epsilon = 0.
		self.anneal_iterations = 100000
		self.epsilon_anneal_rate = (self.initial_epsilon-self.final_epsilon)/self.anneal_iterations

		# Beta value determines mixture of expert and learner. 
		# Beta 1 means completely expert. 
		# Beta 0 means completely learner.
		self.initial_beta = 1.
		self.final_beta = 0.5
		self.test_beta = 0.
		self.beta_anneal_rate = (self.initial_beta-self.final_beta)/self.anneal_iterations

		self.step_size = 1.

		# Training limits. 
		self.number_episodes = 10000

		# Batch size, observation size
		self.batch_size = 25
		self.gamma = 0.99
		self.save_every = 5000
	
		print("Setup Trainer Init.")

	def initialize_memory(self):

		# Number of initial transitions needs to be less than memory size. 
		self.initial_transitions = 5000        
		# transition must have: obs, action taken, terminal?, reward, success, next_state 

		# While memory isn't full:
		#while self.memory.check_full()==0:
		self.max_timesteps = 200
		print("Starting Memory Burn In.")
		self.set_parameters(0)

		# While number of transitions is less than initial_transitions.
		while self.memory.memory_len<self.initial_transitions:
			
			# Start a new episode. 
			counter=0
			state = self.environment.reset()
			terminal = False

			while counter<self.max_timesteps and self.memory.memory_len<self.initial_transitions and not(terminal):
			
				# Put in new transitions. 
				# action = self.environment.action_space.sample()
				# action = self.step_size*self.select_action_beta(state)
				action, expert_action = self.select_action_beta(state)
				# print(action)

				# Take a step in the environment. 
				next_state, onestep_reward, terminal, success = self.environment.step(action)

				# If render flag on, render environment.
				if self.args.render: 
					self.environment.render()				

				# Store in instance of transition class. 
				new_transition = Transition(state,expert_action,next_state,onestep_reward,terminal,success)

				# Append new transition to memory. 
				self.memory.append_to_memory(new_transition)

				# Copy next state into state. 
				state = copy.deepcopy(next_state)

				# Increment counter. 
				counter+=1

		self.max_timesteps = 2000
		print("Memory Burn In Complete.")

	def set_parameters(self,iteration_number):
		# Setting parameters.
		if self.args.train:
			if iteration_number<self.anneal_iterations:
				self.annealed_epsilon = self.initial_epsilon-iteration_number*self.epsilon_anneal_rate
				self.annealed_beta = self.initial_beta-iteration_number*self.beta_anneal_rate			
			else:
				self.annealed_epsilon = self.final_epsilon
				self.annealed_beta = self.final_beta				
		else:
			self.annealed_epsilon = self.test_epsilon	
			self.annealed_beta = self.test_beta

	def assemble_state(self,state):
		# Take state from envrionment (achieved goal,desired goal, and observation blah blah)
		# Assemble into one vector. 
		# (USED FOR FORWARD PASS, AND FEEDING FROM MEMORY)
		return npy.concatenate((state['achieved_goal'],state['desired_goal'],state['observation']))

	def select_action_from_expert(self, state):
		action = npy.zeros((4))
		action[:3] = state['desired_goal']-state['achieved_goal']
		return action

	def select_action_from_policy(self, state):
		# Greedy selection of action from policy. 
		# Here we have a continuous stochastic policy, parameterized as a Gaussian policy. 
		# Just simply select the mean of the Gaussian as the action? 

		assembled_state = npy.reshape(self.assemble_state(state),(1,self.PolicyModel.input_dimensions))

		return self.sess.run(self.PolicyModel.predicted_action,
			feed_dict={self.PolicyModel.input: assembled_state})[0]
		
	def select_action(self, state):
		# Select an action either from the policy or randomly. 
		random_probability = npy.random.random()				

		# If less than epsilon. 
		if random_probability < self.annealed_epsilon:
			action = self.environment.action_space.sample()			
		else:
			# Greedily select action from policy. 
			action = self.select_action_from_policy(state)	

		return action

	def select_action_beta(self, state):
		# Select an action either from the policy or randomly. 
		random_probability = npy.random.random()				

		expert_action = self.select_action_from_expert(state)
		action = npy.zeros((4))
		# If less than beta, use expert. 
		if random_probability < self.annealed_beta:
			action = expert_action
		else:
			# Greedily select action from policy. 
			action = self.select_action_from_policy(state)				

		return action, expert_action

	def policy_update(self, iter_num):
		# Must construct target Q value here
		# Remember, 1 step TD here, not the MC version.

		# Batch
		self.batch_states = npy.zeros((self.batch_size, self.PolicyModel.input_dimensions))
		self.batch_actions = npy.zeros((self.batch_size, self.PolicyModel.output_dimensions))		
		# self.batch_next_states = npy.zeros((self.batch_size, self.ACModel.actor_network.input_dimensions))

		# self.batch_target_Qvalues = npy.zeros((self.batch_size,1))
		# self.batch_onestep_rewards = npy.zeros((self.batch_size,1))
		# self.batch_terminal = npy.zeros((self.batch_size,1),dtype=int)

		# Sample from memory
		indices = self.memory.sample_batch()

		for k in range(len(indices)):

			self.batch_states[k] = self.assemble_state(self.memory.memory[indices[k]].state)
			# self.batch_next_states[k] = self.assemble_state(self.memory.memory[indices[k]].next_state)
			self.batch_actions[k] = self.memory.memory[indices[k]].action
			# self.batch_onestep_rewards[k] = self.memory.memory[indices[k]].onestep_reward
			# self.batch_terminal[k] = self.memory.memory[indices[k]].terminal
	
		# Update Critic and Actor
		merged, _ = self.sess.run([self.PolicyModel.merged_summaries, self.PolicyModel.train_actor],
			feed_dict={self.PolicyModel.input: self.batch_states,
						self.PolicyModel.target_action: self.batch_actions})

		self.PolicyModel.tf_writer.add_summary(merged, iter_num)		

	def meta_training(self):
		# Interacting with the environment: 
		# For initialize_memory, just randomly sample actions from the action space, use env.action_space.sample()

		self.initialize_memory()
		# Train for at least these many episodes. 

		print("Starting Main Training Procedure.")
		meta_counter = 0

		for e in range(self.number_episodes):

			# Maintain coujnter to keep track of updating the policy regularly. 
			# And to check if we are exceeding max number of timesteps .
			counter = 0			

			# Reset environment.
			state = self.environment.reset()
			terminal = False
						
			# Within each episode, just keep going until you terminate or we reach max number of timesteps. 
			while not(terminal) and counter<self.max_timesteps:

				self.set_parameters(meta_counter)

				# SAMPLE ACTION FROM POLICY(STATE)				
				# action = self.step_size*self.select_action_beta(state)
				action, expert_action = self.select_action_beta(state)

				# TAKE STEP WITH ACTION
				next_state, onestep_reward, terminal, success = self.environment.step(action)				
				# embed()
				# If render flag on, render environment.
				if self.args.render: 
					self.environment.render()				

				# STORE TRANSITION IN MEMORY WITH EXPERT ACTION: 
				new_transition = Transition(state,expert_action,next_state,onestep_reward,terminal,success)
				self.memory.append_to_memory(new_transition)

				# UPDATE POLICY (need to decide whether to do thios at every step, or less frequently).
				self.policy_update(counter)

				state = copy.deepcopy(next_state)

				# Increment counter. 
				counter+=1
				meta_counter+=1 
				# If counter % save_
				if meta_counter%self.save_every==0 and self.args.train:
					self.PolicyModel.save_model(meta_counter)
					print("Reached Iteration",meta_counter)

			# embed()





		








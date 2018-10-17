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

		self.number_epochs = 500
		self.anneal_iterations = self.number_epochs
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
		self.save_every = 1
	
		print("Setup Trainer Init.")

	def initialize_memory(self):

		# Now we are going to initialize the memory with a set number of demonstrations. 
		self.number_demonstrations = 500
		# transition must have: obs, action taken, terminal?, reward, success, next_state 

		self.max_timesteps = 200
		self.number_episodes = 0
		print("Starting Memory Burn In.")
		self.set_parameters(0)

		# For INITIALIZING MEMORY ALONE: Set the beta value to 1. - collect expert demonstrations.
		self.annealed_beta = 1.

		# While number of episodes less than number of demonstrations.
		while self.number_episodes<self.number_demonstrations:
	
			# Start a new episode. 
			counter = 0
			state = self.environment.reset()
			terminal = False

			episode = []

			while counter<self.max_timesteps and not(terminal):
		
				# Retrieve action - with beta ==1, this will return expert, expert. 
				action, expert_action = self.select_action_beta(state)

				# Take a step in the environment. 
				next_state, onestep_reward, terminal, success = self.environment.step(action)

				# If render flag on, render environment.
				if self.args.render: 
					self.environment.render()				

				# Store in instance of transition class. 
				# Remember, here we are adding EXPERT action to the memory. 
				new_transition = Transition(state,expert_action,next_state,onestep_reward,terminal,success)

				# Do not append transition to memory yet. 
				# Append to episode, then append episode.
				episode.append(new_transition)
				
				# Copy next state into state. 
				state = copy.deepcopy(next_state)

				# Increment counter. 
				counter+=1

			# Append new episode to memory. 
			self.memory.append_to_memory(episode)
			self.number_episodes += 1

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
		# print("THE EXPERT IS HERE, have no fear!")
		action = npy.zeros((4))
		action[:3] = state['desired_goal']-state['achieved_goal']
		return action

	def select_action_from_policy(self, state):
		# Greedy selection of action from policy. 
		# Here we have a continuous stochastic policy, parameterized as a Gaussian policy. 
		# Just simply select the mean of the Gaussian as the action? 

		assembled_state = npy.reshape(self.assemble_state(state),(1,self.PolicyModel.input_dimensions))

		return self.sess.run(self.PolicyModel.sample_action,
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
		action = npy.zeros((self.PolicyModel.output_dimensions))
		# If less than beta, use expert. 
		if random_probability < self.annealed_beta:
			action = expert_action
		else:
			# Greedily select action from policy. 
			action = self.select_action_from_policy(state)				

		return action, expert_action

	def policy_update(self, iter_num, episode_index):

		episode = copy.deepcopy(self.memory.memory[episode_index])
		self.batch_size = len(episode)

		if not(self.batch_size):
			embed()

		# Compute likelihood ratios for every transition in the episode.
		self.batch_states = npy.zeros((self.batch_size, self.PolicyModel.input_dimensions))
		self.batch_actions = npy.zeros((self.batch_size, self.PolicyModel.output_dimensions))		
		self.batch_likelihood_ratios = npy.ones((self.batch_size,1))
		
		for kx in range(self.batch_size):

			# Instead of accessing memory.memory[k].state, use episode[k].state.
			self.batch_states[kx] = self.assemble_state(episode[kx].state)
			# This should be the expert action.
			self.batch_actions[kx] = episode[kx].action
		

		# IN THE DEMO_BC CODE, WE ARE SETTING BATCH likelihood RATIOS TO 1. 
		# # PROBABILITY OF THE ACTION
		# self.batch_probabilities = self.sess.run(self.PolicyModel.action_likelihood,
		# 	feed_dict={self.PolicyModel.sampled_action: self.batch_actions,
		# 				self.PolicyModel.input: self.batch_states})

		# # Run forward the policy on batch_state[kx] and batch_actions[kx] to get the probability. 
		# # Use this probability for likelihood ratio. 				
		
		# # Set likelihood ratios. 		
		# for kx in range(self.batch_size):
		# 	# Iteratively set the likelihood ratios from timestep 1 to T. 
		# 	self.batch_likelihood_ratios[kx] = self.batch_likelihood_ratios[kx-1]*self.batch_probabilities[kx]

		merged , _ = self.sess.run([self.PolicyModel.merged_summaries, self.PolicyModel.train_actor],
			feed_dict={self.PolicyModel.sampled_action: self.batch_actions,
						self.PolicyModel.input: self.batch_states,
						self.PolicyModel.likelihood_ratio_weight: self.batch_likelihood_ratios})

		self.PolicyModel.tf_writer.add_summary(merged, iter_num)		

	def meta_training(self):
		# Interacting with the environment: 
		if self.args.train:
			self.initialize_memory()
		
		print("Starting Main Training Procedure.")
		meta_counter = 0
		self.set_parameters(meta_counter)

		# Now measuring things in terms of epochs. 
		for	e in range(self.number_epochs):
			print("Epoch Number: ",e)
			# Maintain coujnter to keep track of updating the policy regularly. 
			# And to check if we are exceeding max number of timesteps .
			counter = 0

			demo_index_list = npy.arange(self.number_demonstrations)
			if self.args.train:
				npy.random.shuffle(demo_index_list)

			# SHOULD NOT NEED TO TOUCH THE ENVIRONMENT NOW! 
			# For each episode in the memory:
			for j in range(self.memory.num_mem_episodes):

				# print("Episode # : ",j)

				# Select which demo index to train on.
				episode_index = demo_index_list[j]

				if len(self.memory.memory[episode_index])==0:
					embed()

				if self.args.train:
					self.policy_update(meta_counter, episode_index)

				# Increment counter. 				
				counter+=1
				meta_counter+=1

				# If counter % save_
				# if counter%self.save_every==0 and self.args.train:
			self.PolicyModel.save_model(e)				
			
	def meta_testing(self):
		
		self.set_parameters(0)

		# Number test episodes.
		self.number_test_episodes = 50

		for e in range(self.number_test_episodes):

			# Reset environment.
			state = self.environment.reset()
			terminal = False
			eps_reward = 0.

			while not(terminal):

				# Select action with policy.
				action = self.select_action(state)
				# Take step.
				next_state, onestep_reward, terminal, success = self.environment.step(action)

				eps_reward += copy.deepcopy(onestep_reward)
				# If render flag on, render environment.
				if self.args.render: 
					self.environment.render()				

			print("Episode: ",e," Reward: ",eps_reward, "Terminal: ",terminal)



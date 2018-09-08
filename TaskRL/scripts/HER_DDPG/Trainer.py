#!/usr/bin/env python
from headers import *
from Memory import ReplayMemory
from Policy import ActorCriticModel
from Transitions import Transition

class Trainer():

	def __init__(self, sess=None, policy=None, environment=None, memory=None, args=None):

		# self.policy = policy
		self.ACModel = policy
		self.environment = environment
		self.memory = memory
		self.sess = sess
		self.args = args

		self.max_timesteps = 2000
		
		self.initial_epsilon = 0.6
		self.final_epsilon = 0.1
		self.test_epsilon = 0.
		self.anneal_iterations = 1000000
		self.epsilon_anneal_rate = (self.initial_epsilon-self.final_epsilon)/self.anneal_iterations

		# Beta value determines mixture of expert and learner. 
		# Beta 1 means completely expert. 
		# Beta 0 means completely learner.
		self.initial_beta = 1.
		self.final_beta = 0.5
		self.test_beta = 0.
		self.beta_anneal_rate = (self.initial_beta-self.final_beta)/self.anneal_iterations

		# Training limits. 
		if self.args.env=='MountainCarContinuous-v0':
			self.number_episodes = 10000
		# elif self.args.env=='InvertedPendulum-v2':
		else:
			self.number_episodes = 100000

		# Batch size, observation size
		self.batch_size = 25
		self.gamma = 0.99
		self.save_every = 10000
	
		print("Setup Trainer Init.")

	def initialize_memory(self):

		# Number of initial transitions needs to be less than memory size. 
		self.initial_transitions = 5000        
		# transition must have: obs, action taken, terminal?, reward, success, next_state 

		# While memory isn't full:
		#while self.memory.check_full()==0:
		self.max_timesteps = 500
		print("Starting Memory Burn In.")
		self.set_parameters(0)

		# While number of transitions is less than initial_transitions.
		while self.memory.memory_len<self.initial_transitions:
			
			# Start a new episode. 
			counter=0
			state = self.environment.reset()
			terminal = False

			# Create a list of transitions that represnt the episode. 
			episode_transition_list = []

			while counter<self.max_timesteps and self.memory.memory_len<self.initial_transitions and not(terminal):
			
				# Put in new transitions. 
				action = self.environment.action_space.sample()
				# action = self.select_action_beta(state)

				# Take a step in the environment. 
				next_state, onestep_reward, terminal, success = self.environment.step(action)

				# # If render flag on, render environment.
				# if self.args.render: 
				# 	self.environment.render()				

				# Store in instance of transition class. 
				new_transition = Transition(state,action,next_state,onestep_reward,terminal,success)

				episode_transition_list.append(new_transition)

				# DO NOT append to memory now. 
				# Change goal to the goal finally achieved at the end of the episode. 

				# # Append new transition to memory. 
				# self.memory.append_to_memory(new_transition)

				# Copy next state into state. 
				state = copy.deepcopy(next_state)

				# Increment counter. 
				counter+=1

			# Now that the episode is done, 
			# Change all the "Desired goal" variables to the goal actually achieved. 
			achieved_goal = copy.deepcopy(state['achieved_goal'])

			# Copy the Actually achieved goal as the desired goal to all transitions in the memory. 
			# Now append the transiiton into the memory. 
			for k in range(len(episode_transition_list)):
				episode_transition_list[k].state['desired_goal'] = copy.deepcopy(achieved_goal)
				episode_transition_list[k].next_state['desired_goal'] = copy.deepcopy(achieved_goal)

				# Append into memory. 
				self.memory.append_to_memory(episode_transition_list[k])

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
		if self.args.env=='FetchReach-v0' or self.args.env=='FetchPush-v0':
			temp = npy.concatenate((state['achieved_goal'],state['desired_goal'],state['observation']))
			return npy.reshape(temp,(1,self.ACModel.actor_network.input_dimensions))
		else:
			return npy.reshape(state,(1,self.ACModel.actor_network.input_dimensions))
	# def select_action_from_expert(self, state):
	# 	action = npy.zeros((4))
	# 	action[:3] = state['desired_goal']-state['achieved_goal']
	# 	return action

	def select_action_from_policy(self, state):
		# Greedy selection of action from policy. 
		# Here we have a continuous stochastic policy, parameterized as a Gaussian policy. 
		# Just simply select the mean of the Gaussian as the action? 

		assembled_state = self.assemble_state(state)

		return self.sess.run(self.ACModel.actor_network.predicted_action,
			feed_dict={self.ACModel.actor_network.input: assembled_state})[0]
		
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

	# def select_action_beta(self, state):
	# 	# Select an action either from the policy or randomly. 
	# 	random_probability = npy.random.random()				

	# 	# If less than beta. 
	# 	if random_probability < self.annealed_beta:
	# 		action = self.select_action_from_expert(state)
	# 	else:
	# 		# Greedily select action from policy. 
	# 		action = self.select_action_from_policy(state)				

	# 	return action

	def policy_update(self, iter_num):
		# Must construct target Q value here
		# Remember, 1 step TD here, not the MC version.

		# Batch
		self.batch_states = npy.zeros((self.batch_size, self.ACModel.actor_network.input_dimensions))
		self.batch_next_states = npy.zeros((self.batch_size, self.ACModel.actor_network.input_dimensions))
		self.batch_actions = npy.zeros((self.batch_size, self.ACModel.actor_network.output_dimensions))		
		self.batch_actor_actions = npy.zeros((self.batch_size, self.ACModel.actor_network.output_dimensions))		

		self.batch_target_Qvalues = npy.zeros((self.batch_size,1))
		self.batch_onestep_rewards = npy.zeros((self.batch_size,1))
		self.batch_terminal = npy.zeros((self.batch_size,1),dtype=int)

		# Sample from memory
		indices = self.memory.sample_batch()
		# indices = self.memory.retrieve_batch()

		for k in range(len(indices)):
			self.batch_states[k] = self.assemble_state(self.memory.memory[indices[k]].state)
			self.batch_next_states[k] = self.assemble_state(self.memory.memory[indices[k]].next_state)
			self.batch_actions[k] = self.memory.memory[indices[k]].action
			self.batch_onestep_rewards[k] = self.memory.memory[indices[k]].onestep_reward
			self.batch_terminal[k] = self.memory.memory[indices[k]].terminal

		# Computing the target Q values: 
		# Set target Q values - will need forward prop of critic for Q(s',a').
		# Don't use Q(s',a'), instead, use Q(s', pi(s')). (DDPG, DPG,) etc.,)

		# Critic network's input action_taken is either a placeholder for using actions stored in the memory, 
		# or uses the actor network's predicted action.
		# In our earlier DDPG implementation, we only needed the actor network's predicted action,
		# Because we never forward propagated the critic network with a selected action.
		# In AC-OffPG, we used .. placeholders. 

		next_actions = self.sess.run(self.ACModel.actor_network.predicted_action,
			feed_dict={self.ACModel.actor_network.input: self.batch_next_states})

		# First evaluate critic estimates of Q(s',pi(s')).
		critic_estimates = self.sess.run(self.ACModel.critic_network.predicted_Qvalue, 
			feed_dict={self.ACModel.critic_network.input: self.batch_next_states,
						self.ACModel.critic_network.action_taken: next_actions})
						# self.ACModel.actor_network.input: self.batch_next_states})

		# Next construct target Q as r+gamma Q(s',pi(s')).
		self.batch_target_Qvalues = self.batch_onestep_rewards+self.gamma*(1-self.batch_terminal)*critic_estimates
		# self.batch_target_Qvalues = self.batch_onestep_rewards+self.gamma*critic_estimates
		# embed()	
		# Update Critic and Actor
		_ = self.sess.run(self.ACModel.train_critic,
			feed_dict={self.ACModel.critic_network.input: self.batch_states,
						# THIS IS ONLY BATCH ACTIONS FOR TRAINING CRITIC. 
						# FOR TRAINING ACTOR, SHOULD BE ACTOR's ACTION. 
						self.ACModel.critic_network.action_taken: self.batch_actions,
						self.ACModel.target_Qvalue: self.batch_target_Qvalues})

		# Retrieve batch_actor_actions. 
		self.batch_actor_actions = self.sess.run(self.ACModel.actor_network.predicted_action,
			feed_dict={self.ACModel.actor_network.input: self.batch_states})

		_ = self.sess.run(self.ACModel.train_actor, 
			feed_dict={self.ACModel.critic_network.input: self.batch_states,
						self.ACModel.actor_network.input: self.batch_states,
						self.ACModel.critic_network.action_taken: self.batch_actor_actions})

		merged = self.sess.run(self.ACModel.merged_summaries, 
			feed_dict={self.ACModel.critic_network.input: self.batch_states,
						self.ACModel.actor_network.input: self.batch_states,
						self.ACModel.critic_network.action_taken: self.batch_actions,
						self.ACModel.target_Qvalue: self.batch_target_Qvalues})

		# NOTES: 
		# 1) The Actor is being updated with Current critic's estimate of Q(s,a). 
		# 		This means we aren't feeding in Q(s,a) to be multiplied with the log prob. 
		# 		The model does this automatically by forward proppping critic. 

		# 2) The Critic uses r+\gamma Q(s',pi(s')) as the target, which is precomputed. 
		# 		Precomputation removes gradients with respect to Q(s',pi(s')) that actually depend on critic parameters. 
		self.ACModel.tf_writer.add_summary(merged, iter_num)		

	def meta_training(self):
		# Interacting with the environment: 
		# For initialize_memory, just randomly sample actions from the action space, use env.action_space.sample()
		if self.args.train:
			self.initialize_memory()		
			# self.initial_epsilon = 0.8
		# Train for at least these many episodes. 
		embed()
		print("Starting Main Training Procedure.")
		meta_counter = 0
		episode_counter = 0
		self.set_parameters(meta_counter)

		for e in range(self.number_episodes):

			# Maintain coujnter to keep track of updating the policy regularly. 
			# And to check if we are exceeding max number of timesteps .
			counter = 0			

			# Reset environment.
			state = self.environment.reset()
			terminal = False
			eps_reward = 0.

			episode_transition_list = []

			# Within each episode, just keep going until you terminate or we reach max number of timesteps. 
			while not(terminal) and counter<self.max_timesteps:

				self.set_parameters(meta_counter)

				# SAMPLE ACTION FROM POLICY(STATE)			
				action = self.select_action(state)
				# action = self.select_action_beta(state)
				# print(action)
				# TAKE STEP WITH ACTION
				next_state, onestep_reward, terminal, success = self.environment.step(action)				

				eps_reward += copy.deepcopy(onestep_reward)

				if self.args.env=='MountainCarContinuous-v0':
					print("Special Mountain Car Termination")
					if next_state[0]>0.5:
						terminal = True
						success = True
					else:
						if terminal==True:
							terminal = False


				# If render flag on, render environment.
				if self.args.render: 
					self.environment.render()				

				if self.args.train:
					# STORE TRANSITION IN MEMORY. 
					new_transition = Transition(state,action,next_state,onestep_reward,terminal,success)
					# self.memory.append_to_memory(new_transition)

					episode_transition_list = []

					# UPDATE POLICY (need to decide whether to do thios at every step, or less frequently).	
					self.policy_update(meta_counter)
				else: 
					print(action)

				state = copy.deepcopy(next_state)

				# Increment counter. 
				counter+=1
				meta_counter+=1 
				# If counter % save_
				if meta_counter%self.save_every==0 and self.args.train:
					self.ACModel.save_model(meta_counter)
					print("Reached Iteration",meta_counter)

			achieved_goal = copy.deepcopy(state['achieved_goal'])	

			for k in range(len(episode_transition_list)):
				episode_transition_list[k].state['desired_goal'] = copy.deepcopy(achieved_goal)
				episode_transition_list[k].next_state['desired_goal'] = copy.deepcopy(achieved_goal)

				self.memory.append_to_memory(episode_transition_list[k])

			# embed()

			print("Episode: ",episode_counter," Reward: ",eps_reward, " Counter:", counter, terminal)
			episode_counter +=1 




		








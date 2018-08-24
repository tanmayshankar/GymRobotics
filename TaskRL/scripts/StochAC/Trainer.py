#!/usr/bin/env python
from headers import *
from Memory import ReplayMemory
from Policy import ActorCriticModel
from Transitions import Transition

class Trainer():

	def __init__(self, ACpolicy=None, environment=None, memory=None, args=None):

		# self.policy = policy
		self.ACModel = ACpolicy
		self.environment = environment
		self.memory = memory
		self.args = args

		self.max_timesteps = 500
		
		self.initial_epsilon = 0.5
		self.final_epislon = 0.05
		self.anneal_iterations = 100000
		self.epsilon_anneal_rate = (self.initial_epsilon-self.final_epislon)/self.anneal_epochs

		# Training limits. 
		self.number_episodes = 10000

		# Batch size, observation size
		self.batch_size = 25
		self.gamma = 0.99
	
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

	def set_parameters(self,iteration_number):
		# Setting parameters.
		if self.args.train:
			if iteration_number<self.anneal_epochs:
				self.annealed_epsilon = self.initial_epsilon-iteration_number*self.anneal_epsilon_rate
				# self.annealed_beta = self.initial_beta-e*self.beta_anneal_rate			
			else:
				self.annealed_epsilon = self.final_epsilon
				# self.annealed_beta = self.final_beta				
		else:
			self.annealed_epsilon = self.test_epsilon	
			# self.annealed_beta = self.final_beta

	def assemble_state(self,state):
		# Take state from envrionment (achieved goal,desired goal, and observation blah blah)
		# Assemble into one vector. 
		# (USED FOR FORWARD PASS, AND FEEDING FROM MEMORY)
		return npy.concatenate((state['achieved_goal'],state['desired_goal'],state['observation']))

	def select_action_from_policy(self, state):
		# Greedy selection of action from policy. 
		# Here we have a continuous stochastic policy, parameterized as a Gaussian policy. 
		# Just simply select the mean of the Gaussian as the action? 
		return self.sess.run(self.ACModel.actor_network.normal_means, 
			feed_dict={self.ACModel.actor_network.input: self.assemble_state(state) })
		
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

	def policy_update(self, iter_num):
		# Must construct target Q value here
		# Remember, 1 step TD here, not the MC version.

		# Batch
		self.batch_states = npy.zeros((self.batch_size, self.ACModel.actor_network.input_dimensions))
		self.batch_actions = npy.zeros((self.batch_size, self.ACModel.actor_network.output_dimensions))		
		self.batch_next_states = npy.zeros((self.batch_size, self.ACModel.actor_network.input_dimensions))

		self.batch_target_Qvalues = npy.zeros((self.batch_size,1))
		self.batch_onestep_rewards = npy.zeros((self.batch_size,1))
		self.batch_terminal = npy.zeros((self.batch_size,1),dtype=int)

		# Sample from memory
		indices = self.memory.sample_batch()

		for k in range(len(indices)):

			self.batch_states[k] = self.assemble_state(self.memory.memory[indices[k]].state)
			self.batch_next_states[k] = self.assemble_state(self.memory.memory[indices[k]].next_state)
			self.batch_actions[k] = self.memory.memory[indices[k]].action
			self.batch_onestep_rewards[k] = self.memory.memory[indices[k]].onestep_reward
			self.batch_terminal[k] = self.memory.memory[indices[k]].terminal

		# Computing the target Q values: 
		# Set target Q values - will need forward prop of critic for Q(s',a').
		# Don't use Q(s',a'), instead, use Q(s', pi(s')). (DDPG, DPG,) etc.,)

		# First evaluate critic estimates of Q(s',pi(s')).
		critic_estimates = self.sess.run(self.ACModel.critic_network.predicted_Qvalue, 
			feed_dict={self.ACModel.critic_network.input: self.batch_next_states,
						self.ACModel.critic_network.action_taken: self.ACModel.actor_network.sample_action,
						self.ACModel.actor_network.input: self.batch_next_states})

		# Next construct target Q as r+gamma Q(s',pi(s')).
		self.batch_target_Qvalues = self.batch_onestep_rewards+self.gamma*critic_estimates
		
		# Update Critic and Actor
		merged, _, _ = self.sess.run([self.ACModel.merged_summaries, self.ACModel.train_critic, self.ACModel.train_actor],
			feed_dict={self.ACModel.critic_network.input: self.batch_states,
						self.ACModel.critic_network.action_taken: self.batch_actions,
						self.ACModel.actor_network.sampled_action: self.batch_actions,
						self.ACModel.actor_network.input: self.batch_states, 
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

		# Train for at least these many episodes. 
		for e in range(self.number_episodes):

			# Maintain coujnter to keep track of updating the policy regularly. 
			# And to check if we are exceeding max number of timesteps .
			counter = 0

			# Reset environment.
			state = self.environment.reset()
			terminal = False
			
			# Within each episode, just keep going until you terminate or we reach max number of timesteps. 
			while not(terminal) and counter<self.max_timesteps:

				self.set_parameters(counter)

				# SAMPLE ACTION FROM POLICY(STATE)				
				self.select_action(state)

				# TAKE STEP WITH ACTION
				next_state, onestep_reward, terminal, success = self.environment.step(action)				

				# If render flag on, render environment.
				if self.args.render: 
					self.environment.render()				

				# STORE TRANSITION IN MEMORY. 
				new_transition = Transition(state,action,next_state,onestep_reward,terminal,success)
				self.memory.append_to_memory(new_transition)

				# UPDATE POLICY (need to decide whether to do thios at every step, or less frequently).
				self.policy_update(counter)

				state = copy.deepcopy(next_state)

				# Increment counter. 
				counter+=1






		








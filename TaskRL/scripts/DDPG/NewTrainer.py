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

    def environment_step(self):

        

    def initialize_memory(self):




#!/usr/bin/env python
from headers import *

class ReplayMemory():

	def __init__(self, memory_size=100000):
		
		# Implementing the memory as a list of transitions. 
		# This acts as a queue. 
		self.memory = []

		# Accessing the memory with indices should be constant time, so it's okay to use a list. 
		# Not using a priority either. 
		self.memory_len = 0
		self.memory_size = memory_size

		print("Setup Memory.")

	def append_to_memory(self, transition):

		if self.check_full():
			# Remove first transition in the memory (queue).
			self.memory.pop(0)
			# Now push the transition to the end of hte queue. 
			self.memory.append(transition)
		else:
			self.memory.append(transition)

		self.memory_len+=1

	def sample_batch(self, batch_size=25):

		self.memory_len = len(self.memory)

		indices = npy.random.randint(0,high=self.memory_len,size=(batch_size))

		return indices

	def retrieve_batch(self, batch_size=25):
		# self.memory_len = len(self.memory)

		return npy.arange(0,batch_size)

	def check_full(self):

		self.memory_len = len(self.memory)

		if self.memory_len<self.memory_size:
			return 0 
		else:
			return 1 






#!/usr/bin/env python
from headers import *

class PolicyNetwork():

	def __init__(self, input_dimensions, output_dimensions, num_layers=4):

		self.input_dimensions = input_dimensions
		self.output_dimensions = output_dimensions
		self.num_layers = num_layers

	def initialize_base_model(self, sess, model_file=None, to_train=None):

		# Initializing the session.
		self.sess = sess
		self.to_train = to_train

		# Set number of hidden units. 
		self.hidden_units = 30*npy.ones((self.num_layers+1),dtype=int)        
		self.hidden_units[-1] = self.output_dimensions

		# Input placeholder.
		self.input = tf.placeholder(tf.float32,shape=[None,self.input_dimensions],name='input')

		# Defining hidden layers of MLP. 
		self.hidden_layers = [[] for i in range(self.num_layers)]        
		self.hidden_layers[0] = tf.layers.dense(self.input, self.hidden_units[0], activation=tf.nn.tanh)

		for i in range(1,self.num_layers):
			self.hidden_layers[i] = tf.layers.dense(self.hidden_layers[i-1],self.hidden_units[i],activation=tf.nn.tanh)

		# Create Normal distribution for action. 
		# Use the fact that GYM takes care of bounding actions --> This is wrong, but start with this. 		
		self.normal_means = tf.layers.dense(self.hidden_layers[-1],self.output_dimensions,activation=tf.nn.sigmoid)
		self.normal_vars = tf.layers.dense(self.hidden_layers[-1],self.output_dimensions,activation=tf.nn.softplus)
		self.normal_dist = tf.distributions.Normal(loc=self.normal_means,scale=self.normal_vars)

		# Sample and placeholder for sampled actions. 
		self.sample_action = self.normal_dist.sample()
		
	def logging_ops(self):
		if self.to_train:
			# Create file writer to write summaries.        
			self.tf_writer = tf.summary.FileWriter('train_logging'+'/',self.sess.graph)

			# Create summaries for: Log likelihood, reward weight, and total reward on the full image. 
			self.action_loglikelihood_summary = tf.summary.scalar('Action_LogLikelihood',tf.reduce_mean(self.action_loglikelihood))
			self.reward_weight_summary = tf.summary.scalar('Reward_Weight',tf.reduce_mean(self.return_weight))

			# Merge summaries. 
			self.merged_summaries = tf.summary.merge_all()      

	def loss_ops(self):
		# Return placeholder for MCPG. 
		self.return_weight = tf.placeholder(tf.float32,shape=(None,1),name='return_weight')

		# Sampled action placeholder. 
		self.sampled_action = tf.placeholder(tf.float32,shape=(None,self.output_dimensions),name='sampled_action')

		# Evaluate log likelihood of sampled action. 
		self.action_loglikelihood = self.normal_dist.log_prob(self.sampled_action)

		#Evaluate loss
		self.loss = -tf.multiply(self.action_loglikelihood,self.return_weight)

	def train_ops(self):

		self.optimizer = tf.train.AdamOptimizer(1e-4)
		self.train = self.optimizer.minimize(self.loss,name='Adam_Optimizer')

		# Writing graph and other summaries in tensorflow.
		self.writer = tf.summary.FileWriter('training',self.sess.graph)
		init = tf.global_variables_initializer()
		self.sess.run(init)

	def model_load_alt(self, model_file):
		print("RESTORING MODEL FROM:", model_file)
		saver = tf.train.Saver(max_to_keep=None)
		saver.restore(self.sess,model_file)

	def save_model(self, model_index, iteration_number=-1):
		if not(os.path.isdir("saved_models")):
			os.mkdir("saved_models")

		self.saver = tf.train.Saver(max_to_keep=None)           

		if not(iteration_number==-1):
			save_path = self.saver.save(self.sess,'saved_models/model_epoch{0}_iter{1}.ckpt'.format(model_index,iteration_number))
		else:
			save_path = self.saver.save(self.sess,'saved_models/model_epoch{0}.ckpt'.format(model_index))

	def create_policy_network(self, sess, pretrained_weights=None, to_train=False):

		self.initialize_base_model(sess, to_train=to_train)
		self.logging_ops()
		self.loss_ops()
		self.train_ops()

		if pretrained_weights:
			self.model_load_alt(pretrained_weights)
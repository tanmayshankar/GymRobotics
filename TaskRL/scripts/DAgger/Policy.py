#!/usr/bin/env python
from headers import *

class DAggerPolicy():

	def __init__(self, input_dimensions, output_dimensions, num_layers=4,name_scope=None):
		self.input_dimensions = input_dimensions
		self.output_dimensions = output_dimensions
		self.num_layers = num_layers
		if name_scope:
			self.name_scope = name_scope

		print("Setup Actor Model Variables.")

	def define_base_model(self, sess, to_train=None):
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

		print("Setup Base Model.")

	def define_policy_layers(self):
		self.predicted_action = tf.layers.dense(self.hidden_layers[-1],self.output_dimensions,activation=tf.nn.tanh, name='predicted_action')
		print("Setup Policy Model Policy layers.")
		
	def define_actor_model(self,sess, to_train=None):
		with tf.variable_scope(self.name_scope):
			self.define_base_model(sess,to_train)
			self.define_policy_layers()

		print("Setup Policy Model.")

	def define_actor_training_ops(self):

		# Here just DAgger supervision. 
		self.target_action = tf.placeholder(tf.float32, shape=[None,self.output_dimensions],name='target_actions')		
		self.actor_loss = tf.losses.mean_squared_error(self.target_action, self.predicted_actions)

		# Must get actor variables. 
		self.actor_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='ActorModel')
		self.actor_optimizer = tf.train.AdamOptimizer(1e-4)
		# self.train_actor = self.actor_optimizer.minimize(self.actor_loss,name='Train_Actor',var_list=self.actor_variables)
		self.clip_value = 10
		# Clipping gradients because of NaN values. 
		self.actor_gradients_vars = self.actor_optimizer.compute_gradients(self.actor_loss,var_list=self.actor_variables)
		self.actor_clipped_gradients = [(tf.clip_by_norm(grad,self.clip_value),var) for grad, var in self.actor_gradients_vars]
		# self.train_actor = self.actor_optimizer.apply_gradients(self.actor_gradients_vars)
		self.train_actor = self.actor_optimizer.apply_gradients(self.actor_clipped_gradients)

		print("Defined Actor Training Ops.")			

	def define_logging_ops(self):
		if self.to_train:
			# Create file writer to write summaries.        
			self.tf_writer = tf.summary.FileWriter('train_logging'+'/',self.sess.graph)

			# # Create summaries for: Log likelihood, reward weight, and total reward on the full image. 
			# self.action_loglikelihood_summary = tf.summary.scalar('Action_LogLikelihood',tf.reduce_mean(self.actor_loglikelihood))
			# self.reward_weight_summary = tf.summary.scalar('Reward_Weight',tf.reduce_mean(self.return_weight))
			
			# Create summaries for: Log likelihood, reward weight, and total reward on the full image. 
			self.actor_loss_summary = tf.summary.scalar('Actor_Loss',tf.reduce_mean(self.actor_loss))

			# Merge summaries. 
			self.merged_summaries = tf.summary.merge_all()      

		print("Defined Logging Ops.")			

	def define_training_ops(self):

		self.define_actor_training_ops()
		# Writing graph and other summaries in tensorflow.
		if self.to_train:
			self.writer = tf.summary.FileWriter('training',self.sess.graph)			
		init = tf.global_variables_initializer()
		self.sess.run(init)

		print("Defined All Training Ops.")			

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
		self.define_actor_model(sess,to_train=to_train)
		self.define_training_ops()
		self.define_logging_ops()

		if pretrained_weights:
			self.model_load_alt(pretrained_weights)

		print("Done creating PolicyModel.")
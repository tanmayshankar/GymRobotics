#!/usr/bin/env python
from headers import *

class ActorModel():

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

		print("Setup Base Actor Model.")

	def define_actor_layers(self):
		# Now constructing deterministic policy representation. 
		self.predicted_action = tf.layers.dense(self.hidden_layers[-1],self.output_dimensions,activation=tf.nn.tanh)
		print("Setup Actor Model Actor Layers.")
		
	def define_actor_model(self,sess, to_train=None):
		with tf.variable_scope(self.name_scope):
			self.define_base_model(sess,to_train)
			self.define_actor_layers()

		print("Setup Actor Model.")

class CriticModel():

	def __init__(self, input_dimensions, output_dimensions, num_layers=4, name_scope=None):
		# The critic takes in state as well as the action dimensions. 
		self.input_dimensions = input_dimensions
		self.output_dimensions = output_dimensions
		self.num_layers = num_layers
		if name_scope:
			self.name_scope = name_scope

		print("Setup Critic Model Variables.")

	def define_base_model(self, sess, to_train=None):
		# Initializing the session.
		self.sess = sess
		self.to_train = to_train

		# Set number of hidden units. 
		self.hidden_units = 30*npy.ones((self.num_layers+1),dtype=int)        
		self.hidden_units[-1] = self.output_dimensions

		# Input placeholder.
		self.input = tf.placeholder(tf.float32,shape=[None,self.input_dimensions],name='input')		
		# Use placeholder. Feed action from actor network for computing Q(s',\pi(s')). 
		# Use action from memory to compute Q(s,a) when training critic. 
		# To compute actor gradients, multiply or use gradients_ys from tf.gradients.
		# self.action_taken = actor_action
		self.action_taken = tf.placeholder(tf.float32, shape=[None, self.output_dimensions],name='action_taken')

		self.concat_input = tf.concat([self.input,self.action_taken],axis=1,name='concat')

		# Defining hidden layers of MLP. 
		self.hidden_layers = [[] for i in range(self.num_layers)]        
		self.hidden_layers[0] = tf.layers.dense(self.concat_input, self.hidden_units[0], activation=tf.nn.tanh)

		for i in range(1,self.num_layers):
			self.hidden_layers[i] = tf.layers.dense(self.hidden_layers[i-1],self.hidden_units[i],activation=tf.nn.tanh)

		print("Setup Base Critic Model.")			

	def define_critic_layers(self):
		# self.initialization_val = 3e-4
		# self.predicted_Qvalue = tf.layers.dense(self.hidden_layers[-1],1,name='predicted_Qvalue',
		# 	kernel_initializer=tf.random_uniform_initializer(minval=-self.initialization_val,maxval=self.initialization_val),
		# 	bias_initializer=tf.random_uniform_initializer(minval=-self.initialization_val,maxval=self.initialization_val))
		self.predicted_Qvalue = tf.layers.dense(self.hidden_layers[-1],1,name='predicted_Qvalue')
		print("Setup Critic Model Critic Layer.")			

	def define_critic_model(self,sess, to_train=None):
		with tf.variable_scope(self.name_scope):
			self.define_base_model(sess,to_train)
			self.define_critic_layers()

		print("Setup Critic Model.")

class ActorCriticModel():

	def __init__(self, input_dimensions, output_dimensions, sess=None, to_train=True):

		self.sess = sess
		self.to_train = to_train		
		# Here we instantiate the actor and critic (don't inherit).

		self.actor_network = ActorModel(input_dimensions,output_dimensions,name_scope='ActorModel')
		self.actor_network.define_actor_model(sess,to_train=to_train)		

		self.critic_network = CriticModel(input_dimensions,output_dimensions,name_scope='CriticModel')
		self.critic_network.define_critic_model(sess,to_train=to_train)

		print("Setup Actor Critic Model Init.")			

	def define_critic_training_ops(self):
		self.target_Qvalue = tf.placeholder(tf.float32, shape=(None,1), name='target_Qvalue')
		self.critic_loss = tf.losses.mean_squared_error(self.target_Qvalue, self.critic_network.predicted_Qvalue)

		# Get critic variables, to ensure gradients don't propagate through the actor.
		self.critic_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=self.critic_network.name_scope)

		# Creating a training operation to minimize the critic loss.
		self.critic_optimizer = tf.train.AdamOptimizer(1e-4)
		# self.train_critic = self.critic_optimizer.minimize(self.critic_loss,name='Train_Critic',var_list=self.critic_variables)

		# Clipping gradients because of NaN values. 
		self.clip_value = 10
		self.critic_gradients_vars = self.critic_optimizer.compute_gradients(self.critic_loss,var_list=self.critic_variables)
		self.critic_clipped_gradients = [(tf.clip_by_norm(grad,self.clip_value),var) for grad, var in self.critic_gradients_vars]
		# self.train_critic = self.critic_optimizer.apply_gradients(self.critic_gradients_vars)
		self.train_critic = self.critic_optimizer.apply_gradients(self.critic_clipped_gradients)

		print("Defined Critic Training Ops.")			

	def define_actor_training_ops(self):
		# NOW DDPG Implementation. 

		# Must get actor variables. 
		self.actor_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=self.actor_network.name_scope)
		self.actor_optimizer = tf.train.AdamOptimizer(1e-4)

		# First compute: \nabla_{a} Q(s_t,a) | a=\mu(s_t|\theta) 		
		self.critic_grad_wrt_action = tf.gradients(-self.critic_network.predicted_Qvalue, self.critic_network.action_taken)

		# Now compute: \nabla_{\theta} \mu(s_t|\theta). 
		# Use GRAD_YS parameter in tf.gradients to multiply this with \nabla_{a} Q(s_t,a) | a=\mu(s_t|\theta).
		self.actor_gradients = tf.gradients(self.actor_network.predicted_action,self.actor_variables, grad_ys=self.critic_grad_wrt_action)		
		# Remember, tf.gradients only returns gradients. 
		# Optimizer.compute_gradients used to return var list and gradients. 

		# # WITHOUT CLIPPING GRADIENTS: 
		# # No explicit loss that we're keeping track of. 
		# # self.actor_loss = -tf.multiply(self.actor_loglikelihood, self.critic_network.predicted_Qvalue)
		# # self.train_actor = self.actor_optimizer.minimize(self.actor_loss,name='Train_Actor',var_list=self.actor_variables)
		# self.train_actor = self.actor_optimizer.apply_gradients(zip(self.actor_gradients,self.actor_variables))		

		# WITH CLIPPING GRADIENTS: 
		self.actor_gradients_vars = zip(self.actor_gradients,self.actor_variables)
		self.actor_clipped_gradients = [(tf.clip_by_norm(grad,self.clip_value),var) for grad, var in self.actor_gradients_vars]
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
			# self.actor_loss_summary = tf.summary.scalar('Actor_Loss',tf.reduce_mean(self.actor_loss))
			self.critic_loss_summary = tf.summary.scalar('Critic_Loss',tf.reduce_mean(self.critic_loss))

			# Merge summaries. 
			self.merged_summaries = tf.summary.merge_all()      

		print("Defined Logging Ops.")			

	def define_training_ops(self):

		self.define_critic_training_ops()
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
		self.define_training_ops()
		self.define_logging_ops()

		if pretrained_weights:
			self.model_load_alt(pretrained_weights)

		print("Done creating ACModel.")
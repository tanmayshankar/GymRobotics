#!/usr/bin/env python
from headers import *
from Memory import ReplayMemory
# from Policy import ActorCriticModel
from Policy import DAggerPolicy
from Trainer import Trainer

class PolicyManager():

	def __init__(self, session=None, arguments=None):

		self.sess = session
		self.args = arguments

		# Initialize Gym environment.
		self.environment = gym.make(self.args.env)        

		if self.args.env=='FetchReach-v0':
			input_dimensions = 16
		elif self.args.env=='FetchPush-v0':
			input_dimensions = 31		
	
		output_dimensions = 4

		# Initialize a polivy network. 
		# self.ACModel = ActorCriticModel(input_dimensions,output_dimensions,sess=session,to_train=self.args.train)
		self.PolicyModel = DAggerPolicy(input_dimensions,output_dimensions,name_scope='PolicyModel',sess=session,to_train=self.args.train)

		# Create the actual network
		if self.args.weights:
			self.PolicyModel.create_policy_network(session, pretrained_weights=self.args.weights,to_train=self.args.train)
		else:
			self.PolicyModel.create_policy_network(session, to_train=self.args.train)			

		# Initialize a memory replay. 
		self.memory = ReplayMemory()

		# Create a trainer instance. 
		self.trainer = Trainer(sess=session,policy=self.PolicyModel, environment=self.environment, memory=self.memory,args=self.args)

	def train(self):
		self.trainer.meta_training()

def parse_arguments():
	parser = argparse.ArgumentParser(description='Model Learning')
	parser.add_argument('--weights',dest='weights',type=str)
	parser.add_argument('--train',dest='train',type=int,default=1)
	parser.add_argument('--env',dest='env',type=str)
	parser.add_argument('--gpu',dest='gpu',type=str)
	parser.add_argument('--render',dest='render',type=int,default=0)	

	return parser.parse_args()

def main(args):
	args = parse_arguments()

	# Create a TensorFlow session with limits on GPU usage.
	gpu_ops = tf.GPUOptions(allow_growth=True,visible_device_list=args.gpu)
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)

	policy_manager = PolicyManager(session=sess,arguments=args)
	policy_manager.train()    

if __name__ == '__main__':
	main(sys.argv)




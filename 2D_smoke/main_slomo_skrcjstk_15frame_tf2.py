from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import math
import tensorflow as tf
import numpy as np
import random
from PIL import Image
from module_slomo_skrcjstk_15frame_tf2 import SloMo_model, Slomo_inference_model
import uniio
from utils import getSemiLagrPosBatch

# The operation used to print out the configuration
def print_configuration_op(FLAGS):
	print('[Configurations]:')
	# pdb.set_trace()
	for name in FLAGS.__flags.keys():
		value = getattr(FLAGS, name)
		if type(value) == float:
			print('\t%s: %f' % (name, value))
		elif type(value) == int:
			print('\t%s: %d' % (name, value))
		elif type(value) == str:
			print('\t%s: %s' % (name, value))
		elif type(value) == bool:
			print('\t%s: %s' % (name, value))
		else:
			print('\t%s: %s' % (name, value))

	print('End of configuration')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

Flags = tf.compat.v1.app.flags

# The system parameter
Flags.DEFINE_string('output_dir', None, 'The output directory of the checkpoint')
Flags.DEFINE_string('summary_dir', None, 'The dirctory to output the summary')
Flags.DEFINE_string('mode', 'train', 'The mode of the model train, test.')
Flags.DEFINE_string('checkpoint', None, 'If provided, the weight will be restored from the provided checkpoint.'
										'Checkpoint folder (Latest checkpoint will be taken)')
Flags.DEFINE_boolean('pre_trained_model', False,
					 'If set True, the weight will be loaded but the global_step will still '
					 'be 0. If set False, you are going to continue the training. That is, '
					 'the global_step will be initiallized from the checkpoint, too')
Flags.DEFINE_string('vgg_ckpt', './vgg19/vgg_19.ckpt', 'path to checkpoint file for the vgg19')
Flags.DEFINE_integer('batch_size', 2, 'Batch size of the input batch')

# model configurations
Flags.DEFINE_integer('first_kernel', 7, 'First conv kernel size in flow computation network')
Flags.DEFINE_integer('second_kernel', 5, 'First conv kernel size in flow computation network')
Flags.DEFINE_float('epsilon', 1e-12, 'The eps added to prevent nan')
Flags.DEFINE_string('perceptual_mode', 'VGG54', 'The type of feature used in perceptual loss')
Flags.DEFINE_float('reconstruction_scaling', 0.1, 'The scaling factor for the reconstruction loss')
Flags.DEFINE_float('perceptual_scaling', 1.0, 'The scaling factor for the perceptual loss')
Flags.DEFINE_float('mask_scaling', 1.0, 'The scaling factor for the wrapping loss')
Flags.DEFINE_float('smoothness_scaling', 50.0, 'The scaling factor for the smoothness loss')

# Trainer Parameters
Flags.DEFINE_float('learning_rate', 0.0001, 'The learning rate for the network')
Flags.DEFINE_integer('decay_step', 500000, 'The steps needed to decay the learning rate')
Flags.DEFINE_float('decay_rate', 0.1, 'The decay rate of each decay step')
Flags.DEFINE_boolean('stair', False, 'Whether perform staircase decay. True => decay in discrete interval.')
Flags.DEFINE_float('beta', 0.9, 'The beta1 parameter for the Adam optimizer')
Flags.DEFINE_integer('max_epoch', None, 'The max epoch for the training')
Flags.DEFINE_integer('max_iter', 1000000, 'The max iteration of the training')
Flags.DEFINE_integer('display_freq', 200, 'The diplay frequency of the training process')
Flags.DEFINE_integer('summary_freq', 200, 'The frequency of writing summary')
Flags.DEFINE_integer('save_freq', 1000, 'The frequency of saving checkpoint')
Flags.DEFINE_integer('test_step', 0, 'The step of test generation')
Flags.DEFINE_integer('start_step', 0, 'The step of start checkpoint')
Flags.DEFINE_integer('over_lap', 0, 'The number of pixels for inference')

Flags.DEFINE_integer('trainingSetStart', 0, '...')
Flags.DEFINE_integer('trainingSetLimit', 0, '...')
Flags.DEFINE_integer('testSetStart', 0, '...')
Flags.DEFINE_integer('testSetLimit', 0, '...')
Flags.DEFINE_integer('frameLimit', 0, '...')

Flags.DEFINE_integer('save_uni', 0, 'Boolean for save uni files')
Flags.DEFINE_string('save_uni_dir', None, 'The output directory of uni files')

FLAGS = Flags.FLAGS

# path to sim data, trained models and output are also saved here
BASE_PATH = '/media/cgnadeep2/hdd2/sharedfolder/skrcjstk_data/manta_data/data_skrcjstk7/'
SAVE_PATH = FLAGS.output_dir

IS_SAVE_UNI = bool(FLAGS.save_uni)
if IS_SAVE_UNI is True:
	SAVE_UNI_PATH = FLAGS.save_uni_dir
else:
	SAVE_UNI_PATH = FLAGS.output_dir

if not os.path.exists(SAVE_PATH): os.mkdir(SAVE_PATH)
LOG_DIR = os.path.join(SAVE_PATH, "log")
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)

trainingSetStart = FLAGS.trainingSetStart
trainingSetLimit = FLAGS.trainingSetLimit
testSetStart 	 = FLAGS.testSetStart
testSetLimit 	 = FLAGS.testSetLimit
frameLimit       = FLAGS.frameLimit
batchSize        = FLAGS.batch_size
res 		 	 = 256
steps_per_epoch  = 5
start_step 		 = int(FLAGS.start_step)
test_step 		 = int(FLAGS.test_step)
overlap 		 = int(FLAGS.over_lap)
dt 				 = 0.5
dim 			 = 2

# Print the configuration of the model
print_configuration_op(FLAGS)

def advect_vel(macgrid_input, time_dt):
	dtArray = np.array([i * time_dt for i in range(1,2)], dtype=np.float32)
	if (dim == 2):
		dtArray = dtArray.reshape((-1, 1, 1, 1))
	else:
		dtArray = dtArray.reshape((-1, 1, 1, 1, 1)) 
	#print(dtArray)
	
	#print('macgrid_input: ', macgrid_input.shape)
	vel_pos_high_inter = getSemiLagrPosBatch(macgrid_input, dtArray, res).reshape((1, -1))

	return vel_pos_high_inter

def load_data(idx, frameIdx):
	densities = []
	velocities = []
	header_density = 'dummy'
	if os.path.exists( "%s/simSimple_%04d" % (BASE_PATH, idx) ):
		for i in range(-8, 9): 
			filename = "%s/simSimple_%04d/density_%04d.uni" 
			uniPath = filename % (BASE_PATH, idx, frameIdx+i)  # 100 files per sim
			header_density, content = uniio.readUni(uniPath) # returns [Z,Y,X,C] np array
			h = header_density['dimX']
			w = header_density['dimY']
			arr = content[:, :, :, :] # reverse order of Y axis
			arr = np.reshape(arr, [w, h, 1]) # discard Z
			densities.append( arr )
		densities = np.reshape( densities, (len(densities), res,res,1) )
		for i in range(-8, 9): 
			filename = "%s/simSimple_%04d/vel_%04d.uni" 
			uniPath = filename % (BASE_PATH, idx, frameIdx+i)  # 100 files per sim
			header, content = uniio.readUni(uniPath) # returns [Z,Y,X,C] np array
			h = header['dimX']
			w = header['dimY']
			arr = content[:, :, :, :] # reverse order of Y axis
			arr = np.reshape(arr, [w, h, 3]) # discard Z
			velocities.append( arr )
		velocities = np.reshape( velocities, (len(velocities), res,res,3) )
	return densities, velocities, header_density

def load_test_data(idx, frameIdx):
	densities_test = []
	velocities_test = []
	header_density = 'dummy'
	if os.path.exists( "%s/testSimple_%04d" % (BASE_PATH, idx) ):
		for i in range(-8, 9): 
			filename = "%s/testSimple_%04d/density_%04d.uni" 
			uniPath = filename % (BASE_PATH, idx, frameIdx+i)  # 100 files per sim
			header_density, content = uniio.readUni(uniPath) # returns [Z,Y,X,C] np array
			h = header_density['dimX']
			w = header_density['dimY']
			arr = content[:, :, :, :] # reverse order of Y axis
			arr = np.reshape(arr, [w, h, 1]) # discard Z
			densities_test.append( arr )
		densities_test = np.reshape( densities_test, (len(densities_test), res,res,1) )
		for i in range(-8, 9): 
			filename = "%s/testSimple_%04d/vel_%04d.uni" 
			uniPath = filename % (BASE_PATH, idx, frameIdx+i)  # 100 files per sim
			header, content = uniio.readUni(uniPath) # returns [Z,Y,X,C] np array
			h = header['dimX']
			w = header['dimY']
			arr = content[:, :, :, :] # reverse order of Y axis
			arr = np.reshape(arr, [w, h, 3]) # discard Z
			velocities_test.append( arr )
		velocities_test = np.reshape( velocities_test, (len(velocities_test), res,res,3) )
	return densities_test, velocities_test, header_density

# Check Directories
if FLAGS.output_dir is None or FLAGS.summary_dir is None:
	raise ValueError('The output directory and summary directory are needed')

# Check the output directory to save the checkpoint
if not os.path.exists(FLAGS.output_dir):
	os.mkdir(FLAGS.output_dir)

# Check the summary directory to save the event
if not os.path.exists(FLAGS.summary_dir):
	os.mkdir(FLAGS.summary_dir)

class slomo_GAN(object):
	def __init__(self, sess):
		self.sess       = sess
		self._build_model()
		self.saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=3)

	def tensorResample(self, value, pos, name='Resample'):
		pos_shape = pos.get_shape().as_list()
		dim = len(pos_shape) - 2  # batch and channels are ignored
		assert (dim == pos_shape[-1])
		floors = tf.cast(tf.floor(pos - 0.5), tf.int32)
		ceils = floors + 1

		# clamp min
		floors = tf.maximum(floors, tf.zeros_like(floors))
		ceils = tf.maximum(ceils, tf.zeros_like(ceils))

		# clamp max
		floors = tf.minimum(floors, tf.constant(value.get_shape().as_list()[1:dim + 1], dtype=tf.int32) - 1)
		ceils = tf.minimum(ceils, tf.constant(value.get_shape().as_list()[1:dim + 1], dtype=tf.int32) - 1)

		_broadcaster = tf.ones_like(ceils)
		cell_value_list = []
		cell_weight_list = []
		for axis_x in range(int(pow(2, dim))):  # 3d, 0-7; 2d, 0-3;...
			condition_list = [bool(axis_x & int(pow(2, i))) for i in range(dim)]
			condition_ = (_broadcaster > 0) & condition_list
			axis_idx = tf.cast(
				tf.compat.v1.where(condition_, ceils, floors),
				tf.int32)

			# only support linear interpolation...
			axis_wei = 1.0 - tf.abs((pos - 0.5) - tf.cast(axis_idx, tf.float32))  # shape (..., res_x2, res_x1, dim)
			axis_wei = tf.reduce_prod(input_tensor=axis_wei, axis=-1, keepdims=True)
			cell_weight_list.append(axis_wei)  # single scalar(..., res_x2, res_x1, 1)
			first_idx = tf.ones_like(axis_wei, dtype=tf.int32)
			first_idx = tf.cumsum(first_idx, axis=0, exclusive=True)
			cell_value_list.append(tf.concat([first_idx, axis_idx], -1))

		values_new = tf.gather_nd(value, cell_value_list[0]) * cell_weight_list[0]  # broadcasting used, shape (..., res_x2, res_x1, channels )
		for cell_idx in range(1, len(cell_value_list)):
			values_new = values_new + tf.gather_nd(value, cell_value_list[cell_idx]) * cell_weight_list[cell_idx]
		return values_new  # shape (..., res_x2, res_x1, channels)

	def _build_model(self):
		if FLAGS.mode == "train" :
			self.x0_den = tf.compat.v1.placeholder(tf.float32, shape=[None, res, res, 1])
			self.x4_den = tf.compat.v1.placeholder(tf.float32, shape=[None, res, res, 1])
			
			self.x0_vel = tf.compat.v1.placeholder(tf.float32, shape=[None, res, res, 2])
			self.x1_vel = tf.compat.v1.placeholder(tf.float32, shape=[None, res, res, 2])
			self.x2_vel = tf.compat.v1.placeholder(tf.float32, shape=[None, res, res, 2])
			self.x3_vel = tf.compat.v1.placeholder(tf.float32, shape=[None, res, res, 2])
			self.x4_vel = tf.compat.v1.placeholder(tf.float32, shape=[None, res, res, 2])
			self.x5_vel = tf.compat.v1.placeholder(tf.float32, shape=[None, res, res, 2])
			self.x6_vel = tf.compat.v1.placeholder(tf.float32, shape=[None, res, res, 2])
			self.x7_vel = tf.compat.v1.placeholder(tf.float32, shape=[None, res, res, 2])
			self.x8_vel = tf.compat.v1.placeholder(tf.float32, shape=[None, res, res, 2])
			self.x9_vel = tf.compat.v1.placeholder(tf.float32, shape=[None, res, res, 2])
			self.x10_vel = tf.compat.v1.placeholder(tf.float32, shape=[None, res, res, 2])
			self.x11_vel = tf.compat.v1.placeholder(tf.float32, shape=[None, res, res, 2])

			def n25():
				adv1 = self.tensorResample(self.x0_den, self.x0_vel)
				adv2 = self.tensorResample(adv1, self.x1_vel)
				adv3 = self.tensorResample(adv2, self.x2_vel)
				return self.tensorResample(adv3, self.x3_vel)

			def n50():
				adv1 = self.tensorResample(self.x0_den, self.x0_vel)
				adv2 = self.tensorResample(adv1, self.x1_vel)
				adv3 = self.tensorResample(adv2, self.x2_vel)
				adv4 = self.tensorResample(adv3, self.x3_vel)
				adv5 = self.tensorResample(adv4, self.x4_vel)
				adv6 = self.tensorResample(adv5, self.x5_vel)
				adv7 = self.tensorResample(adv6, self.x6_vel)
				return self.tensorResample(adv7, self.x7_vel)

			def n75():
				adv1 = self.tensorResample(self.x0_den, self.x0_vel)
				adv2 = self.tensorResample(adv1, self.x1_vel)
				adv3 = self.tensorResample(adv2, self.x2_vel)
				adv4 = self.tensorResample(adv3, self.x3_vel)
				adv5 = self.tensorResample(adv4, self.x4_vel)
				adv6 = self.tensorResample(adv5, self.x5_vel)
				adv7 = self.tensorResample(adv6, self.x6_vel)
				adv8 = self.tensorResample(adv7, self.x7_vel)
				adv9 = self.tensorResample(adv8, self.x8_vel)
				adv10 = self.tensorResample(adv9, self.x9_vel)
				adv11 = self.tensorResample(adv10, self.x10_vel)
				return self.tensorResample(adv11, self.x11_vel)

			self.y2_den = tf.compat.v1.placeholder(tf.float32, shape=[None, res, res, 1])
			self.timestamp = tf.compat.v1.placeholder(tf.float32, shape=[1])

			self.x2_tmp_den = tf.case({tf.math.equal(self.timestamp[0], tf.constant(0.25)):n25,
									   tf.math.equal(self.timestamp[0], tf.constant(0.50)):n50,
									   tf.math.equal(self.timestamp[0], tf.constant(0.75)):n75,
									   }, default=None, exclusive=True)

			self.net_train = SloMo_model(self.x0_den, self.x4_den, self.x2_tmp_den, self.y2_den, FLAGS, self.timestamp)

			print('Finish building the network!!!')
		
			# Add scalar summaries
			tl_sum = tf.compat.v1.summary.scalar("total_loss", self.net_train.total_loss)
			rl_sum = tf.compat.v1.summary.scalar("reconstruction_loss", self.net_train.reconstruction_loss)
			ml_sum = tf.compat.v1.summary.scalar("mask_loss", self.net_train.mask_loss)
			sl_sum = tf.compat.v1.summary.scalar("smoothness_loss", self.net_train.smoothness_loss)
			al_sum = tf.compat.v1.summary.scalar("gradient_loss", self.net_train.advection_loss)
			lr_sum = tf.compat.v1.summary.scalar('learning_rate', self.net_train.learning_rate)

			self.d_sum = tf.compat.v1.summary.merge([tl_sum, rl_sum, ml_sum, sl_sum, al_sum, lr_sum])
		else:
			self.x0_den = tf.compat.v1.placeholder(tf.float32, shape=[None, res, res, 1])
			self.x4_den = tf.compat.v1.placeholder(tf.float32, shape=[None, res, res, 1])
			
			self.x0_vel = tf.compat.v1.placeholder(tf.float32, shape=[None, res, res, 2])
			self.x1_vel = tf.compat.v1.placeholder(tf.float32, shape=[None, res, res, 2])
			self.x2_vel = tf.compat.v1.placeholder(tf.float32, shape=[None, res, res, 2])
			self.x3_vel = tf.compat.v1.placeholder(tf.float32, shape=[None, res, res, 2])
			self.x4_vel = tf.compat.v1.placeholder(tf.float32, shape=[None, res, res, 2])
			self.x5_vel = tf.compat.v1.placeholder(tf.float32, shape=[None, res, res, 2])
			self.x6_vel = tf.compat.v1.placeholder(tf.float32, shape=[None, res, res, 2])
			self.x7_vel = tf.compat.v1.placeholder(tf.float32, shape=[None, res, res, 2])
			self.x8_vel = tf.compat.v1.placeholder(tf.float32, shape=[None, res, res, 2])
			self.x9_vel = tf.compat.v1.placeholder(tf.float32, shape=[None, res, res, 2])
			self.x10_vel = tf.compat.v1.placeholder(tf.float32, shape=[None, res, res, 2])
			self.x11_vel = tf.compat.v1.placeholder(tf.float32, shape=[None, res, res, 2])

			def n25():
				adv1 = self.tensorResample(self.x0_den, self.x0_vel)
				adv2 = self.tensorResample(adv1, self.x1_vel)
				adv3 = self.tensorResample(adv2, self.x2_vel)
				return self.tensorResample(adv3, self.x3_vel)

			def n50():
				adv1 = self.tensorResample(self.x0_den, self.x0_vel)
				adv2 = self.tensorResample(adv1, self.x1_vel)
				adv3 = self.tensorResample(adv2, self.x2_vel)
				adv4 = self.tensorResample(adv3, self.x3_vel)
				adv5 = self.tensorResample(adv4, self.x4_vel)
				adv6 = self.tensorResample(adv5, self.x5_vel)
				adv7 = self.tensorResample(adv6, self.x6_vel)
				return self.tensorResample(adv7, self.x7_vel)

			def n75():
				adv1 = self.tensorResample(self.x0_den, self.x0_vel)
				adv2 = self.tensorResample(adv1, self.x1_vel)
				adv3 = self.tensorResample(adv2, self.x2_vel)
				adv4 = self.tensorResample(adv3, self.x3_vel)
				adv5 = self.tensorResample(adv4, self.x4_vel)
				adv6 = self.tensorResample(adv5, self.x5_vel)
				adv7 = self.tensorResample(adv6, self.x6_vel)
				adv8 = self.tensorResample(adv7, self.x7_vel)
				adv9 = self.tensorResample(adv8, self.x8_vel)
				adv10 = self.tensorResample(adv9, self.x9_vel)
				adv11 = self.tensorResample(adv10, self.x10_vel)
				return self.tensorResample(adv11, self.x11_vel)

			self.y2_den = tf.compat.v1.placeholder(tf.float32, shape=[None, res, res, 1])
			self.timestamp = tf.compat.v1.placeholder(tf.float32, shape=[1])

			self.x2_tmp_den = tf.case({tf.math.equal(self.timestamp[0], tf.constant(0.25)):n25,
									   tf.math.equal(self.timestamp[0], tf.constant(0.50)):n50,
									   tf.math.equal(self.timestamp[0], tf.constant(0.75)):n75,
									   }, default=None, exclusive=True)

			self.net_train = Slomo_inference_model(self.x0_den, self.x4_den, self.x2_tmp_den, self.y2_den, FLAGS, self.timestamp)

	def train(self):
		# now we can start training...
		print("Starting training...")
		ckpt = tf.train.get_checkpoint_state(SAVE_PATH)
		if ckpt and tf.compat.v1.train.checkpoint_exists(ckpt.model_checkpoint_path):
			self.saver.restore(self.sess, ckpt.model_checkpoint_path)
			print('model is restored.')
		else:
			self.sess.run(tf.compat.v1.global_variables_initializer())
			print('model is initialized.')

		self.train_writer = tf.compat.v1.summary.FileWriter(os.path.join(LOG_DIR, "train"), self.sess.graph)
		#self.test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, "test"))
		
		# Performing the training
		if FLAGS.max_epoch is None:
			if FLAGS.max_iter is None:
				raise ValueError('one of max_epoch or max_iter should be provided')
			else:
				max_iter = FLAGS.max_iter
		else:
			max_iter = FLAGS.max_epoch * steps_per_epoch

		print('Optimization starts!!!')
		start = time.time()

		for step in range(start_step, max_iter):
			fetches = {
				"train": self.net_train.train,
				"global_step": self.net_train.global_step
			}

			if ((step + 1) % FLAGS.display_freq) == 0:
				fetches["total_loss"] = self.net_train.total_loss
				fetches["reconstruction_loss"] = self.net_train.reconstruction_loss
				fetches["learning_rate"] = self.net_train.learning_rate
				fetches["global_step"] = self.net_train.global_step
				fetches["mask_loss"] = self.net_train.mask_loss
				fetches["smoothness_loss"] = self.net_train.smoothness_loss
				fetches["advection_loss"] = self.net_train.advection_loss
				fetches["rec_loss_w_from1"] = self.net_train.rec_loss_w_from1
				fetches["rec_loss_w_tmp2"] = self.net_train.rec_loss_w_tmp2

			if ((step + 1) % FLAGS.summary_freq) == 0:
				fetches["summary"] = self.d_sum
				fetches["pred_frameT2"] = self.net_train.pred_frameT2
				fetches["interp_t_vis"] = self.net_train.interp_t_vis
				fetches["pred2_from_1"] = self.net_train.pred2_from_1
				fetches["ftmpD2"] = self.net_train.ftmpD2
								

			#data loading
			batchx0_den, batchx4_den, batchy1_den, batchy2_den, batchy3_den, = [],[],[],[],[]
			batchx0_vel, batchx1_vel, batchx2_vel, batchx3_vel, batchx4_vel, batchx5_vel = [],[],[],[],[],[]
			batchx6_vel, batchx7_vel, batchx8_vel, batchx9_vel, batchx10_vel, batchx11_vel = [],[],[],[],[],[]

			for i in range(batchSize):
				idxSet = random.sample(range(trainingSetStart, trainingSetLimit), 1)
				frameSet = random.sample(range(8, frameLimit-8), 1)
				densities, velocities, _ = load_data(idxSet[0], frameSet[0])
			
				batchx0_den.append( densities[0])
				batchx4_den.append( densities[16])
				batchy1_den.append( densities[4])
				batchy2_den.append( densities[8])
				batchy3_den.append( densities[12])
				
				tmp = []
				tmp.append(velocities[0])
				batchx0_vel.append(np.reshape(advect_vel(np.expand_dims(tmp, axis=1), dt), [res, res, 2]))
				tmp = []
				tmp.append(velocities[1])
				batchx1_vel.append(np.reshape(advect_vel(np.expand_dims(tmp, axis=1), dt), [res, res, 2]))
				tmp = []
				tmp.append(velocities[2])
				batchx2_vel.append(np.reshape(advect_vel(np.expand_dims(tmp, axis=1), dt), [res, res, 2]))
				tmp = []
				tmp.append(velocities[3])
				batchx3_vel.append(np.reshape(advect_vel(np.expand_dims(tmp, axis=1), dt), [res, res, 2]))
				tmp = []
				tmp.append(velocities[4])
				batchx4_vel.append(np.reshape(advect_vel(np.expand_dims(tmp, axis=1), dt), [res, res, 2]))
				tmp = []
				tmp.append(velocities[5])
				batchx5_vel.append(np.reshape(advect_vel(np.expand_dims(tmp, axis=1), dt), [res, res, 2]))
				tmp = []
				tmp.append(velocities[6])
				batchx6_vel.append(np.reshape(advect_vel(np.expand_dims(tmp, axis=1), dt), [res, res, 2]))
				tmp = []
				tmp.append(velocities[7])
				batchx7_vel.append(np.reshape(advect_vel(np.expand_dims(tmp, axis=1), dt), [res, res, 2]))
				tmp = []
				tmp.append(velocities[8])
				batchx8_vel.append(np.reshape(advect_vel(np.expand_dims(tmp, axis=1), dt), [res, res, 2]))
				tmp = []
				tmp.append(velocities[9])
				batchx9_vel.append(np.reshape(advect_vel(np.expand_dims(tmp, axis=1), dt), [res, res, 2]))
				tmp = []
				tmp.append(velocities[10])
				batchx10_vel.append(np.reshape(advect_vel(np.expand_dims(tmp, axis=1), dt), [res, res, 2]))
				tmp = []
				tmp.append(velocities[11])
				batchx11_vel.append(np.reshape(advect_vel(np.expand_dims(tmp, axis=1), dt), [res, res, 2]))

			reconst_loss_x1, reconst_loss_x2, reconst_loss_x3 = 0,0,0
			feeds = {self.x0_den: batchx0_den, self.x4_den: batchx4_den, self.y2_den: batchy1_den, 
					 self.x0_vel: batchx0_vel, self.x4_vel: batchx4_vel, self.x8_vel: batchx8_vel,
					 self.x1_vel: batchx1_vel, self.x5_vel: batchx5_vel, self.x9_vel: batchx9_vel,
					 self.x2_vel: batchx2_vel, self.x6_vel: batchx6_vel, self.x10_vel: batchx10_vel,
					 self.x3_vel: batchx3_vel, self.x7_vel: batchx7_vel, self.x11_vel: batchx11_vel,
					 self.timestamp:[0.25]}
			results1 = self.sess.run(fetches, feed_dict=feeds)
			if ((step + 1) % FLAGS.display_freq) == 0:
				reconst_loss_x1 = results1["reconstruction_loss"]
				reconst_loss_w_from1_x1 = results1["rec_loss_w_from1"]
				reconst_loss_w_tmp_x1 = results1["rec_loss_w_tmp2"]
			
			feeds = {self.x0_den: batchx0_den, self.x4_den: batchx4_den, self.y2_den: batchy2_den, 
					 self.x0_vel: batchx0_vel, self.x4_vel: batchx4_vel, self.x8_vel: batchx8_vel,
					 self.x1_vel: batchx1_vel, self.x5_vel: batchx5_vel, self.x9_vel: batchx9_vel,
					 self.x2_vel: batchx2_vel, self.x6_vel: batchx6_vel, self.x10_vel: batchx10_vel,
					 self.x3_vel: batchx3_vel, self.x7_vel: batchx7_vel, self.x11_vel: batchx11_vel,
					 self.timestamp:[0.50]}
			results2 = self.sess.run(fetches, feed_dict=feeds)
			if ((step + 1) % FLAGS.display_freq) == 0:
				reconst_loss_x2 = results2["reconstruction_loss"]
				reconst_loss_w_from1_x2 = results2["rec_loss_w_from1"]
				reconst_loss_w_tmp_x2 = results2["rec_loss_w_tmp2"]

			feeds = {self.x0_den: batchx0_den, self.x4_den: batchx4_den, self.y2_den: batchy3_den, 
					 self.x0_vel: batchx0_vel, self.x4_vel: batchx4_vel, self.x8_vel: batchx8_vel,
					 self.x1_vel: batchx1_vel, self.x5_vel: batchx5_vel, self.x9_vel: batchx9_vel,
					 self.x2_vel: batchx2_vel, self.x6_vel: batchx6_vel, self.x10_vel: batchx10_vel,
					 self.x3_vel: batchx3_vel, self.x7_vel: batchx7_vel, self.x11_vel: batchx11_vel,
					 self.timestamp:[0.75]}
			results3 = self.sess.run(fetches, feed_dict=feeds)
			if ((step + 1) % FLAGS.display_freq) == 0:
				reconst_loss_x3 = results3["reconstruction_loss"]
				reconst_loss_w_from1_x3 = results3["rec_loss_w_from1"]
				reconst_loss_w_tmp_x3 = results3["rec_loss_w_tmp2"]
			
			

			if ((step + 1) % FLAGS.display_freq) == 0:
				train_epoch = math.ceil(results3["global_step"] / steps_per_epoch)
				train_step = (results3["global_step"] - 1) % steps_per_epoch + 1
				rate = (step - start_step + 1) * FLAGS.batch_size / (time.time() - start)
				remaining = (max_iter - step) * FLAGS.batch_size / rate
				print('step : %d' %(step))
				print('max_iter : %d' %(max_iter))
				#print('remaining : %f' %(remaining))
				print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (
					train_epoch, train_step, rate, remaining / 60))
				print("global_step", results3["global_step"])
				print("learning_rate", results3['learning_rate'])
				print("total_loss", results3["total_loss"])
				print("mask_loss", results3["mask_loss"])
				print("smoothness_loss", results3["smoothness_loss"])
				print('')
				print("rec_loss_w_from1_x1", reconst_loss_w_from1_x1)
				print("reconstruction_loss_x1", reconst_loss_x1)
				print("rec_loss_w_tmp2_x1", reconst_loss_w_tmp_x1)
				print('')
				print("rec_loss_w_from1", reconst_loss_w_from1_x2)
				print("reconstruction_loss_x2", reconst_loss_x2)
				print("rec_loss_w_tmp2", reconst_loss_w_tmp_x2)
				print('')
				print("rec_loss_w_from1_x3", reconst_loss_w_from1_x3)
				print("reconstruction_loss_x3", reconst_loss_x3)
				print("rec_loss_w_tmp2_x3", reconst_loss_w_tmp_x3)
				print('')
				print("advection_loss", results3["advection_loss"])
				print(' ')

			if ((step + 1) % FLAGS.summary_freq) == 0:
				print('Recording summary !!!!')
				self.train_writer.add_summary(results3['summary'], results3['global_step'])

			if ((step + 1) % FLAGS.save_freq) == 0:
				print('Save the checkpoint !!!!')
				self.saver.save(self.sess, os.path.join(SAVE_PATH, 'model'), global_step=self.net_train.global_step)

				frame_x1 = results1["pred_frameT2"]
				pred2_from_1_x1 = results1["pred2_from_1"]
				interp_t_vis_x1 = results1["interp_t_vis"]
				ftmp_x1 = results1["ftmpD2"]

				frame_x3 = results3["pred_frameT2"]
				pred2_from_1_x3 = results3["pred2_from_1"]
				interp_t_vis_x3 = results3["interp_t_vis"]
				ftmp_x3 = results3["ftmpD2"]

				frame_x2 = results2["pred_frameT2"]
				pred2_from_1_x2 = results2["pred2_from_1"]
				interp_t_vis_x2 = results2["interp_t_vis"]
				ftmp_x2 = results2["ftmpD2"]

				outDir = "%s/testGen_in_training_%d/" % (SAVE_PATH,step)
				if not os.path.exists(outDir): os.makedirs(outDir)

				filename = "%s/est_%d.png"
				Image.fromarray(np.reshape(np.flip(batchx0_den[0],0)*256, [res, res])).convert('RGB').save(filename % (outDir, 0))
				Image.fromarray(np.reshape(np.flip(frame_x1[0],0)*256, [res, res])).convert('RGB').save(filename % (outDir, 1))
				Image.fromarray(np.reshape(np.flip(frame_x2[0],0)*256, [res, res])).convert('RGB').save(filename % (outDir, 2))
				Image.fromarray(np.reshape(np.flip(frame_x3[0],0)*256, [res, res])).convert('RGB').save(filename % (outDir, 3))
				Image.fromarray(np.reshape(np.flip(batchx4_den[0],0)*256, [res, res])).convert('RGB').save(filename % (outDir, 4))

				Image.fromarray(np.reshape(np.flip(pred2_from_1_x1[0],0)*256, [res, res])).convert('RGB').save("%s/est_from1_%d.png" % (outDir, 1))
				Image.fromarray(np.reshape(np.flip(pred2_from_1_x2[0],0)*256, [res, res])).convert('RGB').save("%s/est_from1_%d.png" % (outDir, 2))
				Image.fromarray(np.reshape(np.flip(pred2_from_1_x3[0],0)*256, [res, res])).convert('RGB').save("%s/est_from1_%d.png" % (outDir, 3))
								
				Image.fromarray(np.reshape(np.flip(ftmp_x1[0],0)*256, [res, res])).convert('RGB').save("%s/est_tmp_%d.png" % (outDir, 1))
				Image.fromarray(np.reshape(np.flip(ftmp_x2[0],0)*256, [res, res])).convert('RGB').save("%s/est_tmp_%d.png" % (outDir, 2))
				Image.fromarray(np.reshape(np.flip(ftmp_x3[0],0)*256, [res, res])).convert('RGB').save("%s/est_tmp_%d.png" % (outDir, 3))

				Image.fromarray(np.reshape(np.flip(batchy1_den[0],0)*256, [res, res])).convert('RGB').save("%s/gt_%d.png" % (outDir, 1))
				Image.fromarray(np.reshape(np.flip(batchy2_den[0],0)*256, [res, res])).convert('RGB').save("%s/gt_%d.png" % (outDir, 2))
				Image.fromarray(np.reshape(np.flip(batchy3_den[0],0)*256, [res, res])).convert('RGB').save("%s/gt_%d.png" % (outDir, 3))

				dump = np.zeros([res, res, 3], np.uint8)
				dump[:,:,0:1] = np.reshape(np.flip((interp_t_vis_x1[0]),0)*256, [res, res, 1])
				Image.fromarray(dump).convert('RGB').save("%s/est_interp_t_vis_%d.png" % (outDir, 1))
				dump = np.zeros([res, res, 3], np.uint8)
				dump[:,:,0:1] = np.reshape(np.flip((interp_t_vis_x2[0]),0)*256, [res, res, 1])
				Image.fromarray(dump).convert('RGB').save("%s/est_interp_t_vis_%d.png" % (outDir, 2))
				dump = np.zeros([res, res, 3], np.uint8)
				dump[:,:,0:1] = np.reshape(np.flip((interp_t_vis_x3[0]),0)*256, [res, res, 1])
				Image.fromarray(dump).convert('RGB').save("%s/est_interp_t_vis_%d.png" % (outDir, 3))

		print('Optimization done!!!!!!!!!!!!')

	def test(self):
		# now we can start training...
		print("Starting testing...")
		ckpt = tf.train.get_checkpoint_state(SAVE_PATH)
		if ckpt and tf.compat.v1.train.checkpoint_exists(ckpt.model_checkpoint_path):
			self.saver.restore(self.sess, ckpt.model_checkpoint_path)
			print('model is restored.')
		else:
			self.sess.run(tf.compat.v1.global_variables_initializer())
			print('model is initialized.')

		fetches = {
			"pred_frameT2": self.net_train.pred_frameT2,
			"ftmpD2": self.net_train.ftmpD2,
			"l1_x2": self.net_train.l1_x2,
			"interp_t_vis": self.net_train.interp_t_vis
			}

		iterCnt = 8
		
		start = time.time()
		idx = 0
		avg_total_x1, avg_total_x2, avg_total_x3 = 0,0,0
		for testIdx in range(testSetStart, testSetLimit):
			outDir = "%s/testGen_%04d_test/" % (SAVE_PATH, testIdx)
			if not os.path.exists(outDir): os.makedirs(outDir)

			avg_x1, avg_x2, avg_x3 = 0,0,0
			for epoch in range(iterCnt):
				offset = 16* epoch + 8
				densities, velocities, header = load_test_data(testIdx, offset)

				batchx0_den, batchx4_den, batchy1_den, batchy2_den, batchy3_den, = [],[],[],[],[]
				batchx0_vel, batchx1_vel, batchx2_vel, batchx3_vel, batchx4_vel, batchx5_vel = [],[],[],[],[],[]
				batchx6_vel, batchx7_vel, batchx8_vel, batchx9_vel, batchx10_vel, batchx11_vel = [],[],[],[],[],[]

				batchx0_den.append( densities[0])
				batchx4_den.append( densities[16])
				batchy1_den.append( densities[4])
				batchy2_den.append( densities[8])
				batchy3_den.append( densities[12])

				tmp = []
				tmp.append(velocities[0])
				batchx0_vel.append(np.reshape(advect_vel(np.expand_dims(tmp, axis=1), dt), [res, res, 2]))
				tmp = []
				tmp.append(velocities[1])
				batchx1_vel.append(np.reshape(advect_vel(np.expand_dims(tmp, axis=1), dt), [res, res, 2]))
				tmp = []
				tmp.append(velocities[2])
				batchx2_vel.append(np.reshape(advect_vel(np.expand_dims(tmp, axis=1), dt), [res, res, 2]))
				tmp = []
				tmp.append(velocities[3])
				batchx3_vel.append(np.reshape(advect_vel(np.expand_dims(tmp, axis=1), dt), [res, res, 2]))
				tmp = []
				tmp.append(velocities[4])
				batchx4_vel.append(np.reshape(advect_vel(np.expand_dims(tmp, axis=1), dt), [res, res, 2]))
				tmp = []
				tmp.append(velocities[5])
				batchx5_vel.append(np.reshape(advect_vel(np.expand_dims(tmp, axis=1), dt), [res, res, 2]))
				tmp = []
				tmp.append(velocities[6])
				batchx6_vel.append(np.reshape(advect_vel(np.expand_dims(tmp, axis=1), dt), [res, res, 2]))
				tmp = []
				tmp.append(velocities[7])
				batchx7_vel.append(np.reshape(advect_vel(np.expand_dims(tmp, axis=1), dt), [res, res, 2]))
				tmp = []
				tmp.append(velocities[8])
				batchx8_vel.append(np.reshape(advect_vel(np.expand_dims(tmp, axis=1), dt), [res, res, 2]))
				tmp = []
				tmp.append(velocities[9])
				batchx9_vel.append(np.reshape(advect_vel(np.expand_dims(tmp, axis=1), dt), [res, res, 2]))
				tmp = []
				tmp.append(velocities[10])
				batchx10_vel.append(np.reshape(advect_vel(np.expand_dims(tmp, axis=1), dt), [res, res, 2]))
				tmp = []
				tmp.append(velocities[11])
				batchx11_vel.append(np.reshape(advect_vel(np.expand_dims(tmp, axis=1), dt), [res, res, 2]))

				feeds = {self.x0_den: batchx0_den, self.x4_den: batchx4_den, self.y2_den: batchy1_den, 
						 self.x0_vel: batchx0_vel, self.x4_vel: batchx4_vel, self.x8_vel: batchx8_vel,
						 self.x1_vel: batchx1_vel, self.x5_vel: batchx5_vel, self.x9_vel: batchx9_vel,
						 self.x2_vel: batchx2_vel, self.x6_vel: batchx6_vel, self.x10_vel: batchx10_vel,
						 self.x3_vel: batchx3_vel, self.x7_vel: batchx7_vel, self.x11_vel: batchx11_vel,
						 self.timestamp:[0.25]}
				results = self.sess.run(fetches, feed_dict=feeds)
				frame_x1 = results["pred_frameT2"]
				avg_x1 += results["l1_x2"]
				interp_t_vis_x1 = results["interp_t_vis"]
				ftmp_x1 = results["ftmpD2"]

				feeds = {self.x0_den: batchx0_den, self.x4_den: batchx4_den, self.y2_den: batchy3_den, 
						 self.x0_vel: batchx0_vel, self.x4_vel: batchx4_vel, self.x8_vel: batchx8_vel,
						 self.x1_vel: batchx1_vel, self.x5_vel: batchx5_vel, self.x9_vel: batchx9_vel,
						 self.x2_vel: batchx2_vel, self.x6_vel: batchx6_vel, self.x10_vel: batchx10_vel,
						 self.x3_vel: batchx3_vel, self.x7_vel: batchx7_vel, self.x11_vel: batchx11_vel,
						 self.timestamp:[0.75]}
				results = self.sess.run(fetches, feed_dict=feeds)
				frame_x3 = results["pred_frameT2"]
				avg_x3 += results["l1_x2"]
				interp_t_vis_x3 = results["interp_t_vis"]
				ftmp_x3 = results["ftmpD2"]

				feeds = {self.x0_den: batchx0_den, self.x4_den: batchx4_den, self.y2_den: batchy2_den, 
						 self.x0_vel: batchx0_vel, self.x4_vel: batchx4_vel, self.x8_vel: batchx8_vel,
						 self.x1_vel: batchx1_vel, self.x5_vel: batchx5_vel, self.x9_vel: batchx9_vel,
						 self.x2_vel: batchx2_vel, self.x6_vel: batchx6_vel, self.x10_vel: batchx10_vel,
						 self.x3_vel: batchx3_vel, self.x7_vel: batchx7_vel, self.x11_vel: batchx11_vel,
						 self.timestamp:[0.50]}
				results = self.sess.run(fetches, feed_dict=feeds)
				frame_x2 = results["pred_frameT2"]
				avg_x2 += results["l1_x2"]
				interp_t_vis_x2 = results["interp_t_vis"]
				ftmp_x2 = results["ftmpD2"]

				filename = "%s/est_%d.png"
				Image.fromarray(np.reshape(np.flip(densities[0],0)*256, [res, res])).convert('RGB').save(filename % (outDir, offset-8))
				Image.fromarray(np.reshape(np.flip(frame_x1[0],0)*256, [res, res])).convert('RGB').save(filename % (outDir, offset-4))
				Image.fromarray(np.reshape(np.flip(frame_x2[0],0)*256, [res, res])).convert('RGB').save(filename % (outDir, offset))
				Image.fromarray(np.reshape(np.flip(frame_x3[0],0)*256, [res, res])).convert('RGB').save(filename % (outDir, offset+4))
				Image.fromarray(np.reshape(np.flip(densities[16],0)*256, [res, res])).convert('RGB').save(filename % (outDir, offset+8))
				
				#dump = np.zeros([res, res, 3], np.uint8)
				#dump[:,:,0:1] = np.reshape(np.flip((interp_t_vis_x1[0]),0)*256, [res, res, 1])
				#Image.fromarray(dump).convert('RGB').save("%s/est_interp_t_vis_%d.png" % (outDir, idx+1))
				#dump = np.zeros([res, res, 3], np.uint8)
				#dump[:,:,0:1] = np.reshape(np.flip((interp_t_vis_x2[0]),0)*256, [res, res, 1])
				#Image.fromarray(dump).convert('RGB').save("%s/est_interp_t_vis_%d.png" % (outDir, idx+2))
				#dump = np.zeros([res, res, 3], np.uint8)
				#dump[:,:,0:1] = np.reshape(np.flip((interp_t_vis_x3[0]),0)*256, [res, res, 1])
				#Image.fromarray(dump).convert('RGB').save("%s/est_interp_t_vis_%d.png" % (outDir, idx+3))

				#print('interp_t_x1:',interp_t_vis_x1[0])
				#print('interp_t_x2:',interp_t_vis_x2[0])
				#print('interp_t_x3:',interp_t_vis_x3[0])

				#Image.fromarray(np.reshape(np.flip(ftmp_x1[0],0)*256, [res, res])).convert('RGB').save("%s/est_tmp_%d.png" % (outDir, offset-8))
				#Image.fromarray(np.reshape(np.flip(ftmp_x2[0],0)*256, [res, res])).convert('RGB').save("%s/est_tmp_%d.png" % (outDir, offset))
				#Image.fromarray(np.reshape(np.flip(ftmp_x3[0],0)*256, [res, res])).convert('RGB').save("%s/est_tmp_%d.png" % (outDir, offset+8))

				#Image.fromarray(np.reshape(np.flip(densities[4],0)*256, [res, res])).convert('RGB').save("%s/gt_%d.png" % (outDir, offset+8))
				#Image.fromarray(np.reshape(np.flip(densities[8],0)*256, [res, res])).convert('RGB').save("%s/gt_%d.png" % (outDir, offset))
				#Image.fromarray(np.reshape(np.flip(densities[12],0)*256, [res, res])).convert('RGB').save("%s/gt_%d.png" % (outDir, offset+8))
				#idx += 4

			print('Test avg_x1 : %f' %(avg_x1/iterCnt))
			print('Test avg_x2 : %f' %(avg_x2/iterCnt))
			print('Test avg_x3 : %f' %(avg_x3/iterCnt))

			avg_x17=avg_x1+avg_x2+avg_x3
			print('Test avg_(x1~x3) : %f' %(avg_x17/iterCnt/3))
			print('Test %d done.' %(testIdx))

			avg_total_x1 += avg_x1/iterCnt
			avg_total_x2 += avg_x2/iterCnt
			avg_total_x3 += avg_x3/iterCnt
		
		print('time : ', time.time() - start)

		print('Test avg_total_x1 : %f' %(avg_total_x1/10))
		print('Test avg_total_x2 : %f' %(avg_total_x2/10))
		print('Test avg_total_x3 : %f' %(avg_total_x3/10))

		avg_total_x13=avg_total_x1+avg_total_x2+avg_total_x3
		print('Test total avg_(x1~x3) : %f' %(avg_total_x13/10/3))
			

def main(_):
	tfconfig = tf.compat.v1.ConfigProto(allow_soft_placement=True)
	tfconfig.gpu_options.allow_growth = True
	with tf.compat.v1.Session(config=tfconfig) as sess:
		model = slomo_GAN(sess=sess)
		if FLAGS.mode == "train" :
			model.train() 
		else:
			model.test()

if __name__ == '__main__':
	tf.compat.v1.app.run()

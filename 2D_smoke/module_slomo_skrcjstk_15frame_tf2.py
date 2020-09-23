from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import UpSampling2D 
from utils import flow_back_wrap2D

import tensorflow as tf
import collections

def gradient3D(x):
	# x: bzyxd
	dudx = x[:,:,:,1:,0] - x[:,:,:,:-1,0]
	dudy = x[:,:,1:,:,0] - x[:,:,:-1,:,0]
	dudz = x[:,1:,:,:,0] - x[:,:-1,:,:,0]
	
	dudx = tf.concat((dudx, tf.expand_dims(dudx[:,:,:,-1], axis=3)), axis=3)
	dudy = tf.concat((dudy, tf.expand_dims(dudy[:,:,-1,:], axis=2)), axis=2)
	dudz = tf.concat((dudz, tf.expand_dims(dudz[:,-1,:,:], axis=1)), axis=1)
		
	j = tf.stack([dudx,dudy,dudz], axis=-1)
	
	return j
	
def l1_loss(Ipred, Iref, axis=[3]):
	return tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=tf.abs(Ipred - Iref), axis=axis))  # L1 Norm

def l2_loss(Ipred, Iref, axis=[3]):
	return tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=tf.square(Ipred - Iref), axis=axis))  # L2 Norm

def reconstruction_loss_2D(Ipred, Iref):
	return l1_loss(Ipred, Iref)

def wrapping_loss(frame0, frame1, frameT, F01, F10, Fdasht0, Fdasht1):
	return l1_loss(frame0, flow_back_wrap2D(frame1, F01)) + \
		   l1_loss(frame1, flow_back_wrap2D(frame0, F10)) + \
		   l1_loss(frameT, flow_back_wrap2D(frame0, Fdasht0)) + \
		   l1_loss(frameT, flow_back_wrap2D(frame1, Fdasht1))

def smoothness_loss(F01, F10):
	deltaF01 = tf.reduce_mean(input_tensor=(tf.abs(F01[:, 1:, :, :] - F01[:, :-1, :, :]))) \
				+ tf.reduce_mean(input_tensor=(tf.abs(F01[:, :, 1:, :] - F01[:, :, :-1, :])))
	deltaF10 = tf.reduce_mean(input_tensor=(tf.abs(F10[:, 1:, :, :] - F10[:, :-1, :, :]))) \
				+ tf.reduce_mean(input_tensor=(tf.abs(F10[:, :, 1:, :] - F10[:, :, :-1, :])))
	return 0.5 * (deltaF01 + deltaF10)

# Model Helper Functions
def conv2d(batch_input, output_channels, kernel_size=3, stride=1, scope="conv", activation=None):
	with tf.compat.v1.variable_scope(scope):
		activation_fn = None
		if activation == 'leaky_relu':
			activation_fn = lambda x: tf.nn.leaky_relu(x, alpha=0.2)
		elif activation == 'relu':
			activation_fn = tf.nn.relu

		return tf.keras.layers.Conv2D(filters=output_channels, kernel_size=kernel_size, padding='same', strides=stride, data_format='channels_last', activation=activation_fn)(batch_input)

def lrelu(input, alpha=0.2):
	return tf.nn.leaky_relu(input, alpha=alpha)

def max_pool_2D(input, kernel_size, stride=2, scope="max_pool"):
	return tf.keras.layers.MaxPool2D(pool_size=(kernel_size, kernel_size), strides=stride)(input)

def bilinear_upsampling(input, scale=2, scope="bi_upsample"):
	with tf.compat.v1.variable_scope(scope):
		return tf.keras.layers.UpSampling2D(size=(scale, scale))(input)

def encoder_block(inputs, output_channel, conv_kernel=3, pool_kernel=2, lrelu_alpha=0.1, scope="enc_block"):
	with tf.compat.v1.variable_scope(scope):
		net = conv2d(inputs, output_channel, kernel_size=conv_kernel)
		conv = lrelu(net, lrelu_alpha)
		pool = max_pool_2D(conv, pool_kernel)
		return conv, pool

def decoder_block(input, skip_conn_input, output_channel, conv_kernel=3, up_scale=2, lrelu_alpha=0.1, scope="dec_block"):
	with tf.compat.v1.variable_scope(scope):
		upsample = bilinear_upsampling(input, scale=up_scale)

		upsample_shape = tf.shape(input=upsample)  # get_shape() - Static, Tf.shape() = dynamic
		skip_conn_shape = tf.shape(input=skip_conn_input)

		# upsample shape can differ from skip conn input (becouse of avg-pool and then bi-upsample in case of odd shape)
		xdiff, ydiff = skip_conn_shape[1] - upsample_shape[1], skip_conn_shape[2] - upsample_shape[2]
		upsample = tf.pad(tensor=upsample, paddings=tf.convert_to_tensor(value=[[0, 0], [0, xdiff], [0, ydiff], [0, 0]], dtype=tf.int32))
		block_input = tf.concat([upsample, skip_conn_input], 3)

		net = conv2d(block_input, output_channel, kernel_size=conv_kernel)
		net = lrelu(net, lrelu_alpha)
		return net

def UNet(inputs, output_channels, decoder_extra_input=None, first_kernel=7, second_kernel=5, scope='unet',
		 output_activation=None, reuse=False):
	with tf.compat.v1.variable_scope(scope, reuse=reuse):
		with tf.compat.v1.variable_scope("encoder"):
			layerCount = 8
			econv1, epool1 = encoder_block(inputs, layerCount, conv_kernel=first_kernel, scope="en_conv1")
			econv2, epool2 = encoder_block(epool1, layerCount*2, conv_kernel=second_kernel, scope="en_conv2")
			econv3, epool3 = encoder_block(epool2, layerCount*4, scope="en_conv3")
			econv4, epool4 = encoder_block(epool3, layerCount*8, scope="en_conv4")
			econv5, epool5 = encoder_block(epool4, layerCount*16, scope="en_conv5")
			with tf.compat.v1.variable_scope("en_conv6"):
				econv6 = conv2d(epool5, layerCount*16)
				econv6 = lrelu(econv6, alpha=0.1)

		with tf.compat.v1.variable_scope("decoder"):
			decoder_input = econv6
			if decoder_extra_input is not None:
				decoder_input = tf.concat([decoder_input, decoder_extra_input], axis=3)
			net = decoder_block(decoder_input, econv5, layerCount*16, scope="dec_conv1")
			net = decoder_block(net, econv4, layerCount*8, scope="dec_conv2")
			net = decoder_block(net, econv3, layerCount*4, scope="dec_conv3")
			net = decoder_block(net, econv2, layerCount*2, scope="dec_conv4")
			net = decoder_block(net, econv1, layerCount, scope="dec_conv5")

		with tf.compat.v1.variable_scope("unet_output"):
			net = conv2d(net, output_channels, scope="output")
			if output_activation is not None:
				if output_activation == "tanh":
					net = tf.nn.tanh(net)
				elif output_activation == "lrelu":
					net = lrelu(net, alpha=0.1)
				elif output_activation == "sigmoid":
					net = tf.nn.sigmoid(net)
				else:
					raise ValueError("only lrelu|tanh|sigmoid allowed")
			return net, econv6


def SloMo_inference_frame_step1(frame0, frame1, FLAGS, timestamp, reuse=False):
	with tf.compat.v1.variable_scope("SloMo_model1", reuse=reuse):
		with tf.variable_scope("flow_computation1"):
			flow_comp_input = tf.concat([frame0, frame1], axis=3)
			flow_comp_out, flow_comp_enc_out = UNet(flow_comp_input,
													output_channels=4,  # 2 channel for each flow
													first_kernel=FLAGS.first_kernel,
													second_kernel=FLAGS.second_kernel)
			flow_comp_out = lrelu(flow_comp_out)
			F01, F10 = flow_comp_out[:, :, :, :2], flow_comp_out[:, :, :, 2:4]
			print("Flow Computation1 Graph Initialized !!!!!! ")
		
		with tf.variable_scope("flow_interpolation1"):
			Fdasht0 = (-1 * (1 - timestamp) * timestamp * F01) + (timestamp * timestamp * F10)
			Fdasht1 = ((1 - timestamp) * (1 - timestamp) * F01) - (timestamp * (1 - timestamp) * F10)

			flow_interp_input = tf.concat([frame0, frame1,
										   flow_back_wrap2D(frame1, Fdasht1),
										   flow_back_wrap2D(frame0, Fdasht0),
										   Fdasht0, Fdasht1], axis=3)
			flow_interp_output, _ = UNet(flow_interp_input,
										 output_channels=5,  # 2 channels for each flow, 1 visibilty map.
										 decoder_extra_input=flow_comp_enc_out,
										 first_kernel=3,
										 second_kernel=3)

			deltaFt0, deltaFt1, Vt0 = flow_interp_output[:, :, :, :2], flow_interp_output[:, :, :, 2:4], \
									  flow_interp_output[:, :, :, 4:5]

			deltaFt0 = lrelu(deltaFt0)
			deltaFt1 = lrelu(deltaFt1)
			Vt0 = tf.sigmoid(Vt0)
			#Vt0 = tf.tile(Vt0, [1, 1, 1, 3])  # Copy same in all three channels
			Vt1 = 1 - Vt0

			Ft0, Ft1 = Fdasht0 + deltaFt0, Fdasht1 + deltaFt1

			normalization_factor = 1 / ((1 - timestamp) * Vt0 + timestamp * Vt1 + FLAGS.epsilon)
			pred_frameT = tf.multiply((1 - timestamp) * Vt0, flow_back_wrap2D(frame0, Ft0)) + \
						  tf.multiply(timestamp * Vt1, flow_back_wrap2D(frame1, Ft1))
			pred_frameT = tf.multiply(normalization_factor, pred_frameT)
			print("Flow Interpolation1 Graph Initialized !!!!!! ")

	return pred_frameT, F01, F10, Fdasht0, Fdasht1

def SloMo_inference_frame_step2(pred_frameT, ftmp01, FLAGS, timestamp, reuse=False):
	with tf.compat.v1.variable_scope("SloMo_model2", reuse=reuse):
		with tf.variable_scope("flow_computation2"):
			flow_comp_input = tf.concat([pred_frameT, ftmp01], axis=3)
			flow_comp_out, flow_comp_enc_out = UNet(flow_comp_input,
													output_channels=5,  # 2 channel for each flow
													first_kernel=FLAGS.first_kernel,
													second_kernel=FLAGS.second_kernel)
			flow_comp_out = lrelu(flow_comp_out)
			F01, F10, interp_t = lrelu(flow_comp_out[:,:,:,:2]), lrelu(flow_comp_out[:,:,:,2:4]), \
								 tf.sigmoid(flow_comp_out[:,:,:,4:5])
			print("Flow Computation2 Graph Initialized !!!!!! ")


		with tf.variable_scope("flow_interpolation2"):
			Fdasht0 = (-1 * (1 - interp_t) * interp_t * F01) + (interp_t * interp_t * F10)
			Fdasht1 = ((1 - interp_t) * (1 - interp_t) * F01) - (interp_t * (1 - interp_t) * F10)

			flow_interp_input = tf.concat([pred_frameT, ftmp01,
										   flow_back_wrap2D(ftmp01, Fdasht1),
										   flow_back_wrap2D(pred_frameT, Fdasht0),
										   Fdasht0, Fdasht1,
										   ], axis=3)
			flow_interp_output, _ = UNet(flow_interp_input,
													output_channels=5, 
													decoder_extra_input=flow_comp_enc_out, 
													first_kernel=FLAGS.first_kernel,
													second_kernel=FLAGS.second_kernel)
			deltaFt0, deltaFt1, Vt0 = lrelu(flow_interp_output[:, :, :, :2]), lrelu(flow_interp_output[:, :, :, 2:4]), \
									tf.sigmoid(flow_interp_output[:, :, :, 4:5])
			Vt1 = 1 - Vt0

			Ft0, Ft1 = Fdasht0 + deltaFt0, Fdasht1 + deltaFt1	

			normalization_factor = 1 / ((1 - interp_t) * Vt0 + interp_t * Vt1 + FLAGS.epsilon)
			pred_frameT2 = tf.multiply((1 - interp_t) * Vt0, flow_back_wrap2D(pred_frameT, Ft0)) + \
						  tf.multiply(interp_t * Vt1, flow_back_wrap2D(ftmp01, Ft1))
			pred_frameT2 = tf.multiply(normalization_factor, pred_frameT2)
			print("Flow Interpolation2 Graph Initialized !!!!!! ")

	return pred_frameT2, interp_t, F01, F10, Fdasht0, Fdasht1

def Slomo_inference_model(fD0, fD4, ftmpD2, fD2, FLAGS, timestamp, reuse=False):
	# Define the container of the parameter
	if FLAGS is None:
		raise ValueError('No FLAGS is provided for generator')

	Network = collections.namedtuple('Network', 'pred_frameT2, ftmpD2, l1_x2, interp_t_vis')


	enable_mask_x0 = tf.not_equal(fD0, 0.0)
	enable_mask_x8 = tf.not_equal(fD4, 0.0)
	enable_mask_x08 = tf.logical_or(enable_mask_x0, enable_mask_x8)
	enable_mask = tf.cast(enable_mask_x08, tf.float32)

	pred2_from_1, F01_1, F10_1, Fdasht0_1, Fdasht1_1 = SloMo_inference_frame_step1(fD0, fD4, FLAGS, timestamp)
	pred2, interp_t, F01_2, F10_2, Fdasht0_2, Fdasht1_2 = SloMo_inference_frame_step2(pred2_from_1, ftmpD2, FLAGS, timestamp)
	pred_frameT2 = tf.multiply(pred2, enable_mask)
	
	L1_mean_lossT2 = tf.reduce_mean(input_tensor=tf.abs(ftmpD2 - fD2))

	return Network(
		pred_frameT2=pred_frameT2,
		ftmpD2=ftmpD2, 
		l1_x2=L1_mean_lossT2,
		interp_t_vis=interp_t,
	)

# SloMo vanila model
def SloMo_model(fD0, fD4, ftmpD2, fD2, FLAGS, timestamp, reuse=False):
	# Define the container of the parameter
	if FLAGS is None:
		raise ValueError('No FLAGS is provided for generator')

	Network = collections.namedtuple('Network', 'total_loss, reconstruction_loss, advection_loss, \
												smoothness_loss, mask_loss, pred_frameT2, pred2_from_1, ftmpD2,\
												interp_t_vis, rec_loss_w_from1, rec_loss_w_tmp2, \
												grads_and_vars, train, global_step, learning_rate')
	#print(timestamp)
	pred2_from_1, F01_1, F10_1, Fdasht0_1, Fdasht1_1 = SloMo_inference_frame_step1(fD0, fD4, FLAGS, timestamp)
	pred2, interp_t, F01_2, F10_2, Fdasht0_2, Fdasht1_2 = SloMo_inference_frame_step2(pred2_from_1, ftmpD2, FLAGS, timestamp)

	warpScale = FLAGS.mask_scaling
	reconstScale = FLAGS.reconstruction_scaling
	smoothScale = FLAGS.smoothness_scaling

	# reconstruction loss
	rec_loss_w_tmp2 = reconstruction_loss_2D(ftmpD2, fD2)
	rec_loss_w_from1 = reconstruction_loss_2D(pred2_from_1, fD2)
	rec_loss_2nd = reconstruction_loss_2D(pred2, fD2)
	
	rec_loss1 = rec_loss_w_from1
	rec_loss2 = rec_loss_2nd
	
	# smoothness loss
	smooth_loss1 = smoothness_loss(F01_1, F01_1)
	smooth_loss2 = smoothness_loss(F01_2, F01_2)
	
	_frame0_temp, _frame1_temp = tf.expand_dims(fD0, axis=1), tf.expand_dims(fD4, axis=1)
	pred_temp, ref_temp = tf.expand_dims(pred2,  axis=1), tf.expand_dims(fD2,  axis=1)	
	pred_grad = gradient3D(tf.concat((tf.concat((_frame0_temp, pred_temp), axis=1), _frame1_temp), axis=1))	
	ref_grad = gradient3D(tf.concat((tf.concat((_frame0_temp, ref_temp), axis=1), _frame1_temp), axis=1))	
	gradient_loss2 = tf.reduce_mean(input_tensor=tf.abs(pred_grad - ref_grad))

	# mask loss
	wrap_loss1 = wrapping_loss(fD0, fD4, fD2, F01_1, F10_1, Fdasht0_1, Fdasht1_1) 
	wrap_loss2 = wrapping_loss(pred2_from_1, ftmpD2, fD2, F01_2, F10_2, Fdasht0_2, Fdasht1_2) 

	total_loss1 = reconstScale*rec_loss1 + smoothScale*smooth_loss1 + warpScale*wrap_loss1
	total_loss2 = reconstScale*rec_loss2 + smoothScale*smooth_loss2 + warpScale*wrap_loss2 + smoothScale*gradient_loss2

	adv_loss = tf.constant(0.0)

	with tf.compat.v1.variable_scope("global_step_and_learning_rate", reuse=reuse):
		global_step = tf.compat.v1.train.get_or_create_global_step()
		learning_rate = tf.compat.v1.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.decay_step, FLAGS.decay_rate, staircase=FLAGS.stair)
		incr_global_step = tf.compat.v1.assign(global_step, global_step + 1)

	with tf.compat.v1.variable_scope("optimizer1", reuse=reuse):
		with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
			tvars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='SloMo_model1')
			optimizer = tf.keras.optimizers.Adam(learning_rate, amsgrad=True)
			grads_and_vars = optimizer.get_gradients(total_loss1, tvars)
			train_op1 = optimizer.apply_gradients(zip(grads_and_vars, tvars))

	with tf.compat.v1.variable_scope("optimizer2", reuse=reuse):
		with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
			tvars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='SloMo_model2')
			optimizer = tf.keras.optimizers.Adam(learning_rate, amsgrad=True)
			grads_and_vars = optimizer.get_gradients(total_loss2, tvars)
			train_op2 = optimizer.apply_gradients(zip(grads_and_vars, tvars))

	return Network(
		total_loss=(total_loss1+total_loss2),
		reconstruction_loss=rec_loss_2nd,
		rec_loss_w_from1=rec_loss_w_from1, rec_loss_w_tmp2=rec_loss_w_tmp2,
		mask_loss=(wrap_loss1+wrap_loss2),
		smoothness_loss=(smooth_loss1+smooth_loss2),
		advection_loss=gradient_loss2,
		pred_frameT2=pred2,
		pred2_from_1=pred2_from_1,
		ftmpD2=ftmpD2,
		interp_t_vis=interp_t,
		grads_and_vars=grads_and_vars,
		train=tf.group((total_loss1+total_loss2), incr_global_step, train_op1, train_op2),
		global_step=global_step,
		learning_rate=learning_rate
	)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

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

# VGG19 net
def vgg_19(inputs, scope='vgg_19', reuse=False):
	with tf.variable_scope(scope, 'vgg_19', [inputs], reuse=reuse) as sc:
		end_points_collection = sc.name + '_end_points'
		# Collect outputs for conv2d, fully_connected and max_pool2d.
		with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
							outputs_collections=end_points_collection):
			net = slim.repeat(inputs, 2, slim.conv2d, 64, 3, scope='conv1', reuse=reuse)
			net = slim.max_pool2d(net, [2, 2], scope='pool1')
			net = slim.repeat(net, 2, slim.conv2d, 128, 3, scope='conv2', reuse=reuse)
			net = slim.max_pool2d(net, [2, 2], scope='pool2')
			net = slim.repeat(net, 4, slim.conv2d, 256, 3, scope='conv3', reuse=reuse)
			net = slim.max_pool2d(net, [2, 2], scope='pool3')
			net = slim.repeat(net, 4, slim.conv2d, 512, 3, scope='conv4', reuse=reuse)
			net = slim.max_pool2d(net, [2, 2], scope='pool4')
			net = slim.repeat(net, 4, slim.conv2d, 512, 3, scope='conv5', reuse=reuse)
			net = slim.max_pool2d(net, [2, 2], scope='pool5')
			# Use conv2d instead of fully_connected layers.
			# Convert end_points_collection into a end_point dict.
			end_points = slim.utils.convert_collection_to_dict(end_points_collection)

			return net, end_points
			
def flow_back_wrap2D(x, v, resize=False, normalize=False, crop=None, out="CONSTANT"):
	"""
	  Args:
		x - Input tensor [N, D, H, W, C=1]
		v - Vector flow tensor [N, D, H, W, 3], tf.float32
		(optional)
		resize - Whether to resize v as same size as x
		normalize - Whether to normalize v from scale 1 to H (or W).
					h : [-1, 1] -> [-H/2, H/2]
					w : [-1, 1] -> [-W/2, W/2]
		crop - Setting the region to sample. 6-d list [d0, d1, h0, h1, w0, w1]
		out  - Handling out of boundary value.
			   Zero value is used if out="CONSTANT".
			   Boundary values are used if out="EDGE".
	"""

	def _get_grid_array(N, H, W, h, w):
		N_i = tf.range(N)
		H_i = tf.range(h + 1, h + H + 1)
		W_i = tf.range(w + 1, w + W + 1)
		n, h, w, = tf.meshgrid(N_i, H_i, W_i, indexing='ij')
		n = tf.expand_dims(n, axis=3)  # [N, H, W, 1]
		h = tf.expand_dims(h, axis=3)  # [N, H, W, 1]
		w = tf.expand_dims(w, axis=3)  # [N, H, W, 1]
		n = tf.cast(n, tf.float32)  # [N, H, W, 1]
		h = tf.cast(h, tf.float32)  # [N, H, W, 1]
		w = tf.cast(w, tf.float32)  # [N, H, W, 1]

		return n, h, w

	shape = tf.shape(x)  # TRY : Dynamic shape
	N = shape[0]
	if crop is None:
		H_ = H = shape[1]
		W_ = W = shape[2]
		h = w = 0
	else:
		H_ = shape[1]
		W_ = shape[2]
		H = crop[1] - crop[0]
		W = crop[3] - crop[2]
		h = crop[0]
		w = crop[2]

	if resize:
		if callable(resize):
			v = resize(v, [H, W])
		else:
			v = tf.image.resize_bilinear(v, [H, W])

	if out == "CONSTANT":
		x = tf.pad(x,
				   ((0, 0), (1, 1), (1, 1), (0, 0)), mode='CONSTANT')
	elif out == "EDGE":
		x = tf.pad(x,
				   ((0, 0), (1, 1), (1, 1), (0, 0)), mode='REFLECT')

	vy, vx = tf.split(v, 2, axis=3)
	if normalize:
		vy = vy * tf.cast(H, dtype=tf.float32)  # TODO: Check why  vy * (H/2) didn't work
		vy = vy / 2
		vx = vy * tf.cast(W, dtype=tf.float32)
		vx = vx / 2

	n, h, w = _get_grid_array(N, H, W, h, w)  # [N, H, W, 3]
	vx0 = tf.floor(vx)
	vy0 = tf.floor(vy)
	vx1 = vx0 + 1
	vy1 = vy0 + 1  # [N, H, W, 1]

	H_1 = tf.cast(H_ + 1, tf.float32)
	W_1 = tf.cast(W_ + 1, tf.float32)
	iy0 = tf.clip_by_value(vy0 + h, 0., H_1)
	iy1 = tf.clip_by_value(vy1 + h, 0., H_1)
	ix0 = tf.clip_by_value(vx0 + w, 0., W_1)
	ix1 = tf.clip_by_value(vx1 + w, 0., W_1)

	i00 = tf.concat([n, iy0, ix0], 3) # y>x / x>y
	i01 = tf.concat([n, iy1, ix0], 3)
	i10 = tf.concat([n, iy0, ix1], 3)
	i11 = tf.concat([n, iy1, ix1], 3)  # [N, H, W, 3]
	i00 = tf.cast(i00, tf.int32)
	i01 = tf.cast(i01, tf.int32)
	i10 = tf.cast(i10, tf.int32)
	i11 = tf.cast(i11, tf.int32)
	x00 = tf.gather_nd(x, i00)
	x01 = tf.gather_nd(x, i01)
	x10 = tf.gather_nd(x, i10)
	x11 = tf.gather_nd(x, i11)
	w00 = tf.cast((vx1 - vx) * (vy1 - vy), tf.float32)
	w01 = tf.cast((vx1 - vx) * (vy - vy0), tf.float32)
	w10 = tf.cast((vx - vx0) * (vy1 - vy), tf.float32)
	w11 = tf.cast((vx - vx0) * (vy - vy0), tf.float32)
	output = tf.add_n([w00 * x00, w01 * x01, w10 * x10, w11 * x11])

	return output

# Reference: https://github.com/gunshi/appearance-flow-tensorflow/blob/master/bilinear_sampler.py
def flow_back_wrap3D(x, v, resize=False, normalize=False, crop=None, out="CONSTANT"):
	"""
	  Args:
		x - Input tensor [N, D, H, W, C=1]
		v - Vector flow tensor [N, D, H, W, 3], tf.float32
		(optional)
		resize - Whether to resize v as same size as x
		normalize - Whether to normalize v from scale 1 to H (or W).
					h : [-1, 1] -> [-H/2, H/2]
					w : [-1, 1] -> [-W/2, W/2]
		crop - Setting the region to sample. 6-d list [d0, d1, h0, h1, w0, w1]
		out  - Handling out of boundary value.
			   Zero value is used if out="CONSTANT".
			   Boundary values are used if out="EDGE".
	"""

	def _get_grid_array(N, D, H, W, d, h, w):
		N_i = tf.range(N)
		D_i = tf.range(d + 1, d + D + 1)
		H_i = tf.range(h + 1, h + H + 1)
		W_i = tf.range(w + 1, w + W + 1)
		n, d, h, w, = tf.meshgrid(N_i, D_i, H_i, W_i, indexing='ij')
		n = tf.expand_dims(n, axis=4)  # [N, D, H, W, 1]
		d = tf.expand_dims(d, axis=4)  # [N, D, H, W, 1]
		h = tf.expand_dims(h, axis=4)  # [N, D, H, W, 1]
		w = tf.expand_dims(w, axis=4)  # [N, D, H, W, 1]
		n = tf.cast(n, tf.float32)  # [N, D, H, W, 1]
		d = tf.cast(d, tf.float32)  # [N, D, H, W, 1]
		h = tf.cast(h, tf.float32)  # [N, D, H, W, 1]
		w = tf.cast(w, tf.float32)  # [N, D, H, W, 1]

		return n, d, h, w

	shape = tf.shape(input=x)  # TRY : Dynamic shape
	N = shape[0]
	if crop is None:
		D_ = D = shape[1]
		H_ = H = shape[2]
		W_ = W = shape[3]
		d = h = w = 0
	else:
		D_ = shape[1]
		H_ = shape[2]
		W_ = shape[3]
		D = crop[1] - crop[0]
		H = crop[3] - crop[2]
		W = crop[5] - crop[4]
		d = crop[0]
		h = crop[2]
		w = crop[4]

	if resize:
		if callable(resize):
			v = resize(v, [D, H, W])
		else:
			v = tf.image.resize(v, [D, H, W], method=tf.image.ResizeMethod.BILINEAR)

	if out == "CONSTANT":
		x = tf.pad(tensor=x,
				   paddings=((0, 0), (1, 1), (1, 1), (1, 1), (0, 0)), mode='CONSTANT')
	elif out == "EDGE":
		x = tf.pad(tensor=x,
				   paddings=((0, 0), (1, 1), (1, 1), (1, 1), (0, 0)), mode='REFLECT')

	vz, vy, vx = tf.split(v, 3, axis=4)
	if normalize:
		vz = vz * tf.cast(D, dtype=tf.float32)
		vz = vz / 3
		vy = vy * tf.cast(H, dtype=tf.float32)
		vy = vy / 3
		vx = vy * tf.cast(W, dtype=tf.float32)
		vx = vx / 3

	n, d, h, w = _get_grid_array(N, D, H, W, d, h, w)  # [N, D, H, W, 3]
	vx0 = tf.floor(vx)
	vy0 = tf.floor(vy)
	vz0 = tf.floor(vz)
	
	vx1 = vx0 + 1  # [N, D, H, W, 1]
	vy1 = vy0 + 1 
	vz1 = vz0 + 1

	D_1 = tf.cast(D_ + 1, tf.float32)
	H_1 = tf.cast(H_ + 1, tf.float32)
	W_1 = tf.cast(W_ + 1, tf.float32)

	iz0 = tf.clip_by_value(vz0 + d, 0., D_1)
	iz1 = tf.clip_by_value(vz1 + d, 0., D_1)
	iy0 = tf.clip_by_value(vy0 + h, 0., H_1)
	iy1 = tf.clip_by_value(vy1 + h, 0., H_1)
	ix0 = tf.clip_by_value(vx0 + w, 0., W_1)
	ix1 = tf.clip_by_value(vx1 + w, 0., W_1)

	i000 = tf.concat([n, iz0, iy0, ix0], 4)
	i001 = tf.concat([n, iz0, iy0, ix1], 4)
	i010 = tf.concat([n, iz0, iy1, ix0], 4)
	i011 = tf.concat([n, iz0, iy1, ix1], 4)
	i100 = tf.concat([n, iz1, iy0, ix0], 4)
	i101 = tf.concat([n, iz1, iy0, ix1], 4)
	i110 = tf.concat([n, iz1, iy1, ix0], 4)
	i111 = tf.concat([n, iz1, iy1, ix1], 4)

	i000 = tf.cast(i000, tf.int32)
	i001 = tf.cast(i001, tf.int32)
	i010 = tf.cast(i010, tf.int32)
	i011 = tf.cast(i011, tf.int32)
	i100 = tf.cast(i100, tf.int32)
	i101 = tf.cast(i101, tf.int32)
	i110 = tf.cast(i110, tf.int32)
	i111 = tf.cast(i111, tf.int32)
	
	x000 = tf.gather_nd(x, i000)
	x001 = tf.gather_nd(x, i001)
	x010 = tf.gather_nd(x, i010)
	x011 = tf.gather_nd(x, i011)
	x100 = tf.gather_nd(x, i100)
	x101 = tf.gather_nd(x, i101)
	x110 = tf.gather_nd(x, i110)
	x111 = tf.gather_nd(x, i111)

	w000 = tf.cast((vz1 - vz) * (vx1 - vx) * (vy1 - vy), tf.float32)
	w001 = tf.cast((vz1 - vz) * (vx1 - vx) * (vy - vy0), tf.float32)
	w010 = tf.cast((vz1 - vz) * (vx - vx0) * (vy1 - vy), tf.float32)
	w011 = tf.cast((vz1 - vz) * (vx - vx0) * (vy - vy0), tf.float32)

	w100 = tf.cast((vz - vz0) * (vx1 - vx) * (vy1 - vy), tf.float32)
	w101 = tf.cast((vz - vz0) * (vx1 - vx) * (vy - vy0), tf.float32)
	w110 = tf.cast((vz - vz0) * (vx - vx0) * (vy1 - vy), tf.float32)
	w111 = tf.cast((vz - vz0) * (vx - vx0) * (vy - vy0), tf.float32)

	output = tf.add_n([w000 * x000, w001 * x001, w010 * x010, w011 * x011, w100 * x100, w101 * x101, w110 * x110, w111 * x111])

	return output

def compute_psnr(ref, target):
	ref = tf.cast(ref, tf.float32)
	target = tf.cast(target, tf.float32)
	diff = target - ref
	sqr = tf.multiply(diff, diff)
	err = tf.reduce_sum(input_tensor=sqr)
	v = tf.shape(input=diff)[0] * tf.shape(input=diff)[1] * tf.shape(input=diff)[2] * tf.shape(input=diff)[3]
	mse = err / tf.cast(v, tf.float32)
	psnr = 10. * (tf.math.log(255. * 255. / mse) / tf.math.log(10.))

	return psnr

# grid interpolation method, order: only linear tested
# for velocity, macgridbatch.shape should be [b,z,y,x,3] (in 2D, should be [b,1,ny,nx,3])
# for density , macgridsource.shape should be [z,y,x,1] (in 2D, should be [1,ny,nx,1])
def gridInterpolBatch(macgridbatch, targetshape, order=1):
	assert (targetshape[-1] == macgridbatch.shape[-1])  # no interpolation between channels
	assert (len(targetshape) == 5 and len(macgridbatch.shape) == 5)
	dim = 3
	if (macgridbatch.shape[1] == 1 and targetshape[1] == 1):  dim = 2
	
	x_ = np.linspace(0.5, targetshape[3] - 0.5, targetshape[3])
	y_ = np.linspace(0.5, targetshape[2] - 0.5, targetshape[2])
	z_ = np.linspace(0.5, targetshape[1] - 0.5, targetshape[1])
	c_ = np.linspace(0, targetshape[4] - 1, targetshape[4])  # no interpolation between channels
	b_ = np.linspace(0, targetshape[0] - 1, targetshape[0])  # no interpolation between batches
	
	b, z, y, x, c = np.meshgrid(b_, z_, y_, x_, c_, indexing='ij')
	
	# scale
	fx = float(macgridbatch.shape[3]) / targetshape[3]
	fy = float(macgridbatch.shape[2]) / targetshape[2]
	fz = float(macgridbatch.shape[1]) / targetshape[1]
	
	mactargetbatch = scipy.ndimage.map_coordinates(macgridbatch, [b, z * fz, y * fy, x * fx, c], order=order, mode='reflect')
	
	return mactargetbatch;

# macgrid_batch shape b, z, y, x, 3
# return a matrix in size [b,z,y,x,3] ( 2D: [b,y,x,2]), last channel in z-y-x order!(y-x order for 2D)
def getMACGridCenteredBatch(macgrid_batch, is3D):
	_bn, _zn, _yn, _xn, _cn = macgrid_batch.shape
	
	valid_idx = list(range(1, _xn))
	valid_idx.append(_xn - 1)
	add_x = macgrid_batch.take(valid_idx, axis=3)[:, :, :, :, 0]  # shape, [b,z,y,x]
	valid_idx = list(range(1, _yn))
	valid_idx.append(_yn - 1)
	add_y = macgrid_batch.take(valid_idx, axis=2)[:, :, :, :, 1]  # shape, [b,z,y,x]
	
	add_y = add_y.reshape([_bn, _zn, _yn, _xn, 1])
	add_x = add_x.reshape([_bn, _zn, _yn, _xn, 1])
	if (is3D):
		valid_idx = list(range(1, _zn))
		valid_idx.append(_zn - 1)
		add_z = macgrid_batch.take(valid_idx, axis=1)[:, :, :, :, 2]  # shape, [b,z,y,x]
		add_z = add_z.reshape([_bn, _zn, _yn, _xn, 1])
		resultgrid = 0.5 * (macgrid_batch[:, :, :, :, ::-1] + np.concatenate((add_z, add_y, add_x), axis=-1))
		return resultgrid.reshape([_bn, _zn, _yn, _xn, 3])
	
	resultgrid = 0.5 * (macgrid_batch[:, :, :, :, -2::-1] + np.concatenate((add_y, add_x), axis=4))
	return resultgrid.reshape([_bn, _yn, _xn, 2])

# macgrid_batch shape b, z, y, x, 3 ( b,1,y,x,3 for 2D )
# return the re-sampling positions as a matrix, in size of [b,z,y,x,3] ( 2D: [b,y,x,2])
def getSemiLagrPosBatch(macgrid_batch, dt, cube_len_output=-1):  # check interpolation later
	assert (len(macgrid_batch.shape) == 5)
	_bn, _zn, _yn, _xn, _cn = macgrid_batch.shape
	assert (_cn == 3)
	is3D = (_zn > 1)
	if (cube_len_output == -1): cube_len_output = _xn
	factor = float(_xn) / cube_len_output
	x_ = np.linspace(0.5, int(_xn / factor + 0.5) - 0.5, int(_xn / factor + 0.5))
	y_ = np.linspace(0.5, int(_yn / factor + 0.5) - 0.5, int(_yn / factor + 0.5))
	interp_shape = [_bn, int(_zn / factor + 0.5), int(_yn / factor + 0.5), int(_xn / factor + 0.5), 3]
	if (not is3D): interp_shape[1] = 1
	
	if (is3D):
		z_ = np.linspace(0.5, int(_zn / factor + 0.5) - 0.5, int(_zn / factor + 0.5))
		z, y, x = np.meshgrid(z_, y_, x_, indexing='ij')
		posArray = np.stack((z, y, x), axis=-1)  # shape, z,y,x,3
		tarshape = [1, int(_zn / factor + 0.5), int(_yn / factor + 0.5), int(_xn / factor + 0.5), 3]
	else:
		y, x = np.meshgrid(y_, x_, indexing='ij')
		posArray = np.stack((y, x), axis=-1)  # shape, y,x,2
		tarshape = [1, int(_yn / factor + 0.5), int(_xn / factor + 0.5), 2]
	posArray = posArray.reshape(tarshape)
	if (cube_len_output == _xn):
		return (posArray - getMACGridCenteredBatch(macgrid_batch, is3D) * dt)
	# interpolate first
	
	inter_mac_batch = gridInterpolBatch(macgrid_batch, interp_shape, 1)
	inter_mac_batch = getMACGridCenteredBatch(inter_mac_batch, is3D) / factor
	return (posArray - (inter_mac_batch) * dt)


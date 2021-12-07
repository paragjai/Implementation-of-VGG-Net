from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras import backend as K

class PJNet:
	@staticmethod
	def build(width, height, depth, classes):

		# conv2d_1 : input : (None, 96, 96, 3)
		#		   : output : (None, 96, 96, 3)
		#		   : filter1 : 2x2
		#		   : filter2 : 2x2

		# act_1    : input : (None, 96, 96, 2)
		#		   : output : (None, 96, 96, 2)
		#		   : function : relu

		# batchNorm_1 : input : (None, 96, 96, 2)
		#             : output : (None, 96, 96, 2)

		# maxPool_1 : input : (None, 96, 96, 2)
		#			: output : (None, 24, 24, 2)
		#			: filter : 4x4

		# flatten_1 : input : (None, 24, 24, 2)
		#			: output : 24 x 24 x 2 = 576 x 2 = 1152 => (None, 1152)

		# dropout_1 : input : (None, 1152)
		#			: output : (None, 1152)

		# dense_1 : input : (None, 1152)
		#		  : output : (None, 3)

		model = Sequential()
		chanDim = -1
		inputShape = (width, height, depth) # (96, 96, 3)
		
		if K.image_data_format() == "channels_first":
			inputShape = (depth, width, height)
			chanDim = 1

		conv2d_1 = Conv2D(2, (2,2), padding="same", input_shape=inputShape)
		act_1 = Activation("relu")
		batchNorm_1 = BatchNormalization(axis = chanDim)
		maxPool_1 = MaxPooling2D(pool_size=(4,4))
		flatten_1 = Flatten()
		dropout_1 = Dropout(0.5)
		dense_1 = Dense(classes)

		layers = [conv2d_1, act_1, batchNorm_1, maxPool_1, flatten_1, dropout_1, dense_1]
		model.add(con2d_1)
		model.add(act_1)
		model.add(batchNorm_1)
		model.add(maxPool_1)
		model.add(flatten_1)
		model.add(dropout_1)
		model.add(dense_1)

		return model
		
	
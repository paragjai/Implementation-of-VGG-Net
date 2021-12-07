from keras.models import Sequential
from keras import backend as K
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dropout
#input image : 96 x 96 x 3

class SmallerVGGNet:
	@staticmethod
	def build(width, height, depth, classes):
		model = Sequential()
		inputShape = (width, height, depth) # (96, 96, 3)
		chanDim = -1
		
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1
			
		# By default we have an "input layer" which has input as (None, 96, 96, 3)
		#										  and output as (None, 96, 96, 3)
		
		
			
		# CONV => RELU => POOL
		# CONV2D layer 1: input : (None, 96, 96, 3)
		#			   : output : (None, 96, 96, 32) # We get ONLY 32 activation maps because we have 32 filters. 
		#It is not like we have 32 activation maps per input channel/depth.
		
		# Activation/RELU layer 1 : input : (None, 96, 96, 32)
		#						: output : (None, 96, 96, 32)
		
		# BatchNormalization 1: input : (None, 96, 96, 32)
		#					 : output : (None, 96, 96, 32)
		
		
		# MaxPooling Layer 1 : input : (None, 96, 96, 32)
		#				   : output : (None, 32, 32, 32)

		# Dropout Layer 1 : 0.25
		model.add(Conv2D(32, (3,3), padding="same", input_shape = inputShape)) # CONV layer has 32 filters (reflect this in output)
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(3,3)))
		model.add(Dropout(0.25))
		
		
		# Conv layer 2: input : (None, 32, 32, 32)
		#			 : output : (None, 32, 32, 64) #Point to be noted. It is 64 here. We get 64 activation maps for each of 32 input #activation frames we receive from the previous layer. NO. It is infact just 64 activation maps because we are using 
		#64 filters. See the input as ONE input image of width = 32 and height = 32 with depth/channels = 32
		# o = (i + 2p - k)/s + 1 => o = i, so p = ((o - 1)*s + k - i)/2
		# i = 32, o = 32, s = 1, k = 3
		# p = ( ( 32 - 1)*1 + 3 - 32 ) / 2 = (30 + 3 - 32)/2  = 0.5 

		# Activation layer 2: input : (None, 32, 32, 64)
		# 				   : output : (None, 32, 32, 64)
		
		# BatchNormalization layer 2: input : (None, 32, 32, 64)
		#				     : output : (None, 32, 32, 64)
		
		model.add(Conv2D(32*2, (3,3), padding="same", input_shape = inputShape)) # No of filters increased as we go deeper
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))


		# Conv Layer 3: input : (None, 32, 32, 64) # Think of it as 1 image of shape (16, 16) with 64 channels. But ONE image.
		#			 : output : (None, 32, 32, 64)
		
		# Activation Layer 3 input : (None, 32, 32, 64)
		#				   : output : (None, 32, 32, 64)
		
		# BatchNormalization layer 3: input : (None, 32, 32, 64)
		#				    : output : (None, 32, 32, 64)

		model.add(Conv2D(32*2, (3,3), padding="same", input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		
		# MaxPooling Layer 2 : input : (None, 32, 32, 64)
		#			     : output : (None, 16, 16, 64)

		# Dropout Layer 2 : input : (None, 16, 16, 64)
		#			: output : (None, 16, 16, 64)
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.25))

		# Conv Layer 4 : input : (None, 16, 16, 64) # One image of 64 channels.
		#			 : output :(None, 16, 16, 128)
		
		# Activation Layer 4: input : (None, 16, 16, 128)
		#				   : output : (None, 16, 16, 128)
		
		# BatchNormalization layer 4: input : (None, 16, 16, 128)
		#				     : output : (None, 16, 16, 128)

		model.add(Conv2D(32*2*2, (3,3), padding="same", input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))

		# Conv Layer 5 : input : (None, 16, 16, 128) # One image of 128 channels
		#		     : output : (None, 16, 16, 128)

		# Activation Layer 5 : input : (None, 16, 16, 128)
		#			  : output : (None, 16, 16, 128)
		
		# BatchNormalization Layer 5 : input : (None, 16, 16, 128)
		#				       : output : (None, 16, 16, 128)

		model.add(Conv2D(32*2*2, (3,3), padding="same", input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		
		# MaxPool2D layer 3: input : (None, 16, 16, 128)
		#			 : output : (None, 8, 8, 128)

		# Dropout layer 3 : input : (None,  8, 8, 128)
		#			: output : (None, 8, 8, 128)

		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.25))
		
		# Flatten 1: input : (None, 8, 8, 128)
		#		  : output : (None, 8*8*128 = 8192)
		
		# Dense Layer 1: input : (None, 8192)
		#		: output : (None, 1024)
		
		# Activation Layer 6: input : (None, 1024)
		#				   : output : (None, 1024)
		
		# BatchNormalization 6: input : (None, 1024)
		#					 : output : (None, 1024)
		
		# Dropout 4: input : (None, 1024)
		#		  : output : (None, 1024)
		
		# Dense 2: input : (None, 1024)
		#		: output : (None, 3) # Because we have 3 classes
		
		# Activation Layer 7: input : (None, 3)
		#				   : output : (None, 3)
		
		

		model.add(Flatten())
		model.add(Dense(1024))	
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Dropout(0.5))
		
		model.add(Dense(classes)) # For training we will keep the final layer as well
		model.add(Activation("softmax"))

		return model

		

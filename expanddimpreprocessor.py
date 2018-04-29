import numpy as np

class ExpandDimPreprocessor:
	def __init__(self, axis = 0):
		self.axis = axis
		
	def preprocess(self, image):
		return np.expand_dims(image, axis = self.axis)
		

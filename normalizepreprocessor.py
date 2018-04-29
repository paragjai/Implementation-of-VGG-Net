import numpy as np

class NormalizePreprocessor:
	def __init__(self, normalizing_factor = 255):
		self.normalizing_factor = normalizing_factor
		
	def preprocess(self, image):
		return np.array(image, "float")/self.normalizing_factor
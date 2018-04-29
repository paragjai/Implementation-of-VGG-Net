import cv2

class SimpleResizePreprocessor:
		#We are designing our network for width=96, height=96 in this case
		def __init__(self, width=96, height=96, inter=cv2.INTER_AREA):
			self.width = width
			self.height = height
			self.inter = inter
			
		def preprocess(self, image):
			return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
			
		
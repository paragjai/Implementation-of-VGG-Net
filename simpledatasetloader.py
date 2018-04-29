import cv2
import os
import numpy as np

class SimpleDatasetLoader:
	def __init__(self, preprocessors = None):
		self.preprocessors = preprocessors
		
		if self.preprocessors is None:
			self.preprocessors = []
			
	def load(self, imagePaths, verbose = -1):
		data = []
		labels = []
		usefulImagePaths = []
		
		for (i,imagePath) in enumerate(imagePaths):
			#print(imagePath)
			image = cv2.imread(imagePath)
			#print(image.shape)
			if image is not None: # We have problem some images even though the path is right and image ends with the validExtensions.
				label = imagePath.split('\\')[-2] #Windows specific, should ideally use os.path.sep instead of "\\"
				#print(label)
				#print("---")
				if self.preprocessors is not None:
					for p in self.preprocessors:
						image = p.preprocess(image)
					usefulImagePaths.append(imagePath)
					data.append(image)
					labels.append(label)
			
			if verbose > 0 and i > 0 and (i + 1)%verbose == 0:
				print("[INFO] processed {}/{}".format(i+1, len(imagePaths)))
				#print(data[-1])
		
		#images are returned along with their labels
		return (usefulImagePaths, np.array(data), np.array(labels))
		
	''' Test one image at a time '''
	def loadTest(self, imagePath):
			image = cv2.imread(imagePath)
			#print(image.shape)
			if image is not None: # We may have problem loading some images even though the path is right and image ends with the validExtensions.
				if self.preprocessors is not None:
					for p in self.preprocessors:
						image = p.preprocess(image)
					
			else:
				print("Could not load the image")
				exit(1)
					
					
			
				
		
			#images are returned for testing
			return image
		
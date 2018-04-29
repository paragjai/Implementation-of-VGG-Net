'''Command to run:
python extract_feature_vectors.py --dataset datasets --model one_epoch.model
'''
import argparse
import os
from simpleresizepreprocessor import SimpleResizePreprocessor
#from imagetoarraypreprocessor import ImageToArrayPreprocessor # Converts PIL image to Numpy arrays
from normalizepreprocessor import NormalizePreprocessor 
from expanddimpreprocessor import ExpandDimPreprocessor # Required for extracting CNN codes since we are passing one image at a time!
from extractcnncodesasfeatures import ExtractCNNCodeAsFeatures
from simpledatasetloader import SimpleDatasetLoader
from keras.models import load_model # To load the trained model. We used keras.models import Sequential while building the model.
import random
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required = True, help = "path of the dataset")
ap.add_argument("-m", "--model", required = True, help = "path to the trained model")
args = vars(ap.parse_args())

# initialize the path of Images to be loaded.
imagePaths = []

# grab the image paths and randomly shuffle them
print("[INFO] loading image paths ..")

validExtensions = ['jpg', 'jpeg', 'bmp', 'png']
for pathName, folderNames, fileNames in os.walk(args['dataset']):
	for fileName in fileNames:
		if fileName.split(".")[-1] in validExtensions:
			#print(pathName, fileName)
			imagePath = pathName+"\\"+fileName # Windows specific, should ideally use os.path.sep instead of "\\"
			imagePaths.append(imagePath)
			
print("imagePaths[0:5]", imagePaths[0:5])

random.seed(42)
random.shuffle(imagePaths)

print("[INFO] loading trained model ... ")
model = load_model(args["model"])
print("trained model loaded.")


#pre-process the image for classification
srp = SimpleResizePreprocessor(width = 96, height = 96)
nop = NormalizePreprocessor(normalizing_factor = 255.)

# iap = ImageToArrayPreprocessor() # We don't really need it. Since we are passing numpy array itself as input
eap = ExpandDimPreprocessor(axis = 0) # Though we have the entire dataset but we are still passing one image at a time to the model.
eccf = ExtractCNNCodeAsFeatures(model, layer_index = 25) # 25 we are hard-coding. This one can see from model.summary() and choose appropriately.
preprocessors = [srp, nop, eap, eccf] 
sdl = SimpleDatasetLoader(preprocessors)
print("[INFO] loading images to disk ... ")
usefulImagePaths, feature_vectors, labels = sdl.load(imagePaths, verbose=100)

print("no of useful images: ", feature_vectors.shape)
print("no of useful labels: ", labels.shape)
#print("feature_vectors[0]")
#print(feature_vectors[0])
#print("feature_vectors[0].shape")
#print(feature_vectors[0].shape)
#print("labels[0]")
#print(labels[0])

labels = np.expand_dims(labels, axis = 1)
''' Preparing data to write to a csv '''
feature_vectors_with_corresponding_labels = np.hstack((feature_vectors, labels))

print("feature_vectors_with_corresponding_labels.shape")
print(feature_vectors_with_corresponding_labels.shape)


df = pd.DataFrame(feature_vectors_with_corresponding_labels)
df.to_csv("features_vectors_with_corresponding_labels.csv")
print("features and label written to features_vectors_with_corresponding_labels.csv")

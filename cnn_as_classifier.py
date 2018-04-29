'''Command to run:
python cnn_as_classifier.py --imagePath cats_00001.jpg --model one_epoch.model --labelbin label_binarizer.pickle
'''

import argparse
from simpleresizepreprocessor import SimpleResizePreprocessor
# from imagetoarraypreprocessor import ImageToArrayPreprocessor # Required to convert PIL images to numpy arrays. We don't need it. We are directly loading image as a numpy array.
from normalizepreprocessor import NormalizePreprocessor 
from expanddimpreprocessor import ExpandDimPreprocessor # Required for classification since we are passing one image at a time!
from simpledatasetloader import SimpleDatasetLoader
from keras.models import load_model # To load the trained model. We used keras.models import Sequential while building the model.
import pickle
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--imagePath", required = True, help = "path of the image to classify")
ap.add_argument("-m", "--model", required = True, help = "path to the trained model")
ap.add_argument("-l", "--labelbin", required = True, help = "path to the saved labels") # We need it to get the name of classes (ex: whether the image passed belongs to class 1 or class 2 or class 3 etc.)
#I don't know why we would need the labels
args = vars(ap.parse_args())

#pre-process the image for classification
srp = SimpleResizePreprocessor(width = 96, height = 96)
nop = NormalizePreprocessor(normalizing_factor = 255.)
#iap = ImageToArrayPreprocessor()
edp = ExpandDimPreprocessor(axis = 0) # We do it for testing!

preprocessors = [srp, nop, edp] 
sdl = SimpleDatasetLoader(preprocessors)
preprocessed_image = sdl.loadTest(args["imagePath"]) # Path of a single image at a time and NOT entire dataset (unlike sdl.load())

print("Preprocessed image for keras to get classified: " ,preprocessed_image.shape)

print("[INFO] loading trained model ... ")
model = load_model(args["model"])
print(model.summary())
print("trained model loaded.")

print("[INFO] loading labels ...")
lb = pickle.loads(open(args["labelbin"], "rb").read())
print("labels loaded.")

print("[INFO] classifying image ... ")
proba = model.predict(preprocessed_image)[0]
idx = np.argmax(proba)
predicted_label = lb.classes_[idx]
print("predicted label: ", predicted_label)
print("image classified.")



'''Command to run:
python train.py --dataset datasets --model one_epoch.model --labelbin label_binarizer.pickle
'''

import argparse
import os
import random
from simpleresizepreprocessor import SimpleResizePreprocessor
# from imagetoarraypreprocessor import ImageToArrayPreprocessor # Not required as we are already passing numpy arrays as input
# from expanddimpreprocessor import ExpandDimPreprocessor # Not required for training!
from normalizepreprocessor import NormalizePreprocessor 
from simpledatasetloader import SimpleDatasetLoader
from sklearn.preprocessing import LabelEncoder # convert string labels to integers
from sklearn.preprocessing import LabelBinarizer # convert string labels to one-hot encoded labels
from sklearn.model_selection import train_test_split
from smallerVGGNet import SmallerVGGNet
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator # Don't have much of idea about this
import pickle
# import matplotlib # Have to fix matplotlib part later. PENDING FOR NOW!!
# matplotlib.use("Agg")

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset (i.e., directory of images)") # one dataset containing both training and testing data. We do the split ourself.
ap.add_argument("-m", "--model", required=True, help="path to output model")
ap.add_argument("-l", "--labelbin", required=True, help="path to output label binarizer")
#ap.add_arguments("-p", "--plot", type=str, hep="path to output accuracy/loss plot")
args = vars(ap.parse_args())

epochs = 1
init_lr = 1e-3 # 0.001
batch_size = 32
image_dims = (96, 96, 3)

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

srp = SimpleResizePreprocessor(width=96, height=96)
#iap = ImageToArrayPreprocessor() # keras specific preprocessor
#edp = ExpandDimPreprocessor(axis=0) # Not required for training!
nop = NormalizePreprocessor(normalizing_factor = 255.)
preprocessors = [srp, nop]
sdl = SimpleDatasetLoader(preprocessors)

print("[INFO] loading images to disk ... ")
usefulImagePaths, data, labels = sdl.load(imagePaths, verbose=100)

print("no of useful images: ", data.shape)
print("no of useful labels: ", labels.shape)


le = LabelEncoder()	
integer_labels = le.fit_transform(labels)

print("[INFO] one hot encoding labels ... ")
lb = LabelBinarizer()
one_hot_encoded_labels = lb.fit_transform(labels)
print("labels encoded.")
# 75% of the data for training
# 25% of the data for testing
print("[INFO] spliting the dataset to train and test ... ")
(trainX, testX, trainY, testY) = train_test_split(data, one_hot_encoded_labels, test_size=0.25, random_state = 42)
print("dataset got split.")



print("[INFO] compiling model ... ")
# we are making this model for image width = 96 and image height = 96 
model = SmallerVGGNet.build(width = 96, height = 96, depth = 3, classes = len(lb.classes_))
opt = Adam(lr = init_lr)
model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])
print("model compiled.")


print("[INFO] training network ... ")
# Used for generating more training data.
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")
	
# fit_generator returns the history object. Metrics are stored in the history member of the object returned.
# keras.callbacks.History()
# Callback that records events into a History object.
# This callback is automatically applied to every Keras model. The History object gets returned by the fit method of models.
	
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=batch_size), validation_data = (testX, testY), steps_per_epoch = len(trainX),epochs = epochs, verbose = 1) # we are passing trainX, one_hot_encoded_trainY, testX, one_hot_encoded_testY

print("[INFO] serializing/saving trained network ... ")
model.save(args["model"])
print("Saved the trained model to " + args["model"])

print("[INFO] serializing/saving one-hot encoded labels/LabelBinarizer object ... ") #We don't really need to save one-hot encoded labels. 
f = open(args["labelbin"], "wb") # We are doing this to use lb.classes_ which has the integral (integer) mapping
f.write(pickle.dumps(lb))
f.close()
print("Saved the one-hot encoded labels/labelBinarizer object " + args["model"])


plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label = "train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label = "val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label = "train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training loss and accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc = "upper left")
plt.savefig(args["plot"])

	

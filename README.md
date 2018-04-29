<h2> Training Phase : train.py</h2>

<ul> Files for Preprocessing</ul>
<li><b>simpleresizepreprocessor.py</b>
Class implemented to resize the input image
</li>
<li><b>normalizepreprocessor.py</b>
Class implemented to normalize an 8 bit RGB image
</li>
</ul>

<b>simpledatasetloader.py</b>
Class implemented to load the dataset to disk and apply the preprocessor to each of the image.

<b>smallerVGGNet.py</b>
VGGNet model is implemented in this file using keras.

Saves the model and labels to the disk.

Cmd to run train.py:
python train.py --dataset datasets --model one_epoch.model --labelbin label_binarizer.pickle

<h2> Testing/Classification Phase : cnn_as_a_classifier.py </h2>

<ul> Files for Preprocessing</ul>
<li><b>simpleresizepreprocessor.py</b>
Class implemented to resize the input image
</li>
<li><b>normalizepreprocessor.py</b>
Class implemented to normalize an 8 bit RGB image
</li>
<li><b>expanddimpreprocessor.py</b>
Class implemented to change the shape the input image from (some_number, some_number, 3) to (1, some_number, some_number, 3).
It is so because keras expects input to be of 4 dimensions. We don't do this when we are passing entire dataset (during training) as input to keras because then the shape of input is (no of images in the dataset, some_number, some_number, 3). 
</li>
</ul>

<b>simpledatasetloader.py</b>
Class implemented to load the dataset to disk and apply the preprocessor to each of the image.

Loads the model and classifies the image as either cats or dogs or panda ( 3 classes we trained the model for ).

Cmd to run cnn_as_a_classifier.py:
python cnn_as_classifier.py --imagePath cats_00001.jpg --model one_epoch.model --labelbin label_binarizer.pickle



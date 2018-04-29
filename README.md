<h2> Training Phase : train.py</h2>

Saves the model and labels to the disk.<br> Generates .model and .pickle file.

<h3> Files required for preprocessing while training </h3>

<ul> 
<li><b>simpleresizepreprocessor.py :</b> 
Class implemented to resize the input image
</li>
<li><b>normalizepreprocessor.py :</b> 
Class implemented to normalize an 8 bit RGB image
</li>
</ul>

<b>simpledatasetloader.py :</b>
Class implemented to load the dataset to disk and apply the preprocessor to each of the image.

<b>smallerVGGNet.py :</b>
VGGNet model is implemented in this file using keras.


Cmd to run train.py <br>
<b>python train.py --dataset datasets --model one_epoch.model --labelbin label_binarizer.pickle</b>

<h2> Testing/Classification Phase : cnn_as_classifier.py </h2>

Loads the model and classifies the image as either cats or dogs or panda ( 3 classes we trained the model for ). Uses one_epoch.model and label_binarizer.pickle.

<h3> Files required for preprocessing while testing </h3>

<ul> 
<li><b>simpleresizepreprocessor.py :</b> 
Class implemented to resize the input image
</li>
<li><b>normalizepreprocessor.py :</b> 
Class implemented to normalize an 8 bit RGB image
</li>
<li><b>expanddimpreprocessor.py :</b>
Class implemented to change the shape the input image from (some_number, some_number, 3) to (1, some_number, some_number, 3).
It is so because keras expects input to be of 4 dimensions. We don't do this when we are passing entire dataset (during training) as input to keras because then the shape of input is (no of images in the dataset, some_number, some_number, 3). 
</li>
</ul>

<b>simpledatasetloader.py :</b>
Class implemented to load the dataset to disk and apply the preprocessor to each of the image.

Cmd to run cnn_as_a_classifier.py <br>
<b>python cnn_as_classifier.py --imagePath cats_00001.jpg --model one_epoch.model --labelbin label_binarizer.pickle</b>

<h2> Extracting features for Transfer Learning : extract_feature_vectors.py </h2>

Loads the model and generates a csv file containing 1024 dimensional feature vector along with labels.<br> Generates features_vectors_with_corresponding_labels.csv.<br> Uses one_epoch.model and label_binarizer.pickle.<br>
Read <b>Steps_For_Transfer_Learning.txt</b> for more details.

<ul> 
<li><b>simpleresizepreprocessor.py :</b> 
Class implemented to resize the input image
</li>
<li><b>normalizepreprocessor.py :</b> 
Class implemented to normalize an 8 bit RGB image
</li>
<li><b>expanddimpreprocessor.py :</b>
Class implemented to change the shape the input image from (some_number, some_number, 3) to (1, some_number, some_number, 3).
It is so because keras expects input to be of 4 dimensions. Though we are passing entire dataset (during training) as input to extract_feature_vectors.py, still one image at a time is passed to trained keras model. Therefore we expand the dimension of each of the image.
</li>
<li><b>extractcnncodesasfeatures.py :</b>
Class implemented to extract 1024 dimensional vector (CNN code) from the image. This class is also capable of giving output of each of the layers of the network. But we have commented that part of the code so that we can just extract the CNN codes.
</li>
</ul>

<b>simpledatasetloader.py :</b>
Class implemented to load the dataset to disk and apply the preprocessor to each of the image.

Cmd to run extract_feature_vectors.py <br>
<b>python extract_feature_vectors.py --dataset datasets --model one_epoch.model</b>

# **Traffic Sign Recognition** 


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/emilwareus/Trafic_Sign_Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Numpy was used to calculate some basic statistics of the project. 

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3) (RBG)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.


![Number for examples of each class in the training set](https://github.com/emilwareus/Trafic_Sign_Classifier/blob/master/img_for_redme/Capture.PNG)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As color seems like an importation feature for a trafic sign, grayscale convertion was not used. Instead a one-to-one convolutional layer was put as the first layer for the inputs. This will optimize the color-space for the rest of the model. 

The only pre-processing done on the images was normalization to +- 0.5. This code was used to performe the normalization. 
X_train_norm = X_train/255-0.5

Normalization was performed to reduce the size of the paramteters, which helps training. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 1x1 with RELU     	| 1x1 stride, same padding, outputs 32x32x3 	|
| Inception with RELU 	      	| outputs 32x32x64 				|
| Max Pooling	    | Stride 2,2 , ksize 3,3 , same padding, outputs 16x16x64 	|
| Inception with RELU 	      	| outputs 16x16x128 				|
| Max Pooling	    | Stride 2,2 , ksize 3,3 , same padding, outputs 8x8x128 	|
| Fully connected with RELU		| 8192 to 700	|
| Dropout			| 0.5 keep_prob during training        									|
|	Fully connected					|	700 to 43|
|	Softmax					|												|
 
 
 Here is the graph visualized in Tensorboard:
 
![Graph](https://github.com/emilwareus/Trafic_Sign_Classifier/blob/master/img_for_redme/Graph.PNG)




#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model I used the Adam optimizer with a learning rate of 0.001. The learning rate was decresed with 20% every 5th epoch. The batch size was small due to a larg model. So the a batch size of 64 seemed to work best on my computer. This parameter could have been incresed a bit, but this size was a problem before max pooling was implemented and the weights took up a lot of memory. The epochs was first set to 25, but training was canceled after 15 epochs due to decresing the risk of overfitting. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

First off I tried the LeNet architecture, but noticed that the conversion was slow and training was quite slow as well. So I tried adding more convolutions and a one-to-one layer. This incresed the performance a bit, but training was still slow. Then I thought of the idea of inception layers, which I recently studied in an article. So I implemented a architecture of 1 inception and two fully connected. I realized that inceptions produced a lot of weights and implemented max pooling after the inception layer. This model converged fast, but did not manage to quite capture the complexity of the data set and underfitted. So I aded a nother inception and a one-to-one together with dropout to prevent overfitting. This ended up to be my final architecture as it converged fast, trained fast and did not overfit. 

My final model results were:
* validation set accuracy of 93,4 %
* test set accuracy of 94%
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![Label: 0](https://github.com/emilwareus/Trafic_Sign_Classifier/blob/master/test_img/0.jpg) ![Label: 12](https://github.com/emilwareus/Trafic_Sign_Classifier/blob/master/test_img/12.jpg) ![Label: 17](https://github.com/emilwareus/Trafic_Sign_Classifier/blob/master/test_img/17.jpg)
![Label: 3](https://github.com/emilwareus/Trafic_Sign_Classifier/blob/master/test_img/3.jpg) ![Label: 37](https://github.com/emilwareus/Trafic_Sign_Classifier/blob/master/test_img/37.jpg) ![Label: 38](https://github.com/emilwareus/Trafic_Sign_Classifier/blob/master/test_img/38.jpg)

The first image is bright and the label has very few training exampels.
The second image should be easier to classify, as it has a very distinct look compared to the other classes. But it is a bit tilted.
The third image has a bit of a wierd lighting, but is alos very distinct in shape nad color. 
The forth sign will probably have some problems in classification, due to that there are many simular signs and it is not completly round as the photo is taken from the side. 
The fith sign has a little dot in it, which might cause problems. But this image is good and it should be classified correctly. 
The sixth image is very bright, but evenly so. If dropout did its job this image should be classified on shape and not color. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (20km/h)| Speed limit (30km/h)	| 
| Priority road     			| Priority road 										|
| No entry					| No entry											|
| Speed limit (60km/h)	      		| End of speed limit (80km/h)					 				|
| Go straight or left			| Go straight or left      							|
| Keep right			| Keep right      							|


The model was able to correctly guess 4 of the 6 traffic signs, which gives an accuracy of 66.6%. This seems to be a lot worse then the test set. The reasone for this is probably that some of the images are a bit zoomed out. This seems to make it hard for the classifier to predict numbers. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

![Top5](https://github.com/emilwareus/Trafic_Sign_Classifier/blob/master/img_for_redme/Capture2.PNG)

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?



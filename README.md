## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

[//]: # (Image References)

[class_distribution]: ./images/class_distribution.png "Class distribution"
[datasets_normal_distribution]: ./images/datasets_normal_distribution.png "Data sets normal distributions"
[grayscale_conversion]: ./images/grayscale_conversion.png "Grayscale conversion"
[training_validation_accuracy]: ./images/training_validation_accuracy.png "Training and validation accuracy wrt to Epochs"

[test_sign1]: ./test_signs/1.jpeg "Test sign 1"
[test_sign2]: ./test_signs/2.jpeg "Test sign 2"
[test_sign3]: ./test_signs/3.jpeg "Test sign 3"
[test_sign4]: ./test_signs/4.jpeg "Test sign 4"
[test_sign5]: ./test_signs/5.jpg "Test sign 5"

[speed_limit_20]: ./images/speed_limit_20.png "Speed limit 20 accuracy"
[no_passing]: ./images/no_passing.png "No passing accuracy"
[stop_accuracy]: ./images/stop_accuracy.png "Stop accuracy"
[traffic_signals_accuracy]: ./images/traffic_signals_accuracy.png "Traffic signals accuracy"
[no_entry_accuracy]: ./images/no_entry_accuracy.png "No entry accuracy"

[input_image]: ./images/input_image.png "Input image"
[conv1]: ./images/conv1.png "Convolution feature map 1"


### Data Set Summary & Exploration

#### 1. Summary of the data set

The numpy library was used to calculate the summary statistic of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Below is an exploratory visualization of the training, validation and training data sets by class. As shown below, there is a very uneven distribution of classes.

![alt text][class_distribution]

The minimum count for a training label is 180 and the maximum count for a training label is 2010. This large variation of class count have the potential to skew the training weights.

Additional exploration of data sets also found that the mean and standard deviations of each set are fairly similar as evident by the graph below.

* Training set mean: 15.738297077502228
* Training set standard deviation 12.002396466891161
* Validation set mean: 16.183673469387756
* Validation set standard deviation 12.089811535677788
* Test set mean: 15.551068883610451
* Test set standard deviation 11.946650214664762

![alt text][datasets_normal_distribution]

### Design and Test a Model Architecture

#### 1. Image Preprocessing

Prior to training the model with the images, the images will first need to be 
preprocessed. The steps outline to preprocess the image are as follows: 
* Add more augmented data
* Convert the image to grayscale
* Normalize the image pixel

##### Add augmented data
As described from the previous section, the distribution of classes in the training images is sporadic. As such, this can skew the model. To overcome this, augmented data is added to the training data set. In my implementation of the augmented data:
 1. The maximum count for a single label is calculated from the training data set.
 2. For each label, augmented data is generated up till the maximum count.
 3. To augment the data, one of the following image transformation functions are applied to a randomly selected image for the label:
    * Gaussian blur
    * Median blur
    * Dilate image
 4. The augmented data is then added back to the training list.

  **Note:** Other augmentation techniques have been considered but not yet implemented. These techniques include:
  * Rotation
  * Shrinking
  * Central scaling
  * Warping

##### Grayscale conversion
In my tests between rgb and grayscale image for training, the grayscale seems to perform better with better validation accuracy. 

Below is an example of an image before and after grayscale conversion.

![alt text][grayscale_conversion]

##### Normalize data
The next step is to normalize the image data to from 0 to 255 to have a mean of 0 with equal variance. This helps the training to process the data set more efficiently.


#### 2. Model Details

I experimented with a couple of models, namely the LeNet and AlexNet model. With a modified variant of the LeNet model, I managed to achieve a validation accuracy of 0.94.

With a modified variant of the AlexNet model, I managed to achieve a validation accuracy of 0.944.

My final model which is based on the AlexNet model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x48 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x48 				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 12x12x96	|
| RELU					|												|| Max pooling	      	| 2x2 stride,  outputs 6x6x96    				|| Convolution 3x3	    | 1x1 stride, same padding, outputs 6x6x128	    |
| RELU					|												|| Convolution 3x3	    | 1x1 stride, same padding, outputs 6x6x128	    |
| RELU					|												|| Convolution 3x3	    | 1x1 stride, same padding, outputs 6x6x96  	|
| RELU					|												|| Max pooling	      	| 2x2 stride,  outputs 3x3x96    				|
| Flatten               | outputs 400                                   |
| Fully connected		| outputs 120                                   |
| RELU                  |           									|
| Dropout               | Drop out, keep probability 0.5                |
| Fully connected		| outputs 84                                   |
| RELU                  |           									|
| Dropout               | Drop out, keep probability 0.5                |
| Fully connected		| outputs 43                                   |
| RELU                  |           									|
 


#### 3. Model Training

To train the model, the following modules/parameters were used:
* The adam optimizer was used to train the model
* A batch size of 128 was used. Experimentation with a lower batch size yield a poorer accuracy.
* An epochs of 50 was selected based on multiple tests whereby the the training accuracy reaches close to 100% and whereby validation accuracy stabilizes. Below is a graph of the training and validation accuracy.
* After several tests with a significantly lower and higher learning rate, A learning rate of 0.00097 was selected as it achieved the highest training/validation accuracy.

#### 4. Improving accuracy

My final model results were:
* training set accuracy of 0.994
* validation set accuracy of 0.966
* test set accuracy of 0.937

![alt text][training_validation_accuracy]

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

  The first architecture that was tested was the LeNet model as it was the first model that was taught from the course that is able to perform image recognition.

* What were some problems with the initial architecture?
  
  The initial architecture had a limited number of convolution layers and whilst changing the parameters for the LeNet model did manage to achieve a validation set accuracy of at least 0.93, I explored other options to improve the validation set accuracy which included transitioning to a AlexNet model.

  For the LeNet model, a 0.93 accuracy was achieved with the following parameters after several iterative tuning of the various parameters. The following parameters yield a validation accuracy of 0.947:
  
  * Loss probability retention rate of 0.6
  * Convolution 1 with 18 filters
  * Convolution 2 with 32 filters
  * Batch size: 128
  * Epoch: 50   

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

  Whilst the modified LeNet model yield a validation accuracy of at least 0.93 with a a high testing accuracy. This implied over fitting. To improve the accuracy of the model, a different model architecture  was explored and adopted. The model that was chosen was the AlexNet model which was meant to be an improvement to the LeNet model for image classification. The adoption of the model yield an increased in  validation accuracy to about 0.966 with a training accuracy of 0.994.

* Which parameters were tuned? How were they adjusted and why?

  In order to achieve the high validation the following parameters were tuned:
   * Epoch - Used to repeat the training of the model.
   * Learning rate - Used to adjust the rate of change of weights for each Epoch.
   * Dropout retention probability - The dropout regularization technique was applied in the connected layer to reduce over fitting. Several tests have yield 0.5 as the optimal probability.
   * Convolution kernel/filter size - Used to adjust the number of feature maps that are captured for each convolution layer.


* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Some important design choice included:
1. The model had to have multiple convolution layers to detect straight edges, curve, colors, patterns.
2. The model had to have minimal over fitting. This was done by applying a dropout layer in the connected layer.
3. The number of filters for each layers cannot be too large as it will mean a large number of parameters and a significantly long processing time.

A convolution layer works well with this problem as a convolution layer is designed to apply various image filtering to an image such as identity/edge detection/gaussian blurring which can be use as feature maps to match specific features (e.g. straight edge, curves, colors, patterns) to predict the label for an image fed into the model.


If a well known architecture was chosen:
* What architecture was chosen?

  The approach to implementing a model with high validation accuracy adopted both an iterative approach and using a well known architecture. As mentioned above, a LeNet and AlexNet architecture were explored and implemented.
* Why did you believe it would be relevant to the traffic sign application?

   Both the LeNet and AlexNet architecture were design for image classifications. The AlexNet architecture was chosen over the LeNet architecture as it had more convolution layers and as such do not require a larger data set as LeNet would have required to train the model to sufficient accuracy. 
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
   As described previously the final model's accuracy yield a validation set accuracy of 0.966.

### Test a Model on New Images

#### 1. External image classification test

In this section, I explored testing the traffic sign classifier on external 
images found on the web. Below are five German traffic signs were used:

![alt text][test_sign1]
![alt text][test_sign2]
![alt text][test_sign3] 
![alt text][test_sign4]
![alt text][test_sign5]

The first image might be difficult to classify because it is similar to other speed limit signs.

The second image might be difficult to classify because it has to be downsized to 32x32 which may result in the loss in resolution of the image and the potential for skewing of the image.

The third image might be difficult to classify because the image contains some watermarks and had to be downsized to 32x32.

The fourth image might be difficult to classify as the the sign contains multi colors for the traffic sign. As the model is train in grayscale, the model may not be able to correctly classify the sign in grayscale.

The firth image might be hard to classify as the sign features wear marks and may be incorrectly identify by the model as a specific pattern that is not related to the no entry label.

#### 2. Model's predictions on these new traffic signs

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (20km/h)  | Speed limit (20km/h)                     	    |
| No passing     	    | No passing									|
| Stop					| Stop											|
| Traffic signals  		| Go straight or right			 				|
| No entry			    | No entry      					     		|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set. Evaluating the accuracy for each image with the same labels used in the web set within the test set yields the following accuracy results:

    Test set accuracy for Speed limit (20km/h) = 1.000
    Test set accuracy for No passing = 0.990
    Test set accuracy for Stop = 0.904
    Test set accuracy for Traffic signals = 0.850
    Test set accuracy for No entry = 0.908

Based on the test set accuracies, the Traffic signal which has the lowest accuracy is reflected in the web image test accuracy. All the other test set labels with 0.9+ accuracy yields a correct prediction in the web image.

#### 3. Model prediction certainty

The code for making predictions on my final model is located in the Step 3: Test a Model on New image section.

For the first image, the model is very sure that this is a Speed limit (20km/h) sign (probability of 1), and the image does contain a Speed limit (20km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Speed limit (20km/h)     						|
| 0     				| Keep left  	                                |
| 0					    | Turn left ahead						        |
| 0	      		     	| Beware of ice/snow				 		    |
| 0				        | Speed limit (30km/h)     						|

![image meta][speed_limit_20]

For the second image the model is very certain that this is a no passing sign, and the image does contain a no passing sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| No passing   									| 
| 0     				| No passing for vehicles over 3.5 metric tons prohibited  	|
| 0					    | No vehicles							        |
| 0	      		     	| Speed limit (60km/h)					 		|
| 0				        | Dangerous curve to the right     			    |
![image meta][no_passing]

For the third image the model is very certain that this is a stop sign, and the image does contain a stop sign.. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Stop sign   									| 
| 0     				| Go straight or right  	                    |
| 0					    | Speed limit (70km/h)							|
| 0	      		     	| No entry					 	            	|
| 0				        | Speed limit (30km/h)     						|

![image meta][stop_accuracy]

For the fourth image the model is very certain that this is a Roundabout mandatory sign, but the image does contain a traffic signals sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Roundabout mandatory  					    |
| 0     				| Priority road 	                            || 0					    | Right-of-way at the next intersection			|
| 0	      		     	| Double curve	        				 		|
| 0				        | Speed limit (20km/h)     						|
![image meta][traffic_signals_accuracy]

For the fifth image the model is very certain that this is a no entry sign, and the image does contain a no entry sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| No entry   									| 
| 0     				| Dangerour curve to the right                	|
| 0					    | Keep right        							|
| 0	      		     	| Yield             					 		|
| 0				        | Speed limit (20km/h)     						|
![image meta][no_entry_accuracy]

### Visualization of neural network

![image meta][input_image]
Base on the above input, the first convolution layer feature maps are shown below:
![image meta][conv1]

Base on my observation, we can see that in the first convolution layer, the arrow characteristic would have been used to make the classification.
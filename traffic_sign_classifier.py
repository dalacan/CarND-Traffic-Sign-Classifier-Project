# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = "traffic-signs-data/train.p"
validation_file = "traffic-signs-data/valid.p"
testing_file = "traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

import numpy as np
### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = np.amax(y_train)+1 # classes label are 0 - max value

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

from sklearn.utils import shuffle


# Normalize the image
def normalize_image(images):
    '''
    Converts images into normalized form with zero mean
    :param images:
    :return:
    '''
    return (images - 128.0) / 128.0


def rgb2gray(images):
    '''
    Converts images into grayscale form
    :param images:
    :return:
    '''
    return np.dot(images[..., :3], [0.299, 0.587, 0.114])


use_gray = False

if use_gray == True:
    # Convert images to gray scale
    X_train_gray = rgb2gray(X_train)
    X_train_gray = X_train_gray.reshape(n_train, 32, 32, 1)
    X_valid_gray = rgb2gray(X_valid)
    X_valid_gray = X_valid_gray.reshape(n_validation, 32, 32, 1)
    X_test_gray = rgb2gray(X_test)
    X_test_gray = X_test_gray.reshape(n_test, 32, 32, 1)

    # Normalize images
    X_train = normalize_image(X_train_gray)
    X_valid = normalize_image(X_valid_gray)
    X_test = normalize_image(X_test_gray)
else:
    # Normalize images
    X_train = normalize_image(X_train)
    X_valid = normalize_image(X_valid)
    X_test = normalize_image(X_test)


# Load required modules
import tensorflow as tf

# Hyperparameter definitions
# Hyper-parameters
EPOCHS = 50
BATCH_SIZE = 128 # 128
mu = 0
sigma = 0.1
dropout = 0.6 # 0.4 is good


def model_lenet_test(x):
    ##########
    # Layer 1
    ##########
    # Convolution: Input = 32x32x3. Output = 28x28x6.
    f1 = 18  # 24
    wc1 = tf.Variable(tf.truncated_normal([5, 5, 3, f1], mu, sigma))  # height, weight, input depth, output depth
    b1 = tf.Variable(tf.zeros([f1]))
    layer1 = tf.nn.conv2d(x, wc1, strides=[1, 1, 1, 1], padding='VALID')
    layer1 = tf.nn.bias_add(layer1, b1)

    # Activation function.
    layer1 = tf.nn.relu(layer1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    layer1 = tf.nn.max_pool(layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    ##########
    # Layer 2
    ##########
    # Convolution: Input 14x14x6. Output = 10x10x16.
    f2 = 48  # 48
    wc2 = tf.Variable(tf.truncated_normal([5, 5, f1, f2], mu, sigma))  # height, weight, input depth, output depth
    b2 = tf.Variable(tf.zeros([f2]))
    layer2 = tf.nn.conv2d(layer1, wc2, strides=[1, 1, 1, 1], padding='VALID')
    layer2 = tf.nn.bias_add(layer2, b2)

    # Activation function.
    layer2 = tf.nn.relu(layer2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    layer2 = tf.nn.max_pool(layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    ####################
    # Flatten output
    ####################
    fh1 = 5 * 5 * f2
    fc1 = tf.reshape(layer2, [-1, fh1])  # Flatten from 5x5x16 to 400

    ##########
    # Layer 3
    ##########
    # Fully Connected. Input = 400. Output = 120.
    fcw1 = tf.Variable(tf.truncated_normal([fh1, 120], mu, sigma))  # height, weight, input depth, output depth
    fcb1 = tf.Variable(tf.zeros([120]))
    fc1 = tf.add(tf.matmul(fc1, fcw1), fcb1)

    # Activation function
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    ##########
    # Layer 4
    ##########
    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fcw2 = tf.Variable(tf.truncated_normal([120, 84], mu, sigma))  # height, weight, input depth, output depth
    fcb2 = tf.Variable(tf.zeros([84]))
    fc2 = tf.add(tf.matmul(fc1, fcw2), fcb2)

    # Activation function
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, dropout)

    ##########
    # Layer 5
    ##########
    # Layer 5: Fully Connected. Input = 84. Output = 10.
    fcw3 = tf.Variable(tf.truncated_normal([84, n_classes], mu, sigma))  # height, weight, input depth, output depth
    fcb3 = tf.Variable(tf.zeros([n_classes]))
    fc3 = tf.add(tf.matmul(fc2, fcw3), fcb3)

    logits = tf.nn.relu(fc3)
    return logits


def model_gray(x):
    ##########
    # Layer 1
    ##########
    # Convolution: Input = 32x32x1. Output = 28x28x6.
    f1 = 6
    wc1 = tf.Variable(tf.truncated_normal([5, 5, 1, f1], mu, sigma))  # height, weight, input depth, output depth
    b1 = tf.Variable(tf.zeros([f1]))
    layer1 = tf.nn.conv2d(x, wc1, strides=[1, 1, 1, 1], padding='VALID')
    layer1 = tf.nn.bias_add(layer1, b1)

    # Activation function.
    layer1 = tf.nn.relu(layer1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    layer1 = tf.nn.max_pool(layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    ##########
    # Layer 2
    ##########
    # Convolution: Input 14x14x6. Output = 10x10x16.
    f2 = 16
    wc2 = tf.Variable(tf.truncated_normal([5, 5, f1, f2], mu, sigma))  # height, weight, input depth, output depth
    b2 = tf.Variable(tf.zeros([f2]))
    layer2 = tf.nn.conv2d(layer1, wc2, strides=[1, 1, 1, 1], padding='VALID')
    layer2 = tf.nn.bias_add(layer2, b2)

    # Activation function.
    layer2 = tf.nn.relu(layer2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    layer2 = tf.nn.max_pool(layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    ####################
    # Flatten output
    ####################
    fc1 = tf.reshape(layer2, [-1, 5 * 5 * f2])  # Flatten from 5x5x16 to 400

    ##########
    # Layer 3
    ##########
    # Fully Connected. Input = 400. Output = 120.
    fh1 = 5 * 5 * f2
    fcw1 = tf.Variable(tf.truncated_normal([fh1, 120], mu, sigma))  # height, weight, input depth, output depth
    fcb1 = tf.Variable(tf.zeros([120]))
    fc1 = tf.add(tf.matmul(fc1, fcw1), fcb1)

    # Activation function
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    ##########
    # Layer 4
    ##########
    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fcw2 = tf.Variable(tf.truncated_normal([120, 84], mu, sigma))  # height, weight, input depth, output depth
    fcb2 = tf.Variable(tf.zeros([84]))
    fc2 = tf.add(tf.matmul(fc1, fcw2), fcb2)

    # Activation function
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, dropout)

    ##########
    # Layer 5
    ##########
    # Layer 5: Fully Connected. Input = 84. Output = 10.
    fcw3 = tf.Variable(tf.truncated_normal([84, n_classes], mu, sigma))  # height, weight, input depth, output depth
    fcb3 = tf.Variable(tf.zeros([n_classes]))
    fc3 = tf.add(tf.matmul(fc2, fcw3), fcb3)

    logits = tf.nn.relu(fc3)
    return logits


### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected,
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.

# Features and labels
if use_gray == True:
    x = tf.placeholder(tf.float32, (None, 32, 32, 1))  # Grayscale
else:
    x = tf.placeholder(tf.float32, (None, 32, 32, 3))  # RGB

y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

# Training pipeline
rate = 0.001

if use_gray == True:
    logits = model_gray(x)
else:
    logits = model_lenet_test(x)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)

# Model evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# Train the model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './lenet')
    print("Model saved")
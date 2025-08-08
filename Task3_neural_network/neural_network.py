import os # file handling
import cv2  # computer vision to load process images
import numpy as np  # for numpy arrays
import matplotlib.pyplot as plt  # optional library for displaying the identified digits (UI/UX)
import tensorflow as tf  # for defining and training machine learning models


# This project classifies handwritten digits from the MNIST Dataset from tensorflow (images of handwritten digits)
# In this project, a neural network is created by dividing the MNIST dataset into training and testing data
# The training data is used to train the model. The model is saved, loaded, and finally tested on unseen input data
# The drawing samples are images of digits from (1-9) that I created myself using paint, so that I could test the performance and accuracy of the 
# neural network model in classifying these handwritten digits correctly. Each handwritten digit in the MNIST Dataset has a respective label,
# which is the actual number or digit that is hand-drawn

# (This part of the code is only to run one time to download the model unless a change is made)

"""
mnist = tf.keras.datasets.mnist # to load the dataset
 
(x_train, y_train), (x_test, y_test) = mnist.load_data() # x_data is the image, y_data is the classification. load_data() will return two tuples with training and testing data

# normalizing the pixels of the training and testing data which makes it easier for the neural network to do calculations
x_train = tf.keras.utils.normalize(x_train, axis=1) 
x_test = tf.keras.utils.normalize(x_test, axis=1) 

# Creating the neural network
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) # turns it into a line of (28 * 28) pixels
model.add(tf.keras.layers.Dense(128, activation='relu')) # Adds a fully connected dense layer with 128 neurons
model.add(tf.keras.layers.Dense(128, activation='relu')) # relu is Rectified Linear Unit, which is an activation function that introduces linearity
model.add(tf.keras.layers.Dense(10, activation='softmax')) # This is the output layer. Softmax gives the probability for each digit to be the right answer

# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train the model
model.fit(x_train, y_train, epochs=6) # epochs is the number of times a model sees the data all over again

# save the model 
model.save('handwritten.model.h5') 

"""

# This is the part of the code that tests the model

# loading the model
model = tf.keras.models.load_model('handwritten.model.h5')

# code to test the model
image_num = 1
while(os.path.isfile(f"drawing_samples/{image_num}.jpg")):
    try:
        img = cv2.imread(f"drawing_samples/{image_num}.jpg")[:, :, 0]
        img = np.invert(np.array([cv2.resize(img, (28, 28))]))
        prediction = model.predict(img)
        print(f"The digit is most likely a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    
    except:
        print("Error. Please check the file name again.")
    finally:
        image_num += 1 # to transition and check all the digits one by one


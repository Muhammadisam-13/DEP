mnist = tf.keras.datasets.mnist # to load the dataset
 
# To train a model:
# Get the data 
# split the data into training and testing data
# training data is used to train the model
# testing data is used to assess the model to see how it performs on data it has not worked with

(x_train, y_train), (x_test, y_test) = mnist.load_data() # x_data is the image, y_data is the classification
# load data will return two tuples with training and testing data

# normalize or scale down so that every value is 0 or 1 (for the images)
# we normalize the pixels which makes it easier for the neural network to do calculations
x_train = tf.keras.utils.normalize(x_train, axis=1) # normalize the training data
x_test = tf.keras.utils.normalize(x_test, axis=1) # normalize the testing data

# neural network
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) # turns it into a line of (28 * 28) pixels
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax')) # softmax gives the probability for each digit to be the right answer

# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train the model
model.fit(x_train, y_train, epochs=3)

model.save('handwritten.model')
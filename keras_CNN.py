# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Initialising the CNN
model = Sequential()
# Step 1 - Convolution
model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# Step 2 - Pooling
model.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
# Step 3 - Flattening
model.add(Flatten())
# Step 4 - Full connection
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('data/training_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')
test_set = test_datagen.flow_from_directory('data/testing_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')

model.fit_generator(training_set, steps_per_epoch = 200, epochs = 2, validation_data = test_set, validation_steps = 80)


# Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image

image_path = 'data/single_prediction/cat_or_dog.jpg'
test_image = image.load_img(image_path, target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

result = model.predict(test_image)
print('image path - > ', image_path)
print('1st result - > ', result[0][0])

training_set.class_indices

prediction = ''
if result[0][0] == 1:
    prediction = 'dog'
elif result[0][0] == 0:
    prediction = 'cat'
else:
    prediction = 'other'
print(prediction)

 # visuallizing the neural net
# from keras.utils import plot_model
# plot_model(model, to_file='./model.png')
# from keras.datasets import cifar10
#
# (X_train, y_train), (X_test, y_test) = cifar10.load_data()
#
# print(X_train.head(5))


'''
Load Preprocess Analyse - Image Data
'''

# Building the image data generator
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./225)

train_generator = train_datagen.flow_from_directory('data/training_set', batch_size=32, class_mode='input')
test_generator = test_datagen.flow_from_directory('data/testing_set', batch_size=32, class_mode='input')

print('training img data are generated')

# importing  plotting lib
# import matplotlib.pyplot as plt
# x_batch, y_batch = next(train_generator)
# for i in range (0,32):
#     image = x_batch[i]
#     plt.imshow(image.transpose(2,1,0))
#     plt.show()

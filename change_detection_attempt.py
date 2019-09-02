# visualizing_class_activation_map

from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import pickle

from keras.layers import Lambda, concatenate
model = load_model('D:/4 NPS/_Thesis/Images and scripts/vgg19_Coastal_071319.h5')
IM_WIDTH, IM_HEIGHT = 256, 256 #fixed size for Vgg19

print("\n*********************************************************")
print("\n*********************************************************")

print("\nModel loaded!")
    
print("\n*********************************************************")
print("\n*********************************************************")


def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(256, 256))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor

def plot_preds(image, preds):
  """Displays image and the top-n predicted probabilities in a bar graph
  Args:
    image: PIL image
    preds: list of predicted labels and their probabilities
  """
  plt.imshow(image)
  plt.axis('off')

  plt.figure()
  labels = ("CoastalCliffs", "CoastalRocky", "CoastalWaterWay","Dunes","ManMadeStructures","SaltMarshes","SandyBeaches","TidalFlats")
  plt.barh([0, 1, 2, 3, 4, 5, 6, 7], preds, alpha=0.5)
  plt.yticks([0, 1, 2, 3, 4, 5, 6, 7], labels)
  plt.xlabel('Probability')
  plt.xlim(0,1.01)
  plt.tight_layout()
  plt.show()


def plot_img_pred(img_path, preds):
  """Displays image and the top-n predicted probabilities in a bar graph
  Args:
    image: PIL image
    preds: list of predicted labels and their probabilities
  """
  
  img = image.load_img(img_path, target_size=(256, 256))
  img_tensor = image.img_to_array(img)                    # (height, width, channels)
  img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
  img_tensor /= 255.                                      # imshow expects values in the range [0, 1]  
  plt.imshow(img_tensor[0])
  plt.axis('off')

  plt.figure()
  labels = ("CoastalCliffs", "CoastalRocky", "CoastalWaterWay","Dunes","ManMadeStructures","RipRap","SaltMarshes","SandyBeaches","TidalFlats")
  plt.barh([0, 1, 2, 3, 4, 5, 6, 7, 8], preds, alpha=0.5)
  plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8], labels)
  plt.xlabel('Probability')
  plt.xlim(0,1.01)
  plt.tight_layout()
  plt.show()

'''
# ******** IMAGE PATHS for tiled images**********************
# D:/4 NPS/_Thesis/Images and scripts/Tiling_Georectified/Dec06_rotate_V2.jpg
# heatmap shape (7621, 5784, 3)

# D:/4 NPS/_Thesis/Images and scripts/Tiling_Georectified/Jan10_rotate_V2.jpg
# heatmap shape (7621, 5784, 3)

# ******** IMAGE PATHS for test images**********************
D:/4 NPS/_Thesis/Images and scripts/test/CoastalWaterWay/795_resized.jpg
D:/4 NPS/_Thesis/Images and scripts/tile_0.jpg
D:/4 NPS/_Thesis/Images and scripts/tile_1.jpg
D:/4 NPS/_Thesis/Images and scripts/tile_3.jpg
D:/4 NPS/_Thesis/Images and scripts/tile_4.jpg

D:/4 NPS/_Thesis/Images and scripts/Image_slicing/3162_resized.jpg

D:/4 NPS/_Thesis/Images and scripts/Image_slicing/img0_0.jpg
D:/4 NPS/_Thesis/Images and scripts/Image_slicing/img0_100.jpg
D:/4 NPS/_Thesis/Images and scripts/Image_slicing/img0_200.jpg
D:/4 NPS/_Thesis/Images and scripts/Image_slicing/img100_0.jpg
D:/4 NPS/_Thesis/Images and scripts/Image_slicing/img100_100.jpg
D:/4 NPS/_Thesis/Images and scripts/Image_slicing/img100_200.jpg
D:/4 NPS/_Thesis/Images and scripts/Image_slicing/img200_0.jpg
D:/4 NPS/_Thesis/Images and scripts/Image_slicing/img200_100.jpg
D:/4 NPS/_Thesis/Images and scripts/Image_slicing/img200_200.jpg
'''

img_path = input("Please enter image path: ")
# `img` is a PIL image of size 224x224
img = image.load_img(img_path, target_size=(256, 256))

# `x` is a float32 Numpy array of shape (224, 224, 3)
x = image.img_to_array(img)

# We add a dimension to transform our array into a "batch"
# of size (1, 224, 224, 3)
x = np.expand_dims(x, axis=0)

# Finally we preprocess the batch
# (this does channel-wise color normalization)  figure out how to use input function for this!
x /= 255.

# Its shape is (1, 150, 150, 3)
print("\nShape")
print(x.shape)
print("\n")

preds = model.predict(x)
print(preds[0])
print("\n")
for i in range(len(preds[0])):
	print(preds[0][i])
print("\n")
for i in range(len(preds[0])):
	print('{:.2%}'.format(preds[0][i]))

test_image = plot_img_pred(img_path, preds[0])
'''
print("\n*********************************************************")
print("\n*********************************************************")
print("\nNow Save the Prediction List!")

text_file_name = input("Please enter file name for prediction list: ")
with open(text_file_name+'.txt', 'wb') as fp:
	pickle.dump(preds[0], fp)


#	for item in preds[0]:
#		f.write("%s\n" % item)

print("\n*********************************************************")
print("\n*********************************************************")


# this is your main classification output, what network predicted
np.argmax(preds[0])

model.output

# This is the "Dunes" entry in the prediction vector
#dunes_output = model.output[:, np.argmax(preds[0])]
# 0  = CoastalCliffs, 1 = CoastalRocky etc..
# ************ CHANGE THE NUMBER TO MATCH THE CLASS ************************************
class_output = model.output[: , 2]

# The is the output feature map of the `block5_conv4` layer,
# the last convolutional layer in VGG19 that we are using
last_conv_layer = model.get_layer('block5_conv4')

# This is the gradient of the "Dunes" class with regard to
# the output feature map of `block5_conv4`
grads = K.gradients(class_output, last_conv_layer.output)[0]

# This is a vector of shape (512,), where each entry
# is the mean intensity of the gradient over a specific feature map channel
pooled_grads = K.mean(grads, axis=(0, 1, 2))

# This function allows us to access the values of the quantities we just defined:
# `pooled_grads` and the output feature map of `block5_conv3`,
# given a sample image
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

# These are the values of these two quantities, as Numpy arrays,
# given our sample image of two elephants
pooled_grads_value, conv_layer_output_value = iterate([x])

# We multiply each channel in the feature map array
# by "how important this channel is" with regard to the elephant class
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

# The channel-wise mean of the resulting feature map
# is our heatmap of class activation
heatmap = np.mean(conv_layer_output_value, axis=-1)



heatmap = np.maximum(heatmap, 0)
print(heatmap)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()

save_heatmap = input("Please enter file name for image heatmap: ")
with open(save_heatmap+'.txt', 'wb') as fp:
	pickle.dump(heatmap, fp)

import cv2

# We use cv2 to load the original image
img = cv2.imread(img_path)

# We resize the heatmap to have the same size as the original image
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

# We convert the heatmap to RGB
heatmap = np.uint8(255 * heatmap)

# We apply the heatmap to the original image
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# 0.4 here is a heatmap intensity factor
superimposed_img = heatmap * 0.3 + img

# **************Save the image to disk ************************
#cv2.imwrite('/Users/Mara/Dropbox (Mara)/MachineLearning/TestImages/USGS_sample_062918/HeatMap/SampleN3d_beach_091218.jpg', superimposed_img)
#cv2.imwrite('/Users/maraorescanin/Dropbox (Mara)/MachineLearning/TestImages/USGS_sample_062918/HeatMap_Dave/06DEC17RS_A1_waterway.jpg', superimposed_img)

heatmap.shape
print("Heatmap shape")
print(heatmap.shape)
'''

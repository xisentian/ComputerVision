# confusion matrix program

from tensorflow.keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
# from imagenet_utils import preprocess_input, decode_predictions

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
  labels = ("CoastalCliffs", "CoastalRocky", "CoastalWaterWay","Dunes","ManMadeStructures","RipRap","SaltMarshes","SandyBeaches","TidalFlats")
  plt.barh([0, 1, 2, 3, 4, 5, 6, 7, 8], preds, alpha=0.5)
  plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8], labels)
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
  

if __name__ == "__main__":

    # load model
    import tensorflow as tf
    from keras.layers import Lambda, concatenate
    model = load_model('D:/4 NPS/_Thesis/Images and scripts/vgg19_Coastal_071319.h5')
    IM_WIDTH, IM_HEIGHT = 256, 256 #fixed size for Vgg19
    NB_EPOCHS = 3
    BAT_SIZE = 1
    batch_size = BAT_SIZE
    
    print("\n*********************************************************")
    print("\n*********************************************************")
    
    print("\nModel loaded!")
    
    print("\n*********************************************************")
    print("\n*********************************************************")

    
    test_dir = "D:/4 NPS/_Thesis/Images and scripts/test"
       
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IM_WIDTH, IM_HEIGHT),
    shuffle = False,
    batch_size=batch_size,
    class_mode='categorical')
    
    filenames = test_generator.filenames
    nb_samples = len(filenames)

    predict = model.predict_generator(test_generator,steps = nb_samples)

# should find 827 images belonging to 9 classes
print("\n*************************************************************")
print("\n*************************************************************\n")

print(len(predict))

print("\n*************************************************************")
print("\n*************************************************************")


print("Predicted array\n")
predict.shape
y_predicted = predict.argmax(axis=-1)
y_predicted.shape
y_predicted
print(y_predicted)
print("\n")

print("True array\n")
y_true = test_generator.classes
y_true.shape
y_true
print(y_true)
print("\n")

import sklearn
from sklearn.metrics import confusion_matrix
cmf=confusion_matrix(y_true, y_predicted, labels=[0, 1, 2, 3, 4, 5, 6, 7])


import itertools 
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, with normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label',fontsize=24)
    plt.xlabel('Predicted label',fontsize=24)


# Plot non-normalized confusion matrix
plt.rcParams.update({'font.size':16})
plt.figure(figsize=(10,10))
plot_confusion_matrix(cmf, normalize = True, classes=['Cliffs', 'Rocky', 'WaterWay', 'Dunes', 'Structures', 'Salt Marshes', 'Sandy Beach', 'TidalFlats' ],
                      title='Confusion matrix, with normalization')
plt.show()	# needed to get confusion matrix to display
                      

def plot_training(history):
  acc = history['acc']
  val_acc = history['val_acc']
  loss = history['loss']
  val_loss = history['val_loss']
  epochs = range(len(acc))

  plt.plot(epochs, acc, 'r', label = 'training')
  plt.plot(epochs, val_acc, 'b', label = 'validation')
  plt.title('Training and validation accuracy')
  plt.legend()
  plt.show()

  plt.figure()
  plt.plot(epochs, loss, 'r',  label = 'training')
  plt.plot(epochs, val_loss, 'b', label = 'validation')
  plt.title('Training and validation loss')
  plt.legend()
  plt.show()


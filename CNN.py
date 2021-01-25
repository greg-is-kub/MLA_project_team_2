import numpy as np
import os
import cv2
from PIL import Image
from tensorflow import keras
from sklearn.model_selection import train_test_split

##Parameters
WIDTH=256
HEIGHT=256
batch_size=128

##Data from database
#using image_dataset_from_directory

x_dataset = []
y_dataset = []
'''
for image in os.listdir('../input/covid19-radiography-database/COVID-19 Radiography Database/Viral Pneumonia/'):
    image = cv2.imread('../input/covid19-radiography-database/COVID-19 Radiography Database/Viral Pneumonia/'+image, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (WIDTH,HEIGHT))
    #image = image/255.
    x_dataset.append(image)
    y_dataset.append(2)
    
    #data augmentation : vertical flip
    #x_dataset.append(np.fliplr(image))
    #y_dataset.append(2)
    
    #Data Augmentation, smoothing images with kernel filter
  
    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(image,-1,kernel)
    x_dataset.append(dst)
    y_dataset.append(2)
   
'''
for image in os.listdir('database/covid/'):
    image = cv2.imread('database/covid/'+image, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (WIDTH,HEIGHT))
    #image = image/255.
    x_dataset.append(image)
    y_dataset.append(1)
    
    #data augmentation : vertical flip
    x_dataset.append(np.fliplr(image))
    y_dataset.append(1)
    
    #Data Augmentation, smoothing images with kernel filter
    kernel = np.ones((2,2),np.float32)/25
    dst = cv2.filter2D(image,-1,kernel)
    x_dataset.append(dst)
    y_dataset.append(1)
    '''
    plt.subplot(121),plt.imshow(image),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
    plt.xticks([]), plt.yticks([])
    plt.show()
    '''
    
    

for image in os.listdir('database/normal/'):
    image = cv2.imread('database/normal/'+image, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (WIDTH,HEIGHT))
    #image = image/255.
    x_dataset.append(image)
    y_dataset.append(0)
    
    #data augmentation : vertical flip
    '''
    x_dataset.append(np.fliplr(image))
    y_dataset.append(0)
    '''
    #Data Augmentation, smoothing images with kernel filter
    
    kernel = np.ones((2,2),np.float32)/25
    dst = cv2.filter2D(image,-1,kernel)
    x_dataset.append(dst)
    y_dataset.append(0)
    
    
x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=0.2, shuffle=True, random_state=3)
    
x_train = np.asarray(x_train)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = np.asarray(x_test)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
print("Nombre de données d'apprentissage",len(x_train))
print("Nombre de données positives (train/test confondus)",y_dataset.count(1)) #covid
print("Nombre de données negatives (train/test confondus)",y_dataset.count(0)) #pas covid

### CNN architecture
import tensorflow as tf
from keras import models
from keras import layers
r = tf.keras.metrics.Recall()
p = tf.keras.metrics.Precision()

model = models.Sequential()
model.add(layers.Input((WIDTH,HEIGHT,1)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(WIDTH, HEIGHT, 1)))
model.add(layers.BatchNormalization(momentum=0.9))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization(momentum=0.9))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization(momentum=0.9))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

##Model fit
history = model.fit(x_train, y_train, epochs=60, batch_size= 128, validation_data=(x_test, y_test), shuffle=True)

##Visualization of the results
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

##Affichage des accuracy et loss des train/test sets

avg_loss=sum(history.history['loss'][40:60])/20
print("Average train loss on 20th last epochs :",avg_loss)
avg_loss_val=sum(history.history['val_loss'][40:60])/20
print("Average test loss on 20th last epochs :",avg_loss_val)
avg_accuracy=sum(history.history['accuracy'][40:60])/20
print("Average train accuracy on 20th last epochs :",avg_accuracy)
avg_accuracy_val=sum(history.history['val_accuracy'][40:60])/20
print("Average test accuracy on 20th last epochs :",avg_accuracy_val)

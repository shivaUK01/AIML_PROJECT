#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np


# In[3]:


import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# In[8]:


imag=datasets.cifar10.load_data()
# print(type(imag))

(X_train, y_train),(X_test,y_test) = datasets.cifar10.load_data()
X_train.shape

# 5000 are training sample
# 32x32 is size
# 3 is specifiying rgb channel


# In[9]:


X_test.shape


# In[10]:


y_train.shape


# In[11]:


y_train[:5]
# we dont want 2d array, we directly want categories in a 1d array, so we convert it into 1d array using reshape.


# In[12]:


y_train = y_train.reshape(-1,)
# -1 is kept when you want it to be same as 10000,
# and you want to flatten this, so we leave it blankk.
y_train[:5]


# In[13]:


y_test = y_test.reshape(-1,)


# In[14]:


classes=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


# In[15]:


classes[8]


# In[16]:


# plt.figure(figsize=(15,2))
# plt.imshow(X_train[1])

# converted above two code into a function:(just)

def plot_sample(X,y,index):
    plt.figure(figsize=(15,2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])


# In[17]:


plot_sample(X_train, y_train, 0)


# In[18]:


plot_sample(X_train, y_train, 1)


# In[19]:


X_train = X_train / 255.0
X_test = X_test / 255.0


# In[20]:


# ann = models.Sequential([
#         layers.Flatten(input_shape=(32,32,3)),
#         layers.Dense(3000, activation='relu'),
#         layers.Dense(1000, activation='relu'),
#         layers.Dense(10, activation='softmax')    
#     ])

# ann.compile(optimizer='SGD',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# ann.fit(X_train, y_train, epochs=5)


# In[21]:


# from sklearn.metrics import confusion_matrix , classification_report
# import numpy as np
# y_pred = ann.predict(X_test)
# y_pred_classes = [np.argmax(element) for element in y_pred]

# print("Classification Report: \n", classification_report(y_test, y_pred_classes))


# In[22]:


cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# In[23]:


cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[24]:


cnn.fit(X_train, y_train, epochs=10)


# In[25]:


cnn.evaluate(X_test,y_test)


# In[26]:


y_pred = cnn.predict(X_test)
y_pred[:5]


# In[27]:


y_classes = [np.argmax(element) for element in y_pred]
y_classes[:5]


# In[28]:


y_test[:5]


# In[29]:


plot_sample(X_test, y_test,3)


# In[35]:


classes[y_classes[3]]


# In[41]:


# import cv2

# img_path = 'images/nine.jpeg'  # Adjust the path accordingly
# img_arr = cv2.imread(img_path)

# if img_arr is None:
#     print("Error: Unable to read the image.")
# else:
#     img_arr = cv2.resize(img_arr, (32, 32))
#     img_arr = img_arr.astype(np.float32) / 255.0  # Normalize the pixel values to the range [0, 1]
#     img_arr = np.expand_dims(img_arr, axis=0)
import cv2
import numpy as np

# Prompt the user to input the image path
img_path = input("Please enter the path to the image: ")

# Read the image from the provided path
img_arr = cv2.imread(img_path)

# Check if the image was successfully read
if img_arr is None:
    print("Error: Unable to read the image. Please check the path and try again.")
else:
    # Resize the image to 32x32 pixels
    img_arr = cv2.resize(img_arr, (32, 32))
    
    # Normalize the pixel values to the range [0, 1]
    img_arr = img_arr.astype(np.float32) / 255.0
    
    # Expand dimensions to match the input shape of most models (1, 32, 32, 3)
    img_arr = np.expand_dims(img_arr, axis=0)
    
    # Print the resulting array shape as a confirmation
    print("Image successfully processed.")
    print(f"Processed image shape: {img_arr.shape}")


# In[42]:


y_pred = cnn.predict(img_arr)
np.argmax(y_pred)
classes[np.argmax(y_pred)]


# In[33]:


cnn.save('model.keras')


# In[34]:


from keras.models import load_model
cnn = load_model('model.h5')  # Load the model


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





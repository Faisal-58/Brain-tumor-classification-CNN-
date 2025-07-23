import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np


# Load the model
model = tf.keras.models.load_model('brain_tumor_model.h5')

# Class names 
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']  

# Load and preprocess the image
img_path = 'test/meningioma/Te-me_0011.jpg'  
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = img_array / 255.0  
img_array = np.expand_dims(img_array, axis=0) 

# Predict
prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]

# Show result
print(f"Predicted class: {predicted_class}")

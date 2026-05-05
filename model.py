import tensorflow as tf
import numpy as np

# Load pretrained AI model
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# IMAGE PREPROCESSING
def preprocess_image(img):
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

# SIMPLE RESULT CONVERSION (temporary logic)
def decode_prediction(pred):
    class_id = np.argmax(pred)

    if class_id < 300:
        return "Normal"
    elif class_id < 600:
        return "Benign (Non-Cancer)"
    else:
        return "Malignant (Cancer)"
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

class Predictor:
    def __init__(self, model_path, img_size=(224, 224)):
        self.model = tf.keras.models.load_model(model_path)
        self.img_size = img_size

    def predict(self, img_path, class_names):
        # Cargar y preprocesar imagen
        img = image.load_img(img_path, target_size=self.img_size)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Hacer predicción
        pred = self.model.predict(img_array)
        class_index = np.argmax(pred)
        return class_names[class_index]

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model

class FruitClassifierModel:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        # Cargar MobileNetV2 sin la capa superior
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False  # Congelar pesos del modelo base

        # Agregar capas personalizadas
        x = Flatten()(base_model.output)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(self.num_classes, activation='softmax')(x)  # Conectar correctamente la capa de salida

        model = Model(inputs=base_model.input, outputs=x)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def get_model(self):
        return self.model


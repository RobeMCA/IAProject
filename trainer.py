import tensorflow as tf

class Trainer:
    def __init__(self, model, train_data, test_data, epochs=10):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.epochs = epochs

    def train(self):
        history = self.model.fit(
            self.train_data,
            validation_data=self.test_data,
            epochs=self.epochs
        )
        return history

    def evaluate(self):
        loss, accuracy = self.model.evaluate(self.test_data)
        print(f'Precisión del modelo: {accuracy * 100:.2f}%')

    def save_model(self, filename='modelo_clasificacion_frutas.h5'):
        self.model.save(filename)
        print(f'Modelo guardado como {filename}')

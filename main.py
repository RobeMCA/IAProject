from dataloader import DataLoader
from model import FruitClassifierModel
from trainer import Trainer
from predictor import Predictor

# Rutas del dataset
train_path = 'C:/Users/Roberto/Desktop/IAFinalProject/Dataset/Training'
test_path = 'C:/Users/Roberto/Desktop/IAFinalProject/Dataset/Test'

# Cargar datos
data_loader = DataLoader(train_path, test_path)
train_data, test_data = data_loader.load_data()

# Definir modelo
num_classes = len(train_data.class_indices)  # Número de clases detectadas
fruit_model = FruitClassifierModel(num_classes)
model = fruit_model.get_model()

# Entrenar modelo
#trainer = Trainer(model, train_data, test_data, epochs=10)
#trainer.train()
#trainer.evaluate()
#trainer.save_model()

# Hacer predicciones
predictor = Predictor('C:/Users/Roberto/Desktop/IAFinalProject/modelo_clasificacion_frutas-360.h5')
class_names = list(train_data.class_indices.keys())

img_path = 'C:/Users/Roberto/Desktop/IAFinalProject/fresa.jpg'  # Ruta de la imagen a predecir
prediction = predictor.predict(img_path, class_names)
print(f'La imagen es de la clase: {prediction}')
print('')


img_path = 'C:/Users/Roberto/Desktop/IAFinalProject/platano.jpg'  # Ruta de la imagen a predecir
prediction = predictor.predict(img_path, class_names)
print(f'La imagen es de la clase: {prediction}')
print('')

img_path = 'C:/Users/Roberto/Desktop/IAFinalProject/naranja.jpg'  # Ruta de la imagen a predecir
prediction = predictor.predict(img_path, class_names)
print(f'La imagen es de la clase: {prediction}')
print('')

img_path = 'C:/Users/Roberto/Desktop/IAFinalProject/manzana.jpg'  # Ruta de la imagen a predecir
prediction = predictor.predict(img_path, class_names)
print(f'La imagen es de la clase: {prediction}')
print('')

img_path = 'C:/Users/Roberto/Desktop/IAFinalProject/pina.jpeg'  # Ruta de la imagen a predecir
prediction = predictor.predict(img_path, class_names)
print(f'La imagen es de la clase: {prediction}')
print('')


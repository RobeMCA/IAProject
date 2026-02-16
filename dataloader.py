from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataLoader:
    def __init__(self, train_path, test_path, img_size=(224, 224), batch_size=32):
        self.train_path = train_path
        self.test_path = test_path
        self.img_size = img_size
        self.batch_size = batch_size
    
    def load_data(self):
        # Generador de imágenes con data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True
        )
        
        # Generador de imágenes para prueba (sin augmentation)
        test_datagen = ImageDataGenerator(rescale=1./255)
        

        train_generator = train_datagen.flow_from_directory(
            self.train_path, target_size=self.img_size, batch_size=self.batch_size, class_mode='categorical'
        )

        test_generator = test_datagen.flow_from_directory(
            self.test_path, target_size=self.img_size, batch_size=self.batch_size, class_mode='categorical'
        )

        return train_generator, test_generator

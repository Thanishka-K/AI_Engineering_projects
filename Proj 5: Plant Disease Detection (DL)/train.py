import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape=(128, 128, 3), num_classes=3):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),# First Convolutional Layer
        layers.MaxPooling2D((2, 2)),
         
        layers.Conv2D(64, (3, 3), activation='relu'),# Second Convolutional Layer
        layers.MaxPooling2D((2, 2)),
      
        layers.Flatten(),# Flattening and Dense Layers
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

print("CNN Architecture initialized.")

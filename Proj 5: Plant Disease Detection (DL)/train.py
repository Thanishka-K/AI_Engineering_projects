import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def train_and_save_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax') #for classes
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    X_train = np.random.rand(10, 128, 128, 3) #for demonstration
    y_train = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

    print("Starting training...")
    model.fit(X_train, y_train, epochs=1) 

    model.save('plant_model.h5')
    print("Success: 'plant_model.h5' has been created.")

if __name__ == "__main__":
    train_and_save_model()

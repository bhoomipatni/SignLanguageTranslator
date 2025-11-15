import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from models.lstm_model import create_model
from data.data_loader import DataLoader

def train_model(config):
    # Load data
    data_loader = DataLoader(config['data_path'], config['labels_path'])
    X_train, y_train = data_loader.load_data()

    # Create model
    model = create_model(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), num_classes=len(data_loader.labels))

    # Compile model
    model.compile(optimizer=Adam(learning_rate=config['learning_rate']), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(X_train, y_train, epochs=config['epochs'], batch_size=config['batch_size'], validation_split=config['validation_split'])

    # Save model
    model.save(os.path.join(config['model_save_path'], 'sign_language_model.h5'))

if __name__ == "__main__":
    # Load configuration
    with open('config.py') as config_file:
        config = json.load(config_file)

    train_model(config)
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from src.data.data_loader import DataLoader

def evaluate_model(model, validation_data, validation_labels):
    predictions = model.predict(validation_data)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(validation_labels, axis=1)

    print("Classification Report:")
    print(classification_report(true_classes, predicted_classes))

    print("Confusion Matrix:")
    print(confusion_matrix(true_classes, predicted_classes))

def main():
    # Load validation data
    data_loader = DataLoader()
    validation_data, validation_labels = data_loader.load_validation_data()

    # Load the trained model
    model = tf.keras.models.load_model('models/saved_models/lstm_model.h5')

    # Evaluate the model
    evaluate_model(model, validation_data, validation_labels)

if __name__ == "__main__":
    main()
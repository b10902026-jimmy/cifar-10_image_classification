from data_preprocess import load_and_preprocess_data
from model import create_model, save_model, convert_model_to_trt, load_trt_model
from augmentation import create_augmentation
from visualization import plot_sample_images, plot_misclassified_images, plot_training_history
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import accuracy_score
import numpy as np
from keras.callbacks import TensorBoard
import time

def main():
    # Load and preprocess data
    x_train, y_train, x_test, y_test = load_and_preprocess_data()

    # Visualize training data 
    plot_sample_images(x_train, y_train)

    # Create and compile the model
    model = create_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Data augmentation
    datagen = create_augmentation()
    datagen.fit(x_train)

    # Model checkpoints and early stopping
    checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)
    early_stop = EarlyStopping(patience=10)

    # Create a TensorBoard callback instance.
    # This will save logs to a subdirectory in the 'logs' directory.
    # The subdirectory name is based on the current time to ensure uniqueness. 
    tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))

    # Train the model
    history = model.fit(datagen.flow(x_train, y_train, batch_size=32), 
                    epochs=50, 
                    validation_data=(x_test, y_test), 
                    callbacks=[checkpoint, early_stop, tensorboard])
    
    predictions = model.predict(x_test, batch_size=64)

    # Convert prediction results into category labels
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)

    # Calculate the accuracy
    accuracy = accuracy_score(true_classes, predicted_classes)
    print(f"Accuracy with TensorRT optimized model: {accuracy * 100:.2f}%")

    # Visualize the inference results
    plot_misclassified_images(x_test, true_classes, predicted_classes)

    # Check if history object is created
    if history is not None:
        plot_training_history(history)
    else:
        print("Training did not complete successfully. No history object was created.")


if __name__ == "__main__":
    main()

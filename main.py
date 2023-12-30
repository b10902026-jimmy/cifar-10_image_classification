from data_preprocess import load_and_preprocess_data
from model import create_model
from augmentation import create_augmentation
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

def main():
    # Load and preprocess data
    x_train, y_train, x_test, y_test = load_and_preprocess_data()

    # Create and compile the model
    model = create_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Optional: Data augmentation
    datagen = create_augmentation()
    datagen.fit(x_train)

    # Model checkpoints and early stopping
    checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)
    early_stop = EarlyStopping(patience=10)

    # Train the model
    history = model.fit(datagen.flow(x_train, y_train, batch_size=32), 
                        epochs=50, 
                        validation_data=(x_test, y_test), 
                        callbacks=[checkpoint, early_stop])

    # Check if history object is created
    if history is not None:
        # Plot training & validation accuracy values
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
    else:
        print("Training did not complete successfully. No history object was created.")

if __name__ == "__main__":
    main()

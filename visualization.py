import matplotlib.pyplot as plt
import numpy as np 

def plot_sample_images(x, y, class_names, num_images=5):
    # Adjust the figure size to accommodate the number of images
    plt.figure(figsize=(num_images * 2, 2)) # Width set to 2 inches per image

    for i in range(num_images):
        plt.subplot(1, num_images, i+1) # Create subplots for each sample image
        plt.xticks([]) # Remove x-axis tick marks
        plt.yticks([]) # Remove y-axis tick marks
        plt.grid(False) # Disable the grid for a cleaner image
        plt.imshow(x[i]) # Display the image
        # Label the image with the corresponding class name
        plt.xlabel(class_names[y[i]])
    
    plt.subplots_adjust(wspace=0.5) # Adjust the spacing between subplots
    plt.show() # Display the figure with the images


def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()

def plot_misclassified_images(x, y_true, y_pred):
    misclassified_indices = np.where(y_pred != y_true)[0]
    num_images = min(5, len(misclassified_indices))
    plt.figure(figsize=(10, 10))
    for i, misclassified_index in enumerate(misclassified_indices[:num_images]):
        plt.subplot(1, num_images, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x[misclassified_index], cmap=plt.cm.binary)
        plt.xlabel(f"True: {y_true[misclassified_index]}, Predicted: {y_pred[misclassified_index]}")
    plt.show()
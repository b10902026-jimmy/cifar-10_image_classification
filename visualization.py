import matplotlib.pyplot as plt
import numpy as np 

def plot_sample_images(x, y, class_names, num_images=20):
    rows = 4  # 四行
    cols = 5  # 每行五列
    plt.figure(figsize=(cols * 2, rows * 2))  # 調整整個圖片的尺寸

    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)  # 正確指定行和列
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x[i])

        # Convert one-hot encoded labels to class index
        label_index = np.argmax(y[i])
        plt.xlabel(class_names[label_index])

    plt.subplots_adjust(wspace=0.5, hspace=0.5)  # 調整子圖之間的水平和垂直間距
    plt.show()



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
    num_images = min(20, len(misclassified_indices))  # 顯示最多20張圖片
    rows = 4  # 四行
    cols = 5  # 每行五列

    plt.figure(figsize=(cols * 2.5, rows * 2.5))  # 調整圖片大小
    for i, misclassified_index in enumerate(misclassified_indices[:num_images]):
        plt.subplot(rows, cols, i + 1)  # 正確指定行和列
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x[misclassified_index], cmap=plt.cm.binary)
        plt.xlabel(f"True: {y_true[misclassified_index]}, Predicted: {y_pred[misclassified_index]}")
    plt.show()

def plot_random_predictions(x, true_classes, predicted_classes, class_names, num_images=20):
    indices = np.random.choice(range(len(x)), num_images, replace=False)
    
    rows = 4  
    cols = 5  
    plt.figure(figsize=(cols * 2.2, rows * 2.5))  # 增加圖像的尺寸以提供更多空間
    
    for i, index in enumerate(indices):
        plt.subplot(rows, cols, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x[index], cmap=plt.cm.binary)
        
        true_label = class_names[true_classes[index]]
        pred_label = class_names[predicted_classes[index]]
        label = f"Facts: {true_label}\nPrediction: {pred_label}"
        plt.xlabel(label, fontsize=9)  # 減小字體大小

    plt.subplots_adjust(wspace=0.1, hspace=0.3) 
    plt.show()


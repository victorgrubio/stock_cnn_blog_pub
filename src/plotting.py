import matplotlib.pyplot as plt
import numpy as np


def plot_training_data(x_train, y_train, show=False):
    fig = plt.figure(figsize=(15, 15))
    columns = rows = 3
    for i in range(1, columns * rows + 1):
        index = np.random.randint(len(x_train))
        img = x_train[index]
        fig.add_subplot(rows, columns, i)
        plt.axis("off")
        plt.title('image_' + str(index) + '_class_' + str(np.argmax(y_train[index])), fontsize=10)
        plt.subplots_adjust(wspace=0.2, hspace=0.2)
        plt.imshow(img)
    if show:
        plt.show()


def plot_model_history_results(history):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['f1_metric'])
    plt.plot(history.history['val_f1_metric'])

    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train_loss', 'val_loss', 'f1', 'val_f1'], loc='upper left')
    plt.show()

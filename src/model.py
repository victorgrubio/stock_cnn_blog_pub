import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score
from tensorflow.python.keras import Sequential, regularizers
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from loguru import logger
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop

from src.metrics import f1_metric


def create_model_cnn(params):
    model = Sequential()

    logger.info("Training with params {}".format(params))

    conv2d_layer1 = Conv2D(params["conv2d_layers"]["conv2d_filters_1"],
                           params["conv2d_layers"]["conv2d_kernel_size_1"],
                           strides=params["conv2d_layers"]["conv2d_strides_1"],
                           kernel_regularizer=regularizers.l2(params["conv2d_layers"]["kernel_regularizer_1"]),
                           padding='same', activation="relu", use_bias=True,
                           kernel_initializer='glorot_uniform',
                           input_shape=(params["input_shape"]))
    model.add(conv2d_layer1)
    if params["conv2d_layers"]['conv2d_mp_1'] > 1:
        model.add(MaxPool2D(pool_size=params["conv2d_layers"]['conv2d_mp_1']))

    model.add(Dropout(params['conv2d_layers']['conv2d_do_1']))
    if params["conv2d_layers"]['layers'] == 'two':
        conv2d_layer2 = Conv2D(params["conv2d_layers"]["conv2d_filters_2"],
                               params["conv2d_layers"]["conv2d_kernel_size_2"],
                               strides=params["conv2d_layers"]["conv2d_strides_2"],
                               kernel_regularizer=regularizers.l2(params["conv2d_layers"]["kernel_regularizer_2"]),
                               padding='same', activation="relu", use_bias=True,
                               kernel_initializer='glorot_uniform')
        model.add(conv2d_layer2)

        if params["conv2d_layers"]['conv2d_mp_2'] > 1:
            model.add(MaxPool2D(pool_size=params["conv2d_layers"]['conv2d_mp_2']))

        model.add(Dropout(params['conv2d_layers']['conv2d_do_2']))

    model.add(Flatten())

    model.add(Dense(params['dense_layers']["dense_nodes_1"], activation='relu'))
    model.add(Dropout(params['dense_layers']['dense_do_1']))

    if params['dense_layers']["layers"] == 'two':
        model.add(Dense(params['dense_layers']["dense_nodes_2"], activation='relu',
                        kernel_regularizer=params['dense_layers']["kernel_regularizer_1"]))
        model.add(Dropout(params['dense_layers']['dense_do_2']))

    model.add(Dense(3, activation='softmax'))

    optimizer = SGD(learning_rate=params["lr"], decay=1e-6, momentum=0.9, nesterov=True)

    if params["optimizer"] == 'rmsprop':
        optimizer = RMSprop(learning_rate=params["lr"])
    elif params["optimizer"] == 'adam':
        optimizer = Adam(learning_rate=params["lr"], beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', f1_metric])
    return model

def show_model_status(model, x_test, y_test, best_model_path):
    test_res = model.evaluate(x_test, y_test, verbose=0)
    logger.info("keras evaluate=", test_res)
    pred = model.predict(x_test)
    pred_classes = np.argmax(pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    check_baseline(pred_classes, y_test_classes)
    conf_mat = confusion_matrix(y_test_classes, pred_classes)
    logger.info(conf_mat)
    labels = [0, 1, 2]

    f1_weighted = f1_score(y_test_classes, pred_classes, labels=None,
                           average='weighted', sample_weight=None)
    logger.info("F1 score (weighted)", f1_weighted)
    logger.info("F1 score (macro)", f1_score(y_test_classes, pred_classes, labels=None,
                                             average='macro', sample_weight=None))
    logger.info("F1 score (micro)", f1_score(y_test_classes, pred_classes, labels=None,
                                             average='micro',
                                             sample_weight=None))  # weighted and micro preferred in case of imbalance

    # https://scikit-learn.org/stable/modules/model_evaluation.html#cohen-s-kappa --> supports multiclass; ref: https://stats.stackexchange.com/questions/82162/cohens-kappa-in-plain-english
    logger.info("cohen's Kappa", cohen_kappa_score(y_test_classes, pred_classes))

    recall = []
    for i, row in enumerate(conf_mat):
        recall.append(np.round(row[i] / np.sum(row), 2))
        logger.info("Recall of class {} = {}".format(i, recall[i]))
    logger.info("Recall avg", sum(recall) / len(recall))


def check_baseline(pred, y_test):
    logger.info("size of test set", len(y_test))
    e = np.equal(pred, y_test)
    logger.info("TP class counts", np.unique(y_test[e], return_counts=True))
    logger.info("True class counts", np.unique(y_test, return_counts=True))
    logger.info("Pred class counts", np.unique(pred, return_counts=True))
    holds = np.unique(y_test, return_counts=True)[1][2]  # number 'hold' predictions
    logger.info("baseline acc:", (holds/len(y_test)*100))


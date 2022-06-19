import os
import sys
import time

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.python.keras.models import load_model

from src.data_generator import DataGenerator
from src.keras_training import check_baseline, plot_history
from src.logger import Logger
from src.utils import seconds_to_minutes


args = sys.argv  # a list of the arguments provided (str)
print("running stock_cnn.py", args)
pd.options.display.width = 0
company_code = args[1]
strategy_type = args[2]
ROOT_PATH = ".."
iter_changes = "fresh_rolling_train"  # label for changes in this run iteration
INPUT_PATH = os.path.join(ROOT_PATH, "stock_history", company_code)
OUTPUT_PATH = os.path.join(ROOT_PATH, "outputs", iter_changes)
LOG_PATH = OUTPUT_PATH + os.sep + "logs"
LOG_FILE_NAME_PREFIX = "log_{}_{}_{}".format(company_code, strategy_type, iter_changes)
PATH_TO_STOCK_HISTORY_DATA = os.path.join(ROOT_PATH, "stock_history")
logger = Logger(LOG_PATH, LOG_FILE_NAME_PREFIX)

print("Tensorflow devices {}".format(tf.test.gpu_device_name()))

start_time = time.time()
# exit here, since training is done with stock_keras.ipynb.
# comment sys.exit() if you want to try out rolling window training.
# sys.exit()

params = {'batch_size': 60, 'conv2d_layers': {'conv2d_do_1': 0.0, 'conv2d_filters_1': 30,
                                               'conv2d_kernel_size_1': 2, 'conv2d_mp_1': 2, 'conv2d_strides_1': 1,
                                               'kernel_regularizer_1':0.0, 'conv2d_do_2': 0.01, 'conv2d_filters_2': 10,
                                               'conv2d_kernel_size_2': 2, 'conv2d_mp_2': 2, 'conv2d_strides_2': 2,
                                               'kernel_regularizer_2':0.0, 'layers': 'two'},
           'dense_layers': {'dense_do_1': 0.07, 'dense_nodes_1': 100, 'kernel_regularizer_1':0.0, 'layers': 'one'},
           'epochs': 3000, 'lr': 0.001, 'optimizer': 'adam', 'input_dim_1': 15, 'input_dim_2': 15, 'input_dim_3': 3}

best_model_path = os.path.join(OUTPUT_PATH, 'best_model_keras')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                   patience=100, min_delta=0.0001)
# csv_logger = CSVLogger(os.path.join(OUTPUT_PATH, 'log_training_batch.log'), append=True)
rlp = ReduceLROnPlateau(monitor='val_loss', factor=0.02, patience=10, verbose=1, mode='min',
                        min_delta=0.001, cooldown=1, min_lr=0.0001)
mcp = ModelCheckpoint(best_model_path, monitor='val_loss', verbose=0,
                      save_best_only=True, save_weights_only=False, mode='min', period=1)


if __name__ == "__main__":
    count = 0
    start_time = time.time()
    PATH_TO_COMPANY_DATA = "TODO"
    data_gen = DataGenerator(company_code, PATH_TO_COMPANY_DATA, OUTPUT_PATH, strategy_type, False, logger)
    while True:
        logger.append_log("training for batch number {}".format(count))
        x_train, y_train, x_cv, y_cv, x_test, y_test,\
        df_batch_train, df_batch_test, sample_weights, is_last_batch = \
            data_gen.get_rolling_data_next(None, 12)

        if os.path.exists(best_model_path) and count > 0:
            model = load_model(best_model_path)
            logger.append_log("loaded previous best model")

        history = model.fit(
            x_train, y_train, epochs=params['epochs'], verbose=0,
            batch_size=64, shuffle=True,
            validation_data=(x_cv, y_cv),
            callbacks=[es, mcp, rlp], sample_weight=sample_weights
        )

        count = count + 1
        plot_history(history, count)
        min_arg = np.argmin(np.array(history.history['val_loss']))
        logger.append_log("Best val_loss is {} and corresponding train_loss is {}".format(
            history.history['val_loss'][min_arg], history.history['loss'][min_arg]))
        test_res = model.evaluate(x_test, y_test, verbose=0)
        logger.append_log("keras evaluate result =" + str(test_res)+", metrics:"+str(model.metrics_names))
        pred = model.predict(x_test)
        check_baseline(np.argmax(pred, axis=1), np.argmax(y_test, axis=1))
        conf_mat = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(pred, axis=1))
        logger.append_log('\n'+str(conf_mat))
        labels = [0, 1, 2]
        f1_weighted = f1_score(
            np.argmax(y_test, axis=1), np.argmax(pred, axis=1), labels=None,
                               average='weighted', sample_weight=None)
        logger.append_log("F1 score (weighted) " + str(f1_weighted))
        logger.append_log(
            "F1 score (macro) " + str(f1_score(np.argmax(y_test, axis=1), np.argmax(pred, axis=1), labels=None,
                                               average='macro', sample_weight=None)))
        logger.append_log(
            "F1 score (micro) " + str(f1_score(np.argmax(y_test, axis=1), np.argmax(pred, axis=1), labels=None,
                                               average='micro',
                                               sample_weight=None)))  # weighted and micro preferred in case of imbalance

        for i, row in enumerate(conf_mat):
            logger.append_log("precision of class {} = {}".format(i, np.round(row[i] / np.sum(row), 2)))

        if is_last_batch:
            break

    logger.append_log("Complete training finished in {}".format(seconds_to_minutes(time.time() - start_time)))
    logger.flush()

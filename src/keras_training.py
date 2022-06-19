import os
from operator import itemgetter

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import f1_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.utils import compute_class_weight

from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils.vis_utils import plot_model

from loguru import logger

from src.config import OUTPUT_PATH, DATASET_PATH
from src.model import create_model_cnn
from src.plotting import plot_training_data, plot_model_history_results


def get_sample_weights(y):
    """
    calculate the sample weights based on class weights. Used for models with
    imbalanced data and one hot encoding prediction.

    params:
        y: class labels as integers
    """

    y = y.astype(int)  # compute_class_weight needs int labels
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y)

    logger.info("real class weights are {}".format(class_weights), np.unique(y))
    logger.info("value_counts", np.unique(y, return_counts=True))
    sample_weights = y.copy().astype(float)
    for i in np.unique(y):
        sample_weights[sample_weights == i] = class_weights[i]  # if i == 2 else 0.8 * class_weights[i]
        # sample_weights = np.where(sample_weights == i, class_weights[int(i)], y_)

    return sample_weights


def reshape_as_image(x, img_width, img_height):
    x_temp = np.zeros((len(x), img_height, img_width))
    for i in range(x.shape[0]):
        # logger.info(type(x), type(x_temp), x.shape)
        x_temp[i] = np.reshape(x[i], (img_height, img_width))
    return x_temp


def check_baseline(pred, y_test):
    logger.info("size of test set", len(y_test))
    e = np.equal(pred, y_test)
    logger.info("TP class counts", np.unique(y_test[e], return_counts=True))
    logger.info("True class counts", np.unique(y_test, return_counts=True))
    logger.info("Pred class counts", np.unique(pred, return_counts=True))
    holds = np.unique(y_test, return_counts=True)[1][2]  # number 'hold' predictions
    logger.info("baseline acc:", (holds/len(y_test)*100))


def check_baseline(pred, y_test):
    e = np.equal(pred, y_test)
    logger.info("TP class counts", np.unique(y_test[e], return_counts=True))
    logger.info("True class counts", np.unique(y_test, return_counts=True))
    logger.info("Pred class counts", np.unique(pred, return_counts=True))
    holds = np.unique(y_test, return_counts=True)[1][2]  # number 'hold' predictions
    logger.info("baseline acc:", str((holds / len(y_test) * 100)))


def show_model_status():
    model = load_model(best_model_path)
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


def get_training_data():
    x_train, x_test, y_train, y_test = train_test_split(
        df.loc[:, 'open':'eom_26'].values, df['labels'].values,
        train_size=0.8,
        test_size=0.2, random_state=2, shuffle=True,
        stratify=df['labels'].values
    )

    # smote = RandomOverSampler(random_state=42, sampling_strategy='not majority')
    # x_train, y_train = smote.fit_resample(x_train, y_train)
    # logger.info('Resampled dataset shape %s' % Counter(y_train))

    if 0.7 * x_train.shape[0] < 2500:
        train_split = 0.8
    else:
        train_split = 0.7
    # train_split = 0.7
    logger.info('train_split =', train_split)
    x_train, x_cv, y_train, y_cv = train_test_split(
        x_train, y_train, train_size=train_split, test_size=1 - train_split,
        random_state=2, shuffle=True, stratify=y_train
    )
    mm_scaler = MinMaxScaler(feature_range=(0, 1))  # or StandardScaler?
    x_train = mm_scaler.fit_transform(x_train)
    x_cv = mm_scaler.transform(x_cv)
    x_test = mm_scaler.transform(x_test)

    x_main = x_train.copy()
    logger.info(
        "Shape of x, y train/cv/test {} {} {} {} {} {}".format(
            x_train.shape, y_train.shape, x_cv.shape, y_cv.shape,
            x_test.shape, y_test.shape)
    )

    num_features = 225  # should be a perfect square
    selection_method = 'all'
    topk = 320 if selection_method == 'all' else num_features
    # if train_split >= 0.8:
    #     topk = 400
    # else:
    #     topk = 300

    if selection_method == 'anova' or selection_method == 'all':
        select_k_best = SelectKBest(f_classif, k=topk)
        if selection_method != 'all':
            x_train = select_k_best.fit_transform(x_main, y_train)
            x_cv = select_k_best.transform(x_cv)
            x_test = select_k_best.transform(x_test)
        else:
            select_k_best.fit(x_main, y_train)

        selected_features_anova = itemgetter(
            *select_k_best.get_support(indices=True)
        )(list_features)
        logger.info(selected_features_anova)
        logger.info(select_k_best.get_support(indices=True))
        logger.info("****************************************")

    if selection_method == 'mutual_info' or selection_method == 'all':
        select_k_best = SelectKBest(mutual_info_classif, k=topk)
        if selection_method != 'all':
            x_train = select_k_best.fit_transform(x_main, y_train)
            x_cv = select_k_best.transform(x_cv)
            x_test = select_k_best.transform(x_test)
        else:
            select_k_best.fit(x_main, y_train)

        selected_features_mic = itemgetter(*select_k_best.get_support(indices=True))(list_features)
        logger.info(
            f"{len(selected_features_mic)}, {selected_features_mic}")
        logger.info(f"{select_k_best.get_support(indices=True)}")
    # %%
    if selection_method == 'all':
        common = list(set(selected_features_anova).intersection(selected_features_mic))
        logger.info("common selected features", len(common), common)
        if len(common) < num_features:
            raise Exception(
                'number of common features found {} < {} required features. Increase "topk variable"'.format(
                    len(common), num_features))
        feat_idx = []
        for c in common:
            feat_idx.append(list_features.index(c))
        feat_idx = sorted(feat_idx[0:225])
        logger.info(feat_idx)
    # %%
    if selection_method == 'all':
        x_train = x_train[:, feat_idx]
        x_cv = x_cv[:, feat_idx]
        x_test = x_test[:, feat_idx]

    logger.info("Shape of x, y train/cv/test {} {} {} {} {} {}".format(
        x_train.shape, y_train.shape, x_cv.shape, y_cv.shape,
        x_test.shape, y_test.shape)
    )
    # %%
    _labels, _counts = np.unique(y_train, return_counts=True)
    logger.info("percentage of class 0 = {}, class 1 = {}".format(_counts[0] / len(y_train) * 100,
                                                                  _counts[1] / len(y_train) * 100))
    sample_weights = get_sample_weights(y_train)
    logger.info("Test sample_weights")
    rand_idx = np.random.randint(0, 1000, 30)
    logger.info(y_train[rand_idx])
    logger.info(sample_weights[rand_idx])
    # %%
    one_hot_enc = OneHotEncoder(sparse=False, categories='auto')  # , categories='auto'
    y_train = one_hot_enc.fit_transform(y_train.reshape(-1, 1))
    logger.info("y_train", y_train.shape)
    y_cv = one_hot_enc.transform(y_cv.reshape(-1, 1))
    y_test = one_hot_enc.transform(y_test.reshape(-1, 1))
    # %%
    dim = int(np.sqrt(num_features))
    x_train = reshape_as_image(x_train, dim, dim)
    x_cv = reshape_as_image(x_cv, dim, dim)
    x_test = reshape_as_image(x_test, dim, dim)
    # adding a 1-dim for channels (3)
    x_train = np.stack((x_train,) * 3, axis=-1)
    x_test = np.stack((x_test,) * 3, axis=-1)
    x_cv = np.stack((x_cv,) * 3, axis=-1)
    logger.info(
        "final shape of x, y train/test {} {} {} {}".format(
            x_train.shape, y_train.shape, x_test.shape, y_test.shape
        ))

    return x_train, y_train, x_test, y_test, x_cv, y_cv


if __name__ == "__main__":
    np.random.seed(2)
    company_code = 'WMT'
    strategy_type = 'original'
    # use the path printed in above output cell after running stock_cnn.py. It's in below format
    df = pd.read_csv(DATASET_PATH / f"df_{company_code}.csv")
    df['labels'] = df['labels'].astype(np.int8)
    if 'dividend_amount' in df.columns:
        df.drop(columns=['dividend_amount', 'split_coefficient'], inplace=True)
    logger.info(df.head())

    list_features = list(df.loc[:, 'open':'eom_26'].columns)
    logger.info('Total number of features', len(list_features))
    x_train, y_train, x_test, y_test, x_cv, y_cv = get_training_data()
    sample_weights = get_sample_weights(y_train)
    plot_training_data(x_train, y_train)
    params = {
        'input_shape': (15, 15, 3),
        'batch_size': 80,
        'conv2d_layers': {
        'conv2d_do_1': 0.2, 'conv2d_filters_1': 32, 'conv2d_kernel_size_1': 3, 'conv2d_mp_1': 0,
        'conv2d_strides_1': 1, 'kernel_regularizer_1': 0.0, 'conv2d_do_2': 0.3,
        'conv2d_filters_2': 64, 'conv2d_kernel_size_2': 3, 'conv2d_mp_2': 2,
        'conv2d_strides_2': 1,
        'kernel_regularizer_2': 0.0, 'layers': 'two'
        },
        'dense_layers': {
          'dense_do_1': 0.3, 'dense_nodes_1': 128,
          'kernel_regularizer_1': 0.0, 'layers': 'one'
        },
        'epochs': 3000, 'lr': 0.001, 'optimizer': 'adam'
    }
    if params.get('input_shape', None) is None:
        params['input_shape'] = x_train[0].shape()

    model = create_model_cnn(params)
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)

    best_model_path = os.path.join('.', 'best_model_keras')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                       patience=100, min_delta=0.0001)
    csv_logger = CSVLogger(os.path.join(OUTPUT_PATH, 'log_training_batch.csv'), append=True)
    rlp = ReduceLROnPlateau(monitor='val_loss', factor=0.02, patience=20, verbose=1, mode='min',
                            min_delta=0.001, cooldown=1, min_lr=0.0001)
    mcp = ModelCheckpoint(best_model_path, monitor='val_f1_metric', verbose=1,
                          save_best_only=True, save_weights_only=False, mode='max', period=1)  # val_f1_metric

    history = model.fit(
        x_train, y_train, epochs=params['epochs'], verbose=1,
        batch_size=64, shuffle=True,
        # validation_split=0.3,
        validation_data=(x_cv, y_cv),
        callbacks=[mcp, rlp, es],
        sample_weight=sample_weights
    )
    plot_model_history_results(history)
    show_model_status(model)

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from configparser import ConfigParser
from datetime import datetime
from src import preprocess

cfg = ConfigParser()
cfg.read('../config.cfg')
DEFAULTS = cfg['DEFAULT']


def train_model(model_type, X_train, y_train):
    y_train = y_train.astype('int')
    if model_type == DEFAULTS['ModelSVM']:
        clf = svm.SVC(
            kernel=DEFAULTS['Kernel'],
            C=float(DEFAULTS['C']),
            gamma=float(DEFAULTS['Gamma']),
            degree=int(DEFAULTS['Degree'])
        )
    elif model_type == DEFAULTS['ModelRandomForest']:
        clf = RandomForestClassifier(
            random_state=int(DEFAULTS['RFRandomState']),
            n_estimators=int(DEFAULTS['NEstimators']),
            class_weight=DEFAULTS['ClassWeight']
        )
    elif model_type == DEFAULTS['ModelDecisionTree']:
        clf = DecisionTreeClassifier(
            random_state=int(DEFAULTS['DTRandomState']),
            max_depth=int(DEFAULTS['MaxDepth'])
        )

    clf.fit(X_train, y_train)
    return clf


def train(jr_train, mv_train, hp_train):
    start_time = datetime.now()

    # msk = np.random.rand(len(mv_train)) < 0.8
    #
    # mv_train_sampled = preprocess.undersample(mv_train[msk])
    # mv_test_sampled = mv_train[~msk]
    #
    # mv_y_train = mv_train_sampled[DEFAULTS['Target']]
    # mv_X_train = mv_train_sampled.drop(DEFAULTS['TrainingTargets'].split(','), axis=1)
    #
    # mv_y_test = mv_test_sampled[DEFAULTS['Target']]
    # mv_X_test = mv_test_sampled.drop(DEFAULTS['TrainingTargets'].split(','), axis=1)

    mv_y = mv_train[DEFAULTS['Target']]
    mv_X = mv_train.drop(DEFAULTS['TrainingTargets'].split(','), axis=1)
    mv_X_train, mv_X_test, mv_y_train, mv_y_test = \
        train_test_split(mv_X, mv_y, test_size=0.2, random_state=1)

    model_type = DEFAULTS['Model']
    clf = train_model(model_type, mv_X_train, mv_y_train)

    y_pred = clf.predict(mv_X_test)

    df = pd.DataFrame({'Actual': mv_y_test, 'Predicted': y_pred.flatten()})
    df.to_csv(DEFAULTS['TrainingOutputFile'])

    precision = metrics.precision_score(mv_y_test, y_pred)
    recall = metrics.recall_score(mv_y_test, y_pred)
    f1_score = (precision * recall) / (precision + recall)
    print('Accuracy: ', metrics.accuracy_score(mv_y_test, y_pred))
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1-Score: ', f1_score)
    print('Time Taken: ', datetime.now() - start_time)


def predict(mv_train, mv_test, output_df):
    start_time = datetime.now()

    # clf = None
    # max_acc = 0
    # for _ in range(100):
    #     msk = np.random.rand(len(mv_train)) < 0.8
    #
    #     mv_train_sampled = preprocess.undersample(mv_train[msk])
    #     mv_test_sampled = mv_train[~msk]
    #
    #     mv_y_train = mv_train_sampled[DEFAULTS['Target']]
    #     mv_X_train = mv_train_sampled.drop(DEFAULTS['TrainingTargets'].split(','), axis=1)
    #
    #     mv_y_test = mv_test_sampled[DEFAULTS['Target']]
    #     mv_X_test = mv_test_sampled.drop(DEFAULTS['TrainingTargets'].split(','), axis=1)
    #
    #     model_type = DEFAULTS['Model']
    #     test_clf = train_model(model_type, mv_X_train, mv_y_train)
    #
    #     y_pred = test_clf.predict(mv_X_test)
    #     acc = metrics.accuracy_score(mv_y_test, y_pred)
    #
    #     if acc > max_acc:
    #         clf = test_clf
    #         max_acc = acc
    #
    # print(max_acc)

    mv_y = mv_train[DEFAULTS['Target']]
    mv_X = mv_train.drop(DEFAULTS['TrainingTargets'].split(','), axis=1)

    mv_test = mv_test.drop(DEFAULTS['TrainingTargets'].split(','), axis=1)

    model_type = DEFAULTS['Model']
    clf = train_model(model_type, mv_X, mv_y)

    y_pred = clf.predict(mv_test)

    output_df['set_clicked'] = y_pred.flatten()
    output_df.to_csv(DEFAULTS['OutputFile'])

    print(output_df['set_clicked'].value_counts())
    print('Time Taken: ', datetime.now() - start_time)


def main(training):
    """
    Runs train or predict based on the boolean passed in.
    :param training:
    :return:
    """
    jabref_train, myvolts_train, homepage_train = preprocess.get_trainings_dfs()
    if training:
        train(jabref_train, myvolts_train, homepage_train)
        # tune_hyperparams(jabref_train, myvolts_train, homepage_train)
    else:
        myvolts_train, myvolts_test, output_df = preprocess.get_training_and_test_dfs()
        predict(myvolts_train, myvolts_test, output_df)


if __name__ == '__main__':
    main(training=int(DEFAULTS['Training']))

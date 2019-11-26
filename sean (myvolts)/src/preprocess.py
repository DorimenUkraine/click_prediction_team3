import pandas as pd
from configparser import ConfigParser
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from category_encoders import CatBoostEncoder

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

cfg = ConfigParser()
cfg.read('../config.cfg')
DEFAULTS = cfg['DEFAULT']


def get_trainings_dfs():
    """
    Read in the training data csv file and return a cleaned dataframe for each organization
    :return:
    """
    missing_values = DEFAULTS['MissingValues'].split(',')
    df = pd.read_csv(DEFAULTS['TrainingFile'], na_values=missing_values)

    jabref_df, myvolts_df, homepage_df = split_data(df)

    jabref_df = clean_jabref(jabref_df)
    myvolts_df = clean_myvolts(myvolts_df)
    homepage_df = clean_homepage(homepage_df)

    return jabref_df, myvolts_df, homepage_df


def get_training_and_test_dfs():
    """
    Read in the test data csv file and return a dataframe for each organization
    :return:
    """
    missing_values = DEFAULTS['MissingValues'].split(',')
    df_train = pd.read_csv(DEFAULTS['TrainingFile'], na_values=missing_values)
    df_test = pd.read_csv(DEFAULTS['TestFile'], na_values=missing_values)
    output_df = pd.DataFrame(columns=['recommendation_set_id', 'set_clicked'])

    jabref_train, myvolts_train, homepage_train = split_data(df_train)
    jabref_test, myvolts_test, homepage_test = split_data(df_test)

    output_df['recommendation_set_id'] = myvolts_test['recommendation_set_id'].copy()

    dropped_cols = DEFAULTS['MyVoltsDroppedCols'].split(',') + DEFAULTS['MyVoltsIgnoredCols'].split(',')
    myvolts_train = myvolts_train.drop(dropped_cols, axis=1)
    myvolts_test = myvolts_test.drop(dropped_cols, axis=1)

    for col in DEFAULTS['MyVoltsNumberCols'].split(','):
        mean = myvolts_train[col].mean()
        myvolts_train[col].fillna(mean, inplace=True)

        mean = myvolts_test[col].mean()
        myvolts_test[col].fillna(mean, inplace=True)

    myvolts_train.fillna('unknown', inplace=True)
    myvolts_test.fillna('unknown', inplace=True)

    # myvolts_train['train'] = 1
    # myvolts_test['train'] = 0

    encode_cols = DEFAULTS['MyVoltsEncodeCols'].split(',')

    cbe = CatBoostEncoder(cols=encode_cols, return_df=True, drop_invariant=True, handle_missing='return_nan')
    cbe.fit(X=myvolts_train, y=myvolts_train['set_clicked'])
    myvolts_train = cbe.transform(myvolts_train)
    myvolts_test = cbe.transform(myvolts_test)

    # combined = pd.concat([myvolts_train, myvolts_test])
    # combined = oh_encode(combined, encode_cols)

    # label_encode_cols = DEFAULTS['MyVoltsLabelEncodeCols'].split(',')
    # combined = label_encode(combined, label_encode_cols)

    # myvolts_train = combined[combined['train'] == 1]
    # myvolts_test = combined[combined['train'] == 0]
    # myvolts_train = myvolts_train.drop(['train'], axis=1)
    # myvolts_test = myvolts_test.drop(['train'], axis=1)

    return myvolts_train, myvolts_test, output_df


def split_data(df):
    """
    Split dataset into three subsets, one for each organization
    :param df:
    :return:
    """
    jabref_df = df[df['organization_id'] == int(DEFAULTS['JabRef'])]
    myvolts_df = df[df['organization_id'] == int(DEFAULTS['MyVolts'])]
    homepage_df = df[df['organization_id'] == int(DEFAULTS['HomePage'])]

    return jabref_df, myvolts_df, homepage_df


def clean_jabref(df):
    dropped_cols =  df.drop(DEFAULTS['JabRefDroppedCols'].split(','), axis=1)
    new_df = df.drop(dropped_cols, axis=1)

    return new_df


def clean_myvolts(df):
    dropped_cols = DEFAULTS['MyVoltsDroppedCols'].split(',') + DEFAULTS['MyVoltsIgnoredCols'].split(',')
    new_df = df.drop(dropped_cols, axis=1)

    # new_df.dropna(inplace=True)

    for col in DEFAULTS['MyVoltsNumberCols'].split(','):
        mean = new_df[col].mean()
        new_df[col].fillna(mean, inplace=True)
    new_df.fillna('unknown', inplace=True)

    encode_cols = DEFAULTS['MyVoltsEncodeCols'].split(',')

    cbe = CatBoostEncoder(cols=encode_cols, return_df=True, drop_invariant=True, handle_missing='return_nan')
    cbe.fit(X=new_df, y=new_df['set_clicked'])
    new_df = cbe.transform(new_df)

    # one_hot_encode_cols = DEFAULTS['MyVoltsOneHotEncodeCols'].split(',')
    # new_df = oh_encode(new_df, one_hot_encode_cols)

    # label_encode_cols = DEFAULTS['MyVoltsLabelEncodeCols'].split(',')
    # new_df = label_encode(new_df, label_encode_cols)

    return new_df


def clean_homepage(df):
    return df.drop(DEFAULTS['HomePageDroppedCols'].split(','), axis=1)


def oh_encode(df, cols):
    """
    One Hot Encode the columns defined in DUMMY_COLS
    :param df:
    :return:
    """
    ohe = OneHotEncoder()
    for col in cols:
        # df[col] = ohe.fit_transform(df[col])
        df = pd.concat((df.drop(columns=col), pd.get_dummies(df[col], drop_first=True)), axis=1)
    return df


def label_encode(df, cols):
    le = LabelEncoder()
    for col in cols:
        df[col] = le.fit_transform(df[col])
    return df


# https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets
def undersample(df):
    count_class_0, count_class_1 = df[DEFAULTS['Target']].value_counts()

    df_class_0 = df[df[DEFAULTS['Target']] == 0]
    df_class_1 = df[df[DEFAULTS['Target']] == 1]

    df_class_0_under = df_class_0.sample(count_class_1)
    df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)

    return df_test_under


def oversample(df):
    count_class_0, count_class_1 = df[DEFAULTS['Target']].value_counts()

    df_class_0 = df[df[DEFAULTS['Target']] == 0]
    df_class_1 = df[df[DEFAULTS['Target']] == 1]

    df_class_1_over = df_class_1.sample(count_class_0, replace=True)
    df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

    return df_test_over

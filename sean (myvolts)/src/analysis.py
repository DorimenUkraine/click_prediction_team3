import pandas as pd
import matplotlib.pyplot as plt
from configparser import ConfigParser
from src import preprocess

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

cfg = ConfigParser()
cfg.read('../config.cfg', encoding='utf-8')
DEFAULTS = cfg['DEFAULT']


def count_values(df, col):
    print(pd.crosstab(df[col], df.set_clicked, normalize='index'))


def show_cols_to_drop(df):
    """
    Find columns in each dataset that have only 1 unique entry and print the list
    :param df:
    :return:
    """
    cols_to_drop = []
    for col in df:
        if df[col].nunique() == 1:
            cols_to_drop.append(col)
    print(','.join(cols_to_drop))


def show_corr_matrix(df):
    corr_matrix = df.corr().abs()
    print(corr_matrix[DEFAULTS['Target']].sort_values(ascending=False))

    plt.matshow(corr_matrix)
    plt.show()


def main():
    # df_train = preprocess.get_df_from_csv(DEFAULTS['TrainingDataFile'])
    # count_values(df_train, 'recommendation_algorithm_id_used')
    # count_values(df_train, 'organization_id')
    # pd.set_option('display.max_columns', 500)
    # print(df_train.describe())

    # show_cols_to_drop(myvolts_df)
    # show_cols_to_drop(jabref_df)
    # show_cols_to_drop(homepage_df)

    jabref_df, myvolts_df, homepage_df = preprocess.get_trainings_dfs()

    # print(len(jabref_df.columns))
    # print(len(myvolts_df.columns))
    # print(len(homepage_df.columns))

    mv_drop = DEFAULTS['MyVoltsDroppedCols'].split(',')
    jr_drop = DEFAULTS['JabRefDroppedCols'].split(',')
    mv_ignore = DEFAULTS['MyVoltsIgnoredCols'].split(',')
    jr_ignore = DEFAULTS['JabRefIgnoredCols'].split(',')
    mv_encode = DEFAULTS['MyVoltsLabelEncodeCols'].split(',')
    jr_encode = DEFAULTS['JabRefLabelEncodeCols'].split(',')
    mv_number_cols = DEFAULTS['MyVoltsNumberCols'].split(',')

    print(jabref_df.columns)

    for col in jr_encode:
        if col in jr_drop:
            print(jabref_df[col].value_counts())

    # show_corr_matrix(myvolts_df)


if __name__ == '__main__':
    main()

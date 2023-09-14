import numpy as np
from sklearn.model_selection import train_test_split

def recode_variables(df):
    # Gender
    df['Female'] = df['Gender'] - 1

    # Ethnicity
    df['Eth_Mex_Am'] = (df['Ethnicity'] == 1).astype(int)
    df.loc[df['Ethnicity'].isnull(), 'Eth_Mex_Am'] = np.nan

    df['Eth_Oth_Hisp'] = (df['Ethnicity'] == 2).astype(int)
    df.loc[df['Ethnicity'].isnull(), 'Eth_Oth_Hisp'] = np.nan

    df['Eth_Nonhisp_White'] = (df['Ethnicity'] == 3).astype(int)
    df.loc[df['Ethnicity'].isnull(), 'Eth_Nonhisp_White'] = np.nan

    df['Eth_Nonhisp_Black'] = (df['Ethnicity'] == 4).astype(int)
    df.loc[df['Ethnicity'].isnull(), 'Eth_Nonhisp_Black'] = np.nan

    df['Eth_Other'] = (df['Ethnicity'] == 5).astype(int)
    df.loc[df['Ethnicity'].isnull(), 'Eth_Other'] = np.nan
    df.drop(columns=['Ethnicity'], inplace=True)

    # Education
    df['Ed_less_9th'] = (df['Education'] == 1).astype(int)
    df.loc[df['Education'].isnull(), 'Ed_less_9th'] = np.nan

    df['Ed_9th_11th'] = (df['Education'] == 2).astype(int)
    df.loc[df['Education'].isnull(), 'Ed_9th_11th'] = np.nan

    df['Ed_HS_GED'] = (df['Education'] == 3).astype(int)
    df.loc[df['Education'].isnull(), 'Ed_HS_GED'] = np.nan

    df['Ed_some_coll_AA'] = (df['Education'] == 4).astype(int)
    df.loc[df['Education'].isnull(), 'Ed_some_coll_AA'] = np.nan

    df['Ed_coll_above'] = (df['Education'] == 5).astype(int)
    df.loc[df['Education'].isnull(), 'Ed_coll_above'] = np.nan

    df['Ed_refused'] = (df['Education'] == 7).astype(int)
    df.loc[df['Education'].isnull(), 'Ed_refused'] = np.nan

    df['Ed_dk'] = (df['Education'] == 9).astype(int)
    df.loc[df['Education'].isnull(), 'Ed_dk'] = np.nan
    df.drop(columns=['Education'], inplace=True)

    return df

def split_train_cal_test(X, y, trn_prop, cal_prop):
    X_train, X_test_cal, y_train, y_test_cal = train_test_split(X, y, train_size=trn_prop)
    cal_prop_rel = cal_prop / (1 - trn_prop)
    X_cal, X_test, y_cal, y_test = train_test_split(X_test_cal, y_test_cal, train_size=cal_prop_rel)
    return X_train, X_cal, X_test, y_train, y_cal, y_test


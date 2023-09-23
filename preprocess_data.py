import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

def recode_variables_NHANES(df):
    df = df.replace(to_replace=' ', value=np.nan)
    df['Education'] = df['Education'].astype(float)

    # RENAME TO CAPS
    df['AGE'] = df['Age']
    df['WEIGHT'] = df['Weight']
    df['HEIGHT'] = df['Height']
    df['WAIST'] = df['Waist']

    # Gender
    df['FEMALE'] = df['Gender'] - 1

    # Ethnicity
    df['HISPANIC'] = ((df['Ethnicity'] == 1) | (df['Ethnicity'] == 2)).astype(int)
    df.loc[df['Ethnicity'].isnull(), 'HISPANIC'] = np.nan

    df['WHITE'] = (df['Ethnicity'] == 3).astype(int)
    df.loc[df['Ethnicity'].isnull(), 'WHITE'] = np.nan

    df['BLACK'] = (df['Ethnicity'] == 4).astype(int)
    df.loc[df['Ethnicity'].isnull(), 'BLACK'] = np.nan

    df['OTHER_MIXED'] = (df['Ethnicity'] == 5).astype(int)
    df.loc[df['Ethnicity'].isnull(), 'OTHER_MIXED'] = np.nan
    df.drop(columns=['Ethnicity'], inplace=True)

    # Education
    df['Ed_less_9th'] = (df['Education'] == 1.).astype(int)
    df.loc[df['Education'].isnull(), 'Ed_less_9th'] = np.nan

    df['Ed_9th_11th'] = (df['Education'] == 2.).astype(int)
    df.loc[df['Education'].isnull(), 'Ed_9th_11th'] = np.nan

    df['Ed_HS_GED'] = (df['Education'] == 3.).astype(int)
    df.loc[df['Education'].isnull(), 'Ed_HS_GED'] = np.nan

    df['Ed_some_coll_AA'] = (df['Education'] == 4.).astype(int)
    df.loc[df['Education'].isnull(), 'Ed_some_coll_AA'] = np.nan

    df['Ed_coll_above'] = (df['Education'] == 5.).astype(int)
    df.loc[df['Education'].isnull(), 'Ed_coll_above'] = np.nan

    df['Ed_refused'] = (df['Education'] == 7.).astype(int)
    df.loc[df['Education'].isnull(), 'Ed_refused'] = np.nan

    df['Ed_dk'] = (df['Education'] == 9.).astype(int)
    df.loc[df['Education'].isnull(), 'Ed_dk'] = np.nan
    df.drop(columns=['Education'], inplace=True)

    return df

def recode_variables_LOOK_AHEAD(df):
    df = df.replace(to_replace='Missing', value=np.nan)
    new_df = pd.DataFrame(index=df['P_ID'])
    new_df['FEMALE'] = df['FEMALE'].map({'Yes': 1, 'No': 0}).values

    # RACE/ETHNICITY
    new_df['WHITE'] = (df['RACEVAR'] == 'White').astype('int').values
    new_df['HISPANIC'] = (df['RACEVAR'] == 'Hispanic').astype('int').values
    new_df['BLACK'] = (df['RACEVAR'] == 'African American / Black (not Hispanic)').astype('int').values
    new_df['OTHER_MIXED'] = (df['RACEVAR'] == 'Other/Mixed').astype('int').values

    new_df['BMI'] = df['bmi'].values
    new_df['AGE'] = df['age'].values
    new_df['WEIGHT'] = ((df['weight1_kg'] + df['weight2_kg']) / 2).values
    new_df['HEIGHT'] = df['eshgt_mean'].values
    new_df['WAIST'] = df['waistcm_mean'].values
    return new_df


def split_train_cal_test(X, y, trn_prop, cal_prop):
    X_train, X_test_cal, y_train, y_test_cal = train_test_split(X, y, train_size=trn_prop)
    cal_prop_rel = cal_prop / (1 - trn_prop)
    X_cal, X_test, y_cal, y_test = train_test_split(X_test_cal, y_test_cal, train_size=cal_prop_rel)
    return X_train, X_cal, X_test, y_train, y_cal, y_test


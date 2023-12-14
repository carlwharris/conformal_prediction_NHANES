import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def recode_variables_NHANES(df):
    """
    Recode variables from the NHANES dataset.

    Args:
        df (pd.DataFrame): Original data.

    Returns:
        pd.DataFrame: Recoded data.
    """

    # Replace empty spaces with NaNs and convert relevant columns to float
    df = df.replace(to_replace=' ', value=np.nan)
    columns_to_float = ['Education', 'Ethnicity', 'Age', 'Weight', 'Height', 'BMI', 'Waist']
    for column in columns_to_float:
        df[column] = df[column].astype(float)

    # Rename columns to upper case
    for column in ['Age', 'Weight', 'Height', 'Waist']:
        df[column.upper()] = df[column]
        df.drop(columns=column, inplace=True)

    # Recode 'Gender' to 'FEMALE' as 0 or 1
    df['FEMALE'] = df['Gender'] - 1
    df.drop(columns=['Gender'], inplace=True)

    # Recode 'Ethnicity' to various ethnic group columns
    ethnicity_mappings = {
        'HISPANIC': [1., 2.],
        'WHITE': [3.],
        'BLACK': [4.],
        'OTHER_MIXED': [5.]
    }
    for group, values in ethnicity_mappings.items():
        df[group] = df['Ethnicity'].isin(values).astype(float)
        df.loc[df['Ethnicity'].isnull(), group] = np.nan
    df.drop(columns=['Ethnicity'], inplace=True)

    # Recode 'Education' to various education level columns
    education_mappings = {
        'ED_LESS_HS': [1., 2.],
        'ED_HS_GED': [3.],
        'ED_SOME_COLL_AA': [4.],
        'ED_COLL_ABOVE': [5.],
        'ED_OTHR_DK': [7., 9.]
    }
    for level, values in education_mappings.items():
        df[level] = df['Education'].isin(values).astype(float)
        df.loc[df['Education'].isnull(), level] = np.nan
    df.drop(columns=['Education'], inplace=True)

    return df


def recode_variables_LOOK_AHEAD(df):
    """
    Recode variables from the LOOK_AHEAD dataset.

    Args:
        df (pd.DataFrame): Original data.

    Returns:
        pd.DataFrame: Recoded data.
    """

    # Replace 'Missing' with NaN
    df = df.replace(to_replace='Missing', value=np.nan)

    # Initialize new dataframe with 'P_ID' as index
    new_df = pd.DataFrame(index=df['P_ID'])

    # Map 'FEMALE' column values to 0 or 1
    new_df['FEMALE'] = df['FEMALE'].map({'Yes': 1, 'No': 0}).values

    # Recode 'RACEVAR' to various ethnic group columns
    race_mappings = {
        'WHITE': 'White',
        'HISPANIC': 'Hispanic',
        'BLACK': 'African American / Black (not Hispanic)',
        'OTHER_MIXED': 'Other/Mixed'
    }
    for group, value in race_mappings.items():
        new_df[group] = (df['RACEVAR'] == value).astype(float).values
        df.loc[df['RACEVAR'].isnull(), group] = np.nan

    # Recode 'SDEDUC' to various education level columns
    education_mappings = {
        'ED_LESS_HS': 'Less than high school',
        'ED_HS_GED': 'High school diploma or equivalency (GED)',
        'ED_SOME_COLL_AA': ['Some college', 'Associate degree (junior college)', 'Some vocational school'],
        'ED_COLL_ABOVE': ["Bachelor's degree", 'Some graduate school', 'Professional (MD, JD, DDS, etc.)'],
        'ED_OTHR_DK': 'Other'
    }
    for level, value in education_mappings.items():
        if isinstance(value, list):
            new_df[level] = df['SDEDUC'].isin(value).astype(float).values
        else:
            new_df[level] = (df['SDEDUC'] == value).astype(float).values
        df.loc[df['SDEDUC'].isnull(), level] = np.nan

    # Assign measurement columns to new dataframe
    new_df['BMI'] = df['bmi'].values
    new_df['AGE'] = df['age'].values
    new_df['WEIGHT'] = ((df['weight1_kg'] + df['weight2_kg']) / 2).values
    new_df['HEIGHT'] = df['eshgt_mean'].values
    new_df['WAIST'] = df['waistcm_mean'].values

    return new_df


def split_train_cal_test(X, y, trn_prop, cal_prop, random_state=42):
    """
    Split the dataset into training, calibration, and test sets.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        trn_prop (float): Proportion of data for training set.
        cal_prop (float): Proportion of data for calibration set.

    Returns:
        tuple: Training, calibration, and test sets.
    """

    X_train, X_test_cal, y_train, y_test_cal = train_test_split(X, y, train_size=trn_prop, random_state=random_state)
    cal_prop_rel = cal_prop / (1 - trn_prop)
    X_cal, X_test, y_cal, y_test = train_test_split(X_test_cal, y_test_cal, train_size=cal_prop_rel, random_state=random_state  )

    X = {}
    X['train'] = X_train
    X['cal'] = X_cal
    X['test'] = X_test

    y = {}
    y['train'] = y_train
    y['cal'] = y_cal
    y['test'] = y_test

    return X,y

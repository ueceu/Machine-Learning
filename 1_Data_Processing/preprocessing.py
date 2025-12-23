import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def load_dataset(path):
    return pd.read_csv(path)


def split_features_target(dataset, target_column):
    X = dataset.drop(target_column, axis=1).values
    y = dataset[target_column].values
    return X, y


def impute_missing_values(X, numeric_columns):
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X[:, numeric_columns] = imputer.fit_transform(X[:, numeric_columns])
    return X


def encode_categorical_feature(X, categorical_index):
    ct = ColumnTransformer(
        transformers=[('encoder', OneHotEncoder(), [categorical_index])],
        remainder='passthrough'
    )
    return np.array(ct.fit_transform(X))


def encode_target(y):
    le = LabelEncoder()
    return le.fit_transform(y)


def split_train_test(X, y, test_size=0.2, random_state=1):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def scale_features(X_train, X_test, start_index):
    sc = StandardScaler()
    X_train[:, start_index:] = sc.fit_transform(X_train[:, start_index:])
    X_test[:, start_index:] = sc.transform(X_test[:, start_index:])
    return X_train, X_test

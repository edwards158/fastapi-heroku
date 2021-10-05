# Script to train machine learning model.
from sklearn.model_selection import train_test_split
import pandas as pd
from joblib import dump
import src.ml_functions


def train_model():
    '''
    Train the model on the train data
    '''
    train_data = pd.read_csv('data/census_train_cleaned.csv')
    train, test = train_test_split(train_data, test_size=0.20)

    categories = list(train.select_dtypes(include='object').columns)[:-1]

    X_train, y_train, encoder, lb = src.ml_functions.process_data(
        train, categories, label="salary", training=True
    )

    model = src.ml_functions.train_model(X_train, y_train)

    dump(model, "model/model.joblib")
    dump(encoder, "model/encoder.joblib")
    dump(lb, "model/lb.joblib")

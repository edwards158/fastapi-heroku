from fastapi import FastAPI
from pydantic import BaseModel
#import census.Census
from typing import Literal
from joblib import load
import src.ml_functions
from pandas.core.frame import DataFrame
import numpy as np
import os

WORKCLASS_VALUES = Literal['State-gov', 'Self-emp-not-inc', 'Private',
                           'Federal-gov', 'Local-gov', 'Self-emp-inc',
                           'Without-pay']
EDUCATION_VALUES = Literal['Bachelors', 'HS-grad', '11th', 'Masters', '9th',
                           'Some-college', 'Assoc-acdm', '7th-8th',
                           'Doctorate', 'Assoc-voc', 'Prof-school',
                           '5th-6th', '10th', 'Preschool', '12th', '1st-4th']
MARITAL_STATUS_VALUES = Literal['Never-married', 'Married-civ-spouse',
                                'Divorced', 'Married-spouse-absent',
                                'Separated', 'Married-AF-spouse', 'Widowed']
OCCUPATION_VALUES = Literal['Adm-clerical', 'Exec-managerial',
                            'Handlers-cleaners', 'Prof-specialty',
                            'Other-service', 'Sales', 'Transport-moving',
                            'Farming-fishing', 'Machine-op-inspct',
                            'Tech-support', 'Craft-repair', 'Protective-serv',
                            'Armed-Forces', 'Priv-house-serv']
RELATIONSHIP_VALUES = Literal['Not-in-family', 'Husband', 'Wife', 'Own-child',
                              'Unmarried', 'Other-relative']
RACE_CLASS = Literal['White', 'Black', 'Asian-Pac-Islander',
                     'Amer-Indian-Eskimo', 'Other']
SEX_CLASS = Literal['Male', 'Female']
NATIVE_COUNTY_CLASS = Literal['United-States', 'Cuba', 'Jamaica', 'India',
                              'Mexico', 'Puerto-Rico', 'Honduras', 'England',
                              'Canada', 'Germany', 'Iran', 'Philippines',
                              'Poland', 'Columbia', 'Cambodia', 'Thailand',
                              'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal',
                              'Dominican-Republic', 'El-Salvador', 'France',
                              'Guatemala', 'Italy', 'China', 'South', 'Japan',
                              'Yugoslavia', 'Peru',
                              'Outlying-US(Guam-USVI-etc)', 'Scotland',
                              'Trinadad&Tobago', 'Greece', 'Nicaragua',
                              'Vietnam', 'Hong', 'Ireland', 'Hungary',
                              'Holand-Netherlands']


class Census(BaseModel):
    workclass: WORKCLASS_VALUES
    education: EDUCATION_VALUES
    marital_status: MARITAL_STATUS_VALUES
    occupation: OCCUPATION_VALUES
    relationship: RELATIONSHIP_VALUES
    race: RACE_CLASS
    sex: SEX_CLASS
    native_country: NATIVE_COUNTY_CLASS
    age: int
    hours_per_week: int


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()


@app.get("/")
async def get_items():
    return {"message": "Hello"}


@app.post('/predict')
async def predict(data: Census):

    try:
        model = load(os.path.join(
            os.getcwd(), "starter/model/model_test.joblib"))
    except FileNotFoundError:
        model = load(os.path.join(os.getcwd(), "model/model_test.joblib"))

    try:
        encoder = load(os.path.join(
            os.getcwd(), "starter/model/encoder_test.joblib"))
    except FileNotFoundError:
        encoder = load(os.path.join(os.getcwd(), "model/encoder_test.joblib"))

    try:
        lb = load(os.path.join(
            os.getcwd(), "starter/model/lb_test.joblib"))
    except FileNotFoundError:
        lb = load(os.path.join(os.getcwd(), "model/lb_test.joblib"))

    data = data.dict()

    array = np.array([[
        data['workclass'],
        data['education'],
        data['marital_status'],
        data['occupation'],
        data['relationship'],
        data['race'],
        data['sex'],
        data['native_country'],
        data['age'],
        data['hours_per_week']
    ]])

    columns = ["workclass",
               "education",
               "marital-status",
               "occupation",
               "relationship",
               "race",
               "sex",
               "native-country",
               "age",
               "hours-per-week",
               ]

    categorical_cols = columns[: -2]

    df = DataFrame(data=array, columns=columns)

    X, _, _, _ = src.ml_functions.process_data(
        df,
        categorical_cols,
        encoder=encoder, lb=lb, training=False)
    pred = src.ml_functions.inference(model, X)
    y = lb.inverse_transform(pred)[0]
    return {"prediction": y}

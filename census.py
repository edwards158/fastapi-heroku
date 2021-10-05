from typing import Literal
from pydantic import BaseModel


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

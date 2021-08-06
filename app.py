## Author
'''
{
    'Objective': 'reusable code for ML training',
    'Author': 'Pinaki Brahma',
    'email': 'pinaki.brahma@walmart.com'
    'created_date': '2021-08-05',
    'modified_date': '2021-08-05'
}
'''

## libraries needed
import pandas as pd
import numpy as np

## load env file (yml)
DIR_PATH = '/Users/prb000j/OneDrive - Walmart Inc/Python Learn Projects/Python Projects/ML_Templates/'
INPUT_DATA_PATH = DIR_PATH + 'input_data/'
INTERM_DATA_PATH = DIR_PATH + 'saved_data/'
MODEL_PATH = DIR_PATH + 'saved_models/'
RESULT_PATH = DIR_PATH + 'saved_results/'

## read data
data_raw = pd.read_csv(INPUT_DATA_PATH + 'train.csv')
print(data_raw.shape)
print(data_raw.head(3))

## Basic EDA of data 
### target data EDA

### features EDA


## Split data into train & test

## Separating catg and numeric features


## Preprocessing data (Numeric features)
### missing values

### outliers


## Preprocessing data (Categorical features)
### missing values

### outliers

### encoding catg features
#### form of labeling / one hot encoding

#### target encoding


## feature reduction


## set of models
### model training

#### test prediction & evaluation

### voting or ensemble


## save pipeline / model


## using a trained model
### load pipeline / model

### use pipeline to preprocess new data

### predict using preprocessed new data


## feedback mechanism for continuous improvement

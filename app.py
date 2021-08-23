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
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

import env
import helper_func_EDA as h_EDA
import helper_func_preprocess as h_PP


print(f"DIR PATH: {env.INPUT_DATA_PATH}")

# ## read data
data_raw = pd.read_csv(env.INPUT_DATA_PATH + 'train.csv')
print(data_raw.shape)
print(data_raw.head(3))

target_feature = 'SalePrice'
EDA_obj = h_EDA.EDA(data_raw, target_feature)


## EDA
print(f"QUANTILE info for {target_feature}")
print(EDA_obj.df[EDA_obj.target_feature].describe(env.QUANTILE_LIST).to_frame().T)

catg_features = EDA_obj.df.select_dtypes(include='object').columns.to_list()
numeric_features = EDA_obj.df.select_dtypes(exclude='object').columns.to_list()

cols_w_manyNAs = EDA_obj.getColsWithManyNAs()
pseudo_catg_features = [col for col in numeric_features if EDA_obj.df[col].nunique()<50]
numeric_features_upd = list(set(numeric_features).difference(set(pseudo_catg_features + [target_feature])))
corr_df, h_corr_pair = EDA_obj.getRankedCorr(numeric_features_upd)
multi_coll_featr = EDA_obj.getMultiCollFeatr(numeric_features_upd)
feature_lists = [pseudo_catg_features, catg_features, cols_w_manyNAs]
cols_w_low_dev = EDA_obj.getCols_w_lowDev(feature_lists)
print(f"cols with low deviation: {cols_w_low_dev}")


print(f"correlated feature pairs")
print(h_corr_pair)
print(f"Drop these features from the above pairs")
print(multi_coll_featr)

### Plots
# plot1 = EDA_obj.plotCatgPlot(cols_w_manyNAs)
# plot2 = EDA_obj.plotCatgPlot(pseudo_catg_features, sns.barplot, def_cols=6)


## Preprocessing steps
PP_obj = h_PP.PreProcess(df=EDA_obj.df, target_fetaure=EDA_obj.target_feature,
        numeric_features=numeric_features_upd, catg_features=catg_features, 
        cols_w_manyNAs=cols_w_manyNAs, cols_w_low_dev=cols_w_low_dev,
        multi_coll_featr=multi_coll_featr)

PP_obj.preProcessData()


## Feature reduction
temp_X = PP_obj.X_test_dim_reduc.copy()
temp_y = PP_obj.y_test.copy()
n_components=2
dim_red_result_df = temp_X.copy()
dim_reduc_cols = [f"FACT_{i+1}" for i in range(n_components)]
print(f"dim reduced cols: {dim_reduc_cols}")
temp_df = temp_X.copy()
temp_df['label'] = temp_y.reset_index(drop=True)
print(f"Factors correlation with Target: {temp_df.corr()['label']}")

### Plots
# plot_dim_reduc = PP_obj.dimReducPlot(df=temp_X, y=temp_y)


## setting up models
### model training
model = RandomForestRegressor(random_state=0, )
model.fit(PP_obj.X_train_dim_reduc, PP_obj.y_train)
y_predict = model.predict(PP_obj.X_test_dim_reduc)

### test prediction & evaluation
metrics = [mean_absolute_percentage_error, mean_squared_error, r2_score]
for metric in metrics:
    print(f"{metric.__name__}: {metric(PP_obj.y_test, y_predict):.4f}")

### voting or ensemble


## save pipeline / model


## using a trained model
### load pipeline / model

### use pipeline to preprocess new data

### predict using preprocessed new data


## feedback mechanism for continuous improvement

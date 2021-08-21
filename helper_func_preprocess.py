import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class PreProcess:
    '''
    Class that handles all the 
    Preprocessing steps on the data before modeling
    Includes steps like splitting data, 
    missing value handling, outlier handling, etc.
    '''
    TEST_SIZE = 0.2
    RANDOM_STATE = 0
    DIM_REDUC_METHOD = TSNE
    DIM_REDUC_N_COMPONENTS = 2

    def __init__(self, df:pd.DataFrame, target_fetaure:str, numeric_features:list=None, catg_features:list=None, cols_w_manyNAs:list=None, cols_w_low_dev:list=None, multi_coll_featr:list=None):
        '''
        initiate object of Preprocess class
        '''
        self.df = df
        self.target_feature = target_fetaure
        self.cols_to_drop = list(set(cols_w_manyNAs + cols_w_low_dev))
        ## update catg & numeric features
        self.catg_features_upd = list(set(catg_features).\
            difference(set(self.cols_to_drop)))
        self.numeric_features_upd = list(set(numeric_features).\
            difference(set(self.cols_to_drop + multi_coll_featr)))
        print(f'Properties of the DF')
        print(f"Shape of df: {self.df.shape}")
        print(f"Target Feature: {self.target_feature}")


    def preProcessData(self, new_df=None, new_data=False, dim_reduc=False):
        if new_data==False:
            ## Create X & y from df
            self.X = self.df[self.catg_features_upd + self.numeric_features_upd]
            self.y = self.df[self.target_feature]
            print(f"Shape of X: {self.X.shape}")
            print(f"Shape of y: {self.y.shape}")
            ## Split DF
            self.splitDf()
            ## Impute Outliers in Numeric Cols
            self.X_train_imp = self.fitTransformOutl(self.X_train)
            self.X_test_imp = self.transformOutl(self.X_test)
            ## Missing Value Treatment - Numeric & Catg
            self.X_train_na_imp = self.fitImputeNa(self.X_train, self.X_train_imp)
            self.X_test_na_imp = self.transformImputeNa(self.X_test, self.X_test_imp)
            ## Bucketing infrequent feature values into 'Other'
            self.getColsOheTargEncd()
            self.X_train_bkt = self.fit_transform_bktInfreqVal(self.X_train_na_imp, self.catg_features_upd)
            self.X_test_bkt = self.transform_bktInfreqVal(self.X_test_na_imp, self.catg_features_upd)
            ## Encoding Catg Features
            self.X_train_enc = self.combEncdNumCols(self.X_train_bkt, self.catg_features_upd, training=True)
            self.X_test_enc = self.combEncdNumCols(self.X_test_bkt, self.catg_features_upd, training=False)
            ## Scaling DF
            self.X_train_sc = self.scaleDf(self.X_train_enc, training=True, scaler=MinMaxScaler)
            self.X_test_sc = self.scaleDf(self.X_test_enc, training=False)
            ## Feature reduction like PCA or t-SNE
            self.X_train_dim_reduc = self.processDimReduc(self.X_train_sc,
                method=self.DIM_REDUC_METHOD, n_components=self.DIM_REDUC_N_COMPONENTS, training=True)
            self.X_test_dim_reduc = self.processDimReduc(self.X_test_sc,
                method=self.DIM_REDUC_METHOD, n_components=self.DIM_REDUC_N_COMPONENTS, training=False)
            

        if new_data==True:
            if new_df.shape[0]==0:
                print(f"You haven't included the new data for processing")
                return None
            new_df_imp = self.transformOutl(new_df)
            new_df_na_imp = self.transformImputeNa(new_df, new_df_imp)
            new_df_bkt = self.transform_bktInfreqVal(new_df_na_imp, self.catg_features_upd)
            new_df_enc = self.combEncdNumCols(new_df_bkt, self.catg_features_upd, training=False)
            new_df_sc = self.scaleDf(new_df_enc, training=False)
            if dim_reduc==True:
                new_df_dim_reduc = self.processDimReduc(new_df_sc,
                    self.DIM_REDUC_METHOD, self.DIM_REDUC_N_COMPONENTS, training=False)
            else:
                new_df_dim_reduc = new_df_sc
            return new_df_dim_reduc


    ## ********* Split DF ********* ##
    def splitDf(self, test_size=TEST_SIZE, random_state=RANDOM_STATE):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
        print("Shape of train & test datasets are as follows:")
        print(self.X_train.shape, self.X_test.shape, self.y_train.shape, self.y_test.shape)


    ## ********* Impute Outliers in Numeric Cols ********* ##
    def fitTransformOutl(self, df):
        '''
        Fit outliers on training data and 
        fit the cut offs and finally 
        transform or treat these outliers
        '''
        cntr=0
        X_train_imp = pd.DataFrame()
        temp_cutoff_dict = {}
        remove_zeros_dict = {}
        for col in self.numeric_features_upd:
            # if cntr > 3:
            #     break
            # print(col)
            remove_zeros_dict[col], temp_cutoff_dict[col], imputed_col = self.cleanedUpOutl(df, col)
            X_train_imp = pd.concat([X_train_imp, imputed_col], axis=1)
            cntr+=1
        assert(X_train_imp.shape[1] == len(self.numeric_features_upd))
        self.remove_zeros_numCols = remove_zeros_dict
        self.imputation_cutoffs_numCols = temp_cutoff_dict
        # self.X_train_imp = X_train_imp
        print(f"cols outlier transformed: {cntr} out of {len(self.numeric_features_upd)} successfully")
        return X_train_imp

    
    def transformOutl(self, df):
        cntr=0
        X_test_imp = pd.DataFrame()
        for col in self.numeric_features_upd:
            # if cntr > 2:
            #     break
            col_ser = df[col].copy()
            cut_off = self.imputation_cutoffs_numCols[col]
            remove_zeros = self.remove_zeros_numCols[col]
            imputed_col = PreProcess.imputeOutl(col_ser, cut_off, remove_zeros, strategy='clip')
            X_test_imp = pd.concat([X_test_imp, imputed_col], axis=1)
            cntr+=1
        assert(X_test_imp.shape[1] == len(self.numeric_features_upd))
        # self.X_test_imp = X_test_imp
        print(f"cols outlier transformed: {cntr} out of {len(self.numeric_features_upd)} successfully")
        return X_test_imp


    def cleanedUpOutl(self, df, col:str):
        QUANTILE_LIST_LOWER = np.arange(0, 0.1, 0.01)
        QUANTILE_LIST_UPPER = np.arange(1, 0.9, -0.01)
        
        col_ser = df[col].copy()
        remove_zeros = PreProcess.removeZeros(col_ser)
        col_ser2 = col_ser[col_ser!=0] if remove_zeros else col_ser
        
        cut_off = {}
        for direction, quantile_list in zip(['L', 'U'], 
                                            [QUANTILE_LIST_LOWER, QUANTILE_LIST_UPPER]):
            cut_off_adj = 0.01 if direction=='L' else 0
            cut_off[direction] = PreProcess.getCutOff(col_ser2, quantile_list, cut_off_adj)
        
        imputed_col = PreProcess.imputeOutl(col_ser, cut_off, remove_zeros, strategy='clip')
        return remove_zeros, cut_off, imputed_col


    @staticmethod
    def removeZeros(col_ser:pd.Series, thresh:float=0.2):
        temp_df = col_ser.copy()
        col_len = temp_df.shape[0]
        zero_cnt = temp_df[temp_df==0].shape[0]
        zero_prop = zero_cnt/col_len
        return True if zero_prop > thresh else False


    @staticmethod
    def getCutOff(col_ser:pd.Series, quantile_list:list, cut_off_adj:float)->float:
        quantile_dict = {q: col_ser.quantile(q) for q in quantile_list}
        temp_df = pd.DataFrame(quantile_dict.items(), columns=['quantile', 'value'])
        temp_df['lag_value'] = temp_df['value'].shift(1)
        temp_df['value_change'] = abs((temp_df['lag_value']-temp_df['value'])/temp_df['value'])
        temp_df.sort_values(['value_change'], ascending=False, inplace=True)
        temp_df.reset_index(drop=True, inplace=True)
        cut_off_q = (temp_df['quantile'][0] - cut_off_adj)
        cut_off = temp_df[temp_df['quantile']==cut_off_q]['value'].to_list()
        if len(cut_off)==0:  # to handle out of index in cut_off
            cut_off_q = (temp_df['quantile'][0])
            cut_off = temp_df[temp_df['quantile']==cut_off_q]['value'].to_list()
        return cut_off[0]


    @staticmethod
    def imputeOutl(col_ser:pd.Series, cut_off:dict, remove_zeros, strategy:str='clip')->pd.Series:
        if strategy != 'clip':
            return col_ser
        if remove_zeros:
            imputed_col = col_ser.apply(lambda x: cut_off['L'] if x < cut_off['L'] else x)
            imputed_col = imputed_col.apply(lambda x: cut_off['U'] if x > cut_off['U'] else x)
        else:
            imputed_col = col_ser.apply(lambda x: cut_off['L'] if (x!=0 and x < cut_off['L']) else x)
            imputed_col = imputed_col.apply(lambda x: cut_off['U'] if (x!=0 and x > cut_off['U']) else x)    
        return imputed_col


    ## ********* Missing Value Treatment - Numeric & Catg ********* ##
    def fitImputeNa(self, df, df_imp):
        self.imputer_na_num = SimpleImputer(strategy='median')
        X_train_na_imp_num = pd.DataFrame(self.imputer_na_num.fit_transform(df_imp[self.numeric_features_upd]), 
                                    columns=self.numeric_features_upd)
        
        self.imputer_na_catg = SimpleImputer(strategy='most_frequent')
        X_train_na_imp_catg = pd.DataFrame(self.imputer_na_catg.fit_transform(df[self.catg_features_upd]), 
                                    columns=self.catg_features_upd)
        X_train_na_imp = pd.concat([X_train_na_imp_num, X_train_na_imp_catg], axis=1)
        assert(X_train_na_imp.shape[1] == len(self.numeric_features_upd + self.catg_features_upd))
        assert(X_train_na_imp.shape[0] == df.shape[0])
        return X_train_na_imp


    def transformImputeNa(self, df, df_imp):
        X_test_na_imp_num = pd.DataFrame(self.imputer_na_num.transform(df_imp[self.numeric_features_upd]), 
                                    columns=self.numeric_features_upd)
        X_test_na_imp_catg = pd.DataFrame(self.imputer_na_catg.transform(df[self.catg_features_upd]), 
                                    columns=self.catg_features_upd)
        X_test_na_imp = pd.concat([X_test_na_imp_num, X_test_na_imp_catg], axis=1)
        assert(X_test_na_imp.shape[1] == len(self.numeric_features_upd + self.catg_features_upd))
        assert(X_test_na_imp.shape[0] == df.shape[0])
        # self.X_test_na_imp = X_test_na_imp
        return X_test_na_imp


    ## ********* Encoding Catg Features ********* ##
    def getColsOheTargEncd(self):
        self.cols_for_target_encd = [col for col in self.catg_features_upd if self.X_train_na_imp[col].nunique() > 10]
        self.cols_for_ohe = list(set(self.catg_features_upd).difference(set(self.cols_for_target_encd)))


    def fit_transform_bktInfreqVal(self, df, catg_col):
        '''
        Fitting on Training data
        check if some values in a column are very rare
        bucket all such values into 'Other'
        Helps in reducing OHE columns later
        '''
        temp_df = df.copy()
        value_prop_dict = {col: temp_df[col].value_counts(normalize=True) for col in catg_col}
        self.value_prop_dict_final = {k: value_prop_dict[k][value_prop_dict[k] < 0.1].index.to_list() for k in value_prop_dict.keys()}
        for k, v in self.value_prop_dict_final.items():
            temp_df[k] = temp_df[k].map(lambda x: 'Other' if x in v else x)
        return temp_df


    def transform_bktInfreqVal(self, df, catg_col):
        '''
        Transform Test data based on Fit
        check if some values in a column are very rare
        bucket all such values into 'Other'
        Helps in reducing OHE columns later
        '''
        temp_df = df.copy()
        for k, v in self.value_prop_dict_final.items():
            temp_df[k] = temp_df[k].map(lambda x: 'Other' if x in v else x)
        return temp_df


    def getOheOutputCols(self, df, catg_col):
        col_ohe_val_dict = {col: list(df[col].unique()) for col in catg_col}
        self.op_cols_for_ohe = [f"{k}_{v}" for k, vals in col_ohe_val_dict.items() for v in vals]
        print(f"len of op cols for OHE: {len(self.op_cols_for_ohe)}")

    
    def combEncdNumCols(self, df, catg_col, training:str='True'):
        temp_df = df.copy()
        print(f"comb encd + num func: {temp_df.shape}")
        ## fit OHE on catg cols in training data
        if training:
            self.getOheOutputCols(temp_df[catg_col], catg_col)
            self.encoder_catg = OneHotEncoder(handle_unknown='ignore', sparse=False)
            temp_df_ohe = pd.DataFrame(self.encoder_catg.fit_transform(temp_df[catg_col]), columns=self.op_cols_for_ohe)
        ## transform catg cols in test data
        else:  # training==False
            temp_df_ohe = pd.DataFrame(self.encoder_catg.transform(temp_df[catg_col]), columns=self.op_cols_for_ohe)
        ## add back numeric columns to OHE cols
        cols_left = list(set(temp_df.columns).difference(set(catg_col)))
        temp_df_ohe2 = pd.concat([temp_df[cols_left], temp_df_ohe], axis=1)
        return temp_df_ohe2


    ## ********* Scaling DF ********* ##
    def scaleDf(self, df, training=True, scaler=MinMaxScaler):
        temp_df = df.copy()

        if training==True:
            if scaler==MinMaxScaler:
                self.scaler_obj = MinMaxScaler(clip=True)
            else:
                self.scaler_obj = StandardScaler()
            temp_df_sc = pd.DataFrame(self.scaler_obj.fit_transform(temp_df), 
                            columns=df.columns.to_list())
        else:  #training==False
            temp_df_sc = pd.DataFrame(self.scaler_obj.transform(temp_df), 
                            columns=df.columns.to_list())
        return temp_df_sc


    ## ********* Feature reduction like PCA or t-SNE ********* ##
    def processDimReduc(self, df, method=DIM_REDUC_METHOD, 
                        n_components=DIM_REDUC_N_COMPONENTS, training=True):
        temp_df = df.copy()
        fact_cols = [f"FACT_{i+1}" for i in range(n_components)]
        if training==True:
            dim_reduc = method(n_components)
            dim_reduc_result = dim_reduc.fit_transform(temp_df)
            dim_reduc_result_df = pd.DataFrame(data = dim_reduc_result, columns = fact_cols)
            self.dim_reduc = dim_reduc
        if training==False:
            dim_reduc_result = self.dim_reduc.transform(temp_df) if method==PCA else self.dim_reduc.fit_transform(temp_df)
            dim_reduc_result_df = pd.DataFrame(data = dim_reduc_result, columns = fact_cols)
        
        return dim_reduc_result_df
    

    def dimReducPlot(self, df, y):
        temp_df = df.copy()
        temp_y = y.copy()
        temp_df['label'] = temp_y.reset_index(drop=True)
        fact_cols = temp_df.columns.to_list()
        fig, ax = plt.subplots(1, figsize=(12, 6))
        sns.scatterplot(x=fact_cols[0], y=fact_cols[1], hue=fact_cols[2], 
            data=temp_df, ax=ax, s=120)
        lim = {col: [temp_df[col].min(), temp_df[col].max()] for col in temp_df.columns}
        ax.set_xlim(lim['FACT_1'])
        ax.set_ylim(lim['FACT_2'])
        ax.set_aspect('equal')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.title(f"Two dimensions explaing {self.target_feature} using {type(self.dim_reduc).__name__}")
        plt.show()

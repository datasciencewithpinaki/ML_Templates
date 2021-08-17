import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer

class PreProcess:
    '''
    Class that handles all the 
    Preprocessing steps on the data before modeling
    Includes steps like splitting data, 
    missing value handling, outlier handling, etc.
    '''
    TEST_SIZE = 0.2
    RANDOM_STATE = 0

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


    def preProcessData(self):
        ## Create X & y from df
        self.X = self.df[self.catg_features_upd + self.numeric_features_upd]
        self.y = self.df[self.target_feature]
        print(f"Shape of X: {self.X.shape}")
        print(f"Shape of y: {self.y.shape}")
        ## Split DF
        self.splitDf()
        ## Impute Outliers in Numeric Cols
        self.fitTransformOutl()
        self.transformOutl()
        ## Missing Value Treatment - Numeric & Catg

        ## Encoding Catg Features

        ## Scaling DF

        ## Feature reduction like PCA or t-SNE


    ## ********* Split DF ********* ##
    def splitDf(self, test_size=TEST_SIZE, random_state=RANDOM_STATE):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
        print("Shape of train & test datasets are as follows:")
        print(self.X_train.shape, self.X_test.shape, self.y_train.shape, self.y_test.shape)


    ## ********* Impute Outliers in Numeric Cols ********* ##
    def fitTransformOutl(self):
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
            remove_zeros_dict[col], temp_cutoff_dict[col], imputed_col = self.cleanedUpOutl(col)
            X_train_imp = pd.concat([X_train_imp, imputed_col], axis=1)
            cntr+=1
        assert(X_train_imp.shape[1] == len(self.numeric_features_upd))
        self.remove_zeros_numCols = remove_zeros_dict
        self.imputation_cutoffs_numCols = temp_cutoff_dict
        self.X_train_imp = X_train_imp
        print(f"cols outlier transformed: {cntr} out of {len(self.numeric_features_upd)} successfully")

    
    def transformOutl(self):
        cntr=0
        X_test_imp = pd.DataFrame()
        for col in self.numeric_features_upd:
            # if cntr > 2:
            #     break
            col_ser = self.X_test[col].copy()
            cut_off = self.imputation_cutoffs_numCols[col]
            remove_zeros = self.remove_zeros_numCols[col]
            imputed_col = PreProcess.imputeOutl(col_ser, cut_off, remove_zeros, strategy='clip')
            X_test_imp = pd.concat([X_test_imp, imputed_col], axis=1)
            cntr+=1
        assert(X_test_imp.shape[1] == len(self.numeric_features_upd))
        self.X_test_imp = X_test_imp
        print(f"cols outlier transformed: {cntr} out of {len(self.numeric_features_upd)} successfully")


    def cleanedUpOutl(self, col:str):
        QUANTILE_LIST_LOWER = np.arange(0, 0.1, 0.01)
        QUANTILE_LIST_UPPER = np.arange(1, 0.9, -0.01)
        
        col_ser = self.X_train[col].copy()
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


    ## ********* Encoding Catg Features ********* ##


    ## ********* Scaling DF ********* ##


    ## ********* Feature reduction like PCA or t-SNE ********* ##


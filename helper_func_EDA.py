## Helper functions that aid to do EDA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class EDA:
    '''
    This class enables us to do the EDA 
    of the data before we preprocess it.
    '''
    NA_ALLOWED = 0.5  # prop of NAs allowed
    DEV_THRESH = 0.1  # threshold of dev in target caused by a single feature
    MAX_QNTL = 0.95  # max quantile value considered rather than the max .. to ignore outliers
    CORR_THRESH = 0.7  # a higher correlation between two independent features need to be avoided in modeling 

    def __init__(self, df, target_feature):
        self.df = df
        self.target_feature = target_feature


    def getColsWithManyNAs(self):
        '''
        cols that have more than the allowed pct of rows 
        will get returned
        '''
        # self.NA_ALLOWED = na_allowed
        thresh_na_count = int(EDA.NA_ALLOWED * self.df.shape[0])
        op_str = f"NA count is at least {thresh_na_count} rows out of {self.df.shape[0]} | {thresh_na_count/ self.df.shape[0]:.2%}"
        print(op_str)
        cols_w_manyNAs = self.df.isna().sum()[self.df.isna().sum() > thresh_na_count].index.to_list()
        return cols_w_manyNAs


    @staticmethod
    def getRowColCountForChart(nbr, def_cols=4):
        '''
        get row and column count for subplots
        based on the number of columns in df
        '''
        MAX_COL = 6
        nrow = 1
        print(f"features to plot: {nbr}")
        if def_cols > MAX_COL:
            print(f"Defaulted to showing {MAX_COL} in one row")
            def_cols = MAX_COL
        if nbr > (def_cols**2):
            print(f"First {def_cols**2} cols selected")
            nbr = def_cols**2
            return def_cols, def_cols
        else:
            ncol = def_cols
            for i in range(1, nbr):
                if i%def_cols == 0:
                    nrow += 1
                        
        return nrow, ncol


    def plotCatgPlot(self, cols, plot_type=sns.boxplot, def_cols=4, figsize2=(24,12)):
        '''
        get the plot for each of the catg 
        within the columns
        '''
        nrow, ncol = EDA.getRowColCountForChart(len(cols), def_cols)
        print(nrow, ncol)
        fig, ax = plt.subplots(nrow, ncol, figsize=figsize2, sharey=True)

        if nrow<2:
            for i, col in enumerate(cols):
                plot_type(x=col, y=self.target_feature, data=self.df, ax=ax[i])

            plt.show()
        
        else:
            cnt=0
            for i in range(0, nrow):
                for j in range(0, ncol):  
                    col = cols[cnt]
                    sns.barplot(x=col, y=self.target_feature, data=self.df, ax=ax[i, j])
                    cnt += 1
            plt.show()


    def explainedDeviationByCols(self, cols):
        '''
        rank the cols based on the 
        deviation in target feature they explain
        '''
        dev=[]
        for i, col in enumerate(cols):
            temp_df = self.df[~self.df[col].isna()]
            temp_gr_df = temp_df.groupby([col], as_index=False)[self.target_feature].median()
            dev.insert(i, temp_gr_df[self.target_feature].std())

        metrics_df = pd.DataFrame({
            'features': cols,
            'dev': dev
        })
        metrics_df.shape
        metrics_df.sort_values(['dev'], ascending=False, inplace=True)
        metrics_df.reset_index(drop=True, inplace=True)
        
        max_dev = metrics_df['dev'].quantile(EDA.MAX_QNTL)
        thresh_l = EDA.DEV_THRESH * max_dev
        cols_w_low_dev = metrics_df[metrics_df['dev'] < thresh_l]['features'].to_list()
        
        return metrics_df, cols_w_low_dev


    def getRankedCorr(self, cols):
        '''
        get ranked correlation between 
        each pair of feature
        '''
        # self.CORR_THRESH = corr_thresh
        pair_key = []
        corr_val = []
        cnt=0
        for i, col_i in enumerate(cols):
            for j, col_j in enumerate(cols):
                if i == j:
                    continue
                temp_pair = col_i + '__' + col_j
                temp_pair_rev = col_j + '__' + col_i
                if len([f for f in pair_key if temp_pair_rev==f])>0:  # avoid repeats
                    continue
                pair_key.insert(cnt, temp_pair)
                corr_val.insert(cnt, self.df[col_i].corr(self.df[col_j]))
                cnt += 1

        corr_df = pd.DataFrame({
            'feature_pair': pair_key,
            'corr_val': corr_val
        })
        corr_df['abs_corr_val'] = abs(corr_df['corr_val'])
        corr_df.sort_values(['abs_corr_val'], ascending=False, inplace=True)
        
        h_corr_pair = corr_df[corr_df['abs_corr_val'] > EDA.CORR_THRESH]['feature_pair'].to_list()
        corr_df.drop(['abs_corr_val'], axis=1, inplace=True)
        corr_df.reset_index(drop=True, inplace=True)
        
        return corr_df, h_corr_pair


    def getMultiCollFeatr(self, cols):
        '''
        input highly correlated features
        return features that need to be dropped 
        based on how are these features correlated 
        with target feature.
        '''
        self.corr_df, h_corr_pair = self.getRankedCorr(cols)

        h_corr_pair_dict = {i:x.split("__") for i, x in enumerate(h_corr_pair)}
        multi_coll_featr = []

        for k, v in h_corr_pair_dict.items():
            temp_dict = {v_one: self.df[self.target_feature].corr(self.df[v_one]) for v_one in v}
            print(temp_dict)
            multi_coll_featr.insert(len(multi_coll_featr), [k1 for k1 in temp_dict.keys() if temp_dict[k1]!=max(temp_dict.values())][0])

        self.multi_coll_featr = multi_coll_featr
        return self.multi_coll_featr


    def getCols_w_lowDev(self, feature_lists):
        cols_w_low_dev = []
        for feature_list in feature_lists:
            _, temp_cols = self.explainedDeviationByCols(feature_list)
            cols_w_low_dev = cols_w_low_dev + temp_cols
        print(f"number of cols with low dev: {len(cols_w_low_dev)}")
        return cols_w_low_dev
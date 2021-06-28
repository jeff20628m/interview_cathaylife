import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn import mixture


class preprocessing_training:
    def __init__(self, df_train):

        self.df = df_train

    def gen_label_encoder(self, target: str) -> object:
        """
        input -> dataframe & the column which you want to label.
        output -> label_encoder (object)
        """
        self.la = LabelEncoder()
        self.la.fit(self.df[target])

        self.mapping = dict(
            zip(self.la.classes_, self.la.transform(self.la.classes_)))

        return self.la, self.mapping

    def label_encoding_feature(self, targets: list):

        mapping_list = []

        for target in targets:
            label_encoding, mapping = self.gen_label_encoder(target)
            self.df[target] = label_encoding.fit_transform(self.df[target])
            mapping_list.append(mapping)

        return self.df, mapping_list

    def preprocessing(self) -> pd.DataFrame:

        # add features

        # add age groups
        self.df['Age_group'] = pd.cut(self.df['Age'], bins=[0, 30, 50, 60, 90])
        self.df['Age_group_codes'] = self.df['Age_group'].cat.codes

        # label categories features
        label_target = ['Gender', 'Vehicle_Damage', 'Vehicle_Age']

        self.df, mapping_list = self.label_encoding_feature(label_target)

        # add special_region
        self.df['special_region'] = 0
        self.special_num = list(self.df[self.df['Response'] == 1]
                                ['Region_Code'].value_counts().index[:10])
        for region in self.special_num:
            special = self.df[self.df['Region_Code'] == region].index
            self.df.loc[special, 'special_region'] = 1

        # add special_channel
        self.df['special_Policy_Sales_Channel'] = 0
        self.special_channel = list(
            self.df[self.df['Response'] ==
                    1]['Policy_Sales_Channel'].value_counts().index[:10])
        for channel in self.special_channel:
            channel = self.df[self.df['Policy_Sales_Channel'] == channel].index
            self.df.loc[channel, 'special_Policy_Sales_Channel'] = 1

        self.df['Vehicle_Damage_region'] = self.df['Vehicle_Damage'] + self.df[
            'special_region']
        self.df['age_channel'] = self.df['Age_group_codes'] * self.df[
            'special_Policy_Sales_Channel']

        # add interesting_score
        self.df['interesting_score'] = self.df['Vehicle_Damage'] - \
            self.df['Previously_Insured'] + 0.25*self.df['Driving_License'] + 0.25*self.df['Gender'] - \
            self.df['Vehicle_Age'] + self.df['special_region'] + \
            self.df['special_Policy_Sales_Channel']
        # add annual_std
        self.df['Annual_log_std'] = np.log1p(self.df['Annual_Premium'])

        # add clustering data

        # add diff cluster
        minmax = MinMaxScaler()
        different_cluster = minmax.fit_transform(
            np.array(self.df[[
                'age_channel', 'Age_group_codes', 'Previously_Insured',
                'Vehicle_Damage', 'interesting_score', 'Vehicle_Damage_region'
            ]]))

        GMcluster = mixture.GaussianMixture(n_components=3,
                                            covariance_type='full',
                                            random_state=26)
        GMcluster_diff = GMcluster.fit(different_cluster)
        GMlabels_diff = GMcluster_diff.predict(different_cluster)
        self.df['different_cluster'] = GMlabels_diff

        # add same cluster
        minmax1 = MinMaxScaler()
        samiliar_cluster = minmax1.fit_transform(
            np.array(self.df[[
                'Annual_log_std', 'Gender', 'Vintage', 'Vehicle_Age',
                'Policy_Sales_Channel', 'Region_Code'
            ]]))
        GMcluster = mixture.GaussianMixture(n_components=3,
                                            covariance_type='full',
                                            random_state=26)
        GMcluster_same = GMcluster.fit(samiliar_cluster)
        GMlabels_same = GMcluster_same.predict(samiliar_cluster)
        self.df['same_cluster'] = GMlabels_same

        # add all cluster
        minmax2 = MinMaxScaler()
        all_columns = [
            'Gender', 'Age', 'Driving_License', 'Region_Code',
            'Previously_Insured', 'Vehicle_Age', 'Annual_Premium',
            'Policy_Sales_Channel', 'Vintage'
        ]
        all_cluster = minmax2.fit_transform(np.array(self.df[all_columns]))
        GMcluster = mixture.GaussianMixture(n_components=4,
                                            covariance_type='full',
                                            random_state=26)
        GMcluster_all = GMcluster.fit(all_cluster)
        GMlabels_all = GMcluster_all.predict(all_cluster)
        self.df['all_cluster'] = GMlabels_all

        # drop object and category data
        for col in self.df.columns:
            if (str(self.df[col].dtype) == 'object') | (str(
                    self.df[col].dtype) == 'category'):
                print(f'Drop {col} type {self.df[col].dtype}')
                self.df = self.df.drop(col, axis=1)

        mapping_list.append(self.special_num)
        mapping_list.append(self.special_channel)

        return self.df, mapping_list, GMcluster_diff, GMcluster_same, GMcluster_all


class preprocessing_testing():
    def __init__(self, df_test, mapping_list, GMcluster_diff, GMcluster_same,
                 GMcluster_all):
        """
        mapping_list label list = 'Gender', 'Vehicle_Damage','Vehicle_Age','special_num','special_channel'
        """
        self.df = df_test
        self.mapping_list = mapping_list
        self.GMcluster_diff = GMcluster_diff
        self.GMcluster_same = GMcluster_same
        self.GMcluster_all = GMcluster_all

    def preprocessing(self):

        # add features
        # label categories features

        self.df['Gender'] = self.df['Gender'].apply(
            lambda x: self.mapping_list[0][x])
        self.df['Vehicle_Damage'] = self.df['Vehicle_Damage'].apply(
            lambda x: self.mapping_list[1][x])
        self.df['Vehicle_Age'] = self.df['Vehicle_Age'].apply(
            lambda x: self.mapping_list[2][x])

        # add age groups
        self.df['Age_group'] = pd.cut(self.df['Age'], bins=[0, 30, 50, 60, 90])
        self.df['Age_group_codes'] = self.df['Age_group'].cat.codes

        # add special_region
        self.df['special_region'] = 0
        for region in self.mapping_list[3]:
            special = self.df[self.df['Region_Code'] == region].index
            self.df.loc[special, 'special_region'] = 1

        self.df['special_Policy_Sales_Channel'] = 0
        for channel in self.mapping_list[4]:
            channel = self.df[self.df['Policy_Sales_Channel'] == channel].index
            self.df.loc[channel, 'special_Policy_Sales_Channel'] = 1

        self.df['Vehicle_Damage_region'] = self.df['Vehicle_Damage'] + self.df[
            'special_region']
        self.df['age_channel'] = self.df['Age_group_codes'] * self.df[
            'special_Policy_Sales_Channel']

        # add interesting_score
        self.df['interesting_score'] = self.df['Vehicle_Damage'] - \
            self.df['Previously_Insured'] + 0.25*self.df['Driving_License'] + 0.25*self.df['Gender'] - \
            self.df['Vehicle_Age'] + self.df['special_region'] + \
            self.df['special_Policy_Sales_Channel']
        # add annual_std
        self.df['Annual_log_std'] = np.log1p(self.df['Annual_Premium'])

        # add clustering data

        # add diff cluster
        minmax = MinMaxScaler()
        different_cluster = minmax.fit_transform(
            np.array(self.df[[
                'age_channel', 'Age_group_codes', 'Previously_Insured',
                'Vehicle_Damage', 'interesting_score', 'Vehicle_Damage_region'
            ]]))

        GMlabels_diff = self.GMcluster_diff.predict(different_cluster)
        self.df['different_cluster'] = GMlabels_diff

        # add same cluster
        minmax1 = MinMaxScaler()
        samiliar_cluster = minmax1.fit_transform(
            np.array(self.df[[
                'Annual_log_std', 'Gender', 'Vintage', 'Vehicle_Age',
                'Policy_Sales_Channel', 'Region_Code'
            ]]))

        GMlabels_same = self.GMcluster_same.predict(samiliar_cluster)
        self.df['same_cluster'] = GMlabels_same

        # add all cluster
        minmax2 = MinMaxScaler()
        all_columns = [
            'Gender', 'Age', 'Driving_License', 'Region_Code',
            'Previously_Insured', 'Vehicle_Age', 'Annual_Premium',
            'Policy_Sales_Channel', 'Vintage'
        ]
        all_cluster = minmax2.fit_transform(np.array(self.df[all_columns]))

        GMlabels_all = self.GMcluster_all.predict(all_cluster)
        self.df['all_cluster'] = GMlabels_all

        # delete the category and object columns
        for col in self.df.columns:
            if (str(self.df[col].dtype) == 'object') | (str(
                    self.df[col].dtype) == 'category'):
                print(f'Drop {col} type {self.df[col].dtype}')
                self.df = self.df.drop(col, axis=1)

        return self.df

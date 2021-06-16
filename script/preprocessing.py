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

    def preprocessing(self) -> pd.DataFrame:

        # add features
        # label categories features

        self.la_gender, self.gender_mapping = self.gen_label_encoder('Gender')
        self.la_vehicle_damage, self.vehicle_damage = self.gen_label_encoder(
            'Vehicle_Damage')
        self.df['Gender'] = self.la_gender.transform(self.df['Gender'])
        self.df['Vehicle_Damage'] = self.la_vehicle_damage.transform(
            self.df['Vehicle_Damage'])
        self.df = pd.concat(
            [self.df, pd.get_dummies(self.df['Vehicle_Age'])], axis=1)

        # add age groups
        self.df['Age_group'] = pd.cut(self.df['Age'], bins=range(10, 90, 10))
        self.df['Age_group_codes'] = self.df['Age_group'].cat.codes

        # add interesting_score
        self.df['interesting_score'] = self.df['Vehicle_Damage'] - \
            self.df['Previously_Insured']

        # add special_region
        self.df['special_region'] = 0
        self.special_num = list(
            self.df[self.df['Response'] == 1]['Region_Code'].value_counts().index[:15])
        for region in self.special_num:
            special = self.df[self.df['Region_Code'] == region].index
            self.df.loc[special, 'special_region'] = 1

        # add age * channel
        self.df['age_channel'] = self.df['Age_group_codes'].astype(
            'str') + self.df['Policy_Sales_Channel'].astype('str')
        self.age_channel_la, self.age_channel_mapping = self.gen_label_encoder(
            'age_channel')
        self.df['age_channel'] = self.age_channel_la.fit_transform(
            self.df['age_channel'])

        # add low & high pay
        self.df['lowest_pay'] = 0
        low_idx = self.df[self.df['Annual_Premium']
                          == self.df['Annual_Premium'].min()].index
        self.df.loc[low_idx, 'lowest_pay'] = 1

        self.df['high_pay'] = 0
        high_idx = self.df[self.df['Annual_Premium'] >= (
            self.df['Annual_Premium'].mean() + 1.5*self.df['Annual_Premium'].std())].index
        self.df.loc[high_idx, 'high_pay'] = 1

        # add annual_std
        self.df['Annual_log_std'] = np.log1p(self.df['Annual_Premium'])

        minmax = MinMaxScaler()
        different_cluster = minmax.fit_transform(np.array(
            self.df[['Policy_Sales_Channel', 'Region_Code', 'Age', 'Previously_Insured', 'Vehicle_Damage']]))
        minmax1 = MinMaxScaler()
        samiliar_cluster = minmax1.fit_transform(np.array(
            self.df[['Annual_Premium', 'Gender', 'Vintage', '1-2 Year', '< 1 Year', '> 2 Years']]))

        GMcluster = mixture.GaussianMixture(
            n_components=2, covariance_type='full', random_state=0)
        GMcluster_fit = GMcluster.fit(different_cluster)
        GMlabels_diff = GMcluster_fit.predict(different_cluster)
        self.df['different_cluster'] = GMlabels_diff

        GMcluster = mixture.GaussianMixture(
            n_components=2, covariance_type='full', random_state=0)
        GMcluster_fit = GMcluster.fit(samiliar_cluster)
        GMlabels_same = GMcluster_fit.predict(samiliar_cluster)
        self.df['same_cluster'] = GMlabels_same

        # drop object and category data
        for col in self.df.columns:
            if (str(self.df[col].dtype) == 'object') | (str(self.df[col].dtype) == 'category'):
                print(f'Drop {col} type {self.df[col].dtype}')
                self.df = self.df.drop(col, axis=1)

        mapping_list = [self.gender_mapping,
                        self.age_channel_mapping, self.vehicle_damage, self.special_num]

        return self.df, mapping_list


class preprocessing_testing():

    def __init__(self, df_test, map_gender, map_age_channel, map_vehicle_damage, special_num):

        self.df = df_test
        self.map_gender = map_gender
        self.map_age_channel = map_age_channel
        self.map_vehicle_damage = map_vehicle_damage
        self.special_num = special_num

    def preprocessing(self):

        # add features
        # label categories features

        self.df['Gender'] = self.df['Gender'].apply(
            lambda x: self.map_gender[x])
        self.df['Vehicle_Damage'] = self.df['Vehicle_Damage'].apply(
            lambda x: self.map_vehicle_damage[x])

        self.df = pd.concat(
            [self.df, pd.get_dummies(self.df['Vehicle_Age'])], axis=1)

        # add age groups
        self.df['Age_group'] = pd.cut(self.df['Age'], bins=range(10, 90, 10))
        self.df['Age_group_codes'] = self.df['Age_group'].cat.codes

        # add interesting_score
        self.df['interesting_score'] = self.df['Vehicle_Damage'] - \
            self.df['Previously_Insured']

        # add special_region
        self.df['special_region'] = 0
        for region in self.special_num:
            special = self.df[self.df['Region_Code'] == region].index
            self.df.loc[special, 'special_region'] = 1

        # add age * channel
        self.df['age_channel'] = self.df['Age_group_codes'].astype(
            'str') + self.df['Policy_Sales_Channel'].astype('str')
        try:
            self.df['age_channel'] = self.df['age_channel'].apply(
                lambda x: self.map_age_channel[x])
        except KeyError:
            print("There's non label data!")
        # it means there's none label data in the mapping list
        # so we deside to add new label for i and show the problem
            for key_value in self.df['age_channel']:
                if key_value not in self.map_age_channel:
                    # add new label as the max length+1
                    self.map_age_channel[key_value] = len(
                        self.map_age_channel) + 1
                else:
                    pass
            self.df['age_channel'] = self.df['age_channel'].apply(
                lambda x: self.map_age_channel[x])

        # add low & high pay
        self.df['lowest_pay'] = 0
        low_idx = self.df[self.df['Annual_Premium']
                          == self.df['Annual_Premium'].min()].index
        self.df.loc[low_idx, 'lowest_pay'] = 1

        self.df['high_pay'] = 0
        high_idx = self.df[self.df['Annual_Premium'] >= (
            self.df['Annual_Premium'].mean() + 1.5*self.df['Annual_Premium'].std())].index
        self.df.loc[high_idx, 'high_pay'] = 1

        # add annual_std
        self.df['Annual_log_std'] = np.log1p(self.df['Annual_Premium'])

        minmax = MinMaxScaler()
        different_cluster = minmax.fit_transform(np.array(
            self.df[['Policy_Sales_Channel', 'Region_Code', 'Age', 'Previously_Insured', 'Vehicle_Damage']]))
        minmax1 = MinMaxScaler()
        samiliar_cluster = minmax1.fit_transform(np.array(
            self.df[['Annual_Premium', 'Gender', 'Vintage', '1-2 Year', '< 1 Year', '> 2 Years']]))

        GMcluster = mixture.GaussianMixture(
            n_components=2, covariance_type='full', random_state=0)
        GMcluster_fit = GMcluster.fit(different_cluster)
        GMlabels_diff = GMcluster_fit.predict(different_cluster)
        self.df['different_cluster'] = GMlabels_diff

        GMcluster = mixture.GaussianMixture(
            n_components=2, covariance_type='full', random_state=0)
        GMcluster_fit = GMcluster.fit(samiliar_cluster)
        GMlabels_same = GMcluster_fit.predict(samiliar_cluster)
        self.df['same_cluster'] = GMlabels_same

        for col in self.df.columns:
            if (str(self.df[col].dtype) == 'object') | (str(self.df[col].dtype) == 'category'):
                print(f'Drop {col} type {self.df[col].dtype}')
                self.df = self.df.drop(col, axis=1)

        return self.df

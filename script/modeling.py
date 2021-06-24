import lightgbm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, precision_score, recall_score, classification_report
from preprocessing import preprocessing_training, preprocessing_testing
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")


class modeling():
    def __init__(self, train, test, model_select=None):

        self.train = train
        self.test = test
        if model_select:
            self.model = model_select
        else:
            print('Auto select best performance model')

    def data_preprocessing(self):

        pre_train = preprocessing_training(self.train)

        self.df, map_list = pre_train.preprocessing()

        pre_test = preprocessing_testing(self.test, map_list)

        self.df_test = pre_test.preprocessing()

        self.df_test = self.df_test.drop(
            ['id', 'Age_group_codes', 'special_Policy_Sales_Channel'], axis=1)

        self.df = self.df.drop(
            ['id', 'Age_group_codes', 'special_Policy_Sales_Channel'], axis=1)

        return self.df, self.df_test

    def model_eval(self, model, X_train, X_test, y_train, y_test):

        # train datset
        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)[::, 1]
        print('Train dataset :')
        print(classification_report(y_train, y_train_pred))
        print('Confusion matrix :\n',
              confusion_matrix(y_train, y_train_pred, labels=[0, 1]))
        print('Accuracy :', accuracy_score(y_train, y_train_pred))
        print('AUC score :', roc_auc_score(y_train, y_train_proba))
        print('F1-score :', f1_score(y_train, y_train_pred))
        print('Precision score :', precision_score(y_train, y_train_pred))
        print('Recall score :', recall_score(y_train, y_train_pred))

        # test datset
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[::, 1]
        print('\n\nTest dataset :')
        print(classification_report(y_test, y_test_pred))
        print('Confusion matrix :\n',
              confusion_matrix(y_test, y_test_pred, labels=[0, 1]))
        print('Accuracy :', accuracy_score(y_test, y_test_pred))
        print('AUC score :', roc_auc_score(y_test, y_test_proba))
        print('F1-score :', f1_score(y_test, y_test_pred))
        print('Precision score :', precision_score(y_test, y_test_pred))
        print('Recall score :', recall_score(y_test, y_test_pred))

    def model_training(self):

        self.df, self.df_test = self.data_preprocessing()
        y_test = self.df_test['Response']
        df_test = self.df_test.drop('Response', axis=1)
        Target = self.df['Response']
        df_final_ = self.df.drop('Response', axis=1)

        # X_train, x_val, Y_train, y_val = train_test_split(df_final_,
        #                                                   Target,
        #                                                   test_size=0.1,
        #                                                   random_state=26)

        lgb = lightgbm.LGBMClassifier(
            n_estimators=2000,
            depth=12,
            learning_rate=0.03,
            metric='auc',
            is_unbalance=True,
            subsample=0.6,
            colsample_bytree=0.6,
            reg_lambda=2,
            reg_alpha=2,
            subsample_freq=13,
            random_state=26,
            n_jobs=-1)  # The parameter here are selected by manual tuning

        fold = StratifiedKFold(n_splits=5, shuffle=True)
        for train_index, test_index in fold.split(df_final_, Target):
            Xf_train, Xf_test = df_final_.iloc[train_index], df_final_.iloc[
                test_index]
            Yf_train, Yf_test = Target.iloc[train_index], Target.iloc[
                test_index]

            lgb = lgb.fit(Xf_train,
                          Yf_train,
                          eval_metric='auc',
                          eval_set=[(Xf_train, Yf_train), (Xf_test, Yf_test),
                                    (df_test, y_test)],
                          verbose=200,
                          early_stopping_rounds=200)

        self.model_eval(lgb, df_final_, df_test, Target, y_test)

        return lgb

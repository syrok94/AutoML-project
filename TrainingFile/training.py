import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing_file.preprocess import Preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from HyperParameterTuning.HyperParameterTuning import HyperParameterTuning
from sklearn.metrics import plot_confusion_matrix,plot_precision_recall_curve,plot_roc_curve
import os
from application_logging.logger import app_logger
class Training:
    def __init__(self,path):
        self.path=path
        self.cwd=os.getcwd()
        self.file_object=open("trainingLOG/train_log.txt","a+")
        self.app_logger=app_logger()
    def train_model(self):
        #self.path=path
        self.app_logger.log(self.file_object,"training started!!")
        preprocess_data = Preprocessing(self.path)

        self.app_logger.log(self.file_object, "reading csv data")
        data = preprocess_data.loading_data_to_csv(self.path)

        self.app_logger.log(self.file_object, "checking for null values")
        null_value = preprocess_data.null_value_count(data)

        self.app_logger.log(self.file_object, "imputing null values")
        data=preprocess_data.null_value_imputer(data)

        self.app_logger.log(self.file_object, "seperating feature and target data column")
        X, Y = preprocess_data.seperate_feature_and_label(data,"Is Diabetic")

        self.app_logger.log(self.file_object, "applying over sampling of data")
        ros_x,ros_y=preprocess_data.oversampling_of_data(X,Y)

        self.app_logger.log(self.file_object, "removing columns with zero variance")
        var = preprocess_data.checking_variance_of_columns(ros_x)

        self.app_logger.log(self.file_object, "appling train test split function on whole data set")
        X_train, X_val, Y_train, Y_val = preprocess_data.splitting_data_to_train_and_validation_set(ros_x, ros_y)

        self.app_logger.log(self.file_object, "again applying train test split on 50% of data")
        x_train, x_test, y_train, y_test = preprocess_data.splitting_train_data_to_train_and_test_set(X_train, Y_train)

        self.app_logger.log(self.file_object, "Hyperparameter tunning started!!")
        model_Hyperparametertunning = HyperParameterTuning()

        self.app_logger.log(self.file_object, "getting best parameter for naive bayes model")
        best_param_naive = model_Hyperparametertunning.best_param_for_naive_bayes_clf(x_train, y_train)

        self.app_logger.log(self.file_object, "getting best parameter for decision tree model")
        best_param_dt_clf = model_Hyperparametertunning.best_param_for_decision_tree_clf(x_train, y_train)

        #best_param_rf_clf = model_Hyperparametertunning.best_param_for_rf_clf_model(x_train, y_train)
        self.app_logger.log(self.file_object, "fitting best parameter for naive bayes model")
        naive_clf = GaussianNB(var_smoothing=best_param_naive['var_smoothing'])
        naive_clf.fit(x_train, y_train)
        naive_predict = naive_clf.predict(x_test)


        self.app_logger.log(self.file_object, "fitting best parameter for decision tree model")
        dt_clf = DecisionTreeClassifier(criterion=best_param_dt_clf['criterion'],
                                        max_depth=best_param_dt_clf['max_depth'],
                                        max_features=best_param_dt_clf['max_features'],
                                        min_samples_leaf=best_param_dt_clf['min_samples_leaf'],
                                        splitter=best_param_dt_clf['splitter'])
        dt_clf.fit(x_train, y_train)
        dt_predict = dt_clf.predict(x_test)

        self.app_logger.log(self.file_object, "stacking best parameter of decision tree and naive model")
        meta_input = np.stack((naive_predict, dt_predict), axis=-1)

        best_param_rf_clf = model_Hyperparametertunning.best_param_for_rf_clf_model(meta_input, y_test)

        self.app_logger.log(self.file_object, "fitting best paramrter to stacked model")
        rf_clf = RandomForestClassifier(n_estimators=best_param_rf_clf["n_estimators"],
                                        max_features=best_param_rf_clf["max_features"],
                                        max_depth=best_param_rf_clf["max_depth"],
                                        criterion=best_param_rf_clf["criterion"],
                                        )
        rf_clf.fit(meta_input,y_test)
        rf_predict = rf_clf.predict(meta_input)
        #print(rf_predict)


        plt.figure(figsize=(5,5))
        plot_precision_recall_curve(rf_clf,meta_input,y_test)
        plt.savefig("precision_recall_curve")
        #plt.title("precision_recall_curve")


        plt.figure(figsize=(5, 5))
        plot_roc_curve(rf_clf, meta_input, y_test)
        plt.savefig("plot_roc_curve")
        #plt.title("plot_roc_curve")


        plt.figure(figsize=(5, 5))
        plot_confusion_matrix(rf_clf, meta_input, y_test)
        plt.savefig("plot_confusion_matrix")
        #plt.title("plot_confusion_matrix")










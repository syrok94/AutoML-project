from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from application_logging.logger import app_logger




class HyperParameterTuning:

    def __init__(self):
        self.naive_model=GaussianNB()
        self.dt_model=DecisionTreeClassifier()
        self.rf_model=RandomForestClassifier()
        self.xg_model=XGBClassifier()
        self.file_object=open("trainingLOG/hyperparameter_tunning_log.txt",'a+')
        self.app_logger=app_logger()




    def best_param_for_naive_bayes_clf(self,x_train,y_train):

        try:
            self.app_logger.log(self.file_object,"hyperparamter tunning of naive model started!")
            self.naive_param = {"var_smoothing": [1e-02, 1e-07, 1e-06, 0.1, 0.01]}
            self.randcv_naive=RandomizedSearchCV(estimator=self.naive_model,param_distributions=self.naive_param
                                                 ,cv=5,n_jobs=-1)
            self.app_logger.log(self.file_object, "hyperparamter tunning of naive model completed!")
            self.app_logger.log(self.file_object, "fitting tunned parameter to tunned model to get best paramrter")

            self.randcv_naive.fit(x_train,y_train)
            self.app_logger.log(self.file_object, "training of tunned model completed!")

            self.naive_model_best_parameter=self.randcv_naive.best_params_
            self.app_logger.log(self.file_object, "best parameter for naive model is stored!")

            return self.naive_model_best_parameter
        except Exception as e:
            raise e

    def best_param_for_decision_tree_clf(self, x_train, y_train):

        try:
            self.app_logger.log(self.file_object, "hyperparamter tunning of decision tree model started!")
            self.dt_clf_param = {"criterion": ['gini', 'entropy'],
                        "splitter": ['best', 'random'],
                        "max_depth": [50, 100, 200, 300, 400, 500],
                        "min_samples_leaf": [1, 2, 3, 4, 5],
                        "max_features": ['auto', 'log2', 'sqrt']
                        }

            self.randcv_dt_clf = RandomizedSearchCV(estimator=self.dt_model, param_distributions=self.dt_clf_param,
                                                   cv=5, n_jobs=-1)
            self.app_logger.log(self.file_object, "hyperparamter tunning of decision tree model completed!")

            self.app_logger.log(self.file_object, "fitting tuned parameters to tunned model")
            self.randcv_dt_clf.fit(x_train, y_train)
            self.app_logger.log(self.file_object, "training of tunned model completed!")
            self.dt_clf_model_best_parameter = self.randcv_dt_clf.best_params_
            self.app_logger.log(self.file_object, "best parameter for decision tree model stored!")
            return self.dt_clf_model_best_parameter
        except Exception as e:
            raise e

    def best_param_for_rf_clf_model(self, x_train, y_train):

        try:
            self.app_logger.log(self.file_object, "hyperparamter tunning of random forest model started!")
            self.rf_param={"n_estimators":[10,20,30,40,50,60,70,80,90,100],
                                "criterion":['gini','entropy'],
                                "max_depth":[50,100,200,300],
                                "max_features":['auto','log2','sqrt']}


            self.randcv_rf_clf = RandomizedSearchCV(estimator=self.rf_model, param_distributions=self.rf_param,
                                                   cv=5, n_jobs=-1)
            self.app_logger.log(self.file_object, "hyperparameter tunning of random fores model completd!")
            self.app_logger.log(self.file_object, "fitting tunned parameter to tuned model")
            self.randcv_rf_clf.fit(x_train, y_train)
            self.rf_clf_model_best_parameter = self.randcv_rf_clf.best_params_
            self.app_logger.log(self.file_object, "best parameter for random forest model stored!")

            return self.rf_clf_model_best_parameter
        except Exception as e:
            raise e

    '''def best_param_for_xgboost_model(self, x_train, y_train):

        try:
            self.xg_param = {"n_estimators": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                "learning_rate": [0.1, 0.01, 0.5, 1]}

            self.randcv_xgboost_clf = RandomizedSearchCV(estimator=self.xg_model, param_distributions=self.xg_param,
                                                    cv=5, n_jobs=-1)

            self.randcv_xgboost_clf.fit(x_train, y_train)
            self.xgboost_clf_model_best_parameter = self.randcv_xgboost_clf.best_params_

            return self.xgboost_clf_model_best_parameter
        except Exception as e:
            raise e'''










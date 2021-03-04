import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from imblearn.over_sampling import RandomOverSampler
from sklearn.impute import KNNImputer

class Preprocessing:

    def __init__(self,path):
        self.path=path
        self.target_name="Is Diabetic"
        #self.ros=RandomOverSampler(random_state=42)



    def loading_data_to_csv(self,path):
        try:
            self.data = pd.read_csv(self.path)

            return self.data
        except Exception as e:
            raise e


    def null_value_count(self,data):
        try:
            #checking for null value
            self.data=data
            self.is_null=self.data.isnull().sum()
            self.null_dataframe=pd.DataFrame(self.is_null,columns=["null_count"])

            #taking sum of null values in csv file
            self.null_dataframe.to_csv("null_count.csv")

            return  self.is_null


        except Exception as e:
            raise e

    def null_value_imputer(self,data):
        try:

            self.data=data
            self.knn=KNNImputer()
            self.knn.fit_transform(data)

            return  self.data
        except Exception as e:
            raise e


    def seperate_feature_and_label(self,data,target_name):
        try:
            self.data=data
            self.target_name=target_name
            #splitting data to feature and column
            self.X=self.data.drop(self.target_name,axis=1)
            self.Y=self.data.get(self.target_name)

            return self.X,self.Y
        except Exception as e:
            raise e

    def oversampling_of_data(self,X,Y):
        try:
            self.X=X
            self.Y=Y
            self.ros=RandomOverSampler(random_state=42)
            self.ros_X,self.ros_Y=self.ros.fit_resample(X,Y)
            plt.figure(figsize=(5,5))
            sns.countplot(self.ros_Y)
            plt.savefig("over_sampling_output.png")

            return self.ros_X,self.ros_Y

        except Exception as e:
            raise e


    def checking_variance_of_columns(self,feature_columns):
        try:
            self.feature_columns=feature_columns
            variance=VarianceThreshold(threshold=0)
            variance.fit(self.feature_columns)

            self.variances_of_columns=variance.variances_
            self.columns_with_zero_variance=variance.get_support()

            return self.columns_with_zero_variance

        except Exception as e:
            raise e


    def splitting_data_to_train_and_validation_set(self,feature_column,target_column):
        try:
            self.feature_column=feature_column
            self.target_column=target_column
            # splitting data to train and validation set
            self.X_train,self.X_val,self.Y_train,self.Y_val=train_test_split(self.feature_column,self.target_column,test_size=0.5,random_state=42)

            return  self.X_train,self.X_val,self.Y_train,self.Y_val
        except Exception as e:
            raise e




    def splitting_train_data_to_train_and_test_set(self,X_train,Y_train):
        try:
            self.X_train=X_train
            self.Y_train=Y_train
            #splitting data to train and test set
            self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.X_train,self.Y_train,test_size=0.2,random_state=42)

            return self.x_train,self.x_test,self.y_train,self.y_test

        except Exception as e:
            raise e









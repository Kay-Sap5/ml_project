import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from src.logger import logging
from src.exception import CustomException

from src.utils import evaluate_model,save_object


@dataclass
class ModelTrainerConfig:
    model_trainer_file_path = os.path.join("artifacts",'model.pkl')


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Split the train and test data")
            x_train,y_train,x_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1])
            
            models = {
                "LinearRegression":LinearRegression(),
                "XGBoostRegressor":XGBRegressor(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "RandomForestRegressor":RandomForestRegressor(),
                "AdaBoostRegressor":AdaBoostRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "CatBoostRegressor":CatBoostRegressor(verbose=False) }
            
            model_reports:dict = evaluate_model(x_train,y_train,x_test,y_test,models)

            best_model_score = max(sorted(model_reports.values()))

            best_model_name = list(model_reports.keys())[
                list(model_reports.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No Best Model Found")
            logging.info(f"Best model in both training and testing dataset")

            save_object(file_path=self.model_trainer_config.model_trainer_file_path,
                        obj = best_model)
            
            predicted = best_model.predict(x_test)
            r_squre_score = r2_score(y_test,predicted)

            return r_squre_score

        except Exception as e:
            raise CustomException(e,sys)
        

from xgboost import XGBClassifier
from src.data import load_dataset_extended,load_dataset_simple,split_data
from src.CONFIG import XGB_PARAMETERS
import joblib
import os

def train_model_extended():
    if os.path.exists("models/extended_model.pkl"):
        print("Model already exists.Loading it")
        model = joblib.load("models/extended_model.pkl")

        df = load_dataset_extended()
        X_train,X_test,y_train,y_test=split_data(df)
        return model, X_test, y_test
    
    df=load_dataset_extended()
    X_train,X_test,y_train,y_test=split_data(df)

    # print(X_train.shape, X_test.shape)
    model=XGBClassifier(**XGB_PARAMETERS) #unpack dictionary items into separate strings
    model.fit(X_train,y_train)
    joblib.dump(model,"models/extended_model.pkl")
    return model,X_test,y_test



def train_model_simple():
    if os.path.exists("models/simple_model.pkl"):
        print("Model already exists.Loading it")
        model = joblib.load("models/simple_model.pkl")

        df = load_dataset_simple()
        X_train,X_test,y_train,y_test=split_data(df)
        return model, X_test, y_test
    
    df=load_dataset_simple()
    X_train,X_test,y_train,y_test=split_data(df)

    # print(X_train.shape, X_test.shape)
    model=XGBClassifier(**XGB_PARAMETERS) #unpack dictionary items into separate strings
    model.fit(X_train,y_train)
    joblib.dump(model,"models/simple_model.pkl")
    return model,X_test,y_test
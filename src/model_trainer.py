from xgboost import XGBClassifier
from src.data import load_dataset,split_data
from CONFIG import PARAMETERS
import joblib

def train_model():
    df=load_dataset()
    X_train,X_test,y_train,y_test=split_data(df)
    model=XGBClassifier(PARAMETERS)
    model.fit(X_train,y_train)
    joblib.dump(model,"models/xgb_model.pkl")
    return model,X_test,y_test
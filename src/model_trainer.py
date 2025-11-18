from xgboost import XGBClassifier
from src.data import load_dataset,split_data
from src.CONFIG import XGB_PARAMETERS
import joblib

def train_model():
    df=load_dataset()
    X_train,X_test,y_train,y_test=split_data(df)

    # print(X_train.shape, X_test.shape)
    model=XGBClassifier(**XGB_PARAMETERS) #unpack dictionary items into separate strings
    model.fit(X_train,y_train)
    joblib.dump(model,"models/xgb_model.pkl")
    return model,X_test,y_test
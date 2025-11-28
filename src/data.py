import pandas as pd
from sklearn.model_selection import train_test_split
from src.CONFIG import DATA_PATH_EXTENDED,DATA_PATH_SIMPLE,DIAGNOSIS


def load_dataset_extended():
    df=pd.read_csv(DATA_PATH_EXTENDED)
    return df

def load_dataset_simple():
    df=pd.read_csv(DATA_PATH_SIMPLE)
    return df


def split_data(df):
    X=df.drop(columns=[DIAGNOSIS])
    y=df[DIAGNOSIS]
    # print(len(y),sum(y))
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=1)
    #we use stratify to make sure data is equally distributed since we are talking about a rare disease with low occurances therefore it may not appear enough in tests or trains
    return X_train,X_test,y_train,y_test
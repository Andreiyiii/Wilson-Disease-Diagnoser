DATA_PATH_EXTENDED="data/Wilson_disease_dataset.csv"
DATA_PATH_SIMPLE="data/Wilson_disease_dataset_simplified.csv"
DIAGNOSIS="Is_Wilson_Disease"
XGB_PARAMETERS={
    "max_depth":3,
    "subsample":0.8,
    "colsample_bytree":0.8,
    "learning_rate":0.05,
    "n_estimators":400,
    "random_state":1,
    "eval_metric":"logloss",
    "scale_pos_weight": 2,
}
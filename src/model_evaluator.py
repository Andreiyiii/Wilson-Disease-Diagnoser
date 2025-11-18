from sklearn.metrics import recall_score, roc_auc_score
def evaluate_model(model,X_test,y_test):
    prediction=model.predict(X_test)
    probability=model.predict_proba(X_test)[:,1] #chance to have the disease since it returns 2 columns,one for false and one for true
    print("Recall: ",recall_score(X_test,prediction))
    print("AUC: ",roc_auc_score(X_test,probability))
    #since we are talking about a rare disease detection ,using the accuracy_score function would
    #not work because the dataset is mostly consisting of healthy examples therefore it could just
    #predict everyone is healthy and still get a decent accuracy
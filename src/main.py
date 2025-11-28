import pandas as pd
import numpy as np
from src.model_trainer import train_model_extended,train_model_simple
from src.model_evaluator import evaluate_model
from src.shap_file import shap_plot


def main():
    model_extended,X_test,y_test=train_model_extended()
    evaluate_model(model_extended,X_test,y_test)
    shap_plot(model_extended,X_test)

    model_simple,X_test,y_test=train_model_simple()
    evaluate_model(model_simple,X_test,y_test)
    shap_plot(model_simple,X_test)

if __name__ == "__main__":
    main()

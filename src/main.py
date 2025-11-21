import pandas as pd
import numpy as np
from model_trainer import train_model
from model_evaluator import evaluate_model
from shap import explain


def main():
    model,X_test,y_test=train_model()
    evaluate_model(model,X_test,y_test)


if __name__ == "__main__":
    main()

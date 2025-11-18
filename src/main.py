import pandas as pd
import numpy as np
from src.model_trainer import train_model
from src.model_evaluator import evaluate_model
from src.shap import explain


def main():
    model,X_test,y_test=train_model()
    evaluate_model(model,X_test,y_test)


if __name__ == "__main__":
    main()
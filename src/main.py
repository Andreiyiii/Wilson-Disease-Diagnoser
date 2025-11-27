import pandas as pd
import numpy as np
from model_trainer import train_model_extended
from model_evaluator import evaluate_model
from shap_file import shap_plot


def main():
    model,X_test,y_test=train_model_extended()
    evaluate_model(model,X_test,y_test)
    shap_plot(model,X_test)


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np

def main():
    dataset=pd.read_csv("data/Wilson_disease_dataset.csv")
    print(dataset.head())


if __name__ == "__main__":
    main()
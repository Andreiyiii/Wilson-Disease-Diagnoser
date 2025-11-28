## Wilson Disease Diagnoser


Streamlit app that estimated probability of Wilson Disease based on lab tests and symptoms.It includes 2 XGBoost models:

- **Simple Model** – trained on Simple dataset(`data/Wilson_disease_dataset_simple.csv`).
- **Extended Model** – trained on Extended dataset (`data/Wilson_disease_dataset.csv`).


## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Training models

```bash
python -m src.main
```
Show ROC-AUC and Recall metrics then generate shap graphs



## Streamlit 
[Website](https://wilson-disease-diagnoser.streamlit.app/)

```bash
streamlit run streamlit_app.py
```
U can select the Simple Model or the Extended Model,introduce the values then press Predict.The expended model also shows a **Shap** plot



## Data
The original dataset is taken from [Kaggle](https://www.kaggle.com/datasets/guldanikaosmonova/wilson-disease-dataset/data)

Furthermore,I removed labels that were almost perfectly correlated with `Is_Wilson_Disease` tag such as `ATB7B Gene Mutation` and also removed useless ones like name and `Socioeconomic Status`.

- `Wilson_disease_dataset.csv` – dataset used for extended model 
- `Wilson_disease_dataset_simple.csv` – subset of Wilson_disease_dataset based on routine tests.
- `clean_sample.csv` , `clean_sample_simple.csv` – small parts of the complete dataset.



## Background of synthetic dataset creation

[GITHUB](https://github.com/Guldanika/Wilson-diseasepre-diction_on-mimic-data/blob/main/wilson_disease_detector_new.ipynb)



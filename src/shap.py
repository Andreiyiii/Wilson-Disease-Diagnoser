import shap
import matplotlib.pyplot as plt
import streamlit as st

def shap_plot(model, X):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    fig = plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    st.pyplot(fig)
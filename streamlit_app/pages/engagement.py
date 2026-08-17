import streamlit as st
import mlflow
import os

MODEL_PATH = "/Workspace/Users/242475r@mymail.nyp.edu.sg/IT3388_BDMP/game_performance_models"

st.write("### MLflow Models Check")

if os.path.exists(MODEL_PATH):
    items = os.listdir(MODEL_PATH)
    
    for item in items:
        full_path = os.path.join(MODEL_PATH, item)
        if os.path.isdir(full_path):
            try:
                model = mlflow.pyfunc.load_model(full_path)
                st.write(f"✅ {item}: Loaded successfully")
            except:
                st.write(f"⚠️ {item}: Not an MLflow model")
else:
    st.write("❌ Directory not found")

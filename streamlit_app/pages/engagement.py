import streamlit as st
import mlflow
import pandas as pd
import os

# =========================================================
# CONFIGURATION
# =========================================================

TABLE_NAME = os.getenv(
    "REVIEW_TABLE",
    "workspace.it3388.gold_review_analysis_final"
)

WAREHOUSE_ID = os.getenv(
    "DATABRICKS_WAREHOUSE_ID"
)

MLFLOW_EXPERIMENT = "/Users/242475r@mymail.nyp.edu.sg/IT3388_BDMP/game_performance_models"

# =========================================================
# LOAD MODELS FROM MLFLOW
# =========================================================

@st.cache_resource
def load_models():
    """Load models from MLflow experiment"""
    st.write("Loading models from MLflow...")
    
    # Set MLflow experiment
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    
    models_ccu = {}
    models_playtime = {}
    
    model_names = ['Linear Regression', 'Random Forest', 'XGBoost', 'LightGBM']
    
    # Load Peak CCU models
    for model_name in model_names:
        try:
            runs = mlflow.search_runs(
                filter_string=f"tags.mlflow.runName LIKE '{model_name} - Peak CCU%'",
                order_by=["metrics.test_r2 DESC"]
            )
            if len(runs) > 0:
                run_id = runs.iloc[0]['run_id']
                models_ccu[model_name] = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
                st.write(f"✅ {model_name} for Peak CCU (R² = {runs.iloc[0]['metrics.test_r2']:.4f})")
        except Exception as e:
            st.write(f"⚠️ {model_name} for Peak CCU: {str(e)}")
    
    # Load Average Playtime models
    for model_name in model_names:
        try:
            runs = mlflow.search_runs(
                filter_string=f"tags.mlflow.runName LIKE '{model_name} - Average Playtime%'",
                order_by=["metrics.test_r2 DESC"]
            )
            if len(runs) > 0:
                run_id = runs.iloc[0]['run_id']
                models_playtime[model_name] = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
                st.write(f"✅ {model_name} for Avg Playtime (R² = {runs.iloc[0]['metrics.test_r2']:.4f})")
        except Exception as e:
            st.write(f"⚠️ {model_name} for Avg Playtime: {str(e)}")
    
    st.write(f"\n✅ Loaded {len(models_ccu)} Peak CCU + {len(models_playtime)} Playtime models")
    
    return models_ccu, models_playtime

# Load models
models_ccu, models_playtime = load_models()

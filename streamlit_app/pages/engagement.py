import streamlit as st
import mlflow
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

# Get Databricks host from environment
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "https://dbc-43c2261c-2c09.cloud.databricks.com")

# =========================================================
# SETUP MLFLOW TO USE DATABRICKS TRACKING
# =========================================================

# Set MLflow tracking URI to Databricks workspace
mlflow.set_tracking_uri("databricks")

st.write("### MLflow Connection Check")
st.write(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

MLFLOW_EXPERIMENT = "/Users/242475r@mymail.nyp.edu.sg/IT3388_BDMP/game_performance_models"

try:
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
    
    if experiment:
        st.write(f"✅ Experiment found: {experiment.experiment_id}")
        
        # Search runs
        all_runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )
        
        st.write(f"Total runs: {len(all_runs)}")
        
        if len(all_runs) > 0:
            st.write("### Available Runs:")
            st.dataframe(all_runs[['run_id', 'tags.mlflow.runName', 'metrics.test_r2']])
        else:
            st.write("⚠️ No runs in experiment")
    else:
        st.write(f"❌ Experiment not found")
        
except Exception as e:
    st.error(f"Error: {str(e)}")
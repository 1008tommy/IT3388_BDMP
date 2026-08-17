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

MLFLOW_EXPERIMENT = "/Users/242475r@mymail.nyp.edu.sg/IT3388_BDMP/game_performance_models"

# =========================================================
# DEBUG: Check MLflow Setup
# =========================================================

st.write("### MLflow Connection Check")

# Check MLflow tracking URI
st.write(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

try:
    # Try to get experiment by name
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
    
    if experiment:
        st.write(f"✅ Experiment found: {experiment.experiment_id}")
        st.write(f"Experiment name: {experiment.name}")
        
        # Search runs using experiment_id
        all_runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )
        
        st.write(f"Total runs: {len(all_runs)}")
        
        if len(all_runs) > 0:
            st.write("### Available Runs:")
            st.dataframe(all_runs[['run_id', 'tags.mlflow.runName', 'metrics.test_r2', 'start_time']])
        else:
            st.write("⚠️ No runs found in experiment")
    else:
        st.write(f"❌ Experiment not found: {MLFLOW_EXPERIMENT}")
        
        # List all experiments
        st.write("\n### Available experiments:")
        all_experiments = mlflow.search_experiments()
        for exp in all_experiments:
            st.write(f"- {exp.name} (ID: {exp.experiment_id})")
            
except Exception as e:
    st.error(f"Error: {str(e)}")
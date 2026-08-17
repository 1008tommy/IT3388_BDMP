import streamlit as st
import mlflow
import os

# =========================================================
# DEBUG: Check MLflow Experiment
# =========================================================

st.write("### MLflow Debugging")

MLFLOW_EXPERIMENT = "/Users/242475r@mymail.nyp.edu.sg/IT3388_BDMP/game_performance_models"

try:
    # Set experiment
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    st.write(f"✅ Experiment set: {MLFLOW_EXPERIMENT}")
    
    # Search ALL runs in this experiment
    all_runs = mlflow.search_runs(order_by=["start_time DESC"])
    
    st.write(f"Total runs found: {len(all_runs)}")
    
    if len(all_runs) > 0:
        st.write("### All Runs:")
        st.dataframe(all_runs[['run_id', 'tags.mlflow.runName', 'metrics.test_r2', 'start_time']])
        
        st.write("### Available Run Names:")
        run_names = all_runs['tags.mlflow.runName'].dropna().unique()
        for name in run_names:
            st.write(f"- {name}")
    else:
        st.write("⚠️ No runs found in this experiment")
        
        # Check if it's a workspace folder instead
        st.write("\n### Checking workspace folder...")
        if os.path.exists(MLFLOW_EXPERIMENT):
            items = os.listdir(MLFLOW_EXPERIMENT)
            st.write(f"Items in folder: {items}")
        else:
            st.write("❌ Path doesn't exist")
            
except Exception as e:
    st.error(f"Error: {str(e)}")
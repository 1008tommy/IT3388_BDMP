import os
import streamlit as st
import mlflow

from databricks import sql
from databricks.sdk.core import Config

# =========================================================
# CONFIGURATION
# =========================================================

TABLE_NAME = os.getenv(
    "REVIEW_TABLE",
    "workspace.it3388.gold_price_summary"
)

WAREHOUSE_ID = os.getenv(
    "DATABRICKS_WAREHOUSE_ID"
)

MLFLOW_EXPERIMENT = "/Users/242475r@mymail.nyp.edu.sg/IT3388_BDMP/game_performance_models"

# =========================================================
# DATABRICKS CONNECTION
# =========================================================

cfg = Config()


def get_connection():
    if not WAREHOUSE_ID:
        st.error(
            "SQL Warehouse resource is not configured. "
            "Set DATABRICKS_WAREHOUSE_ID in the app environment."
        )
        st.stop()

    server_hostname = cfg.host or ""
    if server_hostname.startswith("https://"):
        server_hostname = server_hostname.replace("https://", "")
    elif server_hostname.startswith("http://"):
        server_hostname = server_hostname.replace("http://", "")

    http_path = f"/sql/1.0/warehouses/{WAREHOUSE_ID}"

    return sql.connect(
        server_hostname=server_hostname,
        http_path=http_path,
        credentials_provider=lambda: cfg.authenticate,
        use_cloud_fetch=False,
        _use_arrow_native_complex_types=False
    )


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data(ttl=1800)
def load_price_data():
    conn = get_connection()
    try:
        # OPTIONAL: check columns first
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {TABLE_NAME} LIMIT 0")
            # description may be None depending on driver; skip if so

        with conn.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {TABLE_NAME}")
            df = cursor.fetchall_arrow().to_pandas()
        return df
    finally:
        conn.close()


st.title("💰 Indie Game Pricing Strategy")
st.write("Explore how game features, scope and other characteristics are associated with indie game pricing.")

with st.spinner("Loading price data..."):
    price_df = load_price_data()

if price_df is None or price_df.empty:
    st.warning("No pricing data returned from the configured table.")
    st.stop()

# Simple test metrics
if "steam_id" in price_df.columns:
    total_games = int(price_df["steam_id"].nunique())
elif "app_id" in price_df.columns:
    total_games = int(price_df["app_id"].nunique())
else:
    total_games = len(price_df)

col = st.columns(1)[0]
col.metric("Number of unique games", f"{total_games:,}")

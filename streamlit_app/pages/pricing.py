import streamlit as st
import mlflow
import os

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


# =========================================================
# DATABRICKS CONNECTION
# =========================================================

cfg = Config()


def get_connection():

    if not WAREHOUSE_ID:
        st.error(
            "SQL Warehouse resource is not configured. "
            "Add the SQL Warehouse to the Databricks App."
        )
        st.stop()

    server_hostname = cfg.host

    if server_hostname.startswith("https://"):
        server_hostname = server_hostname.replace(
            "https://",
            ""
        )

    elif server_hostname.startswith("http://"):
        server_hostname = server_hostname.replace(
            "http://",
            ""
        )

    http_path = (
        f"/sql/1.0/warehouses/{WAREHOUSE_ID}"
    )

    return sql.connect(
        server_hostname=server_hostname,
        http_path=http_path,
        credentials_provider=lambda: cfg.authenticate,

        # IMPORTANT:
        # Fetch results through Databricks instead of CloudFetch
        use_cloud_fetch=False,

        _use_arrow_native_complex_types=False
    )

# =========================================================
# LOAD ONLY COLUMNS NEEDED FOR DASHBOARD
# =========================================================

@st.cache_data(ttl=1800)
def load_review_data():

    conn = get_connection()

    try:

        # First check which columns actually exist
        with conn.cursor() as cursor:

            cursor.execute(
                f"""
                SELECT *
                FROM {TABLE_NAME}
                LIMIT 0
                """
            )

        with conn.cursor() as cursor:

            cursor.execute(
                f"""
                SELECT
                    *
                FROM {TABLE_NAME}
                """
            )

            price_df = (
                cursor
                .fetchall_arrow()
                .to_pandas()
            )

        return price_df

    finally:
        conn.close()


# =========================================================
# TEST
# =========================================================
st.header("TEST")


total_reviews = len(price_df)


if "steam_id" in price_df.columns:
    total_games = (
        scope_df["steam_id"]
        .nunique()
    )

else:
    total_games = 0

col1 = st.columns(1)


col1.metric(
    "Number of unique games",
    f"{total_games:,}"
)

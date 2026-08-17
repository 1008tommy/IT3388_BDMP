import streamlit as st
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Indie Game Pricing", layout="wide")
st.title("🎮 Indie Game Pricing Model Dashboard")

# ── Config ────────────────────────────────────────────────────────────────────
PYCARET_RUN_URI = "runs:/3e9c0f4b46e846faba9cacdaaa1aae24/model"
RF_RUN_URI      = "runs:/99c1924e7e674d659dad7348bd0db61b/model"
GOLD_PATH       = "/Volumes/darren/default/it3388/gold_price_summary.csv"
METADATA_PATH   = "/Volumes/darren/default/it3388/silver_game_metadata.csv"
RAW_LIST_COLS   = ["supported_languages", "full_audio_languages", "developers",
                   "publishers", "categories", "genres", "tags"]

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    gold = pd.read_csv(GOLD_PATH).astype({
        "steam_id": "int64", "launch_price": "float64",
        "launched_with_discount": "bool", "min_price": "float64",
        "max_price": "float64", "min_base_price": "float64",
        "max_base_price": "float64", "avg_discount_pct": "float64",
        "pct_time_discounted": "float64", "n_discount_events": "int64",
        "price_volatility": "int64", "n_price_changes": "int64",
        "has_price_history": "bool",
    }).dropna()

    meta = pd.read_csv(METADATA_PATH).astype({
        "app_id": "int64", "name": "string",
        "release_date": "datetime64[ns]", "estimated_owners": "string",
    })

    df = meta.merge(gold, left_on="app_id", right_on="steam_id")
    df = df[df["launch_price"] != 0].drop(columns=RAW_LIST_COLS, errors="ignore")

    genre_dummy_cols = [c for c in df.columns if c.startswith("genres_")]
    df["review_ratio"] = df["positive"] / (df["positive"] + df["negative"]).replace(0, np.nan)

    scope   = ["platform_count", "supported_languages_count", "audio_languages_count",
               "category_count", "genre_count", "tag_count", "total_tag_votes",
               "tag_diversity", "dlc_count", "achievements"]
    control = ["positive", "negative", "average_playtime_forever", "median_playtime_forever",
               "peak_ccu", "recommendations", "days_since_release"]
    feature_names = scope + control + genre_dummy_cols + ["review_ratio"]

    df = df.dropna()
    X, y = df[feature_names], df["launch_price"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_test, y_test, feature_names

# ── Models ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    mlflow.set_tracking_uri("databricks")
    return (
        mlflow.sklearn.load_model(PYCARET_RUN_URI),
        mlflow.sklearn.load_model(RF_RUN_URI),
    )

X_test, y_test, feature_names = load_data()
pycaret_model, rf_model = load_models()

st.title("💰 Indie Game Pricing Strategy")
st.write("Explore how game features, scope and other characteristics are associated with indie game pricing.")

st.heading("Models and model performance")
model_cols = st.columns(2)
for (name, model), col in zip(
    [("PyCaret Best Model", pycaret_model), ("Random Forest", rf_model)], model_cols
):
    y_pred = model.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(
        y_test,
        y_pred,
    )
    mae  = mean_absolute_error(y_test, y_pred)

    with col:
        st.subheader(name)
        m1, m2, m3 = st.columns(3)
        m1.metric("R²",   f"{r2:.4f}")
        m2.metric("RMSE", f"${rmse:.2f}")
        m3.metric("MAE",  f"${mae:.2f}")

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_test, y_pred, alpha=0.3, s=15, color="steelblue")
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", lw=1.5, label="Perfect fit")
        ax.set_xlabel("Actual ($)")
        ax.set_ylabel("Predicted ($)")
        ax.set_title("Predicted vs Actual")
        ax.legend(fontsize=8)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# # ── Tabs ──────────────────────────────────────────────────────────────────────
# tab1, tab2, tab3 = st.tabs(["📊 Model Performance", "📌 Feature Importance", "🔍 SHAP Explainability"])

# # ── Tab 1: Performance ────────────────────────────────────────────────────────
# with tab1:
#     cols = st.columns(2)
#     for (name, model), col in zip(
#         [("PyCaret Best Model", pycaret_model), ("Random Forest", rf_model)], cols
#     ):
#         y_pred = model.predict(X_test)
#         r2   = r2_score(y_test, y_pred)
#         rmse = root_mean_squared_error(y_test, y_pred, squared=False)
#         mae  = mean_absolute_error(y_test, y_pred)

#         with col:
#             st.subheader(name)
#             m1, m2, m3 = st.columns(3)
#             m1.metric("R²",   f"{r2:.4f}")
#             m2.metric("RMSE", f"${rmse:.2f}")
#             m3.metric("MAE",  f"${mae:.2f}")

#             fig, ax = plt.subplots(figsize=(5, 4))
#             ax.scatter(y_test, y_pred, alpha=0.3, s=15, color="steelblue")
#             lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
#             ax.plot(lims, lims, "r--", lw=1.5, label="Perfect fit")
#             ax.set_xlabel("Actual ($)")
#             ax.set_ylabel("Predicted ($)")
#             ax.set_title("Predicted vs Actual")
#             ax.legend(fontsize=8)
#             st.pyplot(fig, use_container_width=True)
#             plt.close(fig)

# # ── Tab 2: Feature Importance ─────────────────────────────────────────────────
# with tab2:
#     st.subheader("Random Forest Feature Importance")
#     top_n = st.slider("Top N features", 5, len(feature_names), 20)
#     fi_df = (
#         pd.DataFrame({"feature": feature_names, "importance": rf_model.feature_importances_})
#         .sort_values("importance", ascending=False)
#         .head(top_n)
#     )

#     fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35)))
#     ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color="steelblue")
#     ax.set_xlabel("Importance")
#     ax.set_title(f"Top {top_n} Features — Random Forest")
#     plt.tight_layout()
#     st.pyplot(fig, use_container_width=True)
#     plt.close(fig)

# # ── Tab 3: SHAP ────────────────────────────────────────────────────────────────
# with tab3:
#     st.subheader("SHAP Explainability — Random Forest")
#     n_samples = st.slider("Sample size", 50, min(300, len(X_test)), 100)
#     X_shap = X_test.sample(n_samples, random_state=42)

#     with st.spinner("Computing SHAP values..."):
#         explainer   = shap.TreeExplainer(rf_model)
#         shap_values = explainer.shap_values(X_shap)

#     st.markdown("**Beeswarm — feature impact distribution**")
#     shap.summary_plot(shap_values, X_shap, feature_names=feature_names, show=False)
#     st.pyplot(plt.gcf(), use_container_width=True)
#     plt.clf()

#     st.markdown("**Bar — mean |SHAP| per feature**")
#     shap.summary_plot(shap_values, X_shap, feature_names=feature_names,
#                       plot_type="bar", show=False)
#     st.pyplot(plt.gcf(), use_container_width=True)
#     plt.clf()

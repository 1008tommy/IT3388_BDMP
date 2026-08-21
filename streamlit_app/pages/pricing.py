import streamlit as st
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import os
from databricks import sql
from databricks.sdk.core import Config

# -- Config --------------------------------------------------------------------
PYCARET_RUN_URI = "runs:/3e9c0f4b46e846faba9cacdaaa1aae24/model"
RF_RUN_URI = "runs:/99c1924e7e674d659dad7348bd0db61b/model"
GOLD_PATH = "/Workspace/Users/242475r@mymail.nyp.edu.sg/IT3388_BDMP/gold_price_summary.csv"
METADATA_PATH = "/Workspace/Users/242475r@mymail.nyp.edu.sg/IT3388_BDMP/silver_game_metadata.csv"
RAW_LIST_COLS = ["supported_languages", "full_audio_languages", "developers", "publishers", "categories", "genres", "tags"]


# -- Load Data -----------------------------------------------------------------

PRICE_TABLE = os.getenv("PRICE_TABLE", "workspace.it3388.gold_price_summary")

METADATA_TABLE = os.getenv("METADATA_TABLE", "workspace.it3388.silver_game_metadata")

WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID")

cfg = Config()


def get_connection():

    if not WAREHOUSE_ID:
        st.error("SQL Warehouse resource is not configured. Add the SQL Warehouse to the Databricks App.")
        st.stop()

    server_hostname = cfg.host

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
        _use_arrow_native_complex_types=False,
    )


@st.cache_data
def load_data():

    conn = get_connection()

    try:
        # Load the two Unity Catalog tables
        gold = pd.read_sql(f"SELECT * FROM {PRICE_TABLE}", conn)

        meta = pd.read_sql(f"SELECT * FROM {METADATA_TABLE}", conn)

    finally:
        conn.close()

    # -------------------------------------------------------------------------
    # Same processing as your original code
    # -------------------------------------------------------------------------

    gold = gold.astype(
        {
            "steam_id": "int64",
            "launch_price": "float64",
            "launched_with_discount": "bool",
            "min_price": "float64",
            "max_price": "float64",
            "min_base_price": "float64",
            "max_base_price": "float64",
            "avg_discount_pct": "float64",
            "pct_time_discounted": "float64",
            "n_discount_events": "int64",
            "price_volatility": "int64",
            "n_price_changes": "int64",
            "has_price_history": "bool",
        }
    ).dropna()

    meta = meta.astype(
        {
            "app_id": "int64",
            "name": "string",
            "release_date": "datetime64[ns]",
            "estimated_owners": "string",
        }
    )

    # Merge price + metadata
    price_df = meta.merge(gold, left_on="app_id", right_on="steam_id")

    # Remove free games
    price_df = price_df[price_df["launch_price"] != 0].copy()

    # Remove raw list columns
    price_df = price_df.drop(columns=RAW_LIST_COLS, errors="ignore")

    # Genre dummy columns
    genre_dummy_cols = [c for c in price_df.columns if c.startswith("genres_")]

    return (
        price_df,
        genre_dummy_cols,
    )


price_df, genre_dummy_cols = load_data()
price_cap = price_df["launch_price"].quantile(0.99)

# -- Overview ------------------------------------------------------------------
st.header("Overview of Indie Games")

total_games = price_df["steam_id"].nunique()
median_price = price_df["launch_price"].median()
mean_price = price_df["launch_price"].mean()
discount_pct = price_df["launched_with_discount"].mean() * 100
dlc_pct = (price_df["dlc_count"] > 0).mean() * 100
median_achievements = price_df["achievements"].median()

overview_cols = st.columns(6)
overview_cols[0].metric("Games analyzed", f"{total_games:,}")
overview_cols[1].metric("Median launch price", f"${median_price:.2f}")
overview_cols[2].metric("Mean launch price", f"${mean_price:.2f}")
overview_cols[3].metric("Launched with discount", f"{discount_pct:.1f}%")
overview_cols[4].metric("Games with DLC", f"{dlc_pct:.1f}%")
overview_cols[5].metric("Median achievements", f"{median_achievements:.0f}")

# -- Price Distribution --------------------------------------------------------
st.header("Launch Price Distribution")

fig, ax = plt.subplots(figsize=(10, 5))

prices = price_df["launch_price"]
upper_limit = prices.quantile(0.99)

ax.hist(prices, bins=50, range=(0, upper_limit), alpha=0.75)
ax.set_xlim(0, upper_limit)
ax.set_xlabel("Launch Price ($)")
ax.set_ylabel("Number of Games")
ax.set_title("Distribution of Indie Game Launch Prices")

st.pyplot(fig, use_container_width=True)
plt.close(fig)

# -- Genre Pricing -------------------------------------------------------------
st.header("Genre and Launch Price")

genre_rows = []

for genre_col in genre_dummy_cols:
    genre_name = genre_col.replace("genres_", "").replace("_", " ").title()

    genre_games = price_df[price_df[genre_col] == 1]

    if len(genre_games) >= 20:
        genre_rows.append(
            {
                "Genre": genre_name,
                "Median Price": genre_games["launch_price"].median(),
                "Mean Price": genre_games["launch_price"].mean(),
                "Games": len(genre_games),
            }
        )

genre_df = pd.DataFrame(genre_rows).sort_values("Median Price", ascending=True)

fig, ax = plt.subplots(figsize=(10, max(5, len(genre_df) * 0.35)))

ax.barh(genre_df["Genre"], genre_df["Median Price"])
ax.set_xlim(0, price_cap)
ax.set_xlabel("Median Launch Price ($)")
ax.set_ylabel("Genre")
ax.set_title("Median Launch Price by Genre")

st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.dataframe(
    genre_df.sort_values("Median Price", ascending=False),
    use_container_width=True,
    hide_index=True,
)

# -- Game Scope vs Price -------------------------------------------------------
st.header("Game Scope and Launch Price")

scope_cols = st.columns(3)

scope_features = [
    ("achievements", "Achievements"),
    ("dlc_count", "DLC Count"),
    ("platform_count", "Platform Count"),
]

for col, (feature, label) in zip(scope_cols, scope_features):
    with col:
        fig, ax = plt.subplots(figsize=(5, 4))

        ax.scatter(price_df[feature], price_df["launch_price"], alpha=0.25, s=12)
        ax.set_ylim(0, price_cap)

        ax.set_xlabel(label)
        ax.set_ylabel("Launch Price ($)")
        ax.set_title(f"{label} vs Launch Price")

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# -- Discount Strategy ---------------------------------------------------------
st.header("Discount Strategy")

discount_cols = st.columns(4)
discount_cols[0].metric("Games launched with discount", f"{discount_pct:.1f}%")
discount_cols[1].metric("Average discount", f"{price_df['avg_discount_pct'].mean():.1f}%")
discount_cols[2].metric("Median time discounted", f"{price_df['pct_time_discounted'].median():.1f}%")
discount_cols[3].metric("Median discount events", f"{price_df['n_discount_events'].median():.0f}")

fig, ax = plt.subplots(figsize=(9, 5))

ax.scatter(price_df["launch_price"], price_df["avg_discount_pct"], alpha=0.25, s=12)
ax.set_xlim(0, price_cap)
ax.set_xlabel("Launch Price ($)")
ax.set_ylabel("Average Discount (%)")
ax.set_title("Launch Price vs Average Discount")

st.pyplot(fig, use_container_width=True)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))

ax.scatter(price_df["n_discount_events"], price_df["launch_price"], alpha=0.25, s=12)
ax.set_ylim(0, price_cap)
ax.set_xlabel("Number of Discount Events")
ax.set_ylabel("Launch Price ($)")
ax.set_title("Discount Frequency vs Launch Price")

st.pyplot(fig, use_container_width=True)
plt.close(fig)

# -- Conclusion ------------------------------------------------------------------
st.markdown("""
## Conclusion

The data shows a weak relationship between game scope and launch price, weaker than the original hypothesis proposed, and driven more by popularity and genre-type signals than by the specific content-scope features named.

**Model performance.** The best model (LightGBM, R²=0.237) outperforms linear regression (R²=0.141) and a mean-only baseline (R²≈0), showing the relationship has some non-linear structure. However, even the best model explains under a quarter of price variance, and performance is unstable across folds (R² ranging 0.10–0.38) — most of what determines an indie game's price isn't captured by scope or metadata at all.

**What actually predicts price doesn't match the hypothesis.** The top predictors are `total_tag_votes`, `days_since_release`, and `peak_ccu` are popularity and engagement signals, not content scope. The features the hypothesis specifically named (**DLC count and platform/language support**) show no visible relationship with price in the data. Achievement count is predictive, but in the opposite direction expected: very high achievement counts cluster at *low* prices, consistent with low-effort "achievement farming" titles rather than content-rich games.

**Genre reflects product type more than content depth.** The highest median prices belong to productivity/creative tools (Web Publishing, Video Production, Game Development, Design & Illustration, \~\$15–20) rather than games. Actual game genres cluster around \$8–10, with Casual/Indie/Free-to-Play lowest (\~\$5–6). This suggests genre is partly separating "software tool" from "game" rather than measuring scope within games.

**Price follows convention, not a continuous function of scope.** Launch prices spike sharply at Steam's standard price points (\$4.99, \$9.99, \$14.99, \$19.99, \$24.99), suggesting developers select from a small set of conventional tiers rather than pricing continuously based on features which limits how well any regression can fit the underlying decision.

**Discounting is a launch-week convention, not a sustained strategy.** While 66.8\% of games launch with a discount, the median game spends only 0.1\% of its lifetime on sale. The discount pattern is dominated by one-time launch promotions rather than repeated, ongoing discounting. Discount frequency and depth show no clear relationship with price tier.

**Overall:** the hypothesis is partially supported (pricing is patterned and non-random) but the mechanism differs from what was proposed. Popularity and product-type signals explain more of the variation than the specific scope features (achievements, DLC, platform/language support) originally hypothesized to drive price.
""")


# # -- Models --------------------------------------------------------------------
# @st.cache_resource
# def load_models():
#     mlflow.set_tracking_uri("databricks")
#     return (
#         mlflow.sklearn.load_model(PYCARET_RUN_URI),
#         mlflow.sklearn.load_model(RF_RUN_URI),
#     )


# pycaret_model, rf_model = load_models()

# st.header("💰 Indie Game Pricing Strategy")
# st.write("Explore how game features, scope and other characteristics are associated with indie game pricing.")

# st.header("Models and model performance")
# model_cols = st.columns(2)
# for (name, model), col in zip(
#     [("PyCaret Best Model", pycaret_model), ("Random Forest", rf_model)],
#     model_cols,
# ):
#     y_pred = model.predict(X_test)
#     r2 = r2_score(y_test, y_pred)
#     rmse = root_mean_squared_error(
#         y_test,
#         y_pred,
#     )
#     mae = mean_absolute_error(y_test, y_pred)

#     with col:
#         st.subheader(name)
#         m1, m2, m3 = st.columns(3)
#         m1.metric("R²", f"{r2:.4f}")
#         m2.metric("RMSE", f"${rmse:.2f}")
#         m3.metric("MAE", f"${mae:.2f}")

#         fig, ax = plt.subplots(figsize=(5, 4))
#         ax.scatter(y_test, y_pred, alpha=0.3, s=15, color="steelblue")
#         lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
#         ax.plot(lims, lims, "r--", lw=1.5, label="Perfect fit")
#         ax.set_xlabel("Actual ($)")
#         ax.set_ylabel("Predicted ($)")
#         ax.set_title("Predicted vs Actual")
#         ax.legend(fontsize=8)
#         st.pyplot(fig, use_container_width=True)
#         plt.close(fig)

# st.header("Random Forest Feature Importance")
# top_n = st.slider("Top N features", 5, len(feature_names), 20)
# fi_df = pd.DataFrame({"feature": feature_names, "importance": rf_model.feature_importances_}).sort_values("importance", ascending=False).head(top_n)

# fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35)))
# ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color="steelblue")
# ax.set_xlabel("Importance")
# ax.set_title(f"Top {top_n} Features — Random Forest")
# plt.tight_layout()
# st.pyplot(fig, use_container_width=True)
# plt.close(fig)


# @st.cache_data
# def calculate_shap_values(_model, X):
#     explainer = shap.TreeExplainer(_model)
#     return explainer.shap_values(X)


# st.header("SHAP Model Explainability")
# n_samples = st.slider("Sample size", 50, min(300, len(X_test)), 100)
# X_shap = X_test.sample(n_samples, random_state=42)

# with st.spinner("Computing SHAP values..."):
#     shap_values = calculate_shap_values(rf_model, X_shap)

# st.markdown("**Beeswarm — feature impact distribution**")
# shap.summary_plot(shap_values, X_shap, feature_names=feature_names, show=False)
# st.pyplot(plt.gcf(), use_container_width=True)
# plt.clf()

# st.markdown("**Bar — mean |SHAP| per feature**")
# shap.summary_plot(
#     shap_values,
#     X_shap,
#     feature_names=feature_names,
#     plot_type="bar",
#     show=False,
# )
# st.pyplot(plt.gcf(), use_container_width=True)
# plt.clf()

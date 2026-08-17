import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px

from databricks import sql
from databricks.sdk.core import Config


# CONFIGURATION

TABLE_NAME = os.getenv(
    "REVIEW_TABLE",
    "it3388.it3388.gold_review_analysis_final"
)

WAREHOUSE_ID = os.getenv(
    "DATABRICKS_WAREHOUSE_ID"
)


# DATABRICKS CONNECTION

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


# LOAD ONLY COLUMNS NEEDED FOR DASHBOARD

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

            available_columns = [
                column[0]
                for column in cursor.description
            ]

        wanted_columns = [
            "app_id",
            "name",
            "game_type",
            "primary_genre",  
            "genres",
            "price",
            "peak_ccu",
            "recommendation_polarity",
            "main_theme",
            "theme_count",
            "review_length_words"
        ]

        columns_to_load = [
            column
            for column in wanted_columns
            if column in available_columns
        ]

        selected_columns = ", ".join(
            [
                f"`{column}`"
                for column in columns_to_load
            ]
        )

        with conn.cursor() as cursor:

            cursor.execute(
                f"""
                SELECT
                    {selected_columns}
                FROM {TABLE_NAME}
                """
            )

            df = (
                cursor
                .fetchall_arrow()
                .to_pandas()
            )

        return df

    finally:
        conn.close()


# LOAD DATA

with st.spinner("Loading review analytics data..."):
    df = load_review_data()


# BASIC CLEANING

required_columns = [
    "main_theme",
    "recommendation_polarity"
]

missing_required = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_required:

    st.error(
        "The following required columns are missing: "
        + ", ".join(missing_required)
    )

    st.stop()


# Numeric columns

for column in [
    "price",
    "peak_ccu",
    "theme_count",
    "review_length_words"
]:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# GENRE PARSING
# Works whether genres came across as:
# ["Action", "Indie"]
# Action, Indie
# or an actual Python list

def parse_genres(value):

    if isinstance(value, list):
        return value

    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    text = str(value).strip()

    if text == "":
        return []

    # Try JSON first
    try:

        parsed = json.loads(text)

        if isinstance(parsed, list):
            return [
                str(item).strip()
                for item in parsed
                if str(item).strip()
            ]

    except Exception:
        pass

    # Otherwise assume comma-separated text
    text = text.strip("[]")

    return [
        part.strip().strip("'\"")
        for part in text.split(",")
        if part.strip().strip("'\"")
    ]


if "genres" in df.columns:

    df["genre_list"] = (
        df["genres"]
        .apply(parse_genres)
    )


# PAGE HEADER

st.title("Player Review Analysis")

st.write(
    """
    Explore what players praise and complain about across
    different types of Steam games. These insights can help
    indie developers identify recurring player expectations,
    compare similar games, and prioritise areas for improvement.
    """
)


# OVERVIEW METRICS

st.header("Review Overview")


# Total reviews
total_reviews = len(df)


# Total games
if "app_id" in df.columns:

    total_games = (
        df["app_id"]
        .nunique()
    )

else:

    total_games = 0


# Positive reviews
positive_reviews = (
    df["recommendation_polarity"]
    == "Positive"
).sum()


positive_percentage = (
    positive_reviews
    / total_reviews
    * 100
    if total_reviews > 0
    else 0
)


# Negative reviews
negative_reviews = (
    df["recommendation_polarity"]
    == "Negative"
).sum()


negative_percentage = (
    negative_reviews
    / total_reviews
    * 100
    if total_reviews > 0
    else 0
)


# DISPLAY OVERVIEW CARDS

col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Reviews Analysed",
    f"{total_reviews:,}"
)


col2.metric(
    "Games Analysed",
    f"{total_games:,}"
)


col3.metric(
    "Positive Reviews",
    f"{positive_percentage:.1f}%"
)


col4.metric(
    "Negative Reviews",
    f"{negative_percentage:.1f}%"
)


st.divider()


# ---------------------------------------------------------
# COUNT THEME + POSITIVE / NEGATIVE
# ---------------------------------------------------------

theme_sentiment_counts = (
    selected_genre_df
    .groupby(
        [
            "main_theme",
            "recommendation_polarity"
        ]
    )
    .size()
    .reset_index(
        name="review_count"
    )
)


# ---------------------------------------------------------
# TOTAL REVIEWS FOR EACH THEME
# ---------------------------------------------------------

theme_sentiment_counts["theme_total"] = (
    theme_sentiment_counts
    .groupby("main_theme")["review_count"]
    .transform("sum")
)


# ---------------------------------------------------------
# POSITIVE / NEGATIVE % WITHIN EACH THEME
#
# EACH BAR ADDS UP TO 100%
# ---------------------------------------------------------

theme_sentiment_counts["percentage"] = (
    theme_sentiment_counts["review_count"]
    / theme_sentiment_counts["theme_total"]
    * 100
)


# ---------------------------------------------------------
# CALCULATE HOW COMMON EACH THEME IS OVERALL
# Used only to sort the themes and show context
# ---------------------------------------------------------

specific_feedback_total = (
    selected_genre_df.shape[0]
)


theme_summary = (
    theme_sentiment_counts
    .groupby("main_theme")
    .agg(
        theme_total=("review_count", "sum")
    )
    .reset_index()
)


theme_summary["theme_share"] = (
    theme_summary["theme_total"]
    / specific_feedback_total
    * 100
)


# ---------------------------------------------------------
# ADD THEME SHARE TO LABEL
#
# Example:
# Gameplay and balance (38.4% of feedback)
# ---------------------------------------------------------

theme_summary["theme_label"] = (
    theme_summary["main_theme"]
    + " ("
    + theme_summary["theme_share"].map(
        lambda x: f"{x:.1f}%"
    )
    + ")"
)


# Add label back into chart dataframe
theme_sentiment_counts = (
    theme_sentiment_counts
    .merge(
        theme_summary[
            [
                "main_theme",
                "theme_label",
                "theme_share"
            ]
        ],
        on="main_theme",
        how="left"
    )
)


# ---------------------------------------------------------
# SORT THEMES
# Most commonly discussed theme at the top
# ---------------------------------------------------------

theme_order = (
    theme_summary
    .sort_values(
        "theme_total",
        ascending=True
    )["theme_label"]
    .tolist()
)


# ---------------------------------------------------------
# CAPTION
# ---------------------------------------------------------

st.caption(
    f"Based on {specific_feedback_total:,} reviews containing "
    f"a specific feedback theme for Indie {selected_genre} games. "
    f"Each bar adds up to 100% and shows the positive vs negative "
    f"split within that feedback theme."
)


# ---------------------------------------------------------
# 100% STACKED BAR CHART
# ---------------------------------------------------------

fig_theme = px.bar(
    theme_sentiment_counts,

    # Positive + Negative = 100% for each theme
    x="percentage",

    y="theme_label",

    color="recommendation_polarity",

    orientation="h",

    barmode="stack",

    # Percentage shown inside each section
    text="percentage",

    category_orders={
        "theme_label": theme_order,

        "recommendation_polarity": [
            "Positive",
            "Negative"
        ]
    },

    color_discrete_map={
        "Positive": "#2ECC71",
        "Negative": "#E74C3C"
    },

    title=(
        f"Positive vs Negative Feedback by Theme "
        f"for Indie {selected_genre} Games"
    ),

    labels={
        "percentage":
            "% Within Theme",

        "theme_label":
            "Feedback Theme",

        "recommendation_polarity":
            "Recommendation"
    },

    custom_data=[
        "review_count",
        "theme_share"
    ]
)


# ---------------------------------------------------------
# CENTRE THE PERCENTAGES INSIDE EACH COLOURED SECTION
# ---------------------------------------------------------

fig_theme.update_traces(

    texttemplate="%{text:.1f}%",

    textposition="inside",

    # Centre percentage in green/red segment
    insidetextanchor="middle",

    textfont=dict(
        color="white",
        size=13
    ),

    hovertemplate=(
        "<b>%{y}</b><br>"
        "Recommendation: %{fullData.name}<br>"
        "Within this theme: %{x:.1f}%<br>"
        "Reviews: %{customdata[0]:,}<br>"
        "Theme share of all specific feedback: "
        "%{customdata[1]:.1f}%"
        "<extra></extra>"
    )
)


# ---------------------------------------------------------
# LAYOUT
# ---------------------------------------------------------

fig_theme.update_layout(

    height=550,

    barmode="stack",

    # Every bar is exactly 100%
    xaxis=dict(
        range=[0, 100],
        tickmode="linear",
        dtick=10,
        ticksuffix="%"
    ),

    xaxis_title=(
        "Positive vs Negative Share Within Theme"
    ),

    yaxis_title=(
        "Feedback Theme (% of All Specific Feedback)"
    ),

    legend_title_text=(
        "Recommendation"
    ),

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),

    # Keep labels visible where possible
    uniformtext_minsize=10,
    uniformtext_mode="show"
)


st.plotly_chart(
    fig_theme,
    width="stretch"
)
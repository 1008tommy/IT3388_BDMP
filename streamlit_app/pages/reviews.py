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


# =========================================================
# CHART 1
# KEY FEEDBACK THEMES BY INDIE PRIMARY GENRE
# POSITIVE VS NEGATIVE STACKED BAR
# =========================================================

st.header("What Feedback Themes Should Indie Developers Look Out For?")

st.write(
    """
    Select an indie game genre to see the specific feedback themes
    most commonly discussed by players. General positive and negative
    feedback are excluded so that the chart focuses on more actionable
    areas such as gameplay, technical issues, content, story and pricing.

    Each feedback theme is split into positive and negative reviews,
    allowing developers to see whether players are mainly praising or
    criticising that aspect of similar games.
    """
)


# ---------------------------------------------------------
# GENRES USED IN THE MODELLING / SAMPLING PLAN
# ---------------------------------------------------------

MODEL_GENRES = [
    "Action",
    "Adventure",
    "Casual",
    "Simulation",
    "Strategy",
    "RPG"
]


# ---------------------------------------------------------
# PREPARE INDIE REVIEW DATA
# ---------------------------------------------------------

if (
    "primary_genre" in df.columns
    and "game_type" in df.columns
    and "recommendation_polarity" in df.columns
):

    indie_genre_df = df[
        (df["game_type"] == "Indie")
        & (df["primary_genre"].isin(MODEL_GENRES))
        & (df["main_theme"].notna())
        & (df["recommendation_polarity"].isin(
            ["Positive", "Negative"]
        ))
    ].copy()


    # ---------------------------------------------------------
    # REMOVE GENERAL FEEDBACK
    # ---------------------------------------------------------

    general_themes = [
        "General positive feedback",
        "General negative feedback"
    ]


    indie_genre_df = indie_genre_df[
        ~indie_genre_df["main_theme"].isin(
            general_themes
        )
    ]


    # ---------------------------------------------------------
    # GENRE SELECTOR
    # ---------------------------------------------------------

    available_genres = [
        genre
        for genre in MODEL_GENRES
        if genre in indie_genre_df[
            "primary_genre"
        ].unique()
    ]


    selected_genre = st.selectbox(
        "Select an indie genre",
        available_genres
    )


    # ---------------------------------------------------------
    # FILTER TO SELECTED PRIMARY GENRE
    # ---------------------------------------------------------

    selected_genre_df = indie_genre_df[
        indie_genre_df["primary_genre"]
        == selected_genre
    ].copy()


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
    # TOTAL SPECIFIC FEEDBACK
    # ---------------------------------------------------------

    specific_feedback_total = (
        theme_sentiment_counts[
            "review_count"
        ].sum()
    )


    # ---------------------------------------------------------
    # PERCENTAGE OF ALL SPECIFIC FEEDBACK
    #
    # Positive + Negative segments across ALL themes
    # therefore add up to 100%
    # ---------------------------------------------------------

    if specific_feedback_total > 0:

        theme_sentiment_counts[
            "percentage"
        ] = (
            theme_sentiment_counts[
                "review_count"
            ]
            / specific_feedback_total
            * 100
        )

    else:

        theme_sentiment_counts[
            "percentage"
        ] = 0


    # ---------------------------------------------------------
    # CALCULATE SENTIMENT SHARE WITHIN EACH THEME
    # Used for hover information
    # ---------------------------------------------------------

    theme_sentiment_counts[
        "theme_total"
    ] = (
        theme_sentiment_counts
        .groupby("main_theme")[
            "review_count"
        ]
        .transform("sum")
    )


    theme_sentiment_counts[
        "within_theme_percentage"
    ] = (
        theme_sentiment_counts[
            "review_count"
        ]
        / theme_sentiment_counts[
            "theme_total"
        ]
        * 100
    )


    # ---------------------------------------------------------
    # FIND TOTAL SHARE OF EACH THEME
    # Used to sort bars
    # ---------------------------------------------------------

    theme_totals = (
        theme_sentiment_counts
        .groupby("main_theme")[
            "percentage"
        ]
        .sum()
        .sort_values(
            ascending=True
        )
    )


    theme_order = (
        theme_totals.index.tolist()
    )


    # ---------------------------------------------------------
    # CAPTION
    # ---------------------------------------------------------

    st.caption(
        f"Based on {specific_feedback_total:,} reviews containing "
        f"a specific feedback theme for Indie {selected_genre} games. "
        f"General positive and negative feedback are excluded. "
        f"All displayed segments together add up to 100%."
    )


    # ---------------------------------------------------------
    # STACKED BAR CHART
    # ---------------------------------------------------------

    fig_theme = px.bar(
        theme_sentiment_counts,

        x="percentage",
        y="main_theme",

        color="recommendation_polarity",

        orientation="h",

        barmode="stack",

        text="percentage",

        category_orders={
            "main_theme": theme_order,
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
            f"Key Feedback Themes for "
            f"Indie {selected_genre} Games"
        ),

        labels={
            "percentage":
                "% of Specific Feedback",

            "main_theme":
                "Feedback Theme",

            "recommendation_polarity":
                "Recommendation"
        },

        custom_data=[
            "review_count",
            "within_theme_percentage"
        ]
    )


    # ---------------------------------------------------------
    # BAR LABELS + HOVER
    # ---------------------------------------------------------

    fig_theme.update_traces(

        texttemplate="%{text:.1f}%",

        textposition="inside",

        hovertemplate=(
            "<b>%{y}</b><br>"
            "Recommendation: %{fullData.name}<br>"
            "Share of all specific feedback: "
            "%{x:.1f}%<br>"
            "Reviews: %{customdata[0]:,}<br>"
            "Within this theme: "
            "%{customdata[1]:.1f}%"
            "<extra></extra>"
        )
    )


    # ---------------------------------------------------------
    # LAYOUT
    # ---------------------------------------------------------

    fig_theme.update_layout(

        height=520,

        barmode="stack",

        xaxis_title=(
            "% of Specific Feedback"
        ),

        yaxis_title=(
            "Feedback Theme"
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

        xaxis=dict(
            range=[
                0,
                max(
                    theme_totals.max() * 1.15,
                    10
                )
            ]
        )
    )


    st.plotly_chart(
        fig_theme,
        width="stretch"
    )

else:

    st.warning(
        "The primary_genre, game_type or "
        "recommendation_polarity column is not available."
    )


st.divider()
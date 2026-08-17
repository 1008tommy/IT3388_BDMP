import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px

from databricks import sql
from databricks.sdk.core import Config


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

            available_columns = [
                column[0]
                for column in cursor.description
            ]

        wanted_columns = [
            "app_id",
            "name",
            "game_type",
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


# =========================================================
# LOAD DATA
# =========================================================

with st.spinner("Loading review analytics data..."):
    df = load_review_data()


# =========================================================
# BASIC CLEANING
# =========================================================

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


# =========================================================
# GENRE PARSING
# Works whether genres came across as:
# ["Action", "Indie"]
# Action, Indie
# or an actual Python list
# =========================================================

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


# =========================================================
# PAGE HEADER
# =========================================================

st.title("💬 Player Review Analysis")

st.write(
    """
    Explore what players praise and complain about across
    different types of Steam games. These insights can help
    indie developers identify recurring player expectations,
    compare similar games, and prioritise areas for improvement.
    """
)


# =========================================================
# ANALYSIS SCOPE
# =========================================================

scope_df = df.copy()


if "game_type" in df.columns:

    available_game_types = sorted(
        df["game_type"]
        .dropna()
        .astype(str)
        .unique()
    )

    scope_option = st.selectbox(
        "Analysis scope",
        ["All games"] + available_game_types
    )

    if scope_option != "All games":

        scope_df = df[
            df["game_type"].astype(str)
            == scope_option
        ].copy()


# =========================================================
# OVERVIEW METRICS
# =========================================================

st.header("📊 Review Overview")


total_reviews = len(scope_df)


if "app_id" in scope_df.columns:

    total_games = (
        scope_df["app_id"]
        .nunique()
    )

else:
    total_games = 0


positive_percentage = (
    (
        scope_df["recommendation_polarity"]
        == "Positive"
    )
    .mean()
    * 100
)


if "review_length_words" in scope_df.columns:

    average_length = (
        scope_df["review_length_words"]
        .mean()
    )

else:
    average_length = 0


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
    "Average Review Length",
    f"{average_length:.1f} words"
)


st.divider()


# =========================================================
# CHART 1
# FEEDBACK THEME DISTRIBUTION BY GENRE
# =========================================================

st.header("1. What Feedback Do Players Give for Different Genres?")

st.write(
    """
    Select a game genre to explore the feedback themes most commonly
    discussed by players. This allows indie developers to understand
    the typical praise, complaints and expectations associated with
    games similar to the one they are developing.
    """
)


# ---------------------------------------------------------
# PREPARE GENRE DATA
# ---------------------------------------------------------

if "genre_list" in df.columns:

    genre_review_df = (
        df[
            [
                "genre_list",
                "main_theme"
            ]
        ]
        .explode("genre_list")
        .rename(
            columns={
                "genre_list": "genre"
            }
        )
    )


    # Clean genre values
    genre_review_df["genre"] = (
        genre_review_df["genre"]
        .astype(str)
        .str.strip()
    )


    # Remove empty genre values
    genre_review_df = genre_review_df[
        (genre_review_df["genre"] != "")
        & genre_review_df["main_theme"].notna()
    ]


    # ---------------------------------------------------------
    # REMOVE INDIE / NON-INDIE FROM GENRE FILTER
    # ---------------------------------------------------------

    excluded_genres = [
        "Indie",
        "Non-Indie",
        "Non Indie"
    ]


    genre_review_df = genre_review_df[
        ~genre_review_df["genre"].isin(
            excluded_genres
        )
    ]


    # ---------------------------------------------------------
    # GENRE SELECTOR
    # ---------------------------------------------------------

    genre_options = sorted(
        genre_review_df["genre"]
        .dropna()
        .unique()
    )


    selected_genre = st.selectbox(
        "🎮 Select a genre",
        genre_options
    )


    # ---------------------------------------------------------
    # FILTER REVIEWS TO SELECTED GENRE
    # ---------------------------------------------------------

    selected_genre_df = genre_review_df[
        genre_review_df["genre"]
        == selected_genre
    ].copy()


    # ---------------------------------------------------------
    # COUNT FEEDBACK THEMES
    # ---------------------------------------------------------

    theme_counts = (
        selected_genre_df
        .groupby("main_theme")
        .size()
        .reset_index(
            name="review_count"
        )
    )


    # Calculate percentage within selected genre
    theme_counts["percentage"] = (
        theme_counts["review_count"]
        / theme_counts["review_count"].sum()
        * 100
    )


    # Sort so largest bar appears at top
    theme_counts = (
        theme_counts
        .sort_values(
            "review_count",
            ascending=True
        )
    )


    # ---------------------------------------------------------
    # CHART
    # ---------------------------------------------------------

    fig_theme = px.bar(
        theme_counts,

        x="percentage",
        y="main_theme",

        orientation="h",

        text="percentage",

        title=(
            f"Player Feedback Theme Distribution "
            f"for {selected_genre} Games"
        ),

        labels={
            "percentage": "% of Reviews",
            "main_theme": "Feedback Theme"
        }
    )


    fig_theme.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )


    fig_theme.update_layout(
        height=550,
        yaxis={
            "categoryorder": "total ascending"
        }
    )


    st.plotly_chart(
        fig_theme,
        width="stretch"
    )


    # ---------------------------------------------------------
    # SIMPLE INSIGHT
    # ---------------------------------------------------------

    if not theme_counts.empty:

        top_theme_row = (
            theme_counts
            .sort_values(
                "percentage",
                ascending=False
            )
            .iloc[0]
        )


        top_theme = (
            top_theme_row["main_theme"]
        )

        top_percentage = (
            top_theme_row["percentage"]
        )


        st.info(
            f"""
            **Developer Insight:** For **{selected_genre}** games,
            **{top_theme}** is the most commonly identified feedback
            theme, accounting for approximately **{top_percentage:.1f}%**
            of classified feedback.

            Indie developers building a {selected_genre} game can use
            this as a reference for areas that players frequently pay
            attention to when reviewing similar games.
            """
        )


else:

    st.warning(
        "Genre information is not available in the review dataset."
    )


st.divider()
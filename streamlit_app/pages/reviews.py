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
            "primary_genre",   # ADD THIS
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

st.title("Player Review Analysis")

st.write(
    """
    Explore what players praise and complain about across
    different types of Steam games. These insights can help
    indie developers identify recurring player expectations,
    compare similar games, and prioritise areas for improvement.
    """
)


# =========================================================
# OVERVIEW METRICS
# =========================================================

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


# =========================================================
# DISPLAY OVERVIEW CARDS
# =========================================================

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
# =========================================================

st.header("1. What Feedback Themes Should Indie Developers Look Out For?")

st.write(
    """
    Select an indie game genre to see the specific feedback themes
    most commonly discussed by players. General positive and negative
    feedback are excluded so that the chart focuses on more actionable
    areas such as gameplay, technical issues, content, story and pricing.

    The percentages are recalculated after removing general feedback,
    so the remaining specific feedback themes add up to 100%.
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
):

    indie_genre_df = df[
        (df["game_type"] == "Indie")
        & (df["primary_genre"].isin(MODEL_GENRES))
        & (df["main_theme"].notna())
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
        "🎮 Select an indie genre",
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
    # COUNT SPECIFIC FEEDBACK THEMES
    # ---------------------------------------------------------

    theme_counts = (
        selected_genre_df
        .groupby("main_theme")
        .size()
        .reset_index(
            name="review_count"
        )
    )


    # ---------------------------------------------------------
    # RECALCULATE PERCENTAGES
    # GENERAL FEEDBACK HAS ALREADY BEEN REMOVED,
    # SO THESE THEMES ADD UP TO 100%
    # ---------------------------------------------------------

    specific_feedback_total = (
        theme_counts["review_count"].sum()
    )


    if specific_feedback_total > 0:

        theme_counts["percentage"] = (
            theme_counts["review_count"]
            / specific_feedback_total
            * 100
        )

    else:

        theme_counts["percentage"] = 0


    # Sort so largest bar appears at top
    theme_counts = (
        theme_counts
        .sort_values(
            "percentage",
            ascending=True
        )
    )


    # ---------------------------------------------------------
    # SHOW NUMBER OF SPECIFIC REVIEWS USED
    # ---------------------------------------------------------

    st.caption(
        f"Based on {specific_feedback_total:,} reviews containing "
        f"a specific feedback theme for Indie {selected_genre} games. "
        f"General positive and negative feedback are excluded."
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
            f"Key Feedback Themes for "
            f"Indie {selected_genre} Games"
        ),

        labels={
            "percentage":
                "% of Specific Feedback",
            "main_theme":
                "Feedback Theme"
        }
    )


    fig_theme.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )


    fig_theme.update_layout(
        height=500,

        yaxis={
            "categoryorder":
                "total ascending"
        },

        xaxis_title=(
            "% of Specific Feedback"
        ),

        yaxis_title=(
            "Feedback Theme"
        ),

        xaxis=dict(
            range=[0, max(
                theme_counts["percentage"].max() * 1.15,
                10
            )]
        )
    )


    st.plotly_chart(
        fig_theme,
        width="stretch"
    )


    # ---------------------------------------------------------
    # DEVELOPER INSIGHT
    # ---------------------------------------------------------

    if not theme_counts.empty:

        ranked_themes = (
            theme_counts
            .sort_values(
                "percentage",
                ascending=False
            )
            .reset_index(drop=True)
        )


        top_theme = (
            ranked_themes
            .iloc[0]["main_theme"]
        )

        top_percentage = (
            ranked_themes
            .iloc[0]["percentage"]
        )


        # Second most common theme
        if len(ranked_themes) > 1:

            second_theme = (
                ranked_themes
                .iloc[1]["main_theme"]
            )

            second_percentage = (
                ranked_themes
                .iloc[1]["percentage"]
            )


            st.info(
                f"""
                **Developer Insight:** Among specific feedback for
                **Indie {selected_genre} games**, the most frequently
                discussed area is **{top_theme}**
                (**{top_percentage:.1f}%**), followed by
                **{second_theme}**
                (**{second_percentage:.1f}%**).

                Since general praise and complaints are excluded,
                the percentages shown represent the distribution of
                actionable feedback themes and add up to **100%**.

                Indie developers creating a {selected_genre} game
                may want to pay particular attention to these areas
                when studying similar games and planning improvements.
                """
            )

        else:

            st.info(
                f"""
                **Developer Insight:** The most frequently discussed
                specific feedback area for **Indie {selected_genre}
                games** is **{top_theme}**, representing approximately
                **{top_percentage:.1f}%** of specific feedback.
                """
            )


else:

    st.warning(
        "The primary_genre or game_type column "
        "is not available."
    )


st.divider()
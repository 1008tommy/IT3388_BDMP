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
# INDIE GENRE FEEDBACK THEMES
# 100% POSITIVE VS NEGATIVE STACKED BAR
# =========================================================

st.header("What Feedback Themes Should Indie Developers Look Out For?")

st.write(
    """
    Select an indie game genre to see the specific feedback themes
    most commonly discussed by players. General positive and negative
    feedback are excluded so that the analysis focuses on more
    actionable areas.

    Each bar represents one feedback theme and adds up to 100%,
    showing the proportion of positive versus negative reviews
    within that theme.
    """
)


# ---------------------------------------------------------
# GENRES USED IN MODELLING
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
# CHECK REQUIRED COLUMNS
# ---------------------------------------------------------

required_chart_columns = [
    "primary_genre",
    "game_type",
    "main_theme",
    "recommendation_polarity"
]


if all(
    column in df.columns
    for column in required_chart_columns
):

    # ---------------------------------------------------------
    # KEEP ONLY INDIE REVIEWS FROM MODELLED GENRES
    # ---------------------------------------------------------

    indie_genre_df = df[
        (df["game_type"] == "Indie")
        & (df["primary_genre"].isin(MODEL_GENRES))
        & (df["main_theme"].notna())
        & (
            df["recommendation_polarity"].isin(
                ["Positive", "Negative"]
            )
        )
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
    ].copy()


    # ---------------------------------------------------------
    # AVAILABLE GENRES
    # ---------------------------------------------------------

    available_genres = [
        genre
        for genre in MODEL_GENRES
        if genre in indie_genre_df[
            "primary_genre"
        ].unique()
    ]


    if len(available_genres) > 0:

        # ---------------------------------------------------------
        # GENRE SELECTOR
        # ---------------------------------------------------------

        selected_genre = st.selectbox(
            "Select an indie genre",
            available_genres,
            key="chart1_genre"
        )


        # ---------------------------------------------------------
        # FILTER TO SELECTED GENRE
        # ---------------------------------------------------------

        selected_genre_df = indie_genre_df[
            indie_genre_df["primary_genre"]
            == selected_genre
        ].copy()


        # ---------------------------------------------------------
        # COUNT POSITIVE / NEGATIVE FOR EACH THEME
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
        # TOTAL REVIEWS WITHIN EACH THEME
        # ---------------------------------------------------------

        theme_sentiment_counts["theme_total"] = (
            theme_sentiment_counts
            .groupby("main_theme")[
                "review_count"
            ]
            .transform("sum")
        )


        # ---------------------------------------------------------
        # POSITIVE / NEGATIVE % WITHIN EACH THEME
        #
        # Used for the text shown INSIDE each bar segment.
        # Positive + Negative = 100% within each theme.
        # ---------------------------------------------------------

        theme_sentiment_counts[
            "within_theme_percentage"
        ] = (
            theme_sentiment_counts["review_count"]
            / theme_sentiment_counts["theme_total"]
            * 100
        )


        # ---------------------------------------------------------
        # TOTAL SPECIFIC FEEDBACK
        # ---------------------------------------------------------

        specific_feedback_total = len(
            selected_genre_df
        )


        # ---------------------------------------------------------
        # SHARE OF ALL SPECIFIC FEEDBACK
        #
        # This determines the ACTUAL LENGTH of each coloured
        # segment and therefore the total length of each bar.
        # ---------------------------------------------------------

        theme_sentiment_counts[
            "overall_segment_percentage"
        ] = (
            theme_sentiment_counts["review_count"]
            / specific_feedback_total
            * 100
        )


        # ---------------------------------------------------------
        # SUMMARY FOR EACH THEME
        # ---------------------------------------------------------

        theme_summary = (
            theme_sentiment_counts
            .groupby("main_theme")
            .agg(
                theme_total=(
                    "review_count",
                    "sum"
                )
            )
            .reset_index()
        )


        # Total length of each bar
        theme_summary["theme_share"] = (
            theme_summary["theme_total"]
            / specific_feedback_total
            * 100
        )


        # ---------------------------------------------------------
        # CREATE Y-AXIS LABEL
        #
        # Example:
        # Gameplay and balance (38.4%)
        # ---------------------------------------------------------

        theme_summary["theme_label"] = (
            theme_summary["main_theme"]
            + " ("
            + theme_summary["theme_share"].map(
                lambda x: f"{x:.1f}%"
            )
            + ")"
        )


        # ---------------------------------------------------------
        # MERGE THEME INFORMATION BACK
        # ---------------------------------------------------------

        theme_sentiment_counts = (
            theme_sentiment_counts
            .merge(
                theme_summary[
                    [
                        "main_theme",
                        "theme_share",
                        "theme_label"
                    ]
                ],
                on="main_theme",
                how="left"
            )
        )


        # ---------------------------------------------------------
        # SORT BY TOTAL THEME SHARE
        # Largest theme at bottom/top depending Plotly orientation
        # ---------------------------------------------------------

        theme_order = (
            theme_summary
            .sort_values(
                "theme_share",
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
            f"Bar length represents how common each theme is, while the "
            f"percentages inside each bar show the positive vs negative "
            f"split within that theme."
        )


        # ---------------------------------------------------------
        # STACKED BAR CHART
        # ---------------------------------------------------------

        fig_theme = px.bar(
            theme_sentiment_counts,

            # IMPORTANT:
            # This controls bar LENGTH
            x="overall_segment_percentage",

            y="theme_label",

            color="recommendation_polarity",

            orientation="h",

            barmode="stack",

            # IMPORTANT:
            # This controls the number shown inside
            text="within_theme_percentage",

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
                "overall_segment_percentage":
                    "% of Specific Feedback",

                "theme_label":
                    "Feedback Theme",

                "recommendation_polarity":
                    "Recommendation"
            },

            custom_data=[
                "review_count",
                "within_theme_percentage",
                "theme_share"
            ]
        )


        # ---------------------------------------------------------
        # CENTRE % INSIDE EACH GREEN / RED SECTION
        # ---------------------------------------------------------

        fig_theme.update_traces(

            texttemplate="%{text:.1f}%",

            textposition="inside",

            # Centre the percentage within its OWN segment
            insidetextanchor="middle",

            textfont=dict(
                color="white",
                size=13
            ),

            hovertemplate=(
                "<b>%{y}</b><br>"
                "Recommendation: %{fullData.name}<br>"
                "Reviews: %{customdata[0]:,}<br>"
                "Within this theme: "
                "%{customdata[1]:.1f}%<br>"
                "Theme share of all specific feedback: "
                "%{customdata[2]:.1f}%"
                "<extra></extra>"
            )
        )


        # ---------------------------------------------------------
        # X-AXIS RANGE
        #
        # Longest bar determines chart scale
        # ---------------------------------------------------------

        max_theme_share = (
            theme_summary["theme_share"].max()
        )


        # ---------------------------------------------------------
        # LAYOUT
        # ---------------------------------------------------------

        fig_theme.update_layout(

            height=560,

            barmode="stack",

            xaxis=dict(
                range=[
                    0,
                    max(
                        max_theme_share * 1.10,
                        10
                    )
                ],

                ticksuffix="%"
            ),

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

            uniformtext_minsize=10,
            uniformtext_mode="show"
        )


        st.plotly_chart(
            fig_theme,
            width="stretch"
        )


    else:

        st.warning(
            "No reviews were found for the selected modelling genres."
        )


else:

    st.warning(
        "One or more required columns are missing: "
        "primary_genre, game_type, main_theme, "
        "or recommendation_polarity."
    )


st.divider()

# =========================================================
# CHART 2
# INDIE VS NON-INDIE REVIEW CONSTRUCTIVENESS
# NUMBER OF FEEDBACK THEMES MENTIONED
# =========================================================

st.header("Do Indie Reviews Contain More Constructive Feedback?")

st.write(
    """
    As a supporting comparison, Indie and Non-Indie reviews are
    compared using the number of specific feedback themes identified
    in each review.

    Theme count is used as a proxy for constructiveness: reviews that
    discuss multiple aspects of a game may provide broader and more
    useful feedback than reviews containing little or no specific
    feedback.
    """
)


# ---------------------------------------------------------
# CHECK REQUIRED COLUMNS
# ---------------------------------------------------------

if (
    "game_type" in df.columns
    and "theme_count" in df.columns
):

    constructiveness_df = df[
        df["game_type"].isin(
            ["Indie", "Non-Indie"]
        )
        & df["theme_count"].notna()
    ].copy()


    # ---------------------------------------------------------
    # MAKE SURE THEME COUNT IS NUMERIC
    # ---------------------------------------------------------

    constructiveness_df["theme_count"] = (
        pd.to_numeric(
            constructiveness_df["theme_count"],
            errors="coerce"
        )
    )


    constructiveness_df = (
        constructiveness_df[
            constructiveness_df["theme_count"].notna()
        ]
        .copy()
    )


    constructiveness_df["theme_count"] = (
        constructiveness_df[
            "theme_count"
        ].astype(int)
    )


    # ---------------------------------------------------------
    # CREATE DISPLAY BUCKETS
    #
    # Keep 0, 1, 2 individually.
    # Combine 3+ because very high theme counts may be rare.
    # ---------------------------------------------------------

    constructiveness_df[
        "theme_count_group"
    ] = constructiveness_df[
        "theme_count"
    ].apply(
        lambda x: (
            "0"
            if x == 0
            else "1"
            if x == 1
            else "2"
            if x == 2
            else "3+"
        )
    )


    # ---------------------------------------------------------
    # COUNT REVIEWS
    # ---------------------------------------------------------

    theme_count_distribution = (
        constructiveness_df
        .groupby(
            [
                "game_type",
                "theme_count_group"
            ]
        )
        .size()
        .reset_index(
            name="review_count"
        )
    )


    # ---------------------------------------------------------
    # TOTAL REVIEWS FOR EACH GAME TYPE
    # ---------------------------------------------------------

    theme_count_distribution[
        "game_type_total"
    ] = (
        theme_count_distribution
        .groupby("game_type")[
            "review_count"
        ]
        .transform("sum")
    )


    # ---------------------------------------------------------
    # CONVERT TO %
    #
    # Each game type adds up to 100%.
    # This avoids Indie having an advantage just because
    # there may be more Indie reviews in the dataset.
    # ---------------------------------------------------------

    theme_count_distribution[
        "percentage"
    ] = (
        theme_count_distribution[
            "review_count"
        ]
        / theme_count_distribution[
            "game_type_total"
        ]
        * 100
    )


    # ---------------------------------------------------------
    # SUMMARY STATISTICS
    # ---------------------------------------------------------

    indie_counts = constructiveness_df[
        constructiveness_df["game_type"]
        == "Indie"
    ]["theme_count"]


    non_indie_counts = constructiveness_df[
        constructiveness_df["game_type"]
        == "Non-Indie"
    ]["theme_count"]


    indie_mean = indie_counts.mean()
    non_indie_mean = non_indie_counts.mean()

    indie_median = indie_counts.median()
    non_indie_median = non_indie_counts.median()


    # % REVIEWS WITH 2 OR MORE THEMES

    indie_multi = (
        (indie_counts >= 2).mean()
        * 100
    )

    non_indie_multi = (
        (non_indie_counts >= 2).mean()
        * 100
    )


    # ---------------------------------------------------------
    # SUMMARY CARDS
    # ---------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "Indie Avg. Themes",
        f"{indie_mean:.2f}"
    )


    col2.metric(
        "Non-Indie Avg. Themes",
        f"{non_indie_mean:.2f}"
    )


    col3.metric(
        "Indie Reviews with 2+ Themes",
        f"{indie_multi:.1f}%"
    )


    col4.metric(
        "Non-Indie Reviews with 2+ Themes",
        f"{non_indie_multi:.1f}%"
    )


    # ---------------------------------------------------------
    # CAPTION
    # ---------------------------------------------------------

    st.caption(
        "Percentages are calculated separately within Indie and "
        "Non-Indie reviews so that differences in dataset size do "
        "not affect the comparison."
    )


    # ---------------------------------------------------------
    # GROUPED BAR CHART
    # ---------------------------------------------------------

    fig_constructiveness = px.bar(
        theme_count_distribution,

        x="theme_count_group",
        y="percentage",

        color="game_type",

        barmode="group",

        text="percentage",

        category_orders={
            "theme_count_group": [
                "0",
                "1",
                "2",
                "3+"
            ],

            "game_type": [
                "Indie",
                "Non-Indie"
            ]
        },

        title=(
            "Distribution of Feedback Themes Mentioned "
            "in Indie vs Non-Indie Reviews"
        ),

        labels={
            "theme_count_group":
                "Number of Feedback Themes Mentioned",

            "percentage":
                "% of Reviews",

            "game_type":
                "Game Type"
        },

        custom_data=[
            "review_count"
        ]
    )


    # ---------------------------------------------------------
    # LABELS + HOVER
    # ---------------------------------------------------------

    fig_constructiveness.update_traces(

        texttemplate="%{text:.1f}%",

        textposition="outside",

        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            "Themes mentioned: %{x}<br>"
            "Reviews: %{customdata[0]:,}<br>"
            "Share of reviews: %{y:.1f}%"
            "<extra></extra>"
        )
    )


    # ---------------------------------------------------------
    # LAYOUT
    # ---------------------------------------------------------

    fig_constructiveness.update_layout(

        height=520,

        xaxis_title=(
            "Number of Feedback Themes Mentioned"
        ),

        yaxis_title=(
            "% of Reviews"
        ),

        yaxis=dict(
            ticksuffix="%"
        ),

        legend_title_text=(
            "Game Type"
        )
    )


    st.plotly_chart(
        fig_constructiveness,
        width="stretch"
    )


else:

    st.warning(
        "The game_type or theme_count column is not available."
    )


st.divider()
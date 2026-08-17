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
# OVERALL FEEDBACK THEME DISTRIBUTION
# =========================================================

st.header("1. What Are Players Talking About?")


theme_counts = (
    scope_df[
        scope_df["main_theme"].notna()
    ]
    .groupby("main_theme")
    .size()
    .reset_index(
        name="review_count"
    )
)


theme_counts["percentage"] = (
    theme_counts["review_count"]
    / theme_counts["review_count"].sum()
    * 100
)


theme_counts = theme_counts.sort_values(
    "review_count",
    ascending=True
)


fig_theme = px.bar(
    theme_counts,
    x="review_count",
    y="main_theme",
    orientation="h",
    text="percentage",
    title="Overall Distribution of Player Feedback Themes",
    labels={
        "review_count": "Number of Reviews",
        "main_theme": "Feedback Theme",
        "percentage": "Percentage"
    }
)


fig_theme.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside"
)


fig_theme.update_layout(
    height=550
)


st.plotly_chart(
    fig_theme,
    width="stretch"
)


st.info(
    """
    **Why this helps developers:** This identifies the areas players
    discuss most frequently. Developers can quickly see whether
    feedback is concentrated around gameplay, technical problems,
    content, visuals, story, or other areas.
    """
)


st.divider()


# =========================================================
# CHART 2
# POSITIVE VS NEGATIVE FEEDBACK BY THEME
# =========================================================

st.header("2. What Gets Praised and What Gets Criticised?")


sentiment_df = scope_df[
    scope_df["main_theme"].notna()
    & scope_df["recommendation_polarity"].notna()
].copy()


# Remove General feedback because its sentiment is already
# directly determined by recommendation status.
sentiment_df = sentiment_df[
    ~sentiment_df["main_theme"]
    .astype(str)
    .str.lower()
    .str.startswith("general")
]


sentiment_theme = (
    sentiment_df
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


theme_totals = (
    sentiment_theme
    .groupby("main_theme")["review_count"]
    .transform("sum")
)


sentiment_theme["percentage"] = (
    sentiment_theme["review_count"]
    / theme_totals
    * 100
)


fig_sentiment = px.bar(
    sentiment_theme,
    x="main_theme",
    y="percentage",
    color="recommendation_polarity",
    barmode="group",
    title="Positive vs Negative Reviews Within Each Feedback Theme",
    labels={
        "main_theme": "Feedback Theme",
        "percentage": "% of Reviews",
        "recommendation_polarity":
            "Recommendation"
    }
)


fig_sentiment.update_layout(
    xaxis_tickangle=-30,
    height=550
)


st.plotly_chart(
    fig_sentiment,
    width="stretch"
)


st.info(
    """
    **Why this helps developers:** A theme appearing frequently does
    not automatically mean it is a problem. This chart separates
    praise from criticism, helping developers identify which areas
    are strengths and which areas may need improvement.
    """
)


st.divider()


# =========================================================
# CHART 3
# GENRE VS FEEDBACK THEME
# =========================================================

st.header("3. Do Different Game Genres Receive Different Feedback?")


if "genre_list" in scope_df.columns:

    genre_theme_df = (
        scope_df[
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


    genre_theme_df["genre"] = (
        genre_theme_df["genre"]
        .astype(str)
        .str.strip()
    )


    genre_theme_df = genre_theme_df[
        (genre_theme_df["genre"] != "")
        & genre_theme_df["main_theme"].notna()
    ]


    # Top genres only so chart remains readable
    top_genres = (
        genre_theme_df["genre"]
        .value_counts()
        .head(10)
        .index
    )


    genre_theme_df = genre_theme_df[
        genre_theme_df["genre"]
        .isin(top_genres)
    ]


    # Exclude General feedback to focus on actionable themes
    specific_genre_theme_df = genre_theme_df[
        ~genre_theme_df["main_theme"]
        .astype(str)
        .str.lower()
        .str.startswith("general")
    ]


    genre_theme_counts = (
        specific_genre_theme_df
        .groupby(
            [
                "genre",
                "main_theme"
            ]
        )
        .size()
        .reset_index(
            name="review_count"
        )
    )


    genre_totals = (
        genre_theme_counts
        .groupby("genre")["review_count"]
        .transform("sum")
    )


    genre_theme_counts["percentage"] = (
        genre_theme_counts["review_count"]
        / genre_totals
        * 100
    )


    heatmap_df = (
        genre_theme_counts
        .pivot(
            index="genre",
            columns="main_theme",
            values="percentage"
        )
        .fillna(0)
    )


    fig_genre = px.imshow(
        heatmap_df,
        text_auto=".1f",
        aspect="auto",
        title="Feedback Theme Distribution by Genre",
        labels={
            "x": "Feedback Theme",
            "y": "Genre",
            "color": "% of Feedback"
        }
    )


    fig_genre.update_layout(
        height=650
    )


    st.plotly_chart(
        fig_genre,
        width="stretch"
    )


    st.info(
        """
        **Why this helps developers:** Developers can use similar
        genres as reference points. For example, if technical issues
        repeatedly appear in one type of game while content-related
        feedback dominates another, developers can prioritise
        different areas depending on the type of game they are
        building.
        """
    )

else:

    st.warning(
        "Genre data is not available for this chart."
    )


st.divider()


# =========================================================
# CHARTS 4 + 5
# INDIE VS NON-INDIE CONSTRUCTIVENESS
# =========================================================

st.header(
    "4. Do Indie Reviews Contain More Detailed Feedback?"
)


if (
    "game_type" in df.columns
    and "review_length_words" in df.columns
):

    constructiveness_df = (
        df[
            df["game_type"].notna()
        ]
        .groupby("game_type")
        .agg(
            average_review_length=(
                "review_length_words",
                "mean"
            ),
            review_count=(
                "game_type",
                "size"
            )
        )
        .reset_index()
    )


    col1, col2 = st.columns(2)


    # -----------------------------------------------------
    # CHART 4
    # REVIEW LENGTH
    # -----------------------------------------------------

    with col1:

        fig_length = px.bar(
            constructiveness_df,
            x="game_type",
            y="average_review_length",
            text_auto=".1f",
            title="Average Review Length",
            labels={
                "game_type": "Game Type",
                "average_review_length":
                    "Average Words per Review"
            }
        )


        fig_length.update_layout(
            height=450
        )


        st.plotly_chart(
            fig_length,
            width="stretch"
        )


    # -----------------------------------------------------
    # CHART 5
    # NUMBER OF THEMES
    # -----------------------------------------------------

    with col2:

        if "theme_count" in df.columns:

            theme_constructiveness = (
                df[
                    df["game_type"].notna()
                ]
                .groupby("game_type")
                .agg(
                    average_theme_count=(
                        "theme_count",
                        "mean"
                    )
                )
                .reset_index()
            )


            fig_theme_count = px.bar(
                theme_constructiveness,
                x="game_type",
                y="average_theme_count",
                text_auto=".2f",
                title="Average Number of Themes Mentioned",
                labels={
                    "game_type":
                        "Game Type",
                    "average_theme_count":
                        "Average Themes per Review"
                }
            )


            fig_theme_count.update_layout(
                height=450
            )


            st.plotly_chart(
                fig_theme_count,
                width="stretch"
            )


    st.info(
        """
        **Why this helps developers:** Longer reviews and reviews
        covering multiple feedback themes may contain more detailed
        information than recommendation labels alone. This provides
        a supporting comparison of how much information players give
        to indie and non-indie developers.
        """
    )

else:

    st.warning(
        "Game type or review length data is not available."
    )


st.divider()


# =========================================================
# CHART 6
# PRICE RANGE VS FEEDBACK THEME
# =========================================================

st.header(
    "5. Does Player Feedback Differ Across Game Prices?"
)


if "price" in scope_df.columns:

    price_df = scope_df[
        scope_df["price"].notna()
        & scope_df["main_theme"].notna()
    ].copy()


    price_df["price_range"] = pd.cut(
        price_df["price"],
        bins=[
            -0.01,
            0,
            5,
            10,
            20,
            30,
            50,
            float("inf")
        ],
        labels=[
            "Free",
            "$0–5",
            "$5–10",
            "$10–20",
            "$20–30",
            "$30–50",
            "$50+"
        ]
    )


    price_df = price_df[
        ~price_df["main_theme"]
        .astype(str)
        .str.lower()
        .str.startswith("general")
    ]


    price_theme = (
        price_df
        .groupby(
            [
                "price_range",
                "main_theme"
            ],
            observed=True
        )
        .size()
        .reset_index(
            name="review_count"
        )
    )


    price_totals = (
        price_theme
        .groupby("price_range")[
            "review_count"
        ]
        .transform("sum")
    )


    price_theme["percentage"] = (
        price_theme["review_count"]
        / price_totals
        * 100
    )


    price_heatmap = (
        price_theme
        .pivot(
            index="price_range",
            columns="main_theme",
            values="percentage"
        )
        .fillna(0)
    )


    fig_price = px.imshow(
        price_heatmap,
        text_auto=".1f",
        aspect="auto",
        title="Feedback Theme Distribution by Game Price Range",
        labels={
            "x": "Feedback Theme",
            "y": "Price Range",
            "color": "% of Feedback"
        }
    )


    fig_price.update_layout(
        height=550
    )


    st.plotly_chart(
        fig_price,
        width="stretch"
    )


    st.info(
        """
        **Why this helps developers:** Players may have different
        expectations at different price points. Understanding these
        patterns can help indie developers see which aspects of a
        game attract more scrutiny as the price increases.
        """
    )

else:

    st.warning(
        "Price data is not available for this chart."
    )


st.divider()


# =========================================================
# DEVELOPER EXPLORER
# =========================================================

st.header("🎮 Developer Reference Explorer")

st.write(
    """
    Select a genre to see the most common praise and complaint
    areas among similar games.
    """
)


if "genre_list" in df.columns:

    all_genre_df = (
        df.explode("genre_list")
        .rename(
            columns={
                "genre_list": "genre"
            }
        )
    )


    all_genre_df["genre"] = (
        all_genre_df["genre"]
        .astype(str)
        .str.strip()
    )


    genre_options = sorted(
        [
            genre
            for genre in all_genre_df[
                "genre"
            ].dropna().unique()
            if genre != ""
        ]
    )


    selected_genre = st.selectbox(
        "Select a genre",
        genre_options
    )


    selected_genre_df = all_genre_df[
        all_genre_df["genre"]
        == selected_genre
    ].copy()


    # -----------------------------------------------------
    # MOST COMMON COMPLAINT
    # -----------------------------------------------------

    negative_df = selected_genre_df[
        selected_genre_df[
            "recommendation_polarity"
        ]
        == "Negative"
    ]


    negative_specific = negative_df[
        ~negative_df["main_theme"]
        .astype(str)
        .str.lower()
        .str.startswith("general")
    ]


    complaint_counts = (
        negative_specific[
            "main_theme"
        ]
        .value_counts()
    )


    # -----------------------------------------------------
    # MOST COMMON PRAISE
    # -----------------------------------------------------

    positive_df = selected_genre_df[
        selected_genre_df[
            "recommendation_polarity"
        ]
        == "Positive"
    ]


    positive_specific = positive_df[
        ~positive_df["main_theme"]
        .astype(str)
        .str.lower()
        .str.startswith("general")
    ]


    praise_counts = (
        positive_specific[
            "main_theme"
        ]
        .value_counts()
    )


    insight1, insight2, insight3 = (
        st.columns(3)
    )


    if not complaint_counts.empty:

        top_complaint = (
            complaint_counts.index[0]
        )

        complaint_pct = (
            complaint_counts.iloc[0]
            / complaint_counts.sum()
            * 100
        )

        insight1.metric(
            "⚠️ Top Complaint Area",
            top_complaint,
            f"{complaint_pct:.1f}% of specific complaints"
        )


    if not praise_counts.empty:

        top_praise = (
            praise_counts.index[0]
        )

        praise_pct = (
            praise_counts.iloc[0]
            / praise_counts.sum()
            * 100
        )

        insight2.metric(
            "👍 Top Praise Area",
            top_praise,
            f"{praise_pct:.1f}% of specific praise"
        )


    genre_positive_pct = (
        (
            selected_genre_df[
                "recommendation_polarity"
            ]
            == "Positive"
        )
        .mean()
        * 100
    )


    insight3.metric(
        "⭐ Positive Reviews",
        f"{genre_positive_pct:.1f}%"
    )


    # -----------------------------------------------------
    # TOP COMPLAINTS CHART FOR SELECTED GENRE
    # -----------------------------------------------------

    st.subheader(
        f"Top Complaint Areas for {selected_genre} Games"
    )


    top_complaints = (
        complaint_counts
        .head(6)
        .reset_index()
    )


    top_complaints.columns = [
        "Feedback Theme",
        "Reviews"
    ]


    top_complaints = (
        top_complaints
        .sort_values(
            "Reviews",
            ascending=True
        )
    )


    fig_complaints = px.bar(
        top_complaints,
        x="Reviews",
        y="Feedback Theme",
        orientation="h",
        title=(
            f"Most Common Negative Feedback "
            f"for {selected_genre} Games"
        )
    )


    st.plotly_chart(
        fig_complaints,
        width="stretch"
    )


    # -----------------------------------------------------
    # REFERENCE GAMES
    # -----------------------------------------------------

    if (
        "name" in selected_genre_df.columns
        and "app_id" in selected_genre_df.columns
    ):

        st.subheader(
            f"Reference Games in {selected_genre}"
        )


        game_summary = (
            selected_genre_df
            .groupby(
                [
                    "app_id",
                    "name"
                ],
                dropna=False
            )
            .agg(
                reviews=(
                    "main_theme",
                    "size"
                ),
                positive_reviews=(
                    "recommendation_polarity",
                    lambda x:
                        (x == "Positive").sum()
                )
            )
            .reset_index()
        )


        game_summary[
            "Positive Review %"
        ] = (
            game_summary[
                "positive_reviews"
            ]
            / game_summary[
                "reviews"
            ]
            * 100
        )


        game_summary = (
            game_summary[
                game_summary["reviews"]
                >= 25
            ]
            .sort_values(
                "reviews",
                ascending=False
            )
            .head(10)
        )


        game_summary = game_summary[
            [
                "name",
                "reviews",
                "Positive Review %"
            ]
        ]


        game_summary.columns = [
            "Game",
            "Reviews Analysed",
            "Positive Review %"
        ]


        game_summary[
            "Positive Review %"
        ] = (
            game_summary[
                "Positive Review %"
            ]
            .round(1)
        )


        st.dataframe(
            game_summary,
            width="stretch",
            hide_index=True
        )


    st.success(
        f"""
        **How an indie developer can use this:** If you are
        developing a **{selected_genre}** game, the dashboard lets
        you examine feedback from existing games in the same genre.
        The recurring complaint areas highlight potential risks to
        address, while the recurring praise areas show features that
        players tend to value.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Steam Indie Game Review Analytics • "
    "Feedback themes classified from Steam player reviews"
)
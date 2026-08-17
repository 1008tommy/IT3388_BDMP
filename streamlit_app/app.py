import streamlit as st


st.set_page_config(
    page_title="Steam Indie Game Analytics",
    page_icon="🎮",
    layout="wide"
)


# Define the three project pages
pricing_page = st.Page(
    "pages/pricing.py",
    title="Pricing Strategy",
)

engagement_page = st.Page(
    "pages/engagement.py",
    title="Player Engagement",
)

reviews_page = st.Page(
    "pages/reviews.py",
    title="Player Reviews",
)


# Navigation
page = st.navigation(
    [
        pricing_page,
        engagement_page,
        reviews_page
    ],
    position="sidebar"
)


page.run()
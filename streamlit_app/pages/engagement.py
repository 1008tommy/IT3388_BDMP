import streamlit as st
import mlflow
import pandas as pd
import numpy as np
from datetime import date, datetime

# =========================================================
# CONFIGURATION
# =========================================================

mlflow.set_tracking_uri("databricks")
MLFLOW_EXPERIMENT = "/Users/242475r@mymail.nyp.edu.sg/IT3388_BDMP/game_performance_models"
mlflow.set_experiment(MLFLOW_EXPERIMENT)

# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
    
    runs_ccu = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName LIKE 'Random Forest - Peak CCU%'",
        max_results=1
    )
    
    runs_playtime = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName LIKE 'XGBoost - Average Playtime%'",
        max_results=1
    )
    
    if len(runs_ccu) > 0 and len(runs_playtime) > 0:
        model_ccu = mlflow.sklearn.load_model(f"runs:/{runs_ccu.iloc[0]['run_id']}/model")
        model_playtime = mlflow.sklearn.load_model(f"runs:/{runs_playtime.iloc[0]['run_id']}/model")
        ccu_r2 = runs_ccu.iloc[0]['metrics.test_r2']
        playtime_r2 = runs_playtime.iloc[0]['metrics.test_r2']
        return model_ccu, model_playtime, ccu_r2, playtime_r2
    return None, None, None, None

model_ccu, model_playtime, ccu_r2, playtime_r2 = load_models()

# =========================================================
# CATEGORIES AND GENRES
# =========================================================

CATEGORIES = [
    'Single_player', 'Multi_player', 'Co_op', 'Online_Co_op', 'Online_PvP', 'PvP',
    'Steam_Achievements', 'Steam_Cloud', 'Steam_Trading_Cards', 'Steam_Workshop',
    'Full_controller_support', 'Partial_Controller_Support', 'Steam_Leaderboards',
    'Stats', 'Remote_Play_Together', 'Remote_Play_on_Phone', 'Remote_Play_on_TV',
    'Remote_Play_on_Tablet', 'Captions_available', 'Commentary_available',
    'Includes_level_editor', 'VR_Support', 'VR_Supported', 'VR_Only',
    'Family_Sharing', 'In_App_Purchases', 'MMO', 'Mods', 'LAN_Co_op', 'LAN_PvP',
    'Shared_Split_Screen', 'Shared_Split_Screen_Co_op', 'Shared_Split_Screen_PvP',
    'Cross_Platform_Multiplayer', 'HDR_available', 'Steam_Input_API_Support',
    'Steam_Timeline', 'Steam_Turn_Notifications', 'Valve_Anti_Cheat_enabled',
    'DualSense_Controller_Support', 'DualShock_Controller_Support', 'Gamepad_Recommended',
    'Keyboard_Only_Option', 'Mouse_Only_Option', 'Touch_Only_Option',
    'Tracked_Controller_Support', 'Includes_Source_SDK', 'Mods_require_HL2',
    'SteamVR_Collectibles', 'Adjustable_Difficulty', 'Adjustable_Text_Size',
    'Camera_Comfort', 'Chat_Speech_to_text', 'Chat_Text_to_speech',
    'Color_Alternatives', 'Custom_Volume_Controls', 'Narrated_Game_Menus',
    'Playable_without_Timed_Input', 'Save_Anytime', 'Stereo_Sound', 'Subtitle_Options',
    'Surround_Sound'
]

GENRES = [
    'Action', 'Adventure', 'Casual', 'Indie', 'RPG', 'Simulation', 'Strategy',
    'Sports', 'Racing', 'Massively_Multiplayer', 'Free_To_Play', 'Early_Access',
    'Animation_and_Modeling', 'Audio_Production', 'Design_and_Illustration',
    'Education', 'Game_Development', 'Photo_Editing', 'Software_Training',
    'Utilities', 'Video_Production', 'Web_Publishing', 'Accounting', 'Documentary',
    'Episodic', 'Gore', 'Movie', 'Nudity', 'Sexual_Content', 'Short', 'Tutorial',
    'Violent', '360_Video'
]

# =========================================================
# UI
# =========================================================

st.title("🎮 Steam Game Performance Predictor")

if model_ccu and model_playtime:
    st.success("✅ Models loaded!")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Peak CCU Model", "Random Forest", f"R² = {ccu_r2:.4f}")
    with col2:
        st.metric("Playtime Model", "XGBoost", f"R² = {playtime_r2:.4f}")
    
    st.divider()
    
    # Basic Features
    st.subheader("📊 Basic Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        price = st.number_input("Price ($)", 0.0, 500.0, 19.99, 0.99)
        dlc_count = st.number_input("DLC Count", 0, 100, 0)
    with col2:
        achievements = st.number_input("Achievements", 0, 1000, 50)
        release_date = st.date_input("Release Date", date.today())
    with col3:
        st.write("**Platforms**")
        windows = st.checkbox("Windows", True)
        mac = st.checkbox("Mac", False)
        linux = st.checkbox("Linux", False)
    
    st.divider()
    
    # Categories
    st.subheader("🎯 Categories")
    selected_categories = []
    col1, col2, col3 = st.columns(3)
    for i, cat in enumerate(sorted(CATEGORIES)):
        with [col1, col2, col3][i % 3]:
            if st.checkbox(cat.replace('_', ' '), key=f"cat_{cat}"):
                selected_categories.append(cat)
    
    st.divider()
    
    # Genres
    st.subheader("🎨 Genres")
    selected_genres = []
    col1, col2, col3 = st.columns(3)
    for i, genre in enumerate(sorted(GENRES)):
        with [col1, col2, col3][i % 3]:
            if st.checkbox(genre.replace('_', ' '), key=f"gen_{genre}"):
                selected_genres.append(genre)
    
    st.divider()
    
    # Advanced Features
    with st.expander("🔧 Advanced Features (Optional)"):
        col1, col2 = st.columns(2)
        with col1:
            supported_languages_count = st.number_input("Supported Languages Count", 0, 120, 10)
            audio_languages_count = st.number_input("Full Audio Languages Count", 0, 120, 5)
        with col2:
            tag_count = st.number_input("Tag Count", 0, 100, 10)
            total_tag_votes = st.number_input("Total Tag Votes", 0, 100000, 1000)
    
    st.divider()
    
    # PREDICT
    if st.button("🔮 Predict Performance", type="primary"):
        try:
            # Calculate engineered features
            platform_count = int(windows) + int(mac) + int(linux)
            category_count = len(selected_categories)
            genre_count = len(selected_genres)
            
            # Date features
            days_since_release = (date.today() - release_date).days
            release_year = release_date.year
            release_month = release_date.month
            release_day_of_week = release_date.isoweekday()
            release_quarter = (release_month - 1) // 3 + 1
            year_bucket_2000s = int(2000 <= release_year < 2010)
            year_bucket_2010s = int(2010 <= release_year < 2020)
            year_bucket_2020s = int(release_year >= 2020)
            
            # Build complete feature set
            input_data = {
                'price': price,
                'dlc_count': dlc_count,
                'windows': int(windows),
                'mac': int(mac),
                'linux': int(linux),
                'achievements': achievements,
                'platform_count': platform_count,
                'supported_languages_count': supported_languages_count,
                'audio_languages_count': audio_languages_count,
                'category_count': category_count,
                'genre_count': genre_count,
                'tag_count': tag_count,
                'total_tag_votes': total_tag_votes,
                'tag_diversity': tag_count,
                'release_year': release_year,
                'release_month': release_month,
                'release_day_of_week': release_day_of_week,
                'release_quarter': release_quarter,
                'days_since_release': days_since_release,
                'year_bucket_2000s': year_bucket_2000s,
                'year_bucket_2010s': year_bucket_2010s,
                'year_bucket_2020s': year_bucket_2020s
            }
            
            # Add category features
            for cat in CATEGORIES:
                input_data[f'categories_{cat}'] = 1 if cat in selected_categories else 0
            
            # Add genre features
            for genre in GENRES:
                input_data[f'genres_{genre}'] = 1 if genre in selected_genres else 0
            
            # Add all language features (set to 0 - we don't have UI for all 120+ languages)
            # You can expand this if needed
            for lang_type in ['supported_languages', 'full_audio_languages']:
                for lang in ['English', 'French', 'German', 'Spanish_Spain', 'Simplified_Chinese', 
                            'Traditional_Chinese', 'Japanese', 'Korean', 'Russian', 'Portuguese_Brazil']:
                    input_data[f'{lang_type}_{lang}'] = 0
            
            input_df = pd.DataFrame([input_data])
            
            # Predict
            pred_ccu_log = model_ccu.predict(input_df)[0]
            pred_playtime_log = model_playtime.predict(input_df)[0]
            
            pred_ccu = np.expm1(pred_ccu_log)
            pred_playtime = np.expm1(pred_playtime_log)
            
            st.success("✅ Prediction Complete!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Peak CCU", f"{int(pred_ccu):,}")
            with col2:
                st.metric("Predicted Avg Playtime", f"{int(pred_playtime):,} mins")
            
            with st.expander("📋 Summary"):
                st.write(f"**Categories:** {', '.join([c.replace('_', ' ') for c in selected_categories]) if selected_categories else 'None'}")
                st.write(f"**Genres:** {', '.join([g.replace('_', ' ') for g in selected_genres]) if selected_genres else 'None'}")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.write("Missing features:", [f for f in model_ccu.feature_names_in_ if f not in input_df.columns])
else:
    st.error("❌ Failed to load models")
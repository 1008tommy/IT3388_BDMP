import streamlit as st
import mlflow
import pandas as pd
import numpy as np
from datetime import date

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
# ALL FEATURES LISTS
# =========================================================

CATEGORIES = ['Adjustable_Difficulty', 'Adjustable_Text_Size', 'Camera_Comfort', 'Captions_available',
    'Chat_Speech_to_text', 'Chat_Text_to_speech', 'Co_op', 'Color_Alternatives', 'Commentary_available',
    'Cross_Platform_Multiplayer', 'Custom_Volume_Controls', 'DualSense_Controller_Support',
    'DualShock_Controller_Support', 'Family_Sharing', 'Full_controller_support', 'Gamepad_Recommended',
    'HDR_available', 'In_App_Purchases', 'Includes_Source_SDK', 'Includes_level_editor',
    'Keyboard_Only_Option', 'LAN_Co_op', 'LAN_PvP', 'MMO', 'Mods', 'Mods_require_HL2',
    'Mouse_Only_Option', 'Multi_player', 'Narrated_Game_Menus', 'Online_Co_op', 'Online_PvP',
    'Partial_Controller_Support', 'Playable_without_Timed_Input', 'PvP', 'Remote_Play_Together',
    'Remote_Play_on_Phone', 'Remote_Play_on_TV', 'Remote_Play_on_Tablet', 'Save_Anytime',
    'Shared_Split_Screen', 'Shared_Split_Screen_Co_op', 'Shared_Split_Screen_PvP', 'Single_player',
    'Stats', 'Steam_Achievements', 'Steam_Cloud', 'Steam_Input_API_Support', 'Steam_Leaderboards',
    'Steam_Timeline', 'Steam_Trading_Cards', 'Steam_Turn_Notifications', 'Steam_Workshop',
    'SteamVR_Collectibles', 'Stereo_Sound', 'Subtitle_Options', 'Surround_Sound', 'Touch_Only_Option',
    'Tracked_Controller_Support', 'VR_Only', 'VR_Support', 'VR_Supported', 'Valve_Anti_Cheat_enabled']

GENRES = ['360_Video', 'Accounting', 'Action', 'Adventure', 'Animation_and_Modeling', 'Audio_Production',
    'Casual', 'Design_and_Illustration', 'Documentary', 'Early_Access', 'Education', 'Episodic',
    'Free_To_Play', 'Game_Development', 'Gore', 'Indie', 'Massively_Multiplayer', 'Movie', 'Nudity',
    'Photo_Editing', 'RPG', 'Racing', 'Sexual_Content', 'Short', 'Simulation', 'Software_Training',
    'Sports', 'Strategy', 'Tutorial', 'Utilities', 'Video_Production', 'Violent', 'Web_Publishing']

SUPPORTED_LANGUAGES = ['Afrikaans', 'Albanian', 'Amharic', 'Arabic', 'Armenian', 'Assamese', 'Azerbaijani',
    'Bangla', 'Basque', 'Belarusian', 'Bosnian', 'Bulgarian', 'Catalan', 'Cherokee', 'Croatian', 'Czech',
    'Danish', 'Dari', 'Dutch', 'English', 'Estonian', 'Filipino', 'Finnish', 'French', 'Galician',
    'Georgian', 'German', 'Greek', 'Gujarati', 'Hausa', 'Hebrew', 'Hindi', 'Hungarian', 'Icelandic',
    'Igbo', 'Indonesian', 'Irish', 'Italian', 'Japanese', 'Kannada', 'Kazakh', 'Khmer', 'Kiche',
    'Kinyarwanda', 'Konkani', 'Korean', 'Kyrgyz', 'Latvian', 'Lithuanian', 'Luxembourgish', 'Macedonian',
    'Malay', 'Malayalam', 'Maltese', 'Maori', 'Marathi', 'Mongolian', 'Nepali', 'Norwegian', 'Odia',
    'Persian', 'Polish', 'Portuguese_Brazil', 'Portuguese_Portugal', 'Punjabi', 'Quechua', 'Romanian',
    'Russian', 'Scots', 'Serbian', 'Simplified_Chinese', 'Sindhi', 'Sinhala', 'Slovak', 'Slovenian',
    'Sorani', 'Sotho', 'Spanish_Latin_America', 'Spanish_Spain', 'Swahili', 'Swedish', 'Tajik', 'Tamil',
    'Tatar', 'Telugu', 'Thai', 'Tigrinya', 'Traditional_Chinese', 'Tswana', 'Turkish', 'Turkmen',
    'Ukrainian', 'Urdu', 'Uyghur', 'Uzbek', 'Valencian', 'Vietnamese', 'Welsh', 'Wolof', 'Xhosa',
    'Yoruba', 'Zulu']

AUDIO_LANGUAGES = ['Afrikaans', 'Albanian', 'Amharic', 'Arabic', 'Armenian', 'Assamese', 'Azerbaijani',
    'Bangla', 'Basque', 'Belarusian', 'Bosnian', 'Bulgarian', 'Catalan', 'Cherokee', 'Croatian', 'Czech',
    'Danish', 'Dari', 'Dutch', 'English', 'Estonian', 'Filipino', 'Finnish', 'French', 'Galician',
    'Georgian', 'German', 'Greek', 'Gujarati', 'Hausa', 'Hebrew', 'Hindi', 'Hungarian', 'Icelandic',
    'Igbo', 'Indonesian', 'Irish', 'Italian', 'Japanese', 'Kannada', 'Kazakh', 'Khmer', 'Kiche',
    'Kinyarwanda', 'Konkani', 'Korean', 'Kyrgyz', 'Latvian', 'Lithuanian', 'Luxembourgish', 'Macedonian',
    'Malay', 'Malayalam', 'Maltese', 'Maori', 'Marathi', 'Mongolian', 'Nepali', 'Norwegian', 'Odia',
    'Persian', 'Polish', 'Portuguese_Brazil', 'Portuguese_Portugal', 'Punjabi', 'Quechua', 'Romanian',
    'Russian', 'Scots', 'Serbian', 'Simplified_Chinese', 'Sindhi', 'Sinhala', 'Slovak', 'Slovenian',
    'Sorani', 'Sotho', 'Spanish_Latin_America', 'Spanish_Spain', 'Swahili', 'Swedish', 'Tajik', 'Tamil',
    'Tatar', 'Telugu', 'Thai', 'Tigrinya', 'Traditional_Chinese', 'Tswana', 'Turkish', 'Turkmen',
    'Ukrainian', 'Urdu', 'Uyghur', 'Uzbek', 'Valencian', 'Vietnamese', 'Welsh', 'Wolof', 'Xhosa',
    'Yoruba', 'Zulu']

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
    
    # =========================================================
    # BASIC FEATURES
    # =========================================================
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
    
    # =========================================================
    # CATEGORIES (MULTISELECT)
    # =========================================================
    st.subheader("🎯 Categories")
    selected_categories = st.multiselect(
        "Select categories:",
        options=[c.replace('_', ' ') for c in CATEGORIES],
        default=['Single player', 'Steam Achievements']
    )
    # Convert back to underscore format
    selected_categories = [c.replace(' ', '_') for c in selected_categories]
    
    st.divider()
    
    # =========================================================
    # GENRES (MULTISELECT)
    # =========================================================
    st.subheader("🎨 Genres")
    selected_genres = st.multiselect(
        "Select genres:",
        options=[g.replace('_', ' ') for g in GENRES],
        default=['Action', 'Indie']
    )
    # Convert back to underscore format
    selected_genres = [g.replace(' ', '_') for g in selected_genres]
    
    st.divider()
    
    # =========================================================
    # LANGUAGES (MULTISELECT)
    # =========================================================
    st.subheader("🌍 Languages")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Supported Languages (Subtitles)**")
        selected_supported_langs = st.multiselect(
            "Select supported languages:",
            options=[l.replace('_', ' ') for l in SUPPORTED_LANGUAGES],
            default=['English'],
            key="supported"
        )
        selected_supported_langs = [l.replace(' ', '_') for l in selected_supported_langs]
        st.info(f"Count: {len(selected_supported_langs)}")
    
    with col2:
        st.write("**Full Audio Languages**")
        selected_audio_langs = st.multiselect(
            "Select audio languages:",
            options=[l.replace('_', ' ') for l in AUDIO_LANGUAGES],
            default=['English'],
            key="audio"
        )
        selected_audio_langs = [l.replace(' ', '_') for l in selected_audio_langs]
        st.info(f"Count: {len(selected_audio_langs)}")
    
    st.divider()
    
    # =========================================================
    # TAGS (DYNAMIC INPUT)
    # =========================================================
    st.subheader("🏷️ Tags")
    st.write("Add tags and their vote counts:")
    
    # Initialize session state for tags
    if 'tags' not in st.session_state:
        st.session_state.tags = {}
    
    col1, col2, col3 = st.columns([3, 2, 1])
    with col1:
        tag_name = st.text_input("Tag name:", placeholder="e.g., Singleplayer")
    with col2:
        tag_votes = st.number_input("Vote count:", 0, 100000, 100, key="tag_votes_input")
    with col3:
        st.write("")
        st.write("")
        if st.button("➕ Add Tag"):
            if tag_name:
                st.session_state.tags[tag_name] = tag_votes
    
    # Display current tags
    if st.session_state.tags:
        st.write("**Current Tags:**")
        for tag, votes in st.session_state.tags.items():
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.write(tag)
            with col2:
                st.write(f"{votes:,} votes")
            with col3:
                if st.button("🗑️", key=f"remove_{tag}"):
                    del st.session_state.tags[tag]
                    st.rerun()
        
        # Auto-calculated metrics
        tag_count = len(st.session_state.tags)
        total_tag_votes = sum(st.session_state.tags.values())
        st.info(f"**Tag Count:** {tag_count} | **Total Votes:** {total_tag_votes:,}")
    else:
        tag_count = 0
        total_tag_votes = 0
        st.info("No tags added (defaults to 0)")
    
    st.divider()
    
    # =========================================================
    # PREDICT BUTTON
    # =========================================================
    if st.button("🔮 Predict Performance", type="primary"):
        try:
            # Calculate engineered features
            platform_count = int(windows) + int(mac) + int(linux)
            category_count = len(selected_categories)
            genre_count = len(selected_genres)
            supported_languages_count = len(selected_supported_langs)
            audio_languages_count = len(selected_audio_langs)
            
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
            
            # Add supported language features
            for lang in SUPPORTED_LANGUAGES:
                input_data[f'supported_languages_{lang}'] = 1 if lang in selected_supported_langs else 0
            
            # Add audio language features
            for lang in AUDIO_LANGUAGES:
                input_data[f'full_audio_languages_{lang}'] = 1 if lang in selected_audio_langs else 0
            
            input_df = pd.DataFrame([input_data])
            
            # Predict
            pred_ccu_log = model_ccu.predict(input_df)[0]
            pred_playtime_log = model_playtime.predict(input_df)[0]
            
            pred_ccu = np.expm1(pred_ccu_log)
            pred_playtime = np.expm1(pred_playtime_log)
            
            st.success("✅ Prediction Complete!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Peak CCU", f"{int(pred_ccu):,}", help="Maximum concurrent users at peak")
            with col2:
                st.metric("Predicted Avg Playtime", f"{int(pred_playtime):,} mins", help="Average minutes played per user")
            
            # Summary
            with st.expander("📋 Input Summary"):
                st.write(f"**Basic:** ${price} | {dlc_count} DLCs | {achievements} achievements")
                st.write(f"**Platforms:** {platform_count} ({', '.join([p for p, v in [('Windows', windows), ('Mac', mac), ('Linux', linux)] if v])})")
                st.write(f"**Categories ({category_count}):** {', '.join([c.replace('_', ' ') for c in selected_categories])}")
                st.write(f"**Genres ({genre_count}):** {', '.join([g.replace('_', ' ') for g in selected_genres])}")
                st.write(f"**Languages:** {supported_languages_count} supported, {audio_languages_count} audio")
                st.write(f"**Tags:** {tag_count} tags, {total_tag_votes:,} total votes")
                st.write(f"**Release:** {release_date} ({days_since_release} days ago)")
                
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.exception(e)
else:
    st.error("❌ Failed to load models")

st.divider()
st.caption("Models trained on Steam game metadata | Random Forest (CCU) + XGBoost (Playtime)")
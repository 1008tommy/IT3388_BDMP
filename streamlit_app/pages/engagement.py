import streamlit as st
import mlflow
import pandas as pd
import numpy as np
from datetime import date
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# CONFIGURATION
# =========================================================

mlflow.set_tracking_uri("databricks")
MLFLOW_EXPERIMENT = "/Users/darrenchor1832@gmail.com/game_performance_models"
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
        
        # Get feature importance
        feature_importance_ccu = None
        feature_importance_playtime = None
        
        if hasattr(model_ccu, 'feature_importances_'):
            feature_importance_ccu = model_ccu.feature_importances_
        
        if hasattr(model_playtime, 'feature_importances_'):
            feature_importance_playtime = model_playtime.feature_importances_
        
        return model_ccu, model_playtime, ccu_r2, playtime_r2, feature_importance_ccu, feature_importance_playtime
    return None, None, None, None, None, None

model_ccu, model_playtime, ccu_r2, playtime_r2, feat_imp_ccu, feat_imp_playtime = load_models()

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

TAGS = ['1980s', '1990s', '2.5D', '2D', '2D_Fighter', '3D', '3D_Platformer', '4X', 'Action',
        'Action_Adventure', 'Action_RPG', 'Adventure', 'Anime', 'Arcade', 'Atmospheric',
        'Base_Building', 'Battle_Royale', 'Beautiful', 'Building', 'Card_Game', 'Casual',
        'Character_Customization', 'Classic', 'Co_op', 'Colorful', 'Comedy', 'Competitive',
        'Cooperative', 'Crafting', 'Dark', 'Dark_Fantasy', 'Difficult', 'Dungeon_Crawler',
        'Exploration', 'FPS', 'Fantasy', 'Fast_Paced', 'Fighting', 'First_Person', 'Funny',
        'Gore', 'Great_Soundtrack', 'Hack_and_Slash', 'Horror', 'Indie', 'JRPG', 'Loot',
        'Mature', 'Medieval', 'Memes', 'Metroidvania', 'MOBA', 'Multiplayer', 'Mystery',
        'Naval', 'Ninja', 'Open_World', 'Party_Based', 'Pixel_Graphics', 'Platformer',
        'Point_and_Click', 'Psychological_Horror', 'Puzzle', 'Racing', 'Retro', 'Rogue_like',
        'Rogue_lite', 'RPG', 'Sandbox', 'Sci_fi', 'Shooter', 'Simulation', 'Singleplayer',
        'Souls_like', 'Space', 'Sports', 'Strategy', 'Survival', 'Tactical', 'Third_Person',
        'Thriller', 'Tower_Defense', 'Turn_Based', 'Turn_Based_Strategy', 'Twin_Stick_Shooter',
        'Visual_Novel', 'VR', 'Walking_Simulator', 'War', 'Zombies']

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_feature_importance_dataframe(feature_importances, feature_names):
    """Convert feature importance to DataFrame"""
    if feature_importances is None:
        return None
    
    if len(feature_importances) == len(feature_names):
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importances
        })
        df = df.sort_values('importance', ascending=True)
        return df
    return None

def plot_feature_importance(df, title, top_n=20):
    """Create a plotly bar chart for feature importance"""
    if df is None or len(df) == 0:
        return None
    
    df_top = df.tail(top_n)
    
    fig = px.bar(df_top, 
                 x='importance', 
                 y='feature',
                 orientation='h',
                 title=title,
                 color='importance',
                 color_continuous_scale='Viridis',
                 height=600)
    
    fig.update_layout(
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    return fig

def get_feature_type(feature_name):
    """Determine the type of feature"""
    if feature_name.startswith('tags_'):
        return 'Tag'
    elif feature_name.startswith('categories_'):
        return 'Category'
    elif feature_name.startswith('genres_'):
        return 'Genre'
    elif feature_name.startswith('supported_languages_'):
        return 'Supported Language'
    elif feature_name.startswith('full_audio_languages_'):
        return 'Audio Language'
    else:
        return 'Basic Feature'

def format_feature_name(feature_name):
    """Format feature name for display"""
    # Remove prefix
    if feature_name.startswith('tags_'):
        return feature_name.replace('tags_', '').replace('_', ' ')
    elif feature_name.startswith('categories_'):
        return feature_name.replace('categories_', '').replace('_', ' ')
    elif feature_name.startswith('genres_'):
        return feature_name.replace('genres_', '').replace('_', ' ')
    elif feature_name.startswith('supported_languages_'):
        return feature_name.replace('supported_languages_', '').replace('_', ' ')
    elif feature_name.startswith('full_audio_languages_'):
        return feature_name.replace('full_audio_languages_', '').replace('_', ' ')
    else:
        return feature_name.replace('_', ' ').title()

def get_top_features_to_add(model, input_data, expected_features, feature_list, current_values, pred_log, model_type='ccu'):
    """
    Get the top features to add based on model impact analysis.
    Tests adding each missing feature one at a time.
    """
    feature_impacts = {}
    base_pred = np.expm1(pred_log) if pred_log else pred_log
    
    # Get current feature values
    current_features = [f for f, v in current_values.items() if v > 0]
    
    # Test each feature that's not already present
    for feature in feature_list:
        if feature in current_features:
            continue
        
        # Create a copy of the input data
        test_data = input_data.copy()
        
        # Add the feature with value 1 (or appropriate value)
        if feature.startswith('supported_languages_') or feature.startswith('full_audio_languages_'):
            # For language features, add with value 1
            test_data[feature] = 1
        elif feature.startswith('tags_'):
            # For tags, add with 1000 votes
            test_data[feature] = 1000
        elif feature.startswith('categories_') or feature.startswith('genres_'):
            # For categories and genres, add with value 1
            test_data[feature] = 1
        else:
            # For other features, add with value 1
            test_data[feature] = 1
        
        # Create DataFrame with correct feature order
        test_df = pd.DataFrame([test_data])[expected_features]
        
        # Predict with this feature
        try:
            new_pred_log = model.predict(test_df)[0]
            new_pred = np.expm1(new_pred_log) if pred_log else new_pred_log
            
            # Calculate absolute and percentage change
            abs_change = new_pred - base_pred
            pct_change = (abs_change / base_pred) * 100 if base_pred != 0 else 0
            
            feature_impacts[feature] = {
                'abs_change': abs_change,
                'pct_change': pct_change,
                'new_value': new_pred,
                'feature_type': get_feature_type(feature),
                'display_name': format_feature_name(feature)
            }
        except:
            continue
    
    # Sort by absolute change
    sorted_features = sorted(feature_impacts.items(), key=lambda x: x[1]['abs_change'], reverse=True)
    return sorted_features[:10]  # Return top 10

def generate_feature_recommendations(feature_impacts, current_ccu, current_playtime):
    """Generate formatted recommendations from feature impacts"""
    recommendations = []
    
    for feature, impact in feature_impacts:
        if impact['abs_change'] > 0:  # Only recommend if it improves
            feature_type = impact['feature_type']
            display_name = impact['display_name']
            
            if feature_type == 'Supported Language':
                rec = {
                    'type': feature_type,
                    'feature': display_name,
                    'impact': impact['abs_change'],
                    'pct_change': impact['pct_change'],
                    'suggestion': f"Add {feature_type} support for '{display_name}'"
                }
            elif feature_type == 'Audio Language':
                rec = {
                    'type': feature_type,
                    'feature': display_name,
                    'impact': impact['abs_change'],
                    'pct_change': impact['pct_change'],
                    'suggestion': f"Add {feature_type} support for '{display_name}'"
                }
            else:
                rec = {
                    'type': feature_type,
                    'feature': display_name,
                    'impact': impact['abs_change'],
                    'pct_change': impact['pct_change'],
                    'suggestion': f"Add {feature_type} '{display_name}'"
                }
            
            # Determine which metric it improves
            if impact['abs_change'] > 0:
                rec['improves'] = 'Both'  # We'll refine this later
                recommendations.append(rec)
    
    return recommendations

# =========================================================
# UI
# =========================================================

st.title("Steam Game Performance Predictor")

if model_ccu and model_playtime:
    st.success("Models loaded successfully!")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Peak CCU Model", "Random Forest", f"R2 = {ccu_r2:.4f}")
    with col2:
        st.metric("Playtime Model", "XGBoost", f"R2 = {playtime_r2:.4f}")
    
    st.divider()
    
    # =========================================================
    # BASIC FEATURES
    # =========================================================
    st.subheader("Basic Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        price = st.number_input("Price ($)", 0.0, 500.0, 19.99, 0.99)
        dlc_count = st.number_input("DLC Count", 0, 100, 0)
    with col2:
        achievements = st.number_input("Achievements", 0, 1000, 50)
        release_date = st.date_input("Release Date", date.today())
    with col3:
        st.write("Platforms")
        windows = st.checkbox("Windows", True)
        mac = st.checkbox("Mac", False)
        linux = st.checkbox("Linux", False)
    
    st.divider()
    
    # =========================================================
    # CATEGORIES
    # =========================================================
    st.subheader("Categories")
    selected_categories = st.multiselect(
        "Select categories:",
        options=[c.replace('_', ' ') for c in CATEGORIES],
        default=['Single player', 'Steam Achievements']
    )
    selected_categories = [c.replace(' ', '_') for c in selected_categories]
    
    st.divider()
    
    # =========================================================
    # GENRES
    # =========================================================
    st.subheader("Genres")
    selected_genres = st.multiselect(
        "Select genres:",
        options=[g.replace('_', ' ') for g in GENRES],
        default=['Action', 'Indie']
    )
    selected_genres = [g.replace(' ', '_') for g in selected_genres]
    
    st.divider()
    
    # =========================================================
    # LANGUAGES
    # =========================================================
    st.subheader("Languages")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Supported Languages (Subtitles)")
        selected_supported_langs = st.multiselect(
            "Select supported languages:",
            options=[l.replace('_', ' ') for l in SUPPORTED_LANGUAGES],
            default=['English'],
            key="supported"
        )
        selected_supported_langs = [l.replace(' ', '_') for l in selected_supported_langs]
        st.info(f"Count: {len(selected_supported_langs)}")
    
    with col2:
        st.write("Full Audio Languages")
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
    # TAGS
    # =========================================================
    st.subheader("Tags")
    st.write("Add tags and their vote counts:")
    
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
        if st.button("Add Tag"):
            if tag_name:
                st.session_state.tags[tag_name] = tag_votes
    
    if st.session_state.tags:
        st.write("Current Tags:")
        for tag, votes in st.session_state.tags.items():
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.write(tag)
            with col2:
                st.write(f"{votes:,} votes")
            with col3:
                if st.button("Remove", key=f"remove_{tag}"):
                    del st.session_state.tags[tag]
                    st.rerun()
        
        tag_count = len(st.session_state.tags)
        total_tag_votes = sum(st.session_state.tags.values())
        st.info(f"Tag Count: {tag_count} | Total Votes: {total_tag_votes:,}")
    else:
        tag_count = 0
        total_tag_votes = 0
        st.info("No tags added (defaults to 0)")
    
    st.divider()
    
    # =========================================================
    # PREDICT BUTTON
    # =========================================================
    if st.button("Predict Performance", type="primary"):
        try:
            platform_count = int(windows) + int(mac) + int(linux)
            category_count = len(selected_categories)
            genre_count = len(selected_genres)
            supported_languages_count = len(selected_supported_langs)
            audio_languages_count = len(selected_audio_langs)
            
            days_since_release = (date.today() - release_date).days
            release_year = release_date.year
            release_month = release_date.month
            release_day_of_week = release_date.isoweekday()
            release_quarter = (release_month - 1) // 3 + 1
            year_bucket_2000s = int(2000 <= release_year < 2010)
            year_bucket_2010s = int(2010 <= release_year < 2020)
            year_bucket_2020s = int(release_year >= 2020)
            
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
            
            # Add all tag features with 0 default
            for tag in TAGS:
                tag_key = f'tags_{tag}'
                input_data[tag_key] = st.session_state.tags.get(tag, 0)
            
            input_df = pd.DataFrame([input_data])
            
            expected_features = model_ccu.feature_names_in_
            
            for feature in expected_features:
                if feature not in input_data:
                    input_data[feature] = 0
            
            input_df = pd.DataFrame([input_data])[expected_features]
            
            # Make predictions
            pred_ccu_log = model_ccu.predict(input_df)[0]
            pred_playtime_log = model_playtime.predict(input_df)[0]
            
            pred_ccu = np.expm1(pred_ccu_log)
            pred_playtime = np.expm1(pred_playtime_log)
            
            st.success("Prediction Complete!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Peak CCU", f"{int(pred_ccu):,}", help="Maximum concurrent users at peak")
            with col2:
                st.metric("Predicted Avg Playtime", f"{int(pred_playtime):,} mins", help="Average minutes played per user")
            
            # =========================================================
            # CALCULATE TOP FEATURES TO ADD
            # =========================================================
            with st.spinner("Analyzing top features to add..."):
                # Get current feature values
                current_values = {}
                for feature in expected_features:
                    current_values[feature] = input_data.get(feature, 0)
                
                # Get all feature types to test
                all_features = {
                    'categories': [f for f in expected_features if f.startswith('categories_') and input_data.get(f, 0) == 0],
                    'genres': [f for f in expected_features if f.startswith('genres_') and input_data.get(f, 0) == 0],
                    'supported_langs': [f for f in expected_features if f.startswith('supported_languages_') and input_data.get(f, 0) == 0],
                    'audio_langs': [f for f in expected_features if f.startswith('full_audio_languages_') and input_data.get(f, 0) == 0],
                    'tags': [f for f in expected_features if f.startswith('tags_') and input_data.get(f, 0) == 0]
                }
                
                # Combine all missing features
                all_missing_features = []
                for feature_list in all_features.values():
                    all_missing_features.extend(feature_list)
                
                # Limit the number of features to test for performance
                if len(all_missing_features) > 50:
                    # Get top features from importance
                    if feat_imp_ccu is not None:
                        importance_df = pd.DataFrame({
                            'feature': expected_features,
                            'importance': feat_imp_ccu
                        })
                        importance_df = importance_df[importance_df['feature'].isin(all_missing_features)]
                        importance_df = importance_df.sort_values('importance', ascending=False)
                        all_missing_features = importance_df['feature'].head(50).tolist()
                    else:
                        all_missing_features = all_missing_features[:50]
                
                # Analyze top features for CCU
                st.info(f"Analyzing {len(all_missing_features)} features for improvement opportunities...")
                
                top_features_ccu = get_top_features_to_add(
                    model_ccu,
                    input_data.copy(),
                    expected_features,
                    all_missing_features,
                    current_values,
                    pred_ccu_log,
                    'ccu'
                )
                
                top_features_playtime = get_top_features_to_add(
                    model_playtime,
                    input_data.copy(),
                    expected_features,
                    all_missing_features,
                    current_values,
                    pred_playtime_log,
                    'playtime'
                )
            
            # =========================================================
            # DISPLAY TOP FEATURE RECOMMENDATIONS
            # =========================================================
            st.subheader("Top 10 Features to Add")
            
            # Combine and sort recommendations
            all_recommendations = []
            
            # Process CCU recommendations
            for feature, impact in top_features_ccu:
                if impact['abs_change'] > 0:
                    display_name = impact['display_name']
                    feature_type = impact['feature_type']
                    
                    if feature_type == 'Supported Language':
                        suggestion = f"Add support for {display_name}"
                    elif feature_type == 'Audio Language':
                        suggestion = f"Add audio support for {display_name}"
                    else:
                        suggestion = f"Add {feature_type} {display_name}"
                    
                    all_recommendations.append({
                        'feature': feature,
                        'display_name': display_name,
                        'type': feature_type,
                        'ccu_impact': impact['abs_change'],
                        'ccu_pct': impact['pct_change'],
                        'playtime_impact': 0,
                        'playtime_pct': 0,
                        'suggestion': suggestion
                    })
            
            # Process Playtime recommendations
            for feature, impact in top_features_playtime:
                if impact['abs_change'] > 0:
                    # Check if already in recommendations
                    existing = next((r for r in all_recommendations if r['feature'] == feature), None)
                    if existing:
                        existing['playtime_impact'] = impact['abs_change']
                        existing['playtime_pct'] = impact['pct_change']
                    else:
                        display_name = impact['display_name']
                        feature_type = impact['feature_type']
                        
                        if feature_type == 'Supported Language':
                            suggestion = f"Add support for {display_name}"
                        elif feature_type == 'Audio Language':
                            suggestion = f"Add audio support for {display_name}"
                        else:
                            suggestion = f"Add {feature_type} {display_name}"
                        
                        all_recommendations.append({
                            'feature': feature,
                            'display_name': display_name,
                            'type': feature_type,
                            'ccu_impact': 0,
                            'ccu_pct': 0,
                            'playtime_impact': impact['abs_change'],
                            'playtime_pct': impact['pct_change'],
                            'suggestion': suggestion
                        })
            
            # Sort by combined impact
            all_recommendations.sort(key=lambda x: abs(x['ccu_impact']) + abs(x['playtime_impact']), reverse=True)
            
            # Display top 10
            if all_recommendations:
                for i, rec in enumerate(all_recommendations[:10]):
                    with st.container():
                        col1, col2, col3 = st.columns([2, 1.5, 1.5])
                        with col1:
                            st.write(f"**{i+1}. {rec['display_name']}**")
                            st.caption(f"Type: {rec['type']}")
                            st.write(rec['suggestion'])
                        with col2:
                            if rec['ccu_impact'] > 0:
                                st.metric(
                                    "CCU Impact", 
                                    f"+{int(rec['ccu_impact']):,}", 
                                    delta=f"{rec['ccu_pct']:.1f}%",
                                    delta_color="normal"
                                )
                            else:
                                st.write("No CCU impact detected")
                        with col3:
                            if rec['playtime_impact'] > 0:
                                st.metric(
                                    "Playtime Impact", 
                                    f"+{int(rec['playtime_impact']):,} min", 
                                    delta=f"{rec['playtime_pct']:.1f}%",
                                    delta_color="normal"
                                )
                            else:
                                st.write("No Playtime impact detected")
                        
                        st.divider()
            else:
                st.success("Your game configuration looks well-optimized! No major improvements needed.")
            
            # =========================================================
            # GENERAL RECOMMENDATIONS
            # =========================================================
            st.subheader("Additional Recommendations")
            
            general_recs = []
            
            # Price recommendations
            if price == 0:
                general_recs.append("Consider adding a price > $0 - free games often have lower perceived value")
            elif price > 50:
                general_recs.append("High price may limit initial user base - consider a lower price point for better adoption")
            
            # Platform recommendations
            platform_count = int(windows) + int(mac) + int(linux)
            if platform_count < 3:
                general_recs.append(f"Add support for {3-platform_count} additional platform(s) to reach wider audience (Mac/Linux)")
            
            # Category recommendations
            if category_count < 3:
                general_recs.append("Add more categories to improve discoverability")
            
            # Multiplayer recommendations
            if input_data.get('categories_Multi_player', 0) == 0 and input_data.get('categories_Co_op', 0) == 0:
                general_recs.append("Consider adding multiplayer or co-op features to boost peak concurrent users")
            
            # Achievement recommendations
            if achievements < 20:
                general_recs.append("Add more achievements (20+) to increase player engagement and playtime")
            
            # Language recommendations
            if supported_languages_count < 5:
                general_recs.append("Add more supported languages to reach international audience (5+ languages recommended)")
            
            # DLC recommendations
            if dlc_count == 0:
                general_recs.append("Consider adding DLCs to extend game lifecycle and revenue")
            
            # VR support
            if input_data.get('categories_VR_Support', 0) == 0:
                general_recs.append("Consider adding VR support to tap into growing VR market")
            
            # Workshop support
            if input_data.get('categories_Steam_Workshop', 0) == 0:
                general_recs.append("Add Steam Workshop support to encourage community content and longer playtime")
            
            if general_recs:
                for rec in general_recs:
                    st.info(rec)
            else:
                st.success("No additional recommendations - your game configuration looks great!")
            
            # =========================================================
            # FEATURE IMPORTANCE VISUALIZATION
            # =========================================================
            st.subheader("Feature Importance Analysis")
            
            tab1, tab2 = st.tabs(["Peak CCU - Feature Importance", "Playtime - Feature Importance"])
            
            with tab1:
                if feat_imp_ccu is not None:
                    df_imp_ccu = get_feature_importance_dataframe(feat_imp_ccu, expected_features)
                    if df_imp_ccu is not None:
                        fig_ccu = plot_feature_importance(df_imp_ccu, "Top 20 Features for Peak CCU Prediction")
                        if fig_ccu:
                            st.plotly_chart(fig_ccu, use_container_width=True)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Top 5 Most Important Features (CCU)**")
                                top_5 = df_imp_ccu.tail(5)[['feature', 'importance']].copy()
                                top_5['importance'] = top_5['importance'].apply(lambda x: f"{x:.4f}")
                                st.dataframe(top_5, hide_index=True)
                            with col2:
                                st.write("**Least Important Features (CCU)**")
                                bottom_5 = df_imp_ccu.head(5)[['feature', 'importance']].copy()
                                bottom_5['importance'] = bottom_5['importance'].apply(lambda x: f"{x:.4f}")
                                st.dataframe(bottom_5, hide_index=True)
                    else:
                        st.warning("Feature importance data not available for CCU model")
                else:
                    st.warning("Feature importance data not available for CCU model")
            
            with tab2:
                if feat_imp_playtime is not None:
                    df_imp_playtime = get_feature_importance_dataframe(feat_imp_playtime, expected_features)
                    if df_imp_playtime is not None:
                        fig_playtime = plot_feature_importance(df_imp_playtime, "Top 20 Features for Playtime Prediction")
                        if fig_playtime:
                            st.plotly_chart(fig_playtime, use_container_width=True)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Top 5 Most Important Features (Playtime)**")
                                top_5 = df_imp_playtime.tail(5)[['feature', 'importance']].copy()
                                top_5['importance'] = top_5['importance'].apply(lambda x: f"{x:.4f}")
                                st.dataframe(top_5, hide_index=True)
                            with col2:
                                st.write("**Least Important Features (Playtime)**")
                                bottom_5 = df_imp_playtime.head(5)[['feature', 'importance']].copy()
                                bottom_5['importance'] = bottom_5['importance'].apply(lambda x: f"{x:.4f}")
                                st.dataframe(bottom_5, hide_index=True)
                    else:
                        st.warning("Feature importance data not available for Playtime model")
                else:
                    st.warning("Feature importance data not available for Playtime model")
            
            # =========================================================
            # INPUT SUMMARY
            # =========================================================
            with st.expander("Input Summary"):
                st.write(f"Basic: ${price} | {dlc_count} DLCs | {achievements} achievements")
                st.write(f"Platforms: {platform_count} ({', '.join([p for p, v in [('Windows', windows), ('Mac', mac), ('Linux', linux)] if v])})")
                st.write(f"Categories ({category_count}): {', '.join([c.replace('_', ' ') for c in selected_categories])}")
                st.write(f"Genres ({genre_count}): {', '.join([g.replace('_', ' ') for g in selected_genres])}")
                st.write(f"Languages: {supported_languages_count} supported, {audio_languages_count} audio")
                st.write(f"Tags: {tag_count} tags, {total_tag_votes:,} total votes")
                st.write(f"Release: {release_date} ({days_since_release} days ago)")
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.exception(e)
else:
    st.error("Failed to load models")

st.divider()
st.caption("Models trained on Steam game metadata | Random Forest (CCU) + XGBoost (Playtime)")
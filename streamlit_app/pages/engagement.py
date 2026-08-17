import streamlit as st
import mlflow
import pandas as pd
import numpy as np
import os

# =========================================================
# CONFIGURATION
# =========================================================

mlflow.set_tracking_uri("databricks")
MLFLOW_EXPERIMENT = "/Users/242475r@mymail.nyp.edu.sg/IT3388_BDMP/game_performance_models"
mlflow.set_experiment(MLFLOW_EXPERIMENT)

# =========================================================
# LOAD BEST MODELS
# =========================================================

@st.cache_resource
def load_best_models():
    """Load the best performing models for Peak CCU and Average Playtime"""
    
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
    
    # Load best Peak CCU model
    runs_ccu = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName LIKE '% - Peak CCU'",
        order_by=["metrics.test_r2 DESC"],
        max_results=1
    )
    
    # Load best Average Playtime model
    runs_playtime = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName LIKE '% - Average Playtime'",
        order_by=["metrics.test_r2 DESC"],
        max_results=1
    )
    
    if len(runs_ccu) > 0 and len(runs_playtime) > 0:
        model_ccu = mlflow.sklearn.load_model(f"runs:/{runs_ccu.iloc[0]['run_id']}/model")
        model_playtime = mlflow.sklearn.load_model(f"runs:/{runs_playtime.iloc[0]['run_id']}/model")
        
        ccu_name = runs_ccu.iloc[0]['tags.mlflow.runName']
        playtime_name = runs_playtime.iloc[0]['tags.mlflow.runName']
        ccu_r2 = runs_ccu.iloc[0]['metrics.test_r2']
        playtime_r2 = runs_playtime.iloc[0]['metrics.test_r2']
        
        return model_ccu, model_playtime, ccu_name, playtime_name, ccu_r2, playtime_r2
    
    return None, None, None, None, None, None

# Load models
model_ccu, model_playtime, ccu_name, playtime_name, ccu_r2, playtime_r2 = load_best_models()

# =========================================================
# DEFINE ALL CATEGORIES AND GENRES
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
# STREAMLIT APP UI
# =========================================================

st.title("🎮 Steam Game Performance Predictor")
st.write("Predict Peak CCU and Average Playtime based on game features")

if model_ccu and model_playtime:
    st.success(f"✅ Models loaded successfully!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Peak CCU Model", ccu_name.split(' - ')[0], f"R² = {ccu_r2:.4f}")
    with col2:
        st.metric("Playtime Model", playtime_name.split(' - ')[0], f"R² = {playtime_r2:.4f}")
    
    st.divider()
    
    # =========================================================
    # INPUT FEATURES
    # =========================================================
    
    st.subheader("📊 Game Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        price = st.number_input("Price ($)", min_value=0.0, max_value=500.0, value=19.99, step=0.99)
        dlc_count = st.number_input("DLC Count", min_value=0, max_value=100, value=0, step=1)
        achievements = st.number_input("Achievements", min_value=0, max_value=1000, value=50, step=1)
    
    with col2:
        st.write("**Platform Support**")
        windows = st.checkbox("Windows", value=True)
        mac = st.checkbox("Mac", value=False)
        linux = st.checkbox("Linux", value=False)
    
    st.divider()
    
    # Categories Selection
    st.subheader("🎯 Categories")
    st.write("Select applicable categories:")
    
    # Popular categories displayed as checkboxes
    selected_categories = []
    
    col1, col2, col3 = st.columns(3)
    for i, category in enumerate(sorted(CATEGORIES)):
        col = [col1, col2, col3][i % 3]
        with col:
            if st.checkbox(category.replace('_', ' '), key=f"cat_{category}"):
                selected_categories.append(category)
    
    st.divider()
    
    # Genres Selection
    st.subheader("🎨 Genres")
    st.write("Select applicable genres:")
    
    selected_genres = []
    
    col1, col2, col3 = st.columns(3)
    for i, genre in enumerate(sorted(GENRES)):
        col = [col1, col2, col3][i % 3]
        with col:
            if st.checkbox(genre.replace('_', ' '), key=f"gen_{genre}"):
                selected_genres.append(genre)
    
    st.divider()
    
    # =========================================================
    # MAKE PREDICTIONS
    # =========================================================
    
    if st.button("🔮 Predict Performance", type="primary"):
        try:
            # Build feature dataframe with ALL features from silver table
            # Start with basic features
            input_data = {
                'price': price,
                'dlc_count': dlc_count,
                'windows': int(windows),
                'mac': int(mac),
                'linux': int(linux),
                'achievements': achievements
            }
            
            # Add all category features (0 by default)
            for cat in CATEGORIES:
                input_data[f'categories_{cat}'] = 1 if cat in selected_categories else 0
            
            # Add all genre features (0 by default)
            for genre in GENRES:
                input_data[f'genres_{genre}'] = 1 if genre in selected_genres else 0
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Make predictions (models were trained on log-transformed targets)
            pred_ccu_log = model_ccu.predict(input_df)[0]
            pred_playtime_log = model_playtime.predict(input_df)[0]
            
            # Transform back to original scale
            pred_ccu = np.expm1(pred_ccu_log)
            pred_playtime = np.expm1(pred_playtime_log)
            
            st.success("✅ Prediction Complete!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Predicted Peak CCU",
                    f"{int(pred_ccu):,}",
                    help="Maximum concurrent users at peak"
                )
            
            with col2:
                st.metric(
                    "Predicted Avg Playtime",
                    f"{int(pred_playtime):,} mins",
                    help="Average playtime per player"
                )
            
            # Show selection summary
            with st.expander("📋 Selection Summary"):
                st.write(f"**Price:** ${price}")
                st.write(f"**DLC Count:** {dlc_count}")
                st.write(f"**Achievements:** {achievements}")
                st.write(f"**Platforms:** {', '.join([p for p, v in [('Windows', windows), ('Mac', mac), ('Linux', linux)] if v])}")
                st.write(f"**Selected Categories ({len(selected_categories)}):** {', '.join([c.replace('_', ' ') for c in selected_categories]) if selected_categories else 'None'}")
                st.write(f"**Selected Genres ({len(selected_genres)}):** {', '.join([g.replace('_', ' ') for g in selected_genres]) if selected_genres else 'None'}")
                
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.exception(e)
else:
    st.error("❌ Failed to load models. Please check MLflow configuration.")

st.divider()
st.caption("Models trained on Steam game metadata dataset")
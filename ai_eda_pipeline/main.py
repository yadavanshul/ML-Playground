import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import json
import time
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.colored_header import colored_header
import uuid

# Import custom modules
from ai_eda_pipeline.utils.data_utils import load_dataset, get_dataset_metadata, detect_data_issues, suggest_preprocessing_steps, apply_preprocessing_step
from ai_eda_pipeline.utils.visualization_utils import get_available_plots, generate_plot
from ai_eda_pipeline.components.ai_agents import MainAIAgent
from ai_eda_pipeline.components.preprocessing_workflow import get_preprocessing_steps_by_category, get_step_config_ui, render_workflow, apply_workflow

# Set page configuration
st.set_page_config(
    page_title="Machine Learning Playground, Using AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.df = None
    st.session_state.dataset_name = None
    st.session_state.analysis = None
    st.session_state.available_plots = {}
    st.session_state.dashboard_plots = []
    st.session_state.insights = {}
    st.session_state.preprocessing_steps = []
    st.session_state.ai_agent = MainAIAgent()
    st.session_state.reasoning_log = []
    st.session_state.plot_configs = {}
    # Preprocessing workflow state
    st.session_state.preprocessing_workflow = []
    st.session_state.processed_df = None
    st.session_state.editing_step = None
    st.session_state.active_tab = "eda"  # Default tab: 'eda' or 'preprocessing' or 'ml'
    st.session_state.preprocessing_messages = []
    # Preview state
    st.session_state.preview_plot = None
    # ML Pipeline state
    st.session_state.ml_models = []
    st.session_state.selected_model = None
    st.session_state.model_results = {}
    st.session_state.feature_importance = {}
    st.session_state.train_test_split = {"test_size": 0.2, "random_state": 42}
    st.session_state.target_column = None
    st.session_state.model_metrics = {}
    # API key state
    st.session_state.openai_api_key = None
    st.session_state.api_key_configured = False

# Add custom CSS
st.markdown("""
<style>
/* General styling */
.stApp {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Dashboard styling */
.dashboard-container {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(450px, 1fr));
    gap: 1.5rem;
    margin-top: 1.5rem;
}

.dashboard-plot {
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 1.2rem;
    background-color: white;
    box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    position: relative;
    transition: all 0.3s ease;
    overflow: hidden;
}

.dashboard-plot:hover {
    box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    transform: translateY(-3px);
}

.plot-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid #f0f0f0;
}

.plot-title {
    font-weight: 600;
    font-size: 1.2rem;
    color: #2E7D32;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.plot-title::before {
    content: "";
    display: inline-block;
    width: 12px;
    height: 12px;
    background-color: #4CAF50;
    border-radius: 50%;
}

.plot-actions {
    display: flex;
    gap: 0.5rem;
}

.plot-actions button {
    background: none;
    border: none;
    cursor: pointer;
    font-size: 1.2rem;
    padding: 0.2rem;
    border-radius: 4px;
    transition: background-color 0.2s;
}

.plot-actions button:hover {
    background-color: #f0f0f0;
}

/* AI Recommendations styling */
.ai-recommendation {
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1.5rem;
    background-color: white;
    box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.ai-recommendation::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    background: linear-gradient(to bottom, #4CAF50, #2196F3);
}

.ai-recommendation:hover {
    box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    transform: translateY(-3px);
}

.ai-recommendation-title {
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
    color: #2E7D32;
}

.ai-recommendation-reason {
    color: #555;
    margin-bottom: 1.2rem;
    font-size: 0.95rem;
    line-height: 1.5;
    padding-left: 0.5rem;
    border-left: 2px solid #f0f0f0;
}

.viz-badge {
    font-size: 0.7rem;
    padding: 0.3rem 0.6rem;
    border-radius: 20px;
    color: white;
    background-color: #4CAF50;
    display: inline-block;
    margin-left: 0.5rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.histogram { background-color: #4CAF50; }
.boxplot { background-color: #2196F3; }
.scatter { background-color: #9C27B0; }
.bar { background-color: #FF9800; }
.pie { background-color: #E91E63; }
.correlation_heatmap { background-color: #F44336; }
.line { background-color: #00BCD4; }
.pairplot { background-color: #795548; }

/* Available plots styling */
.plot-box-container {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.5rem;
}

.plot-box {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 0.8rem;
    background-color: white;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s ease;
    width: 100px;
    height: 100px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    position: relative;
}

.plot-box:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    background-color: #f9f9f9;
}

.plot-box-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}

.plot-box-label {
    font-size: 0.8rem;
    color: #333;
}

.plot-box-tooltip {
    position: absolute;
    bottom: -40px;
    left: 50%;
    transform: translateX(-50%);
    background-color: rgba(0,0,0,0.8);
    color: white;
    padding: 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
    opacity: 0;
    transition: opacity 0.2s;
    pointer-events: none;
    white-space: nowrap;
    z-index: 1000;
}

.plot-box:hover .plot-box-tooltip {
    opacity: 1;
}

/* Preview button styling */
.preview-button {
    position: relative;
    display: inline-block;
    width: 100%;
}

.preview-content {
    display: none;
    position: absolute;
    z-index: 1000;
    background-color: white;
    border: 1px solid #ddd;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    padding: 1rem;
    width: 300px;
    right: 0;
    top: 100%;
}

.preview-button:hover .preview-content {
    display: block;
}

/* Hide the actual plotly chart but keep it for the hover content */
[data-testid="stPlotlyChart"] {
    display: none;
}

.preview-button:hover [data-testid="stPlotlyChart"] {
    display: block;
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
}

/* Insight container styling */
.insight-container {
    background-color: #f8f9fa;
    border-left: 4px solid #4CAF50;
    padding: 1rem;
    margin-top: 1rem;
    border-radius: 0 4px 4px 0;
}

/* Preview container styling */
.preview-container {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 10px;
    margin-top: 10px;
    background-color: white;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .plot-box {
        width: 80px;
        height: 80px;
    }
    
    .plot-box-icon {
        font-size: 1.5rem;
    }
    
    .plot-box-label {
        font-size: 0.7rem;
    }
}

/* Reasoning log styling */
.reasoning-log {
    padding: 8px 12px;
    margin-bottom: 8px;
    border-radius: 6px;
    font-size: 0.9rem;
    line-height: 1.4;
    border-left: 3px solid #ccc;
}

.reasoning-log-thinking {
    background-color: #f0f7ff;
    border-left-color: #2196F3;
}

.reasoning-log-insight {
    background-color: #f0fff4;
    border-left-color: #4CAF50;
}

.reasoning-log-recommendation {
    background-color: #fff8e1;
    border-left-color: #FFC107;
}

.reasoning-log-action {
    background-color: #f5f5f5;
    border-left-color: #9E9E9E;
}

.reasoning-log-timestamp {
    font-weight: 600;
    color: #555;
    margin-right: 6px;
}
</style>
""", unsafe_allow_html=True)

# Helper function to add to reasoning log
def add_to_log(message, is_thinking=False, is_insight=False, is_recommendation=False):
    timestamp = time.strftime("%H:%M:%S")
    
    # Format the message based on type
    if is_thinking:
        formatted_message = f"🧠 [AI Thinking] {message}"
    elif is_insight:
        formatted_message = f"💡 [AI Insight] {message}"
    elif is_recommendation:
        formatted_message = f"🔍 [AI Recommendation] {message}"
    else:
        formatted_message = f"🔄 {message}"
    
    st.session_state.reasoning_log.append(f"[{timestamp}] {formatted_message}")
    if len(st.session_state.reasoning_log) > 100:  # Increased log size
        st.session_state.reasoning_log = st.session_state.reasoning_log[-100:]

# Function to add preprocessing message
def add_preprocessing_message(message, is_error=False):
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.preprocessing_messages.append({
        "timestamp": timestamp,
        "message": message,
        "is_error": is_error
    })
    add_to_log(message)  # Also add to main log

# Function to add a preprocessing step to workflow
def add_preprocessing_step(step_config):
    if not step_config:
        return
    
    if "editing_step" in st.session_state and st.session_state.editing_step is not None:
        # Update existing step
        idx = st.session_state.editing_step
        if idx < len(st.session_state.preprocessing_workflow):
            st.session_state.preprocessing_workflow[idx] = step_config
            add_preprocessing_message(f"Updated {step_config.get('step')} step in workflow")
        st.session_state.editing_step = None
    else:
        # Add new step
        st.session_state.preprocessing_workflow.append(step_config)
        add_preprocessing_message(f"Added {step_config.get('step')} step to workflow")

# Function to apply preprocessing workflow
def apply_preprocessing_workflow():
    if not st.session_state.preprocessing_workflow:
        add_preprocessing_message("No preprocessing steps to apply", is_error=True)
        return False
    
    try:
        # Apply workflow to original dataframe
        processed_df, messages = apply_workflow(st.session_state.df, st.session_state.preprocessing_workflow)
        
        # Store processed dataframe
        st.session_state.processed_df = processed_df
        
        # Add messages to log
        for msg in messages:
            add_preprocessing_message(msg)
        
        return True
    except Exception as e:
        add_preprocessing_message(f"Error applying preprocessing workflow: {str(e)}", is_error=True)
        return False

# Function to load and analyze dataset
def load_and_analyze_dataset(file_buffer=None, dataset_name=None):
    try:
        # Load dataset
        add_to_log("Starting dataset loading process", is_thinking=True)
        df, name = load_dataset(file_buffer=file_buffer, dataset_name=dataset_name)
        
        # Store in session state
        st.session_state.df = df
        st.session_state.dataset_name = name
        
        # Analyze dataset using AI agent
        add_to_log(f"Analyzing dataset: {name}")
        add_to_log(f"Examining data structure: {df.shape[0]} rows, {df.shape[1]} columns", is_thinking=True)
        add_to_log(f"Checking for missing values and data types", is_thinking=True)
        
        # Log column information
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        add_to_log(f"Found {len(numeric_cols)} numeric columns and {len(categorical_cols)} categorical columns", is_thinking=True)
        
        # Analyze dataset
        analysis = st.session_state.ai_agent.analyze_dataset(df, name)
        st.session_state.analysis = analysis
        
        # Log analysis results
        if "issues" in analysis:
            issues = analysis["issues"]
            if issues["missing_values"]:
                add_to_log(f"Detected missing values in {len(issues['missing_values'])} columns", is_thinking=True)
            if issues["outliers"]:
                add_to_log(f"Detected outliers in {len(issues['outliers'])} columns", is_thinking=True)
            if issues["high_correlation"]:
                add_to_log(f"Found {len(issues['high_correlation'])} highly correlated column pairs", is_thinking=True)
        
        # Get available plots
        add_to_log("Determining suitable visualization types for this dataset", is_thinking=True)
        st.session_state.available_plots = get_available_plots(df)
        
        # Get AI-recommended visualizations
        add_to_log("Generating AI-recommended visualizations based on data patterns", is_thinking=True)
        st.session_state.recommended_visualizations = st.session_state.ai_agent.recommend_visualizations(df)
        
        # Log recommendations
        for i, viz in enumerate(st.session_state.recommended_visualizations[:3]):  # Log first 3 recommendations
            add_to_log(f"Recommending {viz['type']} visualization: {viz['reason'][:100]}...", is_recommendation=True)
        
        # Initialize plot configs
        st.session_state.plot_configs = {}
        
        # Clear dashboard plots and insights
        st.session_state.dashboard_plots = []
        st.session_state.insights = {}
        
        add_to_log(f"Dataset loaded and analyzed successfully: {name} ({df.shape[0]} rows, {df.shape[1]} columns)")
        return True
    
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        add_to_log(f"Error loading dataset: {str(e)}")
        return False

# Function to generate a plot and add it to dashboard
def add_plot_to_dashboard(plot_type, config):
    if len(st.session_state.dashboard_plots) >= 6:
        st.warning("Maximum of 6 plots allowed on dashboard. Remove a plot to add a new one.")
        add_to_log("Failed to add plot: Maximum limit reached")
        return False
    
    add_to_log(f"Preparing to add {plot_type} plot to dashboard", is_thinking=True)
    
    # Log configuration details
    config_details = []
    for key, value in config.items():
        if isinstance(value, list) and len(value) > 3:
            value_str = f"{', '.join(str(v) for v in value[:3])}... ({len(value)} items)"
        else:
            value_str = str(value)
        config_details.append(f"{key}: {value_str}")
    
    add_to_log(f"Plot configuration: {'; '.join(config_details)}", is_thinking=True)
    
    # Store the plot configuration directly in the dashboard_plots list
    st.session_state.dashboard_plots.append({
        "type": plot_type,
        "config": config
    })
    
    add_to_log(f"Added {plot_type} plot to dashboard")
    return True

# Function to remove plot from dashboard
def remove_plot_from_dashboard(index):
    if 0 <= index < len(st.session_state.dashboard_plots):
        removed_plot = st.session_state.dashboard_plots.pop(index)
        add_to_log(f"Removed {removed_plot['type']} plot from dashboard")
        return True
    
    return False

# Function to get AI insight for a plot
def get_ai_insight(plot_id):
    if plot_id not in st.session_state.plot_configs:
        return "Error: Plot not found"
    
    plot_config = st.session_state.plot_configs[plot_id]
    add_to_log(f"Generating {plot_config['type']} plot for insight analysis", is_thinking=True)
    plot_data = generate_plot(st.session_state.df, plot_config["type"], plot_config["config"])
    
    add_to_log(f"Requesting AI insight for {plot_config['type']} plot")
    
    # Add thinking steps for different plot types
    if plot_config['type'] == 'histogram':
        add_to_log("Analyzing distribution shape, skewness, and potential outliers", is_thinking=True)
    elif plot_config['type'] == 'scatter':
        add_to_log("Examining correlation patterns, clusters, and potential relationships", is_thinking=True)
    elif plot_config['type'] == 'boxplot':
        add_to_log("Identifying quartiles, median values, and outlier presence", is_thinking=True)
    elif plot_config['type'] == 'correlation_heatmap':
        add_to_log("Detecting strong positive and negative correlations between variables", is_thinking=True)
    else:
        add_to_log(f"Analyzing patterns and trends in the {plot_config['type']} visualization", is_thinking=True)
    
    # Get insight from EDA agent
    insight = st.session_state.ai_agent.get_eda_agent().generate_insight(
        st.session_state.df,
        plot_data,
        st.session_state.dataset_name
    )
    
    # Store insight
    st.session_state.insights[plot_id] = insight
    
    # Log a summary of the insight
    insight_summary = insight.split('.')[0] + '.' if '.' in insight else insight
    add_to_log(f"Key insight: {insight_summary}", is_insight=True)
    
    add_to_log(f"Generated AI insight for {plot_config['type']} plot")
    return insight

def get_plot_config(plot_type, df):
    """
    Generate configuration for a plot type based on the dataframe.
    
    Args:
        plot_type (str): The type of plot to configure
        df (pd.DataFrame): The dataframe to analyze
        
    Returns:
        dict: Configuration for the plot
    """
    # Get column types
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Default configurations based on plot type
    if plot_type == "histogram":
        # Choose the first numeric column
        if numeric_cols:
            return {"column": numeric_cols[0]}
        return {"column": df.columns[0]}  # Fallback
        
    elif plot_type == "boxplot":
        # Choose the first numeric column
        if numeric_cols:
            return {"column": numeric_cols[0]}
        return {"column": df.columns[0]}  # Fallback
        
    elif plot_type == "scatter":
        # Choose the first two numeric columns
        if len(numeric_cols) >= 2:
            return {"x_column": numeric_cols[0], "y_column": numeric_cols[1]}
        elif len(numeric_cols) == 1:
            return {"x_column": numeric_cols[0], "y_column": numeric_cols[0]}
        return {"x_column": df.columns[0], "y_column": df.columns[0]}  # Fallback
        
    elif plot_type == "bar":
        # Choose the first categorical column and first numeric column
        if categorical_cols and numeric_cols:
            return {"x_column": categorical_cols[0], "y_column": numeric_cols[0]}
        elif categorical_cols:
            return {"column": categorical_cols[0]}
        return {"column": df.columns[0]}  # Fallback
        
    elif plot_type == "pie":
        # Choose the first categorical column
        if categorical_cols:
            return {"column": categorical_cols[0]}
        return {"column": df.columns[0]}  # Fallback
        
    elif plot_type == "correlation_heatmap":
        # Use all numeric columns
        if numeric_cols:
            return {"columns": numeric_cols, "method": "pearson"}
        return {"columns": df.columns.tolist(), "method": "pearson"}  # Fallback
        
    elif plot_type == "line":
        # Choose the first two numeric columns
        if len(numeric_cols) >= 2:
            return {"x_column": numeric_cols[0], "y_column": numeric_cols[1]}
        elif len(numeric_cols) == 1:
            return {"x_column": numeric_cols[0], "y_column": numeric_cols[0]}
        return {"x_column": df.columns[0], "y_column": df.columns[0]}  # Fallback
        
    elif plot_type == "pairplot":
        # Choose up to 4 numeric columns
        if numeric_cols:
            selected_cols = numeric_cols[:min(4, len(numeric_cols))]
            hue = categorical_cols[0] if categorical_cols and len(df[categorical_cols[0]].unique()) <= 5 else None
            return {"columns": selected_cols, "hue": hue}
        return {"columns": df.columns.tolist()[:min(4, len(df.columns))], "hue": None}  # Fallback
    
    # Default empty config
    return {}

# Function to render the ML Pipeline tab
def render_ml_pipeline_tab():
    st.markdown("<h2 style='font-size: 1.8rem; font-weight: 600; margin-top: 1rem; margin-bottom: 1.5rem; color: #2E7D32; border-bottom: 2px solid #4CAF50; padding-bottom: 0.5rem;'>Machine Learning Pipeline</h2>", unsafe_allow_html=True)
    
    # Check if we have data to work with
    if st.session_state.df is None:
        st.info("Please load a dataset first to use the ML Pipeline.")
        return
    
    # Use processed data if available, otherwise use original data
    df_to_use = st.session_state.processed_df if st.session_state.processed_df is not None else st.session_state.df
    
    # ML Pipeline Steps
    ml_steps = st.tabs(["1️⃣ Data Preparation", "2️⃣ Model Selection", "3️⃣ Training & Evaluation", "4️⃣ Model Insights"])
    
    # Step 1: Data Preparation
    with ml_steps[0]:
        st.markdown("### Data Preparation")
        
        # Target column selection
        numeric_cols = df_to_use.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df_to_use.select_dtypes(include=['object', 'category']).columns.tolist()
        all_cols = df_to_use.columns.tolist()
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Select target column
            target_col = st.selectbox(
                "Select Target Column (what you want to predict):",
                all_cols,
                index=0 if st.session_state.target_column is None else all_cols.index(st.session_state.target_column) if st.session_state.target_column in all_cols else 0,
                key="target_column_select"
            )
            
            # Store the selected target column
            st.session_state.target_column = target_col
            
            # Determine problem type
            if target_col in categorical_cols or df_to_use[target_col].nunique() < 10:
                problem_type = "Classification"
                st.info(f"Detected problem type: Classification (predicting categories)")
                add_to_log(f"ML Pipeline: Detected classification problem for target '{target_col}'", is_thinking=True)
            else:
                problem_type = "Regression"
                st.info(f"Detected problem type: Regression (predicting numeric values)")
                add_to_log(f"ML Pipeline: Detected regression problem for target '{target_col}'", is_thinking=True)
            
            # Store problem type
            st.session_state.problem_type = problem_type
        
        with col2:
            # Train-test split options
            test_size = st.slider(
                "Test Set Size (%):",
                min_value=10,
                max_value=40,
                value=int(st.session_state.train_test_split["test_size"] * 100),
                step=5,
                key="test_size_slider"
            )
            
            # Update train-test split settings
            st.session_state.train_test_split["test_size"] = test_size / 100
            
            # Random seed for reproducibility
            random_state = st.number_input(
                "Random Seed:",
                min_value=1,
                max_value=1000,
                value=st.session_state.train_test_split["random_state"],
                key="random_state_input"
            )
            
            # Update random state
            st.session_state.train_test_split["random_state"] = random_state
        
        # Feature selection
        st.markdown("### Feature Selection")
        
        # Remove target column from potential features
        potential_features = [col for col in all_cols if col != target_col]
        
        # Select features to use
        selected_features = st.multiselect(
            "Select Features to Use:",
            potential_features,
            default=potential_features,
            key="feature_select"
        )
        
        # Store selected features
        st.session_state.selected_features = selected_features
        
        # Show data preview with selected features and target
        if selected_features:
            st.markdown("### Data Preview (Selected Features and Target)")
            preview_df = df_to_use[selected_features + [target_col]].head(5)
            st.dataframe(preview_df)
            
            # Proceed button
            if st.button("Proceed to Model Selection", key="proceed_to_model_btn"):
                add_to_log(f"ML Pipeline: Data preparation completed with {len(selected_features)} features and target '{target_col}'")
                st.session_state.active_ml_step = 1  # Move to model selection
                st.experimental_rerun()
        else:
            st.warning("Please select at least one feature to proceed.")
    
    # Add a note about the ML Pipeline being a simulation
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-top: 30px;">
        <p style="margin: 0; color: #555; font-size: 0.9rem;">
            <strong>Note:</strong> This ML Pipeline is a simulation for demonstration purposes. In a real application, it would connect to actual machine learning libraries like scikit-learn, TensorFlow, or PyTorch to train and evaluate models.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Main application layout
def main():
    # Create three columns for the layout
    left_col, main_col, right_col = st.columns([1, 2, 1])
    
    # Left sidebar for dataset selection and available plots
    with left_col:
        st.markdown("<div class='sidebar-header' style='font-size: 1.5rem; font-weight: 600; margin-top: 1rem; margin-bottom: 1rem; color: #2E7D32; border-bottom: 2px solid #4CAF50; padding-bottom: 0.5rem;'>Dataset Selection</div>", unsafe_allow_html=True)
        
        # API Key Configuration
        with st.expander("🔑 Configure OpenAI API Key", expanded=not st.session_state.api_key_configured):
            st.markdown("""
            <div style="font-size: 0.9rem; margin-bottom: 10px;">
                To use AI features, you need to provide your own OpenAI API key. 
                Your key is stored only in your browser session and is never saved on any server.
            </div>
            """, unsafe_allow_html=True)
            
            api_key = st.text_input(
                "Enter your OpenAI API Key:",
                type="password",
                value=st.session_state.openai_api_key if st.session_state.openai_api_key else "",
                help="Your API key is stored only in your current session and not saved anywhere."
            )
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Save API Key"):
                    if api_key and api_key.startswith("sk-"):
                        st.session_state.openai_api_key = api_key
                        st.session_state.api_key_configured = True
                        
                        # Reinitialize AI agent with the new API key
                        st.session_state.ai_agent = MainAIAgent(api_key=api_key)
                        st.success("API key saved successfully!")
                    else:
                        st.error("Invalid API key format. It should start with 'sk-'")
            
            with col2:
                if st.button("Clear API Key"):
                    st.session_state.openai_api_key = None
                    st.session_state.api_key_configured = False
                    st.session_state.ai_agent = MainAIAgent()  # Reinitialize without API key
                    st.info("API key cleared.")
            
            st.markdown("""
            <div style="font-size: 0.8rem; margin-top: 10px; color: #666;">
                <strong>Note:</strong> Without an API key, the application will use simulated AI responses.
                <a href="https://platform.openai.com/api-keys" target="_blank">Get an API key from OpenAI</a>
            </div>
            """, unsafe_allow_html=True)
        
        # Dataset selection options
        dataset_option = st.radio(
            "Choose dataset source:",
            ["Upload your own", "Use predefined dataset"]
        )
        
        if dataset_option == "Upload your own":
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
            if uploaded_file is not None:
                if st.button("Load Dataset"):
                    with st.spinner("Loading and analyzing dataset..."):
                        success = load_and_analyze_dataset(file_buffer=uploaded_file)
                        if success:
                            st.success("Dataset loaded successfully!")
        
        else:  # Use predefined dataset
            predefined_datasets = [
                "iris", "wine", "breast_cancer", "diabetes", 
                "titanic", "tips", "planets"
            ]
            selected_dataset = st.selectbox("Select dataset:", predefined_datasets)
            
            if st.button("Load Dataset"):
                with st.spinner("Loading and analyzing dataset..."):
                    success = load_and_analyze_dataset(dataset_name=selected_dataset)
                    if success:
                        st.success("Dataset loaded successfully!")
        
        # Display sidebar content based on active tab
        if st.session_state.df is not None:
            if st.session_state.active_tab == "eda":
                # EDA sidebar content - available plots
                st.markdown("<div class='sidebar-header' style='font-size: 1.5rem; font-weight: 600; margin-top: 2rem; margin-bottom: 1rem; color: #2E7D32; border-bottom: 2px solid #4CAF50; padding-bottom: 0.5rem;'>Available Plots</div>", unsafe_allow_html=True)
                
                # Create a container with max height and scrolling
                plot_container = st.container()
                
                with plot_container:
                    # Add CSS for scrollable container and fancy plot boxes
                    st.markdown("""
                    <style>
                    .scrollable-container {
                        max-height: 500px;
                        overflow-y: auto;
                        padding-right: 10px;
                    }
                    
                    .plot-category {
                        margin-bottom: 15px;
                        border-bottom: 1px solid #eee;
                        padding-bottom: 5px;
                    }
                    
                    .plot-category-title {
                        font-size: 16px;
                        font-weight: 600;
                        color: #333;
                        margin-bottom: 10px;
                        display: flex;
                        align-items: center;
                    }
                    
                    .fancy-plot-box {
                        border: 1px solid #e0e0e0;
                        border-radius: 8px;
                        padding: 12px;
                        margin-bottom: 10px;
                        text-align: center;
                        background: linear-gradient(to bottom, #ffffff, #f9f9f9);
                        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                        transition: all 0.3s ease;
                    }
                    
                    .fancy-plot-box:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                        border-color: #4CAF50;
                    }
                    
                    .plot-icon {
                        font-size: 24px;
                        margin-bottom: 8px;
                        color: #4CAF50;
                    }
                    
                    .plot-name {
                        font-weight: 500;
                        margin-bottom: 5px;
                        color: #333;
                    }
                    
                    .plot-description {
                        font-size: 12px;
                        color: #666;
                        margin-bottom: 10px;
                        height: 36px;
                        overflow: hidden;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<div class='scrollable-container'>", unsafe_allow_html=True)
                    
                    # Group plots by category with better descriptions
                    plot_categories = [
                        {
                            "name": "Distribution", 
                            "icon": "📈",
                            "plots": [
                                {"type": "histogram", "desc": "Shows frequency distribution of a numeric variable"},
                                {"type": "boxplot", "desc": "Displays median, quartiles and outliers"}
                            ]
                        },
                        {
                            "name": "Relationship", 
                            "icon": "🔗",
                            "plots": [
                                {"type": "scatter", "desc": "Shows relationship between two numeric variables"},
                                {"type": "correlation_heatmap", "desc": "Visualizes correlation matrix between variables"},
                                {"type": "pairplot", "desc": "Creates a matrix of scatter plots for multiple variables"}
                            ]
                        },
                        {
                            "name": "Categorical", 
                            "icon": "📊",
                            "plots": [
                                {"type": "bar", "desc": "Compares values across different categories"},
                                {"type": "pie", "desc": "Shows proportion of categories in a whole"}
                            ]
                        },
                        {
                            "name": "Time Series", 
                            "icon": "⏱️",
                            "plots": [
                                {"type": "line", "desc": "Shows trends over a continuous variable"}
                            ]
                        }
                    ]
                    
                    for category in plot_categories:
                        st.markdown(f"<div class='plot-category'><div class='plot-category-title'>{category['name']}</div>", unsafe_allow_html=True)
                        
                        # Create a grid of 2 columns for the plots
                        plot_cols = st.columns(2)
                        col_idx = 0
                        
                        for plot_info in category['plots']:
                            plot_type = plot_info["type"]
                            plot_desc = plot_info["desc"]
                            
                            with plot_cols[col_idx % 2]:
                                st.markdown(f"""
                                <div class="fancy-plot-box">
                                    <div class="plot-icon">
                                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                            <path d="M3 3V21H21" stroke="#4CAF50" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                            <path d="M7 14L11 10L15 14L19 10" stroke="#4CAF50" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                        </svg>
                                    </div>
                                    <div class="plot-name">{plot_type.capitalize()}</div>
                                    <div class="plot-description">{plot_desc}</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                if st.button(f"Add to Dashboard", key=f"add_{plot_type}"):
                                    # Get configuration for the plot type
                                    config = get_plot_config(plot_type, st.session_state.df)
                                    
                                    # Add to dashboard
                                    add_plot_to_dashboard(plot_type, config)
                                    add_to_log(f"Added {plot_type} plot to dashboard")
                                    st.experimental_rerun()
                            
                            col_idx += 1
                        
                        st.markdown("</div>", unsafe_allow_html=True)  # Close plot-category div
                    
                    st.markdown("</div>", unsafe_allow_html=True)  # Close the scrollable container
            
        # Display reasoning log at the bottom
        st.markdown("<div class='sidebar-header' style='font-size: 1.5rem; font-weight: 600; margin-top: 2rem; margin-bottom: 1rem; color: #2E7D32; border-bottom: 2px solid #4CAF50; padding-bottom: 0.5rem;'>AI Reasoning Log</div>", unsafe_allow_html=True)
        
        # Add CSS for reasoning log styling
        st.markdown("""
        <style>
        .reasoning-log {
            padding: 8px 12px;
            margin-bottom: 8px;
            border-radius: 6px;
            font-size: 0.9rem;
            line-height: 1.4;
            border-left: 3px solid #ccc;
        }
        
        .reasoning-log-thinking {
            background-color: #f0f7ff;
            border-left-color: #2196F3;
        }
        
        .reasoning-log-insight {
            background-color: #f0fff4;
            border-left-color: #4CAF50;
        }
        
        .reasoning-log-recommendation {
            background-color: #fff8e1;
            border-left-color: #FFC107;
        }
        
        .reasoning-log-action {
            background-color: #f5f5f5;
            border-left-color: #9E9E9E;
        }
        
        .reasoning-log-timestamp {
            font-weight: 600;
            color: #555;
            margin-right: 6px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        log_container = st.container()
        with log_container:
            for log_entry in st.session_state.reasoning_log:
                # Parse the log entry to determine its type
                timestamp = log_entry.split(']')[0].strip('[')
                message = log_entry.split(']')[1].strip()
                
                log_class = "reasoning-log-action"  # Default class
                
                if "[AI Thinking]" in message:
                    log_class = "reasoning-log-thinking"
                elif "[AI Insight]" in message:
                    log_class = "reasoning-log-insight"
                elif "[AI Recommendation]" in message:
                    log_class = "reasoning-log-recommendation"
                
                st.markdown(f"""
                <div class="reasoning-log {log_class}">
                    <span class="reasoning-log-timestamp">[{timestamp}]</span>
                    {message}
                </div>
                """, unsafe_allow_html=True)
    
    # Main column for dashboard
    with main_col:
        st.title("Machine Learning Playground, Using AI")
        
        if st.session_state.df is not None:
            st.markdown(f"### Dataset: {st.session_state.dataset_name}")
            st.markdown(f"Rows: {st.session_state.df.shape[0]} | Columns: {st.session_state.df.shape[1]}")
            
            # Create tabs for EDA and Preprocessing
            tabs = st.tabs(["📊 Exploratory Data Analysis", "🔄 Preprocessing Pipeline", "🧠 ML Pipeline"])
            
            # Set active tab based on which tab is selected
            if tabs[0]:
                st.session_state.active_tab = "eda"
            elif tabs[1]:
                st.session_state.active_tab = "preprocessing"
            elif tabs[2]:
                st.session_state.active_tab = "ml"
            
            # EDA Tab
            with tabs[0]:
                st.markdown("<h2 style='font-size: 1.8rem; font-weight: 600; margin-top: 1rem; margin-bottom: 1.5rem; color: #2E7D32; border-bottom: 2px solid #4CAF50; padding-bottom: 0.5rem;'>Your Dashboard</h2>", unsafe_allow_html=True)
                
                # Check if dashboard is empty
                if not st.session_state.dashboard_plots:
                    st.markdown("""
                    <div style="text-align: center; padding: 3rem; background: linear-gradient(to bottom right, #f8f9fa, #e9ecef); border-radius: 12px; margin: 2rem 0; box-shadow: 0 4px 8px rgba(0,0,0,0.05);">
                        <svg width="100" height="100" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-bottom: 1.5rem; opacity: 0.6;">
                            <path d="M3 3V21H21" stroke="#4CAF50" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            <path d="M7 14L11 10L15 14L19 10" stroke="#4CAF50" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            <circle cx="11" cy="10" r="1" fill="#4CAF50"/>
                            <circle cx="15" cy="14" r="1" fill="#4CAF50"/>
                            <circle cx="19" cy="10" r="1" fill="#4CAF50"/>
                            <circle cx="7" cy="14" r="1" fill="#4CAF50"/>
                        </svg>
                        <h3 style="color: #2E7D32; font-weight: 500; margin-bottom: 1rem;">Your dashboard is empty</h3>
                        <p style="color: #555; max-width: 400px; margin: 0 auto; line-height: 1.5;">Add plots from the Available Plots section or use AI recommendations to visualize your data and gain insights.</p>
                        <div style="margin-top: 1.5rem;">
                            <span style="display: inline-block; padding: 0.5rem 1rem; background-color: #4CAF50; color: white; border-radius: 20px; font-size: 0.9rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">Start exploring your data</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Create a container for the dashboard plots
                    st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
                    
                    # Display each plot in the dashboard
                    for i, plot_data in enumerate(st.session_state.dashboard_plots):
                        # Generate a unique key for this plot
                        plot_key = f"dashboard_plot_{i}"
                        
                        # Create a container for this plot
                        st.markdown(f'<div class="dashboard-plot" id="{plot_key}">', unsafe_allow_html=True)
                        
                        # Check if plot_data is a string (plot_id) or a dictionary
                        if isinstance(plot_data, str):
                            # It's a plot_id from the old format
                            if plot_data in st.session_state.plot_configs:
                                plot_config = st.session_state.plot_configs[plot_data]
                                plot_type = plot_config["type"]
                                plot_config_data = plot_config["config"]
                            else:
                                st.error(f"Plot configuration not found for ID: {plot_data}")
                                continue
                        else:
                            # It's already a dictionary with type and config
                            plot_type = plot_data["type"]
                            plot_config_data = plot_data["config"]
                        
                        # Plot header with title and actions
                        st.markdown(f"""
                        <div class="plot-header">
                            <div class="plot-title">{plot_type.capitalize()}</div>
                            <div class="plot-actions">
                                <button onclick="document.getElementById('customize_{i}').style.display = document.getElementById('customize_{i}').style.display === 'none' ? 'block' : 'none';">
                                    ⚙️
                                </button>
                                <button onclick="document.getElementById('insight_{i}').style.display = document.getElementById('insight_{i}').style.display === 'none' ? 'block' : 'none';">
                                    💡
                                </button>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add a direct remove button using Streamlit
                        if st.button("❌ Remove Plot", key=f"remove_plot_{i}"):
                            remove_plot_from_dashboard(i)
                            st.experimental_rerun()
                        
                        # Generate the plot
                        plot_result = generate_plot(st.session_state.df, plot_type, plot_config_data)
                        
                        # Display the plot
                        if "error" in plot_result:
                            st.error(plot_result["error"])
                        else:
                            st.plotly_chart(plot_result["figure"], use_container_width=True)
                        
                        # Customization panel (hidden by default)
                        st.markdown(f"""
                        <div id="customize_{i}" style="display: none; padding: 1rem; background-color: #f8f9fa; border-radius: 8px; margin-top: 1rem;">
                            <h4>Customize Plot</h4>
                            <p>Customization options will appear here.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Insight panel (hidden by default)
                        st.markdown(f"""
                        <div id="insight_{i}" style="display: none; padding: 1rem; background-color: #f8f9fa; border-radius: 8px; margin-top: 1rem;">
                            <h4>AI Insight</h4>
                            <p>Click the button below to generate an AI insight for this plot.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Generate insight button
                        if st.button("Generate Insight", key=f"insight_btn_{i}"):
                            with st.spinner("Generating insight..."):
                                # Get the plot data
                                if isinstance(plot_data, str):
                                    # It's a plot_id from the old format
                                    if plot_data in st.session_state.plot_configs:
                                        plot_config = st.session_state.plot_configs[plot_data]
                                        plot_type = plot_config["type"]
                                        plot_config_data = plot_config["config"]
                                    else:
                                        st.error(f"Plot configuration not found for ID: {plot_data}")
                                        continue
                                else:
                                    # It's already a dictionary with type and config
                                    plot_type = plot_data["type"]
                                    plot_config_data = plot_data["config"]
                                
                                # Add thinking logs
                                add_to_log(f"User requested insight for {plot_type} plot")
                                add_to_log(f"Analyzing {plot_type} visualization data patterns", is_thinking=True)
                                
                                # Add specific thinking based on plot type
                                if plot_type == 'histogram':
                                    add_to_log("Examining distribution characteristics: central tendency, spread, and shape", is_thinking=True)
                                    add_to_log("Checking for normality, skewness, and potential outliers", is_thinking=True)
                                elif plot_type == 'scatter':
                                    add_to_log("Analyzing relationship between variables: direction, strength, and form", is_thinking=True)
                                    add_to_log("Looking for clusters, trends, and potential outliers", is_thinking=True)
                                elif plot_type == 'boxplot':
                                    add_to_log("Examining distribution statistics: median, quartiles, and range", is_thinking=True)
                                    add_to_log("Identifying potential outliers and comparing distributions", is_thinking=True)
                                elif plot_type == 'correlation_heatmap':
                                    add_to_log("Identifying strong positive and negative correlations", is_thinking=True)
                                    add_to_log("Looking for patterns and clusters of related variables", is_thinking=True)
                                elif plot_type == 'bar':
                                    add_to_log("Comparing values across categories and identifying key patterns", is_thinking=True)
                                elif plot_type == 'pie':
                                    add_to_log("Analyzing proportional relationships and dominant categories", is_thinking=True)
                                else:
                                    add_to_log(f"Analyzing patterns and trends in the {plot_type} visualization", is_thinking=True)
                                
                                # Generate the plot
                                plot_result = generate_plot(st.session_state.df, plot_type, plot_config_data)
                                
                                # Generate insight
                                add_to_log("Formulating data-driven insights based on visualization patterns", is_thinking=True)
                                insight = st.session_state.ai_agent.get_eda_agent().generate_insight(
                                    st.session_state.df,
                                    plot_result,
                                    st.session_state.dataset_name
                                )
                                
                                # Log the insight
                                insight_summary = insight.split('.')[0] + '.' if '.' in insight else insight
                                add_to_log(f"Key insight: {insight_summary}", is_insight=True)
                                
                                st.markdown(f"""
                                <div class="insight-container">
                                    <h4>AI Insight</h4>
                                    {insight}
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Close the plot container
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Close the dashboard container
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Preprocessing Tab
            with tabs[1]:
                st.markdown("<h2 style='font-size: 1.8rem; font-weight: 600; margin-top: 1rem; margin-bottom: 1.5rem; color: #2E7D32; border-bottom: 2px solid #4CAF50; padding-bottom: 0.5rem;'>Preprocessing Workflow</h2>", unsafe_allow_html=True)
                
                # Step configuration area
                if "configuring_step" in st.session_state and st.session_state.configuring_step:
                    step_id = st.session_state.configuring_step
                    
                    # Get step configuration UI
                    step_config = get_step_config_ui(step_id, st.session_state.df)
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button("Add to Workflow", key="confirm_step"):
                            add_preprocessing_step(step_config)
                            st.session_state.configuring_step = None
                            st.experimental_rerun()
                    
                    with col2:
                        if st.button("Cancel", key="cancel_step"):
                            st.session_state.configuring_step = None
                            st.experimental_rerun()
                
                # Display current workflow
                else:
                    # Workflow visualization
                    st.markdown("<div class='workflow-container'>", unsafe_allow_html=True)
                    
                    if not st.session_state.preprocessing_workflow:
                        st.info("No preprocessing steps added yet. Select steps from the left sidebar to build your workflow.")
                    else:
                        # Render workflow and get index to remove if any
                        remove_idx = render_workflow(st.session_state.preprocessing_workflow, st.session_state.df)
                        
                        # Handle step removal
                        if remove_idx is not None:
                            st.session_state.preprocessing_workflow.pop(remove_idx)
                            add_preprocessing_message(f"Removed step from workflow")
                            st.experimental_rerun()
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Apply workflow button
                    if st.session_state.preprocessing_workflow:
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            if st.button("Apply Preprocessing", key="apply_workflow"):
                                with st.spinner("Applying preprocessing workflow..."):
                                    success = apply_preprocessing_workflow()
                                    if success:
                                        st.success("Preprocessing workflow applied successfully!")
                        
                        with col2:
                            if st.button("Clear Workflow", key="clear_workflow"):
                                st.session_state.preprocessing_workflow = []
                                st.session_state.processed_df = None
                                add_preprocessing_message("Cleared preprocessing workflow")
                                st.experimental_rerun()
                    
                    # Display processed data if available
                    if st.session_state.processed_df is not None:
                        st.markdown("<div class='sidebar-header'>Processed Dataset</div>", unsafe_allow_html=True)
                        
                        # Display processed data info
                        original_shape = st.session_state.df.shape
                        processed_shape = st.session_state.processed_df.shape
                        
                        st.markdown(f"Original: {original_shape[0]} rows, {original_shape[1]} columns")
                        st.markdown(f"Processed: {processed_shape[0]} rows, {processed_shape[1]} columns")
                        
                        # Display sample of processed data
                        st.dataframe(st.session_state.processed_df.head(10))
                        
                        # Option to download processed data
                        csv = st.session_state.processed_df.to_csv(index=False)
                        st.download_button(
                            label="Download Processed Data",
                            data=csv,
                            file_name=f"processed_{st.session_state.dataset_name}.csv",
                            mime="text/csv"
                        )
                        
                        # Compare with original data
                        with st.expander("Compare with Original Data"):
                            # Show column differences
                            original_cols = set(st.session_state.df.columns)
                            processed_cols = set(st.session_state.processed_df.columns)
                            
                            added_cols = processed_cols - original_cols
                            removed_cols = original_cols - processed_cols
                            
                            if added_cols:
                                st.markdown("##### Added Columns")
                                for col in added_cols:
                                    st.markdown(f"- {col}")
                            
                            if removed_cols:
                                st.markdown("##### Removed Columns")
                                for col in removed_cols:
                                    st.markdown(f"- {col}")
                            
                            # Show row count difference
                            row_diff = processed_shape[0] - original_shape[0]
                            if row_diff != 0:
                                st.markdown(f"##### Row Count Change: {row_diff}")
                                if row_diff < 0:
                                    st.markdown(f"{abs(row_diff)} rows were removed")
                                else:
                                    st.markdown(f"{row_diff} rows were added")
                        
                        # Display preprocessing messages
                        if st.session_state.preprocessing_messages:
                            st.markdown("<div class='sidebar-header'>Preprocessing Log</div>", unsafe_allow_html=True)
                            
                            for msg in st.session_state.preprocessing_messages:
                                if msg["is_error"]:
                                    st.error(f"[{msg['timestamp']}] {msg['message']}")
                                else:
                                    st.markdown(f"<div class='message-container'>[{msg['timestamp']}] {msg['message']}</div>", unsafe_allow_html=True)
            
            # ML Pipeline Tab
            with tabs[2]:
                render_ml_pipeline_tab()

    # Right column for AI recommendations
    with right_col:
        if st.session_state.df is not None:
            st.markdown("<div class='sidebar-header' style='font-size: 1.5rem; font-weight: 600; margin-top: 1rem; margin-bottom: 1rem; color: #2E7D32; border-bottom: 2px solid #4CAF50; padding-bottom: 0.5rem;'>AI Recommendations</div>", unsafe_allow_html=True)
            
            if "recommended_visualizations" in st.session_state and st.session_state.recommended_visualizations:
                for i, recommendation in enumerate(st.session_state.recommended_visualizations):
                    plot_type = recommendation["type"]
                    reason = recommendation["reason"]
                    config = recommendation["config"]
                    
                    # Create a unique key for this recommendation
                    rec_key = f"recommendation_{i}"
                    
                    # Display recommendation in a card
                    st.markdown(f"""
                    <div class="ai-recommendation">
                        <div class="ai-recommendation-title">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" stroke="#4CAF50" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                            {plot_type.capitalize()} Visualization
                            <span class="viz-badge {plot_type}">{plot_type}</span>
                        </div>
                        <div class="ai-recommendation-reason">{reason}</div>
                    """, unsafe_allow_html=True)
                    
                    # Add button to add to dashboard
                    if st.button(f"Add to Dashboard", key=f"add_rec_{i}"):
                        add_plot_to_dashboard(plot_type, config)
                        st.experimental_rerun()
                    
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                if st.session_state.df is not None:
                    st.info("Load a dataset to see AI recommendations.")
                    
            # Add dataset insights section if available
            if "analysis" in st.session_state and st.session_state.analysis:
                st.markdown("<div class='sidebar-header' style='font-size: 1.5rem; font-weight: 600; margin-top: 2rem; margin-bottom: 1rem; color: #2E7D32; border-bottom: 2px solid #4CAF50; padding-bottom: 0.5rem;'>Dataset Insights</div>", unsafe_allow_html=True)
                
                analysis = st.session_state.analysis
                
                # Display dataset summary
                if "summary" in analysis:
                    with st.expander("Dataset Summary", expanded=True):
                        summary = analysis["summary"]
                        st.markdown(f"**Rows:** {summary.get('rows', 'N/A')}")
                        st.markdown(f"**Columns:** {summary.get('columns', 'N/A')}")
                        
                        if "column_types" in summary:
                            st.markdown("**Column Types:**")
                            for col_type, count in summary["column_types"].items():
                                st.markdown(f"- {col_type}: {count}")
                
                # Display data overview
                with st.expander("Data Overview", expanded=True):
                    # Show first few rows of the dataset
                    st.markdown("**Sample Data:**")
                    st.dataframe(st.session_state.df.head(5))
                    
                    # Show descriptive statistics
                    st.markdown("**Descriptive Statistics:**")
                    numeric_cols = st.session_state.df.select_dtypes(include=['number'])
                    if not numeric_cols.empty:
                        st.dataframe(numeric_cols.describe())
                    else:
                        st.info("No numeric columns found for statistics.")
                    
                    # Show column information
                    st.markdown("**Column Information:**")
                    col_info = []
                    for col in st.session_state.df.columns:
                        dtype = str(st.session_state.df[col].dtype)
                        non_nulls = st.session_state.df[col].count()
                        nulls = st.session_state.df[col].isna().sum()
                        unique = st.session_state.df[col].nunique()
                        col_info.append({
                            "Column": col,
                            "Type": dtype,
                            "Non-Nulls": non_nulls,
                            "Nulls": nulls,
                            "Unique Values": unique
                        })
                    st.dataframe(pd.DataFrame(col_info))
                
                # Display data issues if any
                if "issues" in analysis and any(analysis["issues"].values()):
                    with st.expander("Data Issues", expanded=True):
                        issues = analysis["issues"]
                        
                        if issues.get("missing_values"):
                            st.markdown("**Missing Values:**")
                            for col, pct in issues["missing_values"].items():
                                st.markdown(f"- {col}: {pct:.1f}%")
                        
                        if issues.get("outliers"):
                            st.markdown("**Outliers Detected:**")
                            for col in issues["outliers"]:
                                st.markdown(f"- {col}")
                        
                        if issues.get("high_correlation"):
                            st.markdown("**High Correlations:**")
                            for corr in issues["high_correlation"][:5]:  # Show top 5
                                st.markdown(f"- {corr['col1']} & {corr['col2']}: {corr['correlation']:.2f}")
                
                # Display AI insights about the dataset
                if "insights" in analysis:
                    with st.expander("AI Insights", expanded=True):
                        insights = analysis.get("insights", [])
                        if insights:
                            for i, insight in enumerate(insights):
                                st.markdown(f"**Insight {i+1}:** {insight}")
                        else:
                            st.info("No AI insights available for this dataset.")

# Run the application
if __name__ == "__main__":
    main() 
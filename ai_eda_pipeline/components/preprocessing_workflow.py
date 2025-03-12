import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple

# Preprocessing step definitions
PREPROCESSING_STEPS = {
    "impute_missing": {
        "name": "Impute Missing Values",
        "description": "Fill missing values in a column",
        "icon": "🔄",
        "methods": {
            "mean": "Mean (numeric)",
            "median": "Median (numeric)",
            "mode": "Mode (categorical)",
            "constant": "Constant value",
            "knn": "KNN imputation (numeric)",
            "new_category": "New category (categorical)"
        },
        "requires_column": True,
        "category": "Missing Values"
    },
    "drop_column": {
        "name": "Drop Column",
        "description": "Remove a column from the dataset",
        "icon": "🗑️",
        "methods": {},
        "requires_column": True,
        "category": "Feature Selection"
    },
    "handle_outliers": {
        "name": "Handle Outliers",
        "description": "Detect and handle outliers in a column",
        "icon": "📊",
        "methods": {
            "remove": "Remove outliers",
            "winsorize": "Winsorize (clip to boundaries)",
            "transform": "Apply transformation"
        },
        "requires_column": True,
        "category": "Outliers"
    },
    "transform": {
        "name": "Transform Column",
        "description": "Apply mathematical transformation to a column",
        "icon": "📈",
        "methods": {
            "log": "Log transform",
            "sqrt": "Square root transform",
            "box-cox": "Box-Cox transform"
        },
        "requires_column": True,
        "category": "Feature Engineering"
    },
    "convert_type": {
        "name": "Convert Type",
        "description": "Convert column to a different data type",
        "icon": "🔄",
        "methods": {
            "to_numeric": "To numeric",
            "to_categorical": "To categorical",
            "to_datetime": "To datetime"
        },
        "requires_column": True,
        "category": "Data Types"
    },
    "reduce_cardinality": {
        "name": "Reduce Cardinality",
        "description": "Reduce the number of unique values in a categorical column",
        "icon": "📉",
        "methods": {
            "group_rare": "Group rare categories"
        },
        "requires_column": True,
        "category": "Feature Engineering"
    },
    "encode": {
        "name": "Encode Categorical",
        "description": "Encode categorical variables for modeling",
        "icon": "🔠",
        "methods": {
            "label": "Label encoding",
            "onehot": "One-hot encoding",
            "target": "Target encoding"
        },
        "requires_column": True,
        "category": "Feature Engineering"
    },
    "scale": {
        "name": "Scale Features",
        "description": "Scale numeric features",
        "icon": "⚖️",
        "methods": {
            "standard": "Standardization (z-score)",
            "minmax": "Min-Max scaling",
            "robust": "Robust scaling"
        },
        "requires_column": False,
        "category": "Feature Engineering"
    },
    "handle_correlation": {
        "name": "Handle Correlation",
        "description": "Handle highly correlated features",
        "icon": "🔗",
        "methods": {
            "drop_one": "Drop one feature",
            "pca": "Apply PCA"
        },
        "requires_column": False,
        "category": "Feature Selection"
    }
}

# Group preprocessing steps by category
PREPROCESSING_CATEGORIES = {
    "Missing Values": ["impute_missing"],
    "Outliers": ["handle_outliers"],
    "Feature Engineering": ["transform", "reduce_cardinality", "encode", "scale"],
    "Feature Selection": ["drop_column", "handle_correlation"],
    "Data Types": ["convert_type"]
}

def get_preprocessing_steps_by_category():
    """Group preprocessing steps by category."""
    steps_by_category = {}
    for category, step_ids in PREPROCESSING_CATEGORIES.items():
        steps_by_category[category] = [
            {"id": step_id, **PREPROCESSING_STEPS[step_id]}
            for step_id in step_ids
        ]
    return steps_by_category

def get_step_config_ui(step_id: str, df: pd.DataFrame, step_config: Dict = None) -> Dict:
    """
    Generate UI for configuring a preprocessing step.
    
    Args:
        step_id: ID of the preprocessing step
        df: DataFrame to preprocess
        step_config: Existing configuration (if any)
        
    Returns:
        Dictionary with step configuration
    """
    if step_id not in PREPROCESSING_STEPS:
        st.error(f"Unknown preprocessing step: {step_id}")
        return {}
    
    step_info = PREPROCESSING_STEPS[step_id]
    config = step_config.copy() if step_config else {"step": step_id}
    
    # Step title
    st.markdown(f"### {step_info['icon']} {step_info['name']}")
    st.markdown(f"*{step_info['description']}*")
    
    # Column selection if required
    if step_info["requires_column"]:
        if step_id == "handle_correlation":
            # For correlation, select two columns
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) < 2:
                st.warning("Need at least 2 numeric columns for correlation handling")
                return {}
            
            col1 = st.selectbox("First column", numeric_cols, key=f"col1_{step_id}")
            col2 = st.selectbox("Second column", numeric_cols, 
                               index=min(1, len(numeric_cols)-1), key=f"col2_{step_id}")
            
            config["columns"] = [col1, col2]
        
        elif step_id == "scale":
            # For scaling, select multiple columns
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            selected_cols = st.multiselect("Select columns to scale", numeric_cols, 
                                          default=numeric_cols, key=f"cols_{step_id}")
            config["columns"] = selected_cols
        
        else:
            # For other steps, select a single column
            if step_id in ["impute_missing", "handle_outliers", "transform"]:
                # Numeric columns for these operations
                cols = df.select_dtypes(include=['number']).columns.tolist()
                col_type = "numeric"
            elif step_id in ["encode", "reduce_cardinality"]:
                # Categorical columns for these operations
                cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                col_type = "categorical"
            else:
                # Any column type for other operations
                cols = df.columns.tolist()
                col_type = "any"
            
            if not cols:
                st.warning(f"No {col_type} columns available for this operation")
                return {}
            
            column = st.selectbox("Select column", cols, key=f"col_{step_id}")
            config["column"] = column
    
    # Method selection if available
    if step_info["methods"]:
        methods = list(step_info["methods"].keys())
        method_labels = list(step_info["methods"].values())
        
        # Default method based on column type if not already set
        default_idx = 0
        if "method" in config:
            default_idx = methods.index(config["method"]) if config["method"] in methods else 0
        
        method = st.selectbox("Method", methods, 
                             format_func=lambda x: step_info["methods"][x],
                             index=default_idx,
                             key=f"method_{step_id}")
        config["method"] = method
        
        # Additional parameters based on method
        if step_id == "impute_missing" and method == "constant":
            value = st.text_input("Constant value", "0", key=f"value_{step_id}")
            try:
                # Try to convert to appropriate type
                if "column" in config and config["column"] in df.columns:
                    col_type = df[config["column"]].dtype
                    if col_type.kind in 'ifc':  # numeric
                        value = float(value)
                config["value"] = value
            except:
                st.warning("Please enter a valid value")
        
        elif step_id == "reduce_cardinality" and method == "group_rare":
            threshold = st.slider("Threshold (%)", 0.1, 10.0, 1.0, 0.1, key=f"threshold_{step_id}")
            config["threshold"] = threshold / 100  # Convert to proportion
    
    # Reason for step
    reason = st.text_area("Reason for this step", 
                         value=config.get("reason", ""), 
                         key=f"reason_{step_id}")
    config["reason"] = reason
    
    return config

def render_workflow(workflow: List[Dict], df: pd.DataFrame):
    """
    Render the preprocessing workflow as a flowchart.
    
    Args:
        workflow: List of preprocessing steps
        df: DataFrame to preprocess
    """
    if not workflow:
        st.info("No preprocessing steps added yet. Drag steps from the left panel to build your workflow.")
        return
    
    # Display workflow as a flowchart
    st.markdown("### Preprocessing Workflow")
    
    # Create a visual representation of the workflow
    cols = st.columns(min(3, len(workflow)))
    
    for i, step in enumerate(workflow):
        col_idx = i % 3
        with cols[col_idx]:
            step_id = step["step"]
            step_info = PREPROCESSING_STEPS.get(step_id, {})
            
            # Step container
            st.markdown(f"""
            <div style="border:1px solid #ddd; border-radius:5px; padding:10px; margin-bottom:10px; background-color:white;">
                <h4>{step_info.get('icon', '🔧')} {step_info.get('name', step_id)}</h4>
                <p><strong>Column:</strong> {step.get('column', step.get('columns', 'All'))}</p>
                <p><strong>Method:</strong> {step.get('method', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Buttons for each step
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Edit", key=f"edit_{i}"):
                    st.session_state.editing_step = i
            with c2:
                if st.button("Remove", key=f"remove_{i}"):
                    return i  # Return index to remove
    
    # Add arrow between steps
    if len(workflow) > 1:
        st.markdown("""
        <div style="text-align:center; margin:10px 0;">
            <span style="font-size:24px;">⬇️</span>
        </div>
        """, unsafe_allow_html=True)
    
    return None  # No step to remove

def apply_workflow(df: pd.DataFrame, workflow: List[Dict]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply preprocessing workflow to DataFrame.
    
    Args:
        df: Input DataFrame
        workflow: List of preprocessing steps
        
    Returns:
        Tuple of (processed_df, messages)
    """
    from utils.data_utils import apply_preprocessing_step
    
    processed_df = df.copy()
    messages = []
    
    for step in workflow:
        try:
            processed_df, message = apply_preprocessing_step(processed_df, step)
            messages.append(message)
        except Exception as e:
            messages.append(f"Error applying step: {str(e)}")
    
    return processed_df, messages 
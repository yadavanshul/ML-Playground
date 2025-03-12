import pandas as pd
import numpy as np
import io
import os
from typing import Dict, List, Tuple, Optional, Union
import seaborn as sns
from sklearn import datasets

def load_dataset(file_path: Optional[str] = None, file_buffer: Optional[io.BytesIO] = None, 
                 dataset_name: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    """
    Load a dataset from a file path, buffer, or predefined dataset.
    
    Args:
        file_path: Path to the dataset file
        file_buffer: File buffer containing the dataset
        dataset_name: Name of predefined dataset
        
    Returns:
        Tuple of (DataFrame, dataset_name)
    """
    if file_path:
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
        elif file_extension == '.json':
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        dataset_name = os.path.basename(file_path)
        
    elif file_buffer:
        df = pd.read_csv(file_buffer)
        dataset_name = "uploaded_dataset"
        
    elif dataset_name:
        if dataset_name == "iris":
            data = datasets.load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
        elif dataset_name == "boston":
            data = datasets.fetch_california_housing()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
        elif dataset_name == "diabetes":
            data = datasets.load_diabetes()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
        elif dataset_name == "wine":
            data = datasets.load_wine()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
        elif dataset_name == "breast_cancer":
            data = datasets.load_breast_cancer()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
        elif dataset_name == "titanic":
            df = sns.load_dataset("titanic")
        elif dataset_name == "tips":
            df = sns.load_dataset("tips")
        elif dataset_name == "planets":
            df = sns.load_dataset("planets")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    else:
        raise ValueError("Must provide either file_path, file_buffer, or dataset_name")
    
    # Limit to 1000 rows as specified in requirements
    if len(df) > 1000:
        df = df.sample(1000, random_state=42)
        
    return df, dataset_name

def get_dataset_metadata(df: pd.DataFrame) -> Dict:
    """
    Extract metadata from a DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary containing metadata
    """
    metadata = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_columns": list(df.select_dtypes(include=['number']).columns),
        "categorical_columns": list(df.select_dtypes(include=['object', 'category']).columns),
        "datetime_columns": list(df.select_dtypes(include=['datetime']).columns),
        "summary_stats": df.describe().to_dict(),
    }
    
    # Add column-specific metadata
    metadata["column_metadata"] = {}
    for col in df.columns:
        col_meta = {
            "dtype": str(df[col].dtype),
            "missing_count": df[col].isnull().sum(),
            "missing_percentage": (df[col].isnull().sum() / len(df)) * 100,
        }
        
        if df[col].dtype.kind in 'ifc':  # numeric
            col_meta.update({
                "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                "median": float(df[col].median()) if not pd.isna(df[col].median()) else None,
                "std": float(df[col].std()) if not pd.isna(df[col].std()) else None,
                "skew": float(df[col].skew()) if not pd.isna(df[col].skew()) else None,
                "kurtosis": float(df[col].kurtosis()) if not pd.isna(df[col].kurtosis()) else None,
                "is_integer": all(df[col].dropna().apply(lambda x: float(x).is_integer())),
                "zeros_count": (df[col] == 0).sum(),
                "negative_count": (df[col] < 0).sum(),
            })
        elif df[col].dtype == 'object' or df[col].dtype.name == 'category':  # categorical
            value_counts = df[col].value_counts()
            col_meta.update({
                "unique_count": df[col].nunique(),
                "top_values": value_counts.head(5).to_dict(),
                "is_binary": df[col].nunique() == 2,
            })
        
        metadata["column_metadata"][col] = col_meta
    
    return metadata

def detect_data_issues(df: pd.DataFrame) -> Dict:
    """
    Detect common issues in the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary of detected issues
    """
    issues = {
        "missing_values": {},
        "outliers": {},
        "inconsistent_types": {},
        "high_cardinality": {},
        "imbalanced_categories": {},
        "zero_variance": [],
        "high_correlation": [],
        "potential_id_columns": [],
    }
    
    # Missing values
    missing_vals = df.isnull().sum()
    missing_percent = (missing_vals / len(df)) * 100
    issues["missing_values"] = {col: {"count": int(count), "percent": float(percent)} 
                               for col, (count, percent) in 
                               zip(missing_vals.index, zip(missing_vals, missing_percent)) 
                               if count > 0}
    
    # Outliers (using IQR method for numeric columns)
    for col in df.select_dtypes(include=['number']).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        if len(outliers) > 0:
            issues["outliers"][col] = {
                "count": len(outliers),
                "percent": (len(outliers) / len(df)) * 100,
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound)
            }
    
    # Inconsistent types (e.g., mixed numeric and string in object columns)
    for col in df.select_dtypes(include=['object']).columns:
        # Check if column contains mixed numeric and non-numeric values
        numeric_count = sum(pd.to_numeric(df[col], errors='coerce').notna())
        if 0 < numeric_count < len(df[col].dropna()):
            issues["inconsistent_types"][col] = {
                "numeric_count": numeric_count,
                "non_numeric_count": len(df[col].dropna()) - numeric_count,
                "percent_numeric": (numeric_count / len(df[col].dropna())) * 100
            }
    
    # High cardinality categorical columns
    for col in df.select_dtypes(include=['object', 'category']).columns:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.5 and df[col].nunique() > 10:
            issues["high_cardinality"][col] = {
                "unique_count": df[col].nunique(),
                "unique_ratio": unique_ratio
            }
            
            # Potential ID columns
            if unique_ratio > 0.9:
                issues["potential_id_columns"].append(col)
    
    # Imbalanced categories
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if df[col].nunique() < 10:  # Only check columns with reasonable number of categories
            value_counts = df[col].value_counts(normalize=True)
            if value_counts.max() > 0.8:  # If dominant category > 80%
                issues["imbalanced_categories"][col] = {
                    "dominant_category": value_counts.idxmax(),
                    "dominant_percent": float(value_counts.max() * 100)
                }
    
    # Zero variance columns
    for col in df.columns:
        if df[col].nunique() <= 1:
            issues["zero_variance"].append(col)
    
    # High correlation between numeric features
    if len(df.select_dtypes(include=['number']).columns) > 1:
        corr_matrix = df.select_dtypes(include=['number']).corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_pairs = [(i, j, corr_matrix.loc[i, j]) 
                          for i in upper_tri.index 
                          for j in upper_tri.columns 
                          if upper_tri.loc[i, j] > 0.9]
        
        if high_corr_pairs:
            issues["high_correlation"] = [
                {"col1": col1, "col2": col2, "correlation": float(corr)}
                for col1, col2, corr in high_corr_pairs
            ]
    
    return issues

def suggest_preprocessing_steps(df: pd.DataFrame, issues: Dict) -> List[Dict]:
    """
    Suggest preprocessing steps based on dataset and detected issues.
    
    Args:
        df: Input DataFrame
        issues: Dictionary of detected issues
        
    Returns:
        List of suggested preprocessing steps
    """
    suggestions = []
    
    # Missing values handling
    if issues["missing_values"]:
        for col, info in issues["missing_values"].items():
            if info["percent"] < 5:  # Low missing percentage
                if df[col].dtype.kind in 'ifc':  # numeric
                    suggestions.append({
                        "step": "impute_missing",
                        "column": col,
                        "method": "median",
                        "reason": f"Fill {info['percent']:.1f}% missing values with median (low missing rate)"
                    })
                else:  # categorical
                    suggestions.append({
                        "step": "impute_missing",
                        "column": col,
                        "method": "mode",
                        "reason": f"Fill {info['percent']:.1f}% missing values with mode (low missing rate)"
                    })
            elif info["percent"] < 30:  # Moderate missing percentage
                if df[col].dtype.kind in 'ifc':  # numeric
                    suggestions.append({
                        "step": "impute_missing",
                        "column": col,
                        "method": "knn",
                        "reason": f"Fill {info['percent']:.1f}% missing values with KNN imputation"
                    })
                else:  # categorical
                    suggestions.append({
                        "step": "impute_missing",
                        "column": col,
                        "method": "new_category",
                        "reason": f"Create new category for {info['percent']:.1f}% missing values"
                    })
            else:  # High missing percentage
                suggestions.append({
                    "step": "drop_column",
                    "column": col,
                    "reason": f"Drop column with {info['percent']:.1f}% missing values"
                })
    
    # Outlier handling
    if issues["outliers"]:
        for col, info in issues["outliers"].items():
            if info["percent"] < 1:  # Very few outliers
                suggestions.append({
                    "step": "handle_outliers",
                    "column": col,
                    "method": "remove",
                    "reason": f"Remove {info['count']} outliers ({info['percent']:.1f}%)"
                })
            elif info["percent"] < 5:  # Some outliers
                suggestions.append({
                    "step": "handle_outliers",
                    "column": col,
                    "method": "winsorize",
                    "reason": f"Winsorize {info['count']} outliers ({info['percent']:.1f}%)"
                })
            else:  # Many outliers - might be a legitimate distribution
                suggestions.append({
                    "step": "transform",
                    "column": col,
                    "method": "log",
                    "reason": f"Apply log transform to handle skewed distribution with {info['percent']:.1f}% outliers"
                })
    
    # Inconsistent types
    if issues["inconsistent_types"]:
        for col, info in issues["inconsistent_types"].items():
            if info["percent_numeric"] > 80:  # Mostly numeric
                suggestions.append({
                    "step": "convert_type",
                    "column": col,
                    "method": "to_numeric",
                    "reason": f"Convert to numeric ({info['percent_numeric']:.1f}% are numeric values)"
                })
            else:
                suggestions.append({
                    "step": "convert_type",
                    "column": col,
                    "method": "to_categorical",
                    "reason": "Convert to categorical (mixed types)"
                })
    
    # High cardinality
    if issues["high_cardinality"]:
        for col, info in issues["high_cardinality"].items():
            if col not in issues["potential_id_columns"]:
                suggestions.append({
                    "step": "reduce_cardinality",
                    "column": col,
                    "method": "group_rare",
                    "reason": f"Group rare categories ({info['unique_count']} unique values)"
                })
    
    # Potential ID columns
    for col in issues["potential_id_columns"]:
        suggestions.append({
            "step": "drop_column",
            "column": col,
            "reason": "Potential ID column with high cardinality"
        })
    
    # Zero variance
    for col in issues["zero_variance"]:
        suggestions.append({
            "step": "drop_column",
            "column": col,
            "reason": "Zero variance column (constant value)"
        })
    
    # High correlation
    if issues["high_correlation"]:
        for corr_info in issues["high_correlation"]:
            suggestions.append({
                "step": "handle_correlation",
                "columns": [corr_info["col1"], corr_info["col2"]],
                "method": "drop_one",
                "reason": f"High correlation ({corr_info['correlation']:.2f}) between features"
            })
    
    # Encoding categorical variables
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if col not in [s["column"] for s in suggestions if s.get("step") == "drop_column"]:
            if df[col].nunique() == 2:
                suggestions.append({
                    "step": "encode",
                    "column": col,
                    "method": "label",
                    "reason": "Binary categorical variable - use label encoding"
                })
            elif df[col].nunique() <= 10:
                suggestions.append({
                    "step": "encode",
                    "column": col,
                    "method": "onehot",
                    "reason": f"Categorical with {df[col].nunique()} categories - use one-hot encoding"
                })
            else:
                suggestions.append({
                    "step": "encode",
                    "column": col,
                    "method": "target",
                    "reason": f"High cardinality categorical ({df[col].nunique()} categories) - use target encoding"
                })
    
    # Scaling numeric features
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        # Check if ranges are very different
        ranges = {col: df[col].max() - df[col].min() for col in numeric_cols if not pd.isna(df[col].max()) and not pd.isna(df[col].min())}
        if ranges and max(ranges.values()) / min(ranges.values()) > 10:
            suggestions.append({
                "step": "scale",
                "columns": list(numeric_cols),
                "method": "standard",
                "reason": "Features have very different scales - use standardization"
            })
    
    return suggestions

def apply_preprocessing_step(df: pd.DataFrame, step: Dict) -> Tuple[pd.DataFrame, str]:
    """
    Apply a preprocessing step to the DataFrame.
    
    Args:
        df: Input DataFrame
        step: Preprocessing step configuration
        
    Returns:
        Tuple of (processed_df, message)
    """
    step_type = step.get("step", "")
    
    if step_type == "impute_missing":
        col = step.get("column")
        method = step.get("method", "median")
        
        if method == "median" and col in df.select_dtypes(include=['number']).columns:
            df[col] = df[col].fillna(df[col].median())
            return df, f"Imputed missing values in '{col}' with median"
        
        elif method == "mean" and col in df.select_dtypes(include=['number']).columns:
            df[col] = df[col].fillna(df[col].mean())
            return df, f"Imputed missing values in '{col}' with mean"
        
        elif method == "mode":
            df[col] = df[col].fillna(df[col].mode()[0])
            return df, f"Imputed missing values in '{col}' with mode"
        
        elif method == "constant":
            value = step.get("value", 0)
            df[col] = df[col].fillna(value)
            return df, f"Imputed missing values in '{col}' with constant ({value})"
        
        elif method == "new_category" and col in df.select_dtypes(include=['object', 'category']).columns:
            df[col] = df[col].fillna("Missing")
            return df, f"Replaced missing values in '{col}' with 'Missing' category"
        
        elif method == "knn":
            # Simple implementation - in practice would use more sophisticated KNN imputation
            from sklearn.impute import KNNImputer
            numeric_cols = df.select_dtypes(include=['number']).columns
            if col in numeric_cols:
                imputer = KNNImputer(n_neighbors=5)
                df[col] = imputer.fit_transform(df[[col]])[:, 0]
                return df, f"Imputed missing values in '{col}' with KNN"
            else:
                return df, f"KNN imputation not applicable for non-numeric column '{col}'"
        
        else:
            return df, f"Invalid imputation method '{method}' for column '{col}'"
    
    elif step_type == "drop_column":
        col = step.get("column")
        if col in df.columns:
            df = df.drop(columns=[col])
            return df, f"Dropped column '{col}'"
        else:
            return df, f"Column '{col}' not found"
    
    elif step_type == "handle_outliers":
        col = step.get("column")
        method = step.get("method", "winsorize")
        
        if col not in df.select_dtypes(include=['number']).columns:
            return df, f"Outlier handling not applicable for non-numeric column '{col}'"
        
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        if method == "remove":
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            return df, f"Removed outliers from '{col}' (outside {lower_bound:.2f}-{upper_bound:.2f})"
        
        elif method == "winsorize":
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            return df, f"Winsorized outliers in '{col}' to range {lower_bound:.2f}-{upper_bound:.2f}"
        
        else:
            return df, f"Invalid outlier handling method '{method}'"
    
    elif step_type == "transform":
        col = step.get("column")
        method = step.get("method", "log")
        
        if col not in df.select_dtypes(include=['number']).columns:
            return df, f"Transformation not applicable for non-numeric column '{col}'"
        
        if method == "log":
            # Handle zero/negative values
            min_val = df[col].min()
            if min_val <= 0:
                shift = abs(min_val) + 1
                df[col] = np.log(df[col] + shift)
                return df, f"Applied log transform to '{col}' with shift of {shift}"
            else:
                df[col] = np.log(df[col])
                return df, f"Applied log transform to '{col}'"
        
        elif method == "sqrt":
            # Handle negative values
            min_val = df[col].min()
            if min_val < 0:
                shift = abs(min_val)
                df[col] = np.sqrt(df[col] + shift)
                return df, f"Applied sqrt transform to '{col}' with shift of {shift}"
            else:
                df[col] = np.sqrt(df[col])
                return df, f"Applied sqrt transform to '{col}'"
        
        elif method == "box-cox":
            from scipy import stats
            # Box-Cox requires positive values
            min_val = df[col].min()
            if min_val <= 0:
                shift = abs(min_val) + 1
                df[col], _ = stats.boxcox(df[col] + shift)
                return df, f"Applied Box-Cox transform to '{col}' with shift of {shift}"
            else:
                df[col], _ = stats.boxcox(df[col])
                return df, f"Applied Box-Cox transform to '{col}'"
        
        else:
            return df, f"Invalid transformation method '{method}'"
    
    elif step_type == "convert_type":
        col = step.get("column")
        method = step.get("method")
        
        if method == "to_numeric":
            df[col] = pd.to_numeric(df[col], errors='coerce')
            return df, f"Converted '{col}' to numeric"
        
        elif method == "to_categorical":
            df[col] = df[col].astype('category')
            return df, f"Converted '{col}' to categorical"
        
        elif method == "to_datetime":
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                return df, f"Converted '{col}' to datetime"
            except:
                return df, f"Failed to convert '{col}' to datetime"
        
        else:
            return df, f"Invalid type conversion method '{method}'"
    
    elif step_type == "reduce_cardinality":
        col = step.get("column")
        method = step.get("method", "group_rare")
        threshold = step.get("threshold", 0.01)
        
        if method == "group_rare":
            value_counts = df[col].value_counts(normalize=True)
            rare_categories = value_counts[value_counts < threshold].index
            df[col] = df[col].replace(rare_categories, 'Other')
            return df, f"Grouped rare categories in '{col}' (frequency < {threshold*100}%) as 'Other'"
        
        else:
            return df, f"Invalid cardinality reduction method '{method}'"
    
    elif step_type == "encode":
        col = step.get("column")
        method = step.get("method", "onehot")
        
        if col not in df.select_dtypes(include=['object', 'category']).columns:
            return df, f"Encoding not applicable for non-categorical column '{col}'"
        
        if method == "label":
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            return df, f"Applied label encoding to '{col}'"
        
        elif method == "onehot":
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            return df, f"Applied one-hot encoding to '{col}'"
        
        elif method == "target":
            # Simple mean target encoding - in practice would use more sophisticated methods
            if 'target' in df.columns and df['target'].dtype.kind in 'ifc':
                means = df.groupby(col)['target'].mean()
                df[f"{col}_encoded"] = df[col].map(means)
                df = df.drop(columns=[col])
                return df, f"Applied target encoding to '{col}'"
            else:
                return df, f"Target encoding requires a numeric 'target' column"
        
        else:
            return df, f"Invalid encoding method '{method}'"
    
    elif step_type == "scale":
        columns = step.get("columns", [])
        method = step.get("method", "standard")
        
        # Filter to only include numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        columns = [col for col in columns if col in numeric_cols]
        
        if not columns:
            return df, "No valid numeric columns to scale"
        
        if method == "standard":
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            df[columns] = scaler.fit_transform(df[columns])
            return df, f"Applied standardization to {len(columns)} columns"
        
        elif method == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            df[columns] = scaler.fit_transform(df[columns])
            return df, f"Applied min-max scaling to {len(columns)} columns"
        
        elif method == "robust":
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            df[columns] = scaler.fit_transform(df[columns])
            return df, f"Applied robust scaling to {len(columns)} columns"
        
        else:
            return df, f"Invalid scaling method '{method}'"
    
    elif step_type == "handle_correlation":
        columns = step.get("columns", [])
        method = step.get("method", "drop_one")
        
        if len(columns) != 2:
            return df, "Correlation handling requires exactly 2 columns"
        
        col1, col2 = columns
        
        if method == "drop_one":
            # Drop the second column by default
            df = df.drop(columns=[col2])
            return df, f"Dropped '{col2}' due to high correlation with '{col1}'"
        
        elif method == "pca":
            from sklearn.decomposition import PCA
            if col1 in df.columns and col2 in df.columns:
                pca = PCA(n_components=1)
                df[f"{col1}_{col2}_pca"] = pca.fit_transform(df[[col1, col2]])
                df = df.drop(columns=[col1, col2])
                return df, f"Combined '{col1}' and '{col2}' using PCA"
            else:
                return df, f"One or both columns not found: {col1}, {col2}"
        
        else:
            return df, f"Invalid correlation handling method '{method}'"
    
    else:
        return df, f"Unknown preprocessing step type: {step_type}" 
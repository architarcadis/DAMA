import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import time
import io
from datetime import datetime

def perform_data_quality_assessment(data_dict, assessment_options, selected_attributes=None, attribute_combinations=None, progress_callback=None):
    """
    Perform comprehensive data quality assessment on all sheets in the data dictionary.
    
    Args:
        data_dict: Dictionary with sheet names as keys and dataframes as values
        assessment_options: Dictionary of assessment types to perform
        selected_attributes: List of specific attributes/columns to assess (if None, assess all)
        attribute_combinations: List of tuples of column names to assess together
        progress_callback: Callback function for progress updates
    
    Returns:
        Dictionary containing assessment results for each sheet
    """
    results = {}
    sheet_count = len(data_dict)
    
    for i, (sheet_name, df) in enumerate(data_dict.items()):
        if progress_callback:
            progress_callback((i / sheet_count) * 100, f"Analyzing sheet: {sheet_name}")
        
        sheet_results = {}
        
        # Filter dataframe to selected attributes if specified
        if selected_attributes:
            # Only keep columns that exist in the dataframe
            valid_columns = [col for col in selected_attributes if col in df.columns]
            if valid_columns:
                assessment_df = df[valid_columns].copy()
            else:
                assessment_df = df.copy()  # Use all if none of the selected columns exist
        else:
            assessment_df = df.copy()
        
        # Basic statistics about the dataset
        sheet_results["basic_stats"] = {
            "row_count": df.shape[0],
            "column_count": df.shape[1],
            "assessed_columns": assessment_df.shape[1],
            "memory_usage": df.memory_usage(deep=True).sum() / (1024 * 1024),  # in MB
        }
        
        # Column metadata
        sheet_results["columns"] = {
            col: {
                "dtype": str(df[col].dtype),
                "is_numeric": pd.api.types.is_numeric_dtype(df[col]),
                "is_datetime": pd.api.types.is_datetime64_dtype(df[col]),
                "is_categorical": pd.api.types.is_categorical_dtype(df[col]),
                "is_selected": selected_attributes is None or col in selected_attributes
            }
            for col in df.columns
        }
        
        # Identify likely date columns (even if not properly formatted as dates)
        date_pattern = re.compile(r'^(\d{1,4}[-/]\d{1,2}[-/]\d{1,4}|\d{2,4}[-/]\d{2}|\d{8}|\d{6})$')
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check a sample of non-null values
                sample = df[col].dropna().astype(str).sample(min(100, len(df[col].dropna())))
                date_matches = [bool(date_pattern.match(val)) for val in sample]
                sheet_results["columns"][col]["likely_date"] = sum(date_matches) / len(sample) > 0.7 if sample.size > 0 else False
            else:
                sheet_results["columns"][col]["likely_date"] = False
        
        # Perform selected assessments on the filtered dataframe (attribute level)
        if assessment_options.get("completeness", True):
            sheet_results["completeness"] = assess_completeness(assessment_df)
            
        if assessment_options.get("consistency", True):
            sheet_results["consistency"] = assess_consistency(assessment_df, sheet_results["columns"])
            
        if assessment_options.get("accuracy", True):
            sheet_results["accuracy"] = assess_accuracy(assessment_df, sheet_results["columns"])
            
        if assessment_options.get("uniqueness", True):
            sheet_results["uniqueness"] = assess_uniqueness(assessment_df)
            
        if assessment_options.get("timeliness", True):
            sheet_results["timeliness"] = assess_timeliness(assessment_df, sheet_results["columns"])
            
        if assessment_options.get("validity", True):
            sheet_results["validity"] = assess_validity(assessment_df, sheet_results["columns"])
        
        # Process attribute combinations if specified
        if attribute_combinations:
            sheet_results["attribute_combinations"] = {}
            
            for i, combo in enumerate(attribute_combinations):
                valid_combo = [col for col in combo if col in df.columns]
                
                if len(valid_combo) >= 2:  # Need at least 2 columns for a combination
                    combo_key = "_".join(valid_combo)
                    combo_df = df[valid_combo].copy()
                    
                    # Analyze the relationship and interdependencies between these attributes
                    combo_results = analyze_attribute_combination(combo_df, valid_combo)
                    sheet_results["attribute_combinations"][combo_key] = combo_results
        
        # Generate visualizations
        sheet_results["visualizations"] = generate_visualizations(df, sheet_results)
        
        # Add to overall results
        results[sheet_name] = sheet_results
    
    # Create summary of results across all sheets
    results["summary"] = create_summary(results)
    
    return results

def assess_completeness(df):
    """Analyze completeness of data - missing values, empty fields"""
    results = {}
    
    # Overall completeness
    total_cells = df.size
    missing_cells = df.isna().sum().sum()
    completeness_pct = 100 * (1 - missing_cells / total_cells) if total_cells > 0 else 100
    
    results["overall"] = {
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "completeness_percentage": completeness_pct,
        "completeness_score": calculate_score(completeness_pct, [95, 98, 99.5]),
    }
    
    # Column-wise completeness
    col_results = {}
    for col in df.columns:
        missing = df[col].isna().sum()
        total = len(df[col])
        completeness = 100 * (1 - missing / total) if total > 0 else 100
        
        col_results[col] = {
            "missing_count": missing,
            "completeness_percentage": completeness,
            "completeness_score": calculate_score(completeness, [95, 98, 99.5]),
        }
    
    results["columns"] = col_results
    
    # Top incomplete columns
    incomplete_cols = {col: data["completeness_percentage"] 
                      for col, data in col_results.items() 
                      if data["completeness_percentage"] < 100}
    results["top_incomplete"] = dict(sorted(incomplete_cols.items(), 
                                          key=lambda x: x[1])[:5])
    
    return results

def assess_consistency(df, column_metadata):
    """Analyze consistency of data - data type consistency, value ranges"""
    results = {}
    
    # Check for mixed data types within columns
    type_consistency = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            # For object columns, check for mixed types
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                # Get types of a sample of values
                sample = non_null_values.sample(min(1000, len(non_null_values)))
                types = [type(val).__name__ for val in sample]
                unique_types = set(types)
                
                type_consistency[col] = {
                    "mixed_types": len(unique_types) > 1,
                    "types_found": list(unique_types),
                    "consistency_score": 100 if len(unique_types) <= 1 else (100 - (len(unique_types) - 1) * 20)
                }
            else:
                type_consistency[col] = {
                    "mixed_types": False,
                    "types_found": [],
                    "consistency_score": 100
                }
        else:
            # For non-object columns, type is consistent
            type_consistency[col] = {
                "mixed_types": False,
                "types_found": [str(df[col].dtype)],
                "consistency_score": 100
            }
    
    results["type_consistency"] = type_consistency
    
    # Check value range consistency for numeric columns
    value_consistency = {}
    for col in df.columns:
        if column_metadata[col]["is_numeric"]:
            non_null = df[col].dropna()
            if len(non_null) > 0:
                q1 = non_null.quantile(0.25)
                q3 = non_null.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = ((non_null < lower_bound) | (non_null > upper_bound)).sum()
                outlier_pct = 100 * outliers / len(non_null)
                
                value_consistency[col] = {
                    "min": float(non_null.min()),
                    "max": float(non_null.max()),
                    "mean": float(non_null.mean()),
                    "median": float(non_null.median()),
                    "std": float(non_null.std()) if len(non_null) > 1 else 0,
                    "outlier_count": int(outliers),
                    "outlier_percentage": float(outlier_pct),
                    "consistency_score": calculate_score(100 - outlier_pct, [95, 98, 99])
                }
            else:
                value_consistency[col] = {
                    "min": None,
                    "max": None,
                    "mean": None,
                    "median": None,
                    "std": None,
                    "outlier_count": 0,
                    "outlier_percentage": 0,
                    "consistency_score": 100
                }
    
    results["value_consistency"] = value_consistency
    
    # Overall consistency score
    type_scores = [data["consistency_score"] for data in type_consistency.values()]
    value_scores = [data["consistency_score"] for data in value_consistency.values() if "consistency_score" in data]
    
    results["overall"] = {
        "type_consistency_score": sum(type_scores) / len(type_scores) if type_scores else 100,
        "value_consistency_score": sum(value_scores) / len(value_scores) if value_scores else 100,
        "overall_consistency_score": (sum(type_scores + value_scores) / 
                                     len(type_scores + value_scores)) if (type_scores + value_scores) else 100
    }
    
    return results

def assess_accuracy(df, column_metadata):
    """Assess accuracy of data - outlier detection, value distribution"""
    results = {}
    
    # Analyze numeric columns for outliers using z-score method
    numeric_accuracy = {}
    for col in df.columns:
        if column_metadata[col]["is_numeric"]:
            data = df[col].dropna()
            if len(data) > 0:
                mean = data.mean()
                std = data.std()
                
                if std > 0:
                    z_scores = abs((data - mean) / std)
                    outliers = (z_scores > 3).sum()
                    outlier_pct = 100 * outliers / len(data)
                else:
                    # If std is 0, all values are the same
                    outliers = 0
                    outlier_pct = 0
                
                numeric_accuracy[col] = {
                    "outlier_count": int(outliers),
                    "outlier_percentage": float(outlier_pct),
                    "accuracy_score": calculate_score(100 - outlier_pct, [95, 98, 99])
                }
    
    results["numeric_accuracy"] = numeric_accuracy
    
    # Check for potential formatting errors in text fields
    text_accuracy = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            data = df[col].dropna().astype(str)
            if len(data) > 0:
                # Check for inconsistent capitalization
                lower_count = (data.str.islower()).sum()
                upper_count = (data.str.isupper()).sum()
                title_count = (data.str.istitle()).sum()
                mixed_count = len(data) - lower_count - upper_count - title_count
                
                # Calculate consistency ratio
                max_style = max(lower_count, upper_count, title_count)
                style_consistency = 100 * max_style / len(data) if len(data) > 0 else 100
                
                text_accuracy[col] = {
                    "style_consistency_percentage": float(style_consistency),
                    "accuracy_score": calculate_score(style_consistency, [70, 85, 95])
                }
    
    results["text_accuracy"] = text_accuracy
    
    # Detect anomalies in distribution (for numeric columns)
    distribution_anomalies = {}
    for col in df.columns:
        if column_metadata[col]["is_numeric"]:
            data = df[col].dropna()
            if len(data) >= 30:  # Need sufficient data for meaningful distribution analysis
                # Check for skewness
                skewness = data.skew()
                # Check for multimodality using a simple histogram
                hist, bin_edges = np.histogram(data, bins='auto')
                peaks = 0
                for i in range(1, len(hist)-1):
                    if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                        peaks += 1
                
                distribution_anomalies[col] = {
                    "skewness": float(skewness),
                    "potential_modes": peaks,
                    "distribution_quality": "Good" if abs(skewness) < 1 and peaks <= 2 else 
                                           "Fair" if abs(skewness) < 2 and peaks <= 3 else 
                                           "Poor"
                }
    
    results["distribution_anomalies"] = distribution_anomalies
    
    # Calculate overall accuracy score
    numeric_scores = [data["accuracy_score"] for data in numeric_accuracy.values()]
    text_scores = [data["accuracy_score"] for data in text_accuracy.values()]
    all_scores = numeric_scores + text_scores
    
    results["overall"] = {
        "numeric_accuracy_score": sum(numeric_scores) / len(numeric_scores) if numeric_scores else 100,
        "text_accuracy_score": sum(text_scores) / len(text_scores) if text_scores else 100,
        "overall_accuracy_score": sum(all_scores) / len(all_scores) if all_scores else 100
    }
    
    return results

def assess_uniqueness(df):
    """Analyze uniqueness of data - duplicate detection"""
    results = {}
    
    # Count overall duplicates
    total_rows = len(df)
    duplicate_rows = df.duplicated().sum()
    uniqueness_pct = 100 * (1 - duplicate_rows / total_rows) if total_rows > 0 else 100
    
    results["overall"] = {
        "total_rows": total_rows,
        "duplicate_rows": int(duplicate_rows),
        "uniqueness_percentage": float(uniqueness_pct),
        "uniqueness_score": calculate_score(uniqueness_pct, [95, 98, 99.5])
    }
    
    # Analyze potential key columns
    key_candidates = []
    key_column_stats = {}
    
    for col in df.columns:
        non_null_values = df[col].dropna()
        unique_values = non_null_values.nunique()
        total_values = len(non_null_values)
        
        if total_values > 0:
            uniqueness_ratio = unique_values / total_values
            
            key_column_stats[col] = {
                "unique_values": int(unique_values),
                "total_values": int(total_values),
                "uniqueness_ratio": float(uniqueness_ratio),
                "is_unique": uniqueness_ratio == 1
            }
            
            if uniqueness_ratio == 1 and total_values == total_rows:
                key_candidates.append(col)
    
    results["column_uniqueness"] = key_column_stats
    results["key_candidates"] = key_candidates
    
    # Identify columns with high cardinality but not unique (potential data issues)
    high_cardinality_non_keys = []
    for col, stats in key_column_stats.items():
        if stats["uniqueness_ratio"] > 0.9 and stats["uniqueness_ratio"] < 1 and stats["total_values"] > 100:
            high_cardinality_non_keys.append({
                "column": col,
                "uniqueness_ratio": stats["uniqueness_ratio"],
                "unique_values": stats["unique_values"]
            })
    
    results["high_cardinality_non_keys"] = high_cardinality_non_keys
    
    return results

def assess_timeliness(df, column_metadata):
    """Assess timeliness of data for date fields"""
    results = {}
    date_columns = []
    
    # Identify datetime columns
    for col in df.columns:
        if column_metadata[col]["is_datetime"] or column_metadata[col]["likely_date"]:
            date_columns.append(col)
    
    if not date_columns:
        results["overall"] = {
            "date_columns_found": 0,
            "timeliness_score": None,
            "timeliness_message": "No date columns identified for timeliness assessment."
        }
        return results
    
    # Analyze each date column
    column_timeliness = {}
    current_date = datetime.now()
    
    for col in date_columns:
        # Convert to datetime if not already
        if column_metadata[col]["is_datetime"]:
            date_series = pd.to_datetime(df[col], errors='coerce')
        else:
            # Try multiple date formats for conversion
            try:
                date_series = pd.to_datetime(df[col], errors='coerce')
            except:
                date_series = pd.Series([pd.NaT] * len(df))
        
        valid_dates = date_series.dropna()
        
        if len(valid_dates) > 0:
            min_date = valid_dates.min()
            max_date = valid_dates.max()
            
            # Calculate time span and recency
            time_span_days = (max_date - min_date).days if max_date > min_date else 0
            recency_days = (current_date - max_date).days
            
            timeliness_score = 100
            if recency_days > 365:  # More than a year old
                timeliness_score = max(0, 100 - (recency_days - 365) / 10)
            elif recency_days > 180:  # More than 6 months old
                timeliness_score = max(70, 100 - (recency_days - 180) / 4)
            elif recency_days > 30:   # More than a month old
                timeliness_score = max(85, 100 - (recency_days - 30) / 5)
            
            column_timeliness[col] = {
                "min_date": min_date.strftime('%Y-%m-%d'),
                "max_date": max_date.strftime('%Y-%m-%d'),
                "time_span_days": time_span_days,
                "recency_days": recency_days,
                "timeliness_score": timeliness_score,
                "timeliness_category": "Current" if recency_days <= 30 else
                                     "Recent" if recency_days <= 180 else
                                     "Outdated" if recency_days <= 365 else
                                     "Obsolete"
            }
        else:
            column_timeliness[col] = {
                "min_date": None,
                "max_date": None,
                "time_span_days": 0,
                "recency_days": None,
                "timeliness_score": 0,
                "timeliness_category": "Invalid Dates"
            }
    
    results["column_timeliness"] = column_timeliness
    
    # Calculate overall timeliness score
    valid_scores = [data["timeliness_score"] for data in column_timeliness.values() 
                   if data["timeliness_score"] is not None]
    
    results["overall"] = {
        "date_columns_found": len(date_columns),
        "timeliness_score": sum(valid_scores) / len(valid_scores) if valid_scores else None,
        "timeliness_message": "Timeliness assessment completed" if valid_scores else 
                             "No valid date data for timeliness assessment."
    }
    
    return results

def assess_validity(df, column_metadata):
    """Validate format and business rule compliance"""
    results = {}
    
    # Check validity of different data types
    validity_checks = {}
    
    # Email validation
    email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    
    # Phone number validation (simple pattern)
    phone_pattern = re.compile(r'^\+?[\d\s\(\)-]{7,20}$')
    
    # URL validation
    url_pattern = re.compile(r'^(http|https)://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$')
    
    # Zip/Postal code validation (general pattern)
    postal_pattern = re.compile(r'^\d{4,5}(-\d{4})?$')
    
    # Numeric range validation for common fields
    for col in df.columns:
        col_lower = col.lower()
        data = df[col].dropna()
        
        if len(data) == 0:
            continue
            
        # For text columns, check for common patterns
        if df[col].dtype == 'object':
            # Convert to string for pattern matching
            str_data = data.astype(str)
            
            # Check for email columns
            if 'email' in col_lower:
                valid_count = str_data.str.match(email_pattern).sum()
                validity_pct = 100 * valid_count / len(str_data)
                validity_checks[col] = {
                    "check_type": "email",
                    "valid_count": int(valid_count),
                    "validity_percentage": float(validity_pct),
                    "validity_score": calculate_score(validity_pct, [90, 95, 99])
                }
            
            # Check for phone columns
            elif any(x in col_lower for x in ['phone', 'mobile', 'tel']):
                valid_count = str_data.str.match(phone_pattern).sum()
                validity_pct = 100 * valid_count / len(str_data)
                validity_checks[col] = {
                    "check_type": "phone",
                    "valid_count": int(valid_count),
                    "validity_percentage": float(validity_pct),
                    "validity_score": calculate_score(validity_pct, [90, 95, 99])
                }
            
            # Check for URL columns
            elif any(x in col_lower for x in ['url', 'website', 'link']):
                valid_count = str_data.str.match(url_pattern).sum()
                validity_pct = 100 * valid_count / len(str_data)
                validity_checks[col] = {
                    "check_type": "url",
                    "valid_count": int(valid_count),
                    "validity_percentage": float(validity_pct),
                    "validity_score": calculate_score(validity_pct, [90, 95, 99])
                }
            
            # Check for postal/zip code columns
            elif any(x in col_lower for x in ['zip', 'postal', 'postcode']):
                valid_count = str_data.str.match(postal_pattern).sum()
                validity_pct = 100 * valid_count / len(str_data)
                validity_checks[col] = {
                    "check_type": "postal",
                    "valid_count": int(valid_count),
                    "validity_percentage": float(validity_pct),
                    "validity_score": calculate_score(validity_pct, [90, 95, 99])
                }
        
        # For numeric columns, check for sensible ranges
        elif column_metadata[col]["is_numeric"]:
            # Age columns should be positive and reasonable
            if any(x in col_lower for x in ['age', 'years']):
                valid_range = (data >= 0) & (data <= 120)
                valid_count = valid_range.sum()
                validity_pct = 100 * valid_count / len(data)
                validity_checks[col] = {
                    "check_type": "numeric_range",
                    "valid_count": int(valid_count),
                    "validity_percentage": float(validity_pct),
                    "validity_score": calculate_score(validity_pct, [95, 98, 99.5])
                }
            
            # Percentage columns should be between 0 and 100
            elif any(x in col_lower for x in ['percent', 'pct', 'rate', 'ratio']):
                valid_range = (data >= 0) & (data <= 100)
                valid_count = valid_range.sum()
                validity_pct = 100 * valid_count / len(data)
                validity_checks[col] = {
                    "check_type": "percentage",
                    "valid_count": int(valid_count),
                    "validity_percentage": float(validity_pct),
                    "validity_score": calculate_score(validity_pct, [95, 98, 99.5])
                }
    
    results["validity_checks"] = validity_checks
    
    # Calculate overall validity score
    validity_scores = [data["validity_score"] for data in validity_checks.values()]
    
    results["overall"] = {
        "checks_performed": len(validity_checks),
        "overall_validity_score": sum(validity_scores) / len(validity_scores) if validity_scores else 100
    }
    
    return results

def generate_visualizations(df, assessment_results):
    """Generate visualizations for data quality assessment"""
    visualizations = {}
    
    # 1. Completeness visualization - Missing values heatmap
    if "completeness" in assessment_results:
        completeness_data = assessment_results["completeness"]["columns"]
        missing_pcts = {col: data["missing_count"] / assessment_results["basic_stats"]["row_count"] * 100 
                       for col, data in completeness_data.items()}
        
        # Only include columns with missing values
        missing_cols = {col: pct for col, pct in missing_pcts.items() if pct > 0}
        
        if missing_cols:
            # Create missing values heatmap data
            sorted_cols = sorted(missing_cols.items(), key=lambda x: x[1], reverse=True)
            cols = [col for col, _ in sorted_cols]
            values = [val for _, val in sorted_cols]
            
            fig = px.bar(
                x=values, 
                y=cols, 
                orientation='h',
                title='Missing Values by Column (%)',
                labels={'x': 'Missing Values (%)', 'y': 'Column'},
                color=values,
                color_continuous_scale='Reds',
            )
            fig.update_layout(height=max(300, len(cols) * 30))
            
            # Convert plot to JSON
            visualizations["missing_values_plot"] = fig.to_json()
    
    # 2. Data type distribution
    column_types = {}
    for col, meta in assessment_results.get("columns", {}).items():
        dtype = meta["dtype"]
        if "float" in dtype or "int" in dtype:
            column_types[col] = "Numeric"
        elif "datetime" in dtype:
            column_types[col] = "DateTime"
        elif "bool" in dtype:
            column_types[col] = "Boolean"
        elif "object" in dtype or "string" in dtype:
            column_types[col] = "Text"
        elif "category" in dtype:
            column_types[col] = "Categorical"
        else:
            column_types[col] = "Other"
    
    type_counts = {"Numeric": 0, "DateTime": 0, "Boolean": 0, "Text": 0, "Categorical": 0, "Other": 0}
    for col_type in column_types.values():
        type_counts[col_type] += 1
    
    fig = px.pie(
        names=list(type_counts.keys()),
        values=list(type_counts.values()),
        title='Data Type Distribution',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    visualizations["datatype_distribution"] = fig.to_json()
    
    # 3. Quality scores by dimension
    quality_dimensions = {}
    
    if "completeness" in assessment_results:
        quality_dimensions["Completeness"] = assessment_results["completeness"]["overall"]["completeness_score"]
    
    if "consistency" in assessment_results:
        quality_dimensions["Consistency"] = assessment_results["consistency"]["overall"]["overall_consistency_score"]
    
    if "accuracy" in assessment_results:
        quality_dimensions["Accuracy"] = assessment_results["accuracy"]["overall"]["overall_accuracy_score"]
    
    if "uniqueness" in assessment_results:
        quality_dimensions["Uniqueness"] = assessment_results["uniqueness"]["overall"]["uniqueness_score"]
    
    if "timeliness" in assessment_results and assessment_results["timeliness"]["overall"]["timeliness_score"] is not None:
        quality_dimensions["Timeliness"] = assessment_results["timeliness"]["overall"]["timeliness_score"]
    
    if "validity" in assessment_results:
        quality_dimensions["Validity"] = assessment_results["validity"]["overall"]["overall_validity_score"]
    
    if quality_dimensions:
        dimensions = list(quality_dimensions.keys())
        scores = list(quality_dimensions.values())
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=dimensions,
            fill='toself',
            name='Data Quality Score'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title='Data Quality Dimensions',
            showlegend=False
        )
        
        visualizations["quality_dimensions_radar"] = fig.to_json()
    
    # 4. Top issues bar chart
    issues = []
    
    # Collect issues from completeness
    if "completeness" in assessment_results:
        incomplete_cols = {col: data["completeness_percentage"] 
                          for col, data in assessment_results["completeness"]["columns"].items() 
                          if data["completeness_percentage"] < 95}
        for col, score in incomplete_cols.items():
            issues.append({
                "column": col,
                "issue": "Incomplete Data",
                "score": score,
                "dimension": "Completeness"
            })
    
    # Collect issues from consistency
    if "consistency" in assessment_results and "type_consistency" in assessment_results["consistency"]:
        for col, data in assessment_results["consistency"]["type_consistency"].items():
            if data.get("mixed_types", False):
                issues.append({
                    "column": col,
                    "issue": "Mixed Data Types",
                    "score": data.get("consistency_score", 50),
                    "dimension": "Consistency"
                })
    
    # Collect issues from accuracy
    if "accuracy" in assessment_results and "numeric_accuracy" in assessment_results["accuracy"]:
        for col, data in assessment_results["accuracy"]["numeric_accuracy"].items():
            if data.get("outlier_percentage", 0) > 5:
                issues.append({
                    "column": col,
                    "issue": "High Outlier %",
                    "score": data.get("accuracy_score", 50),
                    "dimension": "Accuracy"
                })
    
    # Get top issues
    if issues:
        issues.sort(key=lambda x: x["score"])
        top_issues = issues[:8]  # Get top 8 issues
        
        fig = px.bar(
            top_issues,
            x="score",
            y="column",
            color="dimension",
            labels={"score": "Quality Score", "column": "Column", "dimension": "Dimension"},
            title="Top Data Quality Issues",
            orientation='h',
            text="issue"
        )
        fig.update_layout(height=max(300, len(top_issues) * 40))
        fig.update_traces(textposition='inside')
        
        visualizations["top_issues_chart"] = fig.to_json()
    
    # 5. Distribution of values (for numeric columns)
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if numeric_cols:
        # Select a sample of numeric columns (up to 4)
        sample_cols = numeric_cols[:min(4, len(numeric_cols))]
        
        fig = make_subplots(rows=len(sample_cols), cols=1, 
                           subplot_titles=[f"Distribution of {col}" for col in sample_cols],
                           vertical_spacing=0.12)
        
        for i, col in enumerate(sample_cols):
            data = df[col].dropna()
            if len(data) > 0:
                fig.add_trace(
                    go.Histogram(x=data, 
                               name=col, 
                               marker_color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]),
                    row=i+1, col=1
                )
        
        fig.update_layout(
            title_text="Value Distributions for Numeric Columns",
            height=250 * len(sample_cols)
        )
        
        visualizations["numeric_distributions"] = fig.to_json()
    
    return visualizations

def analyze_attribute_combination(df, columns):
    """
    Analyze relationships and interdependencies between multiple attributes
    
    Args:
        df: DataFrame with the selected columns
        columns: List of column names to analyze
    
    Returns:
        Dictionary with analysis results
    """
    results = {}
    
    # Basic statistics
    results["total_rows"] = len(df)
    results["complete_rows"] = df.dropna().shape[0]
    results["complete_rows_percentage"] = 100 * results["complete_rows"] / results["total_rows"] if results["total_rows"] > 0 else 100
    
    # Completeness dependency - check if attributes are missing together
    missing_patterns = {}
    for col in columns:
        missing_patterns[col] = df[col].isna()
    
    # Find co-occurrence of missing values
    co_missing = {}
    for i, col1 in enumerate(columns):
        for col2 in columns[i+1:]:
            both_missing = (missing_patterns[col1] & missing_patterns[col2]).sum()
            col1_missing = missing_patterns[col1].sum()
            col2_missing = missing_patterns[col2].sum()
            
            # Calculate conditional probabilities
            p_col2_missing_given_col1_missing = 100 * both_missing / col1_missing if col1_missing > 0 else 0
            p_col1_missing_given_col2_missing = 100 * both_missing / col2_missing if col2_missing > 0 else 0
            
            co_missing[f"{col1}_{col2}"] = {
                "both_missing": int(both_missing),
                "col1_missing": int(col1_missing),
                "col2_missing": int(col2_missing),
                "p_col2_missing_given_col1_missing": float(p_col2_missing_given_col1_missing),
                "p_col1_missing_given_col2_missing": float(p_col1_missing_given_col2_missing)
            }
    
    results["co_missing_analysis"] = co_missing
    
    # Functional dependency analysis - check if one attribute determines the other
    functional_dependencies = {}
    for i, col1 in enumerate(columns):
        for col2 in columns[i+1:]:
            # Group by col1 and check how many unique values of col2 exist for each col1 value
            if df[col1].nunique() > 1:  # Only analyze if col1 has multiple values
                # Count unique col2 values per col1 value
                unique_counts = df.groupby(col1)[col2].nunique()
                # Calculate functional dependency score:
                # If each col1 value corresponds to exactly one col2 value, fd_score = 100
                fd_score_1_to_2 = 100 * (unique_counts == 1).sum() / len(unique_counts) if len(unique_counts) > 0 else 0
                
                functional_dependencies[f"{col1}->{col2}"] = {
                    "fd_score": float(fd_score_1_to_2),
                    "fd_strength": "Strong" if fd_score_1_to_2 >= 95 else 
                                  "Moderate" if fd_score_1_to_2 >= 70 else 
                                  "Weak",
                    "is_key": fd_score_1_to_2 == 100
                }
            
            # Reverse direction
            if df[col2].nunique() > 1:  # Only analyze if col2 has multiple values
                unique_counts = df.groupby(col2)[col1].nunique()
                fd_score_2_to_1 = 100 * (unique_counts == 1).sum() / len(unique_counts) if len(unique_counts) > 0 else 0
                
                functional_dependencies[f"{col2}->{col1}"] = {
                    "fd_score": float(fd_score_2_to_1),
                    "fd_strength": "Strong" if fd_score_2_to_1 >= 95 else 
                                  "Moderate" if fd_score_2_to_1 >= 70 else 
                                  "Weak",
                    "is_key": fd_score_2_to_1 == 100
                }
    
    results["functional_dependencies"] = functional_dependencies
    
    # Value co-occurrence analysis
    if len(columns) == 2 and len(df) > 0:
        # For pairs of columns, analyze value co-occurrence patterns
        col1, col2 = columns
        
        # Calculate correlation for numeric columns
        if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
            correlation = df[col1].corr(df[col2])
            results["correlation"] = {
                "pearson": float(correlation),
                "strength": "Strong" if abs(correlation) >= 0.7 else 
                           "Moderate" if abs(correlation) >= 0.3 else 
                           "Weak"
            }
        
        # For categorical columns, create contingency table
        if df[col1].nunique() <= 20 and df[col2].nunique() <= 20:  # Limit to manageable number of categories
            contingency = pd.crosstab(df[col1], df[col2], normalize='all')
            results["contingency_table"] = contingency.to_dict()
            
            # Look for strong associations
            strong_associations = []
            for idx in contingency.index:
                for col in contingency.columns:
                    if contingency.loc[idx, col] >= 0.1:  # 10% or more of all data points
                        strong_associations.append({
                            "value1": str(idx),
                            "value2": str(col),
                            "percentage": float(contingency.loc[idx, col] * 100)
                        })
            
            results["strong_value_associations"] = sorted(strong_associations, key=lambda x: x["percentage"], reverse=True)[:5]
    
    # Calculate overall quality score for the combination
    dependency_scores = [item["fd_score"] for item in functional_dependencies.values()]
    
    # Weighted score based on completeness and dependencies
    results["quality_score"] = {
        "completeness": float(results["complete_rows_percentage"]),
        "dependency_strength": float(sum(dependency_scores) / len(dependency_scores)) if dependency_scores else 0,
        "overall": float((results["complete_rows_percentage"] * 0.7 + 
                   (sum(dependency_scores) / len(dependency_scores) if dependency_scores else 0) * 0.3))
    }
    
    # Generate recommendations
    results["recommendations"] = generate_combination_recommendations(results, columns)
    
    return results

def generate_combination_recommendations(results, columns):
    """Generate recommendations for attribute combinations"""
    recommendations = []
    
    # Check for completeness issues
    if results["complete_rows_percentage"] < 95:
        recommendations.append({
            "category": "Completeness",
            "issue": f"Only {results['complete_rows_percentage']:.1f}% of rows have complete data for all attributes in this combination.",
            "recommendation": "Consider implementing data validation rules to ensure all related attributes are populated together."
        })
    
    # Check for strong missing value patterns
    for pattern, data in results["co_missing_analysis"].items():
        if data["p_col2_missing_given_col1_missing"] > 80 or data["p_col1_missing_given_col2_missing"] > 80:
            col1, col2 = pattern.split('_')
            recommendations.append({
                "category": "Co-missing Pattern",
                "issue": f"Strong co-missing pattern detected between {col1} and {col2}.",
                "recommendation": "Investigate business process to understand why these fields are frequently missing together."
            })
    
    # Check for functional dependencies
    strong_dependencies = []
    for dep, data in results["functional_dependencies"].items():
        if data["fd_strength"] == "Strong":
            strong_dependencies.append(dep)
            
    if strong_dependencies:
        recommendations.append({
            "category": "Functional Dependency",
            "issue": f"Strong functional dependencies found: {', '.join(strong_dependencies)}",
            "recommendation": "Consider using these dependencies for data validation rules or to simplify data model."
        })
    
    # Check for correlations if they exist
    if "correlation" in results and abs(results["correlation"]["pearson"]) > 0.9:
        recommendations.append({
            "category": "High Correlation",
            "issue": f"Very high correlation ({results['correlation']['pearson']:.2f}) between attributes.",
            "recommendation": "Consider if both attributes are necessary or if one can be derived from the other."
        })
    
    # Generate recommendations based on strong value associations
    if "strong_value_associations" in results and results["strong_value_associations"]:
        associations = results["strong_value_associations"][0]  # Get strongest association
        recommendations.append({
            "category": "Value Association",
            "issue": f"Strong association between values: {associations['value1']} and {associations['value2']} ({associations['percentage']:.1f}% of data)",
            "recommendation": "Use this pattern to validate data or identify potential business rules."
        })
    
    return recommendations

def calculate_score(value, thresholds):
    """Calculate a score based on thresholds [good, great, excellent]"""
    if value >= thresholds[2]:
        return 100  # Excellent
    elif value >= thresholds[1]:
        return 90   # Great
    elif value >= thresholds[0]:
        return 75   # Good
    elif value >= thresholds[0] * 0.8:
        return 60   # Fair
    elif value >= thresholds[0] * 0.6:
        return 40   # Poor
    else:
        return 20   # Critical

def create_summary(results):
    """Create a summary of results across all sheets"""
    summary = {
        "overall_scores": {},
        "top_issues": [],
        "recommendations": []
    }
    
    # Calculate overall scores
    dimension_scores = {
        "completeness": [],
        "consistency": [],
        "accuracy": [],
        "uniqueness": [],
        "timeliness": [],
        "validity": []
    }
    
    # Collect scores from each sheet
    for sheet_name, sheet_results in results.items():
        if sheet_name == "summary":
            continue
            
        for dimension in dimension_scores.keys():
            if dimension in sheet_results and "overall" in sheet_results[dimension]:
                if dimension == "completeness":
                    score = sheet_results[dimension]["overall"]["completeness_score"]
                elif dimension == "consistency":
                    score = sheet_results[dimension]["overall"]["overall_consistency_score"]
                elif dimension == "accuracy":
                    score = sheet_results[dimension]["overall"]["overall_accuracy_score"]
                elif dimension == "uniqueness":
                    score = sheet_results[dimension]["overall"]["uniqueness_score"]
                elif dimension == "timeliness" and sheet_results[dimension]["overall"]["timeliness_score"] is not None:
                    score = sheet_results[dimension]["overall"]["timeliness_score"]
                elif dimension == "validity":
                    score = sheet_results[dimension]["overall"]["overall_validity_score"]
                else:
                    continue
                
                dimension_scores[dimension].append(score)
    
    # Calculate average scores
    for dimension, scores in dimension_scores.items():
        if scores:
            summary["overall_scores"][dimension] = sum(scores) / len(scores)
    
    # Calculate overall data quality score
    all_scores = [score for scores in dimension_scores.values() for score in scores]
    if all_scores:
        summary["overall_scores"]["overall"] = sum(all_scores) / len(all_scores)
    
    # Identify top issues across all sheets
    issues = []
    
    for sheet_name, sheet_results in results.items():
        if sheet_name == "summary":
            continue
            
        # Check completeness issues
        if "completeness" in sheet_results:
            for col, data in sheet_results["completeness"]["columns"].items():
                if data["completeness_percentage"] < 95:
                    issues.append({
                        "sheet": sheet_name,
                        "column": col,
                        "issue": f"Missing data ({100-data['completeness_percentage']:.1f}%)",
                        "dimension": "Completeness",
                        "severity": (100 - data["completeness_percentage"]) / 10
                    })
        
        # Check consistency issues
        if "consistency" in sheet_results and "type_consistency" in sheet_results["consistency"]:
            for col, data in sheet_results["consistency"]["type_consistency"].items():
                if data.get("mixed_types", False):
                    issues.append({
                        "sheet": sheet_name,
                        "column": col,
                        "issue": "Mixed data types",
                        "dimension": "Consistency",
                        "severity": (100 - data.get("consistency_score", 0)) / 10
                    })
        
        # Check accuracy issues
        if "accuracy" in sheet_results and "numeric_accuracy" in sheet_results["accuracy"]:
            for col, data in sheet_results["accuracy"]["numeric_accuracy"].items():
                if data.get("outlier_percentage", 0) > 5:
                    issues.append({
                        "sheet": sheet_name,
                        "column": col,
                        "issue": f"High outliers ({data['outlier_percentage']:.1f}%)",
                        "dimension": "Accuracy",
                        "severity": data.get("outlier_percentage", 0) / 10
                    })
    
    # Sort issues by severity
    issues.sort(key=lambda x: x["severity"], reverse=True)
    summary["top_issues"] = issues[:10]  # Get top 10 issues
    
    # Generate recommendations
    recommendations = []
    
    # Recommendation based on completeness
    if "completeness" in summary["overall_scores"]:
        completeness_score = summary["overall_scores"]["completeness"]
        if completeness_score < 90:
            recommendations.append({
                "dimension": "Completeness",
                "recommendation": "Focus on improving data collection processes to reduce missing values.",
                "priority": "High" if completeness_score < 75 else "Medium"
            })
    
    # Recommendation based on consistency
    if "consistency" in summary["overall_scores"]:
        consistency_score = summary["overall_scores"]["consistency"]
        if consistency_score < 90:
            recommendations.append({
                "dimension": "Consistency",
                "recommendation": "Standardize data formats and implement validation rules to ensure consistency.",
                "priority": "High" if consistency_score < 75 else "Medium"
            })
    
    # Recommendation based on accuracy
    if "accuracy" in summary["overall_scores"]:
        accuracy_score = summary["overall_scores"]["accuracy"]
        if accuracy_score < 90:
            recommendations.append({
                "dimension": "Accuracy",
                "recommendation": "Implement data validation rules and review outlier detection processes.",
                "priority": "High" if accuracy_score < 75 else "Medium"
            })
    
    # Recommendation based on uniqueness
    if "uniqueness" in summary["overall_scores"]:
        uniqueness_score = summary["overall_scores"]["uniqueness"]
        if uniqueness_score < 95:
            recommendations.append({
                "dimension": "Uniqueness",
                "recommendation": "Establish proper key constraints and deduplication procedures.",
                "priority": "High" if uniqueness_score < 85 else "Medium"
            })
    
    # Recommendation based on timeliness
    if "timeliness" in summary["overall_scores"]:
        timeliness_score = summary["overall_scores"]["timeliness"]
        if timeliness_score < 85:
            recommendations.append({
                "dimension": "Timeliness",
                "recommendation": "Update data collection frequency and implement data freshness monitoring.",
                "priority": "Medium" if timeliness_score < 70 else "Low"
            })
    
    # Recommendation based on validity
    if "validity" in summary["overall_scores"]:
        validity_score = summary["overall_scores"]["validity"]
        if validity_score < 90:
            recommendations.append({
                "dimension": "Validity",
                "recommendation": "Implement stronger data validation rules and format checking.",
                "priority": "High" if validity_score < 75 else "Medium"
            })
    
    summary["recommendations"] = recommendations
    
    return summary

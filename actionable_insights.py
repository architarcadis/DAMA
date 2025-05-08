"""
Actionable Insights Generator Module

This module generates specific, actionable recommendations and insights based on
data quality assessment results. It prioritizes recommendations by:
1. Business impact
2. Implementation effort
3. Expected quality improvement

The insights are tailored to different stakeholder roles (data teams, business users, executives)
and include clear implementation steps.
"""

import pandas as pd
import numpy as np
from collections import defaultdict

# Define the impact categories and thresholds
SEVERITY_THRESHOLDS = {
    "critical": 90,    # Requires immediate attention
    "high": 70,        # Should be addressed soon
    "medium": 50,      # Address as part of regular work
    "low": 30          # Can be addressed in future iterations
}

# Define effort categories (in person-days)
EFFORT_CATEGORIES = {
    "trivial": 1,      # Less than 1 day
    "small": 5,        # 1-5 days
    "moderate": 10,    # 5-10 days
    "large": 20        # More than 10 days
}

def generate_actionable_insights(assessment_results, priority_dimension=None):
    """
    Generate prioritized, actionable insights from assessment results
    
    Args:
        assessment_results: Dictionary containing quality assessment results
        priority_dimension: Optional dimension to prioritize (completeness, consistency, etc.)
    
    Returns:
        Dictionary containing actionable insights categorized by role and priority
    """
    if not assessment_results:
        return {
            "error": "No assessment results available to generate insights."
        }
        
    # Extract key information from assessment results
    summary = assessment_results.get("summary", {})
    sheets = {k: v for k, v in assessment_results.items() if k != "summary"}
    
    # Initialize insights structure
    insights = {
        "top_recommendations": [],
        "by_role": {
            "data_team": [],
            "business_users": [],
            "executive": []
        },
        "by_dimension": defaultdict(list),
        "by_priority": {
            "critical": [],
            "high": [],
            "medium": [],
            "low": []
        },
        "expected_improvements": {}
    }
    
    # 1. Analyze quality dimensions for each sheet
    sheet_insights = []
    
    for sheet_name, sheet_data in sheets.items():
        # Process each dimension
        for dimension, dimension_data in sheet_data.items():
            if dimension in ["completeness", "consistency", "accuracy", "uniqueness", "timeliness", "validity"]:
                dimension_insights = _analyze_dimension(
                    dimension, 
                    dimension_data, 
                    sheet_name
                )
                sheet_insights.extend(dimension_insights)
    
    # 2. Prioritize insights
    prioritized_insights = _prioritize_insights(sheet_insights, priority_dimension)
    
    # 3. Generate top recommendations (top 5)
    insights["top_recommendations"] = prioritized_insights[:5]
    
    # 4. Categorize by role
    insights["by_role"]["data_team"] = [i for i in prioritized_insights 
                                       if i["roles"]["data_team"] > 0.5][:10]
    
    insights["by_role"]["business_users"] = [i for i in prioritized_insights 
                                           if i["roles"]["business_users"] > 0.5][:7]
    
    insights["by_role"]["executive"] = [i for i in prioritized_insights 
                                       if i["roles"]["executive"] > 0.5][:5]
    
    # 5. Categorize by dimension
    for insight in prioritized_insights:
        insights["by_dimension"][insight["dimension"]].append(insight)
        
    # 6. Categorize by priority
    for insight in prioritized_insights:
        insights["by_priority"][insight["priority"]].append(insight)
    
    # 7. Calculate expected improvements
    insights["expected_improvements"] = _calculate_improvements(
        prioritized_insights, 
        summary.get("overall_scores", {})
    )
    
    return insights

def _analyze_dimension(dimension, dimension_data, sheet_name):
    """
    Analyze a specific quality dimension and generate insight recommendations
    
    Args:
        dimension: Name of the dimension (completeness, consistency, etc.)
        dimension_data: Data for this dimension from assessment results
        sheet_name: Name of the sheet/dataset
    
    Returns:
        List of insight dictionaries for this dimension
    """
    insights = []
    
    # Dimension-specific analysis logic
    if dimension == "completeness":
        insights.extend(_analyze_completeness(dimension_data, sheet_name))
    elif dimension == "consistency":
        insights.extend(_analyze_consistency(dimension_data, sheet_name))
    elif dimension == "accuracy":
        insights.extend(_analyze_accuracy(dimension_data, sheet_name))
    elif dimension == "uniqueness":
        insights.extend(_analyze_uniqueness(dimension_data, sheet_name))
    elif dimension == "timeliness":
        insights.extend(_analyze_timeliness(dimension_data, sheet_name))
    elif dimension == "validity":
        insights.extend(_analyze_validity(dimension_data, sheet_name))
    
    return insights

def _analyze_completeness(dimension_data, sheet_name):
    """Generate insights for completeness dimension"""
    insights = []
    
    # Check for missing values in columns
    if "missing_values" in dimension_data:
        missing_cols = dimension_data.get("missing_values", {})
        
        # Get columns with highest missing percentage
        high_missing_cols = []
        for col, data in missing_cols.items():
            missing_pct = data.get("missing_percentage", 0)
            
            if missing_pct > 20:
                high_missing_cols.append((col, missing_pct))
        
        # Sort by missing percentage (descending)
        high_missing_cols.sort(key=lambda x: x[1], reverse=True)
        
        # Generate insights for high missing columns
        for col, missing_pct in high_missing_cols[:3]:  # Focus on top 3
            # Determine severity based on missing percentage
            if missing_pct > 50:
                priority = "critical"
                business_impact = 0.9
            elif missing_pct > 30:
                priority = "high"
                business_impact = 0.7
            elif missing_pct > 15:
                priority = "medium"
                business_impact = 0.5
            else:
                priority = "low"
                business_impact = 0.3
            
            # Generate insight
            insight = {
                "title": f"Fill missing values in {col}",
                "description": f"{col} has {missing_pct:.1f}% missing values, which may affect analysis accuracy.",
                "actions": [
                    f"Review data collection process for {col}",
                    f"Implement validation at data entry for {col}",
                    "Consider data imputation techniques appropriate for this variable"
                ],
                "implementation": {
                    "code_example": f"# Example imputation with median\ndf['{col}'] = df['{col}'].fillna(df['{col}'].median())",
                    "effort": "small" if missing_pct < 30 else "moderate",
                    "tools": ["Data validation forms", "Statistical imputation", "Data collection review"]
                },
                "business_impact": business_impact,
                "priority": priority,
                "dimension": "completeness",
                "sheet_name": sheet_name,
                "improved_metrics": {
                    "completeness_score": f"+{min(missing_pct / 2, 20):.1f}%",
                    "overall_quality": f"+{min(missing_pct / 5, 10):.1f}%"
                },
                "roles": {
                    "data_team": 0.9,
                    "business_users": 0.7, 
                    "executive": 0.3 if missing_pct > 50 else 0.1
                }
            }
            insights.append(insight)
    
    return insights

def _analyze_consistency(dimension_data, sheet_name):
    """Generate insights for consistency dimension"""
    insights = []
    
    # Check for type inconsistencies
    if "type_consistency" in dimension_data:
        inconsistent_cols = []
        for col, data in dimension_data.get("type_consistency", {}).items():
            consistency_pct = data.get("consistency_percentage", 100)
            if consistency_pct < 90:
                inconsistent_cols.append((col, consistency_pct))
        
        # Sort by consistency percentage (ascending)
        inconsistent_cols.sort(key=lambda x: x[1])
        
        # Generate insights for inconsistent columns
        for col, consistency_pct in inconsistent_cols[:3]:  # Focus on top 3
            inconsistency_pct = 100 - consistency_pct
            
            # Determine severity
            if inconsistency_pct > 30:
                priority = "critical"
                business_impact = 0.9
            elif inconsistency_pct > 20:
                priority = "high"
                business_impact = 0.7
            elif inconsistency_pct > 10:
                priority = "medium"
                business_impact = 0.5
            else:
                priority = "low"
                business_impact = 0.3
            
            # Generate insight
            insight = {
                "title": f"Standardize data types in {col}",
                "description": f"{col} has {inconsistency_pct:.1f}% type inconsistency, which may cause processing errors.",
                "actions": [
                    f"Implement type validation for {col}",
                    f"Convert existing values to consistent type",
                    "Update data model to enforce type constraints"
                ],
                "implementation": {
                    "code_example": f"# Example type conversion\ndf['{col}'] = df['{col}'].astype('appropriate_type')",
                    "effort": "small",
                    "tools": ["Data validation", "Type conversion", "Schema enforcement"]
                },
                "business_impact": business_impact,
                "priority": priority,
                "dimension": "consistency",
                "sheet_name": sheet_name,
                "improved_metrics": {
                    "consistency_score": f"+{min(inconsistency_pct / 2, 15):.1f}%",
                    "overall_quality": f"+{min(inconsistency_pct / 6, 5):.1f}%"
                },
                "roles": {
                    "data_team": 0.9,
                    "business_users": 0.5,
                    "executive": 0.1
                }
            }
            insights.append(insight)
    
    # Check for value range consistency
    if "value_ranges" in dimension_data:
        inconsistent_ranges = []
        for col, data in dimension_data.get("value_ranges", {}).items():
            if "inconsistency_score" in data and data["inconsistency_score"] > 0.3:
                inconsistent_ranges.append((col, data["inconsistency_score"]))
        
        # Sort by inconsistency score (descending)
        inconsistent_ranges.sort(key=lambda x: x[1], reverse=True)
        
        # Generate insights for inconsistent ranges
        for col, inconsistency_score in inconsistent_ranges[:2]:  # Focus on top 2
            # Generate insight
            insight = {
                "title": f"Standardize value ranges in {col}",
                "description": f"{col} has inconsistent value ranges, which may indicate data quality issues.",
                "actions": [
                    f"Establish valid value ranges for {col}",
                    "Implement range validation at data entry",
                    "Review outliers and correct if needed"
                ],
                "implementation": {
                    "code_example": f"# Example range validation\nmask = (df['{col}'] < min_val) | (df['{col}'] > max_val)\noutliers = df[mask]",
                    "effort": "moderate",
                    "tools": ["Range validation", "Outlier detection", "Business rule engine"]
                },
                "business_impact": 0.6,
                "priority": "medium",
                "dimension": "consistency",
                "sheet_name": sheet_name,
                "improved_metrics": {
                    "consistency_score": f"+{min(inconsistency_score * 30, 15):.1f}%",
                    "accuracy_score": f"+{min(inconsistency_score * 20, 10):.1f}%",
                    "overall_quality": f"+{min(inconsistency_score * 15, 8):.1f}%"
                },
                "roles": {
                    "data_team": 0.8,
                    "business_users": 0.6,
                    "executive": 0.2
                }
            }
            insights.append(insight)
    
    return insights

def _analyze_accuracy(dimension_data, sheet_name):
    """Generate insights for accuracy dimension"""
    insights = []
    
    # Check for outliers
    if "outliers" in dimension_data:
        outlier_cols = []
        for col, data in dimension_data.get("outliers", {}).items():
            outlier_pct = data.get("outlier_percentage", 0)
            if outlier_pct > 5:
                outlier_cols.append((col, outlier_pct))
        
        # Sort by outlier percentage (descending)
        outlier_cols.sort(key=lambda x: x[1], reverse=True)
        
        # Generate insights for columns with outliers
        for col, outlier_pct in outlier_cols[:3]:  # Focus on top 3
            # Determine severity
            if outlier_pct > 20:
                priority = "critical"
                business_impact = 0.9
            elif outlier_pct > 10:
                priority = "high"
                business_impact = 0.7
            else:
                priority = "medium"
                business_impact = 0.5
            
            # Generate insight
            insight = {
                "title": f"Review outliers in {col}",
                "description": f"{col} has {outlier_pct:.1f}% outliers that may be errors or require special handling.",
                "actions": [
                    f"Investigate outliers in {col} to determine if they're valid",
                    "Implement domain-specific validation rules",
                    "Consider using statistical approaches for extreme value handling"
                ],
                "implementation": {
                    "code_example": f"# Example outlier detection\nfrom scipy import stats\nz_scores = stats.zscore(df['{col}'])\noutliers = df[abs(z_scores) > 3]",
                    "effort": "moderate",
                    "tools": ["Statistical analysis", "Domain validation", "Outlier visualization"]
                },
                "business_impact": business_impact,
                "priority": priority,
                "dimension": "accuracy",
                "sheet_name": sheet_name,
                "improved_metrics": {
                    "accuracy_score": f"+{min(outlier_pct, 15):.1f}%",
                    "overall_quality": f"+{min(outlier_pct / 3, 5):.1f}%"
                },
                "roles": {
                    "data_team": 0.8,
                    "business_users": 0.7,
                    "executive": 0.3 if outlier_pct > 15 else 0.1
                }
            }
            insights.append(insight)
    
    return insights

def _analyze_uniqueness(dimension_data, sheet_name):
    """Generate insights for uniqueness dimension"""
    insights = []
    
    # Check for duplicates
    if "duplicates" in dimension_data:
        dup_pct = dimension_data.get("duplicates", {}).get("duplicate_percentage", 0)
        
        if dup_pct > 1:  # More than 1% duplicates
            # Determine severity
            if dup_pct > 10:
                priority = "critical"
                business_impact = 0.9
            elif dup_pct > 5:
                priority = "high"
                business_impact = 0.8
            else:
                priority = "medium"
                business_impact = 0.6
            
            # Generate insight
            insight = {
                "title": "Remove duplicate records",
                "description": f"The dataset contains {dup_pct:.1f}% duplicate records, which may skew analysis results.",
                "actions": [
                    "Implement deduplication process",
                    "Review data collection/import process to prevent duplicates",
                    "Define and enforce unique constraints"
                ],
                "implementation": {
                    "code_example": "# Example deduplication\ndf = df.drop_duplicates()",
                    "effort": "small",
                    "tools": ["Data deduplication", "Unique constraints", "ETL process review"]
                },
                "business_impact": business_impact,
                "priority": priority,
                "dimension": "uniqueness",
                "sheet_name": sheet_name,
                "improved_metrics": {
                    "uniqueness_score": f"+{min(dup_pct * 2, 20):.1f}%",
                    "overall_quality": f"+{min(dup_pct, 10):.1f}%"
                },
                "roles": {
                    "data_team": 0.9,
                    "business_users": 0.5,
                    "executive": 0.2 if dup_pct > 5 else 0.1
                }
            }
            insights.append(insight)
    
    # Check for duplicate keys
    if "duplicate_keys" in dimension_data:
        key_issues = dimension_data.get("duplicate_keys", {})
        
        for key_col, data in key_issues.items():
            dup_key_pct = data.get("duplicate_percentage", 0)
            
            if dup_key_pct > 0:  # Any duplicate keys
                # This is usually a critical issue for key columns
                priority = "critical"
                business_impact = 0.9
                
                # Generate insight
                insight = {
                    "title": f"Fix duplicate values in key column {key_col}",
                    "description": f"Key column {key_col} has {dup_key_pct:.1f}% duplicate values, violating uniqueness constraints.",
                    "actions": [
                        f"Review and correct duplicate keys in {key_col}",
                        "Implement unique constraint on this column",
                        "Review data entry/integration processes"
                    ],
                    "implementation": {
                        "code_example": f"# Example finding duplicates\nduplicates = df[df.duplicated(['{key_col}'], keep=False)]",
                        "effort": "moderate",
                        "tools": ["Unique constraints", "Data integrity rules", "ETL validation"]
                    },
                    "business_impact": business_impact,
                    "priority": priority,
                    "dimension": "uniqueness",
                    "sheet_name": sheet_name,
                    "improved_metrics": {
                        "uniqueness_score": f"+{min(dup_key_pct * 5, 25):.1f}%",
                        "consistency_score": f"+{min(dup_key_pct * 2, 10):.1f}%",
                        "overall_quality": f"+{min(dup_key_pct * 3, 15):.1f}%"
                    },
                    "roles": {
                        "data_team": 0.9,
                        "business_users": 0.7,
                        "executive": 0.5
                    }
                }
                insights.append(insight)
    
    return insights

def _analyze_timeliness(dimension_data, sheet_name):
    """Generate insights for timeliness dimension"""
    insights = []
    
    # Check for outdated records
    if "outdated_records" in dimension_data:
        outdated_pct = dimension_data.get("outdated_records", {}).get("percentage", 0)
        
        if outdated_pct > 5:  # More than 5% outdated
            # Determine severity
            if outdated_pct > 30:
                priority = "critical"
                business_impact = 0.9
            elif outdated_pct > 15:
                priority = "high"
                business_impact = 0.7
            else:
                priority = "medium"
                business_impact = 0.5
            
            # Generate insight
            insight = {
                "title": "Update outdated records",
                "description": f"{outdated_pct:.1f}% of records are outdated, which may lead to incorrect business decisions.",
                "actions": [
                    "Implement a data refresh process for outdated records",
                    "Set up automated data currency monitoring",
                    "Establish data update frequency standards"
                ],
                "implementation": {
                    "code_example": "# Example identifying outdated records\noutdated = df[df['last_update'] < (datetime.now() - timedelta(days=threshold_days))]",
                    "effort": "moderate",
                    "tools": ["Data refresh automation", "Currency monitoring", "Update validation"]
                },
                "business_impact": business_impact,
                "priority": priority,
                "dimension": "timeliness",
                "sheet_name": sheet_name,
                "improved_metrics": {
                    "timeliness_score": f"+{min(outdated_pct, 20):.1f}%",
                    "overall_quality": f"+{min(outdated_pct / 3, 8):.1f}%"
                },
                "roles": {
                    "data_team": 0.7,
                    "business_users": 0.8,
                    "executive": 0.4 if outdated_pct > 20 else 0.2
                }
            }
            insights.append(insight)
    
    # Check for missing time values
    if "missing_time_values" in dimension_data:
        date_cols_issues = dimension_data.get("missing_time_values", {})
        
        for col, data in date_cols_issues.items():
            missing_pct = data.get("missing_percentage", 0)
            
            if missing_pct > 10:
                # Generate insight
                insight = {
                    "title": f"Fill missing date/time values in {col}",
                    "description": f"{col} has {missing_pct:.1f}% missing values, which affects temporal analysis.",
                    "actions": [
                        f"Review data collection process for {col}",
                        "Implement date/time validation at entry",
                        "Consider appropriate date imputation strategies"
                    ],
                    "implementation": {
                        "code_example": f"# Example date imputation\ndf['{col}'] = df['{col}'].fillna(method='ffill')  # Forward fill",
                        "effort": "small",
                        "tools": ["Date validation", "Temporal imputation", "Data collection review"]
                    },
                    "business_impact": 0.6,
                    "priority": "high" if missing_pct > 25 else "medium",
                    "dimension": "timeliness",
                    "sheet_name": sheet_name,
                    "improved_metrics": {
                        "timeliness_score": f"+{min(missing_pct / 2, 15):.1f}%",
                        "completeness_score": f"+{min(missing_pct / 4, 7):.1f}%",
                        "overall_quality": f"+{min(missing_pct / 5, 5):.1f}%"
                    },
                    "roles": {
                        "data_team": 0.8,
                        "business_users": 0.6,
                        "executive": 0.2
                    }
                }
                insights.append(insight)
    
    return insights

def _analyze_validity(dimension_data, sheet_name):
    """Generate insights for validity dimension"""
    insights = []
    
    # Check for format violations
    if "format_violations" in dimension_data:
        format_issues = dimension_data.get("format_violations", {})
        
        for col, data in format_issues.items():
            violation_pct = data.get("violation_percentage", 0)
            
            if violation_pct > 5:
                # Determine severity
                if violation_pct > 25:
                    priority = "critical"
                    business_impact = 0.8
                elif violation_pct > 15:
                    priority = "high"
                    business_impact = 0.7
                else:
                    priority = "medium"
                    business_impact = 0.5
                
                # Generate insight
                insight = {
                    "title": f"Fix format violations in {col}",
                    "description": f"{col} has {violation_pct:.1f}% format violations that need correction.",
                    "actions": [
                        f"Implement format validation for {col}",
                        "Standardize data entry process",
                        "Correct existing violations"
                    ],
                    "implementation": {
                        "code_example": f"# Example format validation \nimport re\npattern = r'<appropriate_regex>'\nvalid_mask = df['{col}'].str.match(pattern, na=False)",
                        "effort": "moderate",
                        "tools": ["Format validation", "Regex validation", "Data standardization"]
                    },
                    "business_impact": business_impact,
                    "priority": priority,
                    "dimension": "validity",
                    "sheet_name": sheet_name,
                    "improved_metrics": {
                        "validity_score": f"+{min(violation_pct, 20):.1f}%",
                        "overall_quality": f"+{min(violation_pct / 4, 6):.1f}%"
                    },
                    "roles": {
                        "data_team": 0.9,
                        "business_users": 0.5,
                        "executive": 0.2
                    }
                }
                insights.append(insight)
    
    # Check for business rule violations
    if "business_rules" in dimension_data:
        rule_issues = dimension_data.get("business_rules", {})
        
        for rule, data in rule_issues.items():
            violation_pct = data.get("violation_percentage", 0)
            
            if violation_pct > 0:  # Any business rule violation is important
                # Determine severity
                if violation_pct > 10:
                    priority = "critical"
                    business_impact = 0.9
                elif violation_pct > 5:
                    priority = "high"
                    business_impact = 0.8
                else:
                    priority = "medium"
                    business_impact = 0.6
                
                # Generate insight
                insight = {
                    "title": f"Fix business rule violations: {rule}",
                    "description": f"{violation_pct:.1f}% of records violate business rule '{rule}', affecting data reliability.",
                    "actions": [
                        "Review and correct rule violations",
                        "Implement rule validation in data processes",
                        "Update documentation and training materials"
                    ],
                    "implementation": {
                        "code_example": "# Example business rule validation\n# Rule-specific logic would be implemented here",
                        "effort": "moderate" if violation_pct < 10 else "large",
                        "tools": ["Business rule engine", "Data validation", "Process review"]
                    },
                    "business_impact": business_impact,
                    "priority": priority,
                    "dimension": "validity",
                    "sheet_name": sheet_name,
                    "improved_metrics": {
                        "validity_score": f"+{min(violation_pct * 2, 25):.1f}%",
                        "consistency_score": f"+{min(violation_pct, 10):.1f}%",
                        "overall_quality": f"+{min(violation_pct * 1.5, 15):.1f}%"
                    },
                    "roles": {
                        "data_team": 0.8,
                        "business_users": 0.8,
                        "executive": 0.5 if violation_pct > 5 else 0.3
                    }
                }
                insights.append(insight)
    
    return insights

def _prioritize_insights(insights, priority_dimension=None):
    """
    Prioritize insights based on business impact, effort, and dimension priority
    
    Args:
        insights: List of insight dictionaries
        priority_dimension: Optional dimension to prioritize
    
    Returns:
        Sorted list of insights
    """
    # Define weights for prioritization
    weights = {
        "business_impact": 0.6,
        "effort": 0.3,
        "dimension_priority": 0.1
    }
    
    # If a dimension is prioritized, adjust weights
    if priority_dimension:
        weights["dimension_priority"] = 0.3
        weights["effort"] = 0.2
        weights["business_impact"] = 0.5
    
    # Map effort categories to numeric values (lower is better)
    effort_scores = {
        "trivial": 1.0,
        "small": 0.8,
        "moderate": 0.5,
        "large": 0.2
    }
    
    # Set dimension priorities (can be customized)
    dimension_priorities = {
        "completeness": 0.9,
        "accuracy": 0.85,
        "consistency": 0.8,
        "validity": 0.75,
        "uniqueness": 0.7,
        "timeliness": 0.65
    }
    
    # If a dimension is prioritized, boost its score
    if priority_dimension and priority_dimension in dimension_priorities:
        dimension_priorities[priority_dimension] = 1.0
    
    # Calculate priority score for each insight
    for insight in insights:
        effort_score = effort_scores.get(insight["implementation"]["effort"], 0.5)
        dimension_score = dimension_priorities.get(insight["dimension"], 0.5)
        
        # Calculate weighted score
        priority_score = (
            weights["business_impact"] * insight["business_impact"] +
            weights["effort"] * effort_score +
            weights["dimension_priority"] * dimension_score
        )
        
        insight["priority_score"] = priority_score
    
    # Sort by priority score (descending)
    return sorted(insights, key=lambda x: x["priority_score"], reverse=True)

def _calculate_improvements(prioritized_insights, current_scores):
    """
    Calculate expected quality improvements if insights are implemented
    
    Args:
        prioritized_insights: List of prioritized insights
        current_scores: Current quality scores
    
    Returns:
        Dictionary with expected improvement metrics
    """
    # Initialize improvement metrics
    improvements = {
        "overall": 0,
        "by_dimension": defaultdict(float),
        "top_gains": [],
        "implementation_phases": {
            "quick_wins": [],
            "medium_term": [],
            "long_term": []
        }
    }
    
    # Track dimensions with improvements
    dimension_improvements = defaultdict(float)
    
    # Calculate improvements from top insights
    for insight in prioritized_insights:
        dimension = insight["dimension"]
        
        # Extract improvement percentages and convert to numeric
        for metric, value_str in insight["improved_metrics"].items():
            if metric == "overall_quality":
                # Extract numeric value from string like "+5.0%"
                value = float(value_str.strip("+% "))
                improvements["overall"] += value * 0.1  # Scale down to avoid over-optimistic estimates
                
            elif "_score" in metric:
                # Extract dimension name and numeric value
                dim = metric.split("_score")[0]
                value = float(value_str.strip("+% "))
                dimension_improvements[dim] += value * 0.2  # Scale down
        
        # Categorize by implementation phase
        effort = insight["implementation"]["effort"]
        if effort in ["trivial", "small"]:
            improvements["implementation_phases"]["quick_wins"].append({
                "title": insight["title"],
                "gain": f"+{min(float(insight['improved_metrics'].get('overall_quality', '+0%').strip('+%')), 10):.1f}%"
            })
        elif effort == "moderate":
            improvements["implementation_phases"]["medium_term"].append({
                "title": insight["title"],
                "gain": f"+{min(float(insight['improved_metrics'].get('overall_quality', '+0%').strip('+%')), 10):.1f}%"
            })
        else:
            improvements["implementation_phases"]["long_term"].append({
                "title": insight["title"],
                "gain": f"+{min(float(insight['improved_metrics'].get('overall_quality', '+0%').strip('+%')), 10):.1f}%"
            })
    
    # Set dimension improvements
    improvements["by_dimension"] = dict(dimension_improvements)
    
    # Calculate projected scores based on current scores and improvements
    improvements["projected_scores"] = {}
    
    for dimension, current in current_scores.items():
        if dimension in dimension_improvements:
            # Cap improvement to realistic level
            max_improvement = min(100 - current, dimension_improvements[dimension])
            projected = current + max_improvement
            improvements["projected_scores"][dimension] = min(100, projected)
        else:
            improvements["projected_scores"][dimension] = current
    
    # Overall projected score
    if "overall" in current_scores:
        current_overall = current_scores["overall"]
        max_overall_improvement = min(100 - current_overall, improvements["overall"])
        improvements["projected_scores"]["overall"] = min(100, current_overall + max_overall_improvement)
    
    # Find top 3 gains
    dimension_gains = []
    for dim, improvement in dimension_improvements.items():
        dimension_gains.append((dim, improvement))
    
    dimension_gains.sort(key=lambda x: x[1], reverse=True)
    for dim, gain in dimension_gains[:3]:
        improvements["top_gains"].append({
            "dimension": dim,
            "gain": f"+{gain:.1f}%"
        })
    
    return improvements

def format_insight(insight, include_implementation=True):
    """
    Format an insight for display
    
    Args:
        insight: Insight dictionary
        include_implementation: Whether to include implementation details
    
    Returns:
        Formatted string representation of the insight
    """
    # Format main description
    lines = [
        f"## {insight['title']}",
        f"**Priority:** {insight['priority'].title()}",
        f"**Description:** {insight['description']}",
        "### Actions:",
    ]
    
    # Add actions
    for i, action in enumerate(insight['actions']):
        lines.append(f"{i+1}. {action}")
    
    # Add implementation details if requested
    if include_implementation:
        lines.append("\n### Implementation Details:")
        lines.append(f"**Effort:** {insight['implementation']['effort'].title()}")
        lines.append(f"**Tools:** {', '.join(insight['implementation']['tools'])}")
        
        if "code_example" in insight["implementation"]:
            lines.append("\n**Example Implementation:**")
            lines.append(f"```python\n{insight['implementation']['code_example']}\n```")
    
    # Add expected improvements
    lines.append("\n### Expected Improvements:")
    for metric, value in insight['improved_metrics'].items():
        # Format metric name for display
        metric_display = metric.replace("_", " ").title()
        lines.append(f"- {metric_display}: {value}")
    
    return "\n".join(lines)

def get_executive_summary(insights):
    """
    Generate an executive summary from the insights
    
    Args:
        insights: Dictionary of insights from generate_actionable_insights
    
    Returns:
        String with executive summary
    """
    summary_lines = [
        "# Executive Summary of Data Quality Insights",
        "\nThis analysis has identified key actionable insights to improve data quality."
    ]
    
    # Add top recommendations
    summary_lines.append("\n## Top Priority Recommendations")
    
    for i, insight in enumerate(insights.get("top_recommendations", [])[:3]):
        summary_lines.append(f"\n### {i+1}. {insight['title']}")
        summary_lines.append(f"**Priority:** {insight['priority'].title()}")
        summary_lines.append(f"**Impact:** {insight['description']}")
        summary_lines.append("**Key Actions:**")
        for action in insight['actions'][:2]:  # Just show first 2 actions
            summary_lines.append(f"- {action}")
    
    # Add expected improvements
    if "expected_improvements" in insights:
        improvements = insights["expected_improvements"]
        
        summary_lines.append("\n## Expected Quality Improvements")
        
        if "overall" in improvements:
            summary_lines.append(f"\nOverall quality could improve by approximately +{improvements['overall']:.1f}% if recommendations are implemented.")
        
        if "top_gains" in improvements:
            summary_lines.append("\nTop areas of improvement:")
            for gain in improvements["top_gains"]:
                summary_lines.append(f"- {gain['dimension'].title()}: {gain['gain']}")
        
        if "implementation_phases" in improvements:
            phases = improvements["implementation_phases"]
            
            summary_lines.append("\n## Implementation Approach")
            
            if phases.get("quick_wins"):
                summary_lines.append("\n### Quick Wins (Immediate Impact)")
                for win in phases["quick_wins"][:3]:  # Show top 3
                    summary_lines.append(f"- {win['title']} ({win['gain']} improvement)")
            
            if phases.get("medium_term"):
                summary_lines.append("\n### Medium-Term Initiatives")
                for initiative in phases["medium_term"][:2]:  # Show top 2
                    summary_lines.append(f"- {initiative['title']}")
    
    return "\n".join(summary_lines)
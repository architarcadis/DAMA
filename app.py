import streamlit as st

# Set page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="Data Quality Assessment Tool",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import os
import time
import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.colored_header import colored_header
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import json
import base64
from io import BytesIO
from functools import lru_cache

from data_quality import perform_data_quality_assessment
from report_generator import generate_pdf_report
from actionable_insights import generate_actionable_insights, format_insight, get_executive_summary
from animations import loading_animation, processing_animation_with_stages, loading_card
from utils import load_data, get_data_info
from ai_integration import (
    explain_data_quality_issues,
    generate_recommendations,
    classify_issue_severity,
    generate_assessment_documentation,
    answer_data_quality_query
)
from animations import (
    loading_animation, 
    processing_animation_with_stages,
    loading_card
)

# Constants - Arcadis Colors
ARCADIS_ORANGE = "#ee7203"  # (238, 114, 3)
ARCADIS_DARK_ORANGE = "#963a00"
ARCADIS_LIGHT_ORANGE = "#ffb980"
ARCADIS_BLUE = "#1e3c70"  # Arcadis blue
ARCADIS_LIGHT_BLUE = "#4f5e8d"
ARCADIS_GREEN = "#228c22"
ARCADIS_RED = "#c82536"

# Apply custom CSS
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    div[data-testid="stSidebarNav"] {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    div.stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    div.stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 4px 4px 0px 0px;
        padding: 0.5rem 1rem;
        gap: 0.5rem;
    }
    div.stTabs [aria-selected="true"] {
        background-color: #ee7203;
        color: white;
    }
    .main .block-container {
        padding: 2rem;
    }
    h1, h2, h3 {
        color: #1e3c70;
    }
    .stButton > button {
        background-color: #ee7203;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
    }
    .stButton > button:hover {
        background-color: #c85f00;
    }
    div[data-testid="stExpander"] details summary p {
        font-weight: bold;
        color: #1e3c70;
    }
    div[data-testid="stFileUploader"] {
        border: 1px dashed #ee7203;
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
    div[data-testid="stFileUploader"] > label {
        color: #1e3c70;
        font-weight: bold;
    }
    div[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15);
    }
    footer {display: none !important;}
    .reportview-container .main footer {visibility: hidden;}
    .reportview-container .main .block-container {padding-bottom: 0;}
</style>
""", unsafe_allow_html=True)

# Function to initialize session state variables
def initialize_session_state():
    """Initialize all session state variables with default values"""
    
    # Data-related variables
    if "data_dict" not in st.session_state:
        st.session_state.data_dict = None  # Dictionary of dataframes with sheet names as keys
    
    if "file_type" not in st.session_state:
        st.session_state.file_type = None  # "Excel" or "CSV"
    
    if "sheet_names" not in st.session_state:
        st.session_state.sheet_names = []  # List of sheet names
    
    if "selected_sheet" not in st.session_state:
        st.session_state.selected_sheet = None  # Currently selected sheet name
    
    if "file_info" not in st.session_state:
        st.session_state.file_info = None  # Dictionary with file information
    
    if "assessment_results" not in st.session_state:
        st.session_state.assessment_results = None  # Results of the quality assessment
    
    if "current_view" not in st.session_state:
        st.session_state.current_view = "Data Upload"  # Current view in the navigation
    
    if "historical_assessments" not in st.session_state:
        st.session_state.historical_assessments = []  # List of historical assessments
    
    if "report_template" not in st.session_state:
        st.session_state.report_template = "standard"  # Default report template
        
    if "ml_results" not in st.session_state:
        st.session_state.ml_results = {}  # Results from machine learning algorithms
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # History of chat interactions
    
    # AI integration settings
    if "ai_enabled" not in st.session_state:
        st.session_state.ai_enabled = False  # Is AI functionality enabled
    
    if "ai_provider" not in st.session_state:
        st.session_state.ai_provider = "openai"  # Current AI provider
    
    # OpenAI settings
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    
    if "openai_model" not in st.session_state:
        st.session_state.openai_model = "gpt-4o"
    
    # Anthropic settings
    if "anthropic_api_key" not in st.session_state:
        st.session_state.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    
    if "anthropic_model" not in st.session_state:
        st.session_state.anthropic_model = "claude-3-5-sonnet-20241022"
    
    # xAI settings
    if "xai_api_key" not in st.session_state:
        st.session_state.xai_api_key = os.environ.get("XAI_API_KEY", "")
    
    if "xai_model" not in st.session_state:
        st.session_state.xai_model = "grok-2-1212"
    
    # Local LLM settings
    if "local_model_path" not in st.session_state:
        st.session_state.local_model_path = ""
    
    # Token and cache settings
    if "token_usage" not in st.session_state:
        st.session_state.token_usage = {"total": 0, "history": []}
    
    if "ai_explanation_cache" not in st.session_state:
        st.session_state.ai_explanation_cache = {}  # Cache AI explanations for performance
    
    # AI feature enablement settings
    if "nl_explanations_enabled" not in st.session_state:
        st.session_state.nl_explanations_enabled = True
    
    if "auto_docs_enabled" not in st.session_state:
        st.session_state.auto_docs_enabled = True
    
    if "issue_classification_enabled" not in st.session_state:
        st.session_state.issue_classification_enabled = True
    
    if "chat_interface_enabled" not in st.session_state:
        st.session_state.chat_interface_enabled = True
    
    # Application settings
    if "max_rows" not in st.session_state:
        st.session_state.max_rows = 100000
    
    if "cache_ttl" not in st.session_state:
        st.session_state.cache_ttl = 30  # minutes
    
    if "parallel_processing" not in st.session_state:
        st.session_state.parallel_processing = True
    
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False

# Initialize all session state variables
initialize_session_state()

# Function to load and save historical assessments
def save_assessment_history(assessment_results, file_info):
    """Save the current assessment to history with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create a simplified version to save in history (avoiding large objects)
    summary = None
    if "summary" in assessment_results:
        summary = {
            "overall_scores": assessment_results["summary"]["overall_scores"],
            "top_issues": assessment_results["summary"]["top_issues"][:3] if assessment_results["summary"]["top_issues"] else []
        }
    
    historical_entry = {
        "timestamp": timestamp,
        "file_name": file_info.get("file_name", "Unknown"),
        "file_type": file_info.get("file_type", "Unknown"),
        "sheet_count": len(file_info.get("sheet_names", [])),
        "total_rows": file_info.get("total_rows", 0),
        "total_columns": file_info.get("total_columns", 0),
        "summary": summary
    }
    
    st.session_state.historical_assessments.append(historical_entry)
    
    # Keep only the last 10 assessments
    if len(st.session_state.historical_assessments) > 10:
        st.session_state.historical_assessments = st.session_state.historical_assessments[-10:]

def detect_anomalies(df):
    """Use Isolation Forest to detect anomalies in numeric columns"""
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return None, None
    
    # Fill NaN values with column mean
    df_numeric = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Apply Isolation Forest
    model = IsolationForest(contamination=0.05, random_state=42)
    try:
        preds = model.fit_predict(df_numeric)
        anomaly_score = model.decision_function(df_numeric)
        return preds, anomaly_score
    except Exception as e:
        st.warning(f"Could not compute anomalies: {str(e)}")
        return None, None

def perform_clustering(df, n_clusters=3):
    """Use KMeans to identify clusters in the data"""
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return None
    
    # Fill NaN values with column mean
    df_numeric = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Apply KMeans
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(df_numeric)
        return clusters
    except Exception as e:
        st.warning(f"Could not perform clustering: {str(e)}")
        return None

# Function to display sidebar with just AI settings and data info
def display_sidebar():
    with st.sidebar:
        # Custom styling for sidebar header to match design
        st.markdown(
            f"""
            <div style="display: flex; align-items: center;">
                <img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIwIiBoZWlnaHQ9IjQwIiB2aWV3Qm94PSIwIDAgMjQwIDgwIiBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgogIDxwYXRoIGQ9Ik0wLDAgTDI0MCwwIEwyNDAsODAgTDAsODAgTDAsMCBaIiBmaWxsPSJub25lIi8+CiAgPHRleHQgeD0iMTIiIHk9IjQ1IiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMzAiIGZvbnQtd2VpZ2h0PSJib2xkIiBmaWxsPSIjMDQzQjg3Ij5EQU1BIFFVQUxJVFk8L3RleHQ+CiAgPHJlY3QgeD0iMTIiIHk9IjU1IiB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIiIGZpbGw9IiNlZTcyMDMiLz4KICA8dGV4dCB4PSIxMiIgeT0iNzAiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxMiIgZmlsbD0iIzY2NjY2NiI+RGF0YSBRdWFsaXR5IEFzc2Vzc21lbnQgVG9vbDwvdGV4dD4KPC9zdmc+" alt="DAMA Quality" style="width: 100%; max-width: 240px; margin-bottom: 10px;">
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # AI status in the sidebar
        st.markdown("""
        <div style="
            background-color: #f8f9fa;
            border-radius: 5px 5px 0 0;
            padding: 10px 15px;
            font-weight: 500;
            font-size: 16px;
            border-bottom: 1px solid #eee;
            margin-top: 20px;
        ">
            AI Features Status
        </div>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div style="background-color: #f8f9fa; border-radius: 0 0 5px 5px; padding: 10px 15px;">', unsafe_allow_html=True)
            
            ai_status = "Enabled" if st.session_state.get("ai_enabled", False) else "Disabled"
            st.info(f"AI Features: **{ai_status}**")
            
            if st.session_state.get("ai_enabled", False):
                # Show current provider
                provider_name = ""
                if st.session_state.get("ai_provider", "openai") == "openai":
                    provider_name = "OpenAI API"
                    model = st.session_state.get("openai_model", "gpt-4o")
                elif st.session_state.get("ai_provider", "openai") == "anthropic":
                    provider_name = "Anthropic API"
                    model = st.session_state.get("anthropic_model", "claude-3-5-sonnet-20241022")
                elif st.session_state.get("ai_provider", "openai") == "xai":
                    provider_name = "xAI API"
                    model = st.session_state.get("xai_model", "grok-2-1212")
                else:
                    provider_name = "Local LLM"
                    model = "Custom model"
                
                st.success(f"Provider: **{provider_name}**  \nModel: **{model}**")
                
                # Quick link to settings
                if st.button("Configure AI Settings", use_container_width=True):
                    st.session_state.current_view = "Settings"
                    st.rerun()
            else:
                if st.button("Enable AI Features", use_container_width=True):
                    st.session_state.ai_enabled = True
                    st.session_state.current_view = "Settings"
                    st.rerun()
            
            st.caption("AI features enable natural language explanations, recommendations, and conversational interactions.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Show data info if data is loaded
        if st.session_state.data_dict is not None:
            st.markdown("""
            <div style="
                background-color: #f8f9fa;
                border-radius: 5px 5px 0 0;
                padding: 10px 15px;
                font-weight: 500;
                font-size: 16px;
                border-bottom: 1px solid #eee;
                margin-top: 20px;
            ">
                Current Dataset
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div style="background-color: #f8f9fa; border-radius: 0 0 5px 5px; padding: 10px 15px;">', unsafe_allow_html=True)
            st.info(f"""
            **File**: {st.session_state.file_info.get('file_name', 'Unknown')}  
            **Type**: {st.session_state.file_type}  
            **Sheets**: {len(st.session_state.sheet_names)}  
            **Total Rows**: {st.session_state.file_info.get('total_rows', 0):,}  
            **Total Columns**: {st.session_state.file_info.get('total_columns', 0)}
            """)
            st.markdown('</div>', unsafe_allow_html=True)

# Main navigation and layout
def display_main_tabs():
    # Application header with DAMA Quality styling
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <div>
                <h2 style="color: {ARCADIS_BLUE}; margin-bottom: 0;"><strong>DAMA Quality</strong></h2>
                <p style="color: #666666; margin-top: 0; font-size: 0.9rem;">
                    Data Quality Assessment Tool
                </p>
            </div>
            <div style="margin-left: auto;">
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none">
                    <rect x="2" y="2" width="6" height="6" rx="1" fill="{ARCADIS_ORANGE}" />
                    <rect x="10" y="2" width="6" height="6" rx="1" fill="{ARCADIS_ORANGE}" opacity="0.8" />
                    <rect x="18" y="2" width="4" height="6" rx="1" fill="{ARCADIS_ORANGE}" opacity="0.6" />
                    <rect x="2" y="10" width="6" height="6" rx="1" fill="{ARCADIS_ORANGE}" opacity="0.7" />
                    <rect x="10" y="10" width="6" height="6" rx="1" fill="{ARCADIS_ORANGE}" opacity="0.9" />
                    <rect x="18" y="10" width="4" height="6" rx="1" fill="{ARCADIS_ORANGE}" opacity="0.5" />
                    <rect x="2" y="18" width="20" height="4" rx="1" fill="{ARCADIS_ORANGE}" opacity="0.6" />
                </svg>
            </div>
        </div>
        <div style="height: 3px; background-color: {ARCADIS_ORANGE}; margin-bottom: 1rem;"></div>
        """,
        unsafe_allow_html=True
    )
    
    # Custom navigation styling to match the design
    st.markdown(
        f"""
        <style>
        /* Custom styling for navigation */
        div.stHorizontalBlock {{
            background-color: #f8f9fa;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            padding: 0 !important;
        }}
        
        /* Style for the navigation block title */
        div.navigation-header {{
            padding: 10px 15px;
            border-bottom: 1px solid #eee;
            font-size: 16px;
            font-weight: 600;
            color: #333;
        }}
        
        /* Style for the navigation links container */
        div.nav-links-container {{
            display: flex;
            flex-direction: column;
            padding: 0;
        }}
        
        /* Style for navigation links */
        div.nav-link {{
            padding: 12px 15px;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            color: #555;
            text-decoration: none;
            font-size: 14px;
            margin: 0;
            border-left: 3px solid transparent;
        }}
        
        /* Highlighted active link */
        div.nav-link-active {{
            background-color: {ARCADIS_ORANGE};
            color: white;
            font-weight: 500;
            border-left: 3px solid {ARCADIS_ORANGE};
        }}
        
        /* Icon styling for navigation links */
        div.nav-link i {{
            margin-right: 8px;
            font-size: 16px;
            color: {ARCADIS_ORANGE};
        }}
        
        /* Icon color for active link */
        div.nav-link-active i {{
            color: white;
        }}
        </style>
        """, 
        unsafe_allow_html=True
    )
    
    # Navigation options with icons - consolidated to 5 main tabs
    options = [
        "Welcome",  # Welcome and how to use
        "Data Manager",  # Combined Data Upload + Interactive Dashboard
        "Quality Assessment",  # Quality Assessment + Advanced Analytics
        "Insights & Reports",  # Actionable Insights + Report Generation + Historical Trends
        "Settings"
    ]
    
    icons = [
        "house-fill",
        "database-fill-gear",
        "check-circle-fill", 
        "lightbulb-fill",
        "gear-fill"
    ]
    
    # Get current view from session state
    current_view = st.session_state.get("current_view", "Data Upload")
    
    # Create horizontal option menu with custom styling
    selected = option_menu(
        menu_title=None,
        options=options,
        icons=icons,
        default_index=options.index(current_view) if current_view in options else 0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0px", "background-color": "#ffffff", "border-radius": "0px", "box-shadow": "none"},
            "icon": {"color": ARCADIS_ORANGE, "font-size": "16px"}, 
            "nav-link": {
                "font-size": "14px", 
                "text-align": "center", 
                "margin":"2px", 
                "padding": "10px 15px",
                "border-radius": "0px",
                "border-bottom": f"3px solid transparent"
            },
            "nav-link-selected": {
                "background-color": "#ffffff", 
                "color": ARCADIS_ORANGE, 
                "font-weight": "bold",
                "border-radius": "0px",
                "border-bottom": f"3px solid {ARCADIS_ORANGE}"
            },
        }
    )
    
    st.session_state.current_view = selected

# Display sidebar and main navigation tabs
display_sidebar()
display_main_tabs()

# Render the selected view
if st.session_state.current_view == "Welcome":
    st.header("Welcome to the Data Quality Assessment Platform")
    
    st.markdown("""
    This comprehensive platform helps you assess, analyze, and improve data quality.
    It follows DAMA principles for data quality assessment and provides actionable insights
    to enhance your data's value for business decision-making.
    """)
    
    # Create tabs for different welcome sections
    welcome_tab1, welcome_tab2, welcome_tab3, welcome_tab4 = st.tabs([
        "Getting Started", 
        "Understanding Results", 
        "Setting Thresholds",
        "Best Practices"
    ])
    
    # Getting Started Tab
    with welcome_tab1:
        st.markdown("### 📊 Getting Started")
        
        st.markdown("""
        Follow these simple steps to perform your first data quality assessment:
        
        #### 1. Upload Your Data
        - Navigate to the **Data Manager** tab
        - Upload your Excel or CSV file using the file uploader
        - The tool will automatically process your file and provide a quick overview
        
        #### 2. Run Quality Assessment
        - Go to the **Quality Assessment** tab
        - Configure assessment options as needed (or use the defaults)
        - Click "Run Quality Assessment" to begin the analysis
        
        #### 3. Explore Results
        - Review the quality scores and visualizations
        - Click on dimensions to see detailed results for specific quality aspects
        - Use the interactive dashboard to explore your data further
        
        #### 4. Generate Reports & Insights
        - Navigate to the **Insights & Reports** tab
        - Generate actionable insights based on the assessment
        - Create PDF reports for sharing with stakeholders
        
        #### 5. Implement Improvements
        - Follow the recommendations in the Actionable Insights section
        - Re-run assessments after implementing changes to measure improvements
        """)
        
        st.info("**Tip:** Start with a smaller subset of your data for faster results during initial exploration.")
    
    # Understanding Results Tab
    with welcome_tab2:
        st.markdown("### 🔍 Understanding Quality Dimensions")
        
        st.markdown("""
        The assessment follows DAMA principles and evaluates six key dimensions of data quality:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("Completeness", expanded=True):
                st.markdown("""
                **What it measures:** The presence of required data
                
                **Key metrics:**
                - Percentage of missing values
                - Fields with high null rates
                - Required vs. optional field completion
                
                **Good score:** > 95% complete
                """)
            
            with st.expander("Consistency"):
                st.markdown("""
                **What it measures:** Data conformity and coherence
                
                **Key metrics:**
                - Format consistency
                - Value range consistency
                - Cross-field validation
                
                **Good score:** > 90% consistent
                """)
            
            with st.expander("Accuracy"):
                st.markdown("""
                **What it measures:** Correctness of data values
                
                **Key metrics:**
                - Outlier detection
                - Statistical validation
                - Known reference values
                
                **Good score:** > 85% accurate
                """)
        
        with col2:
            with st.expander("Uniqueness"):
                st.markdown("""
                **What it measures:** Absence of duplicates
                
                **Key metrics:**
                - Duplicate records
                - Unique key validation
                - Near-duplicate detection
                
                **Good score:** > 98% unique
                """)
            
            with st.expander("Timeliness"):
                st.markdown("""
                **What it measures:** Data currency and availability
                
                **Key metrics:**
                - Age of data
                - Update frequency
                - Time lag analysis
                
                **Good score:** > 80% timely
                """)
            
            with st.expander("Validity"):
                st.markdown("""
                **What it measures:** Adherence to business rules and formats
                
                **Key metrics:**
                - Format compliance
                - Business rule validation
                - Value domain adherence
                
                **Good score:** > 90% valid
                """)
        
        st.markdown("### 📈 Score Interpretation")
        
        score_col1, score_col2 = st.columns(2)
        
        with score_col1:
            st.markdown("""
            #### Overall Score Ranges
            
            - **90-100**: Excellent - Data meets highest quality standards
            - **80-89**: Good - Minor quality issues, suitable for most analyses
            - **70-79**: Fair - Some quality concerns, may affect certain analyses
            - **50-69**: Poor - Significant quality issues, requires attention
            - **< 50**: Critical - Major quality problems, not suitable for analysis
            """)
        
        with score_col2:
            # Simple gauge chart for score visualization
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = 75,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Example Score"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#ee7203"},
                    'steps': [
                        {'range': [0, 50], 'color': "#DC3545"},
                        {'range': [50, 70], 'color': "#FFC107"},
                        {'range': [70, 80], 'color': "#20c997"},
                        {'range': [80, 100], 'color': "#28A745"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 2},
                        'thickness': 0.75,
                        'value': 75
                    }
                }
            ))
            
            fig.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Setting Thresholds Tab
    with welcome_tab3:
        st.markdown("### 🎚️ Setting Quality Thresholds")
        
        st.markdown("""
        The application uses thresholds to evaluate data quality scores. You can customize these
        thresholds in the Settings tab to match your specific requirements.
        """)
        
        st.markdown("""
        #### Default Thresholds
        
        | Dimension     | Acceptable | Good | Excellent |
        |--------------|------------|------|-----------|
        | Completeness | 80%        | 90%  | 98%       |
        | Consistency  | 75%        | 85%  | 95%       |
        | Accuracy     | 70%        | 80%  | 90%       |
        | Uniqueness   | 95%        | 98%  | 99.5%     |
        | Timeliness   | 70%        | 80%  | 90%       |
        | Validity     | 75%        | 85%  | 95%       |
        """)
        
        st.markdown("""
        #### Considerations for Setting Thresholds
        
        - **Industry standards:** Consider established benchmarks for your industry
        - **Data criticality:** Set higher thresholds for mission-critical data
        - **Use case:** Analysis type may require different quality levels
        - **Improvement goals:** Set progressive thresholds to track improvement
        """)
        
        st.info("""
        **Tip:** Start with the default thresholds and adjust based on your specific needs.
        You can save multiple threshold profiles for different types of assessments.
        """)
    
    # Best Practices Tab
    with welcome_tab4:
        st.markdown("### ✅ Best Practices")
        
        st.markdown("""
        #### Preparing Your Data
        
        1. **Consolidate your data:** Minimize the number of sheets and files
        2. **Include headers:** Ensure your files have clear column headers
        3. **Consistent formats:** Use consistent date and number formats
        4. **Remove formulas:** Convert Excel formulas to values before uploading
        5. **Label samples:** If using sample data, clearly mark it as such
        
        #### Running Assessments
        
        1. **Start small:** Begin with a subset of data for initial exploration
        2. **Define domains:** Configure value domains for validity testing
        3. **Adjust thresholds:** Customize thresholds based on your needs
        4. **Run regularly:** Schedule assessments at regular intervals
        5. **Track history:** Save assessment results to monitor trends
        
        #### Using Reports & Insights
        
        1. **Prioritize issues:** Focus on high-impact, low-effort improvements first
        2. **Share appropriately:** Use report templates suitable for your audience
        3. **Action planning:** Create clear implementation plans for improvements
        4. **Measure impact:** Re-assess after implementing changes
        5. **Document decisions:** Record quality-related decisions and rationales
        """)
        
        with st.expander("Common Pitfalls to Avoid"):
            st.markdown("""
            - **Ignoring context:** Data quality requirements vary by use case
            - **Excessive precision:** Don't set unrealistically high thresholds
            - **Missing root causes:** Look beyond symptoms to underlying issues
            - **Focusing only on scores:** Understand the business impact of quality issues
            - **Skipping validation:** Always validate automated findings
            """)

elif st.session_state.current_view == "Data Manager":
    st.subheader("Data Management & Exploration")
    
    # Create subtabs for Data Upload and Interactive Dashboard
    data_tab1, data_tab2 = st.tabs(["Data Upload", "Interactive Dashboard"])
    
    with data_tab1:
        st.markdown("""
        This tool performs comprehensive data quality assessment on your data files.
        Upload your file below to begin analysis. The tool supports multiple formats and provides advanced analytics capabilities.
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # File upload with better UI
            uploaded_file = st.file_uploader(
                "Upload Excel or CSV file",
                type=["xlsx", "xls", "csv"],
                help="Upload your data file to begin quality assessment"
            )
        
            if uploaded_file is not None:
                # Load data from uploaded file
                try:
                    # Create container for the animation
                    animation_container = st.empty()
                    
                    # First animation stage - File loading
                    with animation_container.container():
                        processing_animation_with_stages(
                            stage=1, 
                            total_stages=3,
                            stage_name="File Loading",
                            description=f"Loading and parsing {uploaded_file.name}..."
                        )
                    
                    time.sleep(0.5)  # Small pause for visual feedback
                    
                    # Load the data
                    data_dict, file_type, sheet_names = load_data(uploaded_file)
                    
                    # Second animation stage - Data analysis
                    with animation_container.container():
                        processing_animation_with_stages(
                            stage=2, 
                            total_stages=3,
                            stage_name="Data Analysis",
                            description="Analyzing data structure and properties..."
                        )
                    
                    time.sleep(0.5)  # Small pause for visual feedback
                    
                    # Store in session state
                    st.session_state.data_dict = data_dict
                    st.session_state.file_type = file_type
                    st.session_state.sheet_names = sheet_names
                    
                    # Get file information
                    file_info = get_data_info(data_dict, sheet_names)
                    file_info["file_name"] = uploaded_file.name  # Add filename to info
                    st.session_state.file_info = file_info
                    
                    # Third animation stage - Preparation complete
                    with animation_container.container():
                        processing_animation_with_stages(
                            stage=3, 
                            total_stages=3,
                            stage_name="Preparation Complete",
                            description="Data successfully loaded and prepared for analysis!"
                        )
                    
                    time.sleep(0.5)  # Small pause for visual feedback
                    
                    # Clear the animation container
                    animation_container.empty()
                    
                    if sheet_names:
                        st.session_state.selected_sheet = sheet_names[0]
                    
                    st.success(f"Successfully loaded {file_type} file with {len(sheet_names):,} sheet(s) and {file_info['total_rows']:,} total rows.")
                    
                    # Show quick stats
                    col_stats1, col_stats2, col_stats3 = st.columns(3)
                    
                    with col_stats1:
                        st.metric(
                            label="Total Rows",
                            value=f"{file_info['total_rows']:,}",
                            delta=None
                        )
                    
                    with col_stats2:
                        st.metric(
                            label="Total Columns", 
                            value=f"{file_info['total_columns']:,}",
                            delta=None
                        )
                    
                    with col_stats3:
                        st.metric(
                            label="File Size",
                            value=f"{sum(details['memory_usage'] for _, details in file_info['sheets'].items()):.2f} MB",
                            delta=None
                        )
                    
                    # Apply custom styling to metrics
                    style_metric_cards()
                    
                    # Next steps guidance
                    st.info("Your data is now loaded! Use the tabs above to navigate to the Interactive Dashboard or Quality Assessment sections to explore your data.")
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
    
    with col2:
        # Display helpful information
        st.markdown("""
        ### Supported Formats
        
        - **Excel Files** (.xlsx, .xls)
            - Multiple sheets supported
            - Formula results are evaluated
        
        - **CSV Files** (.csv)
            - Auto-detects delimiters
            - Handles different encodings
        
        ### Size Limits
        
        For optimal performance:
        - Up to 1 million rows
        - Up to 1,000 columns
        
        ### What Happens Next?
        
        After upload, you can:
        1. Explore data in the interactive dashboard
        2. Run quality assessment
        3. Generate comprehensive reports
        4. Analyze historical trends
        """)

elif st.session_state.current_view == "Interactive Dashboard":
    st.subheader("Interactive Data Dashboard")
    
    if st.session_state.data_dict is None:
        st.warning("Please upload a data file first.")
    else:
        # Sheet selector
        if len(st.session_state.sheet_names) > 1:
            st.session_state.selected_sheet = st.selectbox(
                "Select a sheet to analyze",
                st.session_state.sheet_names,
                index=st.session_state.sheet_names.index(st.session_state.selected_sheet) if st.session_state.selected_sheet in st.session_state.sheet_names else 0
            )
        
        df = st.session_state.data_dict[st.session_state.selected_sheet]
        
        # Dashboard tabs for different views
        dash_tab1, dash_tab2, dash_tab3 = st.tabs(["Data Overview", "Column Analysis", "Distribution Analysis"])
        
        with dash_tab1:
            # Overview metrics
            st.markdown("### Dataset Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            # Calculate metrics
            total_rows = len(df)
            total_cols = len(df.columns)
            missing_data = df.isna().sum().sum()
            missing_pct = missing_data / (total_rows * total_cols) * 100
            duplicated_rows = df.duplicated().sum()
            duplicated_pct = duplicated_rows / total_rows * 100 if total_rows > 0 else 0
            
            with col1:
                st.metric(
                    label="Total Rows",
                    value=f"{total_rows:,}"
                )
            
            with col2:
                st.metric(
                    label="Total Columns",
                    value=f"{total_cols:,}"
                )
            
            with col3:
                st.metric(
                    label="Missing Values",
                    value=f"{missing_data:,}",
                    delta=f"{missing_pct:.1f}%",
                    delta_color="inverse"
                )
            
            with col4:
                st.metric(
                    label="Duplicate Rows",
                    value=f"{duplicated_rows:,}",
                    delta=f"{duplicated_pct:.1f}%",
                    delta_color="inverse"
                )
            
            style_metric_cards()
            
            # Data types distribution
            st.markdown("### Data Type Distribution")
            
            # Count data types
            dtype_counts = df.dtypes.value_counts().reset_index()
            dtype_counts.columns = ['Data Type', 'Count']
            
            # Clean up type names
            dtype_counts['Data Type'] = dtype_counts['Data Type'].astype(str)
            dtype_counts['Data Type'] = dtype_counts['Data Type'].str.replace('float64', 'Numeric (Float)')
            dtype_counts['Data Type'] = dtype_counts['Data Type'].str.replace('int64', 'Numeric (Integer)')
            dtype_counts['Data Type'] = dtype_counts['Data Type'].str.replace('object', 'Text')
            dtype_counts['Data Type'] = dtype_counts['Data Type'].str.replace('datetime64\\[ns\\]', 'Date/Time')
            dtype_counts['Data Type'] = dtype_counts['Data Type'].str.replace('bool', 'Boolean')
            
            # Sort by count
            dtype_counts = dtype_counts.sort_values('Count', ascending=False)
            
            fig_types = px.bar(
                dtype_counts,
                x='Data Type',
                y='Count',
                color='Data Type',
                color_discrete_sequence=px.colors.qualitative.Bold,
                title="Column Data Types"
            )
            
            fig_types.update_layout(
                xaxis_title="Data Type",
                yaxis_title="Number of Columns",
                height=400
            )
            
            st.plotly_chart(fig_types, use_container_width=True)
            
            # Display dataframe sample with improved styling
            with st.expander("Preview Data", expanded=True):
                st.dataframe(
                    df.head(10).style.background_gradient(
                        cmap='YlOrRd',
                        subset=df.select_dtypes(include=['float64', 'int64']).columns
                    ),
                    use_container_width=True
                )
        
        with dash_tab2:
            st.markdown("### Column-Level Analysis")
            
            # Column selector
            selected_column = st.selectbox(
                "Select a column to analyze",
                df.columns.tolist()
            )
            
            if selected_column:
                # Column statistics and visualizations
                col_data = df[selected_column]
                col_type = str(col_data.dtype)
                
                # Display column info
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Summary statistics
                    st.markdown(f"#### Statistics for: {selected_column}")
                    
                    # Different statistics based on data type
                    if col_type.startswith('float') or col_type.startswith('int'):
                        # Numeric data
                        stats = {
                            "Count": col_data.count(),
                            "Missing": col_data.isna().sum(),
                            "Missing %": f"{col_data.isna().mean() * 100:.2f}%",
                            "Mean": f"{col_data.mean():.2f}",
                            "Median": f"{col_data.median():.2f}",
                            "Min": f"{col_data.min():.2f}",
                            "Max": f"{col_data.max():.2f}",
                            "Std Dev": f"{col_data.std():.2f}",
                            "Unique Values": col_data.nunique()
                        }
                    elif col_type.startswith('datetime'):
                        # Date data
                        stats = {
                            "Count": col_data.count(),
                            "Missing": col_data.isna().sum(),
                            "Missing %": f"{col_data.isna().mean() * 100:.2f}%",
                            "Earliest": col_data.min(),
                            "Latest": col_data.max(),
                            "Range (days)": (col_data.max() - col_data.min()).days if not pd.isna(col_data.min()) and not pd.isna(col_data.max()) else "N/A",
                            "Unique Values": col_data.nunique()
                        }
                    else:
                        # Text/categorical data
                        stats = {
                            "Count": col_data.count(),
                            "Missing": col_data.isna().sum(),
                            "Missing %": f"{col_data.isna().mean() * 100:.2f}%",
                            "Unique Values": col_data.nunique(),
                            "Most Common": col_data.value_counts().index[0] if not col_data.value_counts().empty else "N/A",
                            "Most Common Count": col_data.value_counts().iloc[0] if not col_data.value_counts().empty else 0,
                            "Most Common %": f"{col_data.value_counts(normalize=True).iloc[0] * 100:.2f}%" if not col_data.value_counts().empty else "0%"
                        }
                    
                    # Create a dataframe of statistics
                    stats_df = pd.DataFrame(list(stats.items()), columns=['Statistic', 'Value'])
                    st.table(stats_df.set_index('Statistic'))
                
                with col2:
                    # Completeness chart
                    missing_count = col_data.isna().sum()
                    valid_count = col_data.count()
                    
                    fig_completeness = go.Figure(go.Pie(
                        labels=['Valid', 'Missing'],
                        values=[valid_count, missing_count],
                        hole=0.6,
                        marker_colors=[ARCADIS_GREEN, ARCADIS_RED]
                    ))
                    
                    fig_completeness.update_layout(
                        title="Completeness",
                        annotations=[{
                            'text': f"{valid_count/(valid_count+missing_count)*100:.1f}%<br>Complete",
                            'showarrow': False,
                            'font': {'size': 16}
                        }],
                        height=250
                    )
                    
                    st.plotly_chart(fig_completeness, use_container_width=True)
                
                # Column distribution visualization
                if col_type.startswith('float') or col_type.startswith('int'):
                    # Create histogram for numeric data
                    fig_hist = px.histogram(
                        df,
                        x=selected_column,
                        marginal="box",
                        color_discrete_sequence=[ARCADIS_ORANGE],
                        title=f"Distribution of {selected_column}"
                    )
                    
                    fig_hist.update_layout(
                        xaxis_title=selected_column,
                        yaxis_title="Frequency",
                        height=400
                    )
                    
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # Add box plot
                    fig_box = px.box(
                        df,
                        y=selected_column,
                        points="all",
                        color_discrete_sequence=[ARCADIS_BLUE],
                        title=f"Box Plot of {selected_column}"
                    )
                    
                    fig_box.update_layout(
                        yaxis_title=selected_column,
                        height=400
                    )
                    
                    st.plotly_chart(fig_box, use_container_width=True)
                
                elif col_type.startswith('datetime'):
                    # Create timeline for date data
                    if not col_data.dropna().empty:
                        # Count by month/year
                        date_counts = col_data.dropna().dt.to_period('M').value_counts().sort_index()
                        date_counts.index = date_counts.index.astype(str)
                        
                        fig_timeline = px.line(
                            x=date_counts.index,
                            y=date_counts.values,
                            markers=True,
                            title=f"Timeline Distribution of {selected_column}",
                            color_discrete_sequence=[ARCADIS_BLUE]
                        )
                        
                        fig_timeline.update_layout(
                            xaxis_title="Date",
                            yaxis_title="Count",
                            height=400
                        )
                        
                        st.plotly_chart(fig_timeline, use_container_width=True)
                    else:
                        st.warning("No valid date values to visualize.")
                
                else:
                    # Create bar chart for categorical data
                    value_counts = col_data.value_counts().reset_index().head(20)
                    value_counts.columns = ['Value', 'Count']
                    
                    # Calculate percentages
                    total = value_counts['Count'].sum()
                    value_counts['Percentage'] = value_counts['Count'] / total * 100
                    
                    fig_cat = px.bar(
                        value_counts,
                        x='Value',
                        y='Count',
                        color='Value',
                        text=value_counts['Percentage'].apply(lambda x: f'{x:.1f}%'),
                        title=f"Top 20 Values in {selected_column}",
                        color_discrete_sequence=px.colors.qualitative.Bold
                    )
                    
                    fig_cat.update_layout(
                        xaxis_title="Value",
                        yaxis_title="Count",
                        showlegend=False,
                        height=400
                    )
                    
                    st.plotly_chart(fig_cat, use_container_width=True)
        
        with dash_tab3:
            st.markdown("### Distribution Analysis")
            
            # Correlation matrix for numeric columns
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
            if len(numeric_cols) >= 2:
                st.markdown("#### Correlation Matrix")
                
                corr_matrix = df[numeric_cols].corr()
                
                fig_corr = px.imshow(
                    corr_matrix,
                    color_continuous_scale='RdBu_r',
                    zmin=-1,
                    zmax=1,
                    title="Correlation Matrix of Numeric Columns"
                )
                
                fig_corr.update_layout(
                    height=500,
                    coloraxis_colorbar=dict(
                        title=dict(
                            text="Correlation",
                            side="right"
                        )
                    )
                )
                
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Select columns for scatter plot
                st.markdown("#### Relationship Explorer")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    x_axis = st.selectbox("X-Axis", numeric_cols, index=0)
                
                with col2:
                    y_axis = st.selectbox("Y-Axis", numeric_cols, index=min(1, len(numeric_cols)-1))
                
                # Optional color by category
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                color_by = None
                if categorical_cols:
                    color_by = st.selectbox(
                        "Color By (Optional)",
                        ["None"] + categorical_cols
                    )
                    if color_by == "None":
                        color_by = None
                
                # Create scatter plot
                if x_axis != y_axis:
                    fig_scatter = px.scatter(
                        df,
                        x=x_axis,
                        y=y_axis,
                        color=color_by,
                        opacity=0.7,
                        title=f"Relationship between {x_axis} and {y_axis}"
                    )
                    
                    fig_scatter.update_layout(
                        xaxis_title=x_axis,
                        yaxis_title=y_axis,
                        height=500
                    )
                    
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.info("Please select different columns for X and Y axes.")
            else:
                st.warning("Not enough numeric columns for correlation analysis.")
            
            # Missing values pattern
            st.markdown("#### Missing Value Patterns")
            
            # Create a heatmap of missing values
            missing_data = df.isna().astype(int)
            
            # Take a sample if too many rows
            if len(missing_data) > 100:
                missing_sample = missing_data.sample(n=100, random_state=42)
            else:
                missing_sample = missing_data
            
            fig_missing = px.imshow(
                missing_sample.T,
                color_continuous_scale=[[0, 'rgb(0,128,0)'], [1, 'rgb(255,0,0)']],  # Using RGB format for better compatibility
                title="Missing Value Pattern (Sample)",
                labels=dict(x="Row Index", y="Column", color="Missing")
            )
            
            fig_missing.update_layout(
                height=500,
                coloraxis_colorbar=dict(
                    title="Missing",
                    tickvals=[0, 1],
                    ticktext=["Present", "Missing"]
                )
            )
            
            st.plotly_chart(fig_missing, use_container_width=True)
            
            # Missing values by column
            missing_by_col = df.isna().sum().sort_values(ascending=False)
            missing_by_col = missing_by_col[missing_by_col > 0]
            
            if not missing_by_col.empty:
                missing_pct = (missing_by_col / len(df) * 100).round(2)
                missing_df = pd.DataFrame({
                    'Column': missing_by_col.index,
                    'Missing Count': missing_by_col.values,
                    'Missing Percentage': missing_pct.values
                })
                
                fig_missing_cols = px.bar(
                    missing_df,
                    x='Column',
                    y='Missing Percentage',
                    text='Missing Count',
                    color='Missing Percentage',
                    color_continuous_scale='Reds',
                    title="Missing Values by Column"
                )
                
                fig_missing_cols.update_layout(
                    xaxis_title="Column",
                    yaxis_title="Missing Percentage (%)",
                    height=400
                )
                
                st.plotly_chart(fig_missing_cols, use_container_width=True)
            else:
                st.success("No missing values found in the dataset!")

elif st.session_state.current_view == "Quality Assessment":
    st.subheader("Data Quality Assessment")
    
    if st.session_state.data_dict is None:
        st.warning("Please upload a data file first.")
    else:
        # Create assessment options
        st.markdown("### Assessment Options")
        st.markdown("Select the quality dimensions to include in your assessment:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            completeness = st.checkbox("Completeness Analysis", value=True, 
                                     help="Check for missing values and empty fields")
            consistency = st.checkbox("Consistency Checks", value=True, 
                                    help="Verify data type consistency and value ranges")
        
        with col2:
            accuracy = st.checkbox("Accuracy Assessment", value=True, 
                                 help="Detect outliers and validate data accuracy")
            uniqueness = st.checkbox("Uniqueness Analysis", value=True, 
                                   help="Identify duplicate records")
        
        with col3:
            timeliness = st.checkbox("Timeliness Evaluation", value=True, 
                                   help="Assess date fields for timeliness")
            validity = st.checkbox("Validity Checks", value=True, 
                                 help="Validate format and business rule compliance")
        
        # Advanced options
        with st.expander("Advanced Options"):
            adv_tab1, adv_tab2, adv_tab3 = st.tabs(["Threshold Settings", "Attribute Selection", "Attribute Combinations"])
            
            with adv_tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Accuracy Settings**")
                    outlier_threshold = st.slider(
                        "Outlier Detection Threshold (z-score)",
                        min_value=2.0,
                        max_value=5.0,
                        value=3.0,
                        step=0.1,
                        help="Lower values detect more outliers"
                    )
                
                with col2:
                    st.markdown("**Consistency Settings**")
                    type_consistency_threshold = st.slider(
                        "Type Consistency Threshold (%)",
                        min_value=50,
                        max_value=99,
                        value=90,
                        step=1,
                        help="Minimum percentage of consistent type values required"
                    )
            
            # Attribute selection for focused assessment
            with adv_tab2:
                st.markdown("### Attribute-Level Assessment")
                st.markdown("Select specific attributes to focus your assessment on. If none are selected, all attributes will be assessed.")
                
                if st.session_state.data_dict:
                    # Get the first sheet's columns for attribute selection
                    first_sheet = list(st.session_state.data_dict.keys())[0]
                    available_columns = st.session_state.data_dict[first_sheet].columns.tolist()
                    
                    # Allow user to select attributes
                    selected_attributes = st.multiselect(
                        "Select attributes to assess",
                        options=available_columns,
                        default=[],
                        help="Choose specific columns to focus the assessment on"
                    )
                    
                    if selected_attributes:
                        st.success(f"Assessment will focus on {len(selected_attributes)} selected attributes")
                    else:
                        st.info("All attributes will be assessed")
                else:
                    st.warning("Please upload data first to see available attributes")
                    selected_attributes = []
            
            # Attribute combinations for relationship analysis
            with adv_tab3:
                st.markdown("### Attribute Combinations Analysis")
                st.markdown("Analyze relationships and dependencies between specific attribute combinations.")
                
                if st.session_state.data_dict:
                    # Get the first sheet's columns
                    first_sheet = list(st.session_state.data_dict.keys())[0]
                    available_columns = st.session_state.data_dict[first_sheet].columns.tolist()
                    
                    # Initialize attribute combinations container
                    if "attribute_combinations" not in st.session_state:
                        st.session_state.attribute_combinations = []
                    
                    # Add new combination
                    st.subheader("Add Attribute Combination")
                    
                    # Select columns for the combination
                    combo_cols = st.multiselect(
                        "Select 2 or more attributes to analyze together",
                        options=available_columns,
                        default=[],
                        help="Choose attributes to analyze relationships between them"
                    )
                    
                    if st.button("Add Combination", use_container_width=True):
                        if len(combo_cols) >= 2:
                            if tuple(combo_cols) not in st.session_state.attribute_combinations:
                                st.session_state.attribute_combinations.append(tuple(combo_cols))
                                st.success(f"Added combination: {', '.join(combo_cols)}")
                            else:
                                st.warning("This combination already exists")
                        else:
                            st.warning("Please select at least 2 attributes for a combination")
                    
                    # Display current combinations
                    if st.session_state.attribute_combinations:
                        st.subheader("Current Attribute Combinations")
                        for i, combo in enumerate(st.session_state.attribute_combinations):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"{i+1}. {', '.join(combo)}")
                            with col2:
                                if st.button(f"Remove", key=f"remove_combo_{i}"):
                                    st.session_state.attribute_combinations.pop(i)
                                    st.rerun()
                    else:
                        st.info("No attribute combinations added yet")
                else:
                    st.warning("Please upload data first to see available attributes")
                    st.session_state.attribute_combinations = []
        
        # Run assessment
        if st.button("Run Quality Assessment", use_container_width=True):
            assessment_options = {
                "completeness": completeness,
                "consistency": consistency,
                "accuracy": accuracy,
                "uniqueness": uniqueness,
                "timeliness": timeliness,
                "validity": validity,
                "advanced_options": {
                    "outlier_threshold": outlier_threshold,
                    "type_consistency_threshold": type_consistency_threshold / 100
                }
            }
            
            # Create a container for the animation
            animation_container = st.empty()
            
            # Perform assessment
            try:
                # Define the stages for our assessment process
                quality_dimensions = [dim for dim, enabled in assessment_options.items() if enabled and dim != "advanced_options"]
                total_stages = len(quality_dimensions) + 2  # +2 for initial setup and final processing
                
                # Initialize with first stage - Data preparation
                with animation_container.container():
                    processing_animation_with_stages(
                        stage=1, 
                        total_stages=total_stages,
                        stage_name="Data Preparation",
                        description="Preparing data structures for quality assessment..."
                    )
                
                time.sleep(0.5)  # Small pause for visual feedback
                
                # Create progress callback for updating stages
                def progress_callback(progress, message):
                    # Map progress percentage to stage number
                    # progress is 0-1, we need to map it to stage 2 through total_stages-1
                    if progress < 0.05:
                        current_stage = 1  # Still in preparation
                    elif progress > 0.95:
                        current_stage = total_stages - 1  # Almost done
                    else:
                        # Map progress to stages 2 through total_stages-1
                        # We subtract 2 from total_stages and add 2 to the result to skip stages 1 and total_stages
                        current_stage = min(
                            total_stages - 1, 
                            int(2 + (progress * (total_stages - 2)))
                        )
                    
                    # Update animation with current stage
                    with animation_container.container():
                        stage_index = min(current_stage-2, len(quality_dimensions)-1)
                        if stage_index >= 0 and stage_index < len(quality_dimensions):
                            stage_name = f"Assessing {quality_dimensions[stage_index].title()}"
                        else:
                            stage_name = "Processing"
                        
                        processing_animation_with_stages(
                            stage=current_stage, 
                            total_stages=total_stages,
                            stage_name=stage_name,
                            description=message
                        )
                
                # Prepare selected attributes and combinations
                selected_attrs = selected_attributes if 'selected_attributes' in locals() and selected_attributes else None
                attr_combinations = st.session_state.attribute_combinations if hasattr(st.session_state, 'attribute_combinations') and st.session_state.attribute_combinations else None
                
                # Perform quality assessment with our custom progress callback
                assessment_results = perform_data_quality_assessment(
                    st.session_state.data_dict, 
                    assessment_options,
                    selected_attributes=selected_attrs,
                    attribute_combinations=attr_combinations,
                    progress_callback=progress_callback
                )
                
                # Store in session state
                st.session_state.assessment_results = assessment_results
                
                # Save to history
                save_assessment_history(assessment_results, st.session_state.file_info)
                
                # Final stage - Assessment complete
                with animation_container.container():
                    processing_animation_with_stages(
                        stage=total_stages, 
                        total_stages=total_stages,
                        stage_name="Assessment Complete",
                        description="Quality assessment successfully completed!"
                    )
                
                time.sleep(0.8)  # Slightly longer pause at the end for visual confirmation
                
                # Perform ML-based anomaly detection
                if accuracy and st.session_state.data_dict:
                    # Replace animation with new loading card for ML processing
                    animation_container.empty()
                    with animation_container.container():
                        loading_card(
                            "Advanced Machine Learning Analysis", 
                            "Detecting anomalies and patterns in your data using AI techniques...",
                            animation_type="bounce"
                        )
                    
                    # Perform the ML analysis
                    ml_results = {}
                    for sheet_name, df in st.session_state.data_dict.items():
                        # Detect anomalies
                        anomaly_preds, anomaly_scores = detect_anomalies(df)
                        
                        # Perform clustering
                        clusters = perform_clustering(df)
                        
                        if anomaly_preds is not None or clusters is not None:
                            ml_results[sheet_name] = {
                                "anomalies": {
                                    "predictions": anomaly_preds,
                                    "scores": anomaly_scores
                                } if anomaly_preds is not None else None,
                                "clusters": clusters
                            }
                    
                    st.session_state.ml_results = ml_results
                
                # Clear animation container
                animation_container.empty()
                
                # Success message with custom styling
                st.markdown(
                    """
                    <div style="background-color: #f0fff0; border-left: 5px solid #228c22; padding: 15px; border-radius: 4px; margin: 10px 0;">
                        <h3 style="color: #228c22; margin-top: 0;">Assessment Completed Successfully!</h3>
                        <p>Your data quality assessment has been completed. View the results below.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            except Exception as e:
                # Clear animation container and show error
                animation_container.empty()
                st.error(f"An error occurred during assessment: {str(e)}")
        
        # Show assessment results if available
        if st.session_state.assessment_results:
            st.markdown("## Assessment Results")
            
            # Create tabs for results
            result_tab1, result_tab2, result_tab3 = st.tabs(["Summary", "Quality Dimensions", "ML Insights"])
            
            with result_tab1:
                # Show summary of results
                if "summary" in st.session_state.assessment_results:
                    summary = st.session_state.assessment_results["summary"]
                    
                    # Overall score
                    if "overall_scores" in summary and "overall" in summary["overall_scores"]:
                        overall_score = summary["overall_scores"]["overall"]
                        
                        # Score color based on value
                        if overall_score >= 90:
                            score_color = ARCADIS_GREEN
                            score_text = "Excellent"
                        elif overall_score >= 75:
                            score_color = ARCADIS_ORANGE
                            score_text = "Good"
                        else:
                            score_color = ARCADIS_RED
                            score_text = "Needs Improvement"
                        
                        st.markdown(
                            f"""
                            <div style="text-align: center; background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                                <h2 style="color: {score_color};">Overall Quality Score: {overall_score:.1f}/100</h2>
                                <p style="font-size: 18px; color: {score_color};">Data Quality: {score_text}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # Display dimension scores
                        st.markdown("### Quality Dimension Scores")
                        
                        dimension_names = {
                            "completeness": "Completeness",
                            "consistency": "Consistency",
                            "accuracy": "Accuracy",
                            "uniqueness": "Uniqueness",
                            "timeliness": "Timeliness",
                            "validity": "Validity"
                        }
                        
                        # Create columns for dimension scores
                        cols = st.columns(3)
                        col_idx = 0
                        
                        for dimension, name in dimension_names.items():
                            if dimension in summary["overall_scores"]:
                                score = summary["overall_scores"][dimension]
                                
                                # Determine color based on score
                                if score >= 90:
                                    color = ARCADIS_GREEN
                                elif score >= 75:
                                    color = ARCADIS_ORANGE
                                else:
                                    color = ARCADIS_RED
                                
                                with cols[col_idx % 3]:
                                    st.markdown(
                                        f"""
                                        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 10px;">
                                            <h4 style="margin-bottom: 10px;">{name}</h4>
                                            <div style="height: 10px; width: 100%; background-color: #e9ecef; border-radius: 5px;">
                                                <div style="height: 100%; width: {score}%; background-color: {color}; border-radius: 5px;"></div>
                                            </div>
                                            <p style="text-align: right; margin-top: 5px; font-weight: bold; color: {color};">{score:.1f}/100</p>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                                
                                col_idx += 1
                    
                    # Top issues
                    if "top_issues" in summary and summary["top_issues"]:
                        st.markdown("### Top Data Quality Issues")
                        
                        issue_data = []
                        for i, issue in enumerate(summary["top_issues"][:10], 1):
                            sheet = issue["sheet"]
                            column = issue["column"]
                            issue_desc = issue["issue"]
                            dimension = issue["dimension"]
                            severity = issue.get("severity", "Medium")
                            
                            issue_data.append({
                                "Issue #": i,
                                "Sheet": sheet,
                                "Column": column,
                                "Description": issue_desc,
                                "Dimension": dimension,
                                "Severity": severity
                            })
                        
                        issue_df = pd.DataFrame(issue_data)
                        
                        # Apply color highlighting based on severity
                        def color_severity(val):
                            if val == "High":
                                return f"background-color: {ARCADIS_RED}; color: white"
                            elif val == "Medium":
                                return f"background-color: {ARCADIS_ORANGE}; color: white"
                            else:
                                return f"background-color: {ARCADIS_GREEN}; color: white"
                        
                        styled_issues = issue_df.style.applymap(
                            color_severity, 
                            subset=["Severity"]
                        )
                        
                        st.dataframe(styled_issues, use_container_width=True)
                    
                    # Recommendations
                    if "recommendations" in summary and summary["recommendations"]:
                        st.markdown("### Key Recommendations")
                        
                        for i, rec in enumerate(summary["recommendations"][:5], 1):
                            dimension = rec["dimension"]
                            recommendation = rec["recommendation"]
                            priority = rec["priority"]
                            
                            # Priority color
                            if priority == "High":
                                priority_color = ARCADIS_RED
                            elif priority == "Medium":
                                priority_color = ARCADIS_ORANGE
                            else:
                                priority_color = ARCADIS_GREEN
                            
                            st.markdown(
                                f"""
                                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
                                    <h4 style="color: {ARCADIS_BLUE};">{i}. {dimension}</h4>
                                    <p style="margin-left: 20px;">{recommendation}</p>
                                    <p style="text-align: right; font-weight: bold; color: {priority_color};">Priority: {priority}</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
            
            with result_tab2:
                # Allow user to select a sheet to view detailed results
                if len(st.session_state.sheet_names) > 1:
                    selected_sheet = st.selectbox(
                        "Select a sheet to view detailed results",
                        st.session_state.sheet_names,
                        key="results_sheet_selector"
                    )
                else:
                    selected_sheet = st.session_state.sheet_names[0]
                
                # Get results for selected sheet
                if selected_sheet in st.session_state.assessment_results:
                    sheet_results = st.session_state.assessment_results[selected_sheet]
                    
                    # Create view selector for different analysis types
                    view_type = st.radio(
                        "Select analysis type",
                        ["Quality Dimensions", "Attribute Combinations"],
                        horizontal=True,
                        key="analysis_type_selector"
                    )
                    
                    if view_type == "Attribute Combinations":
                        # Show attribute combinations view
                        if "attribute_combinations" in sheet_results:
                            st.markdown("### Attribute Combinations Analysis")
                            st.markdown("This analysis shows the relationships and dependencies between selected combinations of attributes.")
                            
                            if not sheet_results["attribute_combinations"]:
                                st.info("No attribute combinations were analyzed. Add combinations in the 'Advanced Options' section when running an assessment.")
                            else:
                                # Create a selectbox for choosing which combination to view
                                combo_keys = list(sheet_results["attribute_combinations"].keys())
                                selected_combo = st.selectbox(
                                    "Select attribute combination to analyze",
                                    combo_keys,
                                    format_func=lambda x: x.replace("_", " + ")
                                )
                                
                                if selected_combo:
                                    combo_results = sheet_results["attribute_combinations"][selected_combo]
                                    
                                    # Display overall stats
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric("Total Rows", combo_results["total_rows"])
                                    
                                    with col2:
                                        st.metric("Complete Rows", combo_results["complete_rows"])
                                    
                                    with col3:
                                        st.metric("Completeness %", f"{combo_results['complete_rows_percentage']:.1f}%")
                                    
                                    # Quality score
                                    if "quality_score" in combo_results:
                                        score = combo_results["quality_score"]["overall"]
                                        st.markdown(
                                            f"""
                                            <div style="text-align: center; background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                                                <h3>Relationship Quality Score: {score:.1f}/100</h3>
                                            </div>
                                            """,
                                            unsafe_allow_html=True
                                        )
                                    
                                    # Show functional dependencies if they exist
                                    if "functional_dependencies" in combo_results and combo_results["functional_dependencies"]:
                                        st.markdown("### Functional Dependencies")
                                        st.markdown("Functional dependencies indicate when one attribute's value determines another attribute's value.")
                                        
                                        fd_data = []
                                        for dep, data in combo_results["functional_dependencies"].items():
                                            fd_data.append({
                                                "Dependency": dep,
                                                "Strength (%)": f"{data['fd_score']:.1f}%",
                                                "Category": data["fd_strength"],
                                                "Is Key": "Yes" if data["is_key"] else "No"
                                            })
                                        
                                        if fd_data:
                                            fd_df = pd.DataFrame(fd_data)
                                            st.dataframe(fd_df, use_container_width=True)
                                    
                                    # Show correlation if it exists
                                    if "correlation" in combo_results:
                                        st.markdown("### Correlation Analysis")
                                        
                                        correlation = combo_results["correlation"]["pearson"]
                                        strength = combo_results["correlation"]["strength"]
                                        
                                        # Determine color based on strength
                                        if strength == "Strong":
                                            corr_color = ARCADIS_GREEN if correlation > 0 else ARCADIS_RED
                                        elif strength == "Moderate":
                                            corr_color = ARCADIS_ORANGE
                                        else:
                                            corr_color = "gray"
                                        
                                        st.markdown(
                                            f"""
                                            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
                                                <h4>Pearson Correlation: <span style="color: {corr_color}">{correlation:.2f}</span></h4>
                                                <p>Strength: <strong>{strength}</strong></p>
                                                <p>Direction: <strong>{"Positive" if correlation > 0 else "Negative"}</strong></p>
                                            </div>
                                            """,
                                            unsafe_allow_html=True
                                        )
                                    
                                    # Show strong value associations if they exist
                                    if "strong_value_associations" in combo_results and combo_results["strong_value_associations"]:
                                        st.markdown("### Strong Value Associations")
                                        
                                        assoc_data = []
                                        for assoc in combo_results["strong_value_associations"]:
                                            assoc_data.append({
                                                "Value 1": assoc["value1"],
                                                "Value 2": assoc["value2"],
                                                "Co-occurrence (%)": f"{assoc['percentage']:.1f}%"
                                            })
                                        
                                        if assoc_data:
                                            assoc_df = pd.DataFrame(assoc_data)
                                            st.dataframe(assoc_df, use_container_width=True)
                                    
                                    # Show recommendations
                                    if "recommendations" in combo_results and combo_results["recommendations"]:
                                        st.markdown("### Recommendations")
                                        
                                        for rec in combo_results["recommendations"]:
                                            st.markdown(
                                                f"""
                                                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
                                                    <h4>{rec["category"]}</h4>
                                                    <p><strong>Issue:</strong> {rec["issue"]}</p>
                                                    <p><strong>Recommendation:</strong> {rec["recommendation"]}</p>
                                                </div>
                                                """,
                                                unsafe_allow_html=True
                                            )
                        else:
                            st.warning("No attribute combination analysis was performed in this assessment. Run a new assessment with attribute combinations defined.")
                    
                    else:  # Quality Dimensions view
                        # Create sub-tabs for different dimensions
                        quality_tabs = st.tabs([
                            "Completeness", 
                            "Consistency", 
                            "Accuracy", 
                            "Uniqueness", 
                            "Timeliness", 
                            "Validity"
                        ])
                        
                        # Completeness tab
                        with quality_tabs[0]:
                            st.markdown("### Completeness Analysis")
                            
                            if "completeness" in sheet_results:
                                completeness = sheet_results["completeness"]
                                overall_completeness = completeness["overall"]["completeness_percentage"]
                                completeness_score = completeness["overall"]["completeness_score"]
                                
                                # Display completeness score
                                st.markdown(
                                    f"""
                                    <div style="text-align: center; background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                                        <h3>Completeness Score: {completeness_score:.1f}/100</h3>
                                        <p>Overall data completeness: {overall_completeness:.2f}%</p>
                                        <p>Missing cells: {completeness["overall"]["missing_cells"]:,} out of {completeness["overall"]["total_cells"]:,} total cells</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                
                                # Show columns with missing values
                                if completeness["top_incomplete"]:
                                    # Create dataframe of incomplete columns
                                    incomplete_data = []
                                    for col, pct in completeness["top_incomplete"].items():
                                        missing_pct = 100 - pct
                                        missing_count = int(missing_pct * len(st.session_state.data_dict[selected_sheet]) / 100)
                                        incomplete_data.append({
                                            "Column": col,
                                            "Completeness %": pct,
                                            "Missing %": missing_pct,
                                            "Missing Count": missing_count
                                        })
                                    
                                    incomplete_df = pd.DataFrame(incomplete_data)
                                    incomplete_df = incomplete_df.sort_values("Missing %", ascending=False)
                                    
                                    # Bar chart of incomplete columns
                                    fig_incomplete = px.bar(
                                        incomplete_df,
                                        x="Column",
                                        y="Missing %",
                                        text="Missing Count",
                                        color="Missing %",
                                        color_continuous_scale="Reds",
                                        title="Columns with Missing Values"
                                    )
                                    
                                    fig_incomplete.update_layout(
                                        xaxis_title="Column",
                                        yaxis_title="Missing Percentage (%)",
                                        height=400
                                    )
                                    
                                    st.plotly_chart(fig_incomplete, use_container_width=True)
                                    
                                    # Show table of incomplete columns
                                    st.dataframe(incomplete_df, use_container_width=True)
                                else:
                                    st.success("No missing values found in this sheet!")
                                
                                # Add missing values visualization if available
                                if "visualizations" in sheet_results and "missing_values_plot" in sheet_results["visualizations"]:
                                    st.markdown("### Missing Values Pattern")
                                    st.plotly_chart(
                                        go.Figure(go.Layout(
                                            title="Missing Values Heatmap"
                                        )),
                                        use_container_width=True
                                    )
                            else:
                                st.info("Completeness analysis was not performed for this sheet.")
                        
                        # Consistency tab
                        with quality_tabs[1]:
                            st.markdown("### Consistency Analysis")
                            
                            if "consistency" in sheet_results:
                                consistency = sheet_results["consistency"]
                                
                                # Display consistency scores
                                type_score = consistency["overall"]["type_consistency_score"]
                                value_score = consistency["overall"]["value_consistency_score"]
                                overall_score = consistency["overall"]["overall_consistency_score"]
                                
                                # Display scores in metrics
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric(
                                        label="Overall Consistency Score",
                                        value=f"{overall_score:.1f}/100"
                                    )
                                
                                with col2:
                                    st.metric(
                                        label="Type Consistency Score",
                                        value=f"{type_score:.1f}/100"
                                    )
                                
                                with col3:
                                    st.metric(
                                        label="Value Consistency Score",
                                        value=f"{value_score:.1f}/100"
                                    )
                                
                                style_metric_cards()
                                
                                # Show type consistency issues
                                mixed_types = {col: data for col, data in consistency["type_consistency"].items() 
                                             if data.get("mixed_types", False)}
                                
                                if mixed_types:
                                    st.markdown("### Columns with Mixed Data Types")
                                    
                                    for col, data in mixed_types.items():
                                        st.markdown(
                                            f"""
                                            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
                                                <h4>{col}</h4>
                                                <p>Types found: {', '.join(data['types_found'])}</p>
                                                <p>Expected type: {data.get('expected_type', 'Unknown')}</p>
                                                <p>Consistency: {data.get('type_consistency_pct', 0):.2f}%</p>
                                            </div>
                                            """,
                                            unsafe_allow_html=True
                                        )
                                else:
                                    st.success("No data type inconsistencies found!")
                                
                                # Show value consistency issues
                                if "value_consistency" in consistency:
                                    outlier_cols = {col: data for col, data in consistency["value_consistency"].items() 
                                                  if data.get("outlier_percentage", 0) > 1}
                                    
                                    if outlier_cols:
                                        st.markdown("### Columns with Value Range Issues")
                                        
                                        outlier_data = []
                                        for col, data in outlier_cols.items():
                                            outlier_data.append({
                                                "Column": col,
                                                "Min": data["min"],
                                                "Max": data["max"],
                                                "Outlier Count": data["outlier_count"],
                                                "Outlier %": data["outlier_percentage"]
                                            })
                                        
                                        outlier_df = pd.DataFrame(outlier_data)
                                        
                                        # Bar chart of outlier percentages
                                        fig_outliers = px.bar(
                                            outlier_df,
                                            x="Column",
                                            y="Outlier %",
                                            text="Outlier Count",
                                            color="Outlier %",
                                            color_continuous_scale="Reds",
                                            title="Columns with Outliers"
                                        )
                                        
                                        fig_outliers.update_layout(
                                            xaxis_title="Column",
                                            yaxis_title="Outlier Percentage (%)",
                                            height=400
                                        )
                                        
                                        st.plotly_chart(fig_outliers, use_container_width=True)
                                        
                                        # Show table of outlier columns
                                        st.dataframe(outlier_df, use_container_width=True)
                                    else:
                                        st.success("No significant value range issues found!")
                            else:
                                st.info("Consistency analysis was not performed for this sheet.")
                    
                        # Accuracy tab
                        with quality_tabs[2]:
                            st.markdown("### Accuracy Analysis")
                            
                            if "accuracy" in sheet_results:
                                accuracy = sheet_results["accuracy"]
                                
                                # Display accuracy score
                                overall_score = accuracy["overall"]["overall_accuracy_score"]
                            
                                st.markdown(
                                    f"""
                                    <div style="text-align: center; background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                                        <h3>Accuracy Score: {overall_score:.1f}/100</h3>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                
                                # Show outlier information
                                if "column_accuracy" in accuracy:
                                    outlier_data = []
                                    for col, data in accuracy["column_accuracy"].items():
                                        if "outlier_count" in data and data["outlier_count"] > 0:
                                            outlier_data.append({
                                                "Column": col,
                                                "Min": data.get("min", "N/A"),
                                                "Max": data.get("max", "N/A"),
                                                "Mean": data.get("mean", "N/A"),
                                                "Median": data.get("median", "N/A"),
                                                "Std Dev": data.get("std", "N/A"),
                                                "Outlier Count": data["outlier_count"],
                                                "Outlier %": data.get("outlier_percentage", 0)
                                            })
                                    
                                    if outlier_data:
                                        outlier_df = pd.DataFrame(outlier_data)
                                        
                                        # Create visualization of outliers
                                        fig_accuracy = px.scatter(
                                            outlier_df,
                                            x="Column",
                                            y="Outlier %",
                                            size="Outlier Count",
                                            color="Outlier %",
                                            color_continuous_scale="Reds",
                                            title="Accuracy Issues by Column"
                                        )
                                        
                                        fig_accuracy.update_layout(
                                            xaxis_title="Column",
                                            yaxis_title="Outlier Percentage (%)",
                                            height=400
                                        )
                                        
                                        st.plotly_chart(fig_accuracy, use_container_width=True)
                                        
                                        # Show table of outlier information
                                        st.dataframe(outlier_df, use_container_width=True)
                                    else:
                                        st.success("No significant accuracy issues found!")
                                else:
                                    st.info("No column-level accuracy data available.")
                            else:
                                st.info("Accuracy analysis was not performed for this sheet.")
                    
                        # Uniqueness tab
                        with quality_tabs[3]:
                            st.markdown("### Uniqueness Analysis")
                            
                            if "uniqueness" in sheet_results:
                                uniqueness = sheet_results["uniqueness"]
                                
                                # Display uniqueness score
                                uniqueness_score = uniqueness["overall"]["uniqueness_score"]
                                duplicate_rows = uniqueness["overall"]["duplicate_rows"]
                                total_rows = uniqueness["overall"]["total_rows"]
                                duplicate_pct = (duplicate_rows / total_rows * 100) if total_rows > 0 else 0
                                
                                st.markdown(
                                    f"""
                                    <div style="text-align: center; background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                                        <h3>Uniqueness Score: {uniqueness_score:.1f}/100</h3>
                                        <p>Duplicate rows: {duplicate_rows:,} out of {total_rows:,} total rows ({duplicate_pct:.2f}%)</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                
                                # Show potential key columns
                                if uniqueness["key_candidates"]:
                                    st.markdown("### Potential Key Columns")
                                    st.markdown("The following columns have unique values for each row and could serve as primary keys:")
                                    
                                    key_candidates = pd.DataFrame({
                                        "Column": uniqueness["key_candidates"],
                                        "Uniqueness": ["100%" for _ in uniqueness["key_candidates"]]
                                    })
                                    
                                    st.dataframe(key_candidates, use_container_width=True)
                                else:
                                    st.warning("No potential key columns found. Consider adding a unique identifier column.")
                                
                                # Show high cardinality non-key columns
                                if uniqueness["high_cardinality_non_keys"]:
                                    st.markdown("### High Cardinality Non-Key Columns")
                                    st.markdown("These columns have high cardinality but are not unique. They may have data quality issues:")
                                    
                                    cardinality_data = []
                                    for item in uniqueness["high_cardinality_non_keys"]:
                                        cardinality_data.append({
                                            "Column": item["column"],
                                            "Unique Values": item["unique_values"],
                                            "Uniqueness Ratio": f"{item['uniqueness_ratio']*100:.2f}%"
                                        })
                                    
                                    cardinality_df = pd.DataFrame(cardinality_data)
                                    
                                    st.dataframe(cardinality_df, use_container_width=True)
                            else:
                                st.info("Uniqueness analysis was not performed for this sheet.")
                    
                        # Timeliness tab
                        with quality_tabs[4]:
                            st.markdown("### Timeliness Analysis")
                            
                            if "timeliness" in sheet_results and sheet_results["timeliness"]["overall"]["timeliness_score"] is not None:
                                timeliness = sheet_results["timeliness"]
                                
                                # Display timeliness score
                                timeliness_score = timeliness["overall"]["timeliness_score"]
                                date_columns_found = timeliness["overall"]["date_columns_found"]
                            
                                st.markdown(
                                    f"""
                                    <div style="text-align: center; background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                                        <h3>Timeliness Score: {timeliness_score:.1f}/100</h3>
                                        <p>Date columns analyzed: {date_columns_found}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                
                                # Show details for each date column
                                if "column_timeliness" in timeliness and timeliness["column_timeliness"]:
                                    st.markdown("### Date Column Analysis")
                                    
                                    date_data = []
                                    for col, data in timeliness["column_timeliness"].items():
                                        if data["min_date"] and data["max_date"]:
                                            date_data.append({
                                                "Column": col,
                                                "Min Date": data["min_date"],
                                                "Max Date": data["max_date"],
                                                "Time Span (days)": data["time_span_days"],
                                                "Recency (days)": data["recency_days"],
                                                "Timeliness Category": data["timeliness_category"]
                                            })
                                    
                                    if date_data:
                                        date_df = pd.DataFrame(date_data)
                                        
                                        # Create visualization of date ranges
                                        fig_dates = go.Figure()
                                        
                                        for i, row in enumerate(date_data):
                                            fig_dates.add_trace(go.Scatter(
                                                x=[row["Min Date"], row["Max Date"]],
                                                y=[i, i],
                                                mode="lines+markers",
                                                name=row["Column"],
                                                line=dict(width=4),
                                                marker=dict(size=10)
                                            ))
                                        
                                        fig_dates.update_layout(
                                            title="Date Ranges by Column",
                                            xaxis_title="Date",
                                            yaxis=dict(
                                                ticktext=[row["Column"] for row in date_data],
                                                tickvals=list(range(len(date_data))),
                                                title="Column"
                                            ),
                                            height=400
                                        )
                                        
                                        st.plotly_chart(fig_dates, use_container_width=True)
                                        
                                        # Show table of date columns
                                        st.dataframe(date_df, use_container_width=True)
                                    else:
                                        st.info("No valid date columns found in this sheet.")
                                else:
                                    st.info("No date columns found in this sheet.")
                            else:
                                st.info("Timeliness analysis was not performed for this sheet.")
                    
                        # Validity tab
                        with quality_tabs[5]:
                            st.markdown("### Validity Analysis")
                            
                            if "validity" in sheet_results:
                                validity = sheet_results["validity"]
                                
                                # Display validity score
                                overall_score = validity["overall"]["overall_validity_score"]
                                checks_performed = validity["overall"]["checks_performed"]
                            
                                st.markdown(
                                    f"""
                                    <div style="text-align: center; background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                                        <h3>Validity Score: {overall_score:.1f}/100</h3>
                                        <p>Number of checks performed: {checks_performed}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                
                                # Show validity check results
                                if "validity_checks" in validity and validity["validity_checks"]:
                                    st.markdown("### Format Validation Checks")
                                    
                                    validity_data = []
                                    for col, data in validity["validity_checks"].items():
                                        check_type = data["check_type"].replace("_", " ").title()
                                        validity_pct = data["validity_percentage"]
                                        
                                        # Determine status
                                        if validity_pct >= 95:
                                            status = "Valid"
                                        elif validity_pct >= 80:
                                            status = "Mostly Valid"
                                        else:
                                            status = "Invalid"
                                        
                                        validity_data.append({
                                            "Column": col,
                                            "Check Type": check_type,
                                            "Validity %": validity_pct,
                                            "Status": status
                                        })
                                    
                                    validity_df = pd.DataFrame(validity_data)
                                    
                                    # Create bar chart of validity percentages
                                    fig_validity = px.bar(
                                        validity_df,
                                        x="Column",
                                        y="Validity %",
                                        color="Status",
                                        text="Validity %",
                                        color_discrete_map={
                                            "Valid": ARCADIS_GREEN, 
                                            "Mostly Valid": ARCADIS_ORANGE, 
                                            "Invalid": ARCADIS_RED
                                        },
                                        title="Column Format Validity"
                                    )
                                    
                                    fig_validity.update_layout(
                                        xaxis_title="Column",
                                        yaxis_title="Validity Percentage (%)",
                                        height=400
                                    )
                                    
                                    st.plotly_chart(fig_validity, use_container_width=True)
                                    
                                    # Show table of validation results
                                    st.dataframe(validity_df, use_container_width=True)
                                else:
                                    st.info("No format validation checks were performed.")
                            else:
                                st.info("Validity analysis was not performed for this sheet.")
                else:
                    st.warning(f"No assessment results available for sheet: {selected_sheet}")
            
            with result_tab3:
                # Machine Learning Insights
                st.markdown("### Advanced Analytics Insights")
                
                if not st.session_state.ml_results:
                    st.info("Machine learning analysis hasn't been performed yet. Run the quality assessment with 'Accuracy Assessment' enabled.")
                else:
                    # Allow user to select a sheet
                    if len(st.session_state.sheet_names) > 1:
                        selected_ml_sheet = st.selectbox(
                            "Select a sheet to view ML insights",
                            st.session_state.sheet_names,
                            key="ml_sheet_selector"
                        )
                    else:
                        selected_ml_sheet = st.session_state.sheet_names[0]
                    
                    if selected_ml_sheet in st.session_state.ml_results:
                        ml_result = st.session_state.ml_results[selected_ml_sheet]
                        df = st.session_state.data_dict[selected_ml_sheet]
                        
                        # Create tabs for different ML insights
                        ml_tab1, ml_tab2 = st.tabs(["Anomaly Detection", "Cluster Analysis"])
                        
                        with ml_tab1:
                            st.markdown("### Anomaly Detection Results")
                            
                            if ml_result.get("anomalies") is not None:
                                anomaly_preds = ml_result["anomalies"]["predictions"]
                                anomaly_scores = ml_result["anomalies"]["scores"]
                                
                                # Count anomalies
                                anomaly_count = np.sum(anomaly_preds == -1)
                                anomaly_pct = anomaly_count / len(anomaly_preds) * 100
                                
                                st.markdown(
                                    f"""
                                    <div style="text-align: center; background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                                        <h3>Machine Learning Anomaly Detection</h3>
                                        <p>Anomalies found: {anomaly_count:,} out of {len(anomaly_preds):,} rows ({anomaly_pct:.2f}%)</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                
                                # Create anomaly score histogram
                                fig_anomaly = px.histogram(
                                    anomaly_scores,
                                    nbins=50,
                                    title="Distribution of Anomaly Scores",
                                    color_discrete_sequence=[ARCADIS_BLUE]
                                )
                                
                                fig_anomaly.add_vline(
                                    x=np.min(anomaly_scores[anomaly_preds == -1]) if np.any(anomaly_preds == -1) else 0,
                                    line_dash="dash",
                                    line_color=ARCADIS_RED,
                                    annotation_text="Anomaly Threshold"
                                )
                                
                                fig_anomaly.update_layout(
                                    xaxis_title="Anomaly Score (lower is more anomalous)",
                                    yaxis_title="Count",
                                    height=400
                                )
                                
                                st.plotly_chart(fig_anomaly, use_container_width=True)
                                
                                # Show anomalies in the data
                                if anomaly_count > 0:
                                    st.markdown("### Anomalous Records")
                                    st.markdown(f"Showing up to 20 records flagged as anomalies out of {anomaly_count:,} total anomalies.")
                                    
                                    anomaly_indices = np.where(anomaly_preds == -1)[0]
                                    anomaly_df = df.iloc[anomaly_indices].head(20).copy()
                                    
                                    # Add anomaly scores to display
                                    anomaly_df["Anomaly_Score"] = anomaly_scores[anomaly_indices][:20]
                                    
                                    st.dataframe(anomaly_df, use_container_width=True)
                                else:
                                    st.success("No anomalies were detected in this dataset!")
                            else:
                                st.info("Anomaly detection couldn't be performed. This typically happens when there aren't enough numeric columns.")
                        
                        with ml_tab2:
                            st.markdown("### Cluster Analysis")
                            
                            if ml_result.get("clusters") is not None:
                                clusters = ml_result["clusters"]
                                
                                # Count clusters
                                cluster_counts = np.bincount(clusters)
                                n_clusters = len(cluster_counts)
                                
                                st.markdown(
                                    f"""
                                    <div style="text-align: center; background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                                        <h3>Data Clustering Results</h3>
                                        <p>Number of clusters: {n_clusters}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                
                                # Create cluster distribution chart
                                cluster_df = pd.DataFrame({
                                    "Cluster": [f"Cluster {i}" for i in range(n_clusters)],
                                    "Count": cluster_counts
                                })
                                
                                fig_clusters = px.bar(
                                    cluster_df,
                                    x="Cluster",
                                    y="Count",
                                    color="Cluster",
                                    title="Distribution of Data Clusters",
                                    color_discrete_sequence=px.colors.qualitative.Bold
                                )
                                
                                fig_clusters.update_layout(
                                    xaxis_title="Cluster",
                                    yaxis_title="Number of Records",
                                    height=400
                                )
                                
                                st.plotly_chart(fig_clusters, use_container_width=True)
                                
                                # Select numeric columns for visualization
                                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                                
                                if len(numeric_cols) >= 2:
                                    st.markdown("### Cluster Visualization")
                                    st.markdown("Select two numeric columns to visualize clusters:")
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        x_axis = st.selectbox("X-Axis", numeric_cols, index=0, key="cluster_x")
                                    
                                    with col2:
                                        y_axis = st.selectbox("Y-Axis", numeric_cols, index=min(1, len(numeric_cols)-1), key="cluster_y")
                                    
                                    # Create scatter plot colored by cluster
                                    if x_axis != y_axis:
                                        scatter_df = df[[x_axis, y_axis]].copy()
                                        scatter_df["Cluster"] = [f"Cluster {c}" for c in clusters]
                                        
                                        fig_scatter = px.scatter(
                                            scatter_df,
                                            x=x_axis,
                                            y=y_axis,
                                            color="Cluster",
                                            title=f"Cluster Visualization ({x_axis} vs {y_axis})",
                                            color_discrete_sequence=px.colors.qualitative.Bold
                                        )
                                        
                                        fig_scatter.update_layout(
                                            xaxis_title=x_axis,
                                            yaxis_title=y_axis,
                                            height=500
                                        )
                                        
                                        st.plotly_chart(fig_scatter, use_container_width=True)
                                    else:
                                        st.info("Please select different columns for X and Y axes.")
                                
                                # Show data statistics by cluster
                                st.markdown("### Cluster Statistics")
                                
                                # Add cluster column to DataFrame for analysis
                                analysis_df = df.copy()
                                analysis_df["Cluster"] = clusters
                                
                                # Group by cluster and calculate statistics
                                if numeric_cols:
                                    cluster_stats = analysis_df.groupby("Cluster")[numeric_cols].agg(["mean", "min", "max", "std"])
                                    
                                    # Flatten hierarchical index
                                    cluster_stats.columns = [f"{col}_{stat}" for col, stat in cluster_stats.columns]
                                    
                                    # Reset index for nicer display
                                    cluster_stats = cluster_stats.reset_index()
                                    
                                    # Add cluster size
                                    cluster_sizes = analysis_df.groupby("Cluster").size().reset_index(name="Size")
                                    cluster_stats = pd.merge(cluster_stats, cluster_sizes, on="Cluster")
                                    
                                    # Add percentage
                                    cluster_stats["Percentage"] = cluster_stats["Size"] / len(analysis_df) * 100
                                    
                                    # Format cluster column
                                    cluster_stats["Cluster"] = cluster_stats["Cluster"].apply(lambda x: f"Cluster {x}")
                                    
                                    st.dataframe(cluster_stats, use_container_width=True)
                                else:
                                    st.info("No numeric columns available for cluster statistics.")
                            else:
                                st.info("Cluster analysis couldn't be performed. This typically happens when there aren't enough numeric columns.")
                    else:
                        st.warning(f"No machine learning results available for sheet: {selected_ml_sheet}")

elif st.session_state.current_view == "Advanced Analytics":
    st.subheader("Advanced Data Analytics")
    
    if st.session_state.data_dict is None:
        st.warning("Please upload a data file first.")
    else:
        # Sheet selector
        if len(st.session_state.sheet_names) > 1:
            selected_sheet = st.selectbox(
                "Select a sheet to analyze",
                st.session_state.sheet_names,
                key="advanced_sheet_selector"
            )
        else:
            selected_sheet = st.session_state.sheet_names[0]
        
        df = st.session_state.data_dict[selected_sheet]
        
        # Create tabs for different advanced analyses
        adv_tab1, adv_tab2, adv_tab3 = st.tabs(["Anomaly Detection", "Clustering Analysis", "Trend Forecasting"])
        
        with adv_tab1:
            st.markdown("### Anomaly Detection")
            st.markdown("""
            This feature uses machine learning to identify anomalous records in your dataset.
            The algorithm will flag records that significantly deviate from the normal pattern.
            """)
            
            # Anomaly detection settings
            st.markdown("#### Detection Settings")
            
            contamination = st.slider(
                "Contamination Factor (expected % of anomalies)",
                min_value=0.01,
                max_value=0.2,
                value=0.05,
                step=0.01,
                help="Higher values will detect more anomalies"
            )
            
            # Select numeric columns to use
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                st.warning("Anomaly detection requires at least 2 numeric columns. This dataset doesn't have enough numeric columns.")
            else:
                selected_cols = st.multiselect(
                    "Select columns to use for anomaly detection",
                    numeric_cols,
                    default=numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
                )
                
                if len(selected_cols) < 2:
                    st.warning("Please select at least 2 columns for analysis.")
                else:
                    if st.button("Detect Anomalies", use_container_width=True):
                        with st.spinner("Detecting anomalies..."):
                            # Select only the chosen columns
                            df_selected = df[selected_cols].copy()
                            
                            # Fill NaN values with column mean
                            df_selected = df_selected.fillna(df_selected.mean())
                            
                            # Apply Isolation Forest
                            model = IsolationForest(contamination=contamination, random_state=42)
                            preds = model.fit_predict(df_selected)
                            anomaly_score = model.decision_function(df_selected)
                            
                            # Count anomalies
                            anomaly_count = np.sum(preds == -1)
                            anomaly_pct = anomaly_count / len(preds) * 100
                            
                            st.success(f"Analysis complete! Found {anomaly_count} anomalies ({anomaly_pct:.2f}% of data).")
                            
                            # Display results
                            st.markdown("#### Anomaly Detection Results")
                            
                            # Create a histogram of anomaly scores
                            fig_scores = px.histogram(
                                anomaly_score,
                                nbins=50,
                                title="Distribution of Anomaly Scores",
                                color_discrete_sequence=[ARCADIS_BLUE]
                            )
                            
                            fig_scores.add_vline(
                                x=np.min(anomaly_score[preds == -1]) if np.any(preds == -1) else 0,
                                line_dash="dash",
                                line_color=ARCADIS_RED,
                                annotation_text="Anomaly Threshold"
                            )
                            
                            fig_scores.update_layout(
                                xaxis_title="Anomaly Score (lower is more anomalous)",
                                yaxis_title="Count",
                                height=400
                            )
                            
                            st.plotly_chart(fig_scores, use_container_width=True)
                            
                            # Display the most significant anomalies
                            if anomaly_count > 0:
                                st.markdown("#### Top Anomalies")
                                
                                # Add anomaly info to dataframe
                                df_anomaly = df.copy()
                                df_anomaly["Anomaly_Flag"] = preds == -1
                                df_anomaly["Anomaly_Score"] = anomaly_score
                                
                                # Sort by score and show the top anomalies
                                df_anomalies = df_anomaly[df_anomaly["Anomaly_Flag"]].sort_values("Anomaly_Score").head(20)
                                
                                st.dataframe(df_anomalies, use_container_width=True)
                                
                                # Create 2D visualization of anomalies
                                if len(selected_cols) >= 2:
                                    st.markdown("#### Anomaly Visualization")
                                    
                                    # Create a scatter plot with the first two selected columns
                                    x_col = selected_cols[0]
                                    y_col = selected_cols[1]
                                    
                                    fig_scatter = px.scatter(
                                        df_anomaly,
                                        x=x_col,
                                        y=y_col,
                                        color="Anomaly_Flag",
                                        color_discrete_map={True: ARCADIS_RED, False: ARCADIS_BLUE},
                                        opacity=0.7,
                                        title=f"Anomaly Visualization ({x_col} vs {y_col})"
                                    )
                                    
                                    fig_scatter.update_layout(
                                        xaxis_title=x_col,
                                        yaxis_title=y_col,
                                        height=500,
                                        legend_title="Is Anomaly"
                                    )
                                    
                                    st.plotly_chart(fig_scatter, use_container_width=True)
                            else:
                                st.info("No anomalies detected in the dataset with the current settings.")
        
        with adv_tab2:
            st.markdown("### Cluster Analysis")
            st.markdown("""
            This feature uses machine learning to identify natural groupings in your data.
            Clustering can help identify patterns and segment your data into meaningful groups.
            """)
            
            # Clustering settings
            st.markdown("#### Clustering Settings")
            
            n_clusters = st.slider(
                "Number of Clusters",
                min_value=2,
                max_value=10,
                value=3,
                step=1,
                help="The number of clusters to identify in the data"
            )
            
            # Select numeric columns to use
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                st.warning("Clustering requires at least 2 numeric columns. This dataset doesn't have enough numeric columns.")
            else:
                selected_cols = st.multiselect(
                    "Select columns to use for clustering",
                    numeric_cols,
                    default=numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols,
                    key="cluster_col_select"
                )
                
                if len(selected_cols) < 2:
                    st.warning("Please select at least 2 columns for analysis.")
                else:
                    if st.button("Perform Clustering", use_container_width=True):
                        with st.spinner("Identifying clusters..."):
                            # Select only the chosen columns
                            df_selected = df[selected_cols].copy()
                            
                            # Fill NaN values with column mean
                            df_selected = df_selected.fillna(df_selected.mean())
                            
                            # Apply KMeans clustering
                            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                            clusters = kmeans.fit_predict(df_selected)
                            
                            # Add cluster info to dataframe
                            df_clustered = df.copy()
                            df_clustered["Cluster"] = clusters
                            
                            st.success(f"Clustering complete! Identified {n_clusters} clusters in the data.")
                            
                            # Display results
                            st.markdown("#### Cluster Analysis Results")
                            
                            # Count records in each cluster
                            cluster_counts = np.bincount(clusters)
                            
                            # Create a bar chart of cluster sizes
                            cluster_df = pd.DataFrame({
                                "Cluster": [f"Cluster {i}" for i in range(n_clusters)],
                                "Count": cluster_counts,
                                "Percentage": cluster_counts / len(clusters) * 100
                            })
                            
                            fig_counts = px.bar(
                                cluster_df,
                                x="Cluster",
                                y="Count",
                                text=cluster_df["Percentage"].apply(lambda x: f"{x:.1f}%"),
                                color="Cluster",
                                title="Distribution of Records Across Clusters",
                                color_discrete_sequence=px.colors.qualitative.Bold
                            )
                            
                            fig_counts.update_layout(
                                xaxis_title="Cluster",
                                yaxis_title="Number of Records",
                                height=400
                            )
                            
                            st.plotly_chart(fig_counts, use_container_width=True)
                            
                            # Create 2D visualization of clusters
                            st.markdown("#### Cluster Visualization")
                            
                            # Let the user select which columns to visualize
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                x_col = st.selectbox("X-Axis", selected_cols, index=0, key="viz_x_col")
                            
                            with col2:
                                y_col = st.selectbox("Y-Axis", selected_cols, index=min(1, len(selected_cols)-1), key="viz_y_col")
                            
                            # Create scatter plot
                            if x_col != y_col:
                                df_viz = df_clustered[[x_col, y_col, "Cluster"]].copy()
                                df_viz["Cluster"] = df_viz["Cluster"].apply(lambda x: f"Cluster {x}")
                                
                                fig_scatter = px.scatter(
                                    df_viz,
                                    x=x_col,
                                    y=y_col,
                                    color="Cluster",
                                    title=f"Cluster Visualization ({x_col} vs {y_col})",
                                    color_discrete_sequence=px.colors.qualitative.Bold
                                )
                                
                                fig_scatter.update_layout(
                                    xaxis_title=x_col,
                                    yaxis_title=y_col,
                                    height=500
                                )
                                
                                st.plotly_chart(fig_scatter, use_container_width=True)
                            else:
                                st.info("Please select different columns for X and Y axes.")
                            
                            # Show cluster statistics
                            st.markdown("#### Cluster Statistics")
                            
                            # Calculate statistics for numeric columns by cluster
                            cluster_stats = df_clustered.groupby("Cluster")[numeric_cols].agg(["mean", "median", "std", "min", "max"])
                            
                            # Show cluster stats for selected columns
                            for col in selected_cols:
                                col_stats = cluster_stats[col].reset_index()
                                
                                # Format cluster column
                                col_stats["Cluster"] = col_stats["Cluster"].apply(lambda x: f"Cluster {x}")
                                
                                st.markdown(f"##### Statistics for: {col}")
                                st.table(col_stats)
                            
                            # Sample records from each cluster
                            st.markdown("#### Sample Records by Cluster")
                            
                            selected_cluster = st.selectbox(
                                "Select cluster to view sample records",
                                [f"Cluster {i}" for i in range(n_clusters)]
                            )
                            
                            cluster_idx = int(selected_cluster.split(" ")[1])
                            cluster_samples = df_clustered[df_clustered["Cluster"] == cluster_idx].head(10)
                            
                            st.dataframe(cluster_samples, use_container_width=True)
        
        with adv_tab3:
            st.markdown("### Trend Forecasting")
            st.markdown("""
            This feature analyzes time series data to identify trends and forecast future values.
            Select a date column and a target column to begin analysis.
            """)
            
            # Check for datetime columns
            date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
            
            # Try to find potential date columns that might not be in datetime format
            for col in df.select_dtypes(include=["object"]).columns:
                # Sample the column
                sample = df[col].dropna().head(10)
                try:
                    if all(pd.to_datetime(val, errors="coerce") is not pd.NaT for val in sample):
                        date_cols.append(col)
                except:
                    pass
            
            if not date_cols:
                st.warning("No date columns found. Trend analysis requires at least one date/datetime column.")
            else:
                # Date column selection
                date_col = st.selectbox(
                    "Select date column",
                    date_cols
                )
                
                # Target column (numeric only)
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if not numeric_cols:
                    st.warning("No numeric columns found. Trend analysis requires at least one numeric column to forecast.")
                else:
                    target_col = st.selectbox(
                        "Select column to forecast",
                        numeric_cols
                    )
                    
                    # Forecast settings
                    st.markdown("#### Forecast Settings")
                    
                    forecast_periods = st.slider(
                        "Number of periods to forecast",
                        min_value=1,
                        max_value=30,
                        value=7,
                        help="Number of future time periods to predict"
                    )
                    
                    # Group by time period
                    if st.button("Analyze Trend", use_container_width=True):
                        # Ensure date column is datetime
                        try:
                            df_trend = df.copy()
                            if df_trend[date_col].dtype != "datetime64[ns]":
                                df_trend[date_col] = pd.to_datetime(df_trend[date_col], errors="coerce")
                            
                            # Drop rows with invalid dates
                            df_trend = df_trend.dropna(subset=[date_col])
                            
                            # Sort by date
                            df_trend = df_trend.sort_values(date_col)
                            
                            # Group by month
                            df_monthly = df_trend.set_index(date_col).resample("M")[[target_col]].mean().reset_index()
                            
                            # Create time series plot
                            fig_ts = px.line(
                                df_monthly,
                                x=date_col,
                                y=target_col,
                                markers=True,
                                title=f"Time Series Analysis of {target_col}",
                                color_discrete_sequence=[ARCADIS_BLUE]
                            )
                            
                            fig_ts.update_layout(
                                xaxis_title="Date",
                                yaxis_title=target_col,
                                height=400
                            )
                            
                            st.plotly_chart(fig_ts, use_container_width=True)
                            
                            # Simple forecasting
                            st.markdown("#### Trend Forecast")
                            
                            # Create a time index
                            df_forecast = df_monthly.copy()
                            df_forecast["time_idx"] = range(len(df_forecast))
                            
                            # Train a simple linear model
                            from sklearn.linear_model import LinearRegression
                            model = LinearRegression()
                            model.fit(df_forecast[["time_idx"]], df_forecast[target_col])
                            
                            # Generate future dates
                            last_date = df_forecast[date_col].iloc[-1]
                            future_dates = pd.date_range(
                                start=last_date + pd.Timedelta(days=1),
                                periods=forecast_periods,
                                freq="M"
                            )
                            
                            # Create forecast dataframe
                            future_df = pd.DataFrame({
                                date_col: future_dates,
                                "time_idx": range(
                                    df_forecast["time_idx"].max() + 1,
                                    df_forecast["time_idx"].max() + 1 + forecast_periods
                                )
                            })
                            
                            # Generate predictions
                            future_df[target_col] = model.predict(future_df[["time_idx"]])
                            
                            # Combine historical and forecast data
                            forecast_plot_df = pd.concat([
                                df_forecast[[date_col, target_col]].assign(Type="Historical"),
                                future_df[[date_col, target_col]].assign(Type="Forecast")
                            ])
                            
                            # Create forecast plot
                            fig_forecast = px.line(
                                forecast_plot_df,
                                x=date_col,
                                y=target_col,
                                color="Type",
                                line_dash="Type",
                                color_discrete_map={"Historical": ARCADIS_BLUE, "Forecast": ARCADIS_ORANGE},
                                title=f"Forecast of {target_col} ({forecast_periods} periods)"
                            )
                            
                            fig_forecast.update_layout(
                                xaxis_title="Date",
                                yaxis_title=target_col,
                                height=400
                            )
                            
                            st.plotly_chart(fig_forecast, use_container_width=True)
                            
                            # Show forecast values
                            st.markdown("#### Forecast Values")
                            st.dataframe(future_df[[date_col, target_col]], use_container_width=True)
                            
                            # Calculate trend statistics
                            st.markdown("#### Trend Statistics")
                            
                            # Calculate average change
                            historical_values = df_monthly[target_col].values
                            avg_change = np.mean(np.diff(historical_values))
                            pct_change = avg_change / np.mean(historical_values) * 100
                            
                            # Calculate trend direction
                            if avg_change > 0:
                                trend_direction = "Increasing"
                                trend_color = ARCADIS_GREEN
                            elif avg_change < 0:
                                trend_direction = "Decreasing"
                                trend_color = ARCADIS_RED
                            else:
                                trend_direction = "Stable"
                                trend_color = ARCADIS_BLUE
                            
                            # Create metrics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    label="Trend Direction",
                                    value=trend_direction,
                                    delta=f"{pct_change:.2f}% per period",
                                    delta_color="normal" if avg_change >= 0 else "inverse"
                                )
                            
                            with col2:
                                st.metric(
                                    label="Current Value",
                                    value=f"{historical_values[-1]:.2f}",
                                    delta=None
                                )
                            
                            with col3:
                                st.metric(
                                    label="Forecasted End Value",
                                    value=f"{future_df[target_col].iloc[-1]:.2f}",
                                    delta=f"{future_df[target_col].iloc[-1] - historical_values[-1]:.2f}",
                                    delta_color="normal" if future_df[target_col].iloc[-1] >= historical_values[-1] else "inverse"
                                )
                            
                            style_metric_cards()
                            
                            # Calculate seasonality if enough data
                            if len(df_monthly) >= 12:
                                st.markdown("#### Seasonality Analysis")
                                
                                # Decompose the time series
                                try:
                                    from statsmodels.tsa.seasonal import seasonal_decompose
                                    
                                    # Set the date as index
                                    ts = df_monthly.set_index(date_col)[target_col]
                                    
                                    # Decompose the series
                                    result = seasonal_decompose(ts, model='additive', period=12)
                                    
                                    # Create decomposition plots
                                    trend = pd.DataFrame({'Date': result.trend.index, 'Trend': result.trend.values})
                                    seasonal = pd.DataFrame({'Date': result.seasonal.index, 'Seasonality': result.seasonal.values})
                                    residual = pd.DataFrame({'Date': result.resid.index, 'Residual': result.resid.values})
                                    
                                    # Plot trend component
                                    fig_trend = px.line(
                                        trend,
                                        x='Date',
                                        y='Trend',
                                        title="Trend Component",
                                        color_discrete_sequence=[ARCADIS_BLUE]
                                    )
                                    
                                    fig_trend.update_layout(
                                        xaxis_title="Date",
                                        yaxis_title="Trend",
                                        height=300
                                    )
                                    
                                    # Plot seasonal component
                                    fig_seasonal = px.line(
                                        seasonal,
                                        x='Date',
                                        y='Seasonality',
                                        title="Seasonal Component",
                                        color_discrete_sequence=[ARCADIS_ORANGE]
                                    )
                                    
                                    fig_seasonal.update_layout(
                                        xaxis_title="Date",
                                        yaxis_title="Seasonality",
                                        height=300
                                    )
                                    
                                    # Show plots
                                    st.plotly_chart(fig_trend, use_container_width=True)
                                    st.plotly_chart(fig_seasonal, use_container_width=True)
                                    
                                    # Identify seasonal patterns
                                    month_seasonal = seasonal.copy()
                                    month_seasonal['Month'] = month_seasonal['Date'].dt.month
                                    monthly_pattern = month_seasonal.groupby('Month')['Seasonality'].mean().reset_index()
                                    
                                    # Month names
                                    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                                    monthly_pattern['Month_Name'] = monthly_pattern['Month'].apply(lambda x: month_names[x-1])
                                    
                                    # Create monthly pattern plot
                                    fig_pattern = px.bar(
                                        monthly_pattern,
                                        x='Month_Name',
                                        y='Seasonality',
                                        title="Monthly Seasonal Pattern",
                                        color='Seasonality',
                                        color_continuous_scale="RdBu_r"
                                    )
                                    
                                    fig_pattern.update_layout(
                                        xaxis_title="Month",
                                        yaxis_title="Seasonal Effect",
                                        height=300
                                    )
                                    
                                    st.plotly_chart(fig_pattern, use_container_width=True)
                                    
                                    # Identify strong seasonal months
                                    top_seasonal = monthly_pattern.copy()
                                    top_seasonal['Abs_Seasonality'] = abs(top_seasonal['Seasonality'])
                                    top_seasonal = top_seasonal.sort_values('Abs_Seasonality', ascending=False).head(3)
                                    
                                    st.markdown("#### Key Seasonal Insights")
                                    
                                    for _, row in top_seasonal.iterrows():
                                        direction = "increase" if row['Seasonality'] > 0 else "decrease"
                                        st.markdown(f"- **{row['Month_Name']}** typically shows a {direction} of **{abs(row['Seasonality']):.2f}** units")
                                    
                                except Exception as e:
                                    st.warning(f"Could not perform seasonality analysis: {str(e)}")
                            else:
                                st.info("Not enough data for seasonality analysis. At least 12 periods are needed.")
                        
                        except Exception as e:
                            st.error(f"Error in trend analysis: {str(e)}")

elif st.session_state.current_view == "Insights & Reports":
    st.subheader("Insights & Reports")
    
    # Create tabs for the three sections
    ir_tab1, ir_tab2, ir_tab3 = st.tabs(["Actionable Insights", "Reports", "Historical Trends"])
    
    # Actionable Insights Tab
    with ir_tab1:
        if st.session_state.assessment_results is None:
            st.warning("Please run a quality assessment first to generate actionable insights.")
        else:
            st.markdown("""
        This section provides prioritized, actionable recommendations based on the quality assessment results.
        These insights are designed to help improve data quality with specific, implementable actions.
        """)
        
        # Container for the animation
        animation_container = st.empty()
        
        # Create tabs for different views of insights
        insights_tab1, insights_tab2, insights_tab3 = st.tabs(["Top Recommendations", "By Role", "Implementation Plan"])
        
        # Generate insights from assessment results
        if "insights" not in st.session_state or st.button("Regenerate Insights", use_container_width=True):
            
            # Show animation during processing
            with animation_container.container():
                loading_card(
                    "Generating Actionable Insights", 
                    "Analyzing assessment results and prioritizing recommendations...",
                    animation_type="bounce"
                )
                
            # Generate insights
            try:
                insights = generate_actionable_insights(st.session_state.assessment_results)
                st.session_state.insights = insights
                animation_container.empty()
            except Exception as e:
                st.error(f"Error generating insights: {str(e)}")
                animation_container.empty()
                st.stop()
        
        with insights_tab1:
            st.markdown("### Top Priority Recommendations")
            
            # Executive summary metrics
            if "expected_improvements" in st.session_state.insights:
                improvements = st.session_state.insights["expected_improvements"]
                if "projected_scores" in improvements and "overall" in improvements["projected_scores"]:
                    current_score = st.session_state.assessment_results.get("summary", {}).get("overall_scores", {}).get("overall", 0)
                    projected_score = improvements["projected_scores"]["overall"]
                    improvement = projected_score - current_score
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            label="Current Quality Score", 
                            value=f"{current_score:.1f}/100"
                        )
                    with col2:
                        st.metric(
                            label="Projected Quality Score", 
                            value=f"{projected_score:.1f}/100",
                            delta=f"+{improvement:.1f}" if improvement > 0 else f"{improvement:.1f}"
                        )
                    with col3:
                        # Show estimated effort
                        quick_wins = len(improvements.get("implementation_phases", {}).get("quick_wins", []))
                        st.metric(
                            label="Quick Win Opportunities", 
                            value=f"{quick_wins}"
                        )
                    
                    style_metric_cards()
            
            # Display top recommendations
            for i, insight in enumerate(st.session_state.insights.get("top_recommendations", [])):
                with st.expander(f"Priority {i+1}: {insight['title']}", expanded=i == 0):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Description:** {insight['description']}")
                        
                        st.markdown("**Actions:**")
                        for j, action in enumerate(insight['actions']):
                            st.markdown(f"{j+1}. {action}")
                        
                        if "improved_metrics" in insight:
                            st.markdown("**Expected Improvements:**")
                            for metric, value in insight['improved_metrics'].items():
                                metric_name = metric.replace("_", " ").title()
                                st.markdown(f"- {metric_name}: {value}")
                    
                    with col2:
                        # Priority label with color coding
                        priority = insight.get("priority", "medium").lower()
                        if priority == "critical":
                            priority_color = "#DC3545"  # Red
                        elif priority == "high":
                            priority_color = "#FD7E14"  # Orange
                        elif priority == "medium":
                            priority_color = "#FFC107"  # Yellow
                        else:
                            priority_color = "#28A745"  # Green
                        
                        st.markdown(
                            f"""
                            <div style="background-color: {priority_color}; padding: 5px 10px; border-radius: 4px; color: white; text-align: center; margin-bottom: 10px;">
                                <strong>{priority.upper()}</strong>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        
                        # Dimension badge
                        dimension = insight.get("dimension", "").title()
                        st.markdown(
                            f"""
                            <div style="background-color: #6C757D; padding: 5px 10px; border-radius: 4px; color: white; text-align: center; margin-bottom: 10px;">
                                {dimension}
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        
                        # Effort estimation
                        effort = insight.get("implementation", {}).get("effort", "unknown").title()
                        st.markdown(f"**Effort:** {effort}")
                        
                        # Impact visualization
                        impact = insight.get("business_impact", 0.5)
                        st.markdown("**Business Impact:**")
                        st.progress(impact)
            
            # If no recommendations
            if not st.session_state.insights.get("top_recommendations"):
                st.info("No high-priority recommendations identified.")
        
        with insights_tab2:
            st.markdown("### Insights By Role")
            
            role_tab1, role_tab2, role_tab3 = st.tabs(["Data Team", "Business Users", "Executive"])
            
            with role_tab1:
                st.markdown("#### Recommendations for Data Team")
                
                for i, insight in enumerate(st.session_state.insights.get("by_role", {}).get("data_team", [])):
                    with st.expander(f"{i+1}. {insight['title']}", expanded=i == 0):
                        st.markdown(f"**Description:** {insight['description']}")
                        
                        # Code example if available
                        if "implementation" in insight and "code_example" in insight["implementation"]:
                            st.code(insight["implementation"]["code_example"], language="python")
                        
                        # Tools needed
                        if "implementation" in insight and "tools" in insight["implementation"]:
                            st.markdown("**Tools Needed:**")
                            for tool in insight["implementation"]["tools"]:
                                st.markdown(f"- {tool}")
            
            with role_tab2:
                st.markdown("#### Recommendations for Business Users")
                
                for i, insight in enumerate(st.session_state.insights.get("by_role", {}).get("business_users", [])):
                    with st.expander(f"{i+1}. {insight['title']}", expanded=i == 0):
                        st.markdown(f"**Description:** {insight['description']}")
                        
                        st.markdown("**Actions:**")
                        for j, action in enumerate(insight['actions']):
                            st.markdown(f"{j+1}. {action}")
                        
                        if "improved_metrics" in insight:
                            st.markdown("**Business Benefits:**")
                            for metric, value in insight['improved_metrics'].items():
                                if metric == "overall_quality":
                                    st.markdown(f"- Overall Data Quality: {value}")
            
            with role_tab3:
                st.markdown("#### Executive Summary")
                
                # Generate and display executive summary
                if "insights" in st.session_state:
                    exec_summary = get_executive_summary(st.session_state.insights)
                    st.markdown(exec_summary)
                
                # Implementation phases summary
                if "expected_improvements" in st.session_state.insights and "implementation_phases" in st.session_state.insights["expected_improvements"]:
                    phases = st.session_state.insights["expected_improvements"]["implementation_phases"]
                    
                    st.markdown("### Implementation Roadmap")
                    
                    # Phase 1: Quick Wins
                    if phases.get("quick_wins"):
                        st.markdown("#### Phase 1: Quick Wins (1-5 days)")
                        for win in phases["quick_wins"]:
                            st.markdown(f"- {win['title']} ({win['gain']} improvement)")
                    
                    # Phase 2: Medium Term
                    if phases.get("medium_term"):
                        st.markdown("#### Phase 2: Medium-Term Initiatives (1-2 weeks)")
                        for initiative in phases["medium_term"]:
                            st.markdown(f"- {initiative['title']} ({initiative['gain']} improvement)")
                    
                    # Phase 3: Long Term
                    if phases.get("long_term"):
                        st.markdown("#### Phase 3: Strategic Initiatives (2+ weeks)")
                        for initiative in phases["long_term"]:
                            st.markdown(f"- {initiative['title']} ({initiative['gain']} improvement)")
        
        with insights_tab3:
            st.markdown("### Implementation Plan")
            
            # Display insights by priority
            for priority in ["critical", "high", "medium", "low"]:
                priority_insights = st.session_state.insights.get("by_priority", {}).get(priority, [])
                
                if priority_insights:
                    st.markdown(f"#### {priority.title()} Priority Items ({len(priority_insights)})")
                    
                    # Create expandable sections for each insight
                    for i, insight in enumerate(priority_insights):
                        with st.expander(f"{insight['title']}", expanded=i == 0 and priority == "critical"):
                            st.markdown(f"**Description:** {insight['description']}")
                            
                            col1, col2 = st.columns([3, 2])
                            
                            with col1:
                                st.markdown("**Implementation Steps:**")
                                for j, action in enumerate(insight['actions']):
                                    st.markdown(f"{j+1}. {action}")
                                
                                # Show code example if available
                                if "implementation" in insight and "code_example" in insight["implementation"]:
                                    with st.expander("Implementation Code Example"):
                                        st.code(insight["implementation"]["code_example"], language="python")
                            
                            with col2:
                                # Effort and tools
                                st.markdown(f"**Effort:** {insight.get('implementation', {}).get('effort', 'Unknown').title()}")
                                
                                # Expected improvements
                                if "improved_metrics" in insight:
                                    st.markdown("**Expected Improvements:**")
                                    metrics_text = []
                                    for metric, value in insight['improved_metrics'].items():
                                        metric_name = metric.replace("_", " ").title()
                                        metrics_text.append(f"{metric_name}: {value}")
                                    st.markdown("<br>".join(metrics_text), unsafe_allow_html=True)
                                
                                # Key stakeholders based on roles
                                st.markdown("**Key Stakeholders:**")
                                roles = []
                                if insight.get("roles", {}).get("data_team", 0) > 0.6:
                                    roles.append("Data Team")
                                if insight.get("roles", {}).get("business_users", 0) > 0.6:
                                    roles.append("Business Users")
                                if insight.get("roles", {}).get("executive", 0) > 0.3:
                                    roles.append("Executive")
                                
                                st.markdown(", ".join(roles))
            
            # If no implementation plan
            if not any(st.session_state.insights.get("by_priority", {}).values()):
                st.info("No implementation plan available. Please regenerate insights.")
    
    # Reports Tab
    with ir_tab2:
        st.markdown("### PDF Report Generation")
        
        if st.session_state.assessment_results is None:
            st.warning("Please run a quality assessment first to generate a report.")
        else:
            st.markdown("""
            Generate comprehensive PDF reports from your data quality assessment.
            Customize the report format and select which elements to include.
            """)
            
            # Report customization options
            st.subheader("Report Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                report_title = st.text_input(
                    "Report Title",
                    value=f"Data Quality Assessment Report - {st.session_state.file_info.get('file_name', 'Unnamed Dataset')}",
                    help="Enter a title for your report"
                )
                
                organization = st.text_input(
                    "Organization Name",
                    value="",
                    help="Enter your organization name (optional)"
                )
            
            with col2:
                report_type = st.selectbox(
                    "Report Template",
                    options=["Standard", "Executive Summary", "Technical Detail"],
                    index=0,
                    help="Select the type of report to generate"
                )
                
                include_recommendations = st.checkbox(
                    "Include Recommendations",
                    value=True,
                    help="Include actionable recommendations in the report"
                )
            
            # Advanced options in expander
            with st.expander("Advanced Options"):
                col1, col2 = st.columns(2)
                
                with col1:
                    include_visualizations = st.checkbox("Include Visualizations", value=True)
                    include_metadata = st.checkbox("Include Metadata", value=True)
                    include_summary = st.checkbox("Include Executive Summary", value=True)
                
                with col2:
                    include_raw_data = st.checkbox("Include Raw Data Samples", value=False)
                    include_timestamp = st.checkbox("Include Generation Timestamp", value=True)
                    color_coding = st.checkbox("Use Color Coding for Metrics", value=True)
            
            # Generate report button
            report_col1, report_col2 = st.columns([2, 1])
            
            with report_col1:
                generate_button = st.button(
                    "Generate Report",
                    use_container_width=True,
                    type="primary",
                    help="Click to generate a PDF report with the selected options"
                )
            
            # Container for the animation
            animation_container = st.empty()
            
            # Generate the report when button is clicked
            if generate_button:
                # Show animation during processing
                with animation_container.container():
                    loading_card(
                        "Generating PDF Report", 
                        "Creating comprehensive report with visualizations and insights...",
                        animation_type="pulse"
                    )
                
                try:
                    # Check if assessment results are available
                    if st.session_state.assessment_results is None:
                        animation_container.empty()
                        st.warning("No assessment results available. Run a quality assessment first.")
                        st.stop()
                        
                    # Progress callback for report generation
                    def progress_callback(progress, message):
                        # Update the loading card with progress
                        with animation_container.container():
                            processing_animation_with_stages(
                                stage=int(progress * 5) + 1, 
                                total_stages=5,
                                stage_name=f"Report Generation ({int(progress * 100)}%)",
                                description=message
                            )
                    
                    # Call the report generation function with error handling
                    try:
                        report_path = generate_pdf_report(
                            assessment_results=st.session_state.assessment_results,
                            file_info=st.session_state.file_info,
                            report_title=report_title,
                            organization=organization,
                            progress_callback=progress_callback
                        )
                        
                        # Clear the animation
                        animation_container.empty()
                        
                        # Success message with download button
                        st.success("Report generated successfully!")
                        
                        # Create a preview of the report
                        st.session_state.report_preview = {
                            "title": report_title,
                            "organization": organization,
                            "timestamp": datetime.now().strftime('%B %d, %Y %H:%M')
                        }
                        
                        # Read the PDF file as bytes
                        with open(report_path, "rb") as file:
                            pdf_bytes = file.read()
                        
                        # Download button for the report
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"data_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    except ValueError as ve:
                        # Handle value conversion errors
                        if "invalid literal for int()" in str(ve):
                            animation_container.empty()
                            st.error("Error: Invalid number format in the data. This is often caused by empty values or non-numeric characters.")
                            st.info("Please ensure assessment has been run and data contains valid numeric values.")
                        else:
                            animation_container.empty()
                            st.error(f"Value error generating report: {str(ve)}")
                    except Exception as e:
                        # General error handling
                        animation_container.empty()
                        st.error(f"Error generating report: {str(e)}")
                        
                        # Debug output if enabled
                        if st.session_state.get("debug_mode", False):
                            import traceback
                            st.code(traceback.format_exc(), language="python")
                
                except Exception as e:
                    # Clear the animation and show error
                    animation_container.empty()
                    st.error(f"Error in report generation interface: {str(e)}")
            
            # Report preview section
            st.subheader("Report Preview")
            
            if "report_preview" not in st.session_state:
                st.info("Click 'Generate Report' to create and preview your PDF report.")
            else:
                # Display a preview of the report here
                st.markdown("Preview of the report structure:")
                
                # Example preview as expandable sections
                with st.expander("Cover Page", expanded=True):
                    st.markdown(f"**Title:** {report_title}")
                    st.markdown(f"**Organization:** {organization if organization else 'N/A'}")
                    st.markdown(f"**Generation Date:** {datetime.now().strftime('%B %d, %Y')}")
                
                with st.expander("Executive Summary"):
                    st.markdown("Summary of overall data quality scores and key findings.")
                
                with st.expander("Quality Dimension Details"):
                    st.markdown("Detailed analysis of each quality dimension with visualizations.")
                
                with st.expander("Recommendations"):
                    if include_recommendations:
                        st.markdown("Actionable recommendations for improving data quality.")
                    else:
                        st.markdown("Recommendations not included in this report.")
                        
                with st.expander("Appendix"):
                    st.markdown("Additional details and technical information.")
            
            # Tips and information
            st.info("""
            **Tip:** For executive stakeholders, choose the 'Executive Summary' template which focuses 
            on business impact and high-level findings rather than technical details.
            """)
    
    # Historical Trends Tab
    with ir_tab3:
        st.markdown("### Historical Data Quality Trends")
        
        if not st.session_state.get("historical_assessments", []):
            st.info("No historical data available yet. Run at least one quality assessment to see trends over time.")
        else:
            # Display historical data
            st.markdown(f"### Quality Assessment History ({len(st.session_state.historical_assessments)} assessments)")
            
            # Create a dataframe of historical assessments
            history_data = []
            
            for assessment in st.session_state.historical_assessments:
                entry = {
                    "Timestamp": assessment["timestamp"],
                    "File Name": assessment["file_name"],
                    "File Type": assessment["file_type"],
                    "Sheet Count": assessment["sheet_count"],
                    "Total Rows": assessment["total_rows"],
                    "Total Columns": assessment["total_columns"]
                }
                
                # Add quality scores if available
                if assessment.get("summary", {}) and "overall_scores" in assessment["summary"]:
                    scores = assessment["summary"]["overall_scores"]
                    
                    if "overall" in scores:
                        entry["Overall Score"] = scores["overall"]
                    
                    for dimension in ["completeness", "consistency", "accuracy", "uniqueness", "timeliness", "validity"]:
                        if dimension in scores:
                            entry[dimension.capitalize()] = scores[dimension]
                
                history_data.append(entry)
            
            history_df = pd.DataFrame(history_data)
            
            # Convert timestamp to datetime
            history_df["Timestamp"] = pd.to_datetime(history_df["Timestamp"])
            
            # Sort by timestamp (most recent first)
            history_df = history_df.sort_values("Timestamp", ascending=False)
            
            # Display recent assessments
            st.markdown("#### Recent Assessments")
            st.dataframe(history_df, use_container_width=True)
            
            # Create tabs for different trend views
            trend_tab1, trend_tab2 = st.tabs(["Quality Score Trends", "Data Volume Trends"])
            
            with trend_tab1:
                st.markdown("### Quality Score Trends Over Time")
                
                # Check if we have at least 2 assessments with scores
                if "Overall Score" in history_df.columns and len(history_df) >= 2:
                    # Reverse order for charts (oldest first)
                    chart_df = history_df.sort_values("Timestamp")
                    
                    # Create score trend chart
                    score_cols = [col for col in chart_df.columns if col.endswith("Score") or col in ["Completeness", "Consistency", "Accuracy", "Uniqueness", "Timeliness", "Validity"]]
                    
                    if score_cols:
                        # Overall score trend
                        fig_dims = px.line(
                            chart_df,
                            x="Timestamp",
                            y=score_cols,
                            markers=True,
                            title="Quality Scores Over Time",
                            color_discrete_sequence=px.colors.qualitative.Bold
                        )
                        
                        fig_dims.update_layout(
                            xaxis_title="Date",
                            yaxis_title="Score",
                            yaxis=dict(range=[0, 100]),
                            height=400,
                            legend_title="Dimension"
                        )
                        
                        st.plotly_chart(fig_dims, use_container_width=True)
                    else:
                        st.info("No quality scores available in the historical data.")
                else:
                    st.info("Not enough historical data with quality scores to show trends.")
            
            with trend_tab2:
                st.markdown("### Data Volume Trends")
                
                # Check if we have at least 2 assessments
                if len(history_df) >= 2:
                    # Reverse order for charts (oldest first)
                    chart_df = history_df.sort_values("Timestamp")
                    
                    # Create row count trend chart
                    fig_rows = px.line(
                        chart_df,
                        x="Timestamp",
                        y="Total Rows",
                        markers=True,
                        title="Total Rows Over Time",
                        color_discrete_sequence=[ARCADIS_ORANGE]
                    )
                    
                    fig_rows.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Row Count",
                        height=350
                    )
                    
                    st.plotly_chart(fig_rows, use_container_width=True)
                    
                    # Create column count trend chart
                    fig_cols = px.line(
                        chart_df,
                        x="Timestamp",
                        y="Total Columns",
                        markers=True,
                        title="Total Columns Over Time",
                        color_discrete_sequence=["#20c997"]
                    )
                    
                    fig_cols.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Column Count",
                        height=350
                    )
                    
                    st.plotly_chart(fig_cols, use_container_width=True)
                else:
                    st.info("Not enough historical data to show trends. Run at least two assessments.")
                
            # Insight section
            st.markdown("### Trend Insights")
            
            if len(history_df) >= 3 and "Overall Score" in history_df.columns:
                try:
                    # Calculate score changes
                    history_df["Previous Score"] = history_df["Overall Score"].shift(-1)
                    history_df["Score Change"] = history_df["Overall Score"] - history_df["Previous Score"]
                    
                    # Find biggest improvements and declines
                    improvements = history_df[history_df["Score Change"] > 0].sort_values("Score Change", ascending=False)
                    declines = history_df[history_df["Score Change"] < 0].sort_values("Score Change")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Biggest Improvements")
                        if not improvements.empty:
                            for _, row in improvements.head(3).iterrows():
                                date = row["Timestamp"].strftime("%b %d, %Y")
                                file = row["File Name"]
                                change = row["Score Change"]
                                st.markdown(f"- **{date}** (+{change:.1f} points): {file}")
                        else:
                            st.info("No improvements detected in the historical data.")
                    
                    with col2:
                        st.markdown("#### Biggest Declines")
                        if not declines.empty:
                            for _, row in declines.head(3).iterrows():
                                date = row["Timestamp"].strftime("%b %d, %Y")
                                file = row["File Name"]
                                change = row["Score Change"]
                                st.markdown(f"- **{date}** ({change:.1f} points): {file}")
                        else:
                            st.info("No declines detected in the historical data.")
                    
                except Exception as e:
                    st.error(f"Error analyzing trends: {str(e)}")
            else:
                st.info("Not enough historical data to generate trend insights. Run at least three assessments to see meaningful trends.")

elif st.session_state.current_view == "Report Generation":
    st.subheader("Report Generation")
    
    if st.session_state.data_dict is None:
        st.warning("Please upload a data file first.")
    elif st.session_state.assessment_results is None:
        st.warning("Please run a quality assessment before generating a report.")
    else:
        # Report generation options
        st.markdown("### Report Configuration")
        
        # Report title and organization
        col1, col2 = st.columns(2)
        
        with col1:
            report_title = st.text_input(
                "Report Title",
                value=f"Data Quality Assessment Report - {st.session_state.file_info.get('file_name', 'Unknown')}"
            )
        
        with col2:
            organization = st.text_input("Organization Name", value="")
        
        # Report template selection
        st.markdown("### Report Template")
        
        template_options = {
            "standard": "Standard Report (All Details)",
            "executive": "Executive Summary (High-level Overview)",
            "technical": "Technical Report (Detailed Analysis)",
        }
        
        # Use a row of buttons for template selection
        col1, col2, col3 = st.columns(3)
        
        with col1:
            standard_selected = st.session_state.report_template == "standard"
            if st.button(
                "Standard Report",
                use_container_width=True,
                type="secondary" if not standard_selected else "primary",
                help="Comprehensive report with all quality dimensions"
            ):
                st.session_state.report_template = "standard"
        
        with col2:
            executive_selected = st.session_state.report_template == "executive"
            if st.button(
                "Executive Summary",
                use_container_width=True,
                type="secondary" if not executive_selected else "primary",
                help="High-level overview for executives and managers"
            ):
                st.session_state.report_template = "executive"
        
        with col3:
            technical_selected = st.session_state.report_template == "technical"
            if st.button(
                "Technical Report",
                use_container_width=True,
                type="secondary" if not technical_selected else "primary",
                help="Detailed technical analysis for data engineers"
            ):
                st.session_state.report_template = "technical"
        
        # Template preview
        st.markdown(f"Selected template: **{template_options[st.session_state.report_template]}**")
        
        # Customization options
        with st.expander("Customize Report Content"):
            # Sections to include
            st.markdown("#### Sections to Include")
            
            # Use columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                include_summary = st.checkbox("Executive Summary", value=True)
                include_completeness = st.checkbox("Completeness Analysis", value=True)
                include_consistency = st.checkbox("Consistency Analysis", value=True)
            
            with col2:
                include_accuracy = st.checkbox("Accuracy Analysis", value=True)
                include_uniqueness = st.checkbox("Uniqueness Analysis", value=True)
                include_advanced = st.checkbox("Advanced Analytics", value=True)
            
            # Visualization options
            st.markdown("#### Visualization Options")
            chart_style = st.selectbox(
                "Chart Style",
                ["Professional", "Colorful", "Monochrome"],
                index=0
            )
            
            include_radar = st.checkbox("Include Radar Charts", value=True)
            include_3d = st.checkbox("Include 3D Visualizations", value=False)
        
        # Generate report
        if st.button("Generate PDF Report", use_container_width=True):
            # Prepare report options
            report_options = {
                "template": st.session_state.report_template,
                "include_summary": include_summary,
                "include_completeness": include_completeness,
                "include_consistency": include_consistency,
                "include_accuracy": include_accuracy,
                "include_uniqueness": include_uniqueness,
                "include_advanced": include_advanced,
                "chart_style": chart_style,
                "include_radar": include_radar,
                "include_3d": include_3d
            }
            
            # Container for animation
            animation_container = st.empty()
            
            # Define report generation stages
            total_stages = 5  # Planning, Content Creation, Visualization, Assembly, Finalization
            
            # Perform report generation
            try:
                # Stage 1: Report Planning
                with animation_container.container():
                    processing_animation_with_stages(
                        stage=1,
                        total_stages=total_stages,
                        stage_name="Report Planning",
                        description="Preparing template and configuration options..."
                    )
                
                time.sleep(0.5)  # Small pause for visual feedback
                
                # Create progress callback for updating stages
                def progress_callback(progress, message):
                    # Map progress to appropriate stage
                    if progress < 0.2:
                        current_stage = 2  # Content Creation
                        stage_desc = "Generating report content and narratives..."
                    elif progress < 0.5:
                        current_stage = 3  # Visualization
                        stage_desc = "Creating data visualizations and charts..."
                    elif progress < 0.8:
                        current_stage = 4  # Assembly
                        stage_desc = "Assembling report components and formatting..."
                    else:
                        current_stage = 5  # Finalization
                        stage_desc = "Finalizing PDF document..."
                    
                    # Update animation
                    with animation_container.container():
                        processing_animation_with_stages(
                            stage=current_stage,
                            total_stages=total_stages,
                            stage_name=f"Stage {current_stage}: {stage_desc.split('...')[0]}",
                            description=message or stage_desc
                        )
                
                # Generate PDF report with our custom progress callback
                pdf_path = generate_pdf_report(
                    st.session_state.assessment_results,
                    st.session_state.file_info,
                    report_title,
                    organization,
                    progress_callback=progress_callback
                )
                
                # Final animation - Complete
                with animation_container.container():
                    processing_animation_with_stages(
                        stage=total_stages,
                        total_stages=total_stages,
                        stage_name="Report Complete",
                        description="Your PDF report is ready for download!"
                    )
                
                time.sleep(0.8)  # Slightly longer pause at the end
                
                # Clear animation and show success
                animation_container.empty()
                
                # Success message with custom styling
                st.markdown(
                    """
                    <div style="background-color: #f0fff0; border-left: 5px solid #228c22; padding: 15px; border-radius: 4px; margin: 10px 0;">
                        <h3 style="color: #228c22; margin-top: 0;">Report Generated Successfully!</h3>
                        <p>Your PDF report is ready. Click the button below to download it.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Provide download link
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"Data_Quality_Report_{st.session_state.file_info['file_name'].split('.')[0]}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                
                # Clean up temporary file
                os.remove(pdf_path)
                
            except Exception as e:
                # Clear animation container and show error
                animation_container.empty()
                st.error(f"An error occurred during report generation: {str(e)}")
        
        # Report preview with actual content
        st.markdown("### Report Preview")
        
        preview_col1, preview_col2 = st.columns([2, 1])
        
        with preview_col1:
            # Show actual report preview based on selected template
            if st.session_state.report_template == "standard":
                st.markdown(
                    f"""
                    <div style="border: 1px solid #ddd; border-radius: 5px; padding: 20px;">
                        <h2 style="color: rgb(238, 114, 3); text-align: center; margin-bottom: 20px;">DATA QUALITY ASSESSMENT REPORT</h2>
                        <h3 style="color: rgb(238, 114, 3);">Executive Summary</h3>
                        <p>This report provides a comprehensive assessment of the data quality across all DAMA dimensions.</p>
                        <div style="background-color: #f8f9fa; padding: 10px; border-left: 3px solid rgb(238, 114, 3); margin: 15px 0;">
                            <strong>Overall Quality Score:</strong> {st.session_state.assessment_results.get("summary", {}).get("overall_scores", {}).get("overall", 75):.1f}/100
                        </div>
                        <h4>Includes:</h4>
                        <ul>
                            <li>Detailed analysis of all quality dimensions</li>
                            <li>Issue identification with severity ratings</li>
                            <li>Recommended actions prioritized by impact</li>
                            <li>Data visualizations for each dimension</li>
                        </ul>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            elif st.session_state.report_template == "executive":
                st.markdown(
                    f"""
                    <div style="border: 1px solid #ddd; border-radius: 5px; padding: 20px;">
                        <h2 style="color: rgb(238, 114, 3); text-align: center; margin-bottom: 20px;">EXECUTIVE SUMMARY</h2>
                        <p style="font-style: italic;">This executive summary provides key insights from the data quality assessment, focused on business implications.</p>
                        <div style="background-color: #f8f9fa; padding: 10px; border-left: 3px solid rgb(238, 114, 3); margin: 15px 0;">
                            <strong>Overall Assessment:</strong> The data demonstrates 
                            {
                            "excellent quality" if st.session_state.assessment_results.get("summary", {}).get("overall_scores", {}).get("overall", 75) >= 90 else 
                            "good quality with some issues" if st.session_state.assessment_results.get("summary", {}).get("overall_scores", {}).get("overall", 75) >= 75 else 
                            "quality issues that require attention"
                            }
                        </div>
                        <h4>Strategic Recommendations:</h4>
                        <ol>
                            <li>Address critical data gaps in key fields</li>
                            <li>Implement data validation rules at source</li>
                            <li>Establish regular data quality monitoring</li>
                        </ol>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            else:  # technical template
                st.markdown(
                    f"""
                    <div style="border: 1px solid #ddd; border-radius: 5px; padding: 20px;">
                        <h2 style="color: rgb(238, 114, 3); text-align: center; margin-bottom: 20px;">TECHNICAL QUALITY REPORT</h2>
                        <h3 style="color: rgb(238, 114, 3);">Data Quality Metrics</h3>
                        <div style="font-family: monospace; background-color: #f8f9fa; padding: 15px; margin: 10px 0;">
                            <pre>Overall Quality:    {st.session_state.assessment_results.get("summary", {}).get("overall_scores", {}).get("overall", 75):.1f}/100
Completeness:      {st.session_state.assessment_results.get("summary", {}).get("overall_scores", {}).get("completeness", 82):.1f}/100
Consistency:       {st.session_state.assessment_results.get("summary", {}).get("overall_scores", {}).get("consistency", 78):.1f}/100
Accuracy:          {st.session_state.assessment_results.get("summary", {}).get("overall_scores", {}).get("accuracy", 73):.1f}/100
Uniqueness:        {st.session_state.assessment_results.get("summary", {}).get("overall_scores", {}).get("uniqueness", 91):.1f}/100</pre>
                        </div>
                        <h4>Technical Details:</h4>
                        <ul>
                            <li>Comprehensive outlier analysis with statistical thresholds</li>
                            <li>Detailed type consistency validation</li>
                            <li>Pattern-based validity checking</li>
                            <li>Raw data quality metrics with confidence intervals</li>
                        </ul>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        
        with preview_col2:
            st.markdown("#### Report Features")
            
            # Display template-specific features
            if st.session_state.report_template == "standard":
                features = [
                    "Comprehensive assessment",
                    "All quality dimensions",
                    "Detailed visualizations",
                    "Complete recommendations",
                    "Technical appendices"
                ]
            elif st.session_state.report_template == "executive":
                features = [
                    "Business-focused summary",
                    "High-level metrics",
                    "Key issues highlighted",
                    "Strategic recommendations",
                    "Management insights"
                ]
            else:  # technical template
                features = [
                    "In-depth technical analysis",
                    "Statistical details",
                    "Data profiling metrics",
                    "Schema validation results",
                    "Technical appendices"
                ]
            
            for feature in features:
                st.markdown(f"✓ {feature}")
            
            st.markdown("---")
            st.markdown("#### Output Format")
            st.markdown("PDF document with embedded visualizations, searchable text, and bookmarks.")
            
            st.info("The report will be generated with a narrative style that explains the findings in clear language with relevant context and interpretation of results.")

elif st.session_state.current_view == "Historical Trends":
    st.subheader("Historical Data Quality Trends")
    
    if not st.session_state.historical_assessments:
        st.info("No historical data available yet. Run at least one quality assessment to see trends over time.")
    else:
        # Display historical data
        st.markdown(f"### Quality Assessment History ({len(st.session_state.historical_assessments)} assessments)")
        
        # Create a dataframe of historical assessments
        history_data = []
        
        for assessment in st.session_state.historical_assessments:
            entry = {
                "Timestamp": assessment["timestamp"],
                "File Name": assessment["file_name"],
                "File Type": assessment["file_type"],
                "Sheet Count": assessment["sheet_count"],
                "Total Rows": assessment["total_rows"],
                "Total Columns": assessment["total_columns"]
            }
            
            # Add quality scores if available
            if assessment["summary"] and "overall_scores" in assessment["summary"]:
                scores = assessment["summary"]["overall_scores"]
                
                if "overall" in scores:
                    entry["Overall Score"] = scores["overall"]
                
                for dimension in ["completeness", "consistency", "accuracy", "uniqueness", "timeliness", "validity"]:
                    if dimension in scores:
                        entry[dimension.capitalize()] = scores[dimension]
            
            history_data.append(entry)
        
        history_df = pd.DataFrame(history_data)
        
        # Convert timestamp to datetime
        history_df["Timestamp"] = pd.to_datetime(history_df["Timestamp"])
        
        # Sort by timestamp (most recent first)
        history_df = history_df.sort_values("Timestamp", ascending=False)
        
        # Display recent assessments
        st.markdown("#### Recent Assessments")
        st.dataframe(history_df, use_container_width=True)
        
        # Create tabs for different trend views
        trend_tab1, trend_tab2 = st.tabs(["Quality Score Trends", "Data Volume Trends"])
        
        with trend_tab1:
            st.markdown("### Quality Score Trends Over Time")
            
            # Check if we have at least 2 assessments with scores
            if "Overall Score" in history_df.columns and len(history_df) >= 2:
                # Reverse order for charts (oldest first)
                chart_df = history_df.sort_values("Timestamp")
                
                # Create score trend chart
                score_cols = [col for col in chart_df.columns if col.endswith("Score") or col in ["Completeness", "Consistency", "Accuracy", "Uniqueness", "Timeliness", "Validity"]]
                
                if score_cols:
                    # Overall score trend
                    if "Overall Score" in score_cols:
                        fig_overall = px.line(
                            chart_df,
                            x="Timestamp",
                            y="Overall Score",
                            markers=True,
                            title="Overall Quality Score Trend",
                            color_discrete_sequence=[ARCADIS_BLUE]
                        )
                        
                        fig_overall.update_layout(
                            xaxis_title="Date",
                            yaxis_title="Score",
                            yaxis=dict(range=[0, 100]),
                            height=400
                        )
                        
                        st.plotly_chart(fig_overall, use_container_width=True)
                    
                    # Dimension scores trend
                    dimension_cols = [col for col in score_cols if col != "Overall Score"]
                    
                    if dimension_cols:
                        fig_dims = px.line(
                            chart_df,
                            x="Timestamp",
                            y=dimension_cols,
                            markers=True,
                            title="Quality Dimension Scores Over Time",
                            color_discrete_sequence=px.colors.qualitative.Bold
                        )
                        
                        fig_dims.update_layout(
                            xaxis_title="Date",
                            yaxis_title="Score",
                            yaxis=dict(range=[0, 100]),
                            height=400,
                            legend_title="Dimension"
                        )
                        
                        st.plotly_chart(fig_dims, use_container_width=True)
                else:
                    st.info("No quality scores available in the historical data.")
            else:
                st.info("Not enough historical data with quality scores to show trends.")
        
        with trend_tab2:
            st.markdown("### Data Volume Trends")
            
            # Check if we have at least 2 assessments
            if len(history_df) >= 2:
                # Reverse order for charts (oldest first)
                chart_df = history_df.sort_values("Timestamp")
                
                # Create row count trend chart
                fig_rows = px.line(
                    chart_df,
                    x="Timestamp",
                    y="Total Rows",
                    markers=True,
                    title="Total Rows Over Time",
                    color_discrete_sequence=[ARCADIS_ORANGE]
                )
                
                fig_rows.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Row Count",
                    height=350
                )
                
                st.plotly_chart(fig_rows, use_container_width=True)
                
                # Create column count trend chart
                fig_cols = px.line(
                    chart_df,
                    x="Timestamp",
                    y="Total Columns",
                    markers=True,
                    title="Total Columns Over Time",
                    color_discrete_sequence=[ARCADIS_GREEN]
                )
                
                fig_cols.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Column Count",
                    height=350
                )
                
                st.plotly_chart(fig_cols, use_container_width=True)
            else:
                st.info("Not enough historical data to show trends.")

elif st.session_state.current_view == "Settings":
    st.subheader("Application Settings")
    
    settings_tab1, settings_tab2, settings_tab3 = st.tabs(["AI Configuration", "Application", "Advanced"])
    
    with settings_tab1:
        st.markdown("### AI-Powered Features Configuration")
        
        ai_col1, ai_col2 = st.columns([1, 1])
        
        with ai_col1:
            st.markdown("#### General AI Settings")
            
            # Enable/disable AI features
            ai_enabled = st.toggle(
                "Enable AI Features",
                value=st.session_state.get("ai_enabled", False),
                help="Toggle AI-powered features on or off"
            )
            
            if ai_enabled != st.session_state.get("ai_enabled", False):
                st.session_state.ai_enabled = ai_enabled
            
            if st.session_state.get("ai_enabled", False):
                # AI Provider selection
                provider_option = st.radio(
                    "Select AI Provider",
                    options=["OpenAI API", "Anthropic API", "xAI API", "Local LLM"],
                    index=0 if st.session_state.get("ai_provider", "openai") == "openai" else 
                          1 if st.session_state.get("ai_provider", "openai") == "anthropic" else
                          2 if st.session_state.get("ai_provider", "openai") == "xai" else 3,
                    help="Choose between cloud AI APIs or a local LLM model"
                )
                
                # Update provider in session state
                new_provider = "openai" if provider_option == "OpenAI API" else \
                              "anthropic" if provider_option == "Anthropic API" else \
                              "xai" if provider_option == "xAI API" else "local"
                
                if new_provider != st.session_state.get("ai_provider", "openai"):
                    st.session_state.ai_provider = new_provider
            
        with ai_col2:
            st.markdown("#### Provider Settings")
            
            if st.session_state.get("ai_enabled", False):
                if st.session_state.get("ai_provider", "openai") == "openai":
                    # OpenAI settings
                    st.markdown("**OpenAI API Configuration**")
                    
                    # API Key input
                    api_key = st.text_input(
                        "OpenAI API Key",
                        value=st.session_state.get("openai_api_key", ""),
                        type="password",
                        help="Your OpenAI API key for GPT models"
                    )
                    
                    if api_key != st.session_state.get("openai_api_key", ""):
                        st.session_state.openai_api_key = api_key
                        if api_key:
                            os.environ["OPENAI_API_KEY"] = api_key
                    
                    # Model selection
                    model = st.selectbox(
                        "Model",
                        options=["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
                        index=0,
                        help="Select OpenAI model to use"
                    )
                    
                    if model != st.session_state.get("openai_model", "gpt-4o"):
                        st.session_state.openai_model = model
                    
                    # Show token usage if available
                    if st.session_state.get("token_usage", {"total": 0})["total"] > 0:
                        st.info(f"Total token usage: {st.session_state.get('token_usage', {'total': 0})['total']:,} tokens")
                        
                        if st.button("Clear Token Usage History"):
                            st.session_state.token_usage = {"total": 0, "history": []}
                            st.success("Token usage history cleared!")
                    else:
                        st.info("No tokens used yet.")
                
                elif st.session_state.get("ai_provider", "openai") == "anthropic":
                    # Anthropic settings
                    st.markdown("**Anthropic API Configuration**")
                    
                    # API Key input
                    api_key = st.text_input(
                        "Anthropic API Key",
                        value=st.session_state.get("anthropic_api_key", ""),
                        type="password",
                        help="Your Anthropic API key for Claude models"
                    )
                    
                    if api_key != st.session_state.get("anthropic_api_key", ""):
                        st.session_state.anthropic_api_key = api_key
                        if api_key:
                            os.environ["ANTHROPIC_API_KEY"] = api_key
                    
                    # Model selection
                    model = st.selectbox(
                        "Model",
                        options=["claude-3-5-sonnet-20241022", "claude-3-opus", "claude-3-sonnet"],
                        index=0,
                        help="Select Anthropic Claude model to use"
                    )
                    
                    if model != st.session_state.get("anthropic_model", "claude-3-5-sonnet-20241022"):
                        st.session_state.anthropic_model = model
                    
                    st.info("The newest Anthropic model is claude-3-5-sonnet-20241022 which was released October 22, 2024")
                
                elif st.session_state.get("ai_provider", "openai") == "xai":
                    # xAI settings
                    st.markdown("**xAI API Configuration**")
                    
                    # API Key input
                    api_key = st.text_input(
                        "xAI API Key",
                        value=st.session_state.get("xai_api_key", ""),
                        type="password",
                        help="Your xAI API key for Grok models"
                    )
                    
                    if api_key != st.session_state.get("xai_api_key", ""):
                        st.session_state.xai_api_key = api_key
                        if api_key:
                            os.environ["XAI_API_KEY"] = api_key
                    
                    # Model selection
                    model = st.selectbox(
                        "Model",
                        options=["grok-2-1212", "grok-1"],
                        index=0,
                        help="Select xAI Grok model to use"
                    )
                    
                    if model != st.session_state.get("xai_model", "grok-2-1212"):
                        st.session_state.xai_model = model
                
                else:
                    # Local LLM settings
                    st.markdown("**Local LLM Configuration**")
                    
                    # Model path input
                    model_path = st.text_input(
                        "Model Path",
                        value=st.session_state.get("local_model_path", ""),
                        help="Path to your local LLM model"
                    )
                    
                    if model_path != st.session_state.get("local_model_path", ""):
                        st.session_state.local_model_path = model_path
                    
                    # Show placeholder message
                    st.markdown("""
                    ### Local LLM Configuration
                    
                    The Local LLM implementation uses scikit-learn and TF-IDF for basic text processing.
                    This lightweight solution works offline without requiring external API keys.
                    
                    **Features:**
                    - Template-based responses for data quality explanations
                    - Simple text similarity matching
                    - Context-aware recommendations
                    - Consistent response format
                    """)
                    
                    # Model path input
                    model_path = st.text_input(
                        "Custom Model Path (Optional)",
                        value=st.session_state.local_model_path,
                        placeholder="/path/to/model",
                        help="Leave empty to use the built-in templates"
                    )
                    
                    if model_path != st.session_state.local_model_path:
                        st.session_state.local_model_path = model_path
                    
                    # Additional controls
                    if st.button("Test Local LLM"):
                        with st.spinner("Testing local LLM..."):
                            try:
                                # Create a local LLM provider instance
                                from ai_integration import LocalLLMProvider
                                local_llm = LocalLLMProvider(model_path=model_path)
                                
                                # Generate a test response
                                test_response = local_llm.generate_response(
                                    "What are the key data quality issues?",
                                    "explain_issues", 
                                    {"summary": {"overall_scores": {"overall": 75, "completeness": 65, "accuracy": 80}}}
                                )
                                
                                # Show the response
                                st.success("Local LLM is working correctly!")
                                st.info(f"**Sample response:** {test_response}")
                            except Exception as e:
                                st.error(f"Error testing Local LLM: {str(e)}")
            else:
                st.info("Enable AI features to configure provider settings.")
        
        # AI Feature management
        st.markdown("### AI Feature Management")
        
        feature_col1, feature_col2 = st.columns(2)
        
        with feature_col1:
            st.markdown("#### Natural Language Explanations")
            nl_explanations = st.toggle(
                "Enable Natural Language Explanations",
                value=st.session_state.get("nl_explanations_enabled", True),
                help="Generate human-readable explanations of technical findings",
                disabled=not st.session_state.get("ai_enabled", False)
            )
            
            if nl_explanations != st.session_state.get("nl_explanations_enabled", True):
                st.session_state.nl_explanations_enabled = nl_explanations
            
            st.markdown("#### Auto-Documentation")
            auto_docs = st.toggle(
                "Enable Auto-Documentation",
                value=st.session_state.get("auto_docs_enabled", True),
                help="Automatically generate comprehensive documentation",
                disabled=not st.session_state.get("ai_enabled", False)
            )
            
            if auto_docs != st.session_state.get("auto_docs_enabled", True):
                st.session_state.auto_docs_enabled = auto_docs
        
        with feature_col2:
            st.markdown("#### Issue Severity Classification")
            issue_class = st.toggle(
                "Enable Issue Classification",
                value=st.session_state.get("issue_classification_enabled", True),
                help="Automatically classify and prioritize issues",
                disabled=not st.session_state.get("ai_enabled", False)
            )
            
            if issue_class != st.session_state.get("issue_classification_enabled", True):
                st.session_state.issue_classification_enabled = issue_class
            
            st.markdown("#### Conversational Interface")
            conv_interface = st.toggle(
                "Enable Chat Interface",
                value=st.session_state.get("chat_interface_enabled", True),
                help="Enable the conversational interface for data queries",
                disabled=not st.session_state.get("ai_enabled", False)
            )
            
            if conv_interface != st.session_state.get("chat_interface_enabled", True):
                st.session_state.chat_interface_enabled = conv_interface
        
        if not st.session_state.get("ai_enabled", False):
            st.warning("AI features are currently disabled. Enable them above to access these capabilities.")
        
        # AI explanation cache management
        st.markdown("### Cache Management")
        
        cache_col1, cache_col2 = st.columns(2)
        
        with cache_col1:
            cache_size = len(st.session_state.get("ai_explanation_cache", {}))
            st.info(f"AI explanation cache size: {cache_size} items")
        
        with cache_col2:
            if st.button("Clear AI Cache") and cache_size > 0:
                st.session_state.ai_explanation_cache = {}
                st.success("AI explanation cache cleared!")
    
    with settings_tab2:
        st.markdown("### Application Settings")
        
        app_col1, app_col2 = st.columns(2)
        
        with app_col1:
            st.markdown("#### UI Customization")
            
            # Theme selection
            theme = st.selectbox(
                "Color Theme",
                options=["Arcadis Theme", "Blue Theme", "Green Theme", "Neutral Theme"],
                index=0
            )
            
            # This would actually change the theme in a real implementation
            st.info("Theme selection is a placeholder. Currently using Arcadis Theme.")
            
            # Font size
            font_size = st.select_slider(
                "Font Size",
                options=["Small", "Medium", "Large"],
                value="Medium"
            )
            
            st.info(f"Font size setting ({font_size}) is a placeholder.")
        
        with app_col2:
            st.markdown("#### Default Settings")
            
            # Report template selection
            report_template = st.selectbox(
                "Default Report Template",
                options=["Standard", "Executive Summary", "Technical", "Comprehensive"],
                index=0 if st.session_state.get("report_template", "standard") == "standard" else 
                      1 if st.session_state.get("report_template", "standard") == "executive" else
                      2 if st.session_state.get("report_template", "standard") == "technical" else 3
            )
            
            # Update report template in session state
            new_template = report_template.lower()
            if new_template == "executive summary":
                new_template = "executive"
            elif new_template == "comprehensive":
                new_template = "comprehensive"
                
            if new_template != st.session_state.get("report_template", "standard"):
                st.session_state.report_template = new_template
            
            # Visualization style
            viz_style = st.selectbox(
                "Visualization Style",
                options=["Standard", "Minimal", "Detailed"],
                index=0
            )
            
            st.info(f"Visualization style setting ({viz_style}) is a placeholder.")
    
    with settings_tab3:
        st.markdown("### Advanced Settings")
        
        adv_col1, adv_col2 = st.columns(2)
        
        with adv_col1:
            st.markdown("#### Data Processing")
            
            # Sample size settings
            max_rows = st.number_input(
                "Maximum Rows for Analysis",
                min_value=1000,
                max_value=1000000,
                value=st.session_state.get("max_rows", 100000),
                step=10000,
                help="Maximum number of rows to process in analytics functions"
            )
            
            if max_rows != st.session_state.get("max_rows", 100000):
                st.session_state.max_rows = max_rows
                
            # Cache settings
            cache_ttl = st.slider(
                "Cache Time-to-Live (minutes)",
                min_value=5,
                max_value=120,
                value=st.session_state.get("cache_ttl", 30),
                step=5,
                help="How long to keep cached results before recalculating"
            )
            
            if cache_ttl != st.session_state.get("cache_ttl", 30):
                st.session_state.cache_ttl = cache_ttl
        
        with adv_col2:
            st.markdown("#### Performance")
            
            # Parallel processing
            parallel = st.checkbox(
                "Enable Parallel Processing",
                value=st.session_state.get("parallel_processing", True),
                help="Use multiple CPU cores for faster processing when available"
            )
            
            if parallel != st.session_state.get("parallel_processing", True):
                st.session_state.parallel_processing = parallel
            
            # Debug mode
            debug_mode = st.checkbox(
                "Debug Mode",
                value=st.session_state.get("debug_mode", False),
                help="Show additional diagnostic information"
            )
            
            if debug_mode != st.session_state.get("debug_mode", False):
                st.session_state.debug_mode = debug_mode
                
            if st.session_state.get("debug_mode", False):
                st.info("Debug mode enabled. Additional diagnostic information will be shown.")
            
        # Advanced import/export
        st.markdown("#### Import/Export Settings")
        
        ie_col1, ie_col2 = st.columns(2)
        
        with ie_col1:
            settings_dict = {
                "ai_enabled": st.session_state.get("ai_enabled", False),
                "ai_provider": st.session_state.get("ai_provider", "openai"),
                "report_template": st.session_state.get("report_template", "standard"),
                "max_rows": st.session_state.get("max_rows", 100000),
                "cache_ttl": st.session_state.get("cache_ttl", 30),
                "parallel_processing": st.session_state.get("parallel_processing", True),
                "debug_mode": st.session_state.get("debug_mode", False),
                "nl_explanations_enabled": st.session_state.get("nl_explanations_enabled", True),
                "auto_docs_enabled": st.session_state.get("auto_docs_enabled", True),
                "issue_classification_enabled": st.session_state.get("issue_classification_enabled", True),
                "chat_interface_enabled": st.session_state.get("chat_interface_enabled", True)
            }
            
            st.download_button(
                label="Export Settings",
                data=json.dumps(settings_dict, indent=2),
                file_name="data_quality_settings.json",
                mime="application/json"
            )
        
        with ie_col2:
            uploaded_settings = st.file_uploader(
                "Import Settings",
                type=["json"],
                help="Upload a settings file to restore configuration"
            )
            
            if uploaded_settings is not None:
                try:
                    settings_data = json.loads(uploaded_settings.getvalue().decode())
                    
                    # Update session state with uploaded settings
                    for key, value in settings_data.items():
                        if key in settings_dict:
                            st.session_state[key] = value
                    
                    st.success("Settings imported successfully!")
                except Exception as e:
                    st.error(f"Error importing settings: {str(e)}")
                    
    st.markdown("---")
    
    # About section
    st.markdown("### About")
    st.markdown("""
    **Data Quality Assessment Tool v1.0**  
    A comprehensive data quality assessment platform powered by advanced analytics and AI.
    
    Developed by Arcadis Digital Solutions
    """)
    
    # System information
    with st.expander("System Information", expanded=False):
        st.markdown(f"""
        - **Python Version**: {os.environ.get('PYTHON_VERSION', '3.11')}
        - **Operating System**: {os.environ.get('OS_TYPE', 'Linux')}
        - **Application Started**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        """)
                
# Footer
st.markdown("---")
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>Data Quality Assessment Tool</div>
        <div style="text-align: right; font-size: 0.8em; color: #666;">
            Based on DAMA principles | Version 2.0
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

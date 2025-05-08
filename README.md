# DAMA Quality: Data Quality Assessment Tool

A comprehensive data quality assessment tool based on DAMA principles, providing advanced analytics, visualization, and reporting for structured data files.

## Features

- **Multi-format Data Source Processing**: Analyze Excel (multiple sheets) and CSV files
- **Comprehensive Quality Assessment**: Following DAMA principles (Completeness, Consistency, Accuracy, Uniqueness, Timeliness, Validity)
- **Advanced Analytics**: Machine learning-powered anomaly detection and clustering
- **Interactive Visualizations**: Using Plotly for detailed data exploration
- **Customizable PDF Reports**: Professional reports with actionable insights
- **Attribute-Level Assessment**: Focus analysis on specific columns/attributes
- **Attribute Combination Analysis**: Discover relationships between columns

## Setup Instructions

### Prerequisites

- Python 3.8 or newer
- pip (Python package manager)

### Installation

1. Clone this repository to your local machine
   ```
   git clone [repository-url]
   cd [repository-directory]
   ```

2. Create a virtual environment
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

4. Install the required packages
   ```
   pip install -r requirements.txt
   ```
   
   Required packages:
   ```
   streamlit>=1.30.0
   pandas>=2.0.0
   numpy>=1.24.0
   plotly>=5.15.0
   fpdf>=1.7.2
   scikit-learn>=1.2.0
   openpyxl>=3.1.0
   scipy>=1.10.0
   statsmodels>=0.14.0
   streamlit-extras>=0.3.0
   streamlit-metrics>=0.1.0
   streamlit-option-menu>=0.3.2
   ```

### Running the Application

1. Run the Streamlit application
   ```
   streamlit run app.py
   ```

2. The application will open in your default web browser at http://localhost:8501

## Usage Guide

1. **Upload Data**: Use the Data Manager to upload an Excel or CSV file
2. **Configure Assessment**: In the Quality Assessment tab, select which dimensions to analyze
3. **Advanced Options**: 
   - Click the "Advanced Options" expander
   - Use the "Attribute Selection" tab to focus on specific columns
   - Use the "Attribute Combinations" tab to analyze relationships between columns
4. **Run Assessment**: Click the "Run Quality Assessment" button
5. **View Results**: Explore the comprehensive analysis in the Results tab
6. **Generate Reports**: Download PDF reports for sharing with stakeholders

## AI Integration (Optional)

The tool supports integration with:
- OpenAI API (requires API key)
- Anthropic API (requires API key)
- xAI/Grok API (requires API key)

If API keys are available, AI-powered explanations and recommendations will be enabled.

## Structure

- `app.py`: Main application file
- `data_quality.py`: Core assessment functionality
- `report_generator.py`: PDF report generation
- `actionable_insights.py`: Generates prioritized recommendations
- `ai_integration.py`: AI explanations and recommendations
- `animations.py`: UI animations and loading indicators
- `utils.py`: Utility functions

## User Testing

When running user testing, pay attention to:
1. Intuitiveness of the interface
2. Clarity of assessment results
3. Usefulness of visualizations
4. Actionability of recommendations
5. Stability with various data formats and sizes

## License

[Include license information]

## Contact

[Include contact information]
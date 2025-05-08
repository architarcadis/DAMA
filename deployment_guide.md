# DAMA Quality - Local Deployment Guide

This guide provides step-by-step instructions for deploying the DAMA Quality data assessment tool locally for user testing.

## Step 1: Environment Setup

### Option 1: Using Virtual Environment (Recommended)

1. Ensure you have Python 3.8+ installed
   ```bash
   python --version
   ```

2. Clone or download the application code to your local machine

3. Create a virtual environment in the project directory
   ```bash
   python -m venv venv
   ```

4. Activate the virtual environment
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

5. Install the required packages
   ```bash
   pip install streamlit pandas numpy plotly fpdf scikit-learn openpyxl scipy statsmodels streamlit-extras streamlit-metrics streamlit-option-menu
   ```

### Option 2: Using Conda Environment

1. Create a new conda environment
   ```bash
   conda create -n dama-quality python=3.10
   ```

2. Activate the environment
   ```bash
   conda activate dama-quality
   ```

3. Install the required packages
   ```bash
   conda install -c conda-forge streamlit pandas numpy plotly scikit-learn openpyxl scipy statsmodels
   pip install fpdf streamlit-extras streamlit-metrics streamlit-option-menu
   ```

## Step 2: Configuration

1. Create a `.streamlit` directory in the project root if it doesn't exist
   ```bash
   mkdir -p .streamlit
   ```

2. Create a `config.toml` file in the `.streamlit` directory with the following content:
   ```toml
   [server]
   headless = true
   address = "0.0.0.0"
   port = 8501

   [theme]
   primaryColor = "#ee7203"  # Arcadis Orange
   backgroundColor = "#FFFFFF"
   secondaryBackgroundColor = "#F0F2F6"
   textColor = "#262730"
   font = "sans serif"
   ```

## Step 3: Running the Application

1. Start the Streamlit server
   ```bash
   streamlit run app.py
   ```

2. The application will be accessible at `http://localhost:8501`

3. For LAN access (useful for testing on mobile devices):
   - Find your computer's IP address
     - Windows: Run `ipconfig` in Command Prompt
     - macOS/Linux: Run `ifconfig` or `ip addr` in Terminal
   - Access the app from other devices on the same network using `http://YOUR_IP_ADDRESS:8501`

## Step 4: User Testing Setup

1. Prepare test data files:
   - Excel files with multiple sheets
   - CSV files with various data quality issues
   - Files of different sizes (small, medium, large)

2. Prepare user testing tasks:
   - Basic data upload and assessment
   - Advanced options configuration
   - Attribute combination analysis
   - Report generation and customization
   - AI integration testing (if API keys are available)

3. Create a feedback form for testers to document:
   - Usability issues
   - Feature requests
   - Performance observations
   - UI/UX improvement suggestions

## Step 5: Troubleshooting Common Issues

### Application Won't Start

1. Check Python version is 3.8+
2. Verify all dependencies are installed correctly
3. Check for any error messages in the terminal
4. Ensure the port isn't already in use

### Slow Performance

1. For large files, recommend using CSV format instead of Excel
2. Disable unnecessary quality dimensions for initial quick assessment
3. Focus attribute-level assessment on a subset of columns
4. Consider increasing system memory allocation for Python

### PDF Report Generation Issues

1. Ensure the FPDF library is correctly installed
2. Check for any special characters in dataset that might cause encoding issues
3. Verify write permissions in the target directory for saving PDF files

## Step 6: Securing API Keys (Optional)

If using AI integration features:

1. Create a `.env` file in the project root (do NOT commit this to version control)
2. Add API keys to the `.env` file:
   ```
   OPENAI_API_KEY=your_key_here
   ANTHROPIC_API_KEY=your_key_here
   XAI_API_KEY=your_key_here
   ```
3. Install the python-dotenv package:
   ```bash
   pip install python-dotenv
   ```
4. Modify `ai_integration.py` to load keys from the `.env` file if necessary

## Additional Resources

- Streamlit Documentation: https://docs.streamlit.io/
- Pandas Documentation: https://pandas.pydata.org/docs/
- Plotly Documentation: https://plotly.com/python/
- FPDF Documentation: http://fpdf.org/en/doc/index.php
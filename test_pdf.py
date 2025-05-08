import pandas as pd
import traceback
from utils import load_data, get_data_info
from data_quality import perform_data_quality_assessment
from report_generator import generate_pdf_report

try:
    print("Loading test data...")
    # Use a simple approach to load the CSV file
    df = pd.read_csv('test_data.csv')
    data_dict = {'Sheet1': df}
    sheet_names = ['Sheet1']
    file_type = 'CSV'
    
    print("Getting file info...")
    file_info = get_data_info(data_dict, sheet_names)
    
    print("Performing data quality assessment...")
    assessment_options = {
        "completeness": True,
        "consistency": True,
        "accuracy": True,
        "uniqueness": True,
        "timeliness": True,
        "validity": True
    }
    
    def progress_callback(progress, message):
        print(f"Progress: {progress}%, Status: {message}")
    
    assessment_results = perform_data_quality_assessment(
        data_dict, 
        assessment_options,
        progress_callback=progress_callback
    )
    
    print("Generating PDF report...")
    report_title = "Test Data Quality Report"
    organization = "Test Organization"
    
    pdf_path = generate_pdf_report(
        assessment_results, 
        file_info,
        report_title,
        organization,
        progress_callback=progress_callback
    )
    
    print(f"Report generated successfully: {pdf_path}")
    
except Exception as e:
    print(f"Error: {str(e)}")
    traceback.print_exc()
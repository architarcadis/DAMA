import pandas as pd
import numpy as np
import io
import time
import os

def load_data(uploaded_file):
    """
    Load data from uploaded file (Excel or CSV)
    
    Args:
        uploaded_file: Streamlit uploaded file object
    
    Returns:
        data_dict: Dictionary with sheet names as keys and dataframes as values
        file_type: String indicating file type ('Excel' or 'CSV')
        sheet_names: List of sheet names
    """
    # Determine file type based on extension
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension in ['xlsx', 'xls']:
        # Excel file with multiple sheets
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_names = excel_file.sheet_names
        
        # Load each sheet into a dictionary of dataframes
        data_dict = {}
        for sheet in sheet_names:
            data_dict[sheet] = pd.read_excel(excel_file, sheet_name=sheet)
        
        file_type = 'Excel'
    
    elif file_extension == 'csv':
        # CSV file (single sheet)
        df = pd.read_csv(uploaded_file)
        
        # For consistency, store the single dataframe in a dictionary
        data_dict = {'Sheet1': df}
        sheet_names = ['Sheet1']
        file_type = 'CSV'
    
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    return data_dict, file_type, sheet_names

def get_data_info(data_dict, sheet_names):
    """
    Get general information about the loaded data
    
    Args:
        data_dict: Dictionary with sheet names as keys and dataframes as values
        sheet_names: List of sheet names
    
    Returns:
        Dictionary containing data information
    """
    info = {
        "sheet_names": sheet_names,
        "file_type": "Excel" if len(sheet_names) > 1 else "CSV",
        "total_rows": sum(len(data_dict[sheet]) for sheet in sheet_names),
        "total_columns": sum(len(data_dict[sheet].columns) for sheet in sheet_names),
        "sheets": {}
    }
    
    for sheet in sheet_names:
        df = data_dict[sheet]
        
        info["sheets"][sheet] = {
            "rows": len(df),
            "columns": len(df.columns),
            "memory_usage": df.memory_usage(deep=True).sum() / (1024 * 1024),  # in MB
            "column_dtypes": {col: str(df[col].dtype) for col in df.columns}
        }
    
    return info

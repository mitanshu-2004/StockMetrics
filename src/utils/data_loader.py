import pandas as pd
import os
import numpy as np
from src.constants import COMPANY_STOCK_CODES

def load_data():
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    file_path = os.path.join(base_path, 'data.xls')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    stock_df = pd.read_excel(file_path, sheet_name=0)
    fund_df = pd.read_excel(file_path, sheet_name=1)
    
    if stock_df.empty:
        raise ValueError("Empty stock data")
    
    if fund_df.empty:
        raise ValueError("Empty fundamental data")
    
    required_fields = ['Company name', 'Field']
    missing = [f for f in required_fields if f not in fund_df.columns]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")
    
    return stock_df, fund_df

def clean_stock_data(df):
    df = df.iloc[4:].reset_index(drop=True)
    df.columns = ['Date'] + list(df.columns[1:])
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    for code in COMPANY_STOCK_CODES.keys():
        df[code] = pd.to_numeric(df[code], errors='coerce')
    
    df = df.dropna(subset=['Date'])
    df.set_index('Date', inplace=True)
    
    return df.resample('YE').last().pct_change().dropna()
import pandas as pd
import os
from src.constants import COMPANY_STOCK_CODES

def load_data():
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    file_path = os.path.join(base_path, 'data.xls')
    
    stock_df = pd.read_excel(file_path, sheet_name=0)
    fund_df = pd.read_excel(file_path, sheet_name=1)
    
    return stock_df, fund_df

def clean_stock_data(df):
    df = df.iloc[4:].reset_index(drop=True)
    df.columns = ['Date'] + list(df.columns[1:])
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    stock_codes = list(COMPANY_STOCK_CODES.keys())
    for c in stock_codes:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df = df.dropna(subset=['Date'])
    df.set_index('Date', inplace=True)
    
    yearly_returns = df.resample('YE').last().pct_change().dropna()
    return yearly_returns

import pandas as pd
import os
from src.constants import EXPECTED_STOCK_CODE_TO_NAME


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

    required_fields = ['Company name', 'Field', 'ISIN']
    missing = [f for f in required_fields if f not in fund_df.columns]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")

    code_to_name = derive_code_to_name(stock_df, fund_df)
    return stock_df, fund_df, code_to_name


def derive_code_to_name(stock_df, fund_df):
    """Build the authoritative stock-code -> fundamentals-name mapping by joining
    the two sheets on ISIN, the unambiguous security identifier.

    The original code hard-coded this mapping by hand and got three of five pairs
    wrong (e.g. the TCS stock symbol was labelled "Infosys"), silently pairing each
    company's returns with another company's fundamentals. Deriving it from ISIN
    removes that whole class of error; we then cross-check against the documented
    EXPECTED mapping and fail loudly if the file and our expectation disagree.
    """
    # Sheet1 (stock) is laid out with metadata rows under the symbol header:
    # row "Name", row "ISIN Number", row "Exchng Ticker", ... then dated prices.
    label_col = stock_df.columns[0]
    isin_rows = stock_df[stock_df[label_col].astype(str).str.strip() == 'ISIN Number']
    if isin_rows.empty:
        raise ValueError("Could not locate the 'ISIN Number' row in the stock sheet")
    isin_row = isin_rows.iloc[0]

    codes = [c for c in stock_df.columns if c != label_col]
    code_to_isin = {code: str(isin_row[code]).strip() for code in codes}

    # Sheet2 (fundamentals): ISIN -> Company name.
    fund_isin_to_name = {
        str(isin).strip(): name
        for isin, name in fund_df[['ISIN', 'Company name']].drop_duplicates().values
    }

    code_to_name = {}
    for code, isin in code_to_isin.items():
        if isin not in fund_isin_to_name:
            raise ValueError(
                f"Stock symbol {code} (ISIN {isin}) has no matching fundamentals row"
            )
        code_to_name[code] = fund_isin_to_name[isin]

    # Sanity gate: the file-derived mapping must match what we expect. If a future
    # data drop reshuffles symbols or ISINs, this raises instead of silently
    # producing a mislabelled analysis.
    mismatches = {
        c: (code_to_name.get(c), EXPECTED_STOCK_CODE_TO_NAME.get(c))
        for c in EXPECTED_STOCK_CODE_TO_NAME
        if code_to_name.get(c) != EXPECTED_STOCK_CODE_TO_NAME.get(c)
    }
    if mismatches:
        raise ValueError(
            "ISIN-derived mapping disagrees with EXPECTED_STOCK_CODE_TO_NAME "
            f"(derived, expected): {mismatches}. Update constants.py only after "
            "confirming the new data file is correct."
        )

    return code_to_name


def clean_stock_data(df, codes):
    df = df.copy()
    df.columns = ['Date'] + list(df.columns[1:])

    # Metadata rows ("Name", "ISIN Number", ...) parse to NaT and are dropped below,
    # so we no longer hard-code a fixed number of header rows to skip.
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    for code in codes:
        df[code] = pd.to_numeric(df[code], errors='coerce')

    df = df.dropna(subset=['Date'])
    df.set_index('Date', inplace=True)

    return df.resample('YE').last().pct_change().dropna()

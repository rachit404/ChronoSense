import pandas as pd
import numpy as np

def load_data(file_path: str, col: str, date_col: str = "Date", parse_dates: bool = True) -> pd.DataFrame:
    """
    Load CSV into a pandas DataFrame, set the date column as a datetime index (if present),
    and ensure the requested column exists.
    
    Parameters
    ---------
    file_path : str
        Path to the CSV file.
    col : str
        Name of the price column to focus on (e.g., 'Close').
    date_col : str
        Name of the date column. Defaults to 'Date'.
    parse_dates : bool
        Whether to parse the date column as datetime.
    
    Returns
    -------
    pd.DataFrame
        Loaded DataFrame with a datetime index (if date_col exists) and containing the requested column.
    """
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df = df.set_index("Date").sort_index()
    # parse date column and set index if exists
    if parse_dates and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        if df[date_col].isnull().all():
            # failed parsing, keep as-is
            df = df.copy()
        else:
            df = df.set_index(date_col).sort_index()
    # ensure the requested column exists; if not, try to pick a sensible default
    if col not in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        fallback = numeric_cols[0] if numeric_cols else None
        raise ValueError(f"Column '{col}' not found in file. Numeric columns available: {numeric_cols}. "
                         f"Consider using one of those (e.g. '{fallback}').")
    return df
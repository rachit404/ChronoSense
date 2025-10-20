import pandas as pd

def load_dataset(file):
    """
    Load CSV or Excel dataset into a pandas DataFrame.
    """
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file)
        else:
            raise ValueError("Unsupported file format. Please upload CSV or Excel.")
        return df
    except Exception as e:
        raise ValueError(f"Error loading file: {e}")

def validate_dataset(df):
    """
    Validate dataset for time-series analysis.
    Ensures a valid datetime column and Arrow-safe dataframe.
    """
    # Try to detect columns likely to be datetime
    datetime_cols = df.select_dtypes(include=["datetime", "object"]).columns.tolist()

    detected_datetime_col = None

    for col in datetime_cols:
        try:
            parsed_col = pd.to_datetime(df[col], errors="raise")
            df[col] = parsed_col
            detected_datetime_col = col
            break  # Stop at the first valid datetime column
        except (ValueError, TypeError):
            continue

    if not detected_datetime_col:
        raise ValueError("No valid datetime column found. Ensure your dataset has at least one date/time column.")

    # Ensure dataframe is sorted and Arrow-safe
    df = df.sort_values(by=detected_datetime_col)
    df = df.convert_dtypes()  # ensures no 'object' dtype remains

    # Explicitly convert datetime column to string for Streamlit display safety
    df[detected_datetime_col] = df[detected_datetime_col].astype(str)

    return detected_datetime_col, df

def preprocess_dataset(df, datetime_col):
    """
    Basic preprocessing:
    - Sort by datetime
    - Fill missing numeric values with forward fill
    """
    df = df.sort_values(by=datetime_col)
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].ffill()
    return df

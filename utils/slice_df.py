def slice_df(df, slice_range=None):
    """
    Dynamically slice the dataframe if a range is provided.
    Example: slice_range=(750, None) or slice_range=(None, 500)
    """
    if slice_range is not None:
        start, end = slice_range
        return df[start:end]
    return df
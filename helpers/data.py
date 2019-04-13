def extract_label_values(df, values_col=2):
    """
    extract values from dataframe of labels

    Parameters
    ----------
    df : pandas.DataFrame
         labels

    values_col: int
                column index of df where values start
    """
    return df.iloc[:, 2:].values

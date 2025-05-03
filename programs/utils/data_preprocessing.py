import pandas as pd

class Data_overview:
    # return columns list about 
    def _remove_matching_columns(df:pd.DataFrame, exclude_list: list = None) -> list:
        """
        returns a list of column names from the given DataFrame after removing the columns specified in sxclude_list.

        df: Pandas Dataframe
        exclude_list: A list of column names to be edxcluded
        (default:None)

        return: A list of column names excluding those specified in exclude_list
        """
        df_columns = df.columns.tolist()
        if not exclude_list:
            return df_columns
        return [col for col in df_columns if col not in exclude_list]
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import pandas as pd
import numpy as np

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

    def evaluate_null_columns_prediction_model(df:pd.DataFrame, column:str, features: list) -> DataFrame:
        """
        Train and evaluate an XGBoost regression model using data a numeric column (int or float) that may contain missing (null) values.

        Parameters:
        - df: DataFrame where the one column is a numeric feature (int or float) that may contain missing values
        - features: List of feature columns to be used for predicting column's value

        Returns:
        - Prints evaluation metrics and returns a DataFrame with actual and predicted column's values
        """
        df_known = df[df[column].notnull()].copy()

        X = pd.get_dummies(df_known[features])
        y = df_known[column]

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        print(f" RMSE: {rmse:.2f}")
        print(f" MAE : {mae:.2f}")
        print(f" R²  : {r2:.2f}")

        return pd.DataFrame({"Actual": y_val, "Predicted": y_pred})
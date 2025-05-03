import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def remove_matching_columns(df: pd.DataFrame, exclude_list: list = None) -> list:
    """
    Return a list of column names from the given DataFrame after removing those in exclude_list.
    """
    df_columns = df.columns.tolist()
    if not exclude_list:
        return df_columns
    return [col for col in df_columns if col not in exclude_list]


def evaluate_null_columns_prediction_model(df: pd.DataFrame, column: str, features: list) -> pd.DataFrame:
    """
    Train and evaluate an XGBoost regression model on a numeric column with missing values.
    Returns predicted vs actual values (rounded to nearest int).
    """
    df_known = df[df[column].notnull()].copy()

    X = pd.get_dummies(df_known[features])
    y = df_known[column]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_pred_rounded = np.round(y_pred)

    rmse = np.sqrt(mean_squared_error(y_val, y_pred_rounded))
    mae = mean_absolute_error(y_val, y_pred_rounded)
    r2 = r2_score(y_val, y_pred_rounded)

    print(f" RMSE: {rmse:.2f}")
    print(f" MAE : {mae:.2f}")
    print(f" R²  : {r2:.2f}")

    return pd.DataFrame({"Actual": y_val, "Predicted": y_pred_rounded})


def plot_binary_ratio_by_category(df: pd.DataFrame, category_col: str, target_col: str) -> None:
    """
    Visualize the 0/1 ratio of a binary target variable
    for each category of a categorical feature
    using a 100% stacked horizontal bar chart.
    """
    count_df = df.groupby([category_col, target_col]).size().reset_index(name='count')
    count_df['ratio'] = count_df.groupby(category_col)['count'].transform(lambda x: x / x.sum())
    plot_df = count_df.pivot(index=category_col, columns=target_col, values='ratio').fillna(0)

    if 0 in plot_df.columns and 1 in plot_df.columns:
        plot_df = plot_df[[0, 1]]
    else:
        plot_df = plot_df[sorted(plot_df.columns)]

    total_counts = df[category_col].value_counts().to_dict()

    ax = plot_df.plot(kind='barh', stacked=True, color=['lightgray', 'steelblue'], figsize=(12, 8))

    for i, (idx, row) in enumerate(plot_df.iterrows()):
        cumulative = 0
        for col in plot_df.columns:
            val = row[col]
            if val > 0:
                ax.text(cumulative + val / 2, i, f"{val * 100:.0f}%", va='center', ha='center', fontsize=10)
                cumulative += val
        ax.text(1.02, i, f"n={total_counts[idx]}", va='center', ha='left', fontsize=10, fontweight='bold')

    plt.title(f"Normalized Ratio of {target_col} per {category_col}")
    plt.xlabel("Percentage")
    plt.ylabel(category_col)
    plt.xlim(0, 1.10)
    plt.legend(title=target_col, loc='upper left')
    sns.despine()
    plt.tight_layout()
    plt.show()
    return


def cluster_titles_by_survival(df: pd.DataFrame, category_col: str, target_col: str,
                                k_range: range = range(2, 6)) -> dict:
    """
    Automatically cluster categories of a categorical column based on the ratio of a binary target variable.
    The optimal number of clusters is selected using silhouette score.

    Returns:
    - category_cluster_map: A dictionary mapping each category to its assigned cluster.
    """
    summary_df = df.groupby(category_col)[target_col].mean().reset_index()
    X = summary_df[[target_col]].values
    best_k, best_score = None, -1

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k, best_score = k, score

    final_model = KMeans(n_clusters=best_k, random_state=42)
    summary_df['Cluster'] = final_model.fit_predict(X)

    category_cluster_map = dict(zip(summary_df[category_col], summary_df['Cluster']))
    return category_cluster_map
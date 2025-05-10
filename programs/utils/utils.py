import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report
from xgboost import XGBRegressor, XGBClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

def plot_outliers(df, column):
    """
    Show boxplot to visualize outliers in a numerical column.
    """
    plt.figure(figsize=(8, 2))
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot for {column}')
    plt.show()

def remove_matching_columns(df: pd.DataFrame, exclude_list: list = None) -> list:
    """
    Return a list of column names from the given DataFrame after removing those in exclude_list.
    """
    df_columns = df.columns.tolist()
    if not exclude_list:
        return df_columns
    return [col for col in df_columns if col not in exclude_list]


def evaluate_null_int_columns_prediction_model(df: pd.DataFrame, column: str, features: list) -> None:
    """
    Train and evaluate an XGBoost regression model on a numeric column with missing values.
    Performs GridSearchCV to tune hyperparameters.
    Returns predicted vs actual values (rounded to nearest int).
    """
    df_known = df[df[column].notnull()].copy()

    X = pd.get_dummies(df_known[features])
    y = df_known[column]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_estimators': [100, 300, 500, 1000],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.2, 0.5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
    }

    model = XGBRegressor(random_state=42, verbosity=0)
    grid_search = GridSearchCV(estimator=model,
                               param_grid=param_grid,
                               scoring='neg_mean_squared_error',
                               cv=3,
                               verbose=2,
                               n_jobs=-1)

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_val)
    y_pred_rounded = np.round(y_pred)

    rmse = np.sqrt(mean_squared_error(y_val, y_pred_rounded))
    mae = mean_absolute_error(y_val, y_pred_rounded)
    r2 = r2_score(y_val, y_pred_rounded)

    print("Best hyperparameters:")
    print(grid_search.best_params_)
    print(f"\n RMSE: {rmse:.2f}")
    print(f" MAE : {mae:.2f}")
    print(f" R²  : {r2:.2f}")

def evaluate_null_categorical_column_prediction_model(df: pd.DataFrame, column: str, features: list) -> None:
    """
    Train and evaluate an XGBoost classification model on a categorical column with missing values.
    Performs GridSearchCV to tune hyperparameters and prints classification metrics.
    """
    df_known = df[df[column].notnull()].copy()

    X = pd.get_dummies(df_known[features])
    y = df_known[column]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    param_grid = {
        'n_estimators': [100, 300, 500, 1000],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.2, 0.5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
    }
    
    model = XGBClassifier(random_state=42, verbosity=0, use_label_encoder=False)
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='f1_macro',
        cv=3,
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train, sample_weight=sample_weights)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_val)

    print("Best hyperparameters:")
    print(grid_search.best_params_)
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))

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

def evaluate_null_categorical_column_prediction_model_rf(df: pd.DataFrame, category_col: str, features: list) -> None:
    """
    Train and evaluate a RandomForest classification model on a categorical column with missing values.
    Uses Label Encoding instead of One-Hot Encoding for categorical variables.
    Performs GridSearchCV to tune hyperparameters and prints classification metrics.
    """
    df_known = df[df[category_col].notnull()].copy()
    df_encoded = df_known.copy()

    # 범주형 변수에 Label Encoding 적용
    label_encoders = {}
    for col in features:
        if df_encoded[col].dtype == 'object' or df_encoded[col].dtype.name == 'category':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le

    X = df_encoded[features]
    y = df_encoded[category_col]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }

    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train, sample_weight=sample_weights)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_val)

    print("Best hyperparameters:")
    print(grid_search.best_params_)
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))

def fill_embarked_with_mode(df,Mode_Column):
    """
    A function that fills missing values in the Selected column with the mode
    """
    mode_value = df[Mode_Column].mode()[0]
    return df[Mode_Column].fillna(mode_value)

from sklearn.preprocessing import StandardScaler

def standardize_column(df, column):
    scaler = StandardScaler()
    df[column + "_z"] = scaler.fit_transform(df[[column]])
    return df
import numpy as np
import pandas as pd


def add_usage_features(df):
    df = df.copy()

    df["social_gaming_hours"] = df["social_media_hours"] + df["gaming_hours"]

    df["social_media_ratio"] = df["social_media_hours"] / df["daily_screen_time_hours"].replace(0, np.nan)
    df["gaming_ratio"] = df["gaming_hours"] / df["daily_screen_time_hours"].replace(0, np.nan)

    df["weekend_weekday_gap"] = df["weekend_screen_time"] - df["daily_screen_time_hours"]

    df["notif_per_app_open"] = df["notifications_per_day"] / df["app_opens_per_day"].replace(0, np.nan)

    df["screen_time_per_sleep"] = df["daily_screen_time_hours"] / df["sleep_hours"].replace(0, np.nan)

    return df


def target_encode_kfold(train, test, col, target, n_splits=5, seed=42):
    from sklearn.model_selection import StratifiedKFold

    train = train.copy()
    test = test.copy()

    oof = np.zeros(len(train))
    global_mean = train[target].mean()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, valid_idx in skf.split(train, train[target]):
        fold_train = train.iloc[train_idx]
        means = fold_train.groupby(col)[target].mean()
        oof[valid_idx] = train.iloc[valid_idx][col].map(means).fillna(global_mean)

    train[f"{col}_target_enc"] = oof

    full_means = train.groupby(col)[target].mean()
    test[f"{col}_target_enc"] = test[col].map(full_means).fillna(global_mean)

    return train, test
import pandas as pd

NUM_COLS = [
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
]

CAT_COLS = ["gender", "stress_level", "academic_work_impact"]


def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def add_missing_indicators(df, cols=None):
    cols = cols or (NUM_COLS + CAT_COLS)
    df = df.copy()
    for col in cols:
        df[f"{col}_missing"] = df[col].isna().astype(int)
    return df


def fill_missing(df, num_stats=None, cat_stats=None):
    df = df.copy()

    if num_stats is None:
        num_stats = {col: df[col].median() for col in NUM_COLS}
    for col, val in num_stats.items():
        df[col] = df[col].fillna(val)

    if cat_stats is None:
        cat_stats = {col: df[col].mode()[0] for col in CAT_COLS}
    for col, val in cat_stats.items():
        df[col] = df[col].fillna(val)

    return df, num_stats, cat_stats


def encode_categoricals(train, test):
    train = train.copy()
    test = test.copy()
    for col in CAT_COLS:
        categories = sorted(train[col].unique())
        mapping = {cat: i for i, cat in enumerate(categories)}
        train[col] = train[col].map(mapping)
        test[col] = test[col].map(mapping).fillna(-1)
    return train, test


def check_data_quality(df, name="dataset"):
    print(name, df.shape)
    print("id 중복:", df["id"].duplicated().sum())
    print("행 전체 중복:", df.duplicated().sum())
    print("결측률(%)")
    print((df.isna().mean() * 100).round(2).sort_values(ascending=False))
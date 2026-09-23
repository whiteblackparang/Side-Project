import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb


def run_logistic_baseline(X, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    scores = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)

        pred = model.predict_proba(X_valid_scaled)[:, 1]
        oof[valid_idx] = pred

        fold_auc = roc_auc_score(y_valid, pred)
        scores.append(fold_auc)
        print(f"fold {fold} auc: {fold_auc:.5f}")

    print(f"평균 auc: {np.mean(scores):.5f}")
    return oof, scores


def run_lightgbm(X, y, cat_features=None, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    scores = []
    models = []

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "seed": seed,
        "verbosity": -1,
    }

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        train_set = lgb.Dataset(X_train, y_train, categorical_feature=cat_features)
        valid_set = lgb.Dataset(X_valid, y_valid, categorical_feature=cat_features)

        model = lgb.train(
            params,
            train_set,
            valid_sets=[valid_set],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        pred = model.predict(X_valid, num_iteration=model.best_iteration)
        oof[valid_idx] = pred

        fold_auc = roc_auc_score(y_valid, pred)
        scores.append(fold_auc)
        models.append(model)
        print(f"fold {fold} auc: {fold_auc:.5f}")

    print(f"평균 auc: {np.mean(scores):.5f}")
    return oof, scores, models
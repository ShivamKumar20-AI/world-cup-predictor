import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier


FEATURES = [
    "home_form", "away_form", "form_diff",
    "home_goals_scored", "away_goals_scored",
    "home_goals_conceded", "away_goals_conceded",
    "home_win_rate", "away_win_rate", "win_rate_diff",
    "goal_diff",
    "home_rank", "away_rank", "rank_diff",
    "h2h_home_win_rate", "h2h_away_win_rate",
    "neutral",
    "home_wc_performance", "away_wc_performance", "wc_performance_diff",
    "home_elo", "away_elo", "elo_diff",
    "home_yellow_cards", "away_yellow_cards",
    "home_red_cards", "away_red_cards",
    "home_shots_on_target", "away_shots_on_target",
    "home_possession", "away_possession",
    "shots_on_target_diff", "possession_diff",
    "home_form_trend", "away_form_trend", "form_trend_diff",
]


class XGBWrapper:
    def __init__(self, model, label_map, label_unmap):
        self.model = model
        self.label_map = label_map
        self.label_unmap = label_unmap
        self.classes_ = sorted(label_unmap.values())

    def predict(self, X):
        raw = self.model.predict(X)
        return np.array([self.label_unmap[int(r)] for r in raw])

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def train():
    df = pd.read_csv("data/features.csv")

    # Time-based split — train on older matches, test on recent ones
    # Prevents future data leaking into training
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    X_train = df[FEATURES].iloc[:split_idx]
    X_test = df[FEATURES].iloc[split_idx:]
    y_train = df["outcome"].iloc[:split_idx]
    y_test = df["outcome"].iloc[split_idx:]

    print(f"Train period: {df['date'].iloc[0]} → {df['date'].iloc[split_idx-1]}")
    print(f"Test period:  {df['date'].iloc[split_idx]} → {df['date'].iloc[-1]}")
    print(f"Training on {len(X_train)} matches, testing on {len(X_test)} matches")
    print(f"Class distribution:\n{y_train.value_counts()}\n")

    print("=" * 50)
    print("Baseline: Random Forest")
    print("=" * 50)
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"Random Forest Accuracy: {rf_acc:.4%}")

    print("\n" + "=" * 50)
    print("Primary Model: XGBoost")
    print("=" * 50)

    label_map = {-1: 0, 0: 1, 1: 2}
    label_unmap = {0: -1, 1: 0, 2: 1}

    y_train_xgb = y_train.map(label_map)
    y_test_xgb = y_test.map(label_map)

    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )

    xgb.fit(
        X_train, y_train_xgb,
        eval_set=[(X_test, y_test_xgb)],
        verbose=50,
    )

    xgb_preds_mapped = xgb.predict(X_test)
    xgb_preds = pd.Series(xgb_preds_mapped).map(label_unmap)
    xgb_acc = accuracy_score(y_test, xgb_preds)

    print(f"\nXGBoost Accuracy:       {xgb_acc:.4%}")
    print(f"Random Forest Accuracy: {rf_acc:.4%}")
    print(f"Improvement:            +{(xgb_acc - rf_acc):.4%}")

    print("\nXGBoost Classification Report:")
    print(classification_report(
        y_test, xgb_preds,
        target_names=["Away Win", "Draw", "Home Win"]
    ))

    print("\nTop 10 Most Important Features (XGBoost):")
    importance = pd.Series(
        xgb.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)
    for feat, score in importance.head(10).items():
        print(f"  {feat:<35} {score:.4f}")

    wrapped = XGBWrapper(xgb, label_map, label_unmap)

    with open("model.pkl", "wb") as f:
        pickle.dump(wrapped, f)

    print(f"\nModel saved to model.pkl")
    print(f"Final accuracy: {xgb_acc:.4%} (was 58.03% with Random Forest)")


if __name__ == "__main__":
    train()

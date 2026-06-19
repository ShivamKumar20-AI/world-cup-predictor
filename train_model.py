import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

def train():
    df = pd.read_csv("data/features.csv")

    X = df[[
        "home_form",
        "away_form",
        "form_diff",
        "home_goals_scored",
        "away_goals_scored",
        "home_goals_conceded",
        "away_goals_conceded",
        "home_win_rate",
        "away_win_rate",
        "win_rate_diff",
        "goal_diff",
        "home_rank",
        "away_rank",
        "rank_diff"
    ]]

    y = df["outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training on {len(X_train)} matches...")
    print(f"Testing on {len(X_test)} matches...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
          target_names=["Away Win", "Draw", "Home Win"]))

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved to model.pkl")

if __name__ == "__main__":
    train()
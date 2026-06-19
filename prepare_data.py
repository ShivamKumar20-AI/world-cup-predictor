import pandas as pd
import numpy as np

def load_rankings():
    dfs = []
    for f in [
        "data/fifa_ranking-2023-07-20.csv",
        "data/fifa_ranking-2024-04-04.csv",
        "data/fifa_ranking-2024-06-20.csv"
    ]:
        dfs.append(pd.read_csv(f))
    rankings = pd.concat(dfs).drop_duplicates()
    rankings["rank_date"] = pd.to_datetime(rankings["rank_date"])
    return rankings

def get_ranking(rankings, team, date):
    team_ranks = rankings[
        (rankings["country_full"] == team) &
        (rankings["rank_date"] <= date)
    ]
    if len(team_ranks) == 0:
        return 100  # default rank if not found
    return team_ranks.sort_values("rank_date").iloc[-1]["rank"]

def load_and_prepare():
    results = pd.read_csv("data/results.csv")
    results["date"] = pd.to_datetime(results["date"])
    results = results[results["date"] >= "2000-01-01"].copy()
    results = results[results["home_score"].notna() & results["away_score"].notna()].copy()

    rankings = load_rankings()

    print(f"Total matches loaded: {len(results)}")

    def get_outcome(row):
        if row["home_score"] > row["away_score"]:
            return 1
        elif row["home_score"] == row["away_score"]:
            return 0
        else:
            return -1

    results["outcome"] = results.apply(get_outcome, axis=1)

    def get_team_stats(df, team, date, n=10):
        team_matches = df[
            ((df["home_team"] == team) | (df["away_team"] == team)) &
            (df["date"] < date)
        ].tail(n)

        if len(team_matches) == 0:
            return 0.5, 0, 0, 0

        points = []
        goals_scored = []
        goals_conceded = []
        wins = 0

        for _, row in team_matches.iterrows():
            if row["home_team"] == team:
                gs = float(row["home_score"])
                gc = float(row["away_score"])
                if row["outcome"] == 1:
                    points.append(1)
                    wins += 1
                elif row["outcome"] == 0:
                    points.append(0.5)
                else:
                    points.append(0)
            else:
                gs = float(row["away_score"])
                gc = float(row["home_score"])
                if row["outcome"] == -1:
                    points.append(1)
                    wins += 1
                elif row["outcome"] == 0:
                    points.append(0.5)
                else:
                    points.append(0)
            goals_scored.append(gs)
            goals_conceded.append(gc)

        return (
            np.mean(points),
            np.mean(goals_scored),
            np.mean(goals_conceded),
            wins / len(team_matches)
        )

    print("Building features (this may take a few minutes)...")
    rows = []
    for _, match in results.iterrows():
        h_form, h_gf, h_ga, h_wr = get_team_stats(results, match["home_team"], match["date"])
        a_form, a_gf, a_ga, a_wr = get_team_stats(results, match["away_team"], match["date"])

        h_rank = get_ranking(rankings, match["home_team"], match["date"])
        a_rank = get_ranking(rankings, match["away_team"], match["date"])

        rows.append({
            "home_form": h_form,
            "away_form": a_form,
            "form_diff": h_form - a_form,
            "home_goals_scored": h_gf,
            "away_goals_scored": a_gf,
            "home_goals_conceded": h_ga,
            "away_goals_conceded": a_ga,
            "home_win_rate": h_wr,
            "away_win_rate": a_wr,
            "win_rate_diff": h_wr - a_wr,
            "goal_diff": h_gf - a_gf,
            "home_rank": h_rank,
            "away_rank": a_rank,
            "rank_diff": a_rank - h_rank,
            "outcome": match["outcome"]
        })

    df_features = pd.DataFrame(rows)
    df_features.to_csv("data/features.csv", index=False)
    print(f"Done! Features saved with {len(df_features)} matches.")
    return df_features

if __name__ == "__main__":
    load_and_prepare()
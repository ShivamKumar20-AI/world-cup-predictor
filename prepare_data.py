import pandas as pd
import numpy as np
from tqdm import tqdm


# ── Loaders ──────────────────────────────────────────────────────────────────

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


def load_elo():
    elo = pd.read_csv("data/eloratings.csv")
    elo["date"] = pd.to_datetime(elo["date"], format="mixed")
    return elo


def load_wc_stats():
    wc = pd.read_csv("data/fifa_wc_mens_match_dataset_1970_2022.csv")
    team_stats = wc.groupby("team_name").agg(
        avg_yellow_cards=("yellow_cards", "mean"),
        avg_red_cards=("red_cards", "mean"),
        avg_shots_on_target=("shots_on_target", "mean"),
        avg_possession=("possession", "mean")
    ).reset_index()
    return team_stats


# ── Lookup helpers ────────────────────────────────────────────────────────────

def get_wc_team_stats(wc_stats, team):
    row = wc_stats[wc_stats["team_name"] == team]
    if len(row) == 0:
        return 2.0, 0.1, 5.0, 50.0
    return (
        row["avg_yellow_cards"].values[0],
        row["avg_red_cards"].values[0],
        row["avg_shots_on_target"].values[0] if not np.isnan(row["avg_shots_on_target"].values[0]) else 5.0,
        row["avg_possession"].values[0] if not np.isnan(row["avg_possession"].values[0]) else 50.0
    )


def elo_win_prob(home_elo, away_elo):
    """Expected win probability from Elo ratings (non-linear, more informative than raw diff)."""
    return 1 / (1 + 10 ** ((away_elo - home_elo) / 400))


# ── Tournament weight ─────────────────────────────────────────────────────────
# Check qualifiers FIRST so "FIFA World Cup qualification" gets weight 2, not 3.

def tournament_weight(tournament):
    if any(t in tournament for t in ["qualification", "Qualifier"]):
        return 2.0
    elif any(t in tournament for t in ["FIFA World Cup", "UEFA Euro", "Copa America", "AFC Asian Cup"]):
        return 3.0
    return 1.0


# ── Per-team rolling stats (computed once per team, not per row) ──────────────

def build_team_form_cache(results, n=10):
    """
    Pre-compute rolling form stats for every team at every match date.
    Returns a dict: {(team, date) -> (form, avg_gf, avg_ga, win_rate)}
    Much faster than recomputing inside iterrows().
    """
    results = results.sort_values("date").reset_index(drop=True)
    cache = {}

    all_teams = pd.concat([results["home_team"], results["away_team"]]).unique()

    for team in tqdm(all_teams, desc="Building form cache"):
        team_matches = results[
            (results["home_team"] == team) | (results["away_team"] == team)
        ].copy()

        for i, (_, match) in enumerate(team_matches.iterrows()):
            past = team_matches.iloc[max(0, i - n):i]

            if len(past) == 0:
                cache[(team, match["date"])] = (0.5, 0.0, 0.0, 0.5)
                continue

            points, goals_scored, goals_conceded = [], [], []
            wins = 0

            for _, row in past.iterrows():
                w = tournament_weight(row["tournament"])
                if row["home_team"] == team:
                    gs, gc = float(row["home_score"]), float(row["away_score"])
                    if row["outcome"] == 1:   points.append(1 * w); wins += 1
                    elif row["outcome"] == 0: points.append(0.5 * w)
                    else:                     points.append(0)
                else:
                    gs, gc = float(row["away_score"]), float(row["home_score"])
                    if row["outcome"] == -1:  points.append(1 * w); wins += 1
                    elif row["outcome"] == 0: points.append(0.5 * w)
                    else:                     points.append(0)
                goals_scored.append(gs)
                goals_conceded.append(gc)

            form = np.sum(points) / (len(points) * 3.0)
            cache[(team, match["date"])] = (
                form,
                np.mean(goals_scored),
                np.mean(goals_conceded),
                wins / len(past)
            )

    return cache


def build_h2h_cache(results, n=10):
    """
    Pre-compute H2H win rate for every (home_team, away_team, date) triplet.
    Returns a dict: {(home_team, away_team, date) -> home_win_rate}
    """
    results = results.sort_values("date").reset_index(drop=True)
    cache = {}

    pairs = results[["home_team", "away_team"]].drop_duplicates()

    for _, pair in tqdm(pairs.iterrows(), total=len(pairs), desc="Building H2H cache"):
        ht, at = pair["home_team"], pair["away_team"]
        h2h = results[
            ((results["home_team"] == ht) & (results["away_team"] == at)) |
            ((results["home_team"] == at) & (results["away_team"] == ht))
        ].copy()

        for i, (_, match) in enumerate(h2h.iterrows()):
            past = h2h.iloc[max(0, i - n):i]
            if len(past) == 0:
                cache[(ht, at, match["date"])] = 0.5
                continue
            home_wins = sum(
                1 for _, r in past.iterrows()
                if (r["home_team"] == ht and r["outcome"] == 1) or
                   (r["away_team"] == ht and r["outcome"] == -1)
            )
            cache[(ht, at, match["date"])] = min(home_wins / len(past), 1.0)

    return cache


def build_wc_performance_cache(results):
    """
    Pre-compute World Cup performance for every (team, date).
    Returns a dict: {(team, date) -> wc_win_rate}
    """
    wc = results[results["tournament"] == "FIFA World Cup"].copy()
    cache = {}

    all_teams = pd.concat([results["home_team"], results["away_team"]]).unique()

    for team in all_teams:
        team_wc = wc[
            (wc["home_team"] == team) | (wc["away_team"] == team)
        ].sort_values("date")

        match_dates = results[
            (results["home_team"] == team) | (results["away_team"] == team)
        ]["date"].unique()

        for date in match_dates:
            past_wc = team_wc[team_wc["date"] < date]
            if len(past_wc) == 0:
                cache[(team, date)] = 0.5
                continue
            points = []
            for _, row in past_wc.iterrows():
                if row["home_team"] == team:
                    if row["outcome"] == 1:   points.append(1)
                    elif row["outcome"] == 0: points.append(0.5)
                    else:                     points.append(0)
                else:
                    if row["outcome"] == -1:  points.append(1)
                    elif row["outcome"] == 0: points.append(0.5)
                    else:                     points.append(0)
            cache[(team, date)] = np.mean(points)

    return cache


# ── Rankings / Elo via merge_asof (vectorised, no loop) ──────────────────────

def merge_rankings(results, rankings):
    """Attach home and away FIFA rank to each match using merge_asof."""
    ranks = rankings.sort_values("rank_date")

    home_ranks = pd.merge_asof(
        results[["date", "home_team"]].sort_values("date"),
        ranks[["rank_date", "country_full", "rank"]].rename(
            columns={"rank_date": "date", "country_full": "home_team", "rank": "home_rank"}
        ),
        on="date", by="home_team", direction="backward"
    )
    away_ranks = pd.merge_asof(
        results[["date", "away_team"]].sort_values("date"),
        ranks[["rank_date", "country_full", "rank"]].rename(
            columns={"rank_date": "date", "country_full": "away_team", "rank": "away_rank"}
        ),
        on="date", by="away_team", direction="backward"
    )

    results = results.sort_values("date").copy()
    results["home_rank"] = home_ranks["home_rank"].values
    results["away_rank"] = away_ranks["away_rank"].values
    results["home_rank"] = results["home_rank"].fillna(100)
    results["away_rank"] = results["away_rank"].fillna(100)
    return results


def merge_elo(results, elo):
    """Attach home and away Elo rating to each match using merge_asof."""
    elo_sorted = elo.sort_values("date")

    home_elo = pd.merge_asof(
        results[["date", "home_team"]].sort_values("date"),
        elo_sorted[["date", "team", "rating"]].rename(
            columns={"team": "home_team", "rating": "home_elo"}
        ),
        on="date", by="home_team", direction="backward"
    )
    away_elo = pd.merge_asof(
        results[["date", "away_team"]].sort_values("date"),
        elo_sorted[["date", "team", "rating"]].rename(
            columns={"team": "away_team", "rating": "away_elo"}
        ),
        on="date", by="away_team", direction="backward"
    )

    results = results.sort_values("date").copy()
    results["home_elo"] = home_elo["home_elo"].values
    results["away_elo"] = away_elo["away_elo"].values
    results["home_elo"] = results["home_elo"].fillna(1500)
    results["away_elo"] = results["away_elo"].fillna(1500)
    return results


# ── Main pipeline ─────────────────────────────────────────────────────────────

def load_and_prepare():
    results = pd.read_csv("data/results.csv")
    results["date"] = pd.to_datetime(results["date"])
    results = results[results["date"] >= "2000-01-01"].copy()
    results = results[results["home_score"].notna() & results["away_score"].notna()].copy()
    results = results.sort_values("date").reset_index(drop=True)

    rankings = load_rankings()
    elo = load_elo()
    wc_stats = load_wc_stats()

    print(f"Total matches loaded: {len(results)}")

    def get_outcome(row):
        if row["home_score"] > row["away_score"]:    return 1
        elif row["home_score"] == row["away_score"]: return 0
        else:                                         return -1

    results["outcome"] = results.apply(get_outcome, axis=1)

    print("Merging FIFA rankings...")
    results = merge_rankings(results, rankings)

    print("Merging Elo ratings...")
    results = merge_elo(results, elo)

    print("Building form cache...")
    form_cache = build_team_form_cache(results)

    print("Building H2H cache...")
    h2h_cache = build_h2h_cache(results)

    print("Building World Cup performance cache...")
    wc_cache = build_wc_performance_cache(results)

    print("Assembling features...")
    rows = []
    for _, match in tqdm(results.iterrows(), total=len(results), desc="Assembling rows"):
        ht, at, date = match["home_team"], match["away_team"], match["date"]

        h_form, h_gf, h_ga, h_wr = form_cache.get((ht, date), (0.5, 0.0, 0.0, 0.5))
        a_form, a_gf, a_ga, a_wr = form_cache.get((at, date), (0.5, 0.0, 0.0, 0.5))

        h_rank = match["home_rank"]
        a_rank = match["away_rank"]

        h_elo = match["home_elo"]
        a_elo = match["away_elo"]

        h2h_home = h2h_cache.get((ht, at, date), 0.5)
        h2h_away = h2h_cache.get((at, ht, date), 0.5)

        h_wc = wc_cache.get((ht, date), 0.5)
        a_wc = wc_cache.get((at, date), 0.5)

        h_yc, h_rc, h_sot, h_poss = get_wc_team_stats(wc_stats, ht)
        a_yc, a_rc, a_sot, a_poss = get_wc_team_stats(wc_stats, at)

        rows.append({
            # Form
            "home_form":             h_form,
            "away_form":             a_form,
            "form_diff":             h_form - a_form,
            # Goals
            "home_goals_scored":     h_gf,
            "away_goals_scored":     a_gf,
            "home_goals_conceded":   h_ga,
            "away_goals_conceded":   a_ga,
            "goal_diff":             h_gf - a_gf,
            # Win rate
            "home_win_rate":         h_wr,
            "away_win_rate":         a_wr,
            "win_rate_diff":         h_wr - a_wr,
            # FIFA rank (lower = better; positive diff = home team ranked higher)
            "home_rank":             h_rank,
            "away_rank":             a_rank,
            "rank_diff":             a_rank - h_rank,
            # H2H
            "h2h_home_win_rate":     h2h_home,
            "h2h_away_win_rate":     h2h_away,
            # Venue
            "neutral":               int(match["neutral"]),
            # World Cup history
            "home_wc_performance":   h_wc,
            "away_wc_performance":   a_wc,
            "wc_performance_diff":   h_wc - a_wc,
            # Elo
            "home_elo":              h_elo,
            "away_elo":              a_elo,
            "elo_diff":              h_elo - a_elo,
            "home_elo_win_prob":     elo_win_prob(h_elo, a_elo),
            # WC tactical stats
            "home_yellow_cards":     h_yc,
            "away_yellow_cards":     a_yc,
            "home_red_cards":        h_rc,
            "away_red_cards":        a_rc,
            "home_shots_on_target":  h_sot,
            "away_shots_on_target":  a_sot,
            "home_possession":       h_poss,
            "away_possession":       a_poss,
            "shots_on_target_diff":  h_sot - a_sot,
            "possession_diff":       h_poss - a_poss,
            # Meta
            "date":                  date,
            "outcome":               match["outcome"]
        })

    df_features = pd.DataFrame(rows)
    df_features.to_csv("data/features.csv", index=False)

    with open("data/features_meta.txt", "w") as f:
        f.write(
            f"Built: {pd.Timestamp.now()}\n"
            f"Matches: {len(df_features)}\n"
            f"Features: {list(df_features.columns)}\n"
        )

    print(f"Done! Features saved with {len(df_features)} matches and {len(df_features.columns)-2} feature columns.")
    return df_features


if __name__ == "__main__":
    load_and_prepare()
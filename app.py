import streamlit as st
import pickle
import numpy as np
import pandas as pd

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    results = pd.read_csv("data/results.csv")
    teams = sorted(set(results["home_team"].unique()) | set(results["away_team"].unique()))

    dfs = []
    for f in [
        "data/fifa_ranking-2023-07-20.csv",
        "data/fifa_ranking-2024-04-04.csv",
        "data/fifa_ranking-2024-06-20.csv"
    ]:
        dfs.append(pd.read_csv(f))
    rankings = pd.concat(dfs).drop_duplicates()
    rankings["rank_date"] = pd.to_datetime(rankings["rank_date"])

    elo = pd.read_csv("data/eloratings.csv")
    elo["date"] = pd.to_datetime(elo["date"], format="mixed")

    wc = pd.read_csv("data/fifa_wc_mens_match_dataset_1970_2022.csv")
    wc_stats = wc.groupby("team_name").agg(
        avg_yellow_cards=("yellow_cards", "mean"),
        avg_red_cards=("red_cards", "mean"),
        avg_shots_on_target=("shots_on_target", "mean"),
        avg_possession=("possession", "mean")
    ).reset_index()

    return results, teams, rankings, elo, wc_stats

def get_team_stats(df_results, team, n=10):
    team_matches = df_results[
        ((df_results["home_team"] == team) | (df_results["away_team"] == team)) &
        (df_results["home_score"].notna()) &
        (df_results["away_score"].notna())
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
            if row["home_score"] > row["away_score"]:
                points.append(1)
                wins += 1
            elif row["home_score"] == row["away_score"]:
                points.append(0.5)
            else:
                points.append(0)
        else:
            gs = float(row["away_score"])
            gc = float(row["home_score"])
            if row["away_score"] > row["home_score"]:
                points.append(1)
                wins += 1
            elif row["home_score"] == row["away_score"]:
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

def get_ranking(rankings, team):
    team_ranks = rankings[rankings["country_full"] == team]
    if len(team_ranks) == 0:
        return 100
    return team_ranks.sort_values("rank_date").iloc[-1]["rank"]

def get_elo(elo, team):
    team_elo = elo[elo["team"] == team]
    if len(team_elo) == 0:
        return 1500
    return team_elo.sort_values("date").iloc[-1]["rating"]

def get_head_to_head(df_results, home_team, away_team, n=10):
    h2h = df_results[
        (
            ((df_results["home_team"] == home_team) & (df_results["away_team"] == away_team)) |
            ((df_results["home_team"] == away_team) & (df_results["away_team"] == home_team))
        ) &
        (df_results["home_score"].notna())
    ].tail(n)

    if len(h2h) == 0:
        return 0.5

    home_wins = 0
    for _, row in h2h.iterrows():
        if row["home_team"] == home_team and row["home_score"] > row["away_score"]:
            home_wins += 1
        elif row["away_team"] == home_team and row["away_score"] > row["home_score"]:
            home_wins += 1

    return home_wins / len(h2h)

def get_wc_performance(df_results, team):
    wc_matches = df_results[
        (df_results["tournament"] == "FIFA World Cup") &
        ((df_results["home_team"] == team) | (df_results["away_team"] == team)) &
        (df_results["home_score"].notna())
    ]
    if len(wc_matches) == 0:
        return 0.5
    points = []
    for _, row in wc_matches.iterrows():
        if row["home_team"] == team:
            if row["home_score"] > row["away_score"]:
                points.append(1)
            elif row["home_score"] == row["away_score"]:
                points.append(0.5)
            else:
                points.append(0)
        else:
            if row["away_score"] > row["home_score"]:
                points.append(1)
            elif row["home_score"] == row["away_score"]:
                points.append(0.5)
            else:
                points.append(0)
    return np.mean(points)

def get_wc_team_stats(wc_stats, team):
    row = wc_stats[wc_stats["team_name"] == team]
    if len(row) == 0:
        return 2.0, 0.1, 5.0, 50.0
    yc = row["avg_yellow_cards"].values[0]
    rc = row["avg_red_cards"].values[0]
    sot = row["avg_shots_on_target"].values[0]
    poss = row["avg_possession"].values[0]
    return (
        yc,
        rc,
        sot if not np.isnan(sot) else 5.0,
        poss if not np.isnan(poss) else 50.0
    )

st.set_page_config(page_title="2026 World Cup Predictor", page_icon="⚽")
st.title("⚽ 2026 FIFA World Cup Predictor")
st.caption("Select two teams to predict the match outcome")

model = load_model()
df_results, teams, rankings, elo, wc_stats = load_data()

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Home Team", teams, index=teams.index("England"))
with col2:
    away_team = st.selectbox("Away Team", teams, index=teams.index("France"))

if home_team == away_team:
    st.warning("Please select two different teams.")
else:
    if st.button("Predict Match Outcome", type="primary"):
        with st.spinner("Analysing team form and ratings..."):
            h_form, h_gf, h_ga, h_wr = get_team_stats(df_results, home_team)
            a_form, a_gf, a_ga, a_wr = get_team_stats(df_results, away_team)

            h_rank = get_ranking(rankings, home_team)
            a_rank = get_ranking(rankings, away_team)

            h_elo = get_elo(elo, home_team)
            a_elo = get_elo(elo, away_team)

            h2h_home = get_head_to_head(df_results, home_team, away_team)
            h2h_away = get_head_to_head(df_results, away_team, home_team)

            h_wc = get_wc_performance(df_results, home_team)
            a_wc = get_wc_performance(df_results, away_team)

            h_yc, h_rc, h_sot, h_poss = get_wc_team_stats(wc_stats, home_team)
            a_yc, a_rc, a_sot, a_poss = get_wc_team_stats(wc_stats, away_team)

            features = np.array([[
                h_form, a_form, h_form - a_form,
                h_gf, a_gf, h_ga, a_ga,
                h_wr, a_wr, h_wr - a_wr,
                h_gf - a_gf,
                h_rank, a_rank, a_rank - h_rank,
                h2h_home, h2h_away,
                0,
                h_wc, a_wc, h_wc - a_wc,
                h_elo, a_elo, h_elo - a_elo,
                h_yc, a_yc,
                h_rc, a_rc,
                h_sot, a_sot,
                h_poss, a_poss,
                h_sot - a_sot,
                h_poss - a_poss
            ]])

            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            classes = model.classes_

            prob_dict = dict(zip(classes, probabilities))
            away_prob = prob_dict.get(-1, 0)
            draw_prob = prob_dict.get(0, 0)
            home_prob = prob_dict.get(1, 0)

        st.markdown("---")
        st.markdown("### Prediction")

        if prediction == 1:
            st.success(f"🏆 **{home_team} Win**")
        elif prediction == -1:
            st.success(f"🏆 **{away_team} Win**")
        else:
            st.success(f"🤝 **Draw**")

        st.markdown("### Win Probabilities")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{home_team} Win", f"{home_prob:.0%}")
        with col2:
            st.metric("Draw", f"{draw_prob:.0%}")
        with col3:
            st.metric(f"{away_team} Win", f"{away_prob:.0%}")

        st.markdown("### Team Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{home_team}**")
            st.write(f"Elo Rating: {int(h_elo)}")
            st.write(f"FIFA Ranking: #{int(h_rank)}")
            st.write(f"Form score: {h_form:.2f}")
            st.write(f"Avg goals scored: {h_gf:.2f}")
            st.write(f"Avg goals conceded: {h_ga:.2f}")
            st.write(f"Win rate: {h_wr:.0%}")
            st.write(f"H2H win rate: {h2h_home:.0%}")
            st.write(f"World Cup win rate: {h_wc:.0%}")
            st.write(f"Avg shots on target: {h_sot:.1f}")
            st.write(f"Avg possession: {h_poss:.1f}%")
        with col2:
            st.markdown(f"**{away_team}**")
            st.write(f"Elo Rating: {int(a_elo)}")
            st.write(f"FIFA Ranking: #{int(a_rank)}")
            st.write(f"Form score: {a_form:.2f}")
            st.write(f"Avg goals scored: {a_gf:.2f}")
            st.write(f"Avg goals conceded: {a_ga:.2f}")
            st.write(f"Win rate: {a_wr:.0%}")
            st.write(f"H2H win rate: {h2h_away:.0%}")
            st.write(f"World Cup win rate: {a_wc:.0%}")
            st.write(f"Avg shots on target: {a_sot:.1f}")
            st.write(f"Avg possession: {a_poss:.1f}%")
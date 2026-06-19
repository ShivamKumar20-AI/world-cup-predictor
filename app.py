import streamlit as st
import pickle
import numpy as np
import pandas as pd

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_teams():
    results = pd.read_csv("data/results.csv")
    teams = sorted(set(results["home_team"].unique()) | set(results["away_team"].unique()))
    return teams

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

st.set_page_config(page_title="2026 World Cup Predictor", page_icon="⚽")
st.title("⚽ 2026 FIFA World Cup Predictor")
st.caption("Select two teams to predict the match outcome")

model = load_model()
teams = load_teams()
df_results = pd.read_csv("data/results.csv")

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Home Team", teams, index=teams.index("England"))
with col2:
    away_team = st.selectbox("Away Team", teams, index=teams.index("France"))

if home_team == away_team:
    st.warning("Please select two different teams.")
else:
    if st.button("Predict Match Outcome", type="primary"):
        with st.spinner("Analysing team form..."):
            h_form, h_gf, h_ga, h_wr = get_team_stats(df_results, home_team)
            a_form, a_gf, a_ga, a_wr = get_team_stats(df_results, away_team)

            features = np.array([[
                h_form, a_form, h_form - a_form,
                h_gf, a_gf, h_ga, a_ga,
                h_wr, a_wr, h_wr - a_wr,
                h_gf - a_gf
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

        st.markdown("### Team Form (Last 10 Games)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{home_team}**")
            st.write(f"Form score: {h_form:.2f}")
            st.write(f"Avg goals scored: {h_gf:.2f}")
            st.write(f"Avg goals conceded: {h_ga:.2f}")
            st.write(f"Win rate: {h_wr:.0%}")
        with col2:
            st.markdown(f"**{away_team}**")
            st.write(f"Form score: {a_form:.2f}")
            st.write(f"Avg goals scored: {a_gf:.2f}")
            st.write(f"Avg goals conceded: {a_ga:.2f}")
            st.write(f"Win rate: {a_wr:.0%}")
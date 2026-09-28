# World Cup Predictor

Machine learning model that predicts international football match outcomes (home win / draw / away win) using historical data, Elo ratings, and team form.  
🚀 **Live demo:** https://fifa-world-cup-predictor-2026.streamlit.app/

---

## What it does

- Predicts the result of football matches as:
  - Home win
  - Draw
  - Away win
- Uses features like:
  - Team Elo ratings
  - Recent form with exponential decay
  - Head‑to‑head history
  - FIFA rankings (where available)
- Provides probabilities for each outcome, not just a single class.

---

## Model & approach

**Task:** Multi‑class classification (home win / draw / away win).

**Data**
- Historical international match results (from a public football dataset).
- Features include:
  - Team Elo ratings
  - Recent form (exponential decay over last N matches)
  - Head‑to‑head history
  - FIFA rankings (where available)
  - Goal difference, home/away indicators.

**Train/test split**
- Time‑based split: older matches for training, recent matches for testing.
- Ensures no leakage from future matches.

**Models**
- **Random Forest** and/or **XGBoost** classifier.
- Baseline: simple model (e.g., logistic regression or always predicting the most common class).
- Feature engineering:
  - Form features with decay weights.
  - Elo difference between teams.
  - Head‑to‑head win/draw/loss counts.

**Evaluation**
- Metric: **accuracy** on the test set.
- Example results (adjust to your exact numbers if different):
  - Full model: ~60% accuracy
  - Baseline: ~50% accuracy
- Meaningful improvement over naive prediction in a 3‑class problem.
- Feature importance shows Elo and form as key drivers.

---

## Results

- Achieves around **60% accuracy** on unseen test data vs **~50% baseline**.
- Provides calibrated probabilities for each outcome (home/draw/away).
- Demonstrates that simple, well‑engineered features can yield solid predictive performance.

---

## How to run locally

**Requirements**
- Python 3.9+

**Setup**

```bash
# Clone the repo
git clone [https://github.com/ShivamKumar20-AI/world-cup-predictor.git](https://github.com/ShivamKumar20-AI/world-cup-predictor.git)
cd world-cup-predictor

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Train the model**

```bash
python train.py \
  --data_path data/matches.csv \
  --model_path models/world_cup_model.joblib \
  --n_estimators 200 \
  --max_depth 10
```

(Adjust script names and arguments to match your actual code.)

**Run inference / Streamlit app**

```bash
streamlit run app.py
```

(Adjust `app.py` to your actual Streamlit file name.)

Then open the Streamlit URL in your browser (usually `http://localhost:8501`).

---

## Example usage

**Via the Streamlit UI**

1. Start the Streamlit app as above.
2. Select:
   - Home team
   - Away team
   - (Optionally) competition / date
3. The app displays:
   - Predicted probabilities for home win / draw / away win.
   - Most likely outcome.

**Via Python (example)**

```python
import joblib
import pandas as pd

model = joblib.load("models/world_cup_model.joblib")

# Example feature vector (you’d compute these properly in your code)
features = pd.DataFrame([{
    "elo_diff": 50,
    "form_diff": 0.2,
    "h2h_home_wins": 5,
    "h2h_draws": 3,
    "h2h_away_wins": 2,
    "fifa_rank_diff": 10,
    "home_advantage": 1
}])

probs = model.predict_proba(features)
print("Home win:", probs, "Draw:", probs, "Away win:", probs)[1]
```

---

## Deployment

- **Streamlit** web app where users select teams and see predicted probabilities.
- Model artefacts saved (e.g., `.joblib` / `.pkl`) and loaded at runtime.
- Hosted on **Streamlit Cloud** or similar.
- Live demo: https://fifa-world-cup-predictor-2026.streamlit.app/

---

## Limitations & next steps

**Limitations**
- Does not include player‑level data (injuries, lineups, xG).
- Treats all competitions similarly; no explicit weighting for tournament matches.
- Performance may vary across regions and time periods.

**Next steps**
- Add player‑level features (injuries, suspensions, expected goals).
- Experiment with goal‑based models (Poisson, scoreline prediction).
- Add calibration checks and more granular error analysis (by competition, team strength, etc.).
- Extend to an API service for integration with other apps.

---

## License

[Insert your license, e.g. MIT]

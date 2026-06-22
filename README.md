# ⚽ 2026 FIFA World Cup Predictor

An ML-powered match outcome predictor for the 2026 FIFA World Cup. Select any two international teams and get a Win/Draw/Loss prediction with confidence probabilities, powered by an XGBoost classifier trained on 25,000+ international matches.

### 🔗 Live Demo
https://fifa-world-cup-predictor-2026.streamlit.app/

### 📊 Model Performance
| Features Added | Accuracy |
|---|---|
| Form, goals, win rate | 50.58% |
| + FIFA Rankings | 56.93% |
| + Head-to-head record | 56.67% |
| + Neutral venue + WC performance | 56.89% |
| + Elo ratings | 57.79% |
| + WC shots, possession, cards | 58.03% |
| + Time-based split (honest evaluation) | 59.72% |
| + XGBoost + time-decay form weighting | **60.02%** |

### 🧠 How It Works
- **Data Prep:** Historical match data (2000–2026) is cleaned and features (form, goals, rankings, Elo) are engineered with exponential time-decay weighting so recent matches count more than older ones.
- **Model:** An XGBoost classifier trained on 20,000+ matches using a time-based split — trained on matches before 2021, tested on 2021–2026 to prevent data leakage.
- **Inference:** Live predictions pull the latest team stats, passing them through the model to display Win/Draw/Loss probabilities in a Streamlit UI.

### 📦 Data Sources
- International match results (Kaggle: martj42/international-football-results)
- FIFA Rankings (Kaggle: cashncarry/fifaworldranking)
- Elo Ratings (Kaggle: saifalnimri/international-football-elo-ratings)
- World Cup match stats (Kaggle: isfakiqbalchowdhuruy/fifa-mens-world-cup-dataset-1970-2022)

### 🔍 Features Used
Recent form (time-decay weighted), average goals, win rate, FIFA/Elo rankings, head-to-head records, neutral venue status, World Cup historical stats, shots on target, possession, and card discipline.

### 🛠️ Tech Stack
- **ML:** XGBoost, scikit-learn
- **Data:** pandas, numpy
- **UI:** Streamlit
- **Data Access:** Kaggle API

### 🚀 Run Locally
1. Clone the repository and install requirements:
   `git clone https://github.com/ShivamKumar20-AI/world-cup-predictor.git`
   `pip install -r requirements.txt`
2. Download datasets via Kaggle API to the `data/` directory.
3. Execute scripts:
   `python prepare_data.py`
   `python train_model.py`
   `streamlit run app.py`

### 👤 Author
**Shivam Kumar** — AI & ML Engineer | Python Automation | NLP

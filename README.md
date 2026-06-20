# ⚽ 2026 FIFA World Cup Predictor

An ML-powered match outcome predictor for the 2026 FIFA World Cup. Select any two international teams and get a Win/Draw/Loss prediction with confidence probabilities, powered by a Random Forest classifier trained on 25,000+ international matches.

## 🔗 Live Demo
[Try it here]
((https://fifa-world-cup-predictor-2026.streamlit.app/))

## 📊 Model Performance
| Features Added | Accuracy |
|---------------|----------|
| Form, goals, win rate | 50.58% |
| + FIFA Rankings | 56.93% |
| + Head-to-head record | 56.67% |
| + Neutral venue + WC performance | 56.89% |
| + Elo ratings | 57.79% |
| + WC shots, possession, cards | **58.03%** |

## 🧠 How It Works
1. Historical international match data (2000–2026) is loaded and cleaned
2. Features are engineered for each match including form, goals, rankings and Elo ratings
3. A Random Forest classifier is trained on 20,000+ matches
4. For live predictions, the latest stats are pulled for each team and passed to the model
5. Win/Draw/Loss probabilities are displayed in the Streamlit UI

## 📦 Data Sources
- **International match results** — Kaggle (martj42/international-football-results)
- **FIFA Rankings** — Kaggle (cashncarry/fifaworldranking)
- **Elo Ratings** — Kaggle (saifalnimri/international-football-elo-ratings)
- **World Cup match stats** — Kaggle (isfakiqbalchowdhuruy/fifa-mens-world-cup-dataset-1970-2022)

## 🔍 Features Used
- Recent form score (tournament weighted)
- Average goals scored and conceded
- Win rate (last 10 matches)
- FIFA ranking
- Elo rating
- Head-to-head win rate
- Neutral venue flag
- World Cup historical win rate
- Average shots on target
- Average possession
- Average yellow/red cards

## 🛠️ Tech Stack
- **scikit-learn** — Random Forest classifier
- **pandas / numpy** — Data processing and feature engineering
- **Streamlit** — Web UI
- **Kaggle API** — Dataset downloads

## 🚀 Run Locally
```bash
git clone https://github.com/ShivamKumar20-AI/world-cup-predictor.git
cd world-cup-predictor
conda create -n world-cup python=3.10 -y
conda activate world-cup
pip install -r requirements.txt
```

Download datasets via Kaggle API:
```bash
kaggle datasets download -d martj42/international-football-results-from-1872-to-2017 -p data/ --unzip
kaggle datasets download -d cashncarry/fifaworldranking -p data/ --unzip
kaggle datasets download -d saifalnimri/international-football-elo-ratings -p data/ --unzip
kaggle datasets download -d isfakiqbalchowdhuruy/fifa-mens-world-cup-dataset-1970-2022 -p data/ --unzip
```

Then build features and train the model:
```bash
python prepare_data.py
python train_model.py
streamlit run app.py
```

## 👤 Author
**Shivam Kumar** — AI & ML Engineer | Python Automation | NLP
[GitHub](https://github.com/ShivamKumar20-AI) · [LinkedIn](https://linkedin.com/in/shivam-kumar-554395239)

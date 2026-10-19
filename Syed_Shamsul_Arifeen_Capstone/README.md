# CivicShield — AI Toxic Comment Moderation Platform

## Problem and target users
Community moderators and small online platforms receive harmful comments faster than humans can consistently review them. CivicShield gives them a toxicity-risk signal to prioritize **human review**, not to automatically ban users.

## Internship-skills evidence
| Skill | Capstone evidence |
|---|---|
| Week 5 ML foundations | Logistic regression, train/test split, accuracy and F1 evaluation |
| Week 6 NLP | TF-IDF unigrams and bigrams for toxicity classification |
| Week 7 deployment | FastAPI service, Dockerfile, endpoint test, 200-request load test |
| Week 8 MLOps | prediction metadata, user feedback, error-rate monitoring, retraining workflow |

## Dataset
Download Jigsaw Toxic Comment Classification data. Create `data/comments.csv` with columns `text,toxic`, where `toxic` is 0 or 1. Do not commit the dataset or show harmful text unnecessarily.

## Run
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py
uvicorn app.main:app --reload
# second terminal
streamlit run frontend.py
```

## Testing, deployment, and monitoring
Run `pytest` and `python load_test/concurrent_test.py`. Deploy the Docker API to Render/Railway and `frontend.py` to Streamlit Community Cloud; set `API_URL` to the public backend URL. The API stores only timestamp, prediction, probability, length, and voluntary feedback—not comment text. Review `/monitoring/summary` weekly and retrain only after validating a new versioned dataset.

## Honest limitations
Historical English data can miss context, sarcasm, new abuse patterns, coded language, Urdu/Roman Urdu, and unequal errors across groups. Predictions are decision support for moderators, never final decisions.

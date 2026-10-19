# CivicShield — 10–12 minute presentation
1. Title and public demo links (20 seconds).
2. Problem: harmful comments overwhelm community moderators; target small online communities (1 minute).
3. Internship journey: show the Week 5–8 skills table in README (1 minute).
4. Dataset, cleaning, duplicates, and stratified split; explain safe handling of harmful text (1 minute).
5. TF-IDF + balanced logistic regression and why it is a transparent baseline (1 minute).
6. Architecture: User → Streamlit → FastAPI → ML pipeline → result/feedback → SQLite monitoring (1 minute).
7. Results: insert real accuracy, toxic F1, confusion matrix, and two honest errors after training (1 minute).
8. Live demo: one civil and one toxic comment; then feedback and `/monitoring/summary` (2 minutes).
9. Deployment: Docker, FastAPI docs, Streamlit URL, tests and 200-request load-test results (1 minute).
10. Limitations and future work: context, fairness, multilingual support, reviewer workflows and drift review (1 minute).

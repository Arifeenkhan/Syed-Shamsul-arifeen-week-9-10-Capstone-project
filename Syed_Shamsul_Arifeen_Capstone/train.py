import json
from pathlib import Path
import joblib,pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
R=Path(__file__).parent; d=pd.read_csv(R/'data/comments.csv').dropna(subset=['text','toxic']).drop_duplicates('text'); d.text=d.text.astype(str).str.strip(); d=d[d.text.ne('')]
xtr,xte,ytr,yte=train_test_split(d.text,d.toxic.astype(int),test_size=.2,stratify=d.toxic,random_state=42)
p=Pipeline([('tfidf',TfidfVectorizer(stop_words='english',ngram_range=(1,2),min_df=2)),('model',LogisticRegression(max_iter=2000,class_weight='balanced',random_state=42))]); p.fit(xtr,ytr); pred=p.predict(xte)
metrics={'training_rows':len(xtr),'test_rows':len(xte),'accuracy':round(float(accuracy_score(yte,pred)),4),'toxic_f1':round(float(f1_score(yte,pred)),4),'confusion_matrix':confusion_matrix(yte,pred).tolist()}; joblib.dump(p,R/'model/toxic_comment_pipeline.joblib'); (R/'reports/metrics.json').write_text(json.dumps(metrics,indent=2)); (R/'reports/classification_report.txt').write_text(classification_report(yte,pred)); print(metrics)

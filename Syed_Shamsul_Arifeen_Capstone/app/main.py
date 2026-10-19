from pathlib import Path
from datetime import datetime, timezone
import sqlite3, joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT=Path(__file__).resolve().parents[1]; MODEL=ROOT/'model/toxic_comment_pipeline.joblib'; DB=ROOT/'monitoring.db'
app=FastAPI(title='CivicShield Toxicity API',version='1.0.0'); pipeline=None
class Comment(BaseModel): text:str=Field(min_length=1,max_length=5000)
class Feedback(BaseModel): correct:bool
def connection():
 c=sqlite3.connect(DB); c.execute('CREATE TABLE IF NOT EXISTS predictions(id INTEGER PRIMARY KEY, created_at TEXT, label TEXT, probability REAL, text_length INTEGER, feedback TEXT)'); return c
@app.on_event('startup')
def startup():
 global pipeline
 connection().close()
 if MODEL.exists(): pipeline=joblib.load(MODEL)
@app.get('/health')
def health(): return {'status':'healthy' if pipeline else 'model_missing','model_loaded':bool(pipeline)}
@app.post('/predict')
def predict(item:Comment):
 if pipeline is None: raise HTTPException(503,'Run train.py first')
 probability=float(pipeline.predict_proba([item.text])[0][1]); label='toxic' if probability>=.5 else 'non-toxic'
 with connection() as c: cursor=c.execute('INSERT INTO predictions(created_at,label,probability,text_length) VALUES(?,?,?,?)',(datetime.now(timezone.utc).isoformat(),label,probability,len(item.text))); ident=cursor.lastrowid
 return {'label':label,'toxicity_probability':probability,'prediction_id':ident,'model_version':'tfidf-logreg-v1'}
@app.post('/feedback/{ident}')
def feedback(ident:int,item:Feedback):
 with connection() as c: result=c.execute('UPDATE predictions SET feedback=? WHERE id=?',('correct' if item.correct else 'incorrect',ident))
 if not result.rowcount: raise HTTPException(404,'Prediction not found')
 return {'status':'saved'}
@app.get('/monitoring/summary')
def summary():
 with connection() as c:
  total=c.execute('SELECT count(*) FROM predictions').fetchone()[0]; labels=dict(c.execute('SELECT label,count(*) FROM predictions GROUP BY label').fetchall()); feedback=dict(c.execute('SELECT feedback,count(*) FROM predictions WHERE feedback IS NOT NULL GROUP BY feedback').fetchall())
 return {'total_predictions':total,'label_counts':labels,'feedback_counts':feedback,'observed_error_rate':round(feedback.get('incorrect',0)/max(sum(feedback.values()),1),4)}

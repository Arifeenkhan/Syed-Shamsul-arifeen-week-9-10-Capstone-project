
from pathlib import Path
from datetime import datetime, timezone
import logging
import sqlite3

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model" / "toxic_comment_pipeline.joblib"
DB = ROOT / "monitoring.db"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("civicshield")

app = FastAPI(
    title="CivicShield Toxicity API",
    version="1.0.0",
    description="AI-powered toxic comment classification API",
)

pipeline = None


# --------------------------------------------------
# Request models
# --------------------------------------------------

class Comment(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


class Feedback(BaseModel):
    correct: bool


# --------------------------------------------------
# Database
# --------------------------------------------------

def connection():
    conn = sqlite3.connect(DB)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY,
            created_at TEXT,
            label TEXT,
            probability REAL,
            text_length INTEGER,
            feedback TEXT
        )
        """
    )
    return conn


# --------------------------------------------------
# Startup: load trained model
# --------------------------------------------------

@app.on_event("startup")
def startup():
    global pipeline

    with connection():
        pass

    logger.info("Project root: %s", ROOT)
    logger.info("Expected model path: %s", MODEL)
    logger.info("Model file exists: %s", MODEL.is_file())

    if not MODEL.is_file():
        pipeline = None
        logger.error(
            "Trained model is missing. Expected file: %s", MODEL
        )
        return

    try:
        pipeline = joblib.load(MODEL)

        if not hasattr(pipeline, "predict_proba"):
            pipeline = None
            logger.error(
                "Loaded model does not support predict_proba()."
            )
            return

        logger.info("CivicShield model loaded successfully.")

    except Exception:
        pipeline = None
        logger.exception("Failed to load CivicShield model.")


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "CivicShield Toxicity API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    loaded = pipeline is not None

    return {
        "status": "healthy" if loaded else "model_missing",
        "model_loaded": loaded,
        "model_path": str(MODEL),
    }


# --------------------------------------------------
# Toxicity prediction
# --------------------------------------------------

@app.post("/predict")
def predict(item: Comment):
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check the server logs and model file.",
        )

    try:
        probabilities = pipeline.predict_proba([item.text])[0]
        classes = list(pipeline.classes_)

        # Identify the probability belonging to the toxic class.
        toxic_indices = [
            i for i, cls in enumerate(classes)
            if str(cls).strip().lower() in {
                "1", "toxic", "true", "yes"
            }
        ]

        if not toxic_indices:
            raise ValueError(
                f"Cannot identify toxic class in model classes: {classes}"
            )

        probability = float(probabilities[toxic_indices[0]])
        label = "toxic" if probability >= 0.5 else "non-toxic"

        with connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO predictions
                (created_at, label, probability, text_length)
                VALUES (?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    label,
                    probability,
                    len(item.text),
                ),
            )
            prediction_id = cursor.lastrowid

        logger.info(
            "Prediction recorded: id=%s label=%s",
            prediction_id,
            label,
        )

        return {
            "label": label,
            "toxicity_probability": probability,
            "prediction_id": prediction_id,
            "model_version": "tfidf-logreg-v1",
        }

    except HTTPException:
        raise
    except Exception:
        logger.exception("Prediction failed.")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check server logs.",
        )


# --------------------------------------------------
# User feedback
# --------------------------------------------------

@app.post("/feedback/{ident}")
def feedback(ident: int, item: Feedback):
    with connection() as conn:
        cursor = conn.execute(
            """
            UPDATE predictions
            SET feedback = ?
            WHERE id = ?
            """,
            (
                "correct" if item.correct else "incorrect",
                ident,
            ),
        )

        updated = cursor.rowcount

    if not updated:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found",
        )

    return {"status": "saved", "prediction_id": ident}


# --------------------------------------------------
# Monitoring summary
# --------------------------------------------------

@app.get("/monitoring/summary")
def summary():
    with connection() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM predictions"
        ).fetchone()[0]

        labels = dict(
            conn.execute(
                """
                SELECT label, COUNT(*)
                FROM predictions
                GROUP BY label
                """
            ).fetchall()
        )

        feedback_counts = dict(
            conn.execute(
                """
                SELECT feedback, COUNT(*)
                FROM predictions
                WHERE feedback IS NOT NULL
                GROUP BY feedback
                """
            ).fetchall()
        )

    feedback_total = sum(feedback_counts.values())
    incorrect = feedback_counts.get("incorrect", 0)

    return {
        "total_predictions": total,
        "label_counts": labels,
        "feedback_counts": feedback_counts,
        # This measures negative feedback, not independently verified model error.
        "observed_error_rate": round(
            incorrect / feedback_total, 4
        ) if feedback_total else None,
    }

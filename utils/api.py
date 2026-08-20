"""
utils/api.py

FastAPI layer for EduLeap (proposal 4: BACKEND API - authentication,
sessions, learner data). Thin - all real logic lives in orchestrator.py and
the other utils modules.

Run:
    uvicorn utils.api:app --reload
"""

import logging
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from Utils.database import init_db, get_db, LearnerProfile
from Utils import learner as learner_store
from Utils.orchestrator import orchestrator
from Utils.rag import index_knowledge_base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eduleap.api")

app = FastAPI(title="EduLeap API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()
    try:
        index_knowledge_base()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Knowledge base indexing skipped/failed at startup: %s", exc)


# ---------- request/response schemas ----------

class CreateStudentRequest(BaseModel):
    display_name: str


class DiagnosticSubmitRequest(BaseModel):
    student_id: str
    topic: str
    responses: list[dict]  # [{"question","targets","correct"}]


class TeachRequest(BaseModel):
    student_id: str
    topic: str
    struggling: bool = False
    prior_explanation_summary: Optional[str] = None


class AskQuestionRequest(BaseModel):
    student_id: str
    topic: str


class SubmitAnswerRequest(BaseModel):
    student_id: str
    topic: str
    question: str
    correct_answer: str
    student_answer: str


# ---------- routes ----------

@app.post("/students")
def create_student(req: CreateStudentRequest, db: Session = Depends(get_db)):
    student = learner_store.get_or_create_student(db, req.display_name)
    return {"student_id": student.id, "display_name": student.display_name}


@app.get("/profile/{student_id}/{topic}")
def get_profile(student_id: str, topic: str, db: Session = Depends(get_db)):
    profile: LearnerProfile = learner_store.get_or_create_profile(db, student_id, topic)
    return {
        "topic": profile.topic,
        "level": profile.level,
        "known_gaps": profile.known_gaps,
        "misconception_log": profile.misconception_log,
        "human_review_flags": learner_store.recurring_misconceptions(profile),
    }


@app.get("/diagnostic/{topic}")
def start_diagnostic(topic: str):
    """Returns a short diagnostic quiz for the frontend to render."""
    questions = orchestrator.start_diagnostic(topic)
    if not questions:
        raise HTTPException(status_code=502, detail="Failed to generate diagnostic questions.")
    return {"topic": topic, "questions": questions}


@app.post("/diagnostic/submit")
def submit_diagnostic(req: DiagnosticSubmitRequest, db: Session = Depends(get_db)):
    return orchestrator.submit_diagnostic(db, req.student_id, req.topic, req.responses)


@app.post("/tutor/teach")
def teach(req: TeachRequest, db: Session = Depends(get_db)):
    return orchestrator.teach(
        db, req.student_id, req.topic, req.struggling, req.prior_explanation_summary
    )


@app.post("/assessment/question")
def ask_question(req: AskQuestionRequest, db: Session = Depends(get_db)):
    q = orchestrator.ask_question(db, req.student_id, req.topic)
    if "error" in q:
        raise HTTPException(status_code=502, detail="Failed to generate a question.")
    # Do not leak correct_answer to the client in a real deployment; kept
    # here for hackathon demo simplicity / grading transparency.
    return q


@app.post("/assessment/answer")
def submit_answer(req: SubmitAnswerRequest, db: Session = Depends(get_db)):
    return orchestrator.submit_answer(
        db, req.student_id, req.topic, req.question, req.correct_answer, req.student_answer
    )


@app.get("/next-action/{student_id}/{topic}")
def next_action(student_id: str, topic: str, db: Session = Depends(get_db)):
    return orchestrator.get_next_action(db, student_id, topic)


@app.get("/health")
def health():
    return {"status": "ok"}

from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import date
from typing import Literal

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from Ensemble import explain_underwriting_shap
from Inference import UnderwritingPredictor


ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoint"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_ROWS = 10000
predictor = None
checkpoint_error = None


class ApplicationRequest(BaseModel):
    application_id: str = Field(default="LIVE-001", max_length=80)
    employment_type: Literal["Salaried", "Self employed"]
    disbursal_date: date
    disbursed_amount: float = Field(ge=0, le=100_000_000)
    asset_cost: float = Field(ge=0, le=100_000_000)
    ltv: float = Field(ge=0, le=200)
    cibil_score: int = Field(ge=0, le=900)
    average_age_years: int = Field(ge=0, le=100)
    average_age_months: int = Field(ge=0, le=11)
    credit_history_years: int = Field(ge=0, le=100)
    credit_history_months: int = Field(ge=0, le=11)
    primary_active_accounts: int = Field(ge=0, le=1000)
    secondary_active_accounts: int = Field(ge=0, le=1000)
    primary_overdue_accounts: int = Field(ge=0, le=1000)
    secondary_overdue_accounts: int = Field(ge=0, le=1000)
    inquiries: int = Field(ge=0, le=1000)
    recent_delinquencies: int = Field(ge=0, le=1000)
    portfolio_gnpa: float = Field(default=0.02, ge=0, le=1)
    psl_share: float = Field(default=0.0, ge=0, le=1)


def get_predictor():
    if predictor is None:
        detail = checkpoint_error or "Train a model with Main.py before requesting predictions."
        raise HTTPException(status_code=503, detail=detail)
    return predictor


def to_frame(application: ApplicationRequest):
    return pd.DataFrame(
        [
            {
                "UNIQUEID": application.application_id,
                "EMPLOYMENT_TYPE": application.employment_type,
                "DISBURSAL_DATE": application.disbursal_date.strftime("%d-%m-%Y"),
                "DISBURSED_AMOUNT": application.disbursed_amount,
                "ASSET_COST": application.asset_cost,
                "LTV": application.ltv,
                "PERFORM_CNS_SCORE": application.cibil_score,
                "AVERAGE_ACCT_AGE": (
                    f"{application.average_age_years}yrs "
                    f"{application.average_age_months}mon"
                ),
                "CREDIT_HISTORY_LENGTH": (
                    f"{application.credit_history_years}yrs "
                    f"{application.credit_history_months}mon"
                ),
                "PRI_ACTIVE_ACCTS": application.primary_active_accounts,
                "SEC_ACTIVE_ACCTS": application.secondary_active_accounts,
                "PRI_OVERDUE_ACCTS": application.primary_overdue_accounts,
                "SEC_OVERDUE_ACCTS": application.secondary_overdue_accounts,
                "NO_OF_INQUIRIES": application.inquiries,
                "DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS": application.recent_delinquencies,
            }
        ]
    )


def result_payload(predictions, row_index=0):
    row = predictions.iloc[row_index]
    apr = row["PROPOSED_APR_PERCENT"]
    return {
        "application_id": str(row["UNIQUEID"]),
        "decision": str(row["DECISION"]),
        "decline_q_score": float(row["DECLINE_Q_SCORE"]),
        "approve_q_score": float(row["APPROVE_Q_SCORE"]),
        "decision_margin": float(row["DECISION_MARGIN"]),
        "proposed_apr_percent": None if pd.isna(apr) else float(apr),
    }


@asynccontextmanager
async def lifespan(_app):
    global predictor, checkpoint_error
    try:
        predictor = UnderwritingPredictor.load_checkpoint(CHECKPOINT_DIR)
        checkpoint_error = None
    except Exception as error:
        predictor = None
        checkpoint_error = f"Could not load model checkpoint: {error}"
    yield


app = FastAPI(
    title="Micro-loan Underwriting Demo API",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health():
    available = predictor is not None
    return JSONResponse(
        status_code=200 if available else 503,
        content={
            "status": "ok" if available else "model_unavailable",
            "model_loaded": available,
            "explanations_available": bool(
                available and predictor.background_samples is not None
            ),
            "message": checkpoint_error,
        },
    )


@app.post("/api/predict")
def predict(application: ApplicationRequest):
    model = get_predictor()
    try:
        predictions, _ = model.predict_frame(
            to_frame(application),
            portfolio_gnpa=application.portfolio_gnpa,
            psl_share=application.psl_share,
        )
    except (ValueError, KeyError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return result_payload(predictions)


@app.post("/api/predict-csv")
async def predict_csv(
    file: UploadFile = File(...),
    portfolio_gnpa: float = 0.02,
    psl_share: float = 0.0,
):
    if not file.filename or Path(file.filename).suffix.lower() != ".csv":
        raise HTTPException(status_code=400, detail="Upload a CSV file.")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="CSV must be 10 MB or smaller.")
    try:
        frame = pd.read_csv(BytesIO(content))
        if len(frame) > MAX_UPLOAD_ROWS:
            raise HTTPException(status_code=413, detail="Demo uploads are limited to 10,000 rows.")
        predictions, _ = get_predictor().predict_frame(
            frame, portfolio_gnpa=portfolio_gnpa, psl_share=psl_share
        )
    except HTTPException:
        raise
    except (ValueError, KeyError, pd.errors.ParserError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    clean_predictions = predictions.astype(object).where(pd.notna(predictions), None)
    return {"count": len(predictions), "predictions": clean_predictions.to_dict(orient="records")}


@app.post("/api/explain")
def explain(application: ApplicationRequest):
    model = get_predictor()
    if model.background_samples is None:
        raise HTTPException(
            status_code=409,
            detail="This checkpoint has no SHAP background sample. Retrain with Main.py.",
        )
    frame = to_frame(application)
    try:
        predictions, observations = model.predict_frame(
            frame,
            portfolio_gnpa=application.portfolio_gnpa,
            psl_share=application.psl_share,
        )
        prediction = result_payload(predictions)
        with TemporaryDirectory() as temp_dir:
            report_path, figure = explain_underwriting_shap(
                dqn_gate=model.gatekeeper,
                background_samples=model.background_samples,
                applicant_state=observations[0],
                applicant_id=application.application_id,
                q_values=[prediction["decline_q_score"], prediction["approve_q_score"]],
                output_path=Path(temp_dir) / "explanation.html",
            )
            report_html = report_path.read_text(encoding="utf-8")
    except (ValueError, KeyError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    trace = figure.data[0]
    factors = [
        {"feature": str(name), "impact": float(impact)}
        for name, impact in zip(trace.y, trace.x)
    ]
    return {
        "prediction": prediction,
        "factors": factors,
        "report_html": report_html,
    }
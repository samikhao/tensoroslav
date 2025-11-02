from __future__ import annotations

import secrets
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Form, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates


app = FastAPI(title="Tensoroslav Game")


templates = Jinja2Templates(directory="app/templates")
MODEL = joblib.load("ridge_poly.joblib")

D = 10
INT_MIN, INT_MAX = -5, 5
STAR_MIN, STAR_MAX = -5, 5
S_MIN, S_MAX = 0.6, 2.0
B_SCALE = 1.0

SESSIONS: dict[str, dict[str, Any]] = {}


def new_session() -> str:
    sid = secrets.token_urlsafe(16)
    rng = np.random.default_rng()

    Q, _ = np.linalg.qr(rng.normal(size=(D, D)))
    b = rng.normal(loc=0.0, scale=B_SCALE, size=D)
    w_star_int = rng.integers(low=STAR_MIN, high=STAR_MAX + 1, size=D)
    scales = rng.uniform(S_MIN, S_MAX, size=D)
    S = np.diag(scales)

    def _map(wi: np.ndarray) -> np.ndarray:
        centered = wi.astype(float) - w_star_int.astype(float)
        return Q @ (S @ centered) + b

    cols = [f"W{i}" for i in range(D)]
    target_pred = float(MODEL.predict(pd.DataFrame(_map(w_star_int).reshape(1, -1), columns=cols))[0])

    SESSIONS[sid] = {
        "Q": Q, "b": b, "S": S,
        "w_star_int": w_star_int,
        "target_pred": max(target_pred, 0.0),
        "history": [],
    }
    return sid


def get_session(sid: str) -> dict[str, Any]:
    sess = SESSIONS.get(sid)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    return sess


def map_to_model_input(sess: dict, w_int: np.ndarray) -> np.ndarray:
    w_clamped = np.clip(np.rint(w_int).astype(int), INT_MIN, INT_MAX).astype(float)
    centered = w_clamped - sess["w_star_int"].astype(float)
    return sess["Q"] @ (sess["S"] @ centered) + sess["b"]


def oracle_game_mse(sess: dict, w_int: np.ndarray) -> float:
    w_int = np.clip(np.rint(w_int).astype(int), INT_MIN, INT_MAX)
    w_t = map_to_model_input(sess, w_int)
    cols = [f"W{i}" for i in range(D)]
    model_pred = float(MODEL.predict(pd.DataFrame(w_t.reshape(1, -1), columns=cols))[0])
    model_pred = max(model_pred, 0.0)

    dist2 = float(np.sum((w_int - sess["w_star_int"]) ** 2))

    K = 1500.0 / D
    BETA = 0.15

    base = sess["target_pred"]
    mse_game = base + K * dist2 + BETA * (model_pred - base)
    return float(mse_game)


@app.get("/health")
def health():
    return {"status": "ok", "sessions": len(SESSIONS)}

@app.get("/new")
def create_new():
    sid = new_session()
    return RedirectResponse(url=f"/?sid={sid}", status_code=302)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, sid: str | None = Query(default=None)):
    if not sid:
        return RedirectResponse(url="/new", status_code=302)
    _ = get_session(sid)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "sid": sid,
            "history": SESSIONS[sid]["history"],
            "int_min": INT_MIN,
            "int_max": INT_MAX,
            "D": D,
        },
    )

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    sid: str = Form(...),
    w0: float = Form(...), w1: float = Form(...), w2: float = Form(...),
    w3: float = Form(...), w4: float = Form(...), w5: float = Form(...),
    w6: float = Form(...), w7: float = Form(...), w8: float = Form(...),
    w9: float = Form(...),
):
    sess = get_session(sid)
    try:
        w = np.array([w0, w1, w2, w3, w4, w5, w6, w7, w8, w9], dtype=float)
        w = np.clip(np.rint(w), INT_MIN, INT_MAX).astype(int)

        mse_pred = oracle_game_mse(sess, w)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    sess["history"].insert(0, {
        "w": [int(x) for x in w.tolist()],
        "mse": round(mse_pred, 6),
    })

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "sid": sid,
            "history": sess["history"],
            "last_mse": mse_pred,
            "int_min": INT_MIN,
            "int_max": INT_MAX,
            "D": D,
        },
    )

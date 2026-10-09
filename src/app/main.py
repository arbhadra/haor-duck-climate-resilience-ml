"""FastAPI service with a Gradio input form mounted at /ui."""
import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel, Field

from serving.model import load_model, predict

app = FastAPI(title="Haor duck farm flood-resilience predictor")
_model = None


def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


class Farmer(BaseModel):
    age: int = Field(ge=18, le=80)
    gender: str = Field(pattern="^(Female|Male)$")
    education_years: int = Field(ge=0, le=20)
    hh_size: int = Field(ge=1, le=20)
    land_decimals: int = Field(ge=0, le=2000)
    experience_years: int = Field(ge=0, le=60)
    farming_system: str = Field(pattern="^(Free-range|Semi-intensive|Intensive)$")
    flock_size: int = Field(ge=1, le=1000)
    market_distance_km: float = Field(ge=0, le=100)
    extension_training: int = Field(ge=0, le=1)
    credit_access: int = Field(ge=0, le=1)
    group_member: int = Field(ge=0, le=1)
    n_practices: int = Field(ge=0, le=14)


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(farmer: Farmer):
    return predict(get_model(), farmer.model_dump())


def _ui(age, gender, edu, hh, land, exp, system, flock, dist, train, credit, group, n_pr):
    rec = dict(age=age, gender=gender, education_years=edu, hh_size=hh, land_decimals=land,
               experience_years=exp, farming_system=system, flock_size=flock,
               market_distance_km=dist, extension_training=int(train), credit_access=int(credit),
               group_member=int(group), n_practices=n_pr)
    r = predict(get_model(), rec)
    return f"Low flood-season mortality (<=15%)? {r['prediction']}  (probability {r['probability']:.0%})"


demo = gr.Interface(
    fn=_ui,
    inputs=[gr.Slider(18, 80, 37, step=1, label="Age"),
            gr.Radio(["Female", "Male"], value="Female", label="Gender"),
            gr.Slider(0, 16, 5, step=1, label="Education (years)"),
            gr.Slider(1, 15, 5, step=1, label="Household size"),
            gr.Number(value=60, label="Land (decimals)"),
            gr.Slider(0, 40, 10, step=1, label="Duck-farming experience (years)"),
            gr.Radio(["Free-range", "Semi-intensive", "Intensive"], value="Free-range", label="Farming system"),
            gr.Number(value=40, label="Flock size"),
            gr.Number(value=5, label="Distance to market (km)"),
            gr.Checkbox(label="Received extension training"),
            gr.Checkbox(label="Has access to credit"),
            gr.Checkbox(label="Member of a farmer group"),
            gr.Slider(0, 14, 0, step=1, label="Climate-smart practices adopted (0-14)")],
    outputs=gr.Textbox(label="Prediction"),
    title="Haor duck farm flood-resilience predictor",
    description="DEMO trained on SYNTHETIC data - for illustration only, not real advice.",
)
app = gr.mount_gradio_app(app, demo, path="/ui")

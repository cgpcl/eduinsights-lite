"""FastAPI: interfaz, inferencia y lectura de datos históricos."""
import os
from pathlib import Path
from typing import Literal
import httpx
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent
model = joblib.load(BASE / 'modelo.joblib')
app = FastAPI(title='EduInsights Lite', version='1.0')

class Student(BaseModel):
    reading_score: int = Field(ge=0, le=100)
    writing_score: int = Field(ge=0, le=100)
    test_prep: Literal['completed', 'none']

@app.get('/', include_in_schema=False)
def home():
    return FileResponse(BASE / 'index.html', media_type='text/html')

@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': True,
            'supabase_configured': bool(os.getenv('SUPABASE_URL') and os.getenv('SUPABASE_KEY'))}

@app.post('/predict')
def predict(student: Student):
    row = pd.DataFrame([student.model_dump()], columns=['reading_score', 'writing_score', 'test_prep'])
    prediction = int(model.predict(row)[0])
    probability_pass = float(model.predict_proba(row)[0][list(model.classes_).index(1)])
    return {'prediction': prediction, 'label': 'Aprueba' if prediction else 'En riesgo',
            'probability_pass': round(probability_pass, 4)}

@app.get('/students')
def students():
    url, key = os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY')
    if not url or not key:
        raise HTTPException(503, 'Configura SUPABASE_URL y SUPABASE_KEY en Render.')
    try:
        response = httpx.get(f'{url.rstrip("/")}/rest/v1/students',
            headers={'apikey': key}, params={'select': 'id,reading_score,writing_score,test_prep,math_score,pass_math',
                                             'order': 'id.asc', 'limit': 10}, timeout=10)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError):
        raise HTTPException(502, 'No fue posible consultar Supabase. Revisa URL, clave, tabla y políticas RLS.')

import io
import json
import os
import time
from pathlib import Path
import torch
import torchaudio
from fastapi import FastAPI, File, HTTPException, UploadFile
from src.model import CLASSES, SpeechCNN, preprocess

app = FastAPI(title='Speech Commands API', version='3.0.0')
MODEL_PATH = Path(os.getenv('MODEL_PATH', 'artifacts/model.pt'))
_model = None

def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError('Model artifact missing')
        _model = SpeechCNN()
        _model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=True))
        _model.eval()
    return _model

@app.get('/health')
def health():
    return {'status': 'ok'} if MODEL_PATH.exists() else {'status': 'model_missing'}

@app.get('/model-info')
def model_info():
    metrics_path = MODEL_PATH.parent / 'metrics.json'
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    return {'architecture': 'SpeechCNN', 'classes': CLASSES, **metrics}

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    if not (file.filename or '').lower().endswith('.wav'):
        raise HTTPException(400, 'WAV file required')
    try:
        data = await file.read()
        waveform, sr = torchaudio.load(io.BytesIO(data), format='wav')
        x = preprocess(waveform, sr).unsqueeze(0)
        started = time.perf_counter()
        with torch.no_grad():
            scores = torch.softmax(get_model()(x), dim=1)[0]
        latency = (time.perf_counter() - started) * 1000
        index = scores.argmax().item()
        return {'label': CLASSES[index], 'prediction': CLASSES[index],
                'confidence': round(scores[index].item(), 6),
                'latency_ms': round(latency, 3)}
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc

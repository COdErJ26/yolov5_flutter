from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import torch
from PIL import Image
import io
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "yolov5"))

app = FastAPI()

# Load the model
model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    results = model(image)

    results.render()
    buf = io.BytesIO()
    Image.fromarray(results.ims[0]).save(buf, format='JPEG')
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/jpeg")

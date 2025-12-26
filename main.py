from fastapi import FastAPI, UploadFile, File
import uvicorn
import os
import shutil
from src.pipelines.prediction_pipeline import PredictPipeline
from fastapi.responses import JSONResponse

app = FastAPI(title="Hindi Digit Classifier API")

@app.get("/")
def home():
    return {"message": "Welcome to Hindi Digit Classifier API"}

@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):
    if not file:
        return {"error": "No file uploaded"}
    
    # Save temp file
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        pipeline = PredictPipeline()
        label, confidence = pipeline.predict(temp_file)
        
        # Cleanup
        os.remove(temp_file)
        
        return JSONResponse(content={
            "filename": file.filename,
            "predicted_label": label,
            "confidence": confidence
        })
    except Exception as e:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/train")
def train_model():
    from src.pipelines.training_pipeline import TrainPipeline
    try:
        pipeline = TrainPipeline()
        pipeline.run_pipeline()
        return {"message": "Training completed successfully"}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

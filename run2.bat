@echo off
echo Installing dependencies...
uv pip install -r requirements.txt

echo.
echo ===================================================
echo To train the model, place your data in 'data/' folder
echo and run: python train.py
echo ===================================================
echo.
echo To run FastAPI: uvicorn main:app --reload
echo To run Streamlit: streamlit run app.py
echo running fastapi...
uvicorn main:app --reload
echo.
echo or usage Docker: docker-compose up --build
echo.
pause

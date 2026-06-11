<<<<<<< HEAD
<<<<<<< HEAD
# Backend Service - PPE Detection API

FastAPI-based inference service for YOLO model predictions.

## Features

- REST API for model inference
- Support for multiple model formats (ONNX, PT, TensorRT)
- Configurable confidence threshold and image size
- Health check endpoints
- Dynamic model loading
- CPU-optimized for Raspberry Pi

## Installation

```bash
pip install -r requirements.txt
```

## Running the Service

### Development Mode
```bash
python app.py
```

### Production Mode
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker
```bash
docker build -t ppe-backend .
docker run -p 8000:8000 -v $(pwd)/../models:/app/models ppe-backend
```

## API Endpoints

### GET /
Health check endpoint
```bash
curl http://localhost:8000/
```

### GET /health
Detailed health information
```bash
curl http://localhost:8000/health
```

### POST /predict
Perform inference on an image
```bash
curl -X POST "http://localhost:8000/predict?conf_threshold=0.5&imgsz=384" \
  -F "file=@image.jpg"
```

Parameters:
- `file`: Image file (required)
- `conf_threshold`: Confidence threshold (default: 0.5)
- `imgsz`: Inference image size (default: 384)
- `classes`: Comma-separated class IDs (optional)
- `model_path_param`: Path to model file (default: ../models/ppe.onnx)

### POST /load_model
Load or switch model
```bash
curl -X POST "http://localhost:8000/load_model?model_path_param=../models/yolo11n.pt"
```

## API Documentation

Interactive API documentation available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Environment Variables

- `OMP_NUM_THREADS`: Number of OpenMP threads
- `MKL_NUM_THREADS`: Number of MKL threads

## Model Support

Supported model formats:
- `.pt` - PyTorch models
- `.onnx` - ONNX models
- `.engine` - TensorRT engines

Place models in the `../models/` directory.
=======
# RVCE-PPE-backend-service
>>>>>>> 59ec922be728d842523f40bec8d81cf7bf14a7a8
=======
# RVCE-PPE-backend-service
>>>>>>> 6eeccfb351f5d3bc688d238e70967fb10b5c84c2

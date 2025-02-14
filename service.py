import io

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from typing import Optional, List, Dict, Tuple, Union
import uvicorn

from product_categorizer import ProductCategorizer


class PredictionResponse(BaseModel):
    full_path: List[Tuple[str, float]]
    level_predictions: Dict[str, List[Tuple[str, float]]]
    length_info: Optional[Dict[str, Union[str, float]]] = None

    class Config:
        arbitrary_types_allowed = True

app = FastAPI(title="Product Categorizer API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable for the categorizer
categorizer = None

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    global categorizer
    
    # Configure your model parameters here
    model_name = "ViT-L-14"
    pretrained = "/model/open_clip_pytorch_model.bin"
    csv_path = "/dataset/gsshop_fashion_sample_train_small_revision.csv"
    
    print("Initializing model...")
    categorizer = ProductCategorizer(
        model_name=model_name,
        pretrained=pretrained,
        image_weight=0.7,
        text_weight=0.3
    )
    categorizer.load_categories(csv_path)
    print("Model initialized successfully")

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    image: Optional[UploadFile] = File(None),
    base64_image: Optional[str] = Form(None),
    title: str = Form(...)
):
    """
    Predict category for a product image and title
    
    Args:
        image: Upload an image file
        base64_image: Base64 encoded image string
        title: Product title/description
    """
    if categorizer is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        # Process image
        if image:
            # Read uploaded file
            contents = await image.read()
            pil_image = Image.open(io.BytesIO(contents))
        elif base64_image:
            # Process base64 image
            pil_image = categorizer.process_base64_image(base64_image)
        else:
            raise HTTPException(status_code=400, detail="Either image file or base64 image is required")
        
        # Get predictions
        predictions = categorizer.predict_single(pil_image, title)
        
        # Convert numpy float32 to Python float for JSON serialization
        predictions['full_path'] = [
            (path, float(conf)) 
            for path, conf in predictions['full_path']
        ]
        
        for level in predictions['level_predictions']:
            predictions['level_predictions'][level] = [
                (pred, float(conf)) 
                for pred, conf in predictions['level_predictions'][level]
            ]
        
        return predictions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check if the server is running and model is loaded"""
    return {
        "status": "healthy",
        "model_loaded": categorizer is not None
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 
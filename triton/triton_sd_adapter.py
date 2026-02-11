# triton_sd_adapter.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import tritonclient.http as httpclient
import numpy as np
import base64
import time
import io
import os
from PIL import Image
from tritonclient.utils import np_to_triton_dtype

app = FastAPI()

# Configuration
TRITON_URL = os.getenv("TRITON_URL", "localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "stable_diffusion_xl")  # Read from environment


class ImageGenerationRequest(BaseModel):
    prompt: str
    model: str = MODEL_NAME
    n: int = 1
    # size is 512x512 fixed
    response_format: str = "b64_json"  # or "url"
    negative_prompt: Optional[str] = None


class ImageData(BaseModel):
    b64_json: Optional[str] = None
    url: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: List[ImageData]


@app.post("/v1/images/generations", response_model=ImageGenerationResponse)
async def generate_images(request: ImageGenerationRequest):
    """OpenAI-compatible image generation endpoint"""
    try:
        # Initialize Triton client
        triton_client = httpclient.InferenceServerClient(url=TRITON_URL)

        # Prepare prompt input
        text_obj = np.array([request.prompt], dtype="object").reshape((-1, 1))
        input_text = httpclient.InferInput(
            "prompt", text_obj.shape, np_to_triton_dtype(text_obj.dtype)
        )
        input_text.set_data_from_numpy(text_obj)

        inputs = [input_text]

        # Add negative prompt if provided
        if request.negative_prompt:
            neg_obj = np.array([request.negative_prompt], dtype="object").reshape((-1, 1))
            input_neg = httpclient.InferInput(
                "negative_prompt", neg_obj.shape, np_to_triton_dtype(neg_obj.dtype)
            )
            input_neg.set_data_from_numpy(neg_obj)
            inputs.append(input_neg)

        # Request output
        output_img = httpclient.InferRequestedOutput("generated_image")

        # Call Triton
        query_response = triton_client.infer(
            model_name=MODEL_NAME,
            inputs=inputs,
            outputs=[output_img]
        )

        # Get image data
        image_array = query_response.as_numpy("generated_image")

        # Convert to PIL Image
        img = Image.fromarray(np.squeeze(image_array.astype(np.uint8)))

        # Encode to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()

        # Return OpenAI-compatible response
        return ImageGenerationResponse(
            created=int(time.time()),
            data=[ImageData(b64_json=img_b64)]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        triton_client = httpclient.InferenceServerClient(url=TRITON_URL)
        if triton_client.is_server_live():
            return {"status": "healthy", "triton": "connected"}
    except BaseException:
        raise HTTPException(status_code=503, detail="Triton server unavailable")


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)"""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "triton"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("ADAPTER_PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)

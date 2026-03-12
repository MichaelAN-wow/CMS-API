import os
import cv2
import numpy as np
import sys
import base64
import uvicorn
import socket
import warnings
import logging
import asyncio
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# Suppress InsightFace/skimage FutureWarning (estimate deprecated) so deploy doesn't fail
warnings.filterwarnings('ignore', category=FutureWarning, message=r'.*`estimate` is deprecated.*')

# Optional: add local venv to path when present (e.g. on Windows dev); skip in Docker/CI
_venv_site = os.path.join('venv', 'Lib', 'site-packages')
if os.path.isdir(_venv_site):
    sys.path.append(_venv_site)
from insightface.app import FaceAnalysis

app = FastAPI()


@app.get("/health")
def health():
    """Railway (and other platforms) use this to confirm the service is up."""
    return {"status": "ok"}


# Lazy-load InsightFace so the server can start fast and pass health checks (e.g. on Railway).
# Model loads on first request to /api/upload_source.
_face_app = None


def get_face_app():
    global _face_app
    if _face_app is None:
        providers = {'providers': ['CPUExecutionProvider']}
        allow_names = ['detection', 'recognition']
        _face_app = FaceAnalysis(name="buffalo_l", root='./', allowed_modules=allow_names, **providers)
        _face_app.prepare(ctx_id=0, det_size=(640, 640))
    return _face_app


def handle_exception(loop, context):
    """Handle asyncio exceptions gracefully - suppress connection errors"""
    exception = context.get('exception')
    message = context.get('message', '')

    # Suppress connection-related errors (common when clients disconnect)
    if isinstance(exception, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError)):
        # Check if it's a connection error (WinError 10054, etc.)
        if hasattr(exception, 'winerror'):
            if exception.winerror in (10054, 10053, 10058):  # Common Windows connection errors
                return  # Suppress these errors
        elif hasattr(exception, 'errno'):
            if exception.errno in (104, 32, 107):  # Common Linux connection errors
                return  # Suppress these errors
        elif 'connection' in str(exception).lower() or 'reset' in str(exception).lower():
            return  # Suppress connection-related errors

    # Suppress specific callback errors
    if 'ProactorBasePipeTransport' in str(message) or '_call_connection_lost' in str(message):
        return  # Suppress these specific asyncio callback errors

    # Also check the exception message/traceback
    if exception and ('ProactorBasePipeTransport' in str(exception) or
                      '_call_connection_lost' in str(exception) or
                      'WinError 10054' in str(exception) or
                      'forcibly closed' in str(exception)):
        return

    # Log other exceptions using default handler (only if not a connection error)
    # Check one more time for connection errors in the full context
    context_str = str(context)
    if ('ProactorBasePipeTransport' in context_str or
            '_call_connection_lost' in context_str or
            'WinError 10054' in context_str or
            'ConnectionResetError' in context_str or
            'forcibly closed' in context_str):
        return  # Suppress connection errors

    # Log other exceptions using default handler
    if hasattr(loop, 'default_exception_handler') and loop.default_exception_handler:
        loop.default_exception_handler(context)
    # Otherwise, just suppress (don't print)


# Configure asyncio exception handler when starting
def configure_asyncio_handler():
    """Configure asyncio to handle connection errors gracefully"""
    try:
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(handle_exception)
    except RuntimeError:
        # Event loop not running yet - will be configured by uvicorn
        pass


# Get local IP address for network access
def get_local_ip():
    try:
        # Connect to a remote address to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"


class ImageRequest(BaseModel):
    image_base64: str
    kwargs: dict = {}  # This allows you to send arbitrary extra parameters!


@app.post("/api/upload_source")
async def upload_source(request: ImageRequest):
    """Upload source face image as base64, returning cropped face and embedding"""
    try:
        # 2. Extract the base64 string from the request
        b64_str = request.image_base64

        # Strip the data URI header if the frontend included it (e.g., "data:image/jpeg;base64,...")
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]

        # 3. Decode base64 back into bytes, then to a numpy array, then to an OpenCV image
        img_data = base64.b64decode(b64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        source_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if source_img is None:
            raise HTTPException(status_code=400, detail="Invalid image format or corrupted base64 data")

        # Detect face in source image (lazy-loads InsightFace on first call)
        try:
            face_app = get_face_app()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Face model failed to load: {str(e)}. Check server memory and logs.",
            )
        source_faces = face_app.get(source_img)

        if len(source_faces) == 0:
            raise HTTPException(status_code=400, detail="No face detected in source image")

        source_face = source_faces[0]

        # Get bounding box and crop the face
        bbox = source_face.bbox.astype(int)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

        cropped_face = source_img[
            max(0, y1):min(source_img.shape[0], y2),
            max(0, x1):min(source_img.shape[1], x2)
        ]

        # Encode the cropped face to a base64 string for the JSON response
        _, buffer = cv2.imencode('.jpg', cropped_face)
        cropped_face_base64 = base64.b64encode(buffer).decode('utf-8')

        # Extract and convert the embedding (numpy array) to a list
        embedding_list = None
        if hasattr(source_face, 'embedding') and source_face.embedding is not None:
            embedding_list = source_face.embedding.tolist()

        # Return the base64 image, embedding, and echo back the kwargs for verification
        return JSONResponse(content={
            "status": "success",
            "message": "Source face processed successfully",
            "cropped_face_base64": cropped_face_base64,
            "embedding": embedding_list,
            "received_kwargs": request.kwargs
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    local_ip = get_local_ip()

    # Console output for HTTP Server
    print("\n" + "=" * 60)
    print("Face Swap Web Application Server Starting (HTTP)...")
    print("=" * 60)
    print(f"\nServer is running!")
    print(f"\nAccess the web interface at:")
    print(f"  • Local:   http://localhost:8001")
    print(f"  • Local:   http://127.0.0.1:8001")
    print(f"  • Network: http://{local_ip}:8001")
    print(f"\nNote: Server binding to 0.0.0.0:8001 (all network interfaces)")
    print("=" * 60 + "\n")

    # Use explicit uvicorn Config and Server for better control (HTTP ONLY)
    config_params = {
        'app': app,  # <--- CHANGED THIS: Passed the app variable directly instead of the string
        'host': "0.0.0.0",
        'port': 8001,
        'log_level': "info",
        'access_log': True
    }

    config = uvicorn.Config(**config_params)
    server = uvicorn.Server(config)

    # Configure asyncio handler before running
    configure_asyncio_handler()

    # Suppress connection errors in uvicorn logging
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*ProactorBasePipeTransport.*')
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*connection.*reset.*')
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*WinError 10054.*')

    # Also suppress at the logging level
    logging.getLogger('uvicorn.error').setLevel(logging.WARNING)
    logging.getLogger('uvicorn.access').setLevel(logging.WARNING)

    # Start the server
    server.run()

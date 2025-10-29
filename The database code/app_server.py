# main.py
# deps: fastapi, uvicorn, pillow, python-multipart
# run: uvicorn main:app --reload
import json
import logging
from typing import List, Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np
import cv2
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.middleware.base import BaseHTTPMiddleware

from import_db import load_vt_model, patch_spatial_verify_for_tuples
# --- Optional FAISS for faster/more robust k-means ---
try:
    import faiss  # pip install faiss-cpu  (or faiss-gpu)
    FAISS_AVAILABLE = True
except Exception:
    faiss = None
    FAISS_AVAILABLE = False


app = FastAPI(title="Visual Search API", version="1.1.0")

logger = logging.getLogger("visual_search_api")
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Global DB handle populated at startup
db = None  # type: ignore

# Allow browser clients (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handler for better error logging
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error in {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "path": str(request.url.path)}
    )

# ---------- Feature Extraction ----------
def extract_sift(gray_img, nfeatures=500):
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    kps, desc = sift.detectAndCompute(gray_img, None)
    if desc is None:
        return np.empty((0,128), np.float32), []
    return desc.astype(np.float32), kps
def to_rootsift(desc, eps=1e-12, l2_after=True):
    """
    Convert SIFT -> RootSIFT (Arandjelović & Zisserman, 2012).
    Steps: L1-normalize, then sqrt. Optionally L2-normalize after sqrt.
    desc: (N,128) float32
    """
    if desc is None or len(desc) == 0:
        return np.empty((0,128), np.float32)
    # L1-normalize
    desc /= (np.sum(desc, axis=1, keepdims=True) + eps)
    # element-wise sqrt
    desc = np.sqrt(desc, dtype=np.float32)
    if l2_after:
        # optional: stabilize numerics
        norms = np.linalg.norm(desc, axis=1, keepdims=True) + eps
        desc /= norms
    return desc.astype(np.float32)


def query_db(gray_img: np.ndarray, topk: int = 200):
    """Run a query against the vocabulary tree DB using the provided grayscale image.

    Returns list of (image_id, score) sorted by score desc (top 10 by default).
    """
    if gray_img is None or gray_img.size == 0:
        return []
    if db is None:
        logger.error("Database not loaded; cannot query.")
        return []
    q_descs, q_kps = extract_sift(gray_img, nfeatures=1500)
    if q_descs is None or len(q_descs) == 0:
        return []
    qdesc_root = to_rootsift(q_descs)
    try:
        cands = db.query(q_descs, topk=topk)
    except Exception as e:
        logger.exception("Error during DB query: %s", e)
        return []
    reranked = db.spatial_verify(qdesc_root, q_kps, cands, ratio_thresh=0.75, cap=1000, lam=0.01)
    top10 = [(img, score) for img, score, inl in reranked[:10]]
    logger.info("Query returned %d candidates; top10=%s", len(cands), top10[:3])
    return top10

def get_record_ids_by_page_ids(dataset, page_ids, unique=True):
    """
    Given:
      - dataset: a list of objects, each possibly with an "images" list like in your example
      - page_ids: list of pageId strings/ints to look up
    Returns:
      - a list of recordId strings associated with those pageIds.
        By default `unique=True` removes duplicates while preserving order.

    Example dataset shape:
      [
        {
          "images": [
            {
              "recordId": "27166539",
              "pageIds": ["27166540", "27166541", ...],
              ...
            },
            ...
          ]
        },
        ...
      ]
    """
    # 1) Build a mapping: pageId -> recordId
    page_to_record = {}
    for obj in dataset:
        for img in obj.get("images", []):
            rec = img.get("recordId")
            for pid in img.get("pageIds", []):
                if pid is None:
                    continue
                page_to_record[str(pid)] = rec

    # 2) Gather recordIds for the requested page_ids
    out = []
    for pid in page_ids:
        rid = page_to_record.get(str(pid))
        if rid is not None:
            out.append(rid)

    # 3) Optionally deduplicate while preserving order
    if unique:
        seen = set()
        out = [r for r in out if not (r in seen or seen.add(r))]

    return out


# ---------- Optional: faster for repeated queries ----------
def build_page_to_record_index(dataset):
    """
    Precompute and return a dict {pageId(str) -> recordId(str)}.
    Useful if you’ll call lookups many times.
    """
    index = {}
    for obj in dataset:
        for img in obj.get("images", []):
            rec = img.get("recordId")
            for pid in img.get("pageIds", []):
                if pid is None:
                    continue
                index[str(pid)] = rec
    return index

def lookup_record_ids(index, page_ids, unique=True):
    """Use the prebuilt index from build_page_to_record_index()."""
    out = [index.get(str(pid)) for pid in page_ids if index.get(str(pid)) is not None]
    if unique:
        seen = set()
        out = [r for r in out if not (r in seen or seen.add(r))]
    return out


# ---------- Example usage ----------
# pages = ["27166540", "27166551", "27166560"]
# record_ids = get_record_ids_by_page_ids(dataset, pages)          # one-off
# idx = build_page_to_record_index(dataset); record_ids2 = lookup_record_ids(idx, pages)  # repeated
# print(record_ids)


class ProcessResponse(BaseModel):
    record_ids: List[str] = Field(default_factory=list)
    page_ids: List[str] = Field(default_factory=list)
    matches: List[Dict[str, Any]] = Field(default_factory=list, description="Top database matches with scores")


@app.post("/process", response_model=ProcessResponse)
async def process_image_and_dataset(
    image: UploadFile = File(..., description="Image file (query)"),
    dataset: str = Form(..., description="JSON string of dataset objects (expects list)")
):
    """Receive an image and a JSON dataset, return inferred record/page IDs.

    Expected dataset JSON shape (simplified):
    [
      { "images": [ { "recordId": "123", "pageIds": ["456","457"] } ] }, ...
    ]
    """
    logger.info(f"Received /process request - image: {image.filename}, dataset size: {len(dataset)} bytes")
    
    # Decode dataset JSON
    try:
        dataset_obj = json.loads(dataset)
        if not isinstance(dataset_obj, list):
            raise ValueError("Dataset root must be a list")
        
        # Count images in dataset
        total_images = sum(len(obj.get('images', [])) for obj in dataset_obj)
        logger.info(f"Dataset contains {len(dataset_obj)} objects with {total_images} total images")
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON in dataset: {e}")
    except Exception as e:
        logger.error(f"Dataset parsing error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid dataset: {e}")

    # Read image bytes
    try:
        contents = await image.read()
        logger.info(f"Image size: {len(contents)} bytes")
        
        np_buf = np.frombuffer(contents, np.uint8)
        gray = cv2.imdecode(np_buf, cv2.IMREAD_GRAYSCALE)
        
        if gray is None:
            raise ValueError("Could not decode image")
        
        logger.info(f"Image decoded successfully: {gray.shape}")
        
    except Exception as e:
        logger.error(f"Image decode error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Query DB
    logger.info("Starting visual search query...")
    matches = query_db(gray, topk=50)
    page_ids = [m[0] for m in matches]  # assuming image ids correspond to page ids
    
    logger.info(f"Visual search found {len(matches)} matches, top page IDs: {page_ids[:3]}")
    
    # Map page IDs to record IDs
    record_ids = get_record_ids_by_page_ids(dataset=dataset_obj, page_ids=page_ids, unique=True)
    
    logger.info(f"Mapped to {len(record_ids)} unique record IDs: {record_ids[:5]}")
    
    return ProcessResponse(
        record_ids=record_ids,
        page_ids=page_ids,
        matches=[{"image_id": img_id, "score": float(score)} for img_id, score in matches]
    )


# Optional: simple health check on root
@app.get("/health")
def health():
    return {"status": "ok", "version": app.version, "db_loaded": bool(db)}


@app.get("/")
def root():
    return {"message": "Image+Object API", "version": app.version, "endpoints": ["/process", "/health"]}

def test():
    img=cv2.imread('vase.png', cv2.IMREAD_GRAYSCALE)
    results=query_db(img, topk=20)
    print(results)


@app.on_event("startup")
def _load_model():
    global db
    try:
        logger.info("Loading vocabulary tree model ...")
        db = load_vt_model('vocab_db')
        db = patch_spatial_verify_for_tuples(db)
        logger.info("Model loaded successfully")
        logger.info(f"Database contains {len(db.image_meta)} indexed pages")
    except Exception as e:
        logger.exception("Failed to load model: %s", e)


if __name__ == "__main__":
    # Manual run
    # Note: To handle large datasets, the frontend should compress the JSON
    # or the dataset should be sent via a different endpoint
    uvicorn.run(
        "app_server:app", 
        host="127.0.0.1", 
        port=8001, 
        reload=True
    )

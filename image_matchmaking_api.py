
import requests
from fastapi import FastAPI, File, UploadFile, Form, Request, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import uuid
import os
import shutil
import mimetypes
import time
import logging
from e_rara_id_fetcher import search_ids_v2
from e_rara_image_downloader_hack import get_all_page_ids, get_manifest_url



# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# In-memory job store for async jobs (for demo)
jobs = {}


def validate_fields(criteria, from_date, until_date, uploadedImage):
    errors = []
    # Empty submission
    if not criteria and not from_date and not until_date and not uploadedImage:
        errors.append({"field": "criteria", "message": "Empty submission"})
    # Year validation
    if from_date and (len(str(from_date)) != 4 or not str(from_date).isdigit()):
        errors.append({"field": "from_date", "message": "Year must be 4 digits"})
    if until_date and (len(str(until_date)) != 4 or not str(until_date).isdigit()):
        errors.append({"field": "until_date", "message": "Year must be 4 digits"})
    # from_date > until_date
    if from_date and until_date and int(from_date) > int(until_date):
        errors.append({"field": "date_range", "message": "from_date cannot be after until_date"})
    # Too many criteria
    if criteria and len(criteria) > 10:
        errors.append({"field": "criteria", "message": "Too many criteria (>10)"})
    # Too many images
    if uploadedImage and len(uploadedImage) > 5:
        errors.append({"field": "uploadedImage", "message": "Too many images (>5)"})
    # Unsupported file type
    if uploadedImage:
        for img in uploadedImage:
            mime, _ = mimetypes.guess_type(img.filename)
            if mime not in ["image/jpeg", "image/png"]:
                errors.append({"field": "uploadedImage", "message": f"Unsupported file type: {img.filename}"})
    return errors

def parse_criteria(criteria):
    # Accepts either bracket or dot notation, returns list of dicts
    parsed = []
    if not criteria:
        return parsed
    for c in criteria:
        # Example: "Title:Faust:AND" or "criteria[0][field]=Title"
        if isinstance(c, dict):
            parsed.append(c)
        elif isinstance(c, str):
            parts = c.split(":")
            if len(parts) >= 2:
                field = parts[0]
                value = parts[1]
                operator = parts[2] if len(parts) > 2 else "AND"
                parsed.append({"field": field, "value": value, "operator": operator})
    return parsed

def is_valid_image_url(url):
    try:
        resp = requests.head(url, timeout=5, allow_redirects=True)
        if resp.status_code == 405:  # Method not allowed; try GET lightweight
            g = requests.get(url, stream=True, timeout=8)
            ct = g.headers.get('content-type', '')
            ok = g.status_code == 200 and ct.startswith('image/')
            g.close()
            return ok
        return resp.status_code == 200 and resp.headers.get('content-type', '').startswith('image/')
    except Exception as e:
        logger.warning(f"Image URL validation failed for {url}: {e}")
        return False

def build_thumbnail_url(page_id: str, height: int = 150):
    # IIIF pattern: /i3f/v21/{page_id}/full/,{height}/0/default.jpg (height-constrained)
    return f"https://www.e-rara.ch/i3f/v21/{page_id}/full/,{height}/0/default.jpg"

def build_full_url(page_id: str):
    # Full resolution (might be very large)
    return f"https://www.e-rara.ch/i3f/v21/{page_id}/full/full/0/default.jpg"

def expand_record_to_first_page(record_id: str):
    data = get_all_page_ids(record_id)
    pages = data.get('page_ids', []) if isinstance(data, dict) else []
    if pages:
        return pages[0], pages
    return None, []

@app.post("/api/v1/matchmaking/images/search")
async def image_matchmaking_search(
    operation: str = Form(...),
    projectId: str = Form(...),
    agentId: str = Form(...),
    conversationId: Optional[str] = Form(None),
    from_date: Optional[str] = Form(None),
    until_date: Optional[str] = Form(None),
    maxResults: Optional[int] = Form(None),
    pageSize: Optional[int] = Form(None),
    includeMetadata: Optional[bool] = Form(True),
    responseFormat: Optional[str] = Form("json"),
    locale: Optional[str] = Form(None),
    criteria: Optional[List[str]] = Form(None),
    uploadedImage: Optional[List[UploadFile]] = File(None),
    request: Request = None,
    background_tasks: BackgroundTasks = None
):
    logger.info(f"Received request: operation={operation}, projectId={projectId}, agentId={agentId}, criteria={criteria}, from_date={from_date}, until_date={until_date}, maxResults={maxResults}")
    # Validate operation
    if operation != "IMAGE_MATCHMAKING":
        logger.warning("Invalid operation field")
        return JSONResponse(status_code=400, content={"error": "VALIDATION_ERROR", "details": [{"field": "operation", "message": "Must be IMAGE_MATCHMAKING"}]})

    errors = validate_fields(criteria, from_date, until_date, uploadedImage)
    if errors:
        logger.warning(f"Validation errors: {errors}")
        return JSONResponse(status_code=400, content={"error": "VALIDATION_ERROR", "details": errors})

    parsed_criteria = parse_criteria(criteria)
    logger.info(f"Parsed criteria: {parsed_criteria}")

    # Build search filters for e-rara
    filters = {}
    for c in parsed_criteria:
        field = c.get("field", "")
        value = c.get("value", "")
        if field.lower() == "title":
            filters["title"] = value
        elif field.lower() == "author":
            filters["author"] = value
        elif field.lower() == "place":
            filters["place"] = value
        elif field.lower() == "publisher":
            filters["publisher"] = value
    if from_date:
        filters["from_date"] = from_date
    if until_date:
        filters["until_date"] = until_date

    logger.info(f"Search filters: {filters}")

    # Synchronous or async
    if maxResults and maxResults > 100:
        # Async job
        job_id = str(uuid.uuid4())
        jobs[job_id] = {"status": "pending", "results": []}
        logger.info(f"Starting async job: {job_id}")
        # Launch background task
        def process_job(job_id, filters, uploadedImage):
            try:
                ids, total = search_ids_v2(**filters, max_records=maxResults)
                results = []
                for rid in ids:
                    first_page, all_pages = expand_record_to_first_page(rid)
                    if not first_page:
                        logger.warning(f"No pages for record {rid}")
                        continue
                    thumb = build_thumbnail_url(first_page)
                    full_url = build_full_url(first_page)
                    if is_valid_image_url(thumb):
                        results.append({
                            "recordId": rid,
                            "pageId": first_page,
                            "thumbnailUrl": thumb,
                            "fullImageUrl": full_url,
                            "pageCount": len(all_pages),
                            "pageIds": all_pages,
                            "manifest": get_manifest_url(rid)
                        })
                    else:
                        logger.warning(f"Invalid thumbnail for page {first_page} (record {rid})")
                jobs[job_id]["results"] = results
                jobs[job_id]["status"] = "done"
                logger.info(f"Async job {job_id} done: {len(results)} results")
            except Exception as e:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = str(e)
                logger.error(f"Async job {job_id} error: {e}")
        background_tasks.add_task(process_job, job_id, filters, uploadedImage)
        return JSONResponse(content={"jobId": job_id, "status": "pending"})
    else:
        # Synchronous: search and match
        try:
            ids, total = search_ids_v2(**filters, max_records=maxResults)
            images = []
            for rid in ids:
                first_page, all_pages = expand_record_to_first_page(rid)
                if not first_page:
                    logger.warning(f"No pages for record {rid}")
                    continue
                thumb = build_thumbnail_url(first_page)
                full_url = build_full_url(first_page)
                if is_valid_image_url(thumb):
                    images.append({
                        "recordId": rid,
                        "pageId": first_page,
                        "thumbnailUrl": thumb,
                        "fullImageUrl": full_url,
                        "pageCount": len(all_pages),
                        "pageIds": all_pages,
                        "manifest": get_manifest_url(rid)
                    })
                else:
                    logger.warning(f"Invalid thumbnail for page {first_page} (record {rid})")
            logger.info(f"Synchronous search returned {len(images)} images")
            return JSONResponse(content={"images": images, "count": len(images)})
        except Exception as e:
            logger.error(f"Synchronous search error: {e}")
            return JSONResponse(status_code=500, content={"error": "INTERNAL_ERROR", "details": str(e)})


@app.get("/api/v1/matchmaking/images/results")
async def get_matchmaking_results(jobId: str, pageToken: Optional[str] = None):
    job = jobs.get(jobId)
    if not job:
        return JSONResponse(status_code=404, content={"error": "NOT_FOUND", "details": [{"field": "jobId", "message": "Job not found"}]})
    # For demo, return job results
    return JSONResponse(content={"images": job.get("results", []), "status": job.get("status", "pending")})

# SSE streaming endpoint
@app.get("/api/v1/matchmaking/images/stream")
async def stream_matchmaking_results(jobId: str):
    def event_stream():
        while True:
            job = jobs.get(jobId)
            if not job:
                yield f"event: error\ndata: {{'error': 'NOT_FOUND'}}\n\n"
                break
            if job["status"] == "done":
                for img in job["results"]:
                    yield f"event: match\ndata: {img}\n\n"
                yield f"event: done\ndata: {{'status': 'done'}}\n\n"
                break
            elif job["status"] == "error":
                yield f"event: error\ndata: {{'error': '{job.get('error', 'Unknown')}'}}\n\n"
                break
            else:
                yield f"event: progress\ndata: {{'status': 'pending'}}\n\n"
                time.sleep(1)
    return StreamingResponse(event_stream(), media_type="text/event-stream")

# Add more endpoints and validation as needed


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

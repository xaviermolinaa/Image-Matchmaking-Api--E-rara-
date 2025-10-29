
import requests
import json
from fastapi import FastAPI, File, UploadFile, Form, Request, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid
import os
import shutil
import mimetypes
import time
import threading
import logging
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from e_rara_id_fetcher import search_ids_v2
from e_rara_image_downloader_hack import get_all_page_ids, get_manifest_url



# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="E-rara Image Matchmaking API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "2.0.0", "features": ["caching", "concurrency", "smart_page_selection"]}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {"message": "E-rara Image Matchmaking API", "version": "2.0.0", "docs": "/docs"}

# Configuration
class CacheConfig:
    MANIFEST_CACHE_SIZE = 1000  # Cache up to 1000 manifests
    IMAGE_VALIDATION_CACHE_SIZE = 2000  # Cache validation results
    CACHE_TTL_SECONDS = 3600  # 1 hour TTL for demonstration

class ConcurrencyConfig:
    MAX_WORKERS = 5  # Maximum concurrent workers for processing records
    BATCH_SIZE = 10  # Process records in batches

cache_config = CacheConfig()
concurrency_config = ConcurrencyConfig()

# Cached version of expensive operations
@lru_cache(maxsize=cache_config.MANIFEST_CACHE_SIZE)
def get_cached_page_ids(record_id: str):
    """Cached version of get_all_page_ids to avoid repeated API calls"""
    logger.debug(f"Cache miss for record {record_id} - fetching from e-rara")
    return get_all_page_ids(record_id)

@lru_cache(maxsize=cache_config.IMAGE_VALIDATION_CACHE_SIZE)
def get_cached_image_validation(url: str):
    """Cached version of image URL validation"""
    logger.debug(f"Cache miss for image validation {url}")
    return is_valid_image_url_uncached(url)

def clear_caches():
    """Clear all caches - useful for testing or periodic cleanup"""
    get_cached_page_ids.cache_clear()
    get_cached_image_validation.cache_clear()
    logger.info("All caches cleared")

def get_cache_stats():
    """Get cache statistics for monitoring"""
    return {
        "manifest_cache": get_cached_page_ids.cache_info()._asdict(),
        "image_validation_cache": get_cached_image_validation.cache_info()._asdict()
    }


# In-memory job store for async jobs (for demo)
jobs = {}
jobs_lock = threading.Lock()

def create_job() -> str:
    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {
            "status": "pending",
            "created": time.time(),
            "started": None,
            "finished": None,
            "results": [],
            "total_records": 0,
            "processed_records": 0,
            "progress": 0.0,
            "error": None,
            "totalFound": None,
            "warnings": []
        }
    return job_id

def update_job_progress(job_id: str, *, processed: int = None, total: int = None, append_result=None, warnings=None):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        if total is not None:
            job["total_records"] = total
        if processed is not None:
            job["processed_records"] = processed
        if append_result is not None:
            job["results"].append(append_result)
        if warnings:
            job.setdefault("warnings", [])
            job["warnings"].extend(warnings)
        total_val = job.get("total_records") or 0
        proc_val = job.get("processed_records") or 0
        job["progress"] = float(proc_val / total_val) if total_val else 0.0

def finalize_job(job_id: str, status: str, error: str = None):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        job["status"] = status
        job["finished"] = time.time()
        if error:
            job["error"] = error

# Pydantic models for JSON requests
class SearchCriteria(BaseModel):
    field: str
    value: str
    operator: Optional[str] = "AND"

class ImageMatchmakingRequest(BaseModel):
    operation: str
    projectId: Optional[str] = None
    agentId: Optional[str] = None
    conversationId: Optional[str] = None
    criteria: Optional[List[SearchCriteria]] = None
    from_date: Optional[str] = None
    until_date: Optional[str] = None
    maxResults: Optional[int] = 10
    max_records: Optional[int] = None
    pageSize: Optional[int] = None
    includeMetadata: Optional[bool] = True
    responseFormat: Optional[str] = "json"
    locale: Optional[str] = None
    images: Optional[int] = None
    # New parameters for page selection
    avoid_covers: Optional[bool] = True  # Skip cover pages by default
    page_selection: Optional[str] = "content"  # "content", "first", "random"
    # New parameter for image validation
    validate_images: Optional[bool] = True  # Enable image validation by default


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
        # Handle dict objects (from JSON) or string format
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

def is_valid_image_url_uncached(url):
    """Original image validation function without caching"""
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

def is_valid_image_url(url, use_cache=True):
    """
    Validate image URL with optional caching
    
    Parameters:
    -----------
    url : str
        Image URL to validate
    use_cache : bool
        Whether to use cached results (default: True)
    """
    if use_cache:
        return get_cached_image_validation(url)
    else:
        return is_valid_image_url_uncached(url)

def build_thumbnail_url(page_id: str, height: int = 150):
    # IIIF pattern: /i3f/v21/{page_id}/full/,{height}/0/default.jpg (height-constrained)
    return f"https://www.e-rara.ch/i3f/v21/{page_id}/full/,{height}/0/default.jpg"

def build_full_url(page_id: str):
    # Full resolution (might be very large)
    return f"https://www.e-rara.ch/i3f/v21/{page_id}/full/full/0/default.jpg"

def select_page_from_record(record_id: str, avoid_covers: bool = True, page_selection: str = "content"):
    """
    Select a page from a record based on the specified strategy.
    Uses cached manifest data for better performance.
    
    Parameters:
    -----------
    record_id : str
        The e-rara record ID
    avoid_covers : bool
        Whether to skip likely cover pages
    page_selection : str
        Page selection strategy: "content", "first", "random"
    
    Returns:
    --------
    tuple
        (selected_page_id, all_page_ids)
    """
    # Use cached version for better performance
    data = get_cached_page_ids(record_id)
    pages = data.get('page_ids', []) if isinstance(data, dict) else []
    
    if not pages:
        return None, []
    
    total_pages = len(pages)
    
    # If not avoiding covers or requesting first page, return first page
    if not avoid_covers or page_selection == "first":
        return pages[0], pages
    
    # For very short documents, just return the first page
    if total_pages <= 3:
        return pages[0], pages
    
    if page_selection == "random":
        import random
        # For random, still avoid covers if requested
        if avoid_covers:
            skip_start = min(2, total_pages // 4)
            skip_end = min(1, total_pages // 8)
            content_pages = pages[skip_start:total_pages-skip_end] if skip_start < total_pages-skip_end else pages[1:]
            selected_page = random.choice(content_pages) if content_pages else pages[0]
        else:
            selected_page = random.choice(pages)
        return selected_page, pages
    
    # Default: "content" strategy
    # Skip only the first page (likely cover/title page)
    # Simply return page 2 (index 1)
    if total_pages >= 2:
        selected_page = pages[1]  # Page 2
        logger.info(f"Record {record_id}: Selected page 2 of {total_pages} (avoiding cover page)")
        return selected_page, pages
    
    # Ultimate fallback: return first page if only 1 page exists
    return pages[0], pages

# Keep the old function for backward compatibility but use the new logic
def process_single_record(record_id: str, avoid_covers: bool = True, page_selection: str = "content", validate_images: bool = True):
    """
    Process a single record to extract image information
    
    Parameters:
    -----------
    record_id : str
        The e-rara record ID
    avoid_covers : bool
        Whether to skip likely cover pages
    page_selection : str
        Page selection strategy
    validate_images : bool
        Whether to validate image URLs
        
    Returns:
    --------
    dict or None
        Image information dictionary or None if processing failed
    """
    try:
        data = get_cached_page_ids(record_id)
        metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
        first_page, all_pages = select_page_from_record(record_id, avoid_covers, page_selection)
        if not first_page:
            logger.warning(f"No pages for record {record_id}")
            return None
            
        thumb = build_thumbnail_url(first_page)
        full_url = build_full_url(first_page)
        
        # Use configurable image validation
        if validate_images:
            is_valid = is_valid_image_url(thumb, use_cache=True)
        else:
            # Skip validation for faster response
            is_valid = True
            logger.debug(f"Skipping image validation for {thumb} (validate_images=False)")
        
        if is_valid:
            manifest_url = get_manifest_url(record_id)
            
            # Build comprehensive result with all available metadata
            result = {
                "id": record_id,
                "recordId": record_id,
                "pageId": first_page,
                "pageIds": all_pages,
                "pageCount": len(all_pages),
                "thumbnailUrl": thumb,
                "thumbUrl": thumb,
                "fullImageUrl": full_url,
                "fullUrl": full_url,
                "url": full_url,
                "manifest": manifest_url,
                "metadata": {}
            }
            
            # Extract and promote common metadata fields to both top-level and metadata object
            if isinstance(metadata, dict):
                # Copy all metadata to nested object
                result["metadata"] = metadata.copy()
                
                # Promote key fields to top level for UI convenience
                title = metadata.get("title") or metadata.get("Title") or metadata.get("label")
                if title:
                    result["title"] = title
                    result["metadata"]["title"] = title
                
                # Extract other common fields
                author = metadata.get("Author") or metadata.get("Creator") or metadata.get("author")
                if author:
                    result["author"] = author
                    result["metadata"]["author"] = author
                
                date = metadata.get("Date") or metadata.get("Publication date") or metadata.get("date")
                if date:
                    result["date"] = date
                    result["metadata"]["date"] = date
                
                place = metadata.get("Place") or metadata.get("Publication place") or metadata.get("place")
                if place:
                    result["place"] = place
                    result["metadata"]["place"] = place
                
                publisher = metadata.get("Publisher") or metadata.get("Printer / Publisher") or metadata.get("publisher")
                if publisher:
                    result["publisher"] = publisher
                    result["metadata"]["publisher"] = publisher
                
                # Include raw label if exists
                label = metadata.get("label")
                if label and not title:
                    result["title"] = label
                    result["metadata"]["title"] = label
            
            return result
        else:
            logger.warning(f"Invalid thumbnail for page {first_page} (record {record_id})")
            return None
            
    except Exception as e:
        logger.error(f"Error processing record {record_id}: {e}")
        return None

def process_records_concurrently(record_ids: List[str], avoid_covers: bool = True, 
                                page_selection: str = "content", validate_images: bool = True,
                                max_workers: int = None) -> List[Dict]:
    """
    Process multiple records concurrently for better performance
    
    Parameters:
    -----------
    record_ids : List[str]
        List of e-rara record IDs to process
    avoid_covers : bool
        Whether to skip likely cover pages
    page_selection : str
        Page selection strategy
    validate_images : bool
        Whether to validate image URLs
    max_workers : int
        Maximum number of concurrent workers (defaults to config)
        
    Returns:
    --------
    List[Dict]
        List of successfully processed image information dictionaries
    """
    if max_workers is None:
        max_workers = concurrency_config.MAX_WORKERS
        
    images = []
    
    # For small numbers of records, process sequentially to avoid overhead
    if len(record_ids) <= 2:
        logger.info(f"Processing {len(record_ids)} records sequentially")
        for record_id in record_ids:
            result = process_single_record(record_id, avoid_covers, page_selection, validate_images)
            if result:
                images.append(result)
    else:
        logger.info(f"Processing {len(record_ids)} records concurrently with {max_workers} workers")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_record = {
                executor.submit(process_single_record, record_id, avoid_covers, page_selection, validate_images): record_id
                for record_id in record_ids
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_record):
                record_id = future_to_record[future]
                try:
                    result = future.result()
                    if result:
                        images.append(result)
                except Exception as e:
                    logger.error(f"Error processing record {record_id}: {e}")
    
    return images
    """
    Legacy function - now uses the new select_page_from_record with content strategy and caching
    """
    return select_page_from_record(record_id, avoid_covers=True, page_selection="content")

@app.post("/api/v1/matchmaking/images/search")
async def image_matchmaking_search_json(request_data: ImageMatchmakingRequest):
    """JSON-based image matchmaking search endpoint"""
    logger.info(f"Received JSON request: operation={request_data.operation}, criteria={request_data.criteria}, from_date={request_data.from_date}, until_date={request_data.until_date}, maxResults={request_data.maxResults}")
    
    # Validate operation
    if request_data.operation != "IMAGE_MATCHMAKING":
        logger.warning("Invalid operation field")
        return JSONResponse(status_code=400, content={"error": "VALIDATION_ERROR", "details": [{"field": "operation", "message": "Must be IMAGE_MATCHMAKING"}]})

    # Convert criteria to the format expected by existing logic
    criteria_list = []
    if request_data.criteria:
        for c in request_data.criteria:
            criteria_list.append(c.dict())

    errors = validate_fields(criteria_list, request_data.from_date, request_data.until_date, None)
    if errors:
        logger.warning(f"Validation errors: {errors}")
        return JSONResponse(status_code=400, content={"error": "VALIDATION_ERROR", "details": errors})

    parsed_criteria = parse_criteria(criteria_list)
    logger.info(f"Parsed criteria: {parsed_criteria}")

    # Build search filters for e-rara
    filters = {}
    for c in parsed_criteria:
        field = c.get("field", "")
        value = c.get("value", "")
        # Normalize field names to handle variations
        field_lower = field.lower().strip()
        if field_lower in ["title"]:
            filters["title"] = value
        elif field_lower in ["author", "creator", "author or collaborator", "collaborator"]:
            filters["author"] = value
        elif field_lower in ["place", "publication place", "origin place"]:
            filters["place"] = value
        elif field_lower in ["publisher", "printer", "printer / publisher", "printer/publisher"]:
            filters["publisher"] = value
    if request_data.from_date:
        filters["from_date"] = request_data.from_date
    if request_data.until_date:
        filters["until_date"] = request_data.until_date

    logger.info(f"Search filters: {filters}")

    # Synchronous search (default)
    try:
        effective_max = request_data.max_records or request_data.maxResults
        ids, total = search_ids_v2(**filters, max_records=effective_max)
        
        # Use concurrent processing for better performance
        images = process_records_concurrently(
            record_ids=ids,
            avoid_covers=request_data.avoid_covers,
            page_selection=request_data.page_selection,
            validate_images=request_data.validate_images,
            max_workers=concurrency_config.MAX_WORKERS
        )
        
        response_payload = {
            "images": images,
            "count": len(images),
            "totalFound": total if total is not None else len(ids)
        }
        if not images:
            response_payload["warnings"] = [{"code": "NO_RESULTS", "message": "No images found for the given criteria."}]

        logger.info(f"JSON search returned {len(images)} images (totalFound={total})")
        return JSONResponse(content=response_payload)
    except Exception as e:
        logger.error(f"JSON search error: {e}")
        return JSONResponse(status_code=500, content={"error": "INTERNAL_ERROR", "details": str(e)})


@app.post("/api/v1/matchmaking/images/search/async")
async def image_matchmaking_search_async(request_data: ImageMatchmakingRequest, background_tasks: BackgroundTasks):
    """Submit an async matchmaking job. Returns jobId immediately so client can poll or stream."""
    if request_data.operation != "IMAGE_MATCHMAKING":
        return JSONResponse(status_code=400, content={"error": "VALIDATION_ERROR", "details": [{"field": "operation", "message": "Must be IMAGE_MATCHMAKING"}]})

    # Parse & validate criteria
    criteria_list = [c.dict() for c in (request_data.criteria or [])]
    errors = validate_fields(criteria_list, request_data.from_date, request_data.until_date, None)
    if errors:
        return JSONResponse(status_code=400, content={"error": "VALIDATION_ERROR", "details": errors})
    parsed_criteria = parse_criteria(criteria_list)

    filters = {}
    for c in parsed_criteria:
        field = c.get("field", "").lower().strip()
        value = c.get("value", "")
        if field in ["title"]:
            filters["title"] = value
        elif field in ["author", "creator"]:
            filters["author"] = value
        elif field in ["place", "publication place", "origin place"]:
            filters["place"] = value
        elif field in ["publisher", "printer", "printer / publisher", "printer/publisher"]:
            filters["publisher"] = value
    if request_data.from_date:
        filters["from_date"] = request_data.from_date
    if request_data.until_date:
        filters["until_date"] = request_data.until_date

    job_id = create_job()
    logger.info(f"Created async job {job_id} with filters={filters}")

    def run_job():
        with jobs_lock:
            jobs[job_id]["status"] = "running"
            jobs[job_id]["started"] = time.time()
        try:
            effective_max_async = request_data.max_records or request_data.maxResults
            ids, total = search_ids_v2(**filters, max_records=effective_max_async)
            update_job_progress(job_id, total=total if total is not None else len(ids))

            # Decide processing strategy
            processed = 0
            if len(ids) <= 2:
                for rid in ids:
                    res = process_single_record(rid, request_data.avoid_covers, request_data.page_selection, request_data.validate_images)
                    if res:
                        update_job_progress(job_id, append_result=res)
                    processed += 1
                    update_job_progress(job_id, processed=processed)
            else:
                # Concurrent
                with ThreadPoolExecutor(max_workers=concurrency_config.MAX_WORKERS) as executor:
                    future_to_record = {executor.submit(process_single_record, rid, request_data.avoid_covers, request_data.page_selection, request_data.validate_images): rid for rid in ids}
                    for future in as_completed(future_to_record):
                        rid = future_to_record[future]
                        try:
                            res = future.result()
                            if res:
                                update_job_progress(job_id, append_result=res)
                        except Exception as e:
                            logger.error(f"Async job {job_id} record {rid} error: {e}")
                        processed += 1
                        update_job_progress(job_id, processed=processed)
            total_found = total if total is not None else len(ids)
            with jobs_lock:
                jobs[job_id]["totalFound"] = total_found
                if not jobs[job_id]["results"]:
                    jobs[job_id]["warnings"] = [{"code": "NO_RESULTS", "message": "No images found for the given criteria."}]
            finalize_job(job_id, "done")
            logger.info(f"Async job {job_id} finished with {len(jobs[job_id]['results'])} results (totalFound={total_found})")
        except Exception as e:
            logger.exception(f"Async job {job_id} failed: {e}")
            finalize_job(job_id, "error", error=str(e))

    background_tasks.add_task(run_job)
    return JSONResponse(content={"jobId": job_id, "status": "pending"})


@app.get("/api/v1/matchmaking/images/jobs/{job_id}")
async def get_job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "NOT_FOUND", "details": [{"field": "jobId", "message": "Job not found"}]})
    # Copy without internal references
    safe = {k: v for k, v in job.items() if k not in []}
    safe["count"] = len(job.get("results", []))
    return JSONResponse(content=safe)

@app.post("/api/v1/matchmaking/images/search/form")
async def image_matchmaking_search(
    operation: str = Form(...),
    projectId: str = Form(...),
    agentId: str = Form(...),
    conversationId: Optional[str] = Form(None),
    from_date: Optional[str] = Form(None),
    until_date: Optional[str] = Form(None),
    maxResults: Optional[int] = Form(None),
    max_records: Optional[int] = Form(None),
    pageSize: Optional[int] = Form(None),
    includeMetadata: Optional[bool] = Form(True),
    responseFormat: Optional[str] = Form("json"),
    locale: Optional[str] = Form(None),
    criteria: Optional[List[str]] = Form(None),
    uploadedImage: Optional[List[UploadFile]] = File(None),
    # New optional parameters
    avoid_covers: Optional[bool] = Form(True),
    page_selection: Optional[str] = Form("content"),
    validate_images: Optional[bool] = Form(True),
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
        # Normalize field names to handle variations
        field_lower = field.lower().strip()
        if field_lower in ["title"]:
            filters["title"] = value
        elif field_lower in ["author", "creator"]:
            filters["author"] = value
        elif field_lower in ["place", "publication place", "origin place"]:
            filters["place"] = value
        elif field_lower in ["publisher", "printer", "printer / publisher", "printer/publisher"]:
            filters["publisher"] = value
    if from_date:
        filters["from_date"] = from_date
    if until_date:
        filters["until_date"] = until_date

    logger.info(f"Search filters: {filters}")

    # Synchronous or async
    effective_max = max_records or maxResults

    if effective_max and effective_max > 100:
        # Async job
        job_id = str(uuid.uuid4())
        jobs[job_id] = {"status": "pending", "results": []}
        logger.info(f"Starting async job: {job_id}")
        # Launch background task
        def process_job(job_id, filters, uploadedImage, max_records_value):
            try:
                ids, total = search_ids_v2(**filters, max_records=max_records_value)
                
                # Use concurrent processing for async jobs too
                results = process_records_concurrently(
                    record_ids=ids,
                    avoid_covers=avoid_covers,
                    page_selection=page_selection,
                    validate_images=validate_images,
                    max_workers=concurrency_config.MAX_WORKERS
                )
                
                jobs[job_id]["results"] = results
                jobs[job_id]["status"] = "done"
                logger.info(f"Async job {job_id} done: {len(results)} results")
            except Exception as e:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = str(e)
                logger.error(f"Async job {job_id} error: {e}")
        background_tasks.add_task(process_job, job_id, filters, uploadedImage, effective_max)
        return JSONResponse(content={"jobId": job_id, "status": "pending"})
    else:
        # Synchronous: search and match
        try:
            ids, total = search_ids_v2(**filters, max_records=effective_max)
            
            # Use concurrent processing for better performance
            images = process_records_concurrently(
                record_ids=ids,
                avoid_covers=avoid_covers,
                page_selection=page_selection,
                validate_images=validate_images,
                max_workers=concurrency_config.MAX_WORKERS
            )
            response_payload = {
                "images": images,
                "count": len(images),
                "totalFound": total if total is not None else len(ids)
            }
            if not images:
                response_payload["warnings"] = [{"code": "NO_RESULTS", "message": "No images found for the given criteria."}]
            
            logger.info(f"Synchronous search returned {len(images)} images (totalFound={total})")
            return JSONResponse(content=response_payload)
        except Exception as e:
            logger.error(f"Synchronous search error: {e}")
            return JSONResponse(status_code=500, content={"error": "INTERNAL_ERROR", "details": str(e)})


@app.get("/api/v1/matchmaking/images/results")
async def get_matchmaking_results(jobId: str, pageToken: Optional[str] = None):
    job = jobs.get(jobId)
    if not job:
        return JSONResponse(status_code=404, content={"error": "NOT_FOUND", "details": [{"field": "jobId", "message": "Job not found"}]})
    # For demo, return job results
    images = job.get("results", [])
    payload = {
        "images": images,
        "count": len(images),
        "status": job.get("status", "pending"),
        "totalFound": job.get("totalFound"),
        "warnings": job.get("warnings", [])
    }
    return JSONResponse(content=payload)

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
                    yield f"event: match\ndata: {json.dumps(img)}\n\n"
                summary = {
                    "status": "done",
                    "count": len(job["results"]),
                    "totalFound": job.get("totalFound"),
                    "warnings": job.get("warnings", [])
                }
                yield f"event: done\ndata: {json.dumps(summary)}\n\n"
                break
            elif job["status"] == "error":
                error_payload = {"error": job.get("error", "Unknown"), "status": "error"}
                yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
                break
            else:
                progress_payload = {
                    "status": job.get("status", "pending"),
                    "progress": job.get("progress", 0.0),
                    "processed": job.get("processed_records", 0),
                    "total": job.get("total_records", 0)
                }
                yield f"event: progress\ndata: {json.dumps(progress_payload)}\n\n"
                time.sleep(1)
    return StreamingResponse(event_stream(), media_type="text/event-stream")

# Cache management endpoints
@app.get("/api/v1/cache/stats")
async def get_cache_statistics():
    """Get cache performance statistics"""
    stats = get_cache_stats()
    return JSONResponse(content={
        "cache_stats": stats,
        "config": {
            "manifest_cache_size": cache_config.MANIFEST_CACHE_SIZE,
            "image_validation_cache_size": cache_config.IMAGE_VALIDATION_CACHE_SIZE
        }
    })

@app.post("/api/v1/cache/clear")
async def clear_all_caches():
    """Clear all caches - useful for testing or forcing fresh data"""
    clear_caches()
    return JSONResponse(content={"message": "All caches cleared successfully"})

@app.get("/api/v1/performance/config")
async def get_performance_config():
    """Get current performance configuration"""
    return JSONResponse(content={
        "caching": {
            "manifest_cache_size": cache_config.MANIFEST_CACHE_SIZE,
            "image_validation_cache_size": cache_config.IMAGE_VALIDATION_CACHE_SIZE,
            "cache_ttl_seconds": cache_config.CACHE_TTL_SECONDS
        },
        "concurrency": {
            "max_workers": concurrency_config.MAX_WORKERS,
            "batch_size": concurrency_config.BATCH_SIZE
        },
        "features": {
            "concurrent_processing": True,
            "configurable_image_validation": True,
            "smart_page_selection": True,
            "caching_enabled": True
        }
    })

# Add more endpoints and validation as needed


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

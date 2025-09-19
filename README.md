# E-rara Image Matchmaking API

A FastAPI-based service for searching and retrieving historical images from the e-rara digital library using bibliographic criteria and optional reference images.

## Overview

This API provides an IMAGE_MATCHMAKING operation that allows clients to:
- Search e-rara's collection using metadata filters (author, title, place, publisher, date range)
- Upload reference images for similarity matching
- Receive both thumbnail and full-resolution image URLs
- Handle large result sets asynchronously with job polling or SSE streaming

## Features

- **Multipart form-data input** - Accepts criteria, date ranges, and uploaded images
- **IIIF image URLs** - Returns proper thumbnail and full-resolution URLs
- **Manifest integration** - Expands records to individual pages with full page ID arrays
- **Async processing** - Background jobs for large result sets (>100 images)
- **Streaming support** - Server-Sent Events (SSE) for real-time progress
- **Comprehensive validation** - Input validation, image URL verification, error handling
- **Rich metadata** - Returns record IDs, page counts, manifest URLs, and complete page arrays

## Quick Start

### Prerequisites

```bash
pip install fastapi uvicorn requests beautifulsoup4 python-multipart
```

### Running the API

```bash
uvicorn image_matchmaking_api:app --reload
```

The API will be available at:
- **Base URL**: http://127.0.0.1:8000
- **Interactive docs**: http://127.0.0.1:8000/docs
- **OpenAPI spec**: http://127.0.0.1:8000/openapi.json

## API Endpoints

### POST `/api/v1/matchmaking/images/search`

Main search endpoint accepting multipart/form-data.

#### Required Fields
- `operation` (string): Must be "IMAGE_MATCHMAKING"
- `projectId` (string): Project identifier
- `agentId` (string): Agent identifier

#### Optional Fields
- `conversationId` (string): UUID for traceability
- `from_date` (string): Start year (YYYY format)
- `until_date` (string): End year (YYYY format)
- `maxResults` (integer): Maximum number of results
- `pageSize` (integer): Page size for pagination
- `includeMetadata` (boolean): Include metadata (default: true)
- `responseFormat` (string): "json" or "stream"
- `locale` (string): Language preference
- `criteria` (array): Search criteria in format "field:value:operator"
- `uploadedImage` (files): Reference images for similarity matching

#### Synchronous Response (≤100 results)

```json
{
  "images": [
    {
      "recordId": "6100663",
      "pageId": "6100665",
      "thumbnailUrl": "https://www.e-rara.ch/i3f/v21/6100665/full/,150/0/default.jpg",
      "fullImageUrl": "https://www.e-rara.ch/i3f/v21/6100665/full/full/0/default.jpg",
      "pageCount": 372,
      "pageIds": ["6100665", "6100666", "6100667", "..."],
      "manifest": "https://www.e-rara.ch/i3f/v21/6100663/manifest"
    }
  ],
  "count": 1
}
```

#### Async Response (>100 results)

```json
{
  "jobId": "uuid-string",
  "status": "pending"
}
```

### GET `/api/v1/matchmaking/images/results`

Poll for async job results.

**Parameters:**
- `jobId` (required): Job identifier
- `pageToken` (optional): Pagination token

### GET `/api/v1/matchmaking/images/stream`

Server-Sent Events stream for async job progress.

**Parameters:**
- `jobId` (required): Job identifier

## Search Criteria

### Supported Fields
- `Title` - Document title
- `Author` - Creator/author name
- `Place` - Publication place
- `Publisher` - Publisher/printer name

### Date Filtering
- `from_date` - Start year (e.g., "1600")
- `until_date` - End year (e.g., "1700")
- Automatic splitting for ranges >400 years

## Error Handling

### HTTP Status Codes
- `200` - Success
- `400` - Validation error
- `404` - Job not found
- `413` - Payload too large
- `415` - Unsupported media type
- `422` - Unsupported field
- `429` - Rate limit exceeded
- `500` - Internal server error

### Error Response Format

```json
{
  "error": "VALIDATION_ERROR",
  "details": [
    {
      "field": "from_date",
      "message": "Year must be 4 digits"
    }
  ]
}
```

## Usage Examples

### Basic Search with cURL

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/matchmaking/images/search" \
  -F "operation=IMAGE_MATCHMAKING" \
  -F "projectId=demo" \
  -F "agentId=demo" \
  -F "from_date=1600" \
  -F "until_date=1650" \
  -F "maxResults=5"
```

### Search with Criteria

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/matchmaking/images/search" \
  -F "operation=IMAGE_MATCHMAKING" \
  -F "projectId=demo" \
  -F "agentId=demo" \
  -F "criteria=Title:Historia:AND" \
  -F "criteria=Place:Basel:AND" \
  -F "maxResults=10"
```

### JavaScript Frontend Integration

```javascript
async function searchImages() {
  const formData = new FormData();
  formData.append('operation', 'IMAGE_MATCHMAKING');
  formData.append('projectId', 'your-project-id');
  formData.append('agentId', 'your-agent-id');
  formData.append('from_date', '1600');
  formData.append('until_date', '1700');
  formData.append('maxResults', '10');

  const response = await fetch('/api/v1/matchmaking/images/search', {
    method: 'POST',
    body: formData
  });

  const data = await response.json();
  
  if (data.images) {
    // Synchronous results
    renderImages(data.images);
  } else if (data.jobId) {
    // Async job - poll for results
    pollJobResults(data.jobId);
  }
}

function renderImages(images) {
  images.forEach(img => {
    // Show thumbnail first
    const thumbnail = document.createElement('img');
    thumbnail.src = img.thumbnailUrl;
    thumbnail.onclick = () => {
      // Load full image on click
      thumbnail.src = img.fullImageUrl;
    };
    document.body.appendChild(thumbnail);
  });
}
```

## Image URL Patterns

### IIIF URL Structure
- **Thumbnail**: `https://www.e-rara.ch/i3f/v21/{pageId}/full/,150/0/default.jpg`
- **Full size**: `https://www.e-rara.ch/i3f/v21/{pageId}/full/full/0/default.jpg`
- **Custom size**: `https://www.e-rara.ch/i3f/v21/{pageId}/full/,{height}/0/default.jpg`

### Size Options
- `full` - Original dimensions
- `,150` - Height constrained to 150px
- `300,` - Width constrained to 300px
- `!300,300` - Fit within 300×300 box
- `pct:25` - 25% of original size

## Development

### Project Structure
```
├── image_matchmaking_api.py    # Main FastAPI application
├── e_rara_id_fetcher.py       # E-rara search logic
├── e_rara_image_downloader_hack.py  # IIIF manifest processing
├── README.md                  # This file
└── read.md                   # Original API specification
```

### Dependencies
- **FastAPI** - Web framework
- **Uvicorn** - ASGI server
- **Requests** - HTTP client
- **BeautifulSoup4** - HTML/XML parsing
- **python-multipart** - Form data handling

### Adding Features

To extend the API:

1. **New search criteria**: Update `parse_criteria()` function
2. **Image processing**: Integrate with vision models in `process_job()`
3. **Caching**: Add Redis/memory cache for manifest data
4. **Authentication**: Add JWT/API key middleware
5. **Rate limiting**: Implement request throttling

### Testing

```bash
# Start the development server
uvicorn image_matchmaking_api:app --reload --log-level debug

# Test basic functionality
curl -X POST "http://127.0.0.1:8000/api/v1/matchmaking/images/search" \
  -F "operation=IMAGE_MATCHMAKING" \
  -F "projectId=test" \
  -F "agentId=test" \
  -F "from_date=1600" \
  -F "until_date=1610" \
  -F "maxResults=2"
```

## Performance Considerations

- **Manifest fetching**: Can be slow for large collections; consider caching
- **Image validation**: HTTP HEAD requests add latency; disable for faster responses
- **Full images**: Can be very large (10-50MB+); use progressive loading
- **Rate limiting**: E-rara may throttle requests; implement delays/retries

## Contributing

1. Follow the existing code structure and naming conventions
2. Add logging for new features using the configured logger
3. Include error handling and validation for new endpoints
4. Update this README for any API changes

## License

This project interfaces with e-rara.ch, a service of the ETH Library. Please respect their terms of service and usage guidelines.
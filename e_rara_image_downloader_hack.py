import os
import requests
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import json
import cv2
import numpy as np
from PIL import Image
import io
import torch

# IIIF image endpoint
IIIF_BASE = "https://www.e-rara.ch/i3f/v21/{record_id}/full/{size}/0/default.jpg"
# Manifest endpoint to get all pages for a document
MANIFEST_URL = "https://www.e-rara.ch/i3f/v21/{record_id}/manifest"

def get_image_url(record_id, size="full"):
    """
    Generate the IIIF URL for an image
    
    Parameters:
    -----------
    record_id : str
        The e-rara record ID
    size : str, optional
        Size parameter for IIIF (default: "full")
        Can be: "full", "max", ",150" (for thumbnails), etc.
    
    Returns:
    --------
    str
        Complete IIIF URL for the image
    """
    return IIIF_BASE.format(record_id=record_id, size=size)


def get_manifest_url(record_id):
    """
    Generate the IIIF manifest URL for a record
    """
    return MANIFEST_URL.format(record_id=record_id)

def extract_page_ids_from_manifest(manifest):
    """
    Extract all page IDs from a IIIF manifest
    
    Parameters:
    -----------
    manifest : dict
        The IIIF manifest JSON
    
    Returns:
    --------
    list
        List of page IDs
    """
    page_ids = []
    
    sequences = manifest.get('sequences', [])
    for sequence in sequences:
        canvases = sequence.get('canvases', [])
        for canvas in canvases:
            images = canvas.get('images', [])
            for image in images:
                resource = image.get('resource', {})
                service = resource.get('service', {})
                if '@id' in service:
                    service_id = service['@id']
                    # Extract the ID from the format: https://www.e-rara.ch/i3f/v21/13465786
                    if '/i3f/v21/' in service_id:
                        page_id = service_id.split('/i3f/v21/')[1]
                        page_ids.append(page_id)
    
    return page_ids

def get_all_page_ids(record_id, timeout=30, retries=3):
    """
    Get all page IDs for a given record by fetching its IIIF manifest
    
    Parameters:
    -----------
    record_id : str
        The e-rara record ID
    timeout : int, optional
        Request timeout in seconds
    retries : int, optional
        Number of retry attempts
    
    Returns:
    --------
    dict
        Dictionary with manifest data and page IDs
    """
    manifest_url = get_manifest_url(record_id)
    
    for attempt in range(retries):
        try:
            response = requests.get(manifest_url, timeout=timeout)
            response.raise_for_status()
            
            manifest = response.json()
            
            page_ids = extract_page_ids_from_manifest(manifest)
            
            metadata = {}
            if 'label' in manifest:
                metadata['title'] = manifest['label']
                
            if 'metadata' in manifest:
                for item in manifest['metadata']:
                    if 'label' in item and 'value' in item:
                        metadata[item['label']] = item['value']
            
            return {
                'record_id': record_id,
                'manifest': manifest_url,
                'page_ids': page_ids,
                'page_count': len(page_ids),
                'metadata': metadata
            }
            
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            if attempt < retries - 1:
                time.sleep(1)
                continue
            else:
                return {
                    'record_id': record_id,
                    'error': str(e),
                    'page_ids': []
                }
    
    return {
        'record_id': record_id,
        'error': f"Failed to fetch manifest after {retries} attempts",
        'page_ids': []
    }

def get_all_page_ids_from_records(record_ids, max_workers=5, timeout=30, retries=3):
    """
    Get all page IDs from multiple records
    
    Parameters:
    -----------
    record_ids : list
        List of e-rara record IDs
    max_workers : int, optional
        Maximum number of parallel requests
    timeout : int, optional
        Request timeout in seconds
    retries : int, optional
        Number of retry attempts
        
    Returns:
    --------
    list
        List of all page IDs from all records
    """
    all_page_ids = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(get_all_page_ids, rec_id, timeout, retries) for rec_id in record_ids]
        
        for future in tqdm(futures, desc="Fetching manifests", unit="record"):
            try:
                result = future.result()
                page_ids = result.get('page_ids', [])
                all_page_ids.extend(page_ids)
            except Exception as e:
                print(f"Error processing record: {e}")
    
    return all_page_ids

def bytes2tensor(image_bytes):
    """
    Convert image bytes to a PyTorch tensor
    """
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = np.array(image).astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # Change to CxHxW format
    return torch.tensor(image)

def bytes2cv2(image_bytes):
    """
    Convert image bytes to a cv2 image
    """
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = np.array(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def load_target_image(target_image_path, size=",300"):
    """
    Load the target image, resize it based on height, and convert to tensor
    
    Parameters:
    -----------
    target_image_path : str
        Path to the target image file
    size : str, optional
        Size parameter for height (default: ",300" for 300px height)
        Can be: ",300", "400", "full" (no resize)
    preprocessing : bool, optional
        Whether to apply aggressive border removal (default: False)
        
    Returns:
    --------
    torch.Tensor
        Preprocessed image tensor in CxHxW format
    """
    if not os.path.exists(target_image_path):
        raise FileNotFoundError(f"Target image file not found: {target_image_path}")

    img = cv2.imread(target_image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image file: {target_image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if size.startswith(","):
        target_height = int(size[1:])
        current_height, current_width = img.shape[:2]
        
        if current_height != target_height:
            aspect_ratio = current_width / current_height
            new_width = int(target_height * aspect_ratio)
            img = cv2.resize(img, (new_width, target_height), interpolation=cv2.INTER_AREA)
            
    elif size == "full":
        pass
    else:
        target_height = int(size)
        current_height, current_width = img.shape[:2]
        
        if current_height != target_height:
            aspect_ratio = current_width / current_height
            new_width = int(target_height * aspect_ratio)
            img = cv2.resize(img, (new_width, target_height), interpolation=cv2.INTER_AREA)

    # Convert to tensor: HxWxC -> CxHxW, normalize to [0,1]
    img = img.astype(np.float32) / 255.0
    torch_img = torch.tensor(img).permute(2, 0, 1)
    
    return torch_img

def download_tensor_img(record_id, size=",300", timeout=30, retries=3, preprocessing=True):
    """
    Download a thumbnail image and extract SIFT features WITHOUT saving to disk
    
    Parameters:
    -----------
    record_id : str
        The e-rara record ID
    size : str, optional
        Size parameter for IIIF (default: "300," for better SIFT extraction)
    timeout : int, optional
        Request timeout in seconds
    retries : int, optional
        Number of retry attempts
    max_features : int, optional
        Maximum number of SIFT features to extract
        
    Returns:
    --------
    dict
        Dictionary containing success status, features, and metadata
    """
    url = get_image_url(record_id, size)
    
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            
            numpy_img = bytes2cv2(response.content)
            
            if numpy_img is None:
                return False, f"Failed to decode image for {record_id}"

            numpy_img = cv2.cvtColor(numpy_img, cv2.COLOR_BGR2RGB)
            
            if numpy_img is None or numpy_img.size == 0:
                return False, f"Failed to process image for {record_id}: empty or invalid image data"
            
            numpy_img = numpy_img.astype(np.float32) / 255.0
            torch_img = torch.tensor(numpy_img).permute(2, 0, 1)
            return True, torch_img
            
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                time.sleep(1)
                continue
            else:
                return False, f"Error downloading {record_id}: {str(e)}"
        except Exception as e:
            return False, f"Error processing image for {record_id}: {str(e)}"
    
    return False, f"Failed to download {record_id} after {retries} attempts"
    
def download_full_image(record_id, output_dir, size="full", timeout=30, retries=3):
    """
    Download a full-resolution image and save to disk
    
    Parameters:
    -----------
    record_id : str
        The e-rara record ID
    output_dir : str
        Directory to save the image
    size : str, optional
        Size parameter for IIIF (default: "full")
    timeout : int, optional
        Request timeout in seconds
    retries : int, optional
        Number of retry attempts
        
    Returns:
    --------
    bool
        True if download was successful, False otherwise
    str
        Path to the downloaded image or error message
    """
    url = get_image_url(record_id, size)
    filename = f"{record_id}.jpg"
    output_path = os.path.join(output_dir, filename)
    
    if os.path.exists(output_path):
        return True, output_path
    
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                f.write(response.content)
                
            return True, output_path
            
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                time.sleep(1)
                continue
            else:
                return False, f"Error downloading {record_id}: {str(e)}"
    
    return False, f"Failed to download {record_id} after {retries} attempts"

def download_selected_candidates(candidates, output_dir, size="full", max_workers=5, timeout=30, retries=3):
    """
    Download ONLY the selected candidate images at full resolution
    
    Parameters:
    -----------
    candidates : list
        List of candidate dictionaries from filter_by_target_similarity
    output_dir : str
        Directory to save the images
    size : str, optional
        Size parameter for IIIF (default: "full")
    max_workers : int, optional
        Maximum number of parallel downloads
    timeout : int, optional
        Request timeout in seconds
    retries : int, optional
        Number of retry attempts
        
    Returns:
    --------
    dict
        Dictionary with download results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    download_results = {
        "successful": 0,
        "failed": 0,
        "errors": [],
        "downloaded_files": []
    }
    
    def _download_candidate(candidate):
        record_id = candidate['record_id']
        success, result = download_full_image(record_id, output_dir, size, timeout, retries)
        
        if success:
            return {
                'success': True,
                'record_id': record_id,
                'file_path': result,
                # 'good_matches': candidate['good_matches']
            }
        else:
            return {
                'success': False,
                'record_id': record_id,
                'error': result,
                # 'good_matches': candidate['good_matches']
            }
    
    print(f"Downloading {len(candidates)} selected candidates at full resolution...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_download_candidate, candidate) for candidate in candidates]
        
        for future in tqdm(futures, desc="Downloading full images", unit="image"):
            try:
                result = future.result()
                if result['success']:
                    download_results["successful"] += 1
                    download_results["downloaded_files"].append(result)
                else:
                    download_results["failed"] += 1
                    download_results["errors"].append(result)
            except Exception as e:
                download_results["failed"] += 1
                download_results["errors"].append(f"Unexpected error: {str(e)}")
    
    return download_results

def batch_download_with_target_filtering(record_ids, output_dir,
                                         expand_to_pages=True, max_workers=5, size=",300",
                                         timeout=30, retries=3):
    """
    Complete pipeline: filter images based on matcher inliers and download ONLY candidates
    Parameters:
    -----------
    record_ids : list
        List of e-rara record IDs
    output_dir : str
        Directory to save the images
    min_matches : int, optional
        Minimum number of good SIFT matches required
    min_similarity_score : float, optional
        Minimum similarity score required
    max_candidates : int, optional
        Maximum number of candidates to download
    expand_to_pages : bool, optional
        Whether to expand record IDs to individual page IDs
    size : str, optional
        Size parameter for IIIF (default: "full")
    max_workers : int, optional
        Maximum number of parallel downloads
    timeout : int, optional
        Request timeout in seconds
    retries : int, optional
        Number of retry attempts
    max_features : int, optional
        Maximum number of SIFT features to extract per image
        
    Returns:
    --------
    dict
        Dictionary with complete results
    """

    if expand_to_pages:
        print("Expanding record IDs to individual page IDs...")
        all_page_ids = get_all_page_ids_from_records(record_ids, max_workers, timeout, retries)
        ids_to_process = all_page_ids
        print(f"Expanded {len(record_ids)} records to {len(all_page_ids)} pages")
    else:
        ids_to_process = record_ids
        print(f"Processing {len(ids_to_process)} record IDs directly")

    ids_to_process = [{'record_id': rid} for rid in ids_to_process]

    downloaded_images = download_selected_candidates(ids_to_process, output_dir, size, max_workers, timeout, retries)
        
    return downloaded_images

def main():
    parser = argparse.ArgumentParser(description='Download images from e-rara filtered by SIFT similarity to target image')
    parser.add_argument('ids_file', help='File containing record IDs (one per line) or JSON file with "ids" key')
    parser.add_argument('--output-dir', default='e_rara_candidates', help='Output directory for downloaded images')
    parser.add_argument('--size', default=',300', help='Size parameter for IIIF (default: full)')
    parser.add_argument('--max-workers', type=int, default=5, help='Maximum number of parallel downloads')
    parser.add_argument('--no-expand', action='store_true',
                        help='Do not expand record IDs to individual pages (use record IDs directly)')
    parser.add_argument('--timeout', type=int, default=30, help='Request timeout in seconds')
    parser.add_argument('--retries', type=int, default=3, help='Number of retry attempts')
    
    args = parser.parse_args()

    record_ids = []
    try:
        if args.ids_file.endswith('.json'):
            with open(args.ids_file, 'r') as f:
                data = json.load(f)
                record_ids = data.get('ids', [])
        else:
            with open(args.ids_file, 'r') as f:
                record_ids = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error loading record IDs: {str(e)}")
        return
    
    print(f"Loaded {len(record_ids)} record IDs")
    results = batch_download_with_target_filtering(
        record_ids,
        args.output_dir,
        expand_to_pages=not args.no_expand,
        max_workers=args.max_workers,
        size=args.size,
        timeout=args.timeout,
        retries=args.retries
    )

    if 'error' in results:
        print(f"Error: {results['error']}")
        return
    
    print(f"\nDownload complete:"
          f"\n  Successful downloads: {results['successful']}"
          f"\n  Failed downloads: {results['failed']}")

if __name__ == "__main__":
    main()
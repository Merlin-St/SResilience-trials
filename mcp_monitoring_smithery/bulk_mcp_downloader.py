# %%
# =============================================================================
# Core Logic for Bulk Downloading All MCP Server Data from Smithery API
# =============================================================================
import requests
import time
import json
import concurrent.futures
from mcp_monitoring_smithery.bulk_mcp_config import ( # Assuming you are using bulk_mcp_config.py
    bulk_logger as logger, MCP_API_BASE_URL, MCP_REQUEST_TIMEOUT, 
    API_DELAY_BULK, MCP_PAGE_SIZE_BULK, MCP_MAX_WORKERS_BULK
)
# If you switched to config_utils.py, change the import above to:
# from config_utils import (
#     logger, MCP_API_BASE_URL, MCP_REQUEST_TIMEOUT,
#     API_DELAY_BULK_LIST as API_DELAY_BULK, # Ensure variable names match
#     MCP_PAGE_SIZE_BULK, MCP_MAX_WORKERS_BULK
# )


def _fetch_server_list_page(api_key, page, page_size, base_url):
    """Helper to fetch one page from the /servers endpoint (no query)."""
    endpoint = f"{base_url}/servers"
    headers = {'Authorization': f'Bearer {api_key}', 'Accept': 'application/json'}
    params = {'page': page, 'pageSize': page_size}
    
    logger.debug(f"Fetching server list page: {page}, pageSize: {page_size}")
    try:
        response = requests.get(endpoint, headers=headers, params=params, timeout=MCP_REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error fetching server list page {page}: {e.response.status_code} - {e.response.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error fetching server list page {page}: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for server list page {page}: {e}. Response: {response.text[:200] if response else 'No response'}")
    return None

def get_all_server_summaries(api_key, base_url=MCP_API_BASE_URL, 
                             page_size=MCP_PAGE_SIZE_BULK, delay=API_DELAY_BULK):
    """
    Paginates through the /servers endpoint (no query) to get all server summaries.
    Returns a list of server summary objects.
    """
    all_summaries = []
    current_page = 1
    total_pages = 1 # Initialize to enter loop

    logger.info(f"Starting to fetch all server summaries. Page size: {page_size}")
    while current_page <= total_pages:
        logger.info(f"Fetching server summaries page {current_page} of {total_pages if total_pages > 1 else 'unknown'}...")
        page_data = _fetch_server_list_page(api_key, current_page, page_size, base_url)
        
        if page_data and 'servers' in page_data:
            summaries_on_page = page_data['servers']
            # Ensure all summaries are dicts, filter out if not (highly unlikely from API)
            summaries_on_page = [s for s in summaries_on_page if isinstance(s, dict)]
            all_summaries.extend(summaries_on_page)
            
            pagination_info = page_data.get('pagination', {})
            if current_page == 1: 
                total_pages = pagination_info.get('totalPages', current_page)
                logger.info(f"Total pages to fetch for summaries: {total_pages}. Total servers (approx): {pagination_info.get('totalCount', 'N/A')}")

            logger.info(f"Fetched {len(summaries_on_page)} summaries from page {current_page}. Total summaries so far: {len(all_summaries)}")
            current_page += 1
            if current_page <= total_pages: 
                time.sleep(delay)
        else:
            logger.error(f"Failed to fetch page {current_page} or no 'servers' field in response. Stopping summary collection.")
            break
            
    logger.info(f"Finished fetching all server summaries. Total collected: {len(all_summaries)}")
    return all_summaries

def _fetch_single_server_details_and_merge(server_summary, api_key, base_url):
    """
    Fetches details for one server from /servers/{qualifiedName} and merges with its summary.
    Ensures all fields from summary are kept, and detail fields are added/updated.
    """
    qualified_name = server_summary.get('qualifiedName')
    if not qualified_name:
        logger.warning(f"Server summary missing qualifiedName, cannot fetch details: {str(server_summary)[:200]}")
        return {**server_summary, "error_fetching_details": True, "error_message": "Missing qualifiedName in summary"}

    # Start with a complete copy of the summary data. This is the baseline.
    final_data = server_summary.copy()
    final_data["error_fetching_details"] = False 
    final_data["error_message"] = None

    endpoint = f"{base_url}/servers/{qualified_name}"
    headers = {'Authorization': f'Bearer {api_key}', 'Accept': 'application/json'}
    
    try:
        response = requests.get(endpoint, headers=headers, timeout=MCP_REQUEST_TIMEOUT)
        response.raise_for_status()
        detail_data = response.json()

        if detail_data and isinstance(detail_data, dict):
            # Iterate through all keys from the detail_data.
            # If a key exists in detail_data, its value will be used in final_data.
            # This means if a field like 'displayName' is in both, the one from 'detail_data' takes precedence.
            # If a field like 'description' is ONLY in summary_data (and not in detail_data), it remains untouched.
            # If a field like 'tools' is ONLY in detail_data, it gets added.
            for key, value in detail_data.items():
                final_data[key] = value
        
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error fetching details for '{qualified_name}': {e.response.status_code} - {e.response.text}")
        final_data["error_fetching_details"] = True
        final_data["error_message"] = f"HTTP Error: {e.response.status_code}"
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error fetching details for '{qualified_name}': {e}")
        final_data["error_fetching_details"] = True
        final_data["error_message"] = f"Request Error: {str(e)}"
    except json.JSONDecodeError as e:
        response_text = response.text if 'response' in locals() and hasattr(response, 'text') else "No response text available"
        logger.error(f"JSON decode error for details of '{qualified_name}': {e}. Response: {response_text[:200]}")
        final_data["error_fetching_details"] = True
        final_data["error_message"] = f"JSON Decode Error: {str(e)}"
    
    return final_data


def get_details_for_all_servers(server_summaries, api_key, base_url=MCP_API_BASE_URL, 
                                max_workers=MCP_MAX_WORKERS_BULK):
    """
    Fetches full details for a list of server summaries by merging with detail endpoint data.
    Uses ThreadPoolExecutor for concurrent requests.
    Returns a list of combined server data objects.
    """
    all_combined_server_data = []

    if not server_summaries:
        logger.warning("No server summaries provided to fetch details for.")
        return []
    
    # Ensure server_summaries contains dicts with qualifiedName
    valid_summaries_to_fetch = [
        s for s in server_summaries if isinstance(s, dict) and 'qualifiedName' in s and isinstance(s['qualifiedName'], str)
    ]
    
    if not valid_summaries_to_fetch:
        logger.warning("No valid server summaries (dict with qualifiedName) found to fetch details for.")
        return []

    total_summaries = len(valid_summaries_to_fetch)
    logger.info(f"Starting to fetch and merge full details for {total_summaries} servers using up to {max_workers} workers...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_summary = {
            executor.submit(_fetch_single_server_details_and_merge, summary, api_key, base_url): summary 
            for summary in valid_summaries_to_fetch
        }
        
        processed_count = 0
        for future in concurrent.futures.as_completed(future_to_summary):
            summary_obj = future_to_summary[future]
            qname_for_log = summary_obj.get('qualifiedName', 'Unknown qualifiedName')
            try:
                combined_data = future.result()
                if combined_data: 
                    all_combined_server_data.append(combined_data)
            except Exception as exc:
                logger.error(f"Exception processing future for {qname_for_log}: {exc}", exc_info=True)
                error_entry = summary_obj.copy() 
                error_entry["error_processing_future"] = True
                error_entry["exception_details"] = str(exc)
                all_combined_server_data.append(error_entry)
            
            processed_count += 1
            if processed_count % (max(1, total_summaries // 20)) == 0 or processed_count == total_summaries:
                logger.info(f"Details fetched/merged for {processed_count}/{total_summaries} servers...")

    logger.info(f"Finished fetching and merging all server details. Total combined records: {len(all_combined_server_data)}")
    return all_combined_server_data

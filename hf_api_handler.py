# %%
# =============================================================================
# Hugging Face API Interaction Functions
# =============================================================================
import time
import logging
from tqdm import tqdm # Using plain tqdm
from huggingface_hub import HfApi, list_models
import pandas as pd # Needed for DataFrame creation in run script

# Import logger and constants from config_utils
from config_utils import logger, API_DELAY, MODELS_PER_KEYWORD

# --- Data Collection Functions (Phase 2 Helpers) ---
def safe_get_models(api, search_term, limit, search_type='filter'):
    """Safely fetches models from HF Hub, handling potential errors."""
    try:
        if not isinstance(search_term, str) or not search_term:
             logger.warning(f"Invalid search term provided: {search_term}. Skipping.")
             return []
        # Use 'filter' for tag-based search as per original code logic
        if search_type == 'filter':
             # cardData=False and fetch_config=False for efficiency
             models_iterator = list_models(filter=search_term, limit=limit, sort="downloads", direction=-1, cardData=False, fetch_config=False)
        # Allow 'search' as an alternative, though 'filter' is primary
        elif search_type == 'search':
             models_iterator = list_models(search=search_term, limit=limit, sort="downloads", direction=-1, cardData=False, fetch_config=False)
        else:
             logger.error(f"Invalid search_type: {search_type}. Use 'filter' or 'search'.")
             return []
        models_list = list(models_iterator)
        return models_list
    except Exception as e:
        logger.error(f"API Error fetching '{search_term}' (type: {search_type}): {e}")
        return []

def collect_all_models(sector_data_map, models_per_keyword=MODELS_PER_KEYWORD, api_delay=API_DELAY):
    """Collects model data across all sectors and keywords (uses unique sector ID)."""
    if not sector_data_map:
        logger.error("Sector data map is empty. Cannot collect models.")
        return {}
    api = HfApi()
    try: user_info = api.whoami(); logger.info(f"Logged in to Hugging Face Hub as: {user_info.get('name', 'N/A')}")
    except Exception as e: logger.warning(f"Not logged in or error checking login status: {e}. Public access will be used.")

    all_models_data = {}
    total_keywords = sum(len(data['keywords']) for data in sector_data_map.values())
    logger.info(f"Starting data collection for {len(sector_data_map)} unique sectors, {total_keywords} total keywords...")

    # Use plain tqdm
    pbar_keywords = tqdm(total=total_keywords, desc="Processing Keywords")

    # sector_id is the unique "Code - Name" string
    for sector_id, sector_info in sector_data_map.items():
        keywords = sector_info['keywords']
        logger.debug(f"--- Starting Sector ID: {sector_id} ---")
        for keyword in keywords:
            if not isinstance(keyword, str) or not keyword:
                 logger.warning(f"Skipping invalid keyword '{keyword}' in sector '{sector_id}'"); pbar_keywords.update(1); continue
            pbar_keywords.set_description(f"Keyword: {keyword[:30]} ({sector_id[:35]})") # Truncate long names
            # Use 'filter' search type based on original logic (searching tags)
            logger.debug(f"  Fetching top {models_per_keyword} models for keyword: '{keyword}' using tag filter...")
            models_found = safe_get_models(api, keyword, models_per_keyword, search_type='filter')
            logger.debug(f"    Found {len(models_found)} models for keyword '{keyword}'.")
            for model_info in models_found:
                model_id = model_info.modelId; library_name = getattr(model_info, 'library_name', None)
                if not model_id: continue
                if model_id not in all_models_data:
                    all_models_data[model_id] = { "modelId": model_id, "sha": model_info.sha, "lastModified": model_info.lastModified,
                                                "tags": model_info.tags if model_info.tags else [], "pipeline_tag": model_info.pipeline_tag,
                                                "downloads": model_info.downloads if model_info.downloads else 0, "likes": model_info.likes if model_info.likes else 0,
                                                "library_name": library_name,
                                                "matched_sectors": [sector_id], # <-- Store unique sector ID
                                                "matched_keywords": [keyword] }
                    # NOTE: Add call to get_hf_model_extended_metadata here later for Request 3
                else:
                    # Append unique sector ID and keyword if not already present
                    if sector_id not in all_models_data[model_id]["matched_sectors"]:
                        all_models_data[model_id]["matched_sectors"].append(sector_id)
                    if keyword not in all_models_data[model_id]["matched_keywords"]:
                         all_models_data[model_id]["matched_keywords"].append(keyword)
            time.sleep(api_delay)
            pbar_keywords.update(1)
    pbar_keywords.close()
    logger.info(f"--- Finished collecting data. Found {len(all_models_data)} unique models across all keywords. ---")
    return all_models_data
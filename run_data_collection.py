# %%
# =============================================================================
# Script to Execute Data Collection (Phase 1 & 2)
# =============================================================================
import time
import pandas as pd
import logging # Need logging for potential errors during save

# Import necessary functions and constants
from config_utils import logger, NAICS_KEYWORDS_CSV, OUTPUT_CSV_FILE, MODELS_PER_KEYWORD, API_DELAY
from data_processing import load_naics_keywords
from hf_api_handler import collect_all_models

logger.info("--- Running Data Generation Script ---")

# Phase 1: Load Keywords
logger.info("Starting Phase 1: Load NAICS Keywords")
# load_naics_keywords now returns sector_data_map and sector_cni_map
sector_data_map_gen, _ = load_naics_keywords(NAICS_KEYWORDS_CSV) # Only need data map here
if not sector_data_map_gen:
    logger.error("Keyword loading failed. Stopping data generation.")
    # Optionally raise SystemExit or just exit
    exit() # Stop script if keywords fail
logger.info("Phase 1 Finished.")

# Phase 2: Collect Model Data
logger.info("Starting Phase 2: Collect Model Data")
start_time_phase2 = time.time()
# Pass the sector data map and configured parameters to the collector function
raw_model_data_dict = collect_all_models(sector_data_map_gen, MODELS_PER_KEYWORD, API_DELAY)
# NOTE: Add Frontier AI data fetching call within collect_all_models later
end_time_phase2 = time.time()
logger.info(f"Phase 2 Finished in {end_time_phase2 - start_time_phase2:.2f} seconds.")

# Save collected data to CSV
if raw_model_data_dict:
    logger.info(f"Saving raw collected data ({len(raw_model_data_dict)} models) to {OUTPUT_CSV_FILE}...")
    raw_models_list = list(raw_model_data_dict.values())
    raw_df_to_save = pd.DataFrame(raw_models_list)
    try:
        # Ensure list columns are saved as strings correctly for later parsing
        for col in ['tags', 'matched_sectors', 'matched_keywords']:
             if col in raw_df_to_save.columns:
                  raw_df_to_save[col] = raw_df_to_save[col].astype(str)
        raw_df_to_save.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8')
        logger.info(f"Raw data successfully saved to {OUTPUT_CSV_FILE}")
    except Exception as e:
        logger.error(f"Error saving raw data to CSV {OUTPUT_CSV_FILE}: {e}", exc_info=True)
else:
    logger.warning("No models collected in Phase 2. CSV file may be missing or empty.")

logger.info("--- Data Generation Script Finished ---")
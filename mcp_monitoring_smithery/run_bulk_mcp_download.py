# %%
# =============================================================================
# Main Script to Run Bulk Download of All MCP Server Data
# =============================================================================
import json
import time
from bulk_mcp_config import ( # Assuming you are using bulk_mcp_config.py
    bulk_logger as logger, SMITHERY_API_TOKEN, MCP_API_BASE_URL,
    ALL_SERVERS_SUMMARIES_JSON, ALL_SERVERS_DETAILS_COMPLETE_JSON,
    MCP_PAGE_SIZE_BULK, API_DELAY_BULK, MCP_MAX_WORKERS_BULK
)
# If you switched to config_utils.py, make sure to import the correct variables
# e.g., API_DELAY_BULK_LIST as API_DELAY_BULK
from bulk_mcp_downloader import get_all_server_summaries, get_details_for_all_servers

def main_bulk_download():
    logger.info("--- Starting Bulk MCP Server Data Download Process ---")
    start_time_total = time.time()

    if not SMITHERY_API_TOKEN:
        logger.error("SMITHERY_API_TOKEN is not set. Please configure it in bulk_mcp_config.py or as an environment variable.")
        logger.error("Aborting bulk download.")
        return

    # --- Step 1: Fetch all server summaries ---
    logger.info("Step 1: Fetching all server summaries...")
    start_time_summaries = time.time()
    all_summaries = get_all_server_summaries(
        api_key=SMITHERY_API_TOKEN,
        base_url=MCP_API_BASE_URL,
        page_size=MCP_PAGE_SIZE_BULK,
        delay=API_DELAY_BULK # This should be API_DELAY_BULK_LIST if using config_utils.py
    )
    end_time_summaries = time.time()
    logger.info(f"Fetched {len(all_summaries)} server summaries in {end_time_summaries - start_time_summaries:.2f} seconds.")

    if not all_summaries:
        logger.error("No server summaries were fetched. Cannot proceed to download details. Exiting.")
        return

    # Optional: Save summaries to a file
    try:
        with open(ALL_SERVERS_SUMMARIES_JSON, 'w') as f:
            json.dump(all_summaries, f, indent=2)
        logger.info(f"All server summaries saved to {ALL_SERVERS_SUMMARIES_JSON}")
    except IOError as e:
        logger.error(f"Error saving server summaries to {ALL_SERVERS_SUMMARIES_JSON}: {e}")


    # --- Step 2: Fetch full details for all servers ---
    logger.info("\nStep 2: Fetching full details for all collected server summaries...")
    start_time_details = time.time()
    
    # CRITICAL FIX: Pass the list of summary dictionaries directly
    # The get_details_for_all_servers function will handle extracting qnames internally if needed
    # or more accurately, its helper _fetch_single_server_details_and_merge expects the summary dict.
    
    # No need to pre-extract qualified_names_to_fetch here for the main call.
    # The validation for 'dict with qualifiedName' is inside get_details_for_all_servers.
    logger.info(f"Attempting to fetch details for {len(all_summaries)} server summaries.")
    
    all_server_details_list = get_details_for_all_servers(
        server_summaries=all_summaries, # Pass the list of summary dictionaries
        api_key=SMITHERY_API_TOKEN,
        base_url=MCP_API_BASE_URL,
        max_workers=MCP_MAX_WORKERS_BULK
    )
    end_time_details = time.time()
    logger.info(f"Fetched details for {len(all_server_details_list)} servers in {end_time_details - start_time_details:.2f} seconds.")

    if not all_server_details_list:
        logger.warning("No server details were successfully fetched.")
    else:
        # Save the complete details to a JSON file
        try:
            with open(ALL_SERVERS_DETAILS_COMPLETE_JSON, 'w') as f:
                json.dump(all_server_details_list, f, indent=2)
            logger.info(f"All server details saved to {ALL_SERVERS_DETAILS_COMPLETE_JSON}")
        except IOError as e:
            logger.error(f"Error saving server details to {ALL_SERVERS_DETAILS_COMPLETE_JSON}: {e}")

    end_time_total = time.time()
    logger.info(f"--- Bulk MCP Server Data Download Process Finished in {end_time_total - start_time_total:.2f} seconds ---")

if __name__ == "__main__":
    main_bulk_download()

# %%
# =============================================================================
# MCP Server Analysis Script (Robust Version)
# =============================================================================
import pandas as pd
import re
import json
import nltk # NLTK itself for PorterStemmer
from nltk.stem import PorterStemmer
import logging # Standard logging
import os # For file existence check in test block

# --- Global variable to indicate NLTK 'punkt' status ---
PUNKT_AVAILABLE = False
word_tokenize_func = None

# --- Setup Logger ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Attempt to import custom configurations and update logger if available ---
try:
    from hf_models_monitoring_test.config_utils import logger as custom_logger
    logger = custom_logger
    logger.info("Successfully imported custom logger from hf_models_monitoring_test.config_utils.")
except ImportError:
    logger.warning("Could not import custom logger from 'hf_models_monitoring_test.config_utils'. Using standard Python logger.")

# Initialize BULK_MCP_DETAILS_JSON_FILE with a default, to be overridden by import if successful
BULK_MCP_DETAILS_JSON_FILE = "all_mcp_server_details_complete.json" # Default path

# --- MODIFIED IMPORT ---
# Import directly from bulk_mcp_config.py as it's in the same directory
# when running streamlit from mcp_monitoring_smithery
try:
    from bulk_mcp_config import ( # MODIFIED HERE
        FINANCE_SECTOR_KEYWORDS_CONFIG as FS_CONFIG,
        THREAT_MODEL_KEYWORDS_CONFIG as TM_CONFIG,
        FINANCE_AFFORDANCE_KEYWORDS_CONFIG as FA_CONFIG,
        ALL_SERVERS_DETAILS_COMPLETE_JSON as IMPORTED_BULK_FILE_PATH
    )
    logger.info("Successfully imported keyword configurations and data file path from local 'bulk_mcp_config.py'.")
    FS_CONFIG = FS_CONFIG if FS_CONFIG is not None else {}
    TM_CONFIG = TM_CONFIG if TM_CONFIG is not None else {}
    FA_CONFIG = FA_CONFIG if FA_CONFIG is not None else {}
    BULK_MCP_DETAILS_JSON_FILE = IMPORTED_BULK_FILE_PATH
except ImportError as e:
    logger.error(f"Could not import configurations from local 'bulk_mcp_config.py'. Error: {e}. "
                   "Analysis will proceed with empty keyword sets and default data file path. THIS WILL RESULT IN NO MATCHES.")
    FS_CONFIG = {}
    TM_CONFIG = {}
    FA_CONFIG = {}

# --- NLTK Setup ---
try:
    from nltk.tokenize import word_tokenize
    try:
        # Test if 'punkt' is available by trying to tokenize a sentence
        word_tokenize("test sentence for punkt availability")
        word_tokenize_func = word_tokenize
        PUNKT_AVAILABLE = True
        logger.info("NLTK 'punkt' tokenizer is available and working.")
    except LookupError:
        logger.warning("NLTK 'punkt' resource not found. Attempting to download 'punkt'...")
        try:
            nltk.download('punkt', quiet=True)
            # Verify after download
            word_tokenize("test sentence post-download")
            word_tokenize_func = word_tokenize
            PUNKT_AVAILABLE = True
            logger.info("NLTK 'punkt' downloaded and tokenizer is now available.")
        except Exception as download_err:
            logger.error(f"Failed to download or use NLTK 'punkt' after attempt: {download_err}. "
                           "Please try installing it manually in your Python environment by running: "
                           "import nltk; nltk.download('punkt')\n"
                           "Falling back to basic whitespace tokenizer. Keyword matching accuracy might be reduced.")
            PUNKT_AVAILABLE = False
    except Exception as e:
        logger.error(f"An unexpected error occurred during NLTK word_tokenize setup: {e}. Falling back to basic whitespace tokenizer.")
        PUNKT_AVAILABLE = False

except ImportError:
    logger.error("Failed to import 'nltk.tokenize'. NLTK might not be installed. "
                   "Please install it and try: import nltk; nltk.download('punkt')\n"
                   "Falling back to basic whitespace tokenizer.")
    PUNKT_AVAILABLE = False

if not PUNKT_AVAILABLE or word_tokenize_func is None: # Ensure fallback if any step failed
    def whitespace_tokenizer(text_input_ws):
        if not isinstance(text_input_ws, str): return []
        return text_input_ws.lower().split()
    word_tokenize_func = whitespace_tokenizer
    if PUNKT_AVAILABLE:
        logger.warning("word_tokenize_func was not set despite PUNKT_AVAILABLE=True. Using fallback tokenizer.")
    else:
        logger.info("Using fallback whitespace tokenizer because NLTK 'punkt' is unavailable.")


stemmer = PorterStemmer()

# --- Helper Functions ---
def stem_text(text_to_stem):
    """
    Stems the input text using PorterStemmer.
    Uses NLTK's word_tokenize if 'punkt' is available, otherwise splits by whitespace.
    Args:
        text_to_stem (str): The text to stem.
    Returns:
        str: The stemmed text, or empty string if input is invalid.
    """
    if not isinstance(text_to_stem, str) or not word_tokenize_func:
        return ""
    
    try:
        words = word_tokenize_func(text_to_stem.lower())
        stemmed_words = [stemmer.stem(word) for word in words if word.isalnum()]
        return " ".join(stemmed_words)
    except Exception as e:
        logger.error(f"Error during stemming text: '{text_to_stem[:50]}...': {e}")
        return ""

def get_text_from_server(server_data_item):
    """
    Extracts and combines relevant text fields from server data for keyword matching.
    Includes name, description, and tool names/descriptions.
    Args:
        server_data_item (dict): A dictionary representing a single server's data.
    Returns:
        str: A single string containing all relevant text, or empty string if no data or invalid input.
    """
    if not isinstance(server_data_item, dict):
        return ""

    texts_to_search = []
    def append_if_str(value):
        if isinstance(value, str):
            texts_to_search.append(value)
        elif value is not None:
            logger.debug(f"Non-string value encountered in get_text_from_server: {type(value)}, {str(value)[:50]}")

    append_if_str(server_data_item.get('displayName'))
    append_if_str(server_data_item.get('description'))
    append_if_str(server_data_item.get('qualifiedName'))

    tools = server_data_item.get('tools', [])
    if isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, dict):
                append_if_str(tool.get('name'))
                append_if_str(tool.get('description'))
    elif tools:
        q_name = server_data_item.get('qualifiedName', 'UnknownServer')
        logger.warning(f"Server {q_name} has 'tools' field that is not a list: {type(tools)}. Skipping tools text for this server.")

    return " ".join(filter(None, texts_to_search)).lower()

# --- Main Analysis Functions ---
def match_server_to_keywords_stemmed(df_input, keyword_configs_map_input):
    """
    Matches servers to keyword categories using stemmed keywords and text.
    Calculates a score based on the number of unique stemmed keywords matched per category.
    Args:
        df_input (pd.DataFrame): DataFrame of servers. Must contain 'server_data' column with dicts.
        keyword_configs_map_input (dict): A dictionary where keys are new column names
                                    (e.g., 'matched_finance_sectors') and values are
                                    keyword configuration dicts (e.g., FS_CONFIG).
    Returns:
        pd.DataFrame: The input DataFrame with added columns for matched categories, their scores,
                      and the specific keywords that matched.
    """
    logger.info("Starting keyword matching for servers (stemmed)...")
    if 'server_data' not in df_input.columns:
        logger.error("'server_data' column not found in DataFrame. Cannot perform keyword matching.")
        if isinstance(keyword_configs_map_input, dict):
            for new_col_name_outer in keyword_configs_map_input.keys():
                df_input[new_col_name_outer] = [[] for _ in range(len(df_input))]
                df_input[f"{new_col_name_outer}_scores"] = [{} for _ in range(len(df_input))]
                df_input[f"{new_col_name_outer}_matched_keywords"] = [{} for _ in range(len(df_input))] # ADDED
        return df_input
    
    if not isinstance(keyword_configs_map_input, dict) or not keyword_configs_map_input:
        logger.warning("Keyword configurations map is empty or invalid. No keyword matching will be performed.")
        return df_input

    stemmed_keyword_configs_map = {}
    for col_name, config in keyword_configs_map_input.items():
        if not isinstance(config, dict) or not config: 
            logger.warning(f"Configuration for '{col_name}' is not a dictionary or is empty. Skipping this category set.")
            df_input[col_name] = [[] for _ in range(len(df_input))]
            df_input[f"{col_name}_scores"] = [{} for _ in range(len(df_input))]
            df_input[f"{col_name}_matched_keywords"] = [{} for _ in range(len(df_input))] # ADDED
            continue

        stemmed_config = {}
        for category, keywords in config.items():
            if isinstance(keywords, list):
                stemmed_config[category] = [stem_text(kw) for kw in keywords if isinstance(kw, str) and kw.strip()]
            else:
                logger.warning(f"Keywords for category '{category}' in '{col_name}' is not a list. Skipping.")
        if stemmed_config: 
            stemmed_keyword_configs_map[col_name] = stemmed_config
        else: 
            logger.warning(f"Configuration for '{col_name}' resulted in no valid stemmed keywords. Skipping this category set.")
            df_input[col_name] = [[] for _ in range(len(df_input))]
            df_input[f"{col_name}_scores"] = [{} for _ in range(len(df_input))]
            df_input[f"{col_name}_matched_keywords"] = [{} for _ in range(len(df_input))] # ADDED


    for new_col_name, stemmed_keyword_config in stemmed_keyword_configs_map.items():
        logger.info(f"Processing keyword set for column: {new_col_name}")
        matched_categories_for_servers = []
        match_scores_for_servers = []
        matched_keywords_details_for_servers = [] # ADDED

        for index, row in df_input.iterrows():
            server_data = row['server_data']
            current_server_matches = set()
            current_server_scores = {}
            current_server_matched_keywords = {} # ADDED

            if not isinstance(server_data, dict):
                logger.warning(f"Skipping row {index} due to invalid server_data (not a dict).")
            else:
                text_to_search = get_text_from_server(server_data)
                stemmed_text_to_search = stem_text(text_to_search)

                if stemmed_text_to_search.strip():
                    for category, stemmed_keywords in stemmed_keyword_config.items():
                        matched_keyword_count_for_category = 0
                        keywords_found_for_category = set() # ADDED
                        for stemmed_keyword in stemmed_keywords:
                            if not stemmed_keyword: continue
                            if re.search(r'\b' + re.escape(stemmed_keyword) + r'\b', stemmed_text_to_search, re.IGNORECASE):
                                current_server_matches.add(category)
                                matched_keyword_count_for_category +=1
                                keywords_found_for_category.add(stemmed_keyword) # ADDED
                        
                        if matched_keyword_count_for_category > 0:
                            current_server_scores[category] = matched_keyword_count_for_category
                            current_server_matched_keywords[category] = list(keywords_found_for_category) # ADDED
            
            matched_categories_for_servers.append(list(current_server_matches))
            match_scores_for_servers.append(current_server_scores)
            matched_keywords_details_for_servers.append(current_server_matched_keywords) # ADDED

        df_input[new_col_name] = matched_categories_for_servers
        df_input[f"{new_col_name}_scores"] = match_scores_for_servers
        df_input[f"{new_col_name}_matched_keywords"] = matched_keywords_details_for_servers # ADDED
        logger.info(f"Finished processing for {new_col_name}. Added columns.")
    
    logger.info("Keyword matching (stemmed) complete.")
    return df_input


def analyze_server_affordances(df_input, affordance_keyword_config_input):
    """
    Analyzes server tools for financial affordances using stemmed keywords.
    Args:
        df_input (pd.DataFrame): DataFrame of servers. Must contain 'server_data' column.
        affordance_keyword_config_input (dict): Keywords for different affordance types.
    Returns:
        pd.DataFrame: DataFrame with added boolean columns for each affordance type, lists of matched tools,
                      and lists of matched keywords for each affordance.
    """
    logger.info("Starting server financial affordance analysis (stemmed)...")
    default_aff_types_for_schema = ['execution', 'information_gathering', 'agent_interaction']

    if 'server_data' not in df_input.columns:
        logger.error("'server_data' column not found. Cannot perform affordance analysis.")
        for aff_type_outer in default_aff_types_for_schema: 
            df_input[f"has_finance_{aff_type_outer}"] = False
            df_input[f"finance_{aff_type_outer}_tools"] = [[] for _ in range(len(df_input))]
            df_input[f"finance_{aff_type_outer}_matched_keywords"] = [[] for _ in range(len(df_input))] # ADDED
        return df_input

    if not isinstance(affordance_keyword_config_input, dict) or not affordance_keyword_config_input:
        logger.warning("Affordance keyword configuration is empty or invalid. Skipping affordance analysis.")
        for aff_type_default in default_aff_types_for_schema:
            if f"has_finance_{aff_type_default}" not in df_input.columns:
                 df_input[f"has_finance_{aff_type_default}"] = False
            if f"finance_{aff_type_default}_tools" not in df_input.columns:
                 df_input[f"finance_{aff_type_default}_tools"] = [[] for _ in range(len(df_input))]
            if f"finance_{aff_type_default}_matched_keywords" not in df_input.columns: # ADDED
                 df_input[f"finance_{aff_type_default}_matched_keywords"] = [[] for _ in range(len(df_input))] # ADDED
        return df_input


    stemmed_affordance_config = {}
    for affordance_type, keywords in affordance_keyword_config_input.items():
        if isinstance(keywords, list):
            stemmed_affordance_config[affordance_type] = [stem_text(kw) for kw in keywords if isinstance(kw, str) and kw.strip()]
        else:
            logger.warning(f"Keywords for affordance type '{affordance_type}' is not a list. Skipping this type.")

    active_aff_types = list(stemmed_affordance_config.keys())
    if not active_aff_types: 
        active_aff_types = default_aff_types_for_schema


    affordance_results = {f"has_finance_{aff_type}": [False] * len(df_input) for aff_type in active_aff_types}
    affordance_tool_matches = {f"finance_{aff_type}_tools": [[] for _ in range(len(df_input))] for aff_type in active_aff_types}
    affordance_matched_keywords = {f"finance_{aff_type}_matched_keywords": [set() for _ in range(len(df_input))] for aff_type in active_aff_types} # ADDED (use set for unique keywords per server)


    for index, row in df_input.iterrows():
        server_data = row['server_data']
        if not isinstance(server_data, dict):
            continue
        
        tools = server_data.get('tools', [])
        if not isinstance(tools, list): tools = []

        server_matched_tools_for_affordance = {aff_type: set() for aff_type in active_aff_types}
        # server_matched_keywords_for_affordance initialized per server inside affordance_matched_keywords

        for tool in tools:
            if not isinstance(tool, dict): continue

            tool_name = tool.get('name', '')
            tool_desc = tool.get('description', '')
            tool_text_combined = f"{tool_name} {tool_desc}"
            stemmed_tool_text = stem_text(tool_text_combined)

            if not stemmed_tool_text.strip():
                continue
            
            for affordance_type, stemmed_keywords in stemmed_affordance_config.items(): 
                for stemmed_keyword in stemmed_keywords:
                    if not stemmed_keyword: continue
                    if re.search(r'\b' + re.escape(stemmed_keyword) + r'\b', stemmed_tool_text, re.IGNORECASE):
                        if f"has_finance_{affordance_type}" in affordance_results:
                            affordance_results[f"has_finance_{affordance_type}"][index] = True
                        if affordance_type in server_matched_tools_for_affordance: 
                            server_matched_tools_for_affordance[affordance_type].add(tool_name if tool_name else "UnnamedTool")
                        if f"finance_{affordance_type}_matched_keywords" in affordance_matched_keywords: # ADDED
                            affordance_matched_keywords[f"finance_{affordance_type}_matched_keywords"][index].add(stemmed_keyword) # ADDED
        
        for aff_type in active_aff_types: 
             if f"finance_{aff_type}_tools" in affordance_tool_matches and aff_type in server_matched_tools_for_affordance:
                affordance_tool_matches[f"finance_{aff_type}_tools"][index] = list(server_matched_tools_for_affordance[aff_type])


    for col_name, results_list in affordance_results.items():
        df_input[col_name] = results_list
    for col_name, matched_tools_list_of_lists in affordance_tool_matches.items():
        df_input[col_name] = matched_tools_list_of_lists
    for col_name, matched_keywords_set_list in affordance_matched_keywords.items(): # ADDED
        df_input[col_name] = [list(s) for s in matched_keywords_set_list] # Convert sets to lists for DataFrame # ADDED
    
    for aff_type_schema in default_aff_types_for_schema:
        if f"has_finance_{aff_type_schema}" not in df_input.columns:
            df_input[f"has_finance_{aff_type_schema}"] = False
        if f"finance_{aff_type_schema}_tools" not in df_input.columns:
            df_input[f"finance_{aff_type_schema}_tools"] = [[] for _ in range(len(df_input))]
        if f"finance_{aff_type_schema}_matched_keywords" not in df_input.columns: # ADDED
            df_input[f"finance_{aff_type_schema}_matched_keywords"] = [[] for _ in range(len(df_input))] # ADDED


    logger.info("Server financial affordance analysis (stemmed) complete.")
    return df_input

def run_full_mcp_analysis(df_servers_raw):
    """
    Runs the full MCP analysis pipeline on the raw server data.
    Args:
        df_servers_raw (pd.DataFrame): DataFrame with a 'server_data' column.
    Returns:
        pd.DataFrame: Analyzed DataFrame with new columns.
    """
    logger.info("Starting full MCP analysis pipeline...")
    if not isinstance(df_servers_raw, pd.DataFrame):
        logger.error("Input to run_full_mcp_analysis is not a Pandas DataFrame. Returning empty DataFrame.")
        return pd.DataFrame()
    if df_servers_raw.empty:
        logger.warning("Input DataFrame to run_full_mcp_analysis is empty. Returning empty DataFrame.")
        return pd.DataFrame()

    df_analyzed = df_servers_raw.copy()

    if 'server_data' not in df_analyzed.columns:
        logger.error("Raw DataFrame must contain 'server_data' column. Analysis cannot proceed.")
        df_analyzed['error'] = "'server_data' column missing."
        expected_cols_on_error = [
            'qualifiedName', 'displayName', 'description', 'useCount', 'createdAt', 'toolCount',
            'matched_finance_sectors', 'matched_finance_sectors_scores', 'matched_finance_sectors_matched_keywords', # ADDED
            'matched_threat_models', 'matched_threat_models_scores', 'matched_threat_models_matched_keywords', # ADDED
            'has_finance_execution', 'finance_execution_tools', 'finance_execution_matched_keywords', # ADDED
            'has_finance_information_gathering', 'finance_information_gathering_tools', 'finance_information_gathering_matched_keywords', # ADDED
            'has_finance_agent_interaction', 'finance_agent_interaction_tools', 'finance_agent_interaction_matched_keywords' # ADDED
        ]
        for col in expected_cols_on_error: # Ensure these columns exist for schema consistency
            if col.endswith('_scores') or col.endswith('_tools') or col.endswith('_sectors') or col.endswith('_models') or col.endswith('_matched_keywords'): # UPDATED
                # Default to empty list for list-like columns (tools, sectors, models, keywords)
                # and empty dict for dict-like columns (scores)
                if col.endswith('_scores'):
                     df_analyzed[col] = [{} for _ in range(len(df_analyzed))]
                else:
                     df_analyzed[col] = [[] for _ in range(len(df_analyzed))]

            elif col in ['useCount', 'toolCount']:
                df_analyzed[col] = 0
            elif col.startswith('has_finance_'):
                df_analyzed[col] = False
            elif col == 'createdAt':
                df_analyzed[col] = pd.NaT
            else:
                df_analyzed[col] = 'Error: server_data missing'
        return df_analyzed


    logger.info("Extracting key fields from server_data...")
    df_analyzed['qualifiedName'] = df_analyzed['server_data'].apply(lambda x: x.get('qualifiedName', 'N/A') if isinstance(x, dict) else 'InvalidData')
    df_analyzed['displayName'] = df_analyzed['server_data'].apply(lambda x: x.get('displayName', 'N/A') if isinstance(x, dict) else 'InvalidData')
    df_analyzed['description'] = df_analyzed['server_data'].apply(lambda x: x.get('description', '') if isinstance(x, dict) else '')
    df_analyzed['useCount'] = df_analyzed['server_data'].apply(lambda x: x.get('useCount', 0) if isinstance(x, dict) else 0).fillna(0).astype(int)
    df_analyzed['createdAt'] = pd.to_datetime(df_analyzed['server_data'].apply(lambda x: x.get('createdAt') if isinstance(x, dict) else None), errors='coerce')
    df_analyzed['toolCount'] = df_analyzed['server_data'].apply(lambda x: len(x.get('tools', [])) if isinstance(x, dict) and isinstance(x.get('tools'), list) else 0)

    keyword_configs_to_use = {
        'matched_finance_sectors': FS_CONFIG,
        'matched_threat_models': TM_CONFIG
    }
    
    if not FS_CONFIG:
        logger.warning("FS_CONFIG (Finance Sector Keywords) is empty. 'matched_finance_sectors' will likely be empty.")
    if not TM_CONFIG:
        logger.warning("TM_CONFIG (Threat Model Keywords) is empty. 'matched_threat_models' will likely be empty.")


    df_analyzed = match_server_to_keywords_stemmed(df_analyzed, keyword_configs_to_use)
    df_analyzed = analyze_server_affordances(df_analyzed, FA_CONFIG)
    
    if 'toolCount' in df_analyzed.columns:
        servers_with_tools = df_analyzed[df_analyzed['toolCount'] > 0].shape[0]
        total_servers = df_analyzed.shape[0]
        if total_servers > 0:
            coverage_percent = (servers_with_tools / total_servers) * 100
            logger.info(f"Tool data present for {servers_with_tools}/{total_servers} servers ({coverage_percent:.2f}% coverage).")
            if coverage_percent < 1.0 and servers_with_tools > 0 : 
                 logger.warning(f"Very low percentage of servers with tool details found ({coverage_percent:.2f}%). Affordance analysis might be incomplete.")
            elif servers_with_tools == 0: 
                logger.warning("No servers found with tool details. Affordance analysis will not find any tool-based affordances.")

        elif total_servers == 0:
             logger.info("No servers found in the input data for analysis (after initial load).")
    else:
        logger.warning("'toolCount' column not generated. Cannot assess tool data coverage.")

    logger.info("Full MCP analysis pipeline complete.")
    return df_analyzed

# %%
# =============================================================================
# Main execution block (for testing or direct script run)
# =============================================================================
if __name__ == '__main__':
    logger.info("Running MCP Analysis Script directly for testing...")
    logger.info(f"Attempting to load test data from: {BULK_MCP_DETAILS_JSON_FILE}")

    df_raw_test = pd.DataFrame()

    if os.path.exists(BULK_MCP_DETAILS_JSON_FILE):
        try:
            with open(BULK_MCP_DETAILS_JSON_FILE, 'r') as f:
                raw_data_list_test = json.load(f)
            if isinstance(raw_data_list_test, list):
                df_raw_test = pd.DataFrame({'server_data': raw_data_list_test})
                logger.info(f"Successfully loaded {len(df_raw_test)} records from {BULK_MCP_DETAILS_JSON_FILE} for testing.")
            else:
                logger.error(f"Data in {BULK_MCP_DETAILS_JSON_FILE} is not a list. Cannot use for testing.")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error from {BULK_MCP_DETAILS_JSON_FILE} during testing: {e}")
        except Exception as e:
            logger.error(f"Failed to load or process {BULK_MCP_DETAILS_JSON_FILE} for testing: {e}")
    else:
        logger.warning(f"Test data file '{BULK_MCP_DETAILS_JSON_FILE}' not found. "
                       "Proceeding with an empty DataFrame for testing. Analysis will be minimal.")

    logger.info(f"Using FS_CONFIG for test: {'Populated' if FS_CONFIG else 'Empty'}")
    logger.info(f"Using TM_CONFIG for test: {'Populated' if TM_CONFIG else 'Empty'}")
    logger.info(f"Using FA_CONFIG for test: {'Populated' if FA_CONFIG else 'Empty'}")
    
    if not df_raw_test.empty:
        df_analyzed_test = run_full_mcp_analysis(df_raw_test)

        logger.info(f"\n--- Analyzed DataFrame Head (Test Run) ---\n{df_analyzed_test.head().to_string() if not df_analyzed_test.empty else 'DataFrame is empty'}")
        
        cols_to_show = [
            'qualifiedName', 'displayName', 'toolCount',
            'matched_finance_sectors', 'matched_finance_sectors_scores', 'matched_finance_sectors_matched_keywords',
            'matched_threat_models', 'matched_threat_models_scores', 'matched_threat_models_matched_keywords',
            'has_finance_execution', 'finance_execution_tools', 'finance_execution_matched_keywords',
            'has_finance_information_gathering', 'finance_information_gathering_tools', 'finance_information_gathering_matched_keywords',
            'has_finance_agent_interaction', 'finance_agent_interaction_tools', 'finance_agent_interaction_matched_keywords'
        ]
        
        if not df_analyzed_test.empty:
            cols_to_show_existing = [col for col in cols_to_show if col in df_analyzed_test.columns]
            if cols_to_show_existing:
                 logger.info(f"\n--- Relevant Columns from Test Run ---\n{df_analyzed_test[cols_to_show_existing].to_string()}")
            else:
                logger.info("No relevant columns to show from the analyzed test data (they might be missing).")
        else:
            logger.info("Analyzed DataFrame from test run is empty.")
    else:
        logger.warning("Raw test DataFrame is empty (likely because test data file was not found or was empty). Skipping analysis for test run.")

    logger.info("\n--- NLTK 'punkt' availability status for this run ---")
    logger.info(f"PUNKT_AVAILABLE: {PUNKT_AVAILABLE}")
    logger.info(f"Using tokenizer: {'NLTK word_tokenize' if PUNKT_AVAILABLE and word_tokenize_func.__name__ == 'word_tokenize' else 'Fallback whitespace_tokenizer'}")
    logger.info("\n--- Testing complete. Check log for details. ---")
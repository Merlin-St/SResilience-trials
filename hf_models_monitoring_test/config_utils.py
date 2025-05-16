# %%
# =============================================================================
# Configuration, Logging, and Utilities
# =============================================================================
import logging
import os
import pandas as pd
import ast  # For safely evaluating string representations of lists
from io import StringIO # Needed for logging
import json # For parsing/storing complex data like tools

# --- General Configuration ---
NAICS_KEYWORDS_CSV = "naics_cni_percentages.csv" # Input for Phase 1 (HF & NAICS mapping)
OUTPUT_CSV_FILE = "huggingface_naics_sector_models_database.csv" # Output from HF Data Collection
LOG_FILE = "ai_monitoring_workflow.log" # General log file

# --- Hugging Face Specific Configuration ---
MODELS_PER_KEYWORD = 10 # Reduced for quicker test runs, adjust as needed
API_DELAY = 0.001         # Seconds to wait between API calls

# --- Smithery MCP Specific Configuration ---
with open(os.path.expanduser("~/.cache/smithery-api/token")) as f:
    SMITHERY_API_TOKEN = f.read().strip() # IMPORTANT: Set this environment variable in the setup file as aws secret
MCP_API_BASE_URL = "https://registry.smithery.ai"
MCP_OUTPUT_CSV_FILE = "mcp_servers_database.csv"
MCP_REQUEST_TIMEOUT = 30  # Seconds for API request timeout
MCP_PAGE_SIZE = 10000 # Number of items per page for Smithery API
MCP_MAX_WORKERS = 20 # Max concurrent workers for fetching server details

# --- Streamlit Log Handler (if needed for error display during loading) ---
log_stream = StringIO()

# --- Logging Setup ---
# Remove existing handlers if re-running cells in interactive environments
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure logging (log to console, file, and potentially stream)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
                    handlers=[
                        logging.FileHandler(LOG_FILE, mode='w'), # Overwrite log file each run
                        logging.StreamHandler(), # Also print logs to console
                        logging.StreamHandler(log_stream) # Keep handler for potential error display
                    ])

logger = logging.getLogger(__name__)

logger.info("Configuration and Logging Setup Complete.")

try:
    # Get current timestamp for display in the app
    CURRENT_DATE_STR = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
except Exception: # Fallback if timestamp fails
    CURRENT_DATE_STR = "N/A"

# --- Utility Functions ---
def parse_list_string(s):
    """Safely parse a string representation of a list or JSON array."""
    if isinstance(s, list): return s
    if pd.isna(s) or not isinstance(s, str): return []
    if s.startswith('[') and s.endswith(']'):
        try:
            return ast.literal_eval(s) # For simple lists like ['item1', 'item2']
        except (ValueError, SyntaxError, TypeError, MemoryError):
            try:
                # Try parsing as JSON if ast.literal_eval fails (e.g. for list of dicts)
                return json.loads(s)
            except (json.JSONDecodeError, TypeError) as e_json:
                logger.warning(f"Could not parse list/JSON string: '{str(s)[:100]}...' Errors: ast: N/A, json: {e_json}")
                return []
    return [s] # If it's not a list-like string, return it as a single-item list if non-empty

def parse_json_string(s, default_val=None):
    """Safely parse a string that should be a JSON object (dict)."""
    if isinstance(s, dict): return s
    if pd.isna(s) or not isinstance(s, str): return default_val if default_val is not None else {}
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Could not parse JSON string: '{str(s)[:100]}...' Error: {e}")
        return default_val if default_val is not None else {}
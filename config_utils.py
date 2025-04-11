# %%
# =============================================================================
# Configuration, Logging, and Utilities
# =============================================================================
import logging
import os
import pandas as pd
import ast  # For safely evaluating string representations of lists
from io import StringIO # Needed for logging

# --- Configuration ---
NAICS_KEYWORDS_CSV = "naics_cni_percentages.csv" # Input for Phase 1
OUTPUT_CSV_FILE = "huggingface_naics_sector_models_database.csv" # Output from Phase 2 / Input for Streamlit App
LOG_FILE = "hf_naics_workflow.log" # Log file on disk

MODELS_PER_KEYWORD = 100  # Target number of models per keyword (Adjust as needed for Phase 2)
API_DELAY = 0.01         # Seconds to wait between API calls (Adjust if rate limited)

# --- Streamlit Log Handler (if needed for error display during loading) ---
# Create a string buffer to capture logs
log_stream = StringIO()

# --- Logging Setup ---
# Remove existing handlers if re-running cells in interactive environments
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure logging (log to console, file, and potentially stream)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
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
    """Safely parse a string representation of a list."""
    if isinstance(s, list): return s
    if pd.isna(s) or not isinstance(s, str) or not s.startswith('['): return []
    try: return ast.literal_eval(s)
    except (ValueError, SyntaxError, TypeError, MemoryError) as e: logger.warning(f"Could not parse list string: '{str(s)[:100]}...' Error: {e}"); return []
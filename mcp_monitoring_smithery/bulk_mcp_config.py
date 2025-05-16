# %%
# =============================================================================
# Configuration for Bulk MCP Server Data Download
# =============================================================================
import logging
import os
from io import StringIO

# --- API and File Configuration ---
# IMPORTANT: Set SMITHERY_API_TOKEN environment variable or replace "your-smithery-api-token-here"
with open(os.path.expanduser("~/.cache/smithery-api/token")) as f:
    SMITHERY_API_TOKEN = f.read().strip() # IMPORTANT: Set this environment variable in the setup file as aws secret
MCP_API_BASE_URL = "https://registry.smithery.ai"
MCP_OUTPUT_CSV_FILE = "mcp_servers_database.csv"
MCP_REQUEST_TIMEOUT = 30  # Seconds for API request timeout
MCP_PAGE_SIZE_BULK = 10000 # Number of items per page for Smithery API
MCP_MAX_WORKERS_BULK = 20 # Max concurrent workers for fetching server details

# Output file for storing the list of all server summaries (from /servers endpoint)
ALL_SERVERS_SUMMARIES_JSON = "all_mcp_server_summaries.json"
# Output file for storing the full details of ALL servers (from /servers/{qualifiedName})
ALL_SERVERS_DETAILS_COMPLETE_JSON = "all_mcp_server_details_complete.json"

API_DELAY_BULK = 0.05     # Small delay (in seconds) between paginated calls to the /servers list endpoint

# --- Logging Setup for Bulk Download ---
LOG_FILE_BULK = "bulk_mcp_download.log"
log_stream_bulk = StringIO()

# Remove existing handlers if re-running in an interactive environment
for handler in logging.root.handlers[:]:
    if isinstance(handler, logging.FileHandler) and handler.baseFilename.endswith(LOG_FILE_BULK):
        logging.root.removeHandler(handler)
    if isinstance(handler, logging.StreamHandler) and handler.stream == log_stream_bulk:
        logging.root.removeHandler(handler)
    # Avoid removing general console StreamHandler if shared across modules
    # For simplicity here, we'll re-add. In a larger app, manage logger instances.



# --- Keyword Definitions for MCP Monitoring ---

# Finance Sector Keywords
FINANCE_SECTOR_KEYWORDS_CONFIG = {
    "Payments": ["wallet", "pay"]
}
# FINANCE_SECTOR_KEYWORDS_CONFIG = {
#     "Banks": ["banking", "financial institution", "depository", "lending", "credit union", "account management", "monetary policy"],
#     "Financial Trading and Markets": ["trading", "stock market", "securities", "brokerage", "exchange", "derivatives", "forex", "order execution", "market making", "clearing house"],
#     "Insurance": ["insurance", "underwriting", "actuarial", "claims processing", "reinsurance", "policy management", "insurtech"],
#     "Payments": ["payment processing", "money transfer", "digital wallet", "fintech payments", "transaction services", "settlement", "SWIFT", "ACH", "real-time payments"]
# }

# Threat Model Keywords (Inspired by NAICS where possible, focused on AI risk)
THREAT_MODEL_KEYWORDS_CONFIG = {
     "Oversight Correlation Risk": ["credit scoring"]}
# THREAT_MODEL_KEYWORDS_CONFIG = {
#     "Oversight Correlation Risk": ["regulatory compliance AI", "risk assessment AI", "credit scoring AI", "financial monitoring AI", "regtech llm"],
#     "Report Generation Risk": ["financial reporting AI", "audit automation AI", "risk analysis generation AI", "llm for audit"],
#     "Complex Product Design Risk": ["financial engineering AI", "derivatives modeling AI", "structured finance AI", "algorithmic product design AI", "synthetic assets AI"],
#     "Index Manipulation Risk": ["benchmark trading AI", "SOFR AI", "LIBOR AI", "market surveillance AI", "algorithmic trading monitoring"],
#     "Trading Algorithm Bug Risk": ["automated trading systems AI", "high-frequency trading AI", "AI code generation finance", "trading bot development AI", "algorithmic trading bugs"],
#     "Third-Party AI Failure Risk": ["foundation model finance", "shared AI infrastructure risk", "AI vendor risk management", "third party AI dependency"],
#     "AI-Enhanced Cyberattack Risk": ["cybersecurity finance AI", "AI fraud detection", "market abuse AI", "trade surveillance AI", "AI hacking tools finance"],
#     "Service Disruption Risk": ["core banking systems AI", "payment gateway AI", "operational resilience AI", "AI cyber attack payments", "AI denial of service finance"],
#     "Misinformation Bank Run Risk": ["market sentiment AI", "financial news analysis AI", "deepfake detection finance", "social media monitoring finance", "AI rumor detection"],
#     "Identity Fraud Risk": ["KYC AI", "AML AI", "biometric authentication finance", "synthetic identity detection AI", "AI identity theft finance"]
# }

# Finance Affordance Keywords (for analyzing MCP server tools)
FINANCE_AFFORDANCE_KEYWORDS_CONFIG = {
    "execution": ["trade"]}
# FINANCE_AFFORDANCE_KEYWORDS_CONFIG = {
#     "execution": ["trade", "order", "execute", "wallet", "transaction", "send money", "payment", "settlement", "buy", "sell", "transfer", "manage account", "disburse", "submit", "invest", "withdraw", "deposit", "approve", "authorize", "instruct"],
#     "information_gathering": ["news", "screen", "market data", "research", "balance", "portfolio", "analytics", "financial statement", "get price", "fetch report", "query transaction", "monitor", "lookup", "retrieve", "verify", "check status", "history", "quote"],
#     "agent_interaction": ["chat", "negotiate", "message", "collaborate", "share data", "coordinate", "discuss", "alert", "notify", "communicate", "request information", "propose"]
# }



logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
                    handlers=[
                        logging.FileHandler(LOG_FILE_BULK, mode='w'),
                        logging.StreamHandler(), # Console output
                        logging.StreamHandler(log_stream_bulk)
                    ])

bulk_logger = logging.getLogger(__name__)
bulk_logger.info("Bulk MCP Download Configuration and Logging Setup Complete.")

if SMITHERY_API_TOKEN == "your-smithery-api-token-here":
    bulk_logger.warning("SMITHERY_API_TOKEN is using the placeholder value. Bulk download will likely fail authentication.")


# %%
# =============================================================================
# Streamlit Dashboard for MCP Server Monitoring
# =============================================================================
import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import plotly.express as px
import logging # Use standard Python logging

# --- Configuration & Constants ---
PAGE_TITLE = "MCP Server Financial Stability Monitor"
MAX_SERVERS_TO_DISPLAY_IN_TABLE = 200 # For performance in st.dataframe

# --- Setup Logger ---
# Configure the default logger for the application
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.info("Dashboard logger initialized.")

# --- PAGE CONFIG MUST BE THE FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title=PAGE_TITLE, layout="wide")

# --- Global placeholder for the analysis function ---
# This will be replaced by the imported function or the fallback
run_full_mcp_analysis = None 

# --- Local Imports for Analysis and Config ---
try:
    # Attempt to import the specific analysis function and config variable
    from local_mcp_analysis import run_full_mcp_analysis as imported_run_full_mcp_analysis
    from bulk_mcp_config import ALL_SERVERS_DETAILS_COMPLETE_JSON as BULK_MCP_DETAILS_JSON_FILE
    
    run_full_mcp_analysis = imported_run_full_mcp_analysis # Assign successfully imported function
    logger.info("Successfully imported 'run_full_mcp_analysis' and configurations from 'mcp_monitoring_smithery'.")

except ImportError as e:
    st.error(
        f"CRITICAL IMPORT ERROR: Could not import the main analysis module 'mcp_monitoring_smithery.local_mcp_analysis' or its configurations. Error: {e}. "
        "The dashboard will operate with placeholder data, meaning most charts and metrics will show '0', 'N/A', or 'No Data'. "
        "Please ensure the 'mcp_monitoring_smithery' module is correctly installed, accessible in your PYTHONPATH, and has no internal errors."
    )
    logger.error(f"Failed to import from 'mcp_monitoring_smithery': {e}. Using fallback mechanisms for analysis and data path.")
    
    BULK_MCP_DETAILS_JSON_FILE = "all_mcp_server_details_complete.json" 
    logger.info(f"Dashboard: Using fallback data file path: {BULK_MCP_DETAILS_JSON_FILE}")

    # Define a comprehensive placeholder run_full_mcp_analysis if import fails
    def fallback_run_full_mcp_analysis(df_input):
        logger.error("Dashboard: Using FALLBACK 'run_full_mcp_analysis' because 'mcp_monitoring_smithery.local_mcp_analysis' could not be imported. Analysis results will be placeholders.")
        if not isinstance(df_input, pd.DataFrame):
            logger.error("Fallback analysis: Input is not a DataFrame. Returning empty.")
            # Return an empty DataFrame with all expected columns
            return pd.DataFrame(columns=[
                'server_data', 'qualifiedName', 'displayName', 'description', 'useCount', 
                'createdAt', 'toolCount', 
                'matched_finance_sectors', 'matched_finance_sectors_scores', 'matched_finance_sectors_matched_keywords',
                'matched_threat_models', 'matched_threat_models_scores', 'matched_threat_models_matched_keywords',
                'has_finance_execution', 'has_finance_information_gathering', 'has_finance_agent_interaction',
                'finance_execution_tools', 'finance_information_gathering_tools', 'finance_agent_interaction_tools',
                'finance_execution_matched_keywords', 'finance_information_gathering_matched_keywords', 'finance_agent_interaction_matched_keywords',
                'error'
            ])

        if 'server_data' not in df_input.columns and not df_input.empty:
             logger.warning("Fallback analysis: 'server_data' column missing. Creating dummy 'server_data' to proceed.")
             df_input['server_data'] = df_input.apply(lambda row: {}, axis=1)
        elif df_input.empty:
            logger.warning("Fallback analysis: Input DataFrame is empty. Returning empty DataFrame with expected schema.")
            return pd.DataFrame(columns=[
                'server_data', 'qualifiedName', 'displayName', 'description', 'useCount', 
                'createdAt', 'toolCount', 
                'matched_finance_sectors', 'matched_finance_sectors_scores', 'matched_finance_sectors_matched_keywords',
                'matched_threat_models', 'matched_threat_models_scores', 'matched_threat_models_matched_keywords',
                'has_finance_execution', 'has_finance_information_gathering', 'has_finance_agent_interaction',
                'finance_execution_tools', 'finance_information_gathering_tools', 'finance_agent_interaction_tools',
                'finance_execution_matched_keywords', 'finance_information_gathering_matched_keywords', 'finance_agent_interaction_matched_keywords',
                'error'
            ])

        df_output = df_input.copy()
        df_output['error'] = "Analysis function (run_full_mcp_analysis) was not loaded due to import error from 'mcp_monitoring_smithery'. Using placeholder data."
        
        default_text = 'N/A (placeholder)'
        default_list_col = [[] for _ in range(len(df_output))]
        default_dict_col = [{} for _ in range(len(df_output))]

        df_output['qualifiedName'] = df_output['server_data'].apply(lambda x: x.get('qualifiedName', default_text) if isinstance(x, dict) else default_text)
        df_output['displayName'] = df_output['server_data'].apply(lambda x: x.get('displayName', default_text) if isinstance(x, dict) else default_text)
        df_output['description'] = df_output['server_data'].apply(lambda x: x.get('description', "") if isinstance(x, dict) else "")
        df_output['useCount'] = df_output['server_data'].apply(lambda x: x.get('useCount', 0) if isinstance(x, dict) else 0)
        df_output['createdAt'] = pd.to_datetime(df_output['server_data'].apply(lambda x: x.get('createdAt') if isinstance(x, dict) else None), errors='coerce')
        df_output['toolCount'] = df_output['server_data'].apply(lambda x: len(x.get('tools', [])) if isinstance(x, dict) and isinstance(x.get('tools'), list) else 0)
        
        df_output['matched_finance_sectors'] = default_list_col
        df_output['matched_finance_sectors_scores'] = default_dict_col
        df_output['matched_finance_sectors_matched_keywords'] = default_dict_col # CHANGED TO DICT FOR FALLBACK
        df_output['matched_threat_models'] = default_list_col
        df_output['matched_threat_models_scores'] = default_dict_col
        df_output['matched_threat_models_matched_keywords'] = default_dict_col # CHANGED TO DICT FOR FALLBACK
        df_output['has_finance_execution'] = False
        df_output['has_finance_information_gathering'] = False
        df_output['has_finance_agent_interaction'] = False
        df_output['finance_execution_tools'] = default_list_col
        df_output['finance_information_gathering_tools'] = default_list_col
        df_output['finance_agent_interaction_tools'] = default_list_col
        df_output['finance_execution_matched_keywords'] = default_list_col
        df_output['finance_information_gathering_matched_keywords'] = default_list_col
        df_output['finance_agent_interaction_matched_keywords'] = default_list_col
        return df_output
    
    run_full_mcp_analysis = fallback_run_full_mcp_analysis # Assign fallback function

# --- Helper Functions ---
def get_file_details(filepath):
    """Gets file modification time and size."""
    try:
        timestamp = os.path.getmtime(filepath)
        dt_object = datetime.fromtimestamp(timestamp)
        size = os.path.getsize(filepath)
        return dt_object.strftime("%Y-%m-%d %H:%M:%S"), f"{size / (1024*1024):.2f} MB"
    except Exception as e:
        logger.error(f"Error getting file details for {filepath}: {e}")
        return "N/A", "N/A"

@st.cache_data # Cache the data loading and initial analysis
def load_and_analyze_data(json_file_path):
    """
    Loads MCP server data from the bulk JSON file and runs initial analysis.
    """
    logger.info(f"Attempting to load data from: {json_file_path}")
    if not os.path.exists(json_file_path):
        logger.error(f"Data file not found: {json_file_path}")
        st.error(f"CRITICAL: Data file '{os.path.basename(json_file_path)}' not found at the expected path: '{json_file_path}'. Please ensure `run_bulk_mcp_download.py` has been executed successfully and the path is correct.")
        return pd.DataFrame() 

    try:
        with open(json_file_path, 'r') as f:
            raw_data_list = json.load(f) 
        
        if not isinstance(raw_data_list, list):
            logger.error(f"Loaded data from {json_file_path} is not a list as expected. Type: {type(raw_data_list)}")
            st.error(f"Data format error: Expected a list of servers from '{os.path.basename(json_file_path)}'. Check the file content.")
            return pd.DataFrame()

        if not raw_data_list:
            logger.warning(f"No server data found in {json_file_path}. The file might be empty.")
            st.warning(f"No server data found in '{os.path.basename(json_file_path)}'.")
            return pd.DataFrame()

        df_raw = pd.DataFrame({'server_data': raw_data_list})
        logger.info(f"Successfully loaded {len(df_raw)} servers from {json_file_path}.")
        
        if run_full_mcp_analysis.__name__ == 'fallback_run_full_mcp_analysis':
            logger.warning("load_and_analyze_data: Using the FALLBACK analysis function.")
        else:
            logger.info("load_and_analyze_data: Using the IMPORTED analysis function from 'mcp_monitoring_smithery'.")
            
        df_analyzed = run_full_mcp_analysis(df_raw) 
        
        expected_cols = ['qualifiedName', 'matched_finance_sectors', 'matched_threat_models', 
                         'matched_finance_sectors_matched_keywords', 'matched_threat_models_matched_keywords', # ADDED
                         'has_finance_execution', 'finance_execution_matched_keywords'] # ADDED
        if not df_analyzed.empty and not all(col in df_analyzed.columns for col in expected_cols):
             logger.error(f"Analysis (or fallback) did not produce all expected columns. Available: {df_analyzed.columns.tolist()}")
             st.warning("Analysis pipeline might have encountered issues or is using fallbacks. Some dashboard features may not work correctly or show limited data.")
        elif df_analyzed.empty and raw_data_list: 
            st.error("Data was loaded, but the analysis pipeline returned an empty result. Check logs for errors in `local_mcp_analysis.py` or data processing.")
        
        list_cols_to_check = [
            'matched_finance_sectors', 'matched_threat_models', 
            'finance_execution_tools', 'finance_information_gathering_tools', 'finance_agent_interaction_tools',
            'finance_execution_matched_keywords', 'finance_information_gathering_matched_keywords', 'finance_agent_interaction_matched_keywords' # ADDED
        ]
        dict_cols_to_check = [
            'matched_finance_sectors_scores', 'matched_threat_models_scores',
            'matched_finance_sectors_matched_keywords', 'matched_threat_models_matched_keywords' # ADDED (these are dicts mapping category to list of keywords)
        ]


        for col in list_cols_to_check:
            if col in df_analyzed.columns:
                 # Ensure _matched_keywords for affordances are lists (as they come from analysis)
                if col.endswith("_matched_keywords") and ("finance_execution" in col or "finance_information_gathering" in col or "finance_agent_interaction" in col):
                    df_analyzed[col] = df_analyzed[col].apply(lambda x: x if isinstance(x, list) else [])
                elif not col.endswith("_matched_keywords"): # Original list columns
                    df_analyzed[col] = df_analyzed[col].apply(lambda x: x if isinstance(x, list) else [])
            else: 
                df_analyzed[col] = [[] for _ in range(len(df_analyzed))]
        
        for col in dict_cols_to_check:
            if col in df_analyzed.columns:
                df_analyzed[col] = df_analyzed[col].apply(lambda x: x if isinstance(x, dict) else {})
            else: 
                df_analyzed[col] = [{} for _ in range(len(df_analyzed))]

        return df_analyzed

    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding error from {json_file_path}: {e}")
        st.error(f"Error reading or parsing JSON data from '{os.path.basename(json_file_path)}'. Ensure it's a valid JSON file containing a list of server objects. Error: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"An unexpected error occurred during data loading or analysis: {e}", exc_info=True)
        st.error(f"An unexpected error occurred: {e}. Check the application logs.")
        return pd.DataFrame()

def display_server_details(selected_server_row):
    """Displays detailed information for a selected server."""
    if selected_server_row is None or selected_server_row.empty:
        st.info("Select a server from the table or dropdown to see details.")
        return

    server = selected_server_row.iloc[0] 

    st.subheader(f"Details for: {server.get('displayName', 'N/A')} (`{server.get('qualifiedName', 'N/A')}`)")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Use Count", server.get('useCount', 0))
        st.metric("Tool Count", server.get('toolCount', 0))
    with col2:
        created_at = server.get('createdAt')
        if pd.notna(created_at) and isinstance(created_at, (datetime, pd.Timestamp)): 
            st.metric("Created At", created_at.strftime('%Y-%m-%d %H:%M:%S'))
        elif pd.notna(created_at): 
            try:
                st.metric("Created At", pd.to_datetime(created_at).strftime('%Y-%m-%d %H:%M:%S'))
            except:
                st.metric("Created At", "N/A (invalid date format)")
        else:
            st.metric("Created At", "N/A")

    st.markdown("**Description:**")
    st.markdown(f"> {server.get('description', 'No description available.')}")

    st.markdown("**Matched Finance Sectors:**")
    sectors = server.get('matched_finance_sectors', [])
    scores_sectors = server.get('matched_finance_sectors_scores', {})
    keywords_sectors = server.get('matched_finance_sectors_matched_keywords', {}) # ADDED
    if sectors and isinstance(sectors, list): 
        for sector in sectors:
            score = scores_sectors.get(sector, 0) if isinstance(scores_sectors, dict) else 0
            kws = ", ".join(keywords_sectors.get(sector, [])) if isinstance(keywords_sectors, dict) and isinstance(keywords_sectors.get(sector), list) else "N/A" # ADDED
            st.markdown(f"- {sector} (Score: {score}, Matched Keywords: {kws})") # MODIFIED
    else:
        st.markdown("_No specific finance sectors matched or data is in unexpected format._")

    st.markdown("**Matched Threat Models:**")
    threats = server.get('matched_threat_models', [])
    scores_threats = server.get('matched_threat_models_scores', {})
    keywords_threats = server.get('matched_threat_models_matched_keywords', {}) # ADDED
    if threats and isinstance(threats, list): 
        for threat in threats:
            score = scores_threats.get(threat, 0) if isinstance(scores_threats, dict) else 0
            kws = ", ".join(keywords_threats.get(threat, [])) if isinstance(keywords_threats, dict) and isinstance(keywords_threats.get(threat), list) else "N/A" # ADDED
            st.markdown(f"- {threat} (Score: {score}, Matched Keywords: {kws})") # MODIFIED
    else:
        st.markdown("_No specific threat models matched or data is in unexpected format._")

    st.markdown("**Financial Affordances:**")
    affordances_found = []
    if server.get('has_finance_execution', False): 
        tools = server.get('finance_execution_tools', []) 
        kws_exec = ", ".join(server.get('finance_execution_matched_keywords', [])) if isinstance(server.get('finance_execution_matched_keywords'), list) else "N/A" # ADDED
        affordances_found.append(f"**Execution** (Tools: {', '.join(tools) if isinstance(tools, list) and tools else 'N/A'}, Matched Keywords: {kws_exec})") # MODIFIED
    if server.get('has_finance_information_gathering', False):
        tools = server.get('finance_information_gathering_tools', [])
        kws_info = ", ".join(server.get('finance_information_gathering_matched_keywords', [])) if isinstance(server.get('finance_information_gathering_matched_keywords'), list) else "N/A" # ADDED
        affordances_found.append(f"**Information Gathering** (Tools: {', '.join(tools) if isinstance(tools, list) and tools else 'N/A'}, Matched Keywords: {kws_info})") # MODIFIED
    if server.get('has_finance_agent_interaction', False):
        tools = server.get('finance_agent_interaction_tools', [])
        kws_agent = ", ".join(server.get('finance_agent_interaction_matched_keywords', [])) if isinstance(server.get('finance_agent_interaction_matched_keywords'), list) else "N/A" # ADDED
        affordances_found.append(f"**Agent Interaction** (Tools: {', '.join(tools) if isinstance(tools, list) and tools else 'N/A'}, Matched Keywords: {kws_agent})") # MODIFIED
    
    if affordances_found:
        for aff in affordances_found:
            st.markdown(f"- {aff}")
    else:
        st.markdown("_No specific financial affordances identified (or analysis module not loaded)._")

    with st.expander("View Raw Server Data (JSON)"):
        server_data_dict = server.get('server_data', {}) 
        if isinstance(server_data_dict, dict) and server_data_dict:
            st.json(server_data_dict)
        elif isinstance(server, pd.Series) and 'server_data' not in server and isinstance(server.to_dict(), dict) :
             st.json(server.to_dict())
        else:
            st.markdown("_Raw server data not available in expected format or is empty._")

# --- Streamlit App Layout ---
st.title(f"🏦 {PAGE_TITLE}")

# --- Load Data ---
if 'BULK_MCP_DETAILS_JSON_FILE' not in globals() or BULK_MCP_DETAILS_JSON_FILE is None:
    st.error("CRITICAL: Data source file path (BULK_MCP_DETAILS_JSON_FILE) is not defined. This should have been set during initial imports. Dashboard cannot load data.")
    st.stop()
    
data_file_mod_time, data_file_size = get_file_details(BULK_MCP_DETAILS_JSON_FILE)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Data Source:**")
st.sidebar.markdown(f"`{os.path.basename(BULK_MCP_DETAILS_JSON_FILE)}`")
st.sidebar.markdown(f"**Last Modified:** {data_file_mod_time}")
st.sidebar.markdown(f"**Size:** {data_file_size}")
st.sidebar.markdown("---")

df_analyzed = load_and_analyze_data(BULK_MCP_DETAILS_JSON_FILE)

if df_analyzed.empty:
    st.error("No data available to display. This could be due to an empty data file, issues loading the data, or problems during the analysis phase (e.g., import errors for 'mcp_monitoring_smithery'). Please check the data file and application logs.")
    st.stop() 

# --- Dashboard Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Overview & Stats", "🔍 Explore All MCP Servers", "🛠️ Financial Affordances"])

with tab1: 
    st.header("MCP Ecosystem Overview")
    # ... (no changes in this part of tab1, unless you want to show keyword stats here too) ...
    total_servers = len(df_analyzed)
    st.metric("Total MCP Servers Analyzed", total_servers if total_servers > 0 else "0 (or data loading issue)")

    if 'toolCount' in df_analyzed.columns:
        servers_with_tools = df_analyzed[df_analyzed['toolCount'] > 0].shape[0]
        if total_servers > 0 :
            if (servers_with_tools / total_servers) < 0.1:
                st.warning(f"⚠️ Low percentage of servers with tool details found ({servers_with_tools}/{total_servers}). Affordance analysis might be incomplete (or analysis module not loaded).")
            elif servers_with_tools == 0:
                st.warning(f"⚠️ No servers with tool details found ({servers_with_tools}/{total_servers}). Affordance analysis will be missing (or analysis module not loaded).")
        elif total_servers == 0 and 'error' not in df_analyzed.columns: 
             st.info("No server data loaded to assess tool details.")
    else:
        st.warning("Column 'toolCount' not found in analyzed data. Cannot assess tool details coverage.")

    st.subheader("Distribution by Matched Finance Sectors")
    if 'matched_finance_sectors' in df_analyzed.columns:
        finance_sectors_series = df_analyzed['matched_finance_sectors'].apply(lambda x: x if isinstance(x, list) else [])
        finance_sectors_flat = finance_sectors_series.explode()
        finance_sectors_flat = finance_sectors_flat[finance_sectors_flat.apply(lambda x: isinstance(x, str) and x != '')] 
        if not finance_sectors_flat.empty:
            sector_counts = finance_sectors_flat.value_counts().reset_index()
            sector_counts.columns = ['Finance Sector', 'Number of Servers']
            fig_sectors = px.bar(sector_counts, x='Finance Sector', y='Number of Servers', title="Servers per Matched Finance Sector")
            st.plotly_chart(fig_sectors, use_container_width=True)
        else:
            st.info("No servers matched any finance sectors. This could be due to data, keyword configurations, or the analysis module not being loaded correctly.")
    else:
        st.info("Finance sector data ('matched_finance_sectors') not available for plotting.")

    st.subheader("Distribution by Matched Threat Models")
    if 'matched_threat_models' in df_analyzed.columns:
        threat_models_series = df_analyzed['matched_threat_models'].apply(lambda x: x if isinstance(x, list) else [])
        threat_models_flat = threat_models_series.explode()
        threat_models_flat = threat_models_flat[threat_models_flat.apply(lambda x: isinstance(x, str) and x != '')]
        if not threat_models_flat.empty:
            threat_counts = threat_models_flat.value_counts().reset_index()
            threat_counts.columns = ['Threat Model', 'Number of Servers']
            fig_threats = px.bar(threat_counts, x='Threat Model', y='Number of Servers', title="Servers per Matched Threat Model")
            st.plotly_chart(fig_threats, use_container_width=True)
        else:
            st.info("No servers matched any threat models. This could be due to data, keyword configurations, or the analysis module not being loaded correctly.")
    else:
        st.info("Threat model data ('matched_threat_models') not available for plotting.")

    st.subheader("Distribution of Server Use Counts")
    if 'useCount' in df_analyzed.columns and df_analyzed['useCount'].notna().any() and df_analyzed['useCount'].sum() > 0 :
        fig_use_count = px.histogram(df_analyzed[df_analyzed['useCount'] > 0], x="useCount", nbins=50, title="Server Use Count Distribution (Excluding Zero Counts)")
        st.plotly_chart(fig_use_count, use_container_width=True)
    else:
        st.info("No use count data available, all use counts are zero, or 'useCount' column is missing.")
    
    st.subheader("Servers Over Time (by Creation Date)")
    if 'createdAt' in df_analyzed.columns and df_analyzed['createdAt'].notna().any():
        df_time = df_analyzed.copy()
        df_time['createdAt'] = pd.to_datetime(df_time['createdAt'], errors='coerce')
        df_time = df_time.dropna(subset=['createdAt'])
        if not df_time.empty and 'qualifiedName' in df_time.columns:
            servers_over_time = df_time.set_index('createdAt').resample('ME')['qualifiedName'].count().reset_index()
            servers_over_time.columns = ['Month', 'Number of Servers Created']
            fig_time = px.line(servers_over_time, x='Month', y='Number of Servers Created', title="MCP Servers Created Over Time (Monthly)")
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("Creation date or qualifiedName data is not sufficiently available/valid for a time series plot.")
    else:
        st.info("Creation date data ('createdAt' column) is not available for a time series plot.")


with tab2: 
    st.header("Explore and Filter MCP Servers")
    # ... (sidebar filters remain largely the same) ...
    st.sidebar.title("🔍 Server Filters")

    df_filtered = df_analyzed.copy() 

    if 'matched_finance_sectors' in df_analyzed.columns:
        all_finance_sectors_series = df_analyzed['matched_finance_sectors'].apply(lambda x: x if isinstance(x, list) else [])
        all_finance_sectors = sorted(list(set(fs for sublist in all_finance_sectors_series for fs in sublist if isinstance(fs, str) and fs)))
        if all_finance_sectors:
            selected_finance_sectors = st.sidebar.multiselect("Filter by Finance Sector(s):", options=all_finance_sectors, key="ms_finance_sectors")
            if selected_finance_sectors: 
                df_filtered = df_filtered[df_filtered['matched_finance_sectors'].apply(lambda x: isinstance(x, list) and any(s in x for s in selected_finance_sectors))]
        else:
            st.sidebar.text("No finance sectors available for filtering (or analysis module not loaded).")
    else:
        st.sidebar.text("Finance sector data not available for filtering.")

    if 'matched_threat_models' in df_analyzed.columns:
        all_threat_models_series = df_analyzed['matched_threat_models'].apply(lambda x: x if isinstance(x, list) else [])
        all_threat_models = sorted(list(set(tm for sublist in all_threat_models_series for tm in sublist if isinstance(tm, str) and tm)))
        if all_threat_models:
            selected_threat_models = st.sidebar.multiselect("Filter by Threat Model(s):", options=all_threat_models, key="ms_threat_models")
            if selected_threat_models: 
                df_filtered = df_filtered[df_filtered['matched_threat_models'].apply(lambda x: isinstance(x, list) and any(t in x for t in selected_threat_models))]
        else:
            st.sidebar.text("No threat models available for filtering (or analysis module not loaded).")
    else:
        st.sidebar.text("Threat model data not available for filtering.")
    
    if 'useCount' in df_analyzed.columns and not df_analyzed['useCount'].empty:
        min_uc, max_uc = int(df_analyzed['useCount'].min()), int(df_analyzed['useCount'].max())
        selected_use_range = st.sidebar.slider("Filter by Use Count:", min_uc, max_uc, (min_uc, max_uc), disabled=(min_uc == max_uc), key="slider_use_count")
        if not (min_uc == selected_use_range[0] and max_uc == selected_use_range[1]) or min_uc != max_uc :
            df_filtered = df_filtered[(df_filtered['useCount'] >= selected_use_range[0]) & (df_filtered['useCount'] <= selected_use_range[1])]
    else:
        st.sidebar.text("Use count data not available for filtering.")
        
    st.sidebar.markdown("**Filter by Financial Affordances:**")
    if 'has_finance_execution' in df_filtered.columns:
        filter_exec_tab2 = st.sidebar.checkbox("Has Execution Affordance", value=False, key="cb_exec_tab2_unique")
        if filter_exec_tab2: df_filtered = df_filtered[df_filtered['has_finance_execution'] == True]
    if 'has_finance_information_gathering' in df_filtered.columns:
        filter_info_tab2 = st.sidebar.checkbox("Has Info Gathering Affordance", value=False, key="cb_info_tab2_unique")
        if filter_info_tab2: df_filtered = df_filtered[df_filtered['has_finance_information_gathering'] == True]
    if 'has_finance_agent_interaction' in df_filtered.columns:
        filter_interact_tab2 = st.sidebar.checkbox("Has Agent Interaction Affordance", value=False, key="cb_interact_tab2_unique")
        if filter_interact_tab2: df_filtered = df_filtered[df_filtered['has_finance_agent_interaction'] == True]
    
    if not any(col_name in df_filtered.columns for col_name in ['has_finance_execution', 'has_finance_information_gathering', 'has_finance_agent_interaction']):
        st.sidebar.text("Financial affordance data not available for filtering (or analysis module not loaded).")


    st.info(f"Showing {len(df_filtered)} of {len(df_analyzed)} servers based on current filters.")

    if not df_filtered.empty:
        cols_display = [
            'displayName', 'qualifiedName', 'useCount', 'toolCount', 'createdAt', 
            'matched_finance_sectors', 'matched_finance_sectors_matched_keywords', # ADDED
            'matched_threat_models', 'matched_threat_models_matched_keywords'      # ADDED
        ]
        # Filter for columns that actually exist to prevent errors
        cols_exist = [col for col in cols_display if col in df_filtered.columns]
        
        # Convert dicts in keyword columns to string for display in main table
        df_display_tab2 = df_filtered[cols_exist].copy()
        if 'matched_finance_sectors_matched_keywords' in df_display_tab2:
            df_display_tab2['matched_finance_sectors_matched_keywords'] = df_display_tab2['matched_finance_sectors_matched_keywords'].apply(
                lambda d: "; ".join([f"{k}: {', '.join(v)}" for k, v in d.items()]) if isinstance(d, dict) else ""
            )
        if 'matched_threat_models_matched_keywords' in df_display_tab2:
            df_display_tab2['matched_threat_models_matched_keywords'] = df_display_tab2['matched_threat_models_matched_keywords'].apply(
                lambda d: "; ".join([f"{k}: {', '.join(v)}" for k, v in d.items()]) if isinstance(d, dict) else ""
            )

        st.dataframe(df_display_tab2.head(MAX_SERVERS_TO_DISPLAY_IN_TABLE), use_container_width=True, height=600)

        if len(df_filtered) > MAX_SERVERS_TO_DISPLAY_IN_TABLE:
            st.caption(f"Displaying top {MAX_SERVERS_TO_DISPLAY_IN_TABLE} results. Refine filters for more specific results.")

        st.subheader("View Server Details")
        if 'qualifiedName' in df_filtered.columns:
            # Ensure options are strings and handle potential NaN or None values gracefully
            valid_qnames = df_filtered['qualifiedName'].dropna().astype(str).unique().tolist()
            options = [""] + sorted(valid_qnames)
            sel_qname = st.selectbox("Select Server (by Qualified Name):", options, index=0, key="server_sel_tab2_unique")
            if sel_qname: 
                server_detail_df = df_filtered[df_filtered['qualifiedName'] == sel_qname]
                if not server_detail_df.empty:
                    display_server_details(server_detail_df)
            else:
                st.info("Select a server from the dropdown to view details.") 
        else:
            st.warning("QualifiedName column missing, cannot select server for details.")
    else:
        st.info("No servers match the current filter criteria.")

with tab3: 
    st.header("Analysis of Financial Affordances in MCP Servers")
    aff_cols = ['has_finance_execution', 'has_finance_information_gathering', 'has_finance_agent_interaction']
    
    if not all(col in df_analyzed.columns for col in aff_cols):
        st.error("One or more financial affordance columns are missing from the analyzed data. This tab cannot be fully displayed. This may be due to the analysis module ('mcp_monitoring_smithery') not being loaded correctly.")
    else:
        # ... (bar chart for affordance counts remains the same) ...
        aff_counts_data = []
        display_names = ['Execution', 'Information Gathering', 'Agent Interaction']
        for col, d_name in zip(aff_cols, display_names):
            if col in df_analyzed.columns: 
                aff_counts_data.append({'Affordance Type': d_name, 'Number of Servers': df_analyzed[col].sum()})
        
        if aff_counts_data:
            aff_counts_df = pd.DataFrame(aff_counts_data)
            if not aff_counts_df.empty and aff_counts_df['Number of Servers'].sum() > 0 :
                fig_aff = px.bar(aff_counts_df, x='Affordance Type', y='Number of Servers', title="Servers by Identified Financial Affordance")
                st.plotly_chart(fig_aff, use_container_width=True)
            else:
                st.info("No servers identified with any financial affordances (or analysis module not loaded).")
        else:
            st.info("No financial affordance data available to plot (possibly due to missing columns or all counts being zero).")


        st.subheader("Servers with Specific Affordances")
        for aff_col, aff_display_name in zip(aff_cols, display_names):
            if aff_col not in df_analyzed.columns:
                st.markdown(f"_{aff_display_name} data column ('{aff_col}') not available._")
                continue

            sum_servers = df_analyzed[aff_col].sum()
            with st.expander(f"Servers with {aff_display_name} Affordance ({sum_servers} servers)"):
                df_subset = df_analyzed[df_analyzed[aff_col] == True]
                if not df_subset.empty:
                    tool_col_key = f"finance_{aff_display_name.lower().replace(' ', '_')}_tools"
                    keyword_col_key = f"finance_{aff_display_name.lower().replace(' ', '_')}_matched_keywords" # ADDED
                    
                    cols_to_show = ['displayName', 'qualifiedName', 'useCount']
                    if tool_col_key in df_subset.columns:
                        cols_to_show.append(tool_col_key)
                    if keyword_col_key in df_subset.columns: # ADDED
                        cols_to_show.append(keyword_col_key) # ADDED
                    
                    existing_cols_to_show = [c for c in cols_to_show if c in df_subset.columns]
                    st.dataframe(df_subset[existing_cols_to_show].head(MAX_SERVERS_TO_DISPLAY_IN_TABLE), use_container_width=True)
                    if len(df_subset) > MAX_SERVERS_TO_DISPLAY_IN_TABLE:
                        st.caption(f"Displaying top {MAX_SERVERS_TO_DISPLAY_IN_TABLE} results.")
                else:
                    st.markdown(f"_No servers identified with {aff_display_name} affordance (or analysis module not loaded)._")
# %%
# To run: streamlit run local_mcp_dashboard.py
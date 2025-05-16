# %%
# =============================================================================
# Data Loading, Preprocessing, and Analysis Functions
# =============================================================================
import pandas as pd
import logging
import os
from tqdm import tqdm
from collections import Counter # Used indirectly via get_top_tasks_per_sector
import json # For parsing JSON strings from CSV

# Import logger and utility functions from config_utils
from hf_models_monitoring_test.config_utils import logger, parse_list_string, parse_json_string

# --- Keyword Loading Function (Phase 1 Helper for NAICS) ---
def load_naics_keywords(csv_path):
    """
    Loads sector codes, names, keywords, and CNI info from the NAICS CSV.
    Returns:
        sector_data_map (dict): Maps unique sector ID (Code - Name) to details {'code', 'name', 'keywords', 'cni_percentage'}.
        sector_cni_map (dict): Maps unique sector ID to CNI percentage.
    """
    logger.info(f"Loading NAICS data from {csv_path}...")
    if not os.path.exists(csv_path):
        logger.error(f"ERROR: NAICS keyword file not found: {csv_path}")
        return None, None

    try:
        df_naics = pd.read_csv(csv_path)
        required_cols = ['sector_code', 'sector_name', 'keywords']
        if not all(col in df_naics.columns for col in required_cols):
            logger.error(f"NAICS CSV must contain {required_cols} columns.")
            return None, None

        sector_data_map = {}
        processed_codes = set()
        duplicates_found = False

        for _, row in df_naics.iterrows():
            sector_code = row['sector_code']
            sector_name = str(row['sector_name']).strip()
            keywords_str = row['keywords']

            if pd.isna(sector_code) or pd.isna(sector_name) or sector_name == "" or pd.isna(keywords_str):
                logger.warning(f"Skipping row due to missing code, name, or keywords: Code={sector_code}, Name='{sector_name}'")
                continue

            sector_code_str = str(sector_code).strip()

            if sector_code_str in processed_codes:
                logger.warning(f"Duplicate sector code '{sector_code_str}' found. Using first occurrence.")
                duplicates_found = True
                continue
            processed_codes.add(sector_code_str)

            sector_id = f"{sector_code_str} - {sector_name}"
            keywords_list = [k.strip() for k in str(keywords_str).split(',') if k.strip()]
            if not keywords_list:
                 logger.warning(f"Skipping sector '{sector_id}' due to no valid keywords parsed.")
                 continue

            cni_percentage = pd.to_numeric(row.get('cni_percentage'), errors='coerce')
            cni_percentage = 0.0 if pd.isna(cni_percentage) else cni_percentage

            sector_data_map[sector_id] = {
                'code': sector_code_str,
                'name': sector_name,
                'keywords': keywords_list,
                'cni_percentage': cni_percentage
            }

        logger.info(f"Loaded data for {len(sector_data_map)} unique sectors from NAICS CSV.")
        if duplicates_found:
             logger.warning("Duplicate NAICS sector codes were encountered and skipped (first occurrence kept).")
        sector_cni_map = {sid: data['cni_percentage'] for sid, data in sector_data_map.items()}
        return sector_data_map, sector_cni_map

    except Exception as e:
        logger.error(f"Error reading or parsing NAICS CSV '{csv_path}': {e}", exc_info=True)
        return None, None


# --- Data Loading and Preprocessing for Hugging Face Models (Phase 4 Helper) ---
def load_and_preprocess_hf_data(csv_file):
    """Loads and preprocesses the collected Hugging Face model data from CSV."""
    logger.info(f"Loading Hugging Face model data from {csv_file}...")
    if not os.path.exists(csv_file):
        logger.error(f"ERROR: Hugging Face Input file not found: {csv_file}")
        return None
    try:
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded {len(df)} Hugging Face model entries from CSV.")
        tqdm.pandas(desc="Parsing HF List Strings")
        for col in ['tags', 'matched_sectors', 'matched_keywords']:
            if col in df.columns:
                logger.info(f"Parsing list strings in HF column: {col}")
                df[col] = df[col].progress_apply(parse_list_string)
            else:
                 logger.warning(f"HF Column '{col}' not found in CSV during preprocessing.")
                 df[col] = pd.Series([[] for _ in range(len(df))])

        df['lastModified'] = pd.to_datetime(df['lastModified'], errors='coerce')
        df['downloads'] = pd.to_numeric(df['downloads'], errors='coerce').fillna(0).astype(int)
        df['likes'] = pd.to_numeric(df['likes'], errors='coerce').fillna(0).astype(int)
        if 'matched_sectors' in df.columns: # Check if column exists before applying len
             df['sector_count'] = df['matched_sectors'].apply(len)
        else:
             df['sector_count'] = 0 # Default to 0 if column is missing
        df['pipeline_tag'] = df['pipeline_tag'].fillna('unknown')
        logger.info("Hugging Face model data preprocessing complete.")
        return df.reset_index(drop=True)
    except Exception as e:
        logger.error(f"Error loading or preprocessing Hugging Face data from {csv_file}: {e}", exc_info=True)
        return None

# --- Data Loading and Preprocessing for MCP Servers ---
def load_and_preprocess_mcp_data(csv_file):
    """Loads and preprocesses the collected MCP server data from CSV."""
    logger.info(f"Loading MCP server data from {csv_file}...")
    if not os.path.exists(csv_file):
        logger.error(f"ERROR: MCP server data file not found: {csv_file}")
        return None
    try:
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded {len(df)} MCP server entries from CSV.")
        
        # Columns that are expected to be JSON strings representing lists/objects
        json_string_cols = [
            'connections_info', 'all_tools_details', 
            'finance_execution_tools_list', 'finance_info_tools_list', 
            'finance_interaction_tools_list', 'matched_keywords', 'keyword_categories'
        ]
        
        tqdm.pandas(desc="Parsing MCP JSON/List Strings")
        for col in json_string_cols:
            if col in df.columns:
                logger.debug(f"Parsing JSON/list strings in MCP column: {col}")
                # Use parse_json_string for dict-like and parse_list_string for list-like
                if 'list' in col or 'categories' in col or 'keywords' in col: # Heuristic
                    df[col] = df[col].progress_apply(parse_list_string)
                else: # Assume dict-like (e.g. connections_info, all_tools_details)
                    df[col] = df[col].progress_apply(lambda x: parse_json_string(x, default_val=[] if 'tools_details' in col else {}))
            else:
                logger.warning(f"MCP Column '{col}' not found in CSV during preprocessing.")
                df[col] = pd.Series([[] if 'list' in col or 'categories' in col or 'keywords' in col else {} for _ in range(len(df))])

        # Type conversions for other columns
        if 'createdAt_list' in df.columns:
            df['createdAt_list'] = pd.to_datetime(df['createdAt_list'], errors='coerce')
        if 'usage_tool_calls' in df.columns:
            df['usage_tool_calls'] = pd.to_numeric(df['usage_tool_calls'], errors='coerce').fillna(0).astype(int)
        if 'isDeployed' in df.columns:
            df['isDeployed'] = df['isDeployed'].astype(bool)
        if 'security_scan_passed' in df.columns:
             # Handle various ways booleans might be stored or be NaN
            df['security_scan_passed'] = df['security_scan_passed'].apply(
                lambda x: True if str(x).lower() == 'true' else (False if str(x).lower() == 'false' else None)
            ).astype('object') # Keep as object to allow None, or use pd.BooleanDtype() for pandas >= 1.0


        # Add a column for easier affordance checking in dashboard
        for affordance in ['execution', 'info', 'interaction']:
            col_name = f'has_finance_{affordance}'
            list_col = f'finance_{affordance}_tools_list'
            if list_col in df.columns:
                 df[col_name] = df[list_col].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)
            else:
                 df[col_name] = False


        logger.info("MCP server data preprocessing complete.")
        return df.reset_index(drop=True)
    except Exception as e:
        logger.error(f"Error loading or preprocessing MCP server data from {csv_file}: {e}", exc_info=True)
        return None


# --- Analysis Functions for Hugging Face Data (existing, ensure they work with preprocessed data) ---
def get_sector_summary(df, sector_data_map):
    """Calculates summary statistics per sector for HF models."""
    logger.info("Calculating HF sector summary statistics...")
    if df is None or 'matched_sectors' not in df.columns or df.empty:
        logger.error("HF DataFrame invalid or empty for sector summary.")
        return pd.DataFrame()
    if sector_data_map is None:
        logger.error("Sector data map is required for HF sector summary.")
        return pd.DataFrame()

    # Ensure 'matched_sectors' contains lists
    if not df['matched_sectors'].apply(isinstance, args=(list,)).all():
         logger.warning("Found non-list entries in HF 'matched_sectors'. Attempting to fix.")
         df['matched_sectors'] = df['matched_sectors'].apply(lambda x: parse_list_string(x) if not isinstance(x, list) else x)
    
    try:
        df_exploded = df.explode('matched_sectors')
        df_exploded = df_exploded.dropna(subset=['matched_sectors'])
    except Exception as e:
        logger.error(f"Error exploding HF sectors: {e}", exc_info=True)
        return pd.DataFrame()

    if df_exploded.empty:
        logger.warning("HF DataFrame empty after exploding sectors.")
        return pd.DataFrame()
    
    logger.debug("Aggregating base HF sector stats...")
    sector_summary_base = df_exploded.groupby('matched_sectors').agg(
        model_count=('modelId', 'nunique'),
        total_downloads=('downloads', 'sum'),
        total_likes=('likes', 'sum'),
        average_downloads=('downloads', 'mean')
    ).reset_index()

    # Find Top Model (simplified for brevity, assumes 'downloads' and 'modelId' exist)
    top_model_details = df_exploded.loc[df_exploded.groupby('matched_sectors')['downloads'].idxmax()][
        ['matched_sectors', 'modelId', 'downloads', 'matched_keywords']
    ].rename(columns={
        'modelId': 'top_model_id', 
        'downloads': 'top_model_downloads',
        'matched_keywords': 'top_model_keywords' # Keywords of the top model
    })
    # Drop duplicates in case idxmax picked multiple if downloads are same
    top_model_details = top_model_details.drop_duplicates(subset=['matched_sectors'])


    sector_summary_merged = pd.merge(sector_summary_base, top_model_details, on='matched_sectors', how='left')
    
    sector_summary_merged['sector_code'] = sector_summary_merged['matched_sectors'].map(lambda sid: sector_data_map.get(sid, {}).get('code', 'N/A'))
    sector_summary_merged['sector_name'] = sector_summary_merged['matched_sectors'].map(lambda sid: sector_data_map.get(sid, {}).get('name', 'N/A'))
    sector_summary_final = sector_summary_merged.rename(columns={'matched_sectors': 'sector_id_full'})

    numeric_cols = ['average_downloads', 'total_downloads', 'top_model_downloads']
    for col in numeric_cols:
        if col in sector_summary_final.columns:
            sector_summary_final[col] = pd.to_numeric(sector_summary_final[col], errors='coerce').fillna(0)

    if 'total_downloads' in sector_summary_final.columns:
        sector_summary_final['total_downloads_str'] = sector_summary_final['total_downloads'].apply(lambda x: f"{int(x):,}")
    if 'top_model_downloads' in sector_summary_final.columns:
        sector_summary_final['top_model_downloads_str'] = sector_summary_final['top_model_downloads'].apply(lambda x: f"{int(x):,}")
    if 'top_model_keywords' in sector_summary_final.columns:
        sector_summary_final['top_model_keywords_str'] = sector_summary_final['top_model_keywords'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) else ''
        )
    else:
        sector_summary_final['top_model_keywords_str'] = ''


    sector_summary_final = sector_summary_final.sort_values(by='model_count', ascending=False).reset_index(drop=True)
    logger.info("HF sector summary calculation complete.")
    
    final_cols_ordered = [
        'sector_code', 'sector_name', 'model_count', 'total_downloads_str', 'average_downloads',
        'top_model_id', 'top_model_downloads_str', 'top_model_keywords_str',
        'total_downloads', 'total_likes', 'top_model_downloads','top_model_keywords', # Keep original numeric/list for plots
        'sector_id_full'
    ]
    return sector_summary_final[[col for col in final_cols_ordered if col in sector_summary_final.columns]]


def get_keyword_summary(df):
    """Calculates summary statistics per keyword for HF models."""
    logger.info("Calculating HF keyword summary statistics...")
    if df is None or 'matched_keywords' not in df.columns or df.empty:
        logger.error("HF DataFrame invalid or empty for keyword summary."); return pd.DataFrame()
    if not df['matched_keywords'].apply(isinstance, args=(list,)).all():
         logger.warning("Found non-list entries in HF 'matched_keywords'. Attempting to fix.")
         df['matched_keywords'] = df['matched_keywords'].apply(lambda x: parse_list_string(x) if not isinstance(x, list) else x)

    try:
        df_kw_exploded = df.explode('matched_keywords')
        df_kw_exploded = df_kw_exploded.dropna(subset=['matched_keywords'])
        df_kw_exploded = df_kw_exploded[df_kw_exploded['matched_keywords'] != ''] # Filter empty keywords
    except Exception as e:
        logger.error(f"Error exploding HF keywords: {e}"); return pd.DataFrame()
    
    if df_kw_exploded.empty:
        logger.warning("HF DataFrame empty after exploding keywords."); return pd.DataFrame()

    # Find the index for the row with max downloads within each keyword group
    # This gives the top model for each keyword
    idx_kw_max_downloads = df_kw_exploded.groupby('matched_keywords')['downloads'].idxmax()
    keyword_summary_df = df_kw_exploded.loc[idx_kw_max_downloads].reset_index(drop=True)

    keyword_summary = keyword_summary_df.rename(columns={
        'matched_keywords': 'Keyword', 
        'matched_sectors': 'Sector(s) of Top Model', 
        'modelId': 'Top Model ID', 
        'downloads': 'Top Model Downloads', 
        'pipeline_tag': 'Top Model Task'
    })

    if 'Sector(s) of Top Model' in keyword_summary.columns:
        keyword_summary['Sector(s) of Top Model'] = keyword_summary['Sector(s) of Top Model'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) else (x if isinstance(x,str) else '')
        )
    if 'Top Model Downloads' in keyword_summary.columns:
        keyword_summary['Top Model Downloads Str'] = keyword_summary['Top Model Downloads'].apply(lambda x: f"{int(x):,}")
    
    keyword_summary = keyword_summary.sort_values(by='Top Model Downloads', ascending=False).reset_index(drop=True)
    logger.info("HF keyword summary calculation complete.")
    
    final_cols = ['Keyword', 'Sector(s) of Top Model', 'Top Model ID', 'Top Model Downloads Str', 'Top Model Task', 'Top Model Downloads']
    return keyword_summary[[col for col in final_cols if col in keyword_summary.columns]]

def get_top_tasks_per_sector(df):
    """Calculates the top 3 pipeline tags (tasks) per sector for HF models, weighted by downloads."""
    logger.info("Calculating top HF tasks per sector...")
    if df is None or df.empty or 'matched_sectors' not in df.columns or 'pipeline_tag' not in df.columns:
        logger.warning("HF DataFrame invalid or missing required columns for top task calculation.")
        return pd.Series(dtype=str)

    try:
        df_exploded = df.explode('matched_sectors')
        df_exploded = df_exploded.dropna(subset=['matched_sectors', 'pipeline_tag', 'downloads'])
        df_exploded = df_exploded[df_exploded['pipeline_tag'] != 'unknown']

        if df_exploded.empty:
            logger.warning("No valid HF data after exploding/filtering for top task calculation.")
            return pd.Series(dtype=str)

        task_downloads = df_exploded.groupby(['matched_sectors', 'pipeline_tag'])['downloads'].sum().reset_index()
        top_tasks = task_downloads.sort_values(['matched_sectors', 'downloads'], ascending=[True, False]) \
                                  .groupby('matched_sectors')['pipeline_tag'] \
                                  .apply(lambda x: ', '.join(x.head(3))) \
                                  .rename('top_3_tasks')
        logger.info("Top HF tasks calculation complete.")
        return top_tasks
    except Exception as e:
        logger.error(f"Error calculating top HF tasks per sector: {e}", exc_info=True)
        return pd.Series(dtype=str)

# --- Analysis Functions for MCP Server Data ---
def get_mcp_summary_by_category(df_mcp, main_category_col='keyword_categories', sub_category_col=None):
    """
    Generates summary statistics for MCP servers, primarily grouped by their keyword categories.
    Can also be used for NAICS sectors if descriptions are matched later.
    """
    logger.info(f"Calculating MCP summary by category: {main_category_col}")
    if df_mcp is None or df_mcp.empty or main_category_col not in df_mcp.columns:
        logger.warning(f"MCP DataFrame invalid or empty, or missing '{main_category_col}' for summary.")
        return pd.DataFrame()

    # Explode by the main category column (e.g., 'keyword_categories')
    df_exploded = df_mcp.explode(main_category_col)
    df_exploded = df_exploded.dropna(subset=[main_category_col])
    
    if df_exploded.empty:
        logger.warning(f"MCP DataFrame empty after exploding on '{main_category_col}'.")
        return pd.DataFrame()

    # Aggregations
    summary = df_exploded.groupby(main_category_col).agg(
        mcp_server_count=('qualifiedName', 'nunique'),
        total_usage_tool_calls=('usage_tool_calls', 'sum'),
        avg_usage_tool_calls=('usage_tool_calls', 'mean'),
        servers_deployed_count=('isDeployed', lambda x: x.astype(bool).sum()), # Count True values
        security_scan_passed_count=('security_scan_passed', lambda x: x.astype(bool).sum()) # Count True values
    ).reset_index()

    # Add counts of finance affordances
    for affordance in ['execution', 'info', 'interaction']:
        affordance_flag_col = f'has_finance_{affordance}'
        if affordance_flag_col in df_exploded.columns:
            affordance_summary = df_exploded.groupby(main_category_col)[affordance_flag_col].sum().reset_index(name=f'count_finance_{affordance}')
            summary = pd.merge(summary, affordance_summary, on=main_category_col, how='left')
            summary[f'count_finance_{affordance}'] = summary[f'count_finance_{affordance}'].fillna(0).astype(int)
        else:
            summary[f'count_finance_{affordance}'] = 0


    summary = summary.sort_values(by='mcp_server_count', ascending=False)
    logger.info("MCP summary by category calculation complete.")
    return summary

def get_mcp_affordance_overview(df_mcp):
    """Provides an overview of financial affordances across all MCP servers."""
    logger.info("Calculating MCP financial affordance overview.")
    if df_mcp is None or df_mcp.empty:
        logger.warning("MCP DataFrame invalid or empty for affordance overview.")
        return pd.DataFrame()

    affordance_counts = {}
    total_servers = df_mcp['qualifiedName'].nunique()
    affordance_counts['Total Unique Servers'] = total_servers
    
    for affordance in ['execution', 'info', 'interaction']:
        col_name = f'has_finance_{affordance}'
        if col_name in df_mcp.columns:
            # Count unique servers that have this affordance
            # Need to be careful if a server appears multiple times due to keyword matching before unique
            # Assuming df_mcp might have duplicates of servers if matched by multiple keywords,
            # so group by server then check affordance.
            unique_servers_with_affordance = df_mcp.drop_duplicates(subset=['qualifiedName']).groupby('qualifiedName')[col_name].any().sum()
            affordance_counts[f'Servers with Finance {affordance.capitalize()}'] = unique_servers_with_affordance
        else:
            affordance_counts[f'Servers with Finance {affordance.capitalize()}'] = 0
            
    overview_df = pd.DataFrame([affordance_counts])
    logger.info("MCP financial affordance overview complete.")
    return overview_df.T.rename(columns={0: 'Count'})
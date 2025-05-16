# %%
# =============================================================================
# Logic for Local Analysis of Bulk MCP Server Data
# =============================================================================
import pandas as pd
import json
import os
import re # For keyword searching
from hf_models_monitoring_test.config_utils import (
    logger,
    FINANCE_SECTOR_KEYWORDS_CONFIG,
    THREAT_MODEL_KEYWORDS_CONFIG,
    FINANCE_AFFORDANCE_KEYWORDS_CONFIG
)

def load_bulk_mcp_data(json_file_path):
    """Loads the bulk MCP server details JSON into a pandas DataFrame."""
    logger.info(f"Loading bulk MCP data from: {json_file_path}")
    if not os.path.exists(json_file_path):
        logger.error(f"Data file not found: {json_file_path}")
        return pd.DataFrame()
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            df = pd.DataFrame(data)
            logger.info(f"Successfully loaded {len(df)} records into DataFrame.")
            # Basic preprocessing: ensure 'tools' is a list of dicts
            if 'tools' in df.columns:
                df['tools'] = df['tools'].apply(lambda x: x if isinstance(x, list) else [])
            else:
                df['tools'] = pd.Series([[] for _ in range(len(df))])
            return df
        else:
            logger.error("JSON root is not a list as expected.")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading or parsing JSON file {json_file_path}: {e}", exc_info=True)
        return pd.DataFrame()

def match_server_to_keywords(df, keyword_configs_map):
    """
    Matches servers to keywords based on displayName, description, and tool text.
    Adds new columns for matched categories.
    keyword_configs_map should be a dict like:
    { "matched_finance_sectors": FINANCE_SECTOR_KEYWORDS_CONFIG,
      "matched_threat_models": THREAT_MODEL_KEYWORDS_CONFIG }
    """
    logger.info("Starting keyword matching for servers...")
    if df.empty:
        logger.warning("Input DataFrame is empty, skipping keyword matching.")
        return df

    for new_col_name, keyword_config in keyword_configs_map.items():
        logger.info(f"Processing for category: {new_col_name}")
        matched_categories_for_servers = []
        for index, server in df.iterrows():
            server_matches = set()
            # Compile text to search: displayName, description
            text_to_search = ""
            if pd.notna(server.get('displayName')):
                text_to_search += server['displayName'].lower() + " "
            if pd.notna(server.get('description')):
                text_to_search += server['description'].lower() + " "
            
            # Add tool names and descriptions to search text
            server_tools = server.get('tools', [])
            if isinstance(server_tools, list):
                for tool in server_tools:
                    if isinstance(tool, dict):
                        if pd.notna(tool.get('name')):
                            text_to_search += tool['name'].lower() + " "
                        if pd.notna(tool.get('description')):
                            text_to_search += tool['description'].lower() + " "
            
            if not text_to_search.strip(): # Skip if no text content
                matched_categories_for_servers.append(list(server_matches))
                continue

            for category, keywords in keyword_config.items():
                for keyword in keywords:
                    # Using regex for whole word matching, case insensitive
                    if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', text_to_search):
                        server_matches.add(category)
                        break # Found a match for this category, move to next category
            matched_categories_for_servers.append(list(server_matches))
        df[new_col_name] = matched_categories_for_servers
        logger.info(f"Finished processing for {new_col_name}. Example matches: {df[new_col_name].apply(lambda x: len(x) > 0).sum()} servers.")
    
    logger.info("Keyword matching complete.")
    return df

def analyze_server_affordances(df, affordance_keyword_config):
    """
    Analyzes server tools for financial affordances.
    Adds 'has_finance_{type}' and 'finance_{type}_tools' columns.
    """
    logger.info("Starting financial affordance analysis for server tools...")
    if df.empty:
        logger.warning("Input DataFrame is empty, skipping affordance analysis.")
        return df
    if 'tools' not in df.columns:
        logger.error("'tools' column not found in DataFrame. Cannot analyze affordances.")
        for affordance_type in affordance_keyword_config.keys():
            df[f'has_finance_{affordance_type}'] = False
            df[f'finance_{affordance_type}_tools'] = pd.Series([[] for _ in range(len(df))])
        return df

    for affordance_type, keywords in affordance_keyword_config.items():
        has_affordance_col = f'has_finance_{affordance_type}'
        tools_list_col = f'finance_{affordance_type}_tools'
        
        affordance_present_flags = []
        affordance_tools_lists = []

        for index, server in df.iterrows():
            current_server_affordance_tools = set()
            server_tools = server.get('tools', []) # Should be a list of dicts
            
            if isinstance(server_tools, list):
                for tool in server_tools:
                    if not isinstance(tool, dict): continue
                    
                    tool_name = tool.get('name', '').lower()
                    tool_description = tool.get('description', '').lower() if pd.notna(tool.get('description')) else ''
                    tool_text = tool_name + " " + tool_description
                    
                    if not tool_text.strip(): continue

                    for keyword in keywords:
                        if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', tool_text):
                            current_server_affordance_tools.add(tool.get('name', 'Unnamed Tool'))
                            # A tool can match multiple keywords for the same affordance,
                            # but we only add its name once to the set for this affordance type.
                
            affordance_tools_lists.append(list(current_server_affordance_tools))
            affordance_present_flags.append(len(current_server_affordance_tools) > 0)

        df[has_affordance_col] = affordance_present_flags
        df[tools_list_col] = affordance_tools_lists
        logger.info(f"Processed affordance '{affordance_type}'. Found in {df[has_affordance_col].sum()} servers.")
        
    logger.info("Financial affordance analysis complete.")
    return df

def generate_category_summary(df, category_col_name, value_col_name='qualifiedName'):
    """Generates a summary count for a given category column."""
    if df.empty or category_col_name not in df.columns:
        logger.warning(f"Cannot generate summary for '{category_col_name}', column missing or df empty.")
        return pd.DataFrame()

    # Explode if the category column contains lists
    if df[category_col_name].apply(isinstance, args=(list,)).any():
        df_exploded = df.explode(category_col_name)
    else:
        df_exploded = df.copy()
    
    df_exploded = df_exploded.dropna(subset=[category_col_name])
    if df_exploded.empty:
        logger.info(f"No data to summarize for '{category_col_name}' after cleaning.")
        return pd.DataFrame()

    summary = df_exploded.groupby(category_col_name)[value_col_name].nunique().reset_index(name='server_count')
    summary = summary.sort_values(by='server_count', ascending=False).reset_index(drop=True)
    return summary


# %%

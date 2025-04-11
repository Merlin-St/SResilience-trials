# %%
# =============================================================================
# Data Loading, Preprocessing, and Analysis Functions
# =============================================================================
import pandas as pd
import logging
import os
from tqdm import tqdm
from collections import Counter # Used indirectly via get_top_tasks_per_sector

# Import logger and utility functions from config_utils
from config_utils import logger, parse_list_string

# --- Keyword Loading Function (Phase 1 Helper) ---
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
        return None, None # Return None for both dicts

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
            sector_name = str(row['sector_name']).strip() # Strip name
            keywords_str = row['keywords']

            # Basic validation
            if pd.isna(sector_code) or pd.isna(sector_name) or sector_name == "" or pd.isna(keywords_str):
                logger.warning(f"Skipping row due to missing code, name, or keywords: Code={sector_code}, Name='{sector_name}'")
                continue

            # Ensure sector_code is treated consistently (e.g., as string)
            sector_code_str = str(sector_code).strip() # Use string representation for consistency

            # Check for duplicate sector codes - use first occurrence if duplicates exist
            if sector_code_str in processed_codes:
                logger.warning(f"Duplicate sector code '{sector_code_str}' found. Using first occurrence.")
                duplicates_found = True
                continue
            processed_codes.add(sector_code_str)

            # Create unique ID (used as key)
            sector_id = f"{sector_code_str} - {sector_name}"

            # Parse keywords
            keywords_list = [k.strip() for k in str(keywords_str).split(',') if k.strip()]
            if not keywords_list:
                 logger.warning(f"Skipping sector '{sector_id}' due to no valid keywords parsed.")
                 continue # Skip sectors with no valid keywords parsed

            # Get CNI percentage (optional column) - ASSUME IT'S ALREADY 0-100 in CSV
            cni_percentage = pd.to_numeric(row.get('cni_percentage'), errors='coerce')
            if pd.isna(cni_percentage):
                cni_percentage = 0.0 # Default to 0.0 float if missing or invalid

            # Store data mapped to the unique ID
            sector_data_map[sector_id] = {
                'code': sector_code_str,
                'name': sector_name,
                'keywords': keywords_list,
                'cni_percentage': cni_percentage # Store percentage as read
            }

        logger.info(f"Loaded data for {len(sector_data_map)} unique sectors.")
        if duplicates_found:
             logger.warning("Duplicate sector codes were encountered and skipped (first occurrence kept).")

        # Create separate CNI map for convenience
        sector_cni_map = {sid: data['cni_percentage'] for sid, data in sector_data_map.items()}

        return sector_data_map, sector_cni_map

    except Exception as e:
        logger.error(f"Error reading or parsing NAICS CSV: {e}", exc_info=True)
        return None, None


# --- Data Loading and Preprocessing Functions (Phase 4 Helper) ---
def load_and_preprocess_data(csv_file):
    """Loads and preprocesses the collected model data from CSV."""
    logger.info(f"Loading collected data from {csv_file}...")
    if not os.path.exists(csv_file):
        logger.error(f"ERROR: Input file not found: {csv_file}")
        return None
    try:
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded {len(df)} unique models from CSV.")
        # Use plain tqdm
        tqdm.pandas()
        for col in ['tags', 'matched_sectors', 'matched_keywords']:
            if col in df.columns:
                logger.info(f"Parsing list strings in column: {col}")
                # Use parse_list_string from config_utils
                df[col] = df[col].progress_apply(parse_list_string)
            else:
                 logger.warning(f"Column '{col}' not found in CSV during preprocessing.")
                 df[col] = [[] for _ in range(len(df))] # Add empty list column if missing

        df['lastModified'] = pd.to_datetime(df['lastModified'], errors='coerce')
        df['downloads'] = pd.to_numeric(df['downloads'], errors='coerce').fillna(0).astype(int)
        df['likes'] = pd.to_numeric(df['likes'], errors='coerce').fillna(0).astype(int)
        df['sector_count'] = df['matched_sectors'].apply(len)
        df['pipeline_tag'] = df['pipeline_tag'].fillna('unknown') # Ensure pipeline tag has a default
        # NOTE: Add handling for new columns (parameter_count etc.) here later for Request 3
        logger.info("Preprocessing complete.")
        return df.reset_index(drop=True)
    except Exception as e:
        logger.error(f"Error loading or preprocessing data from {csv_file}: {e}", exc_info=True)
        return None

# --- Analysis Functions (Phase 4 Helpers) ---

# Helper function for get_sector_summary
def get_top_model_for_group(group):
    """Finds the single row with max downloads within a group (sector)."""
    if group.empty:
        return pd.Series({
            'top_model_id': 'N/A',
            'top_model_downloads': 0,
            'top_model_keywords': []
        })
    # idxmax() finds the index of the *first* occurrence of the max value
    top_model_row = group.loc[group['downloads'].idxmax()]
    return pd.Series({
        'top_model_id': top_model_row['modelId'],
        'top_model_downloads': top_model_row['downloads'],
        # Important: Get the keywords associated with the *original model*
        'top_model_keywords': top_model_row['matched_keywords'] # Assumes 'matched_keywords' is present
    })

def get_sector_summary(df, sector_data_map):
    """
    Calculates summary statistics per sector, separating ID and Name.
    Uses .apply() to robustly find the single top model per sector.
    """
    logger.info("Calculating sector summary statistics (v2)...")
    if df is None or 'matched_sectors' not in df.columns:
        logger.error("DataFrame invalid for sector summary.")
        return pd.DataFrame()
    if sector_data_map is None:
        logger.error("Sector data map is required for sector summary.")
        return pd.DataFrame()

    # Ensure 'matched_sectors' contains lists
    if df['matched_sectors'].apply(lambda x: not isinstance(x, list)).any():
         logger.warning("Found non-list entries in 'matched_sectors'. Attempting to fix.")
         # Use parse_list_string from config_utils
         df['matched_sectors'] = df['matched_sectors'].apply(lambda x: parse_list_string(x) if isinstance(x, str) else (x if isinstance(x, list) else []))
         df = df[df['matched_sectors'].apply(isinstance, args=(list,))]

    try:
        # Explode the dataframe based on the 'matched_sectors' column (unique IDs like "Code - Name")
        df_exploded = df[['modelId', 'downloads', 'likes', 'matched_sectors', 'matched_keywords']].explode('matched_sectors')
        if df_exploded['matched_sectors'].isnull().any():
            logger.warning(f"Found {df_exploded['matched_sectors'].isnull().sum()} null sectors after exploding.")
            df_exploded = df_exploded.dropna(subset=['matched_sectors'])
    except TypeError as e:
        logger.error(f"Error exploding sectors: {e}")
        return pd.DataFrame()
    except KeyError as e:
        logger.error(f"Missing expected column during explosion setup: {e}")
        return pd.DataFrame()


    if df_exploded.empty:
        logger.warning("DataFrame empty after exploding sectors.")
        return pd.DataFrame()

    # --- Calculate Base Summary Stats ---
    logger.info("Aggregating base sector stats...")
    sector_summary_base = df_exploded.groupby('matched_sectors').agg(
        model_count=('modelId', 'nunique'),
        total_downloads=('downloads', 'sum'),
        total_likes=('likes', 'sum'),
        average_downloads=('downloads', 'mean')
    ).reset_index()
    logger.info(f"Base aggregation complete. Found {len(sector_summary_base)} unique sectors.")

    # --- Find Top Model Details using Apply ---
    logger.info("Applying function to find top model per sector group...")
    top_model_details = df_exploded.groupby('matched_sectors').apply(get_top_model_for_group)
    logger.info("Top model details application complete.")

    # --- Merge Top Model Details ---
    logger.info("Merging top model details with base summary...")
    sector_summary_merged = pd.merge(
        sector_summary_base,
        top_model_details,
        on='matched_sectors',
        how='left' # Keep all sectors from the base summary
    )
    logger.info(f"Merge complete. Rows after merge: {len(sector_summary_merged)}")

    # --- Add Sector Code and Name ---
    logger.info("Mapping sector code and name...")
    sector_summary_merged['sector_code'] = sector_summary_merged['matched_sectors'].map(lambda sid: sector_data_map.get(sid, {}).get('code', 'N/A'))
    sector_summary_merged['sector_name'] = sector_summary_merged['matched_sectors'].map(lambda sid: sector_data_map.get(sid, {}).get('name', 'N/A'))

    # --- Final Formatting ---
    logger.info("Applying final formatting...")
    sector_summary_final = sector_summary_merged.rename(columns={'matched_sectors': 'sector_id_full'}) # Keep original ID

    # Fill NaNs and format numeric columns
    sector_summary_final['average_downloads'] = sector_summary_final['average_downloads'].fillna(0).round(0).astype(int)
    sector_summary_final['total_downloads_str'] = sector_summary_final['total_downloads'].apply(lambda x: f"{x:,.0f}")
    sector_summary_final['top_model_downloads'] = sector_summary_final['top_model_downloads'].fillna(0).astype(int)
    sector_summary_final['top_model_downloads_str'] = sector_summary_final['top_model_downloads'].apply(lambda x: f"{x:,.0f}")

    # Format keywords list to string
    if 'top_model_keywords' not in sector_summary_final.columns:
        sector_summary_final['top_model_keywords'] = [[]] * len(sector_summary_final) # Add column if missing
    sector_summary_final['top_model_keywords'] = sector_summary_final['top_model_keywords'].fillna('') # Fill NaN before apply
    sector_summary_final['top_model_keywords_str'] = sector_summary_final['top_model_keywords'].apply(lambda x: ', '.join(x) if isinstance(x, list) and x else '')

    # Sort and select/order columns
    sector_summary_final = sector_summary_final.sort_values(by='model_count', ascending=False).reset_index(drop=True)
    logger.info("Sector summary calculation complete (v2).")

    final_cols_ordered = [
        'sector_code', 'sector_name', 'model_count', 'total_downloads', 'total_likes', 'average_downloads',
        'top_model_id', 'top_model_downloads', 'top_model_keywords',
        'total_downloads_str', 'top_model_downloads_str', 'top_model_keywords_str',
        'sector_id_full'
    ]
    # Ensure only existing columns are selected
    final_cols_ordered = [col for col in final_cols_ordered if col in sector_summary_final.columns]
    return sector_summary_final[final_cols_ordered]

# --- Function to get top tasks per sector weighted by downloads ---
def get_top_tasks_per_sector(df):
    """Calculates the top 3 pipeline tags (tasks) per sector, weighted by downloads."""
    logger.info("Calculating top tasks per sector...")
    if df is None or df.empty or 'matched_sectors' not in df.columns or 'pipeline_tag' not in df.columns:
        logger.warning("DataFrame invalid or missing required columns for top task calculation.")
        return pd.Series(dtype=str) # Return empty Series

    try:
        df_exploded = df.explode('matched_sectors')
        df_exploded = df_exploded.dropna(subset=['matched_sectors', 'pipeline_tag', 'downloads'])
        df_exploded = df_exploded[df_exploded['pipeline_tag'] != 'unknown'] # Exclude unknown tasks

        if df_exploded.empty:
            logger.warning("No valid data after exploding/filtering for top task calculation.")
            return pd.Series(dtype=str)

        # Group by sector and pipeline tag, summing downloads
        task_downloads = df_exploded.groupby(['matched_sectors', 'pipeline_tag'])['downloads'].sum().reset_index()

        # For each sector, find the top 3 tasks by summed downloads
        top_tasks = task_downloads.sort_values(['matched_sectors', 'downloads'], ascending=[True, False]) \
                                  .groupby('matched_sectors')['pipeline_tag'] \
                                  .apply(lambda x: ', '.join(x.head(3))) \
                                  .rename('top_3_tasks')

        logger.info("Top tasks calculation complete.")
        return top_tasks

    except Exception as e:
        logger.error(f"Error calculating top tasks per sector: {e}", exc_info=True)
        return pd.Series(dtype=str) # Return empty Series on error

# --- Keyword Summary Function ---
def get_keyword_summary(df):
    """Calculates summary statistics per keyword."""
    logger.info("Calculating keyword summary statistics...")
    if df is None or 'matched_keywords' not in df.columns: logger.error("DataFrame invalid for keyword summary."); return pd.DataFrame()
    if df['matched_keywords'].apply(lambda x: not isinstance(x, list)).any():
         logger.error("Found non-list entries in 'matched_keywords'. Preprocessing might have failed.")
         df = df[df['matched_keywords'].apply(isinstance, args=(list,))]
    try:
        df_kw_exploded = df.explode('matched_keywords')
        if df_kw_exploded['matched_keywords'].isnull().any(): logger.warning(f"Found {df_kw_exploded['matched_keywords'].isnull().sum()} null keywords after exploding."); df_kw_exploded = df_kw_exploded.dropna(subset=['matched_keywords'])
        df_kw_exploded = df_kw_exploded[df_kw_exploded['matched_keywords'] != ''] # Filter empty keywords
    except TypeError as e: logger.error(f"Error exploding keywords: {e}"); return pd.DataFrame()
    if df_kw_exploded.empty: logger.warning("DataFrame empty after exploding keywords."); return pd.DataFrame()

    # Find the index label for the row with max downloads within each keyword group
    idx_kw_max_downloads = df_kw_exploded.loc[df_kw_exploded.groupby('matched_keywords')['downloads'].idxmax()]

    if idx_kw_max_downloads.empty: logger.warning("Could not determine top models for any keyword."); return pd.DataFrame()

    # Select required columns directly
    keyword_summary = idx_kw_max_downloads[['matched_keywords', 'matched_sectors', 'modelId', 'downloads', 'pipeline_tag']].reset_index(drop=True)
    keyword_summary = keyword_summary.rename(columns={'matched_keywords': 'Keyword', 'matched_sectors': 'Sector(s) of Top Model', 'modelId': 'Top Model ID', 'downloads': 'Top Model Downloads', 'pipeline_tag': 'Top Model Task'})

    # Format columns
    keyword_summary['Sector(s) of Top Model'] = keyword_summary['Sector(s) of Top Model'].apply(lambda x: ', '.join(x) if isinstance(x, list) else '')
    keyword_summary['Top Model Downloads Str'] = keyword_summary['Top Model Downloads'].apply(lambda x: f"{x:,.0f}")
    keyword_summary = keyword_summary.sort_values(by='Top Model Downloads', ascending=False).reset_index(drop=True)

    logger.info("Keyword summary calculation complete.")
    # Select and order final columns
    return keyword_summary[['Keyword', 'Sector(s) of Top Model', 'Top Model ID', 'Top Model Downloads Str', 'Top Model Task']]
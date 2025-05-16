# %%
# =============================================================================
# Inspector for Bulk MCP Download JSON Outputs
# =============================================================================
import json
import pandas as pd
import os

# --- Configuration - Point to your output files ---
# These names should match what's in your bulk_mcp_config.py (or equivalent)
ALL_SERVERS_SUMMARIES_JSON = "all_mcp_server_summaries.json"
ALL_SERVERS_DETAILS_COMPLETE_JSON = "all_mcp_server_details_complete.json"

def load_json_to_dataframe(json_file_path):
    """Loads a JSON file (expected to be a list of objects) into a pandas DataFrame."""
    if not os.path.exists(json_file_path):
        print(f"Error: File not found at {json_file_path}")
        return None
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            df = pd.DataFrame(data)
            print(f"\nSuccessfully loaded {len(df)} records from {json_file_path}")
            return df
        else:
            print(f"Error: JSON file {json_file_path} does not contain a list of objects at the root.")
            return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {json_file_path}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading {json_file_path}: {e}")
        return None

def inspect_dataframe(df, df_name="DataFrame", check_specific_fields=False):
    """Prints basic inspection information for a DataFrame."""
    if df is None or df.empty:
        print(f"{df_name} is empty or not loaded.")
        return

    print(f"\n--- Inspecting {df_name} ---")
    print(f"Shape: {df.shape}") # (rows, columns)
    
    print(f"\nFirst 3 rows of {df_name}:")
    print(df.head(3))
    
    print(f"\n{df_name} Info (memory usage, dtypes, non-null counts):")
    df.info(verbose=True, show_counts=True) # More detailed info
    
    print(f"\n{df_name} Columns (Total: {len(df.columns)}):")
    print(list(df.columns))
    
    if check_specific_fields:
        print(f"\nChecking for specific metadata fields in {df_name}:")
        specific_fields_to_check = ['qualifiedName', 'displayName', 'description', 'homepage', 'useCount', 'createdAt', 'isDeployed', 'tools', 'security', 'iconUrl', 'connections']
        for field in specific_fields_to_check:
            if field in df.columns:
                non_null_count = df[field].notna().sum()
                print(f"- Field '{field}': Present. Non-null count: {non_null_count} out of {len(df)}")
                # Show a sample of non-null values for some fields
                if non_null_count > 0 and field in ['description', 'homepage', 'useCount', 'createdAt', 'isDeployed']:
                    sample_values = df[df[field].notna()][field].head(2).tolist()
                    print(f"  Sample non-null values for '{field}': {sample_values}")
            else:
                print(f"- Field '{field}': NOT FOUND")
    
    # Check for any error flags if this is the details dataframe
    error_flags_to_check = ['error_fetching_details', 'error_processing_future']
    for err_flag in error_flags_to_check:
        if err_flag in df.columns:
            # Ensure the column is treated as boolean, handling potential NaNs if it's object type
            # If it's already boolean, this won't harm. If object, convert True strings to True, others to False or NaN
            bool_series = pd.to_numeric(df[err_flag], errors='coerce').fillna(0).astype(bool) if df[err_flag].dtype == 'object' else df[err_flag].fillna(False)
            errors = df[bool_series]

            if not errors.empty:
                print(f"\nFound {len(errors)} servers with '{err_flag}' = True in {df_name}:")
                cols_to_show_errors = ['qualifiedName', err_flag]
                if 'error_message' in errors.columns:
                    cols_to_show_errors.append('error_message')
                if 'exception_details' in errors.columns:
                     cols_to_show_errors.append('exception_details')
                print(errors[cols_to_show_errors].head())
            else:
                print(f"\nNo servers found with '{err_flag}' = True in {df_name}.")
        else:
            print(f"\nError flag column '{err_flag}' not present in {df_name}.")


# %%
# --- Inspect Server Summaries (from /servers endpoint) ---
print("="*50)
print("Attempting to load and inspect server summaries...")
df_summaries = load_json_to_dataframe(ALL_SERVERS_SUMMARIES_JSON)
if df_summaries is not None:
    inspect_dataframe(df_summaries, "MCP Server Summaries (from /servers list)")
    
    # Specific checks for summaries
    if 'qualifiedName' in df_summaries.columns:
        print(f"\nNumber of unique qualifiedNames in summaries: {df_summaries['qualifiedName'].nunique()}")
        if df_summaries['qualifiedName'].nunique() != len(df_summaries):
            print("Warning: Found duplicate qualifiedNames in the summaries list!")


# %%
# --- Inspect Server Details (merged data from /servers and /servers/{qname}) ---
print("\n\n" + "="*50)
print("Attempting to load and inspect server details (merged data)...")
df_details = load_json_to_dataframe(ALL_SERVERS_DETAILS_COMPLETE_JSON)
if df_details is not None:
    inspect_dataframe(df_details, "MCP Server Details (Merged)", check_specific_fields=True)

    # Example: Look at the structure of the 'tools' for the first server that has them
    if 'tools' in df_details.columns:
        first_server_with_tools = None
        # Iterate to find a row where 'tools' is a non-empty list
        for index, row in df_details.iterrows():
            tools_data = row['tools']
            if isinstance(tools_data, list) and len(tools_data) > 0:
                # Check if the first item in tools is a dictionary (expected structure)
                if isinstance(tools_data[0], dict):
                    first_server_with_tools = row
                    break
        
        if first_server_with_tools is not None:
            print(f"\nExample 'tools' structure from server: {first_server_with_tools.get('qualifiedName', 'N/A')}")
            tools_sample = first_server_with_tools['tools']
            # Pretty print the tools list (or a sample) for the first server
            print(json.dumps(tools_sample[:2], indent=2)) # Print first 2 tools for brevity
        else:
            print("\nNo servers with a correctly structured, non-empty 'tools' list found for sample display.")
            
    # Example: Check distribution of 'security.scanPassed'
    # The 'security' column itself is an object/dict. We need to access 'scanPassed' within it.
    if 'security' in df_details.columns:
        # Create a 'scanPassed' column safely
        def get_scan_passed(security_obj):
            if isinstance(security_obj, dict):
                return security_obj.get('scanPassed')
            return None
        
        df_details['temp_scanPassed'] = df_details['security'].apply(get_scan_passed)
        print("\nDistribution of 'security.scanPassed':")
        print(df_details['temp_scanPassed'].value_counts(dropna=False))
        # df_details.drop(columns=['temp_scanPassed'], inplace=True) # Clean up temp column

print("\n\n" + "="*50)
print("Inspection script finished.")
print(f"Summary file checked: {ALL_SERVERS_SUMMARIES_JSON}")
print(f"Details file checked: {ALL_SERVERS_DETAILS_COMPLETE_JSON}")


# %%

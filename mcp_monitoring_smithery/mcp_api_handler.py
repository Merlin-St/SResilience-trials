# %%
# =============================================================================
# Smithery MCP Registry API Interaction Functions
# =============================================================================
import requests
import time
import json
from urllib.parse import urlencode
import concurrent.futures

# Import logger and constants from config_utils
from hf_models_monitoring_test.config_utils import (
    logger, MCP_API_BASE_URL, MCP_REQUEST_TIMEOUT, 
    API_DELAY, MCP_PAGE_SIZE, MCP_MAX_WORKERS
)

# --- Smithery API Interaction Functions ---

def fetch_mcp_servers(query, api_key, page=1, page_size=MCP_PAGE_SIZE, base_url=MCP_API_BASE_URL):
    """
    Fetches a paginated list of MCP servers from the Smithery Registry API.
    """
    endpoint = f"{base_url}/servers"
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json'
    }
    params = {'page': page, 'pageSize': page_size}
    if query:
        params['q'] = query

    logger.debug(f"Fetching MCP servers list: query='{query}', page={page}, pageSize={page_size}")
    try:
        response = requests.get(endpoint, headers=headers, params=params, timeout=MCP_REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error fetching MCP servers list for query '{query}': {e.response.status_code} - {e.response.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error fetching MCP servers list for query '{query}': {e}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for MCP servers list (query '{query}'): {e}. Response text: {response.text[:200] if response else 'No response text'}")
    return None


def get_mcp_server_details(qualified_name, api_key, base_url=MCP_API_BASE_URL):
    """
    Retrieves detailed information about a specific MCP server by its qualified name.
    """
    endpoint = f"{base_url}/servers/{qualified_name}"
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json'
    }
    # This log will now primarily appear from within the thread
    # logger.debug(f"Preparing to fetch details for MCP server: {qualified_name}") 
    try:
        response = requests.get(endpoint, headers=headers, timeout=MCP_REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error fetching details for MCP server '{qualified_name}': {e.response.status_code} - {e.response.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error fetching details for MCP server '{qualified_name}': {e}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for MCP server details '{qualified_name}': {e}. Response text: {response.text[:200] if response else 'No response text'}")
    return None


def analyze_server_tools(tools_array, affordance_keywords_config):
    identified_affordances = {"execution": [], "information_gathering": [], "agent_interaction": []}
    all_tool_details_list = []
    if not tools_array or not isinstance(tools_array, list):
        return identified_affordances, all_tool_details_list

    for tool in tools_array:
        if not isinstance(tool, dict): continue
        tool_name_original = tool.get('name', 'N/A')
        tool_name_lower = tool_name_original.lower()
        tool_description = tool.get('description', '') if tool.get('description') else ''
        tool_description_lower = tool_description.lower()
        all_tool_details_list.append({'name': tool_name_original, 'description': tool_description})
        combined_text = tool_name_lower + " " + tool_description_lower
        for affordance_type, keywords in affordance_keywords_config.items():
            for keyword in keywords:
                if keyword.lower() in combined_text:
                    if tool_name_original not in identified_affordances[affordance_type]:
                        identified_affordances[affordance_type].append(tool_name_original)
    return identified_affordances, all_tool_details_list


def process_server_summary(server_summary, keyword, category_name, api_key, base_url, affordance_keywords_config):
    """
    Helper function to process a single server summary: fetch details and format data.
    This function is designed to be called by the ThreadPoolExecutor.
    """
    q_name = server_summary.get('qualifiedName')
    if not q_name:
        logger.warning(f"Server summary (in thread) missing qualifiedName: {server_summary}")
        return None

    # logger.info(f"Fetching details for server (via thread): {q_name} (Keyword: '{keyword}')")
    details_response = get_mcp_server_details(q_name, api_key, base_url=base_url)
    
    affordances = {"execution": [], "information_gathering": [], "agent_interaction": []}
    all_tools_list = [] # Ensure initialized

    if details_response:
        server_tools_raw = details_response.get('tools')
        affordances, all_tools_list = analyze_server_tools(server_tools_raw, affordance_keywords_config)
        
        server_data_entry = {
            'qualifiedName': q_name, 'displayName': server_summary.get('displayName'),
            'description': server_summary.get('description'), 'homepage': server_summary.get('homepage'),
            'usage_tool_calls': server_summary.get('useCount', 0), 'isDeployed': server_summary.get('isDeployed'),
            'createdAt_list': server_summary.get('createdAt'), 'iconUrl': details_response.get('iconUrl'),
            'deploymentUrl': details_response.get('deploymentUrl'),
            'connections_info': json.dumps(details_response.get('connections', [])),
            'security_scan_passed': details_response.get('security', {}).get('scanPassed') if details_response.get('security') else None,
            'all_tools_details': json.dumps(all_tools_list),
            'finance_execution_tools_list': json.dumps(affordances['execution']),
            'finance_info_tools_list': json.dumps(affordances['information_gathering']),
            'finance_interaction_tools_list': json.dumps(affordances['agent_interaction']),
            'matched_keywords': [keyword], 'keyword_categories': [category_name]
        }
    else:
        logger.warning(f"Could not fetch details for server (via thread): {q_name}. Storing summary info only.")
        server_data_entry = { 
            'qualifiedName': q_name, 'displayName': server_summary.get('displayName'),
            'description': server_summary.get('description'), 'homepage': server_summary.get('homepage'),
            'usage_tool_calls': server_summary.get('useCount', 0), 'isDeployed': server_summary.get('isDeployed'),
            'createdAt_list': server_summary.get('createdAt'), 'iconUrl': None, 'deploymentUrl': None,
            'connections_info': json.dumps([]), 'security_scan_passed': None,
            'all_tools_details': json.dumps(all_tools_list), 
            'finance_execution_tools_list': json.dumps(affordances['execution']), 
            'finance_info_tools_list': json.dumps(affordances['information_gathering']), 
            'finance_interaction_tools_list': json.dumps(affordances['agent_interaction']), 
            'matched_keywords': [keyword], 'keyword_categories': [category_name]
        }
    return q_name, server_data_entry


def collect_all_mcp_data(keyword_map_by_category, api_key, base_url, affordance_keywords_config):
    if not api_key or api_key == "your-smithery-api-token-here":
        logger.error("Smithery API token is not configured. Please set SMITHERY_API_TOKEN.")
        return {}

    all_mcp_servers_data = {} 
    
    for category_name, keywords in keyword_map_by_category.items():
        logger.info(f"--- Starting MCP data collection for Category: {category_name} ---")
        for keyword in keywords:
            logger.info(f"Processing keyword: '{keyword}' in category '{category_name}'")
            current_page = 1
            total_pages = 1 

            while current_page <= total_pages:
                logger.debug(f"Fetching list page {current_page} for keyword '{keyword}' (Page Size: {MCP_PAGE_SIZE})")
                list_response = fetch_mcp_servers(keyword, api_key, page=current_page, base_url=base_url, page_size=MCP_PAGE_SIZE)
                time.sleep(API_DELAY) 

                if list_response and 'servers' in list_response:
                    servers_on_page = list_response['servers']
                    pagination_info = list_response.get('pagination', {})
                    if current_page == 1: total_pages = pagination_info.get('totalPages', current_page)
                    
                    logger.info(f"Found {len(servers_on_page)} servers on page {current_page}/{total_pages} for '{keyword}'. Submitting for detail retrieval.")

                    tasks_for_page = []
                    for server_summary in servers_on_page:
                        q_name = server_summary.get('qualifiedName')
                        if not q_name: 
                            logger.warning(f"Server summary missing qualifiedName on page {current_page} for keyword '{keyword}'.")
                            continue
                        
                        if q_name in all_mcp_servers_data:
                            if keyword not in all_mcp_servers_data[q_name]['matched_keywords']:
                                all_mcp_servers_data[q_name]['matched_keywords'].append(keyword)
                            if category_name not in all_mcp_servers_data[q_name]['keyword_categories']:
                                all_mcp_servers_data[q_name]['keyword_categories'].append(category_name)
                        else:
                            tasks_for_page.append(server_summary)
                    
                    if tasks_for_page:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=MCP_MAX_WORKERS) as executor:
                            future_to_summary_map = {
                                executor.submit(process_server_summary, summary, keyword, category_name, api_key, base_url, affordance_keywords_config): summary
                                for summary in tasks_for_page
                            }
                            # This is where line 210 from the traceback would be, after this loop.
                            # The code below correctly handles results from futures.
                            for future in concurrent.futures.as_completed(future_to_summary_map):
                                original_summary = future_to_summary_map[future]
                                original_q_name = original_summary.get('qualifiedName', 'unknown_qname')
                                try:
                                    result = future.result() 
                                    if result:
                                        res_q_name, server_data_entry = result
                                        if res_q_name not in all_mcp_servers_data:
                                            all_mcp_servers_data[res_q_name] = server_data_entry
                                        else: 
                                            if keyword not in all_mcp_servers_data[res_q_name]['matched_keywords']:
                                                 all_mcp_servers_data[res_q_name]['matched_keywords'].append(keyword)
                                            if category_name not in all_mcp_servers_data[res_q_name]['keyword_categories']:
                                                 all_mcp_servers_data[res_q_name]['keyword_categories'].append(category_name)
                                    # This 'else' is part of 'if result:' and is correctly placed.
                                    else: 
                                        logger.error(f"Task for {original_q_name} (keyword '{keyword}') returned None from process_server_summary.")
                                        
                                # This 'except' handles exceptions from future.result() or within process_server_summary
                                except Exception as exc: 
                                    logger.error(f"Server detail processing for {original_q_name} (keyword '{keyword}') generated an exception: {exc}", exc_info=True)
                                # There is NO 'else:' or 'finally:' directly attached to this try/except block at this indentation level.
                                # Any 'else:' at this point, if not part of an outer loop/conditional, would be a SyntaxError.
                    current_page += 1
                # This 'else' is part of 'if list_response and 'servers' in list_response:'
                else: 
                    logger.warning(f"No servers found or error in list response for '{keyword}', page {current_page}. Ending pagination.")
                    break # Exit while loop for current_page
            logger.info(f"Finished processing keyword: '{keyword}'")
    logger.info(f"--- Finished MCP data collection. Found {len(all_mcp_servers_data)} unique MCP servers. ---")
    return all_mcp_servers_data
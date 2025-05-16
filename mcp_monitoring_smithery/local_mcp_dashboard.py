# %%
# =============================================================================
# Streamlit Dashboard for Locally Analyzed MCP Server Data
# =============================================================================
import streamlit as st
import pandas as pd
import os
import json # For pretty printing details
from hf_models_monitoring_test.config_utils import (
    logger,
    FINANCE_SECTOR_KEYWORDS_CONFIG,
    THREAT_MODEL_KEYWORDS_CONFIG,
    FINANCE_AFFORDANCE_KEYWORDS_CONFIG
)

# For BULK_MCP_DETAILS_JSON_FILE, it appears to be defined in the bulk download configuration
# If it's available in bulk_mcp_config, you could import it like this:
from mcp_monitoring_smithery.bulk_mcp_config import ALL_SERVERS_DETAILS_COMPLETE_JSON as BULK_MCP_DETAILS_JSON_FILE

from mcp_monitoring_smithery.local_mcp_analysis import (
    load_bulk_mcp_data,
    match_server_to_keywords,
    analyze_server_affordances,
    generate_category_summary
)
import plotly.express as px

# --- Helper Functions for Dashboard ---
@st.cache_data # Cache the loaded and processed data
def get_analyzed_mcp_data():
    logger.info("Attempting to load and analyze MCP data for dashboard...")
    df = load_bulk_mcp_data(BULK_MCP_DETAILS_JSON_FILE)
    if df.empty:
        st.error(f"Failed to load data from {BULK_MCP_DETAILS_JSON_FILE}. Ensure the file exists and is valid.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    keyword_configs_to_match = {
        "matched_finance_sectors": FINANCE_SECTOR_KEYWORDS_CONFIG,
        "matched_threat_models": THREAT_MODEL_KEYWORDS_CONFIG
    }
    df = match_server_to_keywords(df, keyword_configs_to_match)
    df = analyze_server_affordances(df, FINANCE_AFFORDANCE_KEYWORDS_CONFIG) # Using global config for affordances
    
    # Generate summaries
    finance_sector_summary = generate_category_summary(df, 'matched_finance_sectors')
    threat_model_summary = generate_category_summary(df, 'matched_threat_models')
    
    # Affordance overview
    affordance_overview_data = {}
    for aff_type in ['execution', 'info', 'interaction']: # Corrected: 'info' not 'information_gathering' in has_finance_ col
        col = f'has_finance_{aff_type}'
        if col in df.columns:
            affordance_overview_data[f'Servers with Finance {aff_type.capitalize()}'] = df[col].sum()
        else:
            affordance_overview_data[f'Servers with Finance {aff_type.capitalize()}'] = 0
    affordance_overview_df = pd.DataFrame.from_dict(affordance_overview_data, orient='index', columns=['Count'])

    logger.info("Data loading and analysis for dashboard complete.")
    return df, finance_sector_summary, threat_model_summary, affordance_overview_df

def display_server_details(server_row):
    """Displays details for a single server row in Streamlit."""
    st.subheader(f"{server_row.get('displayName', 'N/A')} (`{server_row.get('qualifiedName', 'N/A')}`)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Description:** {server_row.get('description', 'N/A')}")
        st.markdown(f"**Homepage:** {server_row.get('homepage', 'N/A')}")
        st.metric("Usage (Tool Calls)", server_row.get('useCount', 0))
    with col2:
        st.markdown(f"**Deployed:** {server_row.get('isDeployed', 'N/A')}")
        st.markdown(f"**Created At:** {server_row.get('createdAt', 'N/A')}")
        if isinstance(server_row.get('security'), dict):
            st.markdown(f"**Security Scan Passed:** {server_row['security'].get('scanPassed', 'N/A')}")
        else:
            st.markdown(f"**Security Scan Passed:** N/A")

    st.markdown("**Matched Finance Sectors:** " + (", ".join(server_row.get('matched_finance_sectors', [])) if server_row.get('matched_finance_sectors') else "None"))
    st.markdown("**Matched Threat Models:** " + (", ".join(server_row.get('matched_threat_models', [])) if server_row.get('matched_threat_models') else "None"))

    with st.expander("Tools & Financial Affordances"):
        st.markdown("**Identified Financial Tools:**")
        for aff_type in ['execution', 'info', 'interaction']:
            tools_list = server_row.get(f'finance_{aff_type}_tools', [])
            if tools_list:
                st.markdown(f"_{aff_type.capitalize()}_: {', '.join(tools_list)}")
        
        st.markdown("**All Tools (Raw JSON Sample):**")
        all_tools = server_row.get('tools', [])
        if isinstance(all_tools, list) and all_tools:
            st.json(all_tools[:3], expanded=False) # Show sample of first 3 tools
        else:
            st.caption("No tools data or not in expected list format.")
    st.markdown("---")

# --- Main Dashboard App ---
st.set_page_config(layout="wide", page_title="Local MCP Analysis Dashboard")
st.title("📊 Local MCP Server Analysis Dashboard")
st.markdown(f"Analyzing data from `{BULK_MCP_DETAILS_JSON_FILE}`")

# Load and process data
analyzed_df, finance_sector_summary_df, threat_model_summary_df, affordance_overview_df = get_analyzed_mcp_data()

if analyzed_df.empty:
    st.stop() # Stop execution if data loading failed

# --- Dashboard Tabs ---
tab_overview, tab_finance_sectors, tab_threat_models, tab_affordances, tab_explore = st.tabs([
    "📈 Overview", "💰 Finance Sectors", "🚨 Threat Models", "🛠️ Affordances", "🔍 Explore Servers"
])

with tab_overview:
    st.header("Overall MCP Landscape")
    col1, col2 = st.columns(2)
    col1.metric("Total Unique Servers Analyzed", analyzed_df['qualifiedName'].nunique())
    if not affordance_overview_df.empty:
        col2.dataframe(affordance_overview_df)
    
    if not finance_sector_summary_df.empty:
        st.subheader("Servers per Finance Sector")
        fig_fs = px.bar(finance_sector_summary_df, x='matched_finance_sectors', y='server_count', title="MCP Servers by Matched Finance Sector")
        st.plotly_chart(fig_fs, use_container_width=True)
    
    if not threat_model_summary_df.empty:
        st.subheader("Servers per Threat Model")
        fig_tm = px.bar(threat_model_summary_df, x='matched_threat_models', y='server_count', title="MCP Servers by Matched Threat Model")
        st.plotly_chart(fig_tm, use_container_width=True)

with tab_finance_sectors:
    st.header("MCP Servers by Finance Sector")
    if not finance_sector_summary_df.empty:
        selected_sector = st.selectbox(
            "Select a Finance Sector:", 
            options=['All'] + list(FINANCE_SECTOR_KEYWORDS_CONFIG.keys()), # Use keys from config
            index=0
        )
        
        filtered_df_sector = analyzed_df
        if selected_sector != 'All':
            filtered_df_sector = analyzed_df[analyzed_df['matched_finance_sectors'].apply(lambda x: selected_sector in x if isinstance(x, list) else False)]
        
        st.metric(f"Servers matching '{selected_sector}'", len(filtered_df_sector))
        st.dataframe(filtered_df_sector[['qualifiedName', 'displayName', 'description', 'useCount', 'matched_finance_sectors', 'has_finance_execution', 'has_finance_info']].head(20), height=400)
        
        if not filtered_df_sector.empty:
            st.subheader(f"Detailed view of selected servers for '{selected_sector}' (max 5):")
            for i, row_idx in enumerate(filtered_df_sector.head(5).index):
                display_server_details(filtered_df_sector.loc[row_idx])
    else:
        st.warning("Finance sector summary not available.")

with tab_threat_models:
    st.header("MCP Servers by Threat Model")
    if not threat_model_summary_df.empty:
        selected_threat_model = st.selectbox(
            "Select a Threat Model:", 
            options=['All'] + list(THREAT_MODEL_KEYWORDS_CONFIG.keys()), # Use keys from config
            index=0
        )
        
        filtered_df_threat = analyzed_df
        if selected_threat_model != 'All':
            filtered_df_threat = analyzed_df[analyzed_df['matched_threat_models'].apply(lambda x: selected_threat_model in x if isinstance(x, list) else False)]

        st.metric(f"Servers matching '{selected_threat_model}'", len(filtered_df_threat))
        st.dataframe(filtered_df_threat[['qualifiedName', 'displayName', 'description', 'useCount', 'matched_threat_models', 'has_finance_execution', 'has_finance_info']].head(20), height=400)

        if not filtered_df_threat.empty:
            st.subheader(f"Detailed view of selected servers for '{selected_threat_model}' (max 5):")
            for i, row_idx in enumerate(filtered_df_threat.head(5).index):
                display_server_details(filtered_df_threat.loc[row_idx])
    else:
        st.warning("Threat model summary not available.")

with tab_affordances:
    st.header("MCP Servers by Financial Affordance")
    affordance_type_map = {
        "Execution": "has_finance_execution",
        "Information Gathering": "has_finance_info",
        "Agent Interaction": "has_finance_interaction"
    }
    selected_aff_display = st.selectbox("Select Affordance Type:", list(affordance_type_map.keys()))
    
    if selected_aff_display:
        aff_col = affordance_type_map[selected_aff_display]
        if aff_col in analyzed_df.columns:
            filtered_df_aff = analyzed_df[analyzed_df[aff_col] == True]
            st.metric(f"Servers with {selected_aff_display} affordance", len(filtered_df_aff))
            
            tool_list_col = f"finance_{selected_aff_display.lower().replace(' ', '_').replace('gathering', 'info')}_tools" # Construct tool list col name
            
            cols_to_show = ['qualifiedName', 'displayName', 'description', 'useCount']
            if tool_list_col in filtered_df_aff.columns:
                cols_to_show.append(tool_list_col)
            
            st.dataframe(filtered_df_aff[cols_to_show].head(20), height=400)

            if not filtered_df_aff.empty:
                st.subheader(f"Detailed view of selected servers with '{selected_aff_display}' (max 5):")
                for i, row_idx in enumerate(filtered_df_aff.head(5).index):
                    display_server_details(filtered_df_aff.loc[row_idx])
        else:
            st.warning(f"Affordance column '{aff_col}' not found.")
            
with tab_explore:
    st.header("Explore All MCP Servers")
    st.markdown("Displaying all analyzed MCP servers. Use filters to narrow down.")
    
    # Simple text search filter (can be expanded)
    search_term = st.text_input("Search in displayName or description (case insensitive):").lower()
    
    df_to_display = analyzed_df
    if search_term:
        df_to_display = analyzed_df[
            df_to_display['displayName'].astype(str).str.lower().str.contains(search_term, na=False) |
            df_to_display['description'].astype(str).str.lower().str.contains(search_term, na=False)
        ]
    
    st.metric("Servers matching filter", len(df_to_display))
    
    # Display a limited set of columns for the long list for performance
    long_list_cols = ['qualifiedName', 'displayName', 'description', 'useCount', 'isDeployed', 'createdAt', 
                      'matched_finance_sectors', 'matched_threat_models', 
                      'has_finance_execution', 'has_finance_info', 'has_finance_interaction']
    st.dataframe(df_to_display[[col for col in long_list_cols if col in df_to_display.columns]].head(100), height=600) # Show first 100

    if not df_to_display.empty:
        st.subheader("Select a server from the filtered list to see full details:")
        selected_qname = st.selectbox("Select Qualified Name:", options=[""] + df_to_display['qualifiedName'].tolist())
        if selected_qname:
            selected_server_row = df_to_display[df_to_display['qualifiedName'] == selected_qname].iloc[0]
            display_server_details(selected_server_row)

logger.info("Streamlit dashboard initialized.")


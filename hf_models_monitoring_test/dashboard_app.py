# %%
# =============================================================================
# Streamlit Dashboard Application
# =============================================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import logging
from io import BytesIO # Needed for image buffer
from wordcloud import WordCloud # If still used for HF
import matplotlib.pyplot as plt # If still used for HF

# Import config, utilities, data processing functions
import hf_models_monitoring_test.config_utils as config_utils
import hf_models_monitoring_test.data_processing as data_processing

logger = config_utils.logger

# --- Plotting Functions (Generic, can be reused) ---
def plot_metric_bar_chart(df, metric_col, category_col, title, yaxis_title, orientation='h'):
    default_layout = go.Layout(title=title, annotations=[go.layout.Annotation(text="No data available", showarrow=False)])
    if df.empty or metric_col not in df.columns or category_col not in df.columns:
        logger.warning(f"Cannot plot '{title}': DataFrame empty or columns missing ({metric_col}, {category_col}).")
        return go.Figure(layout=default_layout)

    plot_df = df.sort_values(by=metric_col, ascending=(True if orientation == 'h' else False))
    
    if orientation == 'h':
        fig = px.bar(plot_df, y=category_col, x=metric_col, title=title,
                     labels={category_col: 'Category', metric_col: yaxis_title},
                     height=max(400, 50 + len(plot_df) * (20 if len(plot_df) < 30 else 15) ), orientation='h')
        fig.update_layout(yaxis_title=None, margin=dict(l=max(150, plot_df[category_col].astype(str).map(len).max() * 7))) # Dynamic left margin
    else: # vertical
        fig = px.bar(plot_df, x=category_col, y=metric_col, title=title,
                     labels={category_col: 'Category', metric_col: yaxis_title},
                     height=500)
        fig.update_layout(xaxis_title=None)
        
    fig.update_layout(xaxis=dict(type='linear'))
    return fig

# --- Streamlit App Setup ---
logger.info("Setting up Streamlit application...")
st.set_page_config(layout="wide", page_title="AI Societal Resilience Monitoring")

# --- Data Loading and Caching ---
@st.cache_data
def load_hf_data_for_app(csv_path):
    logger.info("(Cache Check) Attempting to load Hugging Face data for app...")
    df = data_processing.load_and_preprocess_hf_data(csv_path)
    if df is not None: logger.info("(Cache Check) HF Data loaded successfully.")
    else: logger.error("(Cache Check) Failed to load HF data.")
    return df

@st.cache_data
def calculate_hf_summaries(_df_hf, naics_csv_path):
    logger.info("(Cache Check) Attempting to calculate Hugging Face summaries...")
    if _df_hf is None or _df_hf.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    sector_data_map_app, sector_cni_map_by_id = data_processing.load_naics_keywords(naics_csv_path)
    if not sector_data_map_app: sector_data_map_app = {}
    if not sector_cni_map_by_id: sector_cni_map_by_id = {}

    sector_summary = data_processing.get_sector_summary(_df_hf, sector_data_map_app)
    keyword_summary = data_processing.get_keyword_summary(_df_hf)
    top_3_tasks_series = data_processing.get_top_tasks_per_sector(_df_hf)

    if not sector_summary.empty:
        if sector_cni_map_by_id:
            sector_summary['CNI Percentage'] = sector_summary['sector_id_full'].map(sector_cni_map_by_id).fillna(0.0)
        else:
            sector_summary['CNI Percentage'] = 0.0
        if not top_3_tasks_series.empty:
            sector_summary = pd.merge(sector_summary, top_3_tasks_series, left_on='sector_id_full', right_index=True, how='left')
            sector_summary['top_3_tasks'] = sector_summary['top_3_tasks'].fillna('N/A')
        else:
            sector_summary['top_3_tasks'] = 'N/A'
    
    # Keyword downloads for word cloud (simplified, ensure columns exist)
    keyword_sector_downloads = pd.DataFrame()
    if 'matched_sectors' in _df_hf.columns and 'matched_keywords' in _df_hf.columns and 'downloads' in _df_hf.columns:
        try:
            _df_wc = _df_hf.copy()
            _df_wc = _df_wc[_df_wc['matched_sectors'].apply(isinstance, args=(list,))]
            _df_wc = _df_wc[_df_wc['matched_keywords'].apply(isinstance, args=(list,))]
            df_exploded_sectors = _df_wc.explode('matched_sectors')
            df_exploded_keywords = df_exploded_sectors.explode('matched_keywords')
            df_exploded_keywords = df_exploded_keywords.dropna(subset=['matched_sectors', 'matched_keywords'])
            df_exploded_keywords = df_exploded_keywords[df_exploded_keywords['matched_keywords'] != '']
            if not df_exploded_keywords.empty:
                 keyword_sector_downloads = df_exploded_keywords.groupby(['matched_sectors', 'matched_keywords'])['downloads'].sum().reset_index()
                 keyword_sector_downloads = keyword_sector_downloads.rename(columns={'matched_sectors': 'sector_id_full', 'matched_keywords': 'keyword', 'downloads': 'total_keyword_sector_downloads'})
                 if sector_data_map_app:
                      keyword_sector_downloads['sector_name'] = keyword_sector_downloads['sector_id_full'].map(lambda sid: sector_data_map_app.get(sid, {}).get('name', 'N/A'))
                 else: keyword_sector_downloads['sector_name'] = 'N/A'
        except Exception as e: logger.error(f"Error calculating HF keyword downloads for wordcloud: {e}")


    logger.info("(Cache Check) Hugging Face summaries calculated.")
    return sector_summary, keyword_summary, keyword_sector_downloads, sector_data_map_app


@st.cache_data
def load_mcp_data_for_app(csv_path):
    logger.info("(Cache Check) Attempting to load MCP server data for app...")
    df = data_processing.load_and_preprocess_mcp_data(csv_path)
    if df is not None: logger.info("(Cache Check) MCP Data loaded successfully.")
    else: logger.error("(Cache Check) Failed to load MCP data.")
    return df

@st.cache_data
def calculate_mcp_summaries(_df_mcp):
    logger.info("(Cache Check) Attempting to calculate MCP summaries...")
    if _df_mcp is None or _df_mcp.empty:
        return pd.DataFrame(), pd.DataFrame()

    # For MCP, categories are 'Finance Sector Keywords', 'Threat Model Keywords' etc.
    # These are already in the 'keyword_categories' column, which is a list.
    mcp_summary_by_cat = data_processing.get_mcp_summary_by_category(_df_mcp, main_category_col='keyword_categories')
    mcp_affordance_overview = data_processing.get_mcp_affordance_overview(_df_mcp)
    
    logger.info("(Cache Check) MCP summaries calculated.")
    return mcp_summary_by_cat, mcp_affordance_overview


# --- Main App UI ---
st.title("Frontier AI: Societal Resilience Monitoring Dashboard")
st.markdown(f"""
Prototype dashboard to explore indicators related to AI exposure/integration, severity/incidents, and vulnerability/resilience.
*Data sources last updated around: {config_utils.CURRENT_DATE_STR}*
""")

# --- Data Source Selection ---
data_source_type = st.sidebar.radio(
    "Select Data Source to Analyze:",
    ("Hugging Face Models", "MCP Servers"),
    key="data_source_selector"
)
st.sidebar.markdown("---")


# --- Load and Process Data Based on Selection ---
if data_source_type == "Hugging Face Models":
    st.sidebar.info(f"Reading HF model data from: `{config_utils.OUTPUT_CSV_FILE}`")
    st.sidebar.info(f"NAICS keyword source: `{config_utils.NAICS_KEYWORDS_CSV}`")
    if not os.path.exists(config_utils.OUTPUT_CSV_FILE) or not os.path.exists(config_utils.NAICS_KEYWORDS_CSV):
        st.error("Required data files for Hugging Face analysis are missing. Please run data collection.")
        st.stop()
    
    main_hf_df = load_hf_data_for_app(config_utils.OUTPUT_CSV_FILE)
    if main_hf_df is None or main_hf_df.empty:
        st.error(f"Failed to load or process Hugging Face data. Check logs.")
        # Display logs if available
        st.stop()
    
    hf_sector_summary_df, hf_keyword_summary_df, hf_keyword_sector_downloads_df, hf_sector_data_map = calculate_hf_summaries(main_hf_df, config_utils.NAICS_KEYWORDS_CSV)

elif data_source_type == "MCP Servers":
    st.sidebar.info(f"Reading MCP server data from: `{config_utils.MCP_OUTPUT_CSV_FILE}`")
    if not os.path.exists(config_utils.MCP_OUTPUT_CSV_FILE):
        st.error(f"MCP server data file (`{config_utils.MCP_OUTPUT_CSV_FILE}`) not found. Please run MCP data collection (`run_mcp_data_collection.py`).")
        st.stop()

    main_mcp_df = load_mcp_data_for_app(config_utils.MCP_OUTPUT_CSV_FILE)
    if main_mcp_df is None or main_mcp_df.empty:
        st.error(f"Failed to load or process MCP server data. Check logs.")
        st.stop()
    
    mcp_summary_by_category_df, mcp_affordance_overview_df = calculate_mcp_summaries(main_mcp_df)

else: # Should not happen
    st.error("Invalid data source selected.")
    st.stop()


# --- Display Dashboard Based on Data Source ---

if data_source_type == "Hugging Face Models":
    st.header("🤖 Hugging Face Model Analysis")
    st.markdown("Analysis of AI models on Hugging Face Hub, categorized by potential relevance to NAICS sectors.")

    col1, col2 = st.columns(2)
    col1.metric("Total Unique Models Analyzed (HF)", len(main_hf_df) if main_hf_df is not None else 0)
    col2.metric("NAICS Sectors with Models Found (HF)", hf_sector_summary_df['sector_code'].nunique() if not hf_sector_summary_df.empty else 0)
    st.markdown("---")

    tab_hf_sector, tab_hf_keyword, tab_hf_cni, tab_hf_wordcloud, tab_hf_longlist = st.tabs([
        "📊 Sector Analysis (HF)", "🔑 Keyword Analysis (HF)", "💡 CNI Analysis (HF)",
        "☁️ Keyword Clouds (HF)", "📋 Model Longlist (HF)"
    ])

    with tab_hf_sector:
        st.subheader("Sector Summary Statistics (HF Models)")
        if not hf_sector_summary_df.empty:
            display_cols_sector = [
                'sector_code', 'sector_name', 'model_count', 'total_downloads_str', 
                'average_downloads', 'top_model_id', 'top_model_downloads_str', 'top_model_keywords_str'
            ]
            st.dataframe(hf_sector_summary_df[[c for c in display_cols_sector if c in hf_sector_summary_df.columns]].rename(columns={
                'sector_code': 'ID', 'sector_name': 'Sector', 'model_count': 'Models', 
                'total_downloads_str': 'Total Dls', 'average_downloads': 'Avg Dls',
                'top_model_id': 'Top Model', 'top_model_downloads_str': 'Top Model Dls',
                'top_model_keywords_str': 'Top Model Keywords'
            }), use_container_width=True, hide_index=True)
            
            st.plotly_chart(plot_metric_bar_chart(hf_sector_summary_df, 'model_count', 'sector_name', "Model Count per Sector (HF)", "Number of Models"), use_container_width=True)
            st.plotly_chart(plot_metric_bar_chart(hf_sector_summary_df, 'total_downloads', 'sector_name', "Total Downloads per Sector (HF)", "Total Downloads"), use_container_width=True)
        else: st.warning("No HF sector summary data available.")

    with tab_hf_keyword:
        st.subheader("Keyword Analysis (HF Models)")
        st.markdown("Most downloaded model found for each specific keyword used in NAICS mapping.")
        if not hf_keyword_summary_df.empty:
            st.dataframe(hf_keyword_summary_df, use_container_width=True, hide_index=True)
        else: st.warning("No HF keyword summary data available.")

    with tab_hf_cni:
        st.subheader("Critical National Infrastructure (CNI) Analysis (HF Models)")
        if 'CNI Percentage' in hf_sector_summary_df.columns:
            # Simplified CNI display
            cni_display_df = hf_sector_summary_df[hf_sector_summary_df['CNI Percentage'] > 0].sort_values('CNI Percentage', ascending=False)
            cni_cols = ['sector_name', 'CNI Percentage', 'model_count', 'total_downloads_str', 'top_3_tasks']
            st.dataframe(cni_display_df[[c for c in cni_cols if c in cni_display_df.columns]], use_container_width=True, hide_index=True)
        else: st.warning("CNI percentage data not available for HF models.")

    with tab_hf_wordcloud:
        st.subheader("Keyword Download Word Clouds by Sector (HF Models)")
        if hf_keyword_sector_downloads_df is None or hf_keyword_sector_downloads_df.empty:
            st.warning("No data for HF keyword word clouds.")
        else:
            # Simplified word cloud display - showing for top 5 sectors by model count
            if not hf_sector_summary_df.empty and 'sector_name' in hf_keyword_sector_downloads_df.columns:
                top_sectors_for_wc = hf_sector_summary_df.nlargest(5, 'model_count')['sector_name'].tolist()
                for sector_name in top_sectors_for_wc:
                    st.markdown(f"##### {sector_name}")
                    sector_data = hf_keyword_sector_downloads_df[hf_keyword_sector_downloads_df['sector_name'] == sector_name]
                    if not sector_data.empty and sector_data['total_keyword_sector_downloads'].sum() > 0:
                        frequencies = pd.Series(sector_data.total_keyword_sector_downloads.values, index=sector_data.keyword).loc[lambda x: x > 0].to_dict()
                        if frequencies:
                            try:
                                wc = WordCloud(width=400, height=200, background_color='white', colormap='viridis', max_words=30).generate_from_frequencies(frequencies)
                                fig, ax = plt.subplots(figsize=(6,3))
                                ax.imshow(wc, interpolation='bilinear'); ax.axis('off'); plt.tight_layout(pad=0)
                                st.pyplot(fig)
                            except Exception as e: st.error(f"Word cloud error: {e}")
                        else: st.caption("No keywords with downloads.")
                    else: st.caption("No keyword download data.")
            else: st.info("Not enough data for word clouds.")


    with tab_hf_longlist:
        st.subheader("Detailed Model Longlist (HF)")
        if main_hf_df is not None and not main_hf_df.empty:
            longlist_cols = ['modelId', 'downloads', 'likes', 'lastModified', 'pipeline_tag', 'matched_sectors', 'matched_keywords', 'tags']
            display_df = main_hf_df[[c for c in longlist_cols if c in main_hf_df.columns]].copy()
            for col in ['matched_sectors', 'matched_keywords', 'tags']:
                 if col in display_df.columns: display_df[col] = display_df[col].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
            st.dataframe(display_df, use_container_width=True)
        else: st.warning("No HF model longlist data.")


elif data_source_type == "MCP Servers":
    st.header("🔌 MCP Server Analysis (Smithery Registry)")
    st.markdown("Analysis of Model Context Protocol (MCP) servers, their capabilities, and relevance to financial sectors and threat models.")

    col1, col2, col3 = st.columns(3)
    if main_mcp_df is not None:
        col1.metric("Total Unique MCP Servers Analyzed", main_mcp_df['qualifiedName'].nunique())
    if not mcp_affordance_overview_df.empty:
        col2.metric("Servers with Execution Affordance", mcp_affordance_overview_df.loc['Servers with Finance Execution', 'Count'] if 'Servers with Finance Execution' in mcp_affordance_overview_df.index else 0)
        col3.metric("Servers with Info Affordance", mcp_affordance_overview_df.loc['Servers with Finance Information_gathering', 'Count'] if 'Servers with Finance Information_gathering' in mcp_affordance_overview_df.index else 0) # Match key from data_processing
    st.markdown("---")

    tab_mcp_overview, tab_mcp_categories, tab_mcp_affordances, tab_mcp_longlist = st.tabs([
        "📊 MCP Overview", "🗂️ MCPs by Category", "🛠️ Financial Affordances", "📋 MCP Server Longlist"
    ])

    with tab_mcp_overview:
        st.subheader("MCP Server General Overview")
        if not mcp_affordance_overview_df.empty:
            st.dataframe(mcp_affordance_overview_df, use_container_width=True)
        else:
            st.warning("No MCP affordance overview data available.")
        
        if main_mcp_df is not None and not main_mcp_df.empty:
            st.subheader("Key MCP Server Metrics")
            # Example: Top 5 servers by usage_tool_calls
            top_usage = main_mcp_df.sort_values('usage_tool_calls', ascending=False).head(5)
            st.write("Top 5 Servers by Usage Tool Calls:")
            st.dataframe(top_usage[['qualifiedName', 'displayName', 'usage_tool_calls', 'security_scan_passed']], use_container_width=True, hide_index=True)

    with tab_mcp_categories:
        st.subheader("MCP Servers by Matched Keyword Category")
        st.markdown("Categories are based on Finance Sector and Threat Model keyword lists used for searching.")
        if not mcp_summary_by_category_df.empty:
            st.dataframe(mcp_summary_by_category_df.rename(columns={
                'keyword_categories': 'Search Category',
                'mcp_server_count': 'Server Count',
                'total_usage_tool_calls': 'Total Usage',
                'avg_usage_tool_calls': 'Avg Usage',
                'servers_deployed_count': 'Deployed',
                'security_scan_passed_count': 'Scan Passed',
                'count_finance_execution': 'Exec. Tools',
                'count_finance_info': 'Info Tools',
                'count_finance_interaction': 'Interact. Tools'
            }), use_container_width=True, hide_index=True)
            
            st.plotly_chart(plot_metric_bar_chart(mcp_summary_by_category_df, 
                                                  'mcp_server_count', 'keyword_categories', 
                                                  "MCP Server Count by Search Category", "Number of Servers"), 
                            use_container_width=True)
        else:
            st.warning("No MCP summary by category data available.")

    with tab_mcp_affordances:
        st.subheader("MCP Servers with Specific Financial Affordances")
        if main_mcp_df is not None and not main_mcp_df.empty:
            affordance_cols_map = {
                "Execution": "has_finance_execution",
                "Information Gathering": "has_finance_info",
                "Agent Interaction": "has_finance_interaction"
            }
            for display_name, col_name in affordance_cols_map.items():
                if col_name in main_mcp_df.columns:
                    st.markdown(f"#### Servers with {display_name} Affordance")
                    filtered_df = main_mcp_df[main_mcp_df[col_name] == True].drop_duplicates(subset=['qualifiedName'])
                    if not filtered_df.empty:
                        tool_list_col = f"finance_{col_name.split('_')[-1]}_tools_list" # e.g. finance_execution_tools_list
                        display_cols = ['qualifiedName', 'displayName', 'usage_tool_calls', tool_list_col, 'security_scan_passed']
                        # Ensure tool_list_col exists before trying to access it
                        if tool_list_col not in filtered_df.columns:
                            display_cols.remove(tool_list_col)
                            
                        st.dataframe(filtered_df[[c for c in display_cols if c in filtered_df.columns]].rename(columns={
                            tool_list_col: f'{display_name} Tools (Sample)'
                        }).assign(**{
                             f'{display_name} Tools (Sample)': lambda dfx: dfx[tool_list_col].apply(lambda x: x[:3] if isinstance(x, list) else x) if tool_list_col in dfx else 'N/A'
                        }), use_container_width=True, hide_index=True)
                    else:
                        st.caption(f"No servers found with {display_name} affordance.")
                else:
                    st.warning(f"Affordance column {col_name} not found in MCP data.")
        else:
            st.warning("No MCP data for affordance analysis.")

    with tab_mcp_longlist:
        st.subheader("Detailed MCP Server Longlist")
        if main_mcp_df is not None and not main_mcp_df.empty:
            # Make a copy for display to avoid changing cached data
            display_mcp_df = main_mcp_df.copy()
            
            # Columns to display, ensure they exist
            mcp_longlist_cols = [
                'qualifiedName', 'displayName', 'description', 'usage_tool_calls', 'isDeployed', 
                'security_scan_passed', 'finance_execution_tools_list', 'finance_info_tools_list',
                'finance_interaction_tools_list', 'all_tools_details', 'matched_keywords', 
                'keyword_categories', 'homepage', 'createdAt_list', 'deploymentUrl'
            ]
            display_mcp_df = display_mcp_df[[col for col in mcp_longlist_cols if col in display_mcp_df.columns]]

            # Simplify display of list/JSON cols for readability in dataframe
            for col in ['finance_execution_tools_list', 'finance_info_tools_list', 'finance_interaction_tools_list', 
                        'all_tools_details', 'matched_keywords', 'keyword_categories']:
                if col in display_mcp_df.columns:
                    display_mcp_df[col] = display_mcp_df[col].apply(lambda x: str(x) if isinstance(x, (list, dict)) else x)
            
            st.dataframe(display_mcp_df, height=600, use_container_width=True)
        else:
            st.warning("No MCP server longlist data available.")

st.sidebar.markdown("---")
st.sidebar.caption(f"App Version: 0.2.0 (MCP Integration)")
st.sidebar.caption(f"Current Date: {config_utils.CURRENT_DATE_STR}")

# --- Optional: Display Logs ---
# with st.expander("Show Logs"):
# st.text_area("Log Output", config_utils.log_stream.getvalue(), height=200)
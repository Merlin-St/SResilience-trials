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
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Import config, utilities, data processing functions
import config_utils
import data_processing

logger = config_utils.logger # Use the logger from config_utils

# --- Plotting Functions ---
def plot_sector_metric(sector_summary, metric_col, sector_name_col, title, yaxis_title):
    """Generates a Plotly horizontal bar chart for a given metric per sector, using sector name for labels."""
    default_layout = go.Layout(title=title, annotations=[go.layout.Annotation(text="No data available", showarrow=False)])
    if sector_summary.empty:
        logger.warning(f"Cannot plot '{title}': Sector summary empty.")
        return go.Figure(layout=default_layout)
    if metric_col not in sector_summary.columns:
        logger.error(f"Metric column '{metric_col}' not found for plotting '{title}'.")
        return go.Figure(layout=go.Layout(title=title, annotations=[go.layout.Annotation(text=f"Metric '{metric_col}' not found", showarrow=False)]))
    if sector_name_col not in sector_summary.columns:
        logger.error(f"Sector name column '{sector_name_col}' not found for plotting '{title}'.")
        return go.Figure(layout=go.Layout(title=title, annotations=[go.layout.Annotation(text=f"Sector name column '{sector_name_col}' not found", showarrow=False)]))

    # Sort by the metric for better visualization
    plot_df = sector_summary.sort_values(by=metric_col, ascending=True)

    # Use horizontal bars
    fig = px.bar(plot_df,
                 y=sector_name_col,
                 x=metric_col,
                 title=title,
                 labels={sector_name_col: 'Sector', metric_col: yaxis_title},
                 height=500 + len(plot_df) * 10,
                 orientation='h')

    fig.update_layout(
        xaxis=dict(type='linear'),
        margin=dict(l=250, r=20, t=50, b=50),
        yaxis_title=None
    )
    return fig

# --- Streamlit App Setup ---
logger.info("Setting up Streamlit application...")
st.set_page_config(layout="wide", page_title="HF NAICS Sector Analysis")

# Define functions to load/process data WITH caching for Streamlit app performance
@st.cache_data # Cache the result of loading and preprocessing
def load_data_for_app(csv_path):
    """Loads and preprocesses data specifically for the Streamlit app."""
    logger.info("(Cache Check) Attempting to load data for app...")
    # Use the function from data_processing module
    df = data_processing.load_and_preprocess_data(csv_path)
    if df is not None: logger.info("(Cache Check) Data loaded successfully for app.")
    else: logger.error("(Cache Check) Failed to load data for app.")
    return df

@st.cache_data # Cache analysis results derived from the main df
def calculate_summaries(_df, naics_csv_path):
    """
    Calculates summaries based on the loaded dataframe, using cached results.
    Includes CNI merge, top tasks, AND keyword-download aggregation for wordclouds.
    Args:
        _df (pd.DataFrame): The main dataframe loaded by load_data_for_app.
        naics_csv_path (str): Path to the NAICS keywords CSV file.
    Returns:
        tuple: Contains sector_summary (pd.DataFrame), keyword_summary (pd.DataFrame),
               keyword_sector_downloads (pd.DataFrame).
    """
    logger.info("(Cache Check) Attempting to calculate summaries...")
    if _df is None or _df.empty:
        logger.warning("(Cache Check) Input DataFrame empty, returning empty summaries.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # --- Reload NAICS data to get full map and CNI mapping ---
    logger.info("(Cache Check) Loading NAICS data within cached function...")
    # Use function from data_processing
    sector_data_map_app, sector_cni_map_by_id = data_processing.load_naics_keywords(naics_csv_path)
    if not sector_data_map_app:
        logger.warning("Could not load NAICS data map within calculate_summaries. Sector details will be missing.")
        sector_data_map_app = {}
    if not sector_cni_map_by_id:
        logger.warning("Could not load CNI map within calculate_summaries. CNI analysis will be incomplete.")
        sector_cni_map_by_id = {}

    # Calculate sector summary (requires the sector_data_map)
    # Use function from data_processing
    sector_summary = data_processing.get_sector_summary(_df, sector_data_map_app)

    # Calculate keyword summary
    # Use function from data_processing
    keyword_summary = data_processing.get_keyword_summary(_df)

    # --- Calculate Top 3 Tasks ---
    # Use function from data_processing
    top_3_tasks_series = data_processing.get_top_tasks_per_sector(_df)

    # --- Merge CNI info and Top Tasks into sector_summary ---
    if not sector_summary.empty:
        if sector_cni_map_by_id:
            sector_summary['CNI Percentage'] = sector_summary['sector_id_full'].map(sector_cni_map_by_id).fillna(0.0)
            logger.info("CNI Percentage mapped from CSV data.")
        else:
            sector_summary['CNI Percentage'] = 0.0
            logger.warning("CNI Percentage set to 0.0 due to missing map.")

        if not top_3_tasks_series.empty:
            sector_summary = pd.merge(sector_summary, top_3_tasks_series, left_on='sector_id_full', right_index=True, how='left')
            sector_summary['top_3_tasks'] = sector_summary['top_3_tasks'].fillna('N/A')
        else:
            sector_summary['top_3_tasks'] = 'N/A'
    else:
        # Ensure columns exist even if summary is empty
        if 'CNI Percentage' not in sector_summary.columns: sector_summary['CNI Percentage'] = pd.Series(dtype=float)
        if 'top_3_tasks' not in sector_summary.columns: sector_summary['top_3_tasks'] = pd.Series(dtype=str)
        if 'sector_id_full' not in sector_summary.columns: sector_summary['sector_id_full'] = pd.Series(dtype=str)

    # --- Calculate Keyword Downloads per Sector (for Word Clouds) ---
    logger.info("(Cache Check) Calculating keyword downloads per sector...")
    keyword_sector_downloads = pd.DataFrame() # Initialize empty
    try:
        if 'matched_sectors' in _df.columns and 'matched_keywords' in _df.columns and 'downloads' in _df.columns:
             _df_wc = _df.copy()
             # Ensure lists are parsed (might use config_utils.parse_list_string if needed, but load_and_preprocess should handle it)
             _df_wc = _df_wc[_df_wc['matched_sectors'].apply(isinstance, args=(list,))]
             _df_wc = _df_wc[_df_wc['matched_keywords'].apply(isinstance, args=(list,))]

             df_exploded_sectors = _df_wc.explode('matched_sectors')
             df_exploded_keywords = df_exploded_sectors.explode('matched_keywords')

             df_exploded_keywords = df_exploded_keywords.dropna(subset=['matched_sectors', 'matched_keywords'])
             df_exploded_keywords = df_exploded_keywords[df_exploded_keywords['matched_keywords'] != '']

             if not df_exploded_keywords.empty:
                 keyword_sector_downloads = df_exploded_keywords.groupby(['matched_sectors', 'matched_keywords'])['downloads'].sum().reset_index()
                 keyword_sector_downloads = keyword_sector_downloads.rename(columns={
                     'matched_sectors': 'sector_id_full',
                     'matched_keywords': 'keyword',
                     'downloads': 'total_keyword_sector_downloads'
                 })
                 if sector_data_map_app:
                      keyword_sector_downloads['sector_name'] = keyword_sector_downloads['sector_id_full'].map(lambda sid: sector_data_map_app.get(sid, {}).get('name', 'N/A'))
                 else:
                      keyword_sector_downloads['sector_name'] = 'N/A'
                 logger.info(f"(Cache Check) Keyword downloads per sector calculated. Found {len(keyword_sector_downloads)} keyword-sector pairs.")
             else:
                 logger.warning("(Cache Check) DataFrame empty after exploding keywords/sectors for word cloud data.")
        else:
            logger.warning("(Cache Check) Missing columns required for word cloud data ('matched_sectors', 'matched_keywords', 'downloads').")
    except Exception as e:
        logger.error(f"(Cache Check) Error calculating keyword downloads per sector: {e}", exc_info=True)
        keyword_sector_downloads = pd.DataFrame()

    logger.info("(Cache Check) Summaries calculated successfully.")
    return sector_summary, keyword_summary, keyword_sector_downloads

# --- Streamlit App UI ---

st.title("Hugging Face AI Model Analysis by NAICS Sector")
st.markdown(f"""
Analysis of AI models on Hugging Face Hub, categorized by potential relevance
to NAICS sectors using associated keywords found in model tags. Note: Quick 1-day trial to understand tractability of dashboards. 
Needs update to focus on FrontierAI only & temporal trend.
*Data collected around: {config_utils.CURRENT_DATE_STR}*
""")
st.info(f"Reading model data from: `{config_utils.OUTPUT_CSV_FILE}` | Keyword/CNI source: `{config_utils.NAICS_KEYWORDS_CSV}`")

# --- Load data ---
# Check if essential files exist before attempting to load
if not os.path.exists(config_utils.OUTPUT_CSV_FILE):
    st.error(f"**Model data file not found:** `{config_utils.OUTPUT_CSV_FILE}`")
    st.warning("Please ensure the data collection script (`run_data_collection.py`) has been executed successfully to generate the required CSV file, then refresh this page.")
    st.stop()
if not os.path.exists(config_utils.NAICS_KEYWORDS_CSV):
    st.error(f"**NAICS keyword/CNI file not found:** `{config_utils.NAICS_KEYWORDS_CSV}`")
    st.warning("Please ensure the NAICS keyword and CNI percentage file is available in the correct location.")
    st.stop()

# Load the main model data using the cached function
main_df_app = load_data_for_app(config_utils.OUTPUT_CSV_FILE)

# Handle potential loading failure
if main_df_app is None or main_df_app.empty:
    st.error(f"Failed to load or process data from `{config_utils.OUTPUT_CSV_FILE}`. Check logs (`{config_utils.LOG_FILE}`).")
    st.subheader("Logs")
    log_content = "Could not read log file."
    try:
        with open(config_utils.LOG_FILE, 'r') as f:
            log_content = f.read()
    except Exception as log_e:
        log_content = f"Error reading log file {config_utils.LOG_FILE}: {log_e}\n\nCaptured logs during run:\n{config_utils.log_stream.getvalue()}" # Use log_stream from config
    st.text_area("Log Output", log_content, height=300)
    st.stop()

# --- Calculate Summaries ---
# Use the cached function, passing the NAICS file path from config
sector_summary_df_app, keyword_summary_df_app, keyword_sector_downloads_df = calculate_summaries(main_df_app, config_utils.NAICS_KEYWORDS_CSV)

# --- Display Dashboard Sections ---
st.markdown("---")
st.subheader("Dashboard Overview")

# Display overview metrics
col1, col2 = st.columns(2)
col1.metric("Total Unique Models Analyzed", len(main_df_app))
col2.metric("NAICS Sectors with Models Found", sector_summary_df_app['sector_code'].nunique() if not sector_summary_df_app.empty else 0)

st.markdown("---")

# Define the tabs
tab_sector, tab_keyword, tab_cni, tab_wordcloud, tab_longlist, tab_interp = st.tabs([
    "📊 Sector Analysis",
    "🔑 Keyword Analysis",
    "💡 CNI Analysis",
    "☁️ Keyword Clouds",
    "📋 Model Longlist",
    "📜 Limitations & Next Steps"
])

# --- Tab 1: Sector Analysis ---
with tab_sector:
    # (Content remains the same as original, using sector_summary_df_app and plot_sector_metric)
    st.header("📊 Sector Analysis")
    st.markdown("Overview of models associated with each identified NAICS sector based on keyword matching in tags.")

    st.subheader("Sector Summary Statistics")
    if not sector_summary_df_app.empty:
        display_cols_sector = [
            'sector_code', 'sector_name', 'model_count', 'total_downloads_str', 'average_downloads',
            'top_model_id', 'top_model_downloads_str', 'top_model_keywords_str'
        ]
        display_cols_sector = [col for col in display_cols_sector if col in sector_summary_df_app.columns]
        rename_map_sector = {
            'sector_code': 'Sector ID', 'sector_name': 'Sector Name',
            'model_count': 'Model Count', 'total_downloads_str': 'Total Downloads',
            'average_downloads': 'Avg Downloads', 'top_model_id': 'Most Downloaded Model',
            'top_model_downloads_str': 'Top Model Downloads',
            'top_model_keywords_str': 'Keywords Matched by Top Model'
        }
        rename_map_sector_filtered = {k: v for k, v in rename_map_sector.items() if k in display_cols_sector}
        sector_display_df = sector_summary_df_app[display_cols_sector].rename(columns=rename_map_sector_filtered)
        st.dataframe(sector_display_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No sector summary data available.")

    st.subheader("Visualizations")
    if not sector_summary_df_app.empty:
        fig_sector_count = plot_sector_metric(sector_summary_df_app, metric_col='model_count', sector_name_col='sector_name', title="Model count per sector", yaxis_title="Number of Unique Models")
        st.plotly_chart(fig_sector_count, use_container_width=True)
        fig_sector_downloads = plot_sector_metric(sector_summary_df_app, metric_col='total_downloads', sector_name_col='sector_name', title="Model downloads per sector", yaxis_title="Total Downloads")
        st.plotly_chart(fig_sector_downloads, use_container_width=True)
    else:
        st.warning("No data available for sector visualizations.")


# --- Tab 2: Keyword Analysis ---
with tab_keyword:
    # (Content remains the same as original, using keyword_summary_df_app)
    st.header("🔑 Keyword Analysis")
    st.markdown("""
    Models are associated with sectors by searching for specific keywords within the model tags on Hugging Face Hub (using the `filter` parameter in the API).
    Keywords are extracted from the NAICS 4-digit sector classifications (e.g., Keywords of sector 21 'Mining, Quarrying, and Oil and Gas Extraction' come from all the subsector definitions, e.g., 2111, "'Oil' 'and' 'gas' 'extraction'" to 2131 '...'), excluding common stopwords (e.g., 'and') and words used differently in ML than in industry (e.g., 'extraction').
    The table below shows the most downloaded model found for each specific keyword used.
    """)
    if not keyword_summary_df_app.empty:
        st.dataframe(keyword_summary_df_app, use_container_width=True, hide_index=True)
    else:
        st.warning("No keyword summary data available.")


# --- Tab 3: CNI Analysis ---
with tab_cni:
    # (Content remains the same as original, using sector_summary_df_app and NAICS_KEYWORDS_CSV from config)
    st.header("💡 Critical National Infrastructure (CNI) Analysis")
    st.markdown(f"""
    This section analyzes models associated with sectors based on their potential relevance to CNI.
    The CNI percentage ('CNI %') represents the average share of CNI-related subsectors within a given NAICS 2-digit sector, based on the classifications in `{config_utils.NAICS_KEYWORDS_CSV}`.
    *Disclaimer: This CNI classification is based on an automated assessment (e.g., by an LLM) of whether each 4-digit NAICS subsector description aligns with official US CNI definitions, due to the lack of a direct official mapping. It is intended as an indicator and requires expert validation.*
    The analysis compares sectors deemed predominantly CNI-related (>50% CNI) versus others.
    """)

    if 'CNI Percentage' in sector_summary_df_app.columns:
        sector_summary_df_app['CNI Percentage'] = pd.to_numeric(sector_summary_df_app['CNI Percentage'], errors='coerce').fillna(0.0)
        predominantly_cni_threshold = 50.0
        cni_predom_df = sector_summary_df_app[sector_summary_df_app['CNI Percentage'] > predominantly_cni_threshold].copy()
        non_predom_cni_df = sector_summary_df_app[sector_summary_df_app['CNI Percentage'] <= predominantly_cni_threshold].copy()

        st.subheader("CNI Relevant Sectors Summary (> 0% CNI)")
        cni_relevant_df = sector_summary_df_app[sector_summary_df_app['CNI Percentage'] > 0].copy()
        if not cni_relevant_df.empty:
             cni_display_cols = ['sector_code', 'sector_name', 'CNI Percentage', 'model_count', 'total_downloads_str', 'top_3_tasks']
             cni_display_cols = [col for col in cni_display_cols if col in cni_relevant_df.columns]
             cni_rename_map = {'sector_code': 'Sector ID', 'sector_name': 'Sector Name', 'CNI Percentage': 'CNI %',
                               'model_count': 'Model Count', 'total_downloads_str': 'Total Downloads', 'top_3_tasks': 'Top 3 Model Tasks'}
             cni_rename_map_filtered = {k: v for k, v in cni_rename_map.items() if k in cni_display_cols}
             cni_display_df = cni_relevant_df[cni_display_cols].rename(columns=cni_rename_map_filtered)
             st.dataframe(
                 cni_display_df.style.format({'CNI %': '{:.1f}%'}),
                 use_container_width=True,
                 hide_index=True
             )
        else:
            st.info("No sectors found with CNI Percentage > 0%.")

        st.subheader(f"Comparative Stats: Predominantly CNI Sectors (> {predominantly_cni_threshold}%) vs. Others")
        total_predom_cni_models = cni_predom_df['model_count'].sum()
        total_non_predom_cni_models = non_predom_cni_df['model_count'].sum()
        total_predom_cni_downloads = cni_predom_df['total_downloads'].sum()
        total_non_predom_cni_downloads = non_predom_cni_df['total_downloads'].sum()
        avg_downloads_predom_cni = cni_predom_df['total_downloads'].sum() / cni_predom_df['model_count'].sum() if cni_predom_df['model_count'].sum() > 0 else 0
        avg_downloads_non_predom_cni = non_predom_cni_df['total_downloads'].sum() / non_predom_cni_df['model_count'].sum() if non_predom_cni_df['model_count'].sum() > 0 else 0

        col1_cni, col2_cni = st.columns(2)
        with col1_cni:
            st.metric(f"Models in Predominantly CNI Sectors (> {predominantly_cni_threshold}%)", f"{total_predom_cni_models:,}")
            st.metric(f"Downloads in Predominantly CNI Sectors", f"{total_predom_cni_downloads:,.0f}")
            st.metric(f"Avg Downloads/Model (Predom. CNI)", f"{avg_downloads_predom_cni:,.0f}")
        with col2_cni:
            st.metric(f"Models in Other Sectors (<= {predominantly_cni_threshold}%)", f"{total_non_predom_cni_models:,}")
            st.metric(f"Downloads in Other Sectors", f"{total_non_predom_cni_downloads:,.0f}")
            st.metric(f"Avg Downloads/Model (Other)", f"{avg_downloads_non_predom_cni:,.0f}")

        total_models_overall = total_predom_cni_models + total_non_predom_cni_models
        if total_models_overall > 0:
             predom_cni_model_proportion = total_predom_cni_models / total_models_overall
             st.markdown(f"**Proportion:** Predominantly CNI sectors (> {predominantly_cni_threshold}%) account for approximately **{predom_cni_model_proportion:.1%}** of the unique models found across these sector definitions (note: models can appear in multiple sectors).")
        else:
             st.markdown("No models found to calculate proportion.")

        st.markdown("""*Note: This analysis is based on keyword matching and the provided CNI classifications. See the Limitations tab.*""")
    else:
        st.error("CNI Percentage column not found in analysis results. Cannot perform CNI analysis.")


# --- Tab 4: Keyword Clouds ---
with tab_wordcloud:
    # (Content remains the same as original, using keyword_sector_downloads_df)
    st.header("☁️ Keyword Download Word Clouds by Sector")
    st.markdown("""
    Word clouds visualizing the cumulative downloads of models associated with specific keywords within each NAICS sector.
    The size of each keyword is proportional to the total downloads of all models found matching that keyword *within that sector*.
    This helps identify which specific terms (and potentially underlying concepts) attract the most attention (downloads) in the context of each sector's model landscape on Hugging Face.
    """)

    if keyword_sector_downloads_df is None or keyword_sector_downloads_df.empty:
        st.warning("No data available for generating word clouds. Please ensure the data processing step was successful and produced keyword download data.")
    else:
        if 'sector_name' in keyword_sector_downloads_df.columns:
            available_sectors = sorted(keyword_sector_downloads_df['sector_name'].unique())
        else:
            available_sectors = []
            st.error("Internal Error: 'sector_name' column missing in keyword download data.")

        if not available_sectors:
            st.warning("No sectors with keyword download data found.")
        else:
            NUM_COLS = 4
            for i in range(0, len(available_sectors), NUM_COLS):
                cols = st.columns(NUM_COLS)
                for j in range(NUM_COLS):
                    sector_idx = i + j
                    if sector_idx < len(available_sectors):
                        sector_name = available_sectors[sector_idx]
                        with cols[j]:
                            st.markdown(f"##### {sector_name}")
                            sector_data = keyword_sector_downloads_df[keyword_sector_downloads_df['sector_name'] == sector_name]
                            if sector_data.empty or sector_data['total_keyword_sector_downloads'].sum() <= 0:
                                st.info("No keyword download data (with positive downloads) found for this sector.")
                                continue
                            frequencies = pd.Series(
                                sector_data.total_keyword_sector_downloads.values,
                                index=sector_data.keyword
                            ).loc[lambda x: x > 0].to_dict()
                            if not frequencies:
                                st.info("No keywords with positive downloads found for this sector.")
                                continue
                            fig = None
                            try:
                                wordcloud_gen = WordCloud(width=400, height=200,
                                                        background_color='white', colormap='viridis',
                                                        max_words=50, prefer_horizontal=0.9,
                                                        ).generate_from_frequencies(frequencies)
                                fig, ax = plt.subplots(figsize=(6, 3))
                                ax.imshow(wordcloud_gen, interpolation='bilinear')
                                ax.axis('off')
                                plt.tight_layout(pad=0)
                                buf = BytesIO()
                                fig.savefig(buf, format="png", bbox_inches='tight')
                                buf.seek(0)
                                st.image(buf, use_container_width=True)
                            except ValueError as ve:
                                st.error(f"Could not generate word cloud: {ve}")
                            except Exception as e:
                                st.error(f"An error occurred while generating the word cloud: {e}")
                                logger.error(f"Word cloud generation error for {sector_name}", exc_info=True)
                            finally:
                                if fig is not None:
                                    plt.close(fig)

# --- Tab 5: Model Longlist ---
with tab_longlist:
    # (Content remains the same as original, using main_df_app)
    st.header("📋 Detailed Model Longlist")
    st.markdown("""
    This table provides a detailed view of every unique model identified across all sectors and keywords during the data collection phase.
    Each row represents one unique model from Hugging Face Hub. Note that a single model might be associated with multiple sectors or keywords (shown as comma-separated lists).
    Use the column headers to sort the table interactively (e.g., by Downloads, Likes, Last Modified).
    """)
    if main_df_app is None or main_df_app.empty:
        st.warning("Model data is not available.")
    else:
        longlist_cols = [
            'modelId', 'downloads', 'likes', 'lastModified', 'pipeline_tag',
            'matched_sectors', 'matched_keywords', 'library_name', 'tags'
            # 'parameter_count', # Add later
            # 'is_potentially_frontier', # Add later
        ]
        longlist_cols_exist = [col for col in longlist_cols if col in main_df_app.columns]
        longlist_df_display = main_df_app[longlist_cols_exist].copy()
        for col in ['matched_sectors', 'matched_keywords', 'tags']:
            if col in longlist_df_display.columns:
                 longlist_df_display[col] = longlist_df_display[col].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
        st.dataframe(longlist_df_display, use_container_width=True)
        st.info(f"Total unique models listed: {len(longlist_df_display)}")


# --- Tab 6: Limitations & Next Steps ---
with tab_interp:
    # (Content remains the same as original, using NAICS_KEYWORDS_CSV from config)
    st.header("📜 Limitations & Next Steps")
    st.warning("CRITICAL Limitations & Considerations:")
    st.markdown(f"""
    * **Proxy Data:** Reflects model **availability** and **developer interest** on Hugging Face Hub via keyword matching, **not real-world adoption**.
    * **Not Real-World Adoption:** Does **NOT** measure deployment, economic impact, or true sector 'exposure'.
    * **Model Scope:** Excludes proprietary models. Effectiveness/quality not assessed. CNI classification depends on `{config_utils.NAICS_KEYWORDS_CSV}` methodology.
    * **Download Skew:** Downloads influenced by factors beyond sector use (tutorials, hype, etc.). Foundational models dominate.
    * **Tag/Keyword Dependency:** Relies on user tagging and NAICS keyword relevance/completeness. Keyword matching ≠ primary sector use. Keyword extraction process matters.
    * **Top Model/Task Representation:** 'Most Downloaded' or 'Top Tasks' often reflect general-purpose models due to high download counts.
    """)
    st.info("Next Steps:")
    st.markdown("""
    * Track changes over time (periodic runs).
    * Extend keyword search to other platforms (GitHub, AI agent indices).
    * Extend to task-focused views using other sources (AI Economic Index, Job listings -> ONET/NAICS, Agent framework tools -> ONET/NAICS).
    """)

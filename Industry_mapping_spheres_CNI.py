# %%
import nltk
from nltk.corpus import stopwords

import os
import pandas as pd
import logging
from tqdm import tqdm
import requests
from io import BytesIO
from anthropic import Anthropic
from openai import OpenAI


# Import the function from your module (ensure model_completions.py is accessible)
# from model_completions import process_df_prompts # Keep commented if not running Step 5a

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_naics_data():
    """Load NAICS data from GitHub repository file"""
    logger.info("Loading NAICS data from GitHub file...")
    # Assuming the script is run from the repository root
    # Ensure this path is correct relative to where you run the script
    file_path = "2-6 digit_2022_Codes.xlsx"
    if not os.path.exists(file_path):
        logger.error(f"NAICS file not found at {file_path}. Please ensure the file exists.")
        # Attempt to download if not found (example, adjust URL/logic as needed)
        # url = "URL_TO_YOUR_NAICS_FILE_HERE" # Replace with actual URL if needed
        # logger.info(f"Attempting to download from {url}...")
        # try:
        #     response = requests.get(url)
        #     response.raise_for_status() # Raise an exception for bad status codes
        #     df = pd.read_excel(BytesIO(response.content))
        # except requests.exceptions.RequestException as e:
        #     logger.error(f"Failed to download NAICS data: {e}")
        #     raise FileNotFoundError(f"NAICS file not found at {file_path} and download failed.")
        raise FileNotFoundError(f"NAICS file not found at {file_path}. Please place it in the correct directory.")
    else:
         df = pd.read_excel(file_path)

    df = df.iloc[:, :-2]  # Keep this if you still need to drop the last two columns

    # Rename columns to standardized names right after loading
    column_mapping = {
        'Seq. No.': 'sequence_number',
        '2022 NAICS US   Code': 'naics_code',
        '2022 NAICS US Title': 'naics_title'
    }

    df.rename(columns=column_mapping, inplace=True)
    logger.info(f"Successfully loaded NAICS data with {len(df)} entries")
    return df

def prepare_naics_for_classification(df):
    """Prepare the NAICS dataframe for classification"""
    # Create a working copy
    working_df = df.copy()

    # Handle column names based on the file format
    # Assuming the columns are 'naics_code' and 'naics_title' after renaming
    if 'naics_code' not in working_df.columns or 'naics_title' not in working_df.columns:
        logger.error("Required columns 'naics_code' or 'naics_title' not found after loading.")
        raise ValueError("DataFrame missing expected columns.")

    # Ensure naics_code is string for reliable length checks and slicing
    working_df['naics_code'] = working_df['naics_code'].astype(str)

    # Extract 2-digit sector code
    working_df['sector_code'] = working_df['naics_code'].str.slice(0, 2)

    # Filter to 4-digit NAICS codes for analysis (subsector level)
    # Use .str.len() on the string column
    four_digit_df = working_df[working_df['naics_code'].str.len() == 4].copy()

    # Create sector name mapping from 2-digit codes
    sector_mapping = working_df[working_df['naics_code'].str.len() == 2]
    sector_mapping = dict(zip(sector_mapping['naics_code'], sector_mapping['naics_title']))

    # Add sector name
    four_digit_df['sector_name'] = four_digit_df['sector_code'].map(sector_mapping)

    # Check for any 4-digit codes where the sector name couldn't be mapped (shouldn't happen with standard NAICS)
    if four_digit_df['sector_name'].isnull().any():
        logger.warning("Some 4-digit NAICS codes could not be mapped to a 2-digit sector name.")

    # Create prompts for classification
    four_digit_df['prompt'] = four_digit_df.apply(
        lambda row: f"NAICS Code: {row['naics_code']} - {row['naics_title']}\n\nIs this a Critical Infrastructure sector?",
        axis=1
    )

    return four_digit_df

def create_system_prompt():
    """Create the system prompt for the classification task"""
    return """You are an expert in Critical National Infrastructure (CNI) classification.

Your task is to evaluate if a given NAICS subsector should be classified as Critical Infrastructure according to the 16 official Critical Infrastructure sectors defined by CISA:

1. Chemical Sector: Basic chemicals, specialty chemicals, agricultural chemicals, pharmaceuticals
2. Commercial Facilities Sector: Shopping centers, sports venues, performing arts centers, lodging
3. Communications Sector: Satellite, wireless, wireline, cable, broadcasting infrastructure
4. Critical Manufacturing Sector: Primary metals, machinery, electrical equipment, transportation equipment
5. Dams Sector: Dam projects, navigation locks, levees, hurricane barriers
6. Defense Industrial Base Sector: Military and industrial supply chains
7. Emergency Services Sector: Law enforcement, fire, emergency medical services, public works
8. Energy Sector: Electricity, oil, natural gas systems
9. Financial Services Sector: Banking institutions, securities, insurance companies
10. Food and Agriculture Sector: Farms, restaurants, food manufacturing, distribution
11. Government Facilities Sector: Government buildings, national monuments/icons, election infrastructure
12. Healthcare and Public Health Sector: Hospitals, health departments, pharmaceutical companies
13. Information Technology Sector: Hardware, software, IT systems, services
14. Nuclear Reactors, Materials, and Waste Sector: Nuclear power plants, materials, waste
15. Transportation Systems Sector: Aviation, maritime, rail, mass transit, highway infrastructure
16. Water and Wastewater Systems Sector: Drinking water, wastewater management

Analyze the provided NAICS subsector carefully. Consider its direct relationship to any of the 16 sectors above. If the subsector provides essential services, products, or infrastructure that would be part of a critical infrastructure sector, classify it as CNI.

Respond ONLY in one of these two exact formats:
- "Yes, critical national infrastructure. Mainly in [insert ONE most applicable CNI sector name]."
- "No, not critical national infrastructure."

Do not include any other text, explanations, or reasoning in your response.
"""

def calculate_cni_percentages(classified_df):
    """Calculate the percentage of CNI classifications for each 2-digit NAICS sector"""


    # Group by 2-digit sector and calculate percentages
    sector_stats = classified_df.groupby(['sector_code', 'sector_name']).agg(
        total_subsectors=('naics_code', 'count'),
        cni_subsectors=('is_cni', 'sum')
    ).reset_index()

    # Calculate percentage
    sector_stats['cni_percentage'] = (sector_stats['cni_subsectors'] / sector_stats['total_subsectors'] * 100).round(2)

    return sector_stats.sort_values('sector_code')

def interactive_display():
    """Launch an interactive Streamlit app to display the results."""
    try:
        import streamlit as st
    except ImportError:
        logger.error("Streamlit is not installed. Cannot launch interactive display.")
        logger.info("To install Streamlit, run: pip install streamlit")
        return

    st.title("NAICS CNI Classification Results")

    # Load the results
    results_file = 'naics_cni_percentages.csv'
    if os.path.exists(results_file):
        results = pd.read_csv(results_file)
        st.write("### Critical Infrastructure Percentages by NAICS Sector")
        st.dataframe(results)

        # Create a bar chart
        st.write("### Bar Chart of CNI Percentages")
        # Ensure sector_code is string for categorical axis if needed
        results['sector_code'] = results['sector_code'].astype(str)
        st.bar_chart(results.set_index('sector_code')['cni_percentage'])
    else:
        st.error(f"Results file '{results_file}' not found. Please run the classification and analysis first.")

def visualize_results(results_df):
    """Visualize the CNI percentage results with matplotlib"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("Matplotlib is not installed. Cannot create visualization.")
        logger.info("To install Matplotlib, run: pip install matplotlib")
        return None # Return None if plotting cannot be done

    plt.figure(figsize=(12, 8))
    # Ensure sector_code is treated as a category (string) for plotting if it's numeric
    results_df['sector_code'] = results_df['sector_code'].astype(str)
    bars = plt.bar(results_df['sector_code'], results_df['cni_percentage'], color='darkblue')

    # Add data labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')

    plt.title('Percentage of NAICS Subsectors Classified as Critical Infrastructure by Sector')
    plt.xlabel('NAICS Sector Code')
    plt.ylabel('Percentage of Subsectors Classified as CNI (%)')
    plt.xticks(rotation=45, ha='right') # Adjust rotation for better label readability
    plt.ylim(0, 105)  # Allow space for labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout() # Adjust layout to prevent labels overlapping

    return plt # Return the plot object
# %%
def extract_sector_keywords(results_df, top_n=10):
    """
    Extract the most common and relevant keywords from NAICS titles for each sector.

    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing NAICS classification data with 'sector_code', 'sector_name', and 'naics_title'
    top_n : int, default=10
        Number of top keywords to extract for each sector

    Returns:
    --------
    keywords_df : pandas.DataFrame
        DataFrame with sector_code, sector_name, and keywords columns
    """
    import pandas as pd
    from collections import Counter
    import re
    from nltk.corpus import stopwords
    import nltk

    # Ensure NLTK stopwords are downloaded
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        logger.info("NLTK stopwords not found. Downloading...")
        nltk.download('stopwords')
        logger.info("NLTK stopwords downloaded.")

    # Get English stopwords
    stop_words = set(stopwords.words('english'))

    domain_stopwords = {
        # Original list
        'other', 'related', 'products', 'services',
        'except', 'including', 'and', 'or', 'of', 'for', 'the',
        'etc', 'various', 'miscellaneous', 'general', 'specialized',
        'all', 'equipment', 'supplies', 'activities', 'support',
        'similar', 'administration', 'management', 'operation', 'operations',
        'offices', 'establishments', 'primarily', 'engaged', 'providing',
        'parts', 'without', 'with', 'than', 'merchant', 'wholesalers',
        'retailers', 'stores', 'industries', 'industry', 'from',
        'systems', 'to', 'facilities', 'primarily', 'used', 'like',
        'such', 'those', 'certain', 'primarily', 'based', 'n.e.c', 'nec',

        # additional ambiguous/general words
        'real', 'business', 'extraction', 'generation', 'technical', 'nut',
        'pig', 'human', 'quality', 'crop', 'data', 'structure', 'agents',
        'paper', 'foundation', 'precision', 'development', 'works', 'relations',
        'collection', 'infrastructure', 'planning', 'professional',
        'information', 'system', 'process', 'control', 'program', 'design',
        'analysis', 'research', 'testing', 'application', 'network', 'security',
        'standard', 'value', 'resource', 'level', 'type', 'form', 'group',
        'service', 'product', 'material' # Singular forms if plurals already added
    }
    stop_words.update(domain_stopwords)
    # --------------------------------------------------------------------

    # Function to extract keywords from a list of titles
    def extract_keywords_from_titles(titles):
        # Combine all titles into a single text
        all_text = ' '.join(titles)

        # Convert to lowercase and remove non-alphanumeric characters (keep spaces)
        all_text = all_text.lower()
        # Preserve hyphens if they connect meaningful words (e.g., state-of-the-art) - slightly more complex
        # Simple approach: remove punctuation except maybe hyphens if needed
        all_text = re.sub(r'[^\w\s-]', '', all_text) # Keep words, spaces, hyphens
        all_text = re.sub(r'\s+', ' ', all_text).strip() # Normalize whitespace

        # Tokenize
        words = all_text.split()

        # Remove stopwords and short words (length > 2 or specific allowed short words if any)
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]

        # Count word frequencies
        word_counts = Counter(filtered_words)

        # Get the top N most common words
        top_words = word_counts.most_common(top_n)

        # Format as comma-separated string
        return ', '.join([word for word, count in top_words])

    # Group by sector and extract keywords
    sector_keywords = []
    # Ensure grouping columns exist
    if not all(col in results_df.columns for col in ['sector_code', 'sector_name', 'naics_title']):
         logger.error("Missing required columns for keyword extraction: 'sector_code', 'sector_name', 'naics_title'")
         # Return an empty DataFrame or raise error, depending on desired behavior
         return pd.DataFrame(columns=['sector_code', 'sector_name', 'keywords'])


    for (sector_code, sector_name), group in results_df.groupby(['sector_code', 'sector_name']):
        # Ensure group is not empty and naics_title is not all null
        if not group.empty and not group['naics_title'].isnull().all():
            keywords = extract_keywords_from_titles(group['naics_title'].dropna().tolist())
            sector_keywords.append({
                'sector_code': sector_code,
                'sector_name': sector_name,
                'keywords': keywords
            })
        else:
            logger.warning(f"Skipping keyword extraction for sector {sector_code} ({sector_name}) due to empty or null titles.")
            sector_keywords.append({
                'sector_code': sector_code,
                'sector_name': sector_name,
                'keywords': '' # Add empty keywords for consistency
            })


    # Create DataFrame from results
    keywords_df = pd.DataFrame(sector_keywords)

    return keywords_df


# %% Step 1: Load NAICS data
logger.info("Step 1: Loading NAICS data...")
try:
    naics_df = load_naics_data()
    print(f"Loaded {len(naics_df)} NAICS entries")
    print(naics_df.head())
except FileNotFoundError as e:
    logger.error(f"Failed to load NAICS data: {e}")
    # Exit or handle error appropriately if NAICS data is essential
    exit() # or raise e

# %% Step 2: Prepare data for classification
logger.info("Step 2: Preparing data for classification...")
try:
    classification_df = prepare_naics_for_classification(naics_df)
    print(f"Created {len(classification_df)} classification prompts for 4-digit NAICS codes")
    print(classification_df.head())
except ValueError as e:
     logger.error(f"Failed to prepare data: {e}")
     exit() # or handle error

# %% Step 3: Create system prompt
logger.info("Step 3: Creating system prompt...")
system_prompt = create_system_prompt()
print("System prompt created.")
print(system_prompt[:200] + "...")  # Preview of system prompt

# %% Step 4: Prepare for classification
logger.info("Step 4: Preparing for classification...")
# No need to add system prompt to each row anymore
print("Ready for classification with single system prompt")
print(classification_df.head())

# %% Step 5a: Run the classification via LLM or use step 5b to load results / manually

logger.info("Step 5a: Running CNI classification...")
# Process in batches to avoid rate limits
batch_size = 20
# --- Ensure you have 'model_completions' module and uncomment the import at the top ---

try:
    from model_completions import process_df_prompts # Ensure this is imported
    results_df = process_df_prompts(
        df=classification_df,
        model_type="claude",  # Or "gpt" depending on preference
        system_prompt=system_prompt,  # Pass the single system prompt here
        user_prompt_col='prompt',
        result_col='response',
        max_tokens=100,  # Increased for longer responses that include the CNI sector
        temperature=0.0,  # We want deterministic answers
        batch_size=batch_size,
        max_workers=4,
        show_progress=True
    )

    # Save raw results
    raw_results_file = 'naics_cni_classification_raw.csv'
    results_df.to_csv(raw_results_file, index=False)
    print(f"Classification complete and raw results saved to {raw_results_file}")
    print(results_df[['naics_code', 'naics_title', 'response']].head())

    # Example of printing sample titles per response category
    if 'response' in results_df.columns:
          unique_responses = sorted(results_df['response'].dropna().unique())
          print("\nSample Titles per Response Category:")
          for val in unique_responses:
              sample_titles = results_df[results_df['response'] == val]['naics_title'].sample(min(3, len(results_df[results_df['response'] == val]))).tolist()
              print(f"Response '{val}': " + ", ".join(sample_titles))
    else:
          print("Column 'response' not found in results_df after classification.")

except ImportError:
    logger.error("Module 'model_completions' not found. Cannot run classification.")
    print("Please ensure 'model_completions.py' is available and dependencies are installed.")
    results_df = None # Set results_df to None if classification fails
except Exception as e:
    logger.error(f"An error occurred during classification: {e}")
    results_df = None # Set results_df to None if classification fails


# %% Step 5b: Load results (if skipping Step 5a)
logger.info("Step 5b: Loading existing classification results...")
results_df = None # Initialize results_df
raw_results_file = 'naics_cni_classification_raw.csv'
try:
    results_df = pd.read_csv(raw_results_file)
    print(f"Loaded {len(results_df)} classification results from {raw_results_file}")
    # Basic check for expected columns after loading
    if 'response' not in results_df.columns:
         logger.warning(f"Loaded file {raw_results_file} is missing the 'response' column.")
         # Handle appropriately - maybe raise error or try to proceed cautiously
except FileNotFoundError:
    logger.error(f"Results file '{raw_results_file}' not found.")
    logger.error("Please run Step 5a first or ensure the file exists in the correct directory.")
    # Option 1: Exit the script
    # raise FileNotFoundError(f"Required results file {raw_results_file} not found.")
    # Option 2: Continue if later steps can handle results_df being None (currently they can't)
    print("Cannot proceed without classification results.")
    exit() # Exit if file is mandatory
except pd.errors.EmptyDataError:
    logger.error(f"Results file '{raw_results_file}' is empty.")
    print("Cannot proceed with empty results file.")
    exit() # Exit if file is empty
except Exception as e:
    logger.error(f"An error occurred while loading {raw_results_file}: {e}")
    print("Cannot proceed due to error loading results file.")
    exit() # Exit on other loading errors

# Check if results_df was loaded successfully before proceeding
if results_df is None:
    logger.critical("results_df was not loaded. Aborting script.")
    exit()
# -------------------------------------------------------------

# %% Step 6: Extract CNI sector and create binary flag
logger.info("Step 6: Extracting CNI sector and creating binary flag...")

# Ensure 'response' column exists and is string type
if 'response' not in results_df.columns:
    logger.error("Column 'response' not found in results_df. Cannot perform extraction.")
    # Handle error - maybe exit or skip this step if possible
    exit()
results_df['response'] = results_df['response'].astype(str) # Ensure string type for reliable .str access


# Create binary CNI column (1 for Yes, 0 for No) - Robust check for 'Yes' at the start
results_df['is_cni'] = results_df['response'].str.strip().str.startswith('Yes').astype(int)

# Extract CNI sector when identified (for "Yes" responses) using regex
# Regex explanation:
# - Starts with 'Mainly in '
# - Captures (.+) one or more characters (the sector name)
# - Ends with a literal dot \.
results_df['cni_sector'] = results_df['response'].str.extract(r'Mainly in (.*?)\.?$', expand=False).str.strip()
# Fill NaN for non-Yes responses or if pattern doesn't match
results_df['cni_sector'] = results_df['cni_sector'].where(results_df['is_cni'] == 1, other=pd.NA)


# Display the updated DataFrame with new columns
print("Updated DataFrame with is_cni and cni_sector columns:")
# Use pandas display options for better terminal output if needed
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(results_df[['naics_code', 'naics_title', 'response', 'is_cni', 'cni_sector']].head())

# Count of CNI vs non-CNI classifications
cni_counts = results_df['is_cni'].value_counts()
print(f"\nCNI Classification Counts:\n- Yes (1): {cni_counts.get(1, 0)}\n- No (0): {cni_counts.get(0, 0)}")

# Count of different CNI sectors identified
# Drop NA values before counting to only count identified sectors
cni_sector_counts = results_df['cni_sector'].dropna().value_counts().reset_index()
cni_sector_counts.columns = ['cni_sector', 'count']
print("\nIdentified CNI Sector Distribution:")
print(cni_sector_counts)

# %% Step 7: Calculate percentages by 2-digit sector and add keywords
logger.info("Step 7: Calculating percentages by sector and adding keywords...")

# Calculate CNI percentages per sector
sector_percentages = calculate_cni_percentages(results_df)

# Extract keywords for each sector
# Make sure results_df has the necessary columns: 'sector_code', 'sector_name', 'naics_title'
sector_keywords = extract_sector_keywords(results_df, top_n=30) # Increased top_n for more keywords

# Display the keywords
print("\nSector Keywords Extracted:")
print(sector_keywords)

# Merge keywords with sector_percentages
# Ensure 'sector_code' types match if merging (e.g., both string or both int)
sector_percentages['sector_code'] = sector_percentages['sector_code'].astype(str)
sector_keywords['sector_code'] = sector_keywords['sector_code'].astype(str)

sector_results = pd.merge(
    sector_percentages,
    sector_keywords[['sector_code', 'keywords']],
    on='sector_code',
    how='left' # Keep all sectors, even if keyword extraction failed for some
)
# Fill NA keywords with empty string if any merge mismatches occurred
sector_results['keywords'] = sector_results['keywords'].fillna('')


# Save the enhanced sector results to CSV
percentages_file = 'naics_cni_percentages.csv'
sector_results.to_csv(percentages_file, index=False)
print(f"\nSector percentages with keywords saved to {percentages_file}")

print("\nCritical Infrastructure Percentages by NAICS Sector (with keywords):")
print(sector_results[['sector_code', 'sector_name', 'total_subsectors', 'cni_subsectors', 'cni_percentage', 'keywords']])

# Save the complete results (original data + CNI flags + extracted sector)
complete_results_file = 'naics_cni_classification_complete.csv'
results_df.to_csv(complete_results_file, index=False)
print(f"\nComplete classification results saved to {complete_results_file}")


# %% Step 8: Visualize results with matplotlib
logger.info("Step 8: Creating visualization...")
plt_object = visualize_results(sector_results) # Use the merged results dataframe

# Save the plot if matplotlib is installed and plt_object is returned
if plt_object:
    plot_filename = 'naics_cni_percentages.png'
    try:
        plt_object.savefig(plot_filename, dpi=300, bbox_inches='tight') # Use bbox_inches='tight'
        print(f"Visualization saved to {plot_filename}")
        plt_object.show() # Display the plot
    except Exception as e:
        logger.error(f"Failed to save or show plot: {e}")
else:
    print("Skipping plot saving/displaying as Matplotlib might not be installed or visualization failed.")


# %%

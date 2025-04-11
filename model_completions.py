import os
import logging
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from aisitools.api_key import get_api_key_for_proxy
from anthropic import Anthropic
from openai import OpenAI

# Configure logging 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_claude_completion(system_prompt, conversation_history, anthropic_client, max_tokens=100, temperature=1.0):
    """Get completion from Anthropic's Claude model."""
    try:
        model_name = "claude-3-7-sonnet-20250219"

        # Format the conversation history into Claude's expected format
        messages = []
        for turn in conversation_history:
            if "user" in turn:
                messages.append({"role": "user", "content": turn["user"]})
            if "assistant" in turn:
                messages.append({"role": "assistant", "content": turn["assistant"]})

        response = anthropic_client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            messages=messages,
            system=system_prompt,
            temperature=temperature,
        )

        # Extract the response
        assistant_response = response.content[0].text

        # Get token usage
        token_usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens
        }

        # Log token usage
        logger.info(f"Claude Token Usage: {token_usage}")

        return assistant_response, token_usage

    except Exception as e:
        logger.error(f"Error in getting Claude completion: {str(e)}")
        return None, None


def get_gpt_completion(system_prompt, conversation_history, openai_client, max_tokens=100, temperature=1.0):
    """Get completion from OpenAI's GPT model."""
    try:
        model_name = "gpt-4o-2024-08-06"

        # Format the conversation history into OpenAI's expected format
        messages = [{"role": "system", "content": system_prompt}]

        for turn in conversation_history:
            if "user" in turn:
                messages.append({"role": "user", "content": turn["user"]})
            if "assistant" in turn:
                messages.append({"role": "assistant", "content": turn["assistant"]})

        response = openai_client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Extract the response
        assistant_response = response.choices[0].message.content

        # Get token usage
        token_usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }

        # Log token usage
        logger.info(f"GPT Token Usage: {token_usage}")

        return assistant_response, token_usage

    except Exception as e:
        logger.error(f"Error in getting GPT completion: {str(e)}")
        return None, None


def process_df_prompts(df, model_type, system_prompt="You are a helpful AI assistant.", user_prompt_col='prompt', 
                       result_col='response', max_tokens=100, temperature=1.0, 
                       batch_size=None, max_workers=1, show_progress=True):
    """
    Process a dataframe of prompts with the specified model using a single system prompt.
    
    Args:
        df (pandas.DataFrame): DataFrame containing prompts
        model_type (str): Model to use ("gpt" or "claude")
        system_prompt (str): System prompt to use for all rows (defaults to "You are a helpful AI assistant.")
        user_prompt_col (str): Column name with user prompts
        result_col (str): Column name to store model responses
        max_tokens (int): Maximum tokens to generate
        temperature (float): Sampling temperature (0.0 = deterministic, 1.0 = creative)
        batch_size (int, optional): Process in batches of this size
        max_workers (int): Number of parallel workers
        show_progress (bool): Whether to show progress bar
        
    Returns:
        pandas.DataFrame: Original dataframe with added response and token columns
    """
    # Make a copy of the dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Add result column if it doesn't exist
    if result_col not in result_df.columns:
        result_df[result_col] = None
    
    # Add token usage columns
    result_df['prompt_tokens'] = None
    result_df['completion_tokens'] = None
    result_df['total_tokens'] = None
    
    # Initialize the client based on model type
    client = None
    
    if model_type == "gpt":
        # Check if OPENAI_API_KEY is set
        if not os.environ.get("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY environment variable not set")
            return result_df
            
        # Get API key using the proxy
        api_key = get_api_key_for_proxy(os.environ.get("OPENAI_API_KEY"))
        client = OpenAI(api_key=api_key)
        
    elif model_type == "claude":
        # Check if ANTHROPIC_API_KEY is set
        if not os.environ.get("ANTHROPIC_API_KEY"):
            logger.error("ANTHROPIC_API_KEY environment variable not set")
            return result_df

        # Get API key using the proxy helper
        api_key = get_api_key_for_proxy(os.environ.get("ANTHROPIC_API_KEY"))
        
        # Explicitly set the api_key and the base_url for the proxy
        client = Anthropic(
            api_key=api_key
        )
       
    else:
        logger.error(f"Unsupported model type: {model_type}")
        return result_df
    
    # Function to process a single row
    def process_row(row):
        # Get user prompt
        user_prompt = row[user_prompt_col]
        if pd.isna(user_prompt):
            return None, None
        
        # Create conversation history
        conversation_history = [{"user": user_prompt}]
        
        # Get model completion
        if model_type == "claude":
            return get_claude_completion(
                system_prompt,
                conversation_history,
                client,
                max_tokens=max_tokens,
                temperature=temperature
            )
        elif model_type == "gpt":
            return get_gpt_completion(
                system_prompt,
                conversation_history,
                client,
                max_tokens=max_tokens,
                temperature=temperature
            )
    
    # Process in batches or all at once
    total_rows = len(result_df)
    if batch_size is None:
        batch_size = total_rows
    
    # Process in batches sequentially
    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        batch = result_df.iloc[start_idx:end_idx]
        
        # Wrap with tqdm for progress bar if requested
        iterator = tqdm(batch.iterrows(), total=len(batch), desc=f"Processing {model_type} batch") if show_progress else batch.iterrows()
        
        # Process each row sequentially
        for idx, row in iterator:
            response, token_usage = process_row(row)
            
            if response and token_usage:
                result_df.at[idx, result_col] = response
                result_df.at[idx, 'prompt_tokens'] = token_usage.get('prompt_tokens')
                result_df.at[idx, 'completion_tokens'] = token_usage.get('completion_tokens')
                result_df.at[idx, 'total_tokens'] = token_usage.get('total_tokens')
    
    return result_df

# example usage
import pandas as pd

# Create a dataframe with prompts
df = pd.DataFrame({
    'question': ["What is machine learning?", "How do neural networks work?"]
})

# Process the dataframe with a single system prompt
results = process_df_prompts(
    df,
    model_type="claude",
    system_prompt="Explain AI concepts clearly to beginners.",  # Single system prompt for all rows
    user_prompt_col="question",  # Column containing user prompts
    max_tokens=200
)

# Print results
print(results[['question', 'response', 'total_tokens']])


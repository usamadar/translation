import pandas as pd
import deepl
from openai import OpenAI
import os
from dotenv import load_dotenv

# --------------------------------------------
# 1. API Keys and Client Setup
# --------------------------------------------

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
DEEPL_AUTH_KEY = os.getenv("DEEPL_AUTH_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate that keys are available
if not DEEPL_AUTH_KEY or not OPENAI_API_KEY:
    raise ValueError("Missing API keys. Please check your .env file.")

# Create a DeepL translator instance
translator = deepl.Translator(DEEPL_AUTH_KEY)

# Create an OpenAI client instance using the provided API key
client = OpenAI(api_key=OPENAI_API_KEY)

# --------------------------------------------
# 2. Define the GPT Improvement Function
# --------------------------------------------

def improve_text_with_gpt(translated_text, original_text, column_name):
    """
    Improves the translated text using GPT by also providing the original French text and column context.
    The prompt instructs GPT to improve clarity, style, and correctness while not adding any markdown formatting.
    """
    if not translated_text:
        return ""
    
    # Define a system message with instructions
    system_prompt = (
        "You are an expert e-commerce translator specializing in product descriptions. "
        "Your task is to refine machine-translated text to sound natural and appealing to customers. "
        f"You're working with content from the '{column_name}' field of a product catalog. "
        "Focus on these aspects:\n"
        "1. Use industry-appropriate terminology for fashion/home goods\n"
        "2. Maintain the tone and style appropriate for luxury/premium products\n"
        "3. Ensure technical specifications are accurately preserved\n"
        "4. Make the text concise but compelling\n"
        "5. Preserve all factual information from the original\n"
        "Output plain text only without any formatting."
    )
    
    # Define the user prompt with the additional context
    user_prompt = (
        f"Column: {column_name}\n"
        f"Original French text: {original_text}\n"
        f"Machine translation: {translated_text}\n\n"
        "Please improve this translation to sound natural to native English speakers "
        "while preserving all product details and specifications."
    )
    
    # Request improvement from ChatGPT
    completion = client.chat.completions.create(
        model="gpt-4o",  # Upgrade to a more capable model if budget allows
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3  # Slightly higher temperature for more natural language
    )
    
    # Use attribute access to retrieve the improved text
    improved_text = completion.choices[0].message.content.strip()
    return improved_text

# --------------------------------------------
# 3. Read the Excel File
# --------------------------------------------

# Input file containing French product info (adjust the path as needed)
input_file_path = "input_fr.xlsx"
df = pd.read_excel(input_file_path)

# --------------------------------------------
# 4. Define Columns to Process
# --------------------------------------------

columns_to_translate = [
    "name - fr-FR",
    "color - fr-FR",
    "color_display - fr-FR",
    "material - fr-FR",
    "details - fr-FR",
    "care_label - fr-FR",
    "variant_size - fr-FR",
    "measures - fr-FR"
]

# --------------------------------------------
# 5. Process Each Cell: Translate then Improve
# --------------------------------------------

# For each column, process each text entry individually
for col in columns_to_translate:
    # Create a new column name for the improved English text
    new_col = col.replace("fr-FR", "en_improved")
    improved_texts = []
    
    for original_text in df[col]:
        if pd.isna(original_text):
            improved_texts.append("")
        else:
            try:
                # Translate the French text to English (using British English)
                translation = translator.translate_text(original_text, target_lang="EN-GB")
                translated_text = translation.text
                print("Translated text:", translated_text)
                
                # Improve the translated text using GPT, providing column and original text context
                improved_text = improve_text_with_gpt(translated_text, original_text, col)
                print("Improved text:", improved_text)
                improved_texts.append(improved_text)
            except Exception as e:
                print(f"Error processing text in column '{col}': {original_text} - {e}")
                improved_texts.append("")
    
    # Add the new improved text column to the DataFrame
    df[new_col] = improved_texts

# --------------------------------------------
# 6. Save the Results to a New Excel File
# --------------------------------------------

output_file_path = "westwing_en_improved.xlsx"
df.to_excel(output_file_path, index=False)
print("Translation and improvement done. Results saved to:", output_file_path)

import pandas as pd
import deepl
from openai import OpenAI
import os
from dotenv import load_dotenv
import json

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
# 2. Terminology and Context Management
# --------------------------------------------

# Westwing-specific terminology glossary
TERMINOLOGY = {
    "canapé": "sofa",
    "table basse": "coffee table",
    "lit": "bed",
    "armoire": "wardrobe",
    "coussin": "cushion",
    "décoration": "decor",
    "salon": "living room",
    "chambre": "bedroom",
    # Add more terms as needed
}

# Column-specific handling configurations
COLUMN_CONFIG = {
    "name - fr-FR": {
        "model": "gpt-4o",
        "importance": "high",
        "style": "concise but appealing"
    },
    "details - fr-FR": {
        "model": "gpt-4o",
        "importance": "high",
        "style": "detailed and descriptive"
    },
    "care_label - fr-FR": {
        "model": "gpt-4o",
        "importance": "high",
        "style": "precise and instructional"
    },
    # Default configuration for other columns
    "default": {
        "model": "gpt-4o",
        "importance": "medium",
        "style": "clear and accurate"
    }
}

# --------------------------------------------
# 3. Enhanced Translation Functions
# --------------------------------------------

def get_column_context(df, row_index, current_column):
    """
    Extracts context from neighboring columns to provide more context for translation.
    """
    context = {}
    # Get product name for context if available
    if "name - fr-FR" in df.columns and current_column != "name - fr-FR":
        context["product_name"] = df.iloc[row_index]["name - fr-FR"]
    
    # Get product category if available
    if "category - fr-FR" in df.columns:
        context["category"] = df.iloc[row_index]["category - fr-FR"]
        
    return context

def improve_text_with_gpt(translated_text, original_text, column_name, additional_context=None):
    """
    First-pass improvement of the translated text using GPT with enhanced context.
    """
    if not translated_text:
        return ""
    
    # Get column-specific configuration
    config = COLUMN_CONFIG.get(column_name, COLUMN_CONFIG["default"])
    
    # Define a system message with instructions
    system_prompt = (
        "You are an expert e-commerce translator specializing in premium home and living product descriptions for Westwing. "
        "Your task is to refine machine-translated text to sound natural and appealing to customers. "
        f"You're working with content from the '{column_name}' field of a product catalog. "
        f"This content should be {config['style']}. "
        "Focus on these aspects:\n"
        "1. Use industry-appropriate terminology for home goods and furniture\n"
        "2. Maintain the tone and style appropriate for luxury/premium products\n"
        "3. Ensure technical specifications are accurately preserved\n"
        "4. Make the text concise but compelling\n"
        "5. Preserve all factual information from the original\n"
        "Output plain text only without any formatting."
    )
    
    # Build context information
    context_info = ""
    if additional_context:
        for key, value in additional_context.items():
            if value and not pd.isna(value):
                context_info += f"{key}: {value}\n"
    
    # Apply terminology consistency
    improved_translation = translated_text
    for fr_term, en_term in TERMINOLOGY.items():
        if fr_term.lower() in original_text.lower():
            improved_translation = improved_translation.replace(fr_term, en_term)
    
    # Define the user prompt with the additional context
    user_prompt = (
        f"Column: {column_name}\n"
        f"{context_info}"
        f"Original French text: {original_text}\n"
        f"Machine translation: {improved_translation}\n\n"
        "Please improve this translation to sound natural to native English speakers "
        "while preserving all product details and specifications."
    )
    
    # Request improvement from ChatGPT
    completion = client.chat.completions.create(
        model=config["model"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )
    
    improved_text = completion.choices[0].message.content.strip()
    return improved_text

def refine_translation(first_pass_text, original_text, column_name, quality_criteria=None):
    """
    Second-pass refinement that reviews and improves the first translation.
    """
    if not first_pass_text:
        return ""
    
    # Get column-specific configuration
    config = COLUMN_CONFIG.get(column_name, COLUMN_CONFIG["default"])
    
    system_prompt = (
        "You are a senior editor at Westwing Home & Living, reviewing translated product descriptions. "
        "Your task is to refine the already improved translation to ensure it meets Westwing's premium standards. "
        f"This is for the '{column_name}' field which requires {config['style']} content. "
        "Focus on:\n"
        "1. Ensuring terminology is consistent with luxury home goods industry standards\n"
        "2. Polishing language to sound sophisticated yet accessible\n"
        "3. Maintaining factual accuracy compared to the original\n"
        "4. Addressing any specific quality issues mentioned\n"
    )
    
    quality_info = ""
    if quality_criteria:
        quality_info = f"Pay special attention to these aspects: {quality_criteria}\n\n"
    
    user_prompt = (
        f"Column: {column_name}\n"
        f"Original French: {original_text}\n"
        f"Current translation: {first_pass_text}\n"
        f"{quality_info}"
        "Please refine this translation to meet Westwing's premium standards. "
        "Return only the improved text without explanations."
    )
    
    completion = client.chat.completions.create(
        model=config["model"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2  # Lower temperature for more consistent refinement
    )
    
    refined_text = completion.choices[0].message.content.strip()
    return refined_text

def evaluate_translation_quality(original_text, translated_text, improved_text, column_name):
    """
    Uses GPT to evaluate the quality of translation and provide a rating.
    Returns a score (1-5) and brief feedback.
    """
    if not translated_text or not improved_text:
        return {"score": 0, "feedback": "Empty translation"}
    
    # Define evaluation prompt
    system_prompt = (
        "You are an expert e-commerce translator quality evaluator for Westwing Home & Living. "
        "Your task is to evaluate the quality of product description translations from French to English. "
        "Westwing is a premium home and living e-commerce company that requires high-quality, "
        "natural-sounding translations that maintain the luxury feel of their products."
    )
    
    user_prompt = (
        f"Column: {column_name}\n"
        f"Original French text: {original_text}\n"
        f"Machine translation: {translated_text}\n"
        f"Improved translation: {improved_text}\n\n"
        "Please evaluate the improved translation on a scale of 1-5 where:\n"
        "1 = Poor quality with major errors\n"
        "2 = Below average with noticeable issues\n"
        "3 = Acceptable but could be improved\n"
        "4 = Good quality with minor issues\n"
        "5 = Excellent, natural-sounding translation\n\n"
        "Provide a brief explanation (max 30 words) for your rating and list specific improvement areas if score is below 4."
        "Return your response in JSON format with 'score', 'feedback', and 'improvement_areas' keys."
    )
    
    # Request evaluation from ChatGPT
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.3
    )
    
    # Parse the JSON response
    try:
        evaluation = json.loads(completion.choices[0].message.content)
        return evaluation
    except Exception as e:
        print(f"Error parsing evaluation response: {e}")
        return {"score": 0, "feedback": f"Error: {str(e)}", "improvement_areas": ""}

# --------------------------------------------
# 4. Read the Excel File
# --------------------------------------------

# Input file containing French product info (adjust the path as needed)
input_file_path = "input_fr.xlsx"
df = pd.read_excel(input_file_path)

# --------------------------------------------
# 5. Define Columns to Process
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
# 6. Process Each Cell: Translate, Improve, Refine and Evaluate
# --------------------------------------------

# For each column, process each text entry individually
for col in columns_to_translate:
    # Create new column names
    new_col = col.replace("fr-FR", "en_improved")
    quality_col = col.replace("fr-FR", "quality_score")
    feedback_col = col.replace("fr-FR", "quality_feedback")
    
    improved_texts = []
    quality_scores = []
    quality_feedbacks = []
    
    for idx, original_text in enumerate(df[col]):
        if pd.isna(original_text):
            improved_texts.append("")
            quality_scores.append(0)
            quality_feedbacks.append("")
        else:
            try:
                # Get additional context from neighboring columns
                context = get_column_context(df, idx, col)
                
                # Translate the French text to English (using British English)
                translation = translator.translate_text(original_text, target_lang="EN-GB")
                translated_text = translation.text
                print("Translated text:", translated_text)
                
                # First pass: Improve the translated text using GPT
                first_pass = improve_text_with_gpt(translated_text, original_text, col, context)
                print("First pass improvement:", first_pass)
                
                # Evaluate the first pass translation
                evaluation = evaluate_translation_quality(original_text, translated_text, first_pass, col)
                
                # Second pass: Refine based on evaluation feedback
                if evaluation.get("score", 5) < 4 and "improvement_areas" in evaluation:
                    refined_text = refine_translation(first_pass, original_text, col, evaluation["improvement_areas"])
                    print("Second pass refinement:", refined_text)
                    improved_text = refined_text
                else:
                    improved_text = first_pass
                
                # Final evaluation
                final_evaluation = evaluate_translation_quality(original_text, translated_text, improved_text, col)
                
                improved_texts.append(improved_text)
                quality_scores.append(final_evaluation.get("score", 0))
                quality_feedbacks.append(final_evaluation.get("feedback", ""))
                
                print(f"Quality score: {final_evaluation.get('score', 0)}/5 - {final_evaluation.get('feedback', '')}")
                
            except Exception as e:
                print(f"Error processing text in column '{col}': {original_text} - {e}")
                improved_texts.append("")
                quality_scores.append(0)
                quality_feedbacks.append(f"Error: {str(e)}")
    
    # Add the new columns to the DataFrame
    df[new_col] = improved_texts
    df[quality_col] = quality_scores
    df[feedback_col] = quality_feedbacks

# --------------------------------------------
# 7. Save the Results to a New Excel File
# --------------------------------------------

output_file_path = "westwing_en_improved.xlsx"
df.to_excel(output_file_path, index=False)
print("Translation and improvement done. Results saved to:", output_file_path)

import os
from google import genai
from google.genai.errors import APIError

def generate_content_with_gemini(prompt: str) -> str:
    """
    Connects to the Gemini API, sends a simple text prompt, and returns the response.
    
    Args:
        prompt: The text prompt to send to the model.
        
    Returns:
        The text response from the model, or an error message.
    """
    try:
        # --- 1. Set up the API Client ---
        # The client will automatically look for the GEMINI_API_KEY 
        # environment variable. You can also pass it explicitly like:
        # client = genai.Client(api_key="YOUR_API_KEY")
        # However, using the environment variable is recommended for security.
        client = genai.Client(api_key = "AIzaSyDKVdCntVXrfIpV_DueZQChTZ1rIeMtvIc")
        
        # --- 2. Specify the Model ---
        # gemini-2.5-flash is a fast, capable model for general tasks.
        model_name = 'gemini-2.5-flash'
        
        print(f"Sending prompt to {model_name}...")
        
        # --- 3. Call the API ---
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        
        # --- 4. Return the Text Output ---
        return response.text

    except APIError as e:
        return f"An API Error occurred: {e}"
    except Exception as e:
        # This catches errors like the API key not being set.
        return f"An unexpected error occurred: {e}. Please ensure your GEMINI_API_KEY environment variable is set."

# --- Example Usage ---

# The simple prompt you want to send to the Gemini model
user_prompt = "Explain the concept of a black hole in one short paragraph."

# Get the response
model_response = generate_content_with_gemini(user_prompt)

# Print the results
print("\n--- Input Prompt ---")
print(user_prompt)

print("\n--- Gemini Output ---")
print(model_response)

print("--- End of Output ---")
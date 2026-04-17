import os
from google import genai
from dotenv import load_dotenv

# 1. Load the .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ Error: GEMINI_API_KEY not found in .env!")
else:
    print("✅ API Key found. Connecting via Modern GenAI SDK...")
    
    # 2. Initialize the Modern Client
    # It will automatically find the key if you name it GEMINI_API_KEY, 
    # but we'll be explicit here for safety.
    client = genai.Client(api_key=api_key)

    try:
        # 3. Make the request using Gemini 2.0 Flash (the 2026 standard)
        # You can also use 'gemini-1.5-flash' if you prefer
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents="Say 'Hello Ahmet, LinAI is online with the new SDK!'"
        )
        
        print("-" * 30)
        print(f"🤖 LinAI Response: {response.text}")
        print("-" * 30)
        print("🚀 Success! The modern connection is established.")
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")

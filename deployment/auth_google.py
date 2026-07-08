import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables from .env
load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in your environment variables or .env file.")

# Initialize the Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_google_auth_url():
    """
    Generate the OAuth authentication URL for Google provider.
    After successful login, the user will be redirected to the /dashboard endpoint.
    """
    # Replace 'http://127.0.0.1:5000' with your production URL when deploying.
    # The redirect URL must be whitelisted in your Supabase Dashboard under:
    # Authentication -> URL Configuration -> Redirect URLs
    redirect_url = os.environ.get("REDIRECT_URL", "https://zenstudy.up.railway.app/auth/callback")
    
    response = supabase.auth.sign_in_with_oauth(
        {
            "provider": "google",
            "options": {
                "redirect_to": "https://zenstudy.up.railway.app/auth/callback",
            }
        }
    )
    
    # The response contains the authorization URL that the user needs to visit to log in
    return response.url

if __name__ == "__main__":
    # Test generating the URL (helpful for debugging)
    try:
        url = get_google_auth_url()
        print("Generated Google Auth URL:")
        print(url)
    except Exception as e:
        print(f"Error generating auth URL: {e}")

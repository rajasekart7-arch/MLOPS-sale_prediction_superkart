from huggingface_hub import HfApi

from dotenv import load_dotenv
load_dotenv()

hf_token = os.getenv("HF_TOKEN")

# Define the repository ID for the Streamlit space
repo_id = "Rajse/Superkart-SalesPredictionFrontend"

# Initialize the API
api = HfApi(token=hf_token)

# Upload the frontend files stored in the 'frontend_files' folder
api.upload_folder(
    folder_path="./frontend_files",  # Local folder path
    repo_id=repo_id,  # Hugging Face space id
    repo_type="space"  # Hugging Face repo type "space"
)
print(f"Frontend files uploaded successfully to space: {repo_id}")


# for hugging face space authentication to upload files
from huggingface_hub import HfApi
import os

from dotenv import load_dotenv
load_dotenv()

hf_token = os.getenv("HF_TOKEN")

repo_id = "Rajse/Superkart-SalesPredictionBackend"  # Corrected Hugging Face space id

# Initialize the API
api = HfApi(token=hf_token)

# Upload Streamlit app files stored in the folder called deployment_files
api.upload_folder(
    folder_path="backend_files",  # Local folder path
    repo_id=repo_id,  # Hugging face space id
    repo_type="space",  # Hugging face repo type "space"
)

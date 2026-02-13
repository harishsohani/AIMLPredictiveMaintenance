
# import libraries for enabling interaction with Hugging Face
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo

import os

# ---------------------------------------
# Hugging Face repository IDs
# ---------------------------------------

hf_repo_id_backend  = "harishsohani/AIMLPredictiveMaintenanceBackEnd"
hf_repo_id_frontend = "harishsohani/AIMLPredictiveMaintenance"


# ---------------------------------------
# Initialize Hugging Face API
# ---------------------------------------
api = HfApi(token=os.getenv("HF_TOKEN"))


# ---------------------------------------
# Create BackEnd Space (Docker)
# ---------------------------------------

# Try to create the repository on Hugging Face Space for BackEnd
try:
    create_repo (
        repo_id=hf_repo_id_backend,   # Name of the space to be created for back end
        repo_type="space",            # repository type as 'space'
        space_sdk="docker",           # Specify the space SDK as "docker" to create a Docker space
        private=False,                # This is set as False, so that it is accessble to all
        exist_ok=True                 # ignore if present
    )
    print(f"Backend space created: {hf_repo_id_backend}")
except Exception as e:
    # Handle potential errors during repository creation
    if "RepositoryAlreadyExistsError" in str(e):
        print("Repository already exists. Skipping creation.")
    else:
        print(f"Error in creating repository: {e}")


# ---------------------------------------
# Upload BackEnd files
# ---------------------------------------
api.upload_folder(
    folder_path="capstone_project3/deployment/BackEnd",   # the local folder containing your files
    repo_id=hf_repo_id_backend,                                   # the target repo
    repo_type="space",                                    # dataset, model, or space
    path_in_repo="",                                      # optional: subfolder path inside the repo
)

print("Backend files uploaded successfully.")


#############   NOW Upload File for Front End ##############

# ---------------------------------------
# Create FrontEnd Space (Streamlit)
# ---------------------------------------

try:
    create_repo(
        repo_id=hf_repo_id_frontend,      # Name of the space to be created for Front End
        repo_type="space",                # repository type as 'space'
        space_sdk="streamlit",            # Specify the space
        private=False,                    # This is set as False, so that it is accessble to all
        exist_ok=True                     # ignore if present
    )
    print(f"Frontend space created: {hf_repo_id_frontend}")
except Exception as e:
    # Handle potential errors during repository creation
    if "RepositoryAlreadyExistsError" in str(e):
        print("Frontend space already exists. Skipping creation.")
    else:
        print(f"Error creating frontend space: {e}")


# ---------------------------------------
# Upload FrontEnd files
# ---------------------------------------
api.upload_folder(
    folder_path="capstone_project3/deployment/FrontEnd",  # the local folder containing your files
    repo_id=hf_repo_id_frontend,                                   # the target repo
    repo_type="space",                                    # dataset, model, or space
    path_in_repo="",                                      # optional: subfolder path inside the repo
)

print("Frontend files uploaded successfully.")

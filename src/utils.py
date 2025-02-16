import os

def upload_to_huggingface(csv_file_path):
    from huggingface_hub import HfApi, HfFolder

    hf_api = HfApi()
    token = HfFolder.get_token()
    repo_id = "matthieunlp/spatial_geometry"

    hf_api.upload_file(
        path_or_fileobj=csv_file_path,
        path_in_repo=csv_file_path,
        repo_id=repo_id,
        repo_type="dataset",
        token=token
    )
    print(f"Sentences uploaded to Hugging Face dataset: {repo_id}")

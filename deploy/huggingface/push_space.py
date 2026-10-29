"""Uploads the assembled folder to a Hugging Face Space (used by GitHub Actions).

    HF_TOKEN=... python deploy/huggingface/push_space.py space
Optional env: SPACE_NAME (default lungscan-ai), ADMIN_USER, ADMIN_PASSWORD, ANTHROPIC_API_KEY.
Writes the public URL to $GITHUB_OUTPUT as space_url.
"""
import os
import sys

from huggingface_hub import HfApi

folder = sys.argv[1] if len(sys.argv) > 1 else "space"
api = HfApi(token=os.environ["HF_TOKEN"])
user = api.whoami()["name"]
name = os.environ.get("SPACE_NAME") or "lungscan-ai"
repo_id = f"{user}/{name}"
api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True, private=False)
for key in ("ADMIN_USER", "ADMIN_PASSWORD", "ANTHROPIC_API_KEY"):
    if os.environ.get(key):
        api.add_space_secret(repo_id, key, os.environ[key])
api.upload_folder(folder_path=folder, repo_id=repo_id, repo_type="space",
                  commit_message="Deploy from GitHub " + os.environ.get("GITHUB_SHA", "")[:7],
                  delete_patterns=["*.py", "doctor_app/**", "patient_app/**", "doctor_flutter/**"])
url = f"https://{user.lower().replace('_', '-').replace('.', '-')}-{name.lower()}.hf.space"
print("Space:", f"https://huggingface.co/spaces/{repo_id}")
print("App:  ", url)
if os.environ.get("GITHUB_OUTPUT"):
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"space_url={url}\n")

import argparse
import os
import sys
from huggingface_hub import HfApi

def main():
    parser = argparse.ArgumentParser(description="Push model weights to HuggingFace Hub.")
    parser.add_argument("--file", type=str, required=True, help="Path to .eqx weights file")
    parser.add_argument("--repo", type=str, required=True, help="HuggingFace repository ID (e.g., user/repo)")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace token (optional, can use env var HF_TOKEN)")
    parser.add_argument("--commit-message", type=str, default="Upload model weights", help="Commit message")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("Error: No HuggingFace token provided. Use --token or set HF_TOKEN env var.")
        sys.exit(1)

    api = HfApi(token=token)
    
    filename = os.path.basename(args.file)
    print(f"Uploading {args.file} to {args.repo} as {filename}...")
    
    try:
        api.upload_file(
            path_or_fileobj=args.file,
            path_in_repo=filename,
            repo_id=args.repo,
            repo_type="model",
            commit_message=args.commit_message
        )
        print("Upload successful!")
    except Exception as e:
        print(f"Upload failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

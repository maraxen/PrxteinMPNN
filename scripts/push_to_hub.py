import argparse
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi

DEFAULT_WEIGHTS_PATH = "prxteinmpnn_dropout_weights.eqx"
DEFAULT_REPO_ID = "maraxen/PrxteinMPNN"

def main():
    parser = argparse.ArgumentParser(
        description="Push model weights to HuggingFace Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload migrated weights with default path
  python push_to_hub.py
  
  # Upload custom weights file
  python push_to_hub.py --file custom_weights.eqx
  
  # Upload to different repo
  python push_to_hub.py --repo username/model-name
        """
    )
    parser.add_argument(
        "--file", 
        type=str, 
        default=DEFAULT_WEIGHTS_PATH,
        help=f"Path to .eqx weights file (default: {DEFAULT_WEIGHTS_PATH})"
    )
    parser.add_argument(
        "--repo", 
        type=str, 
        default=DEFAULT_REPO_ID,
        help=f"HuggingFace repository ID (default: {DEFAULT_REPO_ID})"
    )
    parser.add_argument(
        "--token", 
        type=str, 
        default=None, 
        help="HuggingFace token (optional, can use env var HF_TOKEN)"
    )
    parser.add_argument(
        "--commit-message", 
        type=str, 
        default="Upload dropout-enabled model weights", 
        help="Commit message"
    )
    parser.add_argument(
        "--path-in-repo",
        type=str,
        default=None,
        help="Path in repo (default: same as filename)"
    )
    args = parser.parse_args()

    # Check if file exists
    if not Path(args.file).exists():
        print(f"✗ Error: File not found: {args.file}")
        print(f"\nDid you run migrate_weights.py first?")
        sys.exit(1)

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("✗ Error: No HuggingFace token provided.")
        print("Use --token argument or set HF_TOKEN environment variable.")
        sys.exit(1)

    api = HfApi(token=token)
    
    path_in_repo = args.path_in_repo or os.path.basename(args.file)
    
    print(f"Uploading {args.file} to {args.repo}/{path_in_repo}...")
    print(f"Commit message: {args.commit_message}")
    
    try:
        api.upload_file(
            path_or_fileobj=args.file,
            path_in_repo=path_in_repo,
            repo_id=args.repo,
            repo_type="model",
            commit_message=args.commit_message
        )
        print(f"\n✓ Upload successful!")
        print(f"View at: https://huggingface.co/{args.repo}/blob/main/{path_in_repo}")
    except Exception as e:
        print(f"\n✗ Upload failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

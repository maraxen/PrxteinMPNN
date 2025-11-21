#!/usr/bin/env python3
"""Upload all migrated PrxteinMPNN weights to HuggingFace Hub.

This script uploads all migrated weight files from the migrated_weights/ directory
to the HuggingFace Hub, maintaining the same directory structure (eqx/).
"""

import argparse
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi

HF_REPO_ID = "maraxen/prxteinmpnn"
MIGRATED_WEIGHTS_DIR = Path("migrated_weights")


def main():
    parser = argparse.ArgumentParser(
        description="Upload all migrated weights to HuggingFace Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload all migrated weights
  export HF_TOKEN=your_token_here
  python push_all_to_hub.py
  
  # Upload to different repo
  python push_all_to_hub.py --repo username/model-name
  
  # Dry run (show what would be uploaded)
  python push_all_to_hub.py --dry-run
        """
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=HF_REPO_ID,
        help=f"HuggingFace repository ID (default: {HF_REPO_ID})"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (optional, can use env var HF_TOKEN)"
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=MIGRATED_WEIGHTS_DIR,
        help=f"Directory containing migrated weights (default: {MIGRATED_WEIGHTS_DIR})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading"
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload dropout-enabled model weights",
        help="Commit message for all uploads"
    )
    
    args = parser.parse_args()
    
    # Check if weights directory exists
    if not args.weights_dir.exists():
        print(f"✗ Error: Weights directory not found: {args.weights_dir}")
        print("\nDid you run migrate_all_weights.py first?")
        sys.exit(1)
    
    # Find all .eqx files
    weight_files = list(args.weights_dir.glob("*.eqx"))
    if not weight_files:
        print(f"✗ Error: No .eqx files found in {args.weights_dir}")
        sys.exit(1)
    
    print("="*60)
    print("PrxteinMPNN Weight Upload Tool")
    print("="*60)
    print(f"\nRepository: {args.repo}")
    print(f"Weights directory: {args.weights_dir}")
    print(f"Files to upload: {len(weight_files)}")
    print(f"Dry run: {args.dry_run}")
    
    if not args.dry_run:
        token = True # I am logged in not needed, args.token or os.environ.get("HF_TOKEN")
        if not token:
            print("\n✗ Error: No HuggingFace token provided.")
            print("Use --token argument or set HF_TOKEN environment variable.")
            sys.exit(1)
        
        api = HfApi(token=token)
    
    # Upload each file
    successful = 0
    failed = 0
    
    print("\n" + "="*60)
    print("UPLOADING FILES")
    print("="*60)
    
    for weight_file in sorted(weight_files):
        filename = weight_file.name
        path_in_repo = f"eqx/{filename}"
        
        print(f"\n{filename}")
        print(f"  → {args.repo}/{path_in_repo}")
        
        if args.dry_run:
            print("  [DRY RUN] Would upload")
            successful += 1
            continue
        
        try:
            api.upload_file(
                path_or_fileobj=str(weight_file),
                path_in_repo=path_in_repo,
                repo_id=args.repo,
                repo_type="model",
                commit_message=f"{args.commit_message}: {filename}"
            )
            print("  ✓ Uploaded successfully")
            successful += 1
        except Exception as e:
            print(f"  ✗ Upload failed: {e}")
            failed += 1
    
    # Print summary
    print("\n" + "="*60)
    print("UPLOAD SUMMARY")
    print("="*60)
    print(f"Total files: {len(weight_files)}")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    
    if not args.dry_run and successful > 0:
        print(f"\n✓ View uploaded weights at:")
        print(f"  https://huggingface.co/{args.repo}/tree/main/eqx")
    
    if failed > 0:
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✓ All weights uploaded successfully!")
    print("="*60)


if __name__ == "__main__":
    main()

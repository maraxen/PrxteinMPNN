#!/usr/bin/env python3
"""Upload new .eqx models to HuggingFace and clean up old files.

This script:
1. Lists existing files in the HF repo
2. Deletes old .eqx and .pkl files
3. Uploads new .eqx files
4. Updates the model card
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi, list_repo_files

# Configuration
REPO_ID = "maraxen/prxteinmpnn"
REPO_TYPE = "model"
LOCAL_MODEL_DIR = Path(__file__).parent.parent / "models" / "new_format"

# Model files to upload
MODELS = [
  "original_v_48_002.eqx",
  "original_v_48_010.eqx",
  "original_v_48_020.eqx",
  "original_v_48_030.eqx",
  "soluble_v_48_002.eqx",
  "soluble_v_48_010.eqx",
  "soluble_v_48_020.eqx",
  "soluble_v_48_030.eqx",
]


def main():
  """Main upload workflow."""
  api = HfApi()

  print(f"🔍 Connecting to HuggingFace repo: {REPO_ID}")

  # Step 1: List existing files
  print("\n📋 Step 1: Listing existing files in repo...")
  try:
    existing_files = list(list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE))
    print(f"   Found {len(existing_files)} files")

    # Show .eqx and .pkl files
    eqx_files = [f for f in existing_files if f.endswith(".eqx")]
    pkl_files = [f for f in existing_files if f.endswith(".pkl")]

    if eqx_files:
      print(f"\n   Existing .eqx files ({len(eqx_files)}):")
      for f in eqx_files:
        print(f"     - {f}")

    if pkl_files:
      print(f"\n   Existing .pkl files ({len(pkl_files)}):")
      for f in pkl_files:
        print(f"     - {f}")

  except Exception as e:
    print(f"❌ Error listing files: {e}")
    return 1

  # Step 2: Delete old files
  print("\n🗑️  Step 2: Deleting old .eqx and .pkl files...")
  files_to_delete = eqx_files + pkl_files

  if not files_to_delete:
    print("   No files to delete.")
  else:
    confirm = input(f"\n   ⚠️  About to delete {len(files_to_delete)} files. Continue? [y/N]: ")
    if confirm.lower() != "y":
      print("   Aborted.")
      return 0

    for file_path in files_to_delete:
      try:
        print(f"   Deleting: {file_path}")
        api.delete_file(
          path_in_repo=file_path,
          repo_id=REPO_ID,
          repo_type=REPO_TYPE,
          commit_message=f"Remove old {file_path}",
        )
        print("     ✓ Deleted")
      except Exception as e:
        print(f"     ❌ Error: {e}")

  # Step 3: Upload new .eqx files
  print("\n⬆️  Step 3: Uploading new .eqx files...")

  # Verify local files exist
  missing_files = []
  for model_file in MODELS:
    local_path = LOCAL_MODEL_DIR / model_file
    if not local_path.exists():
      missing_files.append(model_file)

  if missing_files:
    print("\n❌ Missing local files:")
    for f in missing_files:
      print(f"   - {f}")
    print(f"\n   Expected location: {LOCAL_MODEL_DIR}")
    return 1

  print(f"\n   Found all {len(MODELS)} model files locally")
  confirm = input(f"\n   📤 Ready to upload {len(MODELS)} files. Continue? [y/N]: ")
  if confirm.lower() != "y":
    print("   Aborted.")
    return 0

  for model_file in MODELS:
    local_path = LOCAL_MODEL_DIR / model_file

    # Determine folder structure (keep original/soluble separate if they exist)
    if (
      "original" in model_file
      or local_path.parent.name == "original"
      or "soluble" in model_file
      or local_path.parent.name == "soluble"
    ):
      path_in_repo = f"eqx/{model_file}"
    else:
      path_in_repo = f"eqx/{model_file}"

    try:
      print(f"\n   Uploading: {model_file}")
      print(f"     Local:  {local_path}")
      print(f"     Remote: {path_in_repo}")

      api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message=f"Upload new Equinox model: {model_file}",
      )
      print("     ✓ Uploaded successfully")
    except Exception as e:
      print(f"     ❌ Error: {e}")
      return 1

  # Step 4: Summary
  print("\n" + "=" * 60)
  print("✅ Upload complete!")
  print("=" * 60)
  print(f"\n📦 Uploaded {len(MODELS)} new .eqx models to:")
  print(f"   https://huggingface.co/{REPO_ID}")
  print("\n💡 Next steps:")
  print("   1. Update the model card on HuggingFace")
  print("   2. Update README with .eqx usage examples")
  print("   3. Test downloading models from HuggingFace")
  print("   4. Update prxteinmpnn.io module to use .eqx by default")

  return 0


if __name__ == "__main__":
  try:
    sys.exit(main())
  except KeyboardInterrupt:
    print("\n\n⚠️  Interrupted by user")
    sys.exit(1)
  except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

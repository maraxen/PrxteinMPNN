#!/usr/bin/env python3
"""Upload the README to HuggingFace model repository."""

from pathlib import Path

from huggingface_hub import HfApi


def main():
  """Upload README to HuggingFace."""
  # Initialize HF API
  api = HfApi()

  # Path to README
  readme_path = Path(__file__).parent.parent / "HUGGINGFACE_README.md"

  if not readme_path.exists():
    print(f"❌ README not found at {readme_path}")
    return

  print("📤 Uploading README to HuggingFace...")
  print(f"   Source: {readme_path}")
  print("   Destination: maraxen/prxteinmpnn/README.md")

  # Upload the README
  api.upload_file(
    path_or_fileobj=str(readme_path),
    path_in_repo="README.md",
    repo_id="maraxen/prxteinmpnn",
    repo_type="model",
    commit_message="Update README with .eqx model usage instructions",
  )

  print("✅ README uploaded successfully!")
  print()
  print("🔗 View at: https://huggingface.co/maraxen/prxteinmpnn")


if __name__ == "__main__":
  main()

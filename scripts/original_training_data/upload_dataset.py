import argparse
import json
import logging
import pathlib
import sys
from typing import Optional

from array_record.python import array_record_module as array_record  # type: ignore[unresolved-import]
from huggingface_hub import HfApi, create_repo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def verify_dataset(record_path: pathlib.Path, index_path: pathlib.Path) -> bool:
    """
    Verifies integrity of the ArrayRecord and its Index before upload.
    Returns True if valid, False otherwise.
    """
    logger.info(f"Verifying dataset integrity...")
    
    # 1. Check File Existence
    if not record_path.exists():
        logger.error(f"ArrayRecord file not found: {record_path}")
        return False
    if not index_path.exists():
        logger.error(f"Index file not found: {index_path}")
        return False

    # 2. Verify ArrayRecord Readable
    try:
        reader = array_record.ArrayRecordReader(str(record_path))
        num_records = reader.num_records()
        logger.info(f"✅ ArrayRecord is readable. Total records: {num_records}")
        
        # Try reading the first and last record to ensure no truncation
        if num_records > 0:
            _ = reader.read()
            _ = reader.read([num_records - 1])
            logger.info("✅ Successfully read first and last records.")
        reader.close()
    except Exception as e:
        logger.error(f"❌ Failed to read ArrayRecord: {e}")
        return False

    # 3. Verify Index JSON
    try:
        with open(index_path, "r") as f:
            index_data = json.load(f)
        
        # Basic structure check
        if not isinstance(index_data, dict):
            logger.error("❌ Index file is not a JSON dictionary.")
            return False
            
        # Check if index size roughly matches (heuristic)
        # Note: Index keys are usually strings (names), not necessarily equal to num_records if 1-to-many
        logger.info(f"✅ Index JSON loaded. Contains {len(index_data)} keys.")
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ Index file is corrupted JSON: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to read Index file: {e}")
        return False

    logger.info("Dataset passed verification checks.")
    return True

def upload_files(
    repo_id: str, 
    record_path: pathlib.Path, 
    index_path: pathlib.Path, 
    token: Optional[str] = None
):
    """Uploads the validated files to Hugging Face Hub."""
    api = HfApi(token=token)
    
    logger.info(f"Creating/Checking repository: {repo_id}")
    create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)
    
    logger.info(f"Uploading {record_path.name}...")
    api.upload_file(
        path_or_fileobj=record_path,
        path_in_repo=record_path.name,
        repo_id=repo_id,
        repo_type="dataset"
    )
    
    logger.info(f"Uploading {index_path.name}...")
    api.upload_file(
        path_or_fileobj=index_path,
        path_in_repo="index.json", # Force standard name 'index.json' in repo
        repo_id=repo_id,
        repo_type="dataset"
    )
    
    logger.info("🎉 Upload complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify and upload ArrayRecord dataset to Hugging Face.")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face Dataset Repo ID (e.g., username/dataset_name)")
    parser.add_argument("--record_path", type=pathlib.Path, default=pathlib.Path("src/prxteinmpnn/training/data/pdb_2021aug02.array_record"), help="Path to the combined ArrayRecord file")
    parser.add_argument("--index_path", type=pathlib.Path, default=pathlib.Path("src/prxteinmpnn/training/data/pdb_2021aug02.index.json"), help="Path to the index JSON file")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API Token (optional if logged in via CLI)")

    args = parser.parse_args()

    if verify_dataset(args.record_path, args.index_path):
        upload_files(args.repo_id, args.record_path, args.index_path, args.token)
    else:
        logger.error("Aborting upload due to verification failure.")
        sys.exit(1)
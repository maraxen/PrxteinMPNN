import argparse
import re
import os
from pathlib import Path
from array_record.python import array_record_module as array_record

def inspect_shards(data_dir: Path, expected_shards: int):
    """
    Audits the data directory to classify shards into Valid, Corrupt, and Missing.
    """
    print(f"🔍 Inspecting {data_dir} for {expected_shards} shards...\n")

    if not data_dir.exists():
        print(f"❌ Error: Directory {data_dir} does not exist.")
        return

    valid_shards = []
    corrupt_shards = []
    
    # Regex to extract shard index from filename: data-00001-of-00064.array_record
    # Adjust regex if your naming convention differs slightly
    pattern = re.compile(r"data-(\d+)-of-(\d+)\.array_record")
    
    files = sorted(list(data_dir.glob("data-*.array_record")))
    
    total_records_found = 0

    print(f"{'SHARD FILE':<35} | {'STATUS':<10} | {'RECORDS':<10} | {'SIZE (MB)':<10}")
    print("-" * 75)

    # 1. Check existing files
    found_indices = set()
    
    for f in files:
        match = pattern.search(f.name)
        if not match:
            print(f"{f.name:<35} | SKIPPED    | N/A        | N/A")
            continue
            
        shard_idx = int(match.group(1))
        found_indices.add(shard_idx)
        
        file_size_mb = f.stat().st_size / (1024 * 1024)
        
        # Status Check
        status = "✅ VALID"
        count = 0
        
        try:
            # Check 1: Is file empty?
            if f.stat().st_size == 0:
                raise ValueError("Empty file")

            # Check 2: Can ArrayRecord C++ reader open it?
            reader = array_record.ArrayRecordReader(str(f))
            count = reader.num_records()
            
            for i in range(count):
                _ = reader.read()
            
            reader.close()
            valid_shards.append(shard_idx)
            total_records_found += count

        except Exception as e:
            print(e)
            status = "❌ CORRUPT"
            corrupt_shards.append(f)
        
        print(f"{f.name:<35} | {status:<10} | {str(count):<10} | {file_size_mb:.2f}")

    # 2. Determine Missing Indices
    all_indices = set(range(expected_shards)) # 0 to 63
    missing_indices = sorted(list(all_indices - found_indices))

    print("-" * 75)
    print("\n📊 SUMMARY REPORT")
    print(f"Total Expected Shards: {expected_shards}")
    print(f"Valid Shards:          {len(valid_shards)} (Safe to combine)")
    print(f"Corrupt Shards:        {len(corrupt_shards)} (Must delete)")
    print(f"Missing Shards:        {len(missing_indices)} (Never started)")
    print(f"Total Records Saved:   {total_records_found}")

    # 3. Action Items
    if corrupt_shards:
        print("\n⚠️  ACTION REQUIRED: Delete these corrupt files before resuming:")
        for cs in corrupt_shards:
            print(f"   rm {cs}")
    
    if missing_indices or corrupt_shards:
        print(f"\n🔄 TO RESUME: You need to re-run processing for shard indices:")
        # Combine missing and corrupted indices into a todo list
        todo_indices = sorted(missing_indices + [int(pattern.search(c.name).group(1)) for c in corrupt_shards])
        print(f"   Target Indices: {todo_indices}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("src/prxteinmpnn/training/data"))
    parser.add_argument("--shards", type=int, default=64, help="Total number of workers/shards used originally")
    args = parser.parse_args()
    
    inspect_shards(args.dir, args.shards)
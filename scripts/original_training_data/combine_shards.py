import argparse
import logging
from pathlib import Path
from array_record.python import array_record_module as array_record

def combine_shards(input_dir: Path, output_path: Path, pattern: str = "data-*.array_record"):
    """
    Reads multiple ArrayRecord shards and writes them into a single valid ArrayRecord file.
    """
    input_files = sorted(list(input_dir.glob(pattern)))
    if not input_files:
        print(f"No files matching {pattern} found in {input_dir}")
        return

    print(f"Combining {len(input_files)} shards into {output_path}...")
    
    # Use the same compression options as your shards
    writer = array_record.ArrayRecordWriter(str(output_path), "zstd:9,group_size:1")
    
    total_records = 0
    for f in input_files:
        try:
            reader = array_record.ArrayRecordReader(str(f))
            num_records = reader.num_records()
            
            for _ in range(num_records):
                record = reader.read()
                writer.write(record)
            
            total_records += num_records
            reader.close()
            print(f"Merged {f.name} ({num_records} records)")
        except Exception as e:
            print(f"Error reading {f}: {e}")

    writer.close()
    print(f"Successfully combined {total_records} records into {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=Path("src/prxteinmpnn/training/data"))
    parser.add_argument("--output_file", type=Path, default=Path("src/prxteinmpnn/training/data/pdb_2021aug02.array_record"))
    args = parser.parse_args()

    combine_shards(args.input_dir, args.output_file)
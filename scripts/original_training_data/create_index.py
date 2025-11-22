import argparse
import json
import csv
import pathlib
import msgpack
import msgpack_numpy as m
import tqdm
from array_record.python import array_record_module as array_record

# Patch msgpack to handle numpy arrays serialized in the records
m.patch()

def load_cluster_file(filepath: pathlib.Path) -> set:
    """Reads a file of cluster IDs (one per line)."""
    clusters = set()
    if not filepath.exists():
        print(f"Warning: Cluster file {filepath} not found. Split will be empty.")
        return clusters
    
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            # Skip comments or empty lines
            if not line or line.startswith("#"):
                continue
            clusters.add(line)
    return clusters

def load_chain_to_cluster_map(csv_path: pathlib.Path) -> dict:
    """
    Parses list.csv to map CHAINID -> CLUSTER.
    Expected columns: CHAINID,DEPOSITION,RESOLUTION,HASH,CLUSTER,SEQUENCE
    """
    mapping = {}
    if not csv_path.exists():
        print(f"Warning: Metadata file {csv_path} not found. Cannot map chains to clusters.")
        return mapping

    print(f"Loading metadata from {csv_path}...")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Store mapping: "5naf_A" -> "12123"
            if "CHAINID" in row and "CLUSTER" in row:
                mapping[row["CHAINID"]] = row["CLUSTER"]
    
    print(f"Mapped {len(mapping)} chains to clusters.")
    return mapping

def determine_split(record_name, record_data, chain_map, valid_clusters, test_clusters):
    """
    Determines if a record belongs to train, valid, or test.
    Handles both single chains (PDBID_CHAINID) and assemblies (PDBID).
    """
    # 1. Try to get the Cluster ID directly for the name
    cluster_id = chain_map.get(record_name)

    # 2. If not found, it might be an assembly (e.g., "1abc") containing multiple chains
    #    Check the 'chains' field in the record data
    chains = record_data.get(b"chains") or record_data.get("chains")
    
    associated_clusters = set()
    if cluster_id:
        associated_clusters.add(cluster_id)
    elif chains:
        # If it's an assembly, gather clusters for all constituent chains
        # chains is likely a list of bytes or strings: [b"1abc_A", b"1abc_B"]
        if isinstance(chains, (list, tuple)):
            for c in chains:
                c_str = c.decode("utf-8") if isinstance(c, bytes) else str(c)
                c_cluster = chain_map.get(c_str)
                if c_cluster:
                    associated_clusters.add(c_cluster)

    # 3. strict splitting logic:
    #    If ANY chain in this record belongs to a test cluster -> Test
    #    Else if ANY chain belongs to a valid cluster -> Valid
    #    Else -> Train
    
    # Check for intersection with splits
    if not associated_clusters.isdisjoint(test_clusters):
        return "test"
    if not associated_clusters.isdisjoint(valid_clusters):
        return "valid"
    
    return "train"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=pathlib.Path, 
                        default=pathlib.Path("src/prxteinmpnn/training/data/pdb_2021aug02.array_record"))
    parser.add_argument("--output_index", type=pathlib.Path, 
                        default=pathlib.Path("src/prxteinmpnn/training/data/pdb_2021aug02.index.json"))
    parser.add_argument("--cluster_dir", type=pathlib.Path, 
                        default=pathlib.Path("src/prxteinmpnn/training/data/pdb_sample/pdb_2021aug02_sample"))
    parser.add_argument("--metadata_csv", type=pathlib.Path,
                        default=None, help="Path to list.csv. Defaults to cluster_dir/list.csv")

    args = parser.parse_args()
    
    # Path Setup
    if args.metadata_csv is None:
        args.metadata_csv = args.cluster_dir / "list.csv"

    # 1. Load Splits (Cluster IDs)
    valid_clusters = load_cluster_file(args.cluster_dir / "valid_clusters.txt")
    test_clusters = load_cluster_file(args.cluster_dir / "test_clusters.txt")
    print(f"Loaded {len(valid_clusters)} validation and {len(test_clusters)} test clusters.")

    # 2. Load Metadata (Chain -> Cluster)
    chain_to_cluster = load_chain_to_cluster_map(args.metadata_csv)

    # 3. Open ArrayRecord
    if not args.input_file.exists():
        print(f"Error: Input file {args.input_file} not found.")
        exit(1)

    try:
        reader = array_record.ArrayRecordReader(str(args.input_file))
    except Exception as e:
        print(f"Error opening ArrayRecord: {e}")
        print("Did you run combine_array_record.py successfully?")
        exit(1)

    num_records = reader.num_records()
    print(f"Indexing {num_records} records...")

    # Index Structure
    index_data = {}
    counts = {"train": 0, "valid": 0, "test": 0}
    
    # 4. Iterate and Index
    # Reading one by one is slow for huge datasets, but necessary for unpacking and inspecting content.
    # If dataset is massive, consider ParallelRead, but sequential is safer for simple indexing logic.
    for i in tqdm.tqdm(range(num_records), desc="Building Index"):
        try:
            # We use read(i, i+1) to get specific records safely if the bindings support it, 
            # or standard sequential read if we iterate naturally. 
            # Assuming sequential access is fastest:
            data_bytes = reader.read([i])[0] # Reading by index list to get specific record
            
            # Unpack
            record = msgpack.unpackb(data_bytes, raw=False)
            
            # Identify Record
            # Dictionary keys might be bytes or strings depending on how they were packed.
            # We try to handle both.
            
            # Common keys in protein datasets: 'name', 'id', 'PDBID', 'chain_id'
            # Based on description: PDBID_CHAINID.pt
            name = record.get("name") or record.get(b"name") or record.get("id") or record.get(b"id")
            
            if isinstance(name, bytes):
                name = name.decode("utf-8")
                
            if not name:
                # Fallback if name isn't explicitly in keys (unlikely for this dataset type)
                name = f"record_{i}"

            split = determine_split(name, record, chain_to_cluster, valid_clusters, test_clusters)
            
            counts[split] += 1
            
            if name not in index_data:
                index_data[name] = {"idx": [], "set": split}
            
            index_data[name]["idx"].append(i)

        except Exception as e:
            print(f"Error processing record {i}: {e}")
            continue

    reader.close()

    # 5. Save Index
    print(f"\nStatistics: {counts}")
    
    # Ensure output directory exists
    args.output_index.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Writing index to {args.output_index}...")
    with open(args.output_index, "w") as f:
        json.dump(index_data, f, indent=2)
        
    print("Done.")

if __name__ == "__main__":
    main()
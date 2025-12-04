import argparse
import json
import csv
import pathlib
import msgpack
import msgpack_numpy as m
import tqdm
from array_record.python import array_record_module as array_record

m.patch()

def load_cluster_file(filepath: pathlib.Path) -> set:
    """Reads a file of cluster IDs (one per line)."""
    clusters = set()
    if not filepath.exists():
        return clusters
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                clusters.add(line)
    return clusters

def load_chain_to_cluster_map(csv_path: pathlib.Path) -> dict:
    """Parses list.csv to map CHAINID -> CLUSTER."""
    mapping = {}
    if not csv_path.exists():
        print(f"Warning: Metadata file {csv_path} not found.")
        return mapping

    print(f"Loading metadata from {csv_path}...")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Map "5naf_A" -> "ClusterID"
            if "CHAINID" in row and "CLUSTER" in row:
                mapping[row["CHAINID"]] = row["CLUSTER"]
    return mapping

def determine_split(name, chain_map, valid_clusters, test_clusters):
    """
    Decides split based on cluster ID.
    Priority: Test > Valid > Train
    """
    # 1. Lookup the cluster for this specific chain
    cluster_id = chain_map.get(name)

    # 2. If metadata is missing for this chain, default to Train
    # (Or you could default to 'valid' to be safe against data leakage)
    if not cluster_id:
        return "train"

    # 3. Check splits
    if cluster_id in test_clusters:
        return "test"
    if cluster_id in valid_clusters:
        return "valid"
    
    return "train"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=pathlib.Path, required=True,
                        help="Path to combined .array_record file")
    parser.add_argument("--output_index", type=pathlib.Path, default=pathlib.Path("pdb_2021aug02.index.json"))
    parser.add_argument("--cluster_dir", type=pathlib.Path, required=True,
                        help="Directory containing valid_clusters.txt and list.csv")
    
    args = parser.parse_args()
    
    # Paths
    metadata_csv = args.cluster_dir / "list.csv"
    valid_file = args.cluster_dir / "valid_clusters.txt"
    test_file = args.cluster_dir / "test_clusters.txt"

    # Load Metadata
    valid_clusters = load_cluster_file(valid_file)
    test_clusters = load_cluster_file(test_file)
    chain_to_cluster = load_chain_to_cluster_map(metadata_csv)

    print(f"Stats: {len(valid_clusters)} valid clusters, {len(test_clusters)} test clusters.")

    # Open Reader
    if not args.input_file.exists():
        print(f"Error: {args.input_file} not found.")
        exit(1)

    reader = array_record.ArrayRecordReader(str(args.input_file))
    num_records = reader.num_records()
    print(f"Scanning {num_records} records...")

    index_data = {}
    counts = {"train": 0, "valid": 0, "test": 0}

    # Iterate sequentially (much faster than random access)
    for i in tqdm.tqdm(range(num_records)):
        # Read the raw bytes at index i
        record_bytes = reader.read([i])[0]
        
        # Unpack
        record = msgpack.unpackb(record_bytes, raw=False)
        
        # Get ID (handle bytes vs string)
        name_raw = record.get("protein_id")
        if isinstance(name_raw, bytes):
            name = name_raw.decode("utf-8")
        else:
            name = str(name_raw)

        # Determine split
        split = determine_split(name, chain_to_cluster, valid_clusters, test_clusters)
        
        counts[split] += 1
        
        if name not in index_data:
            index_data[name] = {"idx": [], "set": split}
        
        index_data[name]["idx"].append(i)

    reader.close()

    print(f"\nFinal Counts: {counts}")
    print(f"Writing index to {args.output_index}...")
    
    with open(args.output_index, "w") as f:
        json.dump(index_data, f, indent=2)

if __name__ == "__main__":
    main()
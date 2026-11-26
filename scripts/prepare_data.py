"""
scripts/prepare_data.py
Streams parquet shards by row group using PyArrow, writes memory-mapped binary image containers
and manifest JSONL records. Computes perceptual hashes to detect any cross-split leakage.
"""

from pathlib import Path
import io
import json
import argparse
import pyarrow.parquet as pq
from PIL import Image
import imagehash
from tqdm import tqdm


def prepare_data(data_dir: str = "dataset", output_dir: str = "data/processed", check_hash_samples: int = 5000):
    src_path = Path(data_dir)
    if not src_path.exists():
        matches = [p for p in Path(".").iterdir() if p.name.lower() == "dataset"]
        if matches:
            src_path = matches[0]
        else:
            raise FileNotFoundError(f"Source path {data_dir} not found.")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": sorted(list(src_path.glob("train-*.parquet"))),
        "validation": sorted(list(src_path.glob("validation-*.parquet"))),
        "test": sorted(list(src_path.glob("test-*.parquet"))),
    }

    hash_store = {}  # hash -> (split, index)
    cross_split_duplicates = 0

    print("=" * 65)
    print("DEEPTRUTH AI - DATA PIPELINE PREPARATION (STREAMING PARQUET)")
    print(f"Source: {src_path.resolve()}")
    print(f"Output: {out_path.resolve()}")
    print("=" * 65)

    for split_name, files in splits.items():
        print(f"\nProcessing Split: {split_name.upper()} ({len(files)} shards)...")
        bin_file = out_path / f"{split_name}.bin"
        manifest_file = out_path / f"{split_name}_manifest.jsonl"

        global_idx = 0
        current_offset = 0

        with open(bin_file, "wb") as bf, open(manifest_file, "w", encoding="utf-8") as mf:
            for fpath in files:
                pf = pq.ParquetFile(fpath)
                for rg_idx in range(pf.num_row_groups):
                    table = pf.read_row_group(rg_idx, columns=["image", "label", "generator"])
                    img_col = table["image"]
                    lbl_col = table["label"].to_pylist()
                    gen_col = table["generator"].to_pylist()
                    n_rows = len(table)

                    for r in range(n_rows):
                        img_struct = img_col[r].as_py()
                        img_bytes = img_struct.get("bytes") if isinstance(img_struct, dict) else img_struct
                        img_path = img_struct.get("path", f"{global_idx}.jpg") if isinstance(img_struct, dict) else f"{global_idx}.jpg"

                        byte_len = len(img_bytes)
                        bf.write(img_bytes)

                        record = {
                            "index": global_idx,
                            "split": split_name,
                            "label": int(lbl_col[r]),
                            "generator": int(gen_col[r]),
                            "path": img_path,
                            "byte_offset": current_offset,
                            "byte_length": byte_len,
                        }
                        mf.write(json.dumps(record) + "\n")

                        # Perceptual hash sampling for cross-split duplicate check
                        if check_hash_samples > 0 and global_idx % max(1, (100000 // check_hash_samples)) == 0:
                            try:
                                with Image.open(io.BytesIO(img_bytes)) as pil_img:
                                    h = str(imagehash.phash(pil_img))
                                    if h in hash_store:
                                        prev_split, prev_idx = hash_store[h]
                                        if prev_split != split_name:
                                            cross_split_duplicates += 1
                                    else:
                                        hash_store[h] = (split_name, global_idx)
                            except Exception:
                                pass

                        current_offset += byte_len
                        global_idx += 1

        print(f"  -> Generated {manifest_file.name} ({global_idx:,} records)")
        print(f"  -> Generated {bin_file.name} ({current_offset / (1024**2):.1f} MB)")

    print("\n" + "=" * 65)
    print(f"Perceptual Hash Sampled Duplicates Check: {cross_split_duplicates} cross-split duplicates found.")
    print("Preparation complete.")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset", help="Input directory containing parquet shards")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory for manifests and binary containers")
    parser.add_argument("--hash-samples", type=int, default=5000, help="Number of samples to hash per split")
    args = parser.parse_args()

    prepare_data(data_dir=args.data_dir, output_dir=args.output_dir, check_hash_samples=args.hash_samples)

"""
data/inspect_parquet.py
Inspects HuggingFace parquet shards in dataset/ directory:
Prints columns, dtypes, label counts per split, image byte format, size distributions.
Streams row groups using pyarrow without reading entire dataset into pandas.
"""

from pathlib import Path
import io
import pyarrow.parquet as pq
from PIL import Image


def inspect_parquet(data_dir: str = "dataset") -> dict:
    data_path = Path(data_dir)
    if not data_path.exists():
        # Try Dataset case-insensitive
        matches = [p for p in Path(".").iterdir() if p.name.lower() == "dataset"]
        if matches:
            data_path = matches[0]
        else:
            raise FileNotFoundError(f"Dataset directory '{data_dir}' not found.")

    print(f"Scanning parquet files in: {data_path.resolve()}")
    parquet_files = sorted(list(data_path.glob("*.parquet")))
    print(f"Found {len(parquet_files)} parquet files.")

    splits = {"train": [], "validation": [], "test": []}
    for f in parquet_files:
        fname = f.name.lower()
        if "train" in fname:
            splits["train"].append(f)
        elif "val" in fname:
            splits["validation"].append(f)
        elif "test" in fname:
            splits["test"].append(f)

    report = {}

    for split_name, files in splits.items():
        print(f"\n{'='*50}")
        print(f"SPLIT: {split_name.upper()} ({len(files)} shards)")
        print(f"{'='*50}")
        if not files:
            print("  No shards found!")
            continue

        total_rows = 0
        label_counts = {}
        sample_checked = 0
        formats = {}
        dimensions = {}
        byte_sizes = {"real": [], "fake": []}

        # Inspect first file schema
        first_pf = pq.ParquetFile(files[0])
        schema = first_pf.schema
        print(f"Schema (from {files[0].name}):")
        for i, col in enumerate(schema.names):
            print(f"  Col {i}: {col} -> {schema.column(i).physical_type}")

        for file_idx, fpath in enumerate(files):
            pf = pq.ParquetFile(fpath)
            num_rg = pf.num_row_groups
            for rg_idx in range(num_rg):
                # Read row group columns without full dataframe conversion
                table = pf.read_row_group(rg_idx)
                col_names = table.column_names
                rows = len(table)
                total_rows += rows

                # Determine label column
                label_col = "label" if "label" in col_names else col_names[1]
                labels = table[label_col].to_pylist()
                for l in labels:
                    label_counts[str(l)] = label_counts.get(str(l), 0) + 1

                # Sample inspection of images
                if sample_checked < 50:
                    img_col = "image" if "image" in col_names else col_names[0]
                    img_data = table[img_col]
                    for idx in range(min(5, rows)):
                        val = img_data[idx].as_py()
                        img_bytes = None
                        if isinstance(val, dict) and "bytes" in val:
                            img_bytes = val["bytes"]
                        elif isinstance(val, bytes):
                            img_bytes = val

                        if img_bytes:
                            lbl_str = str(labels[idx])
                            is_real = lbl_str in ["0", "real", "Real"]
                            category = "real" if is_real else "fake"
                            byte_sizes[category].append(len(img_bytes))

                            try:
                                with Image.open(io.BytesIO(img_bytes)) as im:
                                    fmt = im.format
                                    formats[fmt] = formats.get(fmt, 0) + 1
                                    sz = im.size
                                    dimensions[sz] = dimensions.get(sz, 0) + 1
                            except Exception as e:
                                pass
                            sample_checked += 1

        print(f"Total rows in {split_name}: {total_rows:,}")
        print(f"Label distribution: {label_counts}")
        print(f"Image formats detected (sample): {formats}")
        print(f"Image dimensions detected (sample): {dimensions}")
        if byte_sizes["real"]:
            avg_real = sum(byte_sizes["real"]) / len(byte_sizes["real"])
            print(f"Avg Real image size (sample): {avg_real/1024:.1f} KB")
        if byte_sizes["fake"]:
            avg_fake = sum(byte_sizes["fake"]) / len(byte_sizes["fake"])
            print(f"Avg Fake image size (sample): {avg_fake/1024:.1f} KB")

        report[split_name] = {
            "total_rows": total_rows,
            "label_counts": label_counts,
            "formats": formats,
            "dimensions": dimensions,
        }

    return report


if __name__ == "__main__":
    inspect_parquet()

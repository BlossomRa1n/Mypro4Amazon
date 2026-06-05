"""
Download All_Beauty raw data (review JSONL + meta JSONL) from HF Mirror.
With retry, resume, and long timeout support.
"""
import os
import sys
import requests
import argparse
from tqdm import tqdm
import time


HF_MIRROR = "https://hf-mirror.com/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main"

FILES = {
    "review": "raw/review_categories/All_Beauty.jsonl",
    "meta_jsonl": "raw/meta_categories/meta_All_Beauty.jsonl",
    "meta_parquet": "raw_meta_All_Beauty/full-00000-of-00001.parquet",
}


def download_file(url, dest, desc, max_retries=5):
    """Stream download with progress bar and retry+resume support."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    # Check existing partial
    existing_size = os.path.getsize(dest) if os.path.exists(dest) else 0

    for attempt in range(max_retries):
        try:
            headers = {}
            if existing_size > 0:
                headers['Range'] = f'bytes={existing_size}-'
                print(f"  [RESUME] from byte {existing_size:,}")

            # Use session for connection pooling
            session = requests.Session()
            resp = session.get(url, stream=True, timeout=(60, 600), headers=headers)

            if resp.status_code == 416:
                # Range not satisfiable — file already complete
                print(f"  [SKIP] {desc} already fully downloaded")
                return dest

            resp.raise_for_status()

            # Handle resume: 206 Partial Content vs 200 OK (server ignored range)
            if resp.status_code == 206:
                mode = 'ab'
                downloaded = existing_size
            else:
                mode = 'wb'
                downloaded = 0
                existing_size = 0

            total = int(resp.headers.get("content-length", 0))
            total_display = total + existing_size if total else 0

            with open(dest, mode) as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=desc,
                initial=0
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
                        downloaded += len(chunk)

            print(f"  ✓ {desc} complete ({downloaded + existing_size:,} bytes)")
            return dest

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            print(f"  ✗ Attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                wait = (attempt + 1) * 30
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
                existing_size = os.path.getsize(dest) if os.path.exists(dest) else 0
            else:
                raise


def main():
    parser = argparse.ArgumentParser(description="Download All_Beauty raw data")
    parser.add_argument("--data-dir", default="amazon_reviews")
    parser.add_argument("--meta-format", choices=["jsonl", "parquet", "both"], default="jsonl")
    parser.add_argument("--skip-review", action="store_true", help="Skip review, only download meta")
    parser.add_argument("--skip-meta", action="store_true", help="Skip meta, only download review")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    # 1. Review JSONL (larger file: ~200-500 MB)
    if not args.skip_review:
        review_url = f"{HF_MIRROR}/{FILES['review']}"
        review_dest = os.path.join(data_dir, "All_Beauty_reviews.jsonl")
        if os.path.exists(review_dest):
            size = os.path.getsize(review_dest)
            # Quick sanity check: > 100KB is likely valid
            if size > 100_000:
                print(f"[SKIP] {review_dest} already exists ({size:,} bytes)")
            else:
                print(f"[RE-DOWNLOAD] {review_dest} too small ({size:,} bytes), re-downloading")
                os.remove(review_dest)
                download_file(review_url, review_dest, "All_Beauty reviews")
        else:
            print(f"[DOWNLOAD] {review_url}")
            download_file(review_url, review_dest, "All_Beauty reviews")

    # 2. Meta JSONL (smaller: ~10-50 MB)
    if not args.skip_meta:
        if args.meta_format in ("jsonl", "both"):
            meta_url = f"{HF_MIRROR}/{FILES['meta_jsonl']}"
            meta_dest = os.path.join(data_dir, "All_Beauty_meta.jsonl")
            if os.path.exists(meta_dest) and os.path.getsize(meta_dest) > 10_000:
                print(f"[SKIP] {meta_dest} already exists ({os.path.getsize(meta_dest):,} bytes)")
            else:
                print(f"[DOWNLOAD] {meta_url}")
                download_file(meta_url, meta_dest, "All_Beauty meta JSONL")

    print("\n=== Download complete ===")
    print(f"Files in {data_dir}:")
    for f in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, f)
        if os.path.isfile(path):
            print(f"  {f}  ({os.path.getsize(path):,} bytes)")


if __name__ == "__main__":
    main()

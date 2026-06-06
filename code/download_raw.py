"""
Download Amazon Reviews 2023 raw data (review JSONL + meta JSONL) from HF Mirror.
With retry, resume, long timeout support, and configurable category.

Usage:
  # Download single category (default: All_Beauty)
  python code/download_raw.py

  # Download multiple categories
  python code/download_raw.py --categories Video_Games,CDs_and_Vinyl

  # Download to specific data directory
  python code/download_raw.py --category Video_Games --data-dir /root/autodl-tmp/amazon_data

  # Preview without downloading
  python code/download_raw.py --category Video_Games --dry-run

  # Match your EXT_CATEGORIES config
  python code/download_raw.py --categories All_Beauty,Video_Games,CDs_and_Vinyl
"""
import os
import sys
import requests
import argparse
from tqdm import tqdm
import time


HF_MIRROR = "https://hf-mirror.com/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main"


def get_category_urls(category):
    """Build HF Mirror URLs and local paths for a given category."""
    return {
        "review": {
            "url": f"{HF_MIRROR}/raw/review_categories/{category}.jsonl",
            "dest_subpath": os.path.join("raw", "review_categories", f"{category}.jsonl"),
        },
        "meta_jsonl": {
            "url": f"{HF_MIRROR}/raw/meta_categories/meta_{category}.jsonl",
            "dest_subpath": os.path.join("raw", "meta_categories", f"meta_{category}.jsonl"),
        },
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


def download_category(data_dir, category, skip_review=False, skip_meta=False,
                      meta_format="jsonl", dry_run=False):
    """Download review and/or meta JSONL for one category."""
    urls = get_category_urls(category)

    print(f"\n{'='*60}")
    print(f"  Category: {category}")
    print(f"  Data dir: {data_dir}")
    print(f"{'='*60}")

    # 1. Review JSONL
    if not skip_review:
        url = urls["review"]["url"]
        dest = os.path.join(data_dir, urls["review"]["dest_subpath"])
        if dry_run:
            print(f"  [DRY-RUN] Would download review: {url}")
            print(f"            → {dest}")
        elif os.path.exists(dest):
            size = os.path.getsize(dest)
            if size > 100_000:
                print(f"  [SKIP] Review already exists: {dest} ({size:,} bytes)")
            else:
                print(f"  [RE-DOWNLOAD] {dest} too small ({size:,} bytes)")
                os.remove(dest)
                download_file(url, dest, f"{category} reviews")
        else:
            print(f"  [DOWNLOAD] {url}")
            download_file(url, dest, f"{category} reviews")

    # 2. Meta JSONL
    if not skip_meta and meta_format in ("jsonl", "both"):
        url = urls["meta_jsonl"]["url"]
        dest = os.path.join(data_dir, urls["meta_jsonl"]["dest_subpath"])
        if dry_run:
            print(f"  [DRY-RUN] Would download meta: {url}")
            print(f"            → {dest}")
        elif os.path.exists(dest) and os.path.getsize(dest) > 10_000:
            print(f"  [SKIP] Meta already exists: {dest} ({os.path.getsize(dest):,} bytes)")
        else:
            print(f"  [DOWNLOAD] {url}")
            download_file(url, dest, f"{category} meta")


def main():
    parser = argparse.ArgumentParser(
        description="Download Amazon Reviews 2023 raw JSONL data from HF Mirror"
    )
    parser.add_argument("--data-dir", default="amazon_reviews",
                        help="Root data directory (default: amazon_reviews)")
    parser.add_argument("--category", default=None,
                        help="Single category to download (default: All_Beauty)")
    parser.add_argument("--categories", default=None,
                        help="Comma-separated list of categories, e.g. 'Video_Games,CDs_and_Vinyl'")
    parser.add_argument("--meta-format", choices=["jsonl", "parquet", "both"], default="jsonl")
    parser.add_argument("--skip-review", action="store_true",
                        help="Skip review, only download meta")
    parser.add_argument("--skip-meta", action="store_true",
                        help="Skip meta, only download review")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print download URLs and paths without downloading")
    args = parser.parse_args()

    # Resolve categories list
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    elif args.category:
        categories = [args.category]
    else:
        categories = ["All_Beauty"]

    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    for cat in categories:
        download_category(
            data_dir=data_dir,
            category=cat,
            skip_review=args.skip_review,
            skip_meta=args.skip_meta,
            meta_format=args.meta_format,
            dry_run=args.dry_run,
        )

    if not args.dry_run:
        print("\n=== Download complete ===")
        print(f"Files in {data_dir}:")
        _list_dir(data_dir)


def _list_dir(path, indent=""):
    """List directory contents with indentation."""
    for f in sorted(os.listdir(path)):
        full = os.path.join(path, f)
        if os.path.isfile(full):
            size = os.path.getsize(full)
            print(f"  {indent}{f}  ({size:,} bytes)")
        elif os.path.isdir(full):
            print(f"  {indent}{f}/")
            _list_dir(full, indent + "    ")


if __name__ == "__main__":
    main()

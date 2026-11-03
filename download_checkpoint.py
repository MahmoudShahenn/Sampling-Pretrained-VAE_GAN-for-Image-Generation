#!/usr/bin/env python3
"""High-performance parallel, resumable downloader for the original VAE-GAN CelebA checkpoint.

Uses concurrent HTTP range requests to download from GitHub Git LFS / AWS S3 at maximum speed,
with progress tracking, automatic retries, and full SHA-256 integrity verification.
"""

import os
import sys
import time
import json
import hashlib
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BATCH_URL = "https://github.com/rufinv/VAE-GAN-CelebA.git/info/lfs/objects/batch"
EXPECTED_SIZE = 554_656_600
EXPECTED_SHA256 = "f97e76a9302481d45dca45a523904c91a94c8e39b94922c0f949df02e5c4948b"

OUTPUT_FILE = "vaegan_celeba.ckpt"
PART_FILE = OUTPUT_FILE + ".part"
CHUNK_SIZE = 4 * 1024 * 1024  # 4 MB per chunk
NUM_WORKERS = 8


def get_fresh_download_url(retries=5, delay=3):
    """Retrieve an authenticated S3 download URL via Git LFS batch API."""
    payload = {
        "operation": "download",
        "transfers": ["basic"],
        "objects": [{
            "oid": EXPECTED_SHA256,
            "size": EXPECTED_SIZE
        }]
    }
    request = urllib.request.Request(
        BATCH_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/vnd.git-lfs+json",
            "Content-Type": "application/vnd.git-lfs+json",
            "User-Agent": "git-lfs/3.0.0"
        }
    )

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
            obj = result["objects"][0]
            if "error" in obj:
                raise RuntimeError(f"LFS API error: {obj['error']}")
            url = obj["actions"]["download"]["href"]
            headers = obj["actions"]["download"].get("header", {})
            return url, headers
        except Exception as e:
            if attempt == retries:
                raise
            print(f"[!] Git LFS batch request attempt {attempt} failed ({e}). Retrying in {delay}s...")
            time.sleep(delay)


def verify_sha256(filepath):
    """Verify SHA-256 checksum of the completed file."""
    print(f"\n[*] Calculating SHA-256 integrity checksum for {filepath}...")
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            hasher.update(chunk)
    actual_hash = hasher.hexdigest()
    print(f"[*] Computed SHA-256: {actual_hash}")
    print(f"[*] Expected SHA-256: {EXPECTED_SHA256}")
    return actual_hash.lower() == EXPECTED_SHA256.lower()


def download_chunk(url, headers, start, end, out_filepath, file_lock, progress_tracker):
    """Download a specific byte range and write it directly to the target file offset."""
    req_headers = dict(headers)
    req_headers["Range"] = f"bytes={start}-{end}"
    max_retries = 10

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers=req_headers)
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = resp.read()

            expected_len = end - start + 1
            if len(data) != expected_len:
                raise ValueError(f"Incomplete chunk received: {len(data)} != {expected_len}")

            with file_lock:
                with open(out_filepath, "r+b") as f:
                    f.seek(start)
                    f.write(data)

            progress_tracker.update(len(data))
            return True

        except Exception as e:
            if attempt == max_retries:
                print(f"\n[!] Failed to download chunk {start}-{end} after {max_retries} attempts: {e}")
                raise
            time.sleep(1 + attempt)


class ProgressTracker:
    def __init__(self, already_downloaded, total_size):
        self.lock = threading.Lock()
        self.downloaded = already_downloaded
        self.total = total_size
        self.start_time = time.time()
        self.session_downloaded = 0

    def update(self, num_bytes):
        with self.lock:
            self.downloaded += num_bytes
            self.session_downloaded += num_bytes
            elapsed = time.time() - self.start_time
            speed = (self.session_downloaded / (1024 * 1024)) / (elapsed + 1e-6)
            pct = (self.downloaded / self.total) * 100
            print(
                f"\rProgress: {self.downloaded / (1024*1024):.1f} / {self.total / (1024*1024):.1f} MB "
                f"({pct:.1f}%) [{speed:.2f} MB/s] ",
                end="", flush=True
            )


def download_checkpoint():
    # 1. Check if complete verified file already exists
    if os.path.isfile(OUTPUT_FILE):
        if os.path.getsize(OUTPUT_FILE) == EXPECTED_SIZE:
            print(f"[+] Found {OUTPUT_FILE} ({EXPECTED_SIZE:,} bytes). Verifying...")
            if verify_sha256(OUTPUT_FILE):
                print("[+] Checkpoint is already fully downloaded and verified!")
                return True
            else:
                print("[!] Existing file hash mismatch. Re-downloading...")
                os.remove(OUTPUT_FILE)

    # 2. Inspect existing partial download
    start_offset = 0
    if os.path.isfile(PART_FILE):
        current_part_size = os.path.getsize(PART_FILE)
        if current_part_size >= EXPECTED_SIZE:
            if current_part_size == EXPECTED_SIZE and verify_sha256(PART_FILE):
                os.replace(PART_FILE, OUTPUT_FILE)
                print(f"[+] Checkpoint verified and moved to {OUTPUT_FILE}!")
                return True
            else:
                print("[!] Partial file invalid size/hash. Restarting...")
                os.remove(PART_FILE)
                current_part_size = 0
        else:
            # Align start_offset to CHUNK_SIZE
            start_offset = (current_part_size // CHUNK_SIZE) * CHUNK_SIZE
            print(f"[*] Resuming from byte {start_offset:,} ({start_offset / (1024*1024):.1f} MB)...")

    # 3. Pre-allocate or open partial file
    if not os.path.isfile(PART_FILE):
        with open(PART_FILE, "wb") as f:
            f.seek(EXPECTED_SIZE - 1)
            f.write(b"\0")
    else:
        if os.path.getsize(PART_FILE) != EXPECTED_SIZE:
            with open(PART_FILE, "a+b") as f:
                f.seek(EXPECTED_SIZE - 1)
                f.write(b"\0")

    # 4. Generate chunks to download
    chunks = []
    offset = start_offset
    while offset < EXPECTED_SIZE:
        chunk_end = min(offset + CHUNK_SIZE - 1, EXPECTED_SIZE - 1)
        chunks.append((offset, chunk_end))
        offset = chunk_end + 1

    print(f"[*] Downloading {len(chunks)} chunks using {NUM_WORKERS} parallel threads...")
    url, headers = get_fresh_download_url()

    file_lock = threading.Lock()
    tracker = ProgressTracker(start_offset, EXPECTED_SIZE)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(
                download_chunk, url, headers, start, end, PART_FILE, file_lock, tracker
            ): (start, end)
            for start, end in chunks
        }
        for future in as_completed(futures):
            future.result()

    print("\n[+] All chunks downloaded successfully!")

    # 5. Verify integrity
    if verify_sha256(PART_FILE):
        os.replace(PART_FILE, OUTPUT_FILE)
        print(f"[+] SUCCESS: Pretrained checkpoint verified and saved to '{OUTPUT_FILE}'!")
        return True
    else:
        print("[!] ERROR: Checkpoint failed SHA-256 verification.")
        return False


if __name__ == "__main__":
    success = download_checkpoint()
    sys.exit(0 if success else 1)

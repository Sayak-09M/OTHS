#!/usr/bin/env python3
"""Download the India/Nepal priority subset of MIDV-LAIT from its author FTP."""

from __future__ import annotations

import argparse
import ftplib
import os
import posixpath
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

HOST = "smartengines.com"
REMOTE_ROOT = "/midv-lait/dataset"
DEFAULT_PREFIXES = (
    "ind_aadhaar_bengali_",
    "ind_aadhaar_devanagari_",
    "ind_aadhaar_gujarati_",
    "ind_aadhaar_gurmukhi_",
    "ind_aadhaar_kannada_",
    "ind_aadhaar_malayalam_",
    "ind_aadhaar_odia_",
    "ind_aadhaar_tamil_",
    "ind_aadhaar_telugu_",
    "npl_id_",
)


def connect() -> ftplib.FTP:
    ftp = ftplib.FTP(HOST, timeout=60)
    ftp.login()
    ftp.set_pasv(True)
    return ftp


def safe_local_path(destination: Path, remote_path: str) -> Path:
    relative = PurePosixPath(remote_path).relative_to(PurePosixPath(REMOTE_ROOT))
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"unsafe remote path: {remote_path}")
    return destination.joinpath(*relative.parts)


def selected_directories(ftp: ftplib.FTP, prefixes: tuple[str, ...]) -> list[str]:
    directories: list[str] = []
    for entry in ftp.nlst(REMOTE_ROOT):
        name = posixpath.basename(entry.rstrip("/"))
        if name.startswith(prefixes):
            directories.append(posixpath.join(REMOTE_ROOT, name))
    return sorted(directories)


def walk_files(ftp: ftplib.FTP, remote_directory: str) -> list[tuple[str, int]]:
    files: list[tuple[str, int]] = []
    pending = [remote_directory]
    ftp.voidcmd("TYPE I")
    while pending:
        current = pending.pop()
        for entry in ftp.nlst(current):
            remote_path = (
                entry if entry.startswith("/") else posixpath.join(current, entry)
            )
            try:
                size = ftp.size(remote_path)
            except ftplib.error_perm:
                pending.append(remote_path)
            else:
                files.append((remote_path, int(size or 0)))
    return sorted(files)


def download_file(
    ftp: ftplib.FTP, remote_path: str, expected_size: int, local_path: Path
) -> ftplib.FTP:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = local_path.with_name(local_path.name + ".part")

    if local_path.exists() and local_path.stat().st_size == expected_size:
        return ftp

    for attempt in range(1, 6):
        offset = partial_path.stat().st_size if partial_path.exists() else 0
        try:
            with partial_path.open("ab" if offset else "wb") as output:
                ftp.retrbinary(
                    f"RETR {remote_path}",
                    output.write,
                    blocksize=1024 * 1024,
                    rest=offset or None,
                )
            actual_size = partial_path.stat().st_size
            if expected_size and actual_size != expected_size:
                raise OSError(
                    f"size mismatch for {remote_path}: {actual_size} != {expected_size}"
                )
            os.replace(partial_path, local_path)
            return ftp
        except (OSError, EOFError, ftplib.Error) as error:
            if attempt == 5:
                raise
            print(f"retry {attempt}/5 for {remote_path}: {error}", flush=True)
            time.sleep(attempt * 2)
            try:
                ftp.close()
            finally:
                ftp = connect()
    return ftp


def download_directory(destination: Path, remote_directory: str) -> tuple[int, int]:
    ftp = connect()
    completed_files = 0
    completed_bytes = 0
    try:
        for remote_path, expected_size in walk_files(ftp, remote_directory):
            local_path = safe_local_path(destination, remote_path)
            ftp = download_file(ftp, remote_path, expected_size, local_path)
            completed_files += 1
            completed_bytes += expected_size
        return completed_files, completed_bytes
    finally:
        try:
            ftp.quit()
        except (OSError, EOFError, ftplib.Error):
            ftp.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--prefix", action="append", dest="prefixes")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    prefixes = tuple(args.prefixes or DEFAULT_PREFIXES)
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    with connect() as ftp:
        directories = selected_directories(ftp, prefixes)
    print(f"selected {len(directories)} document directories", flush=True)

    completed_files = 0
    completed_bytes = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(download_directory, destination, directory): directory
            for directory in directories
        }
        for directory_index, future in enumerate(as_completed(futures), start=1):
            file_count, byte_count = future.result()
            completed_files += file_count
            completed_bytes += byte_count
            print(
                f"downloaded directory {directory_index}/{len(directories)}; "
                f"{completed_files} files, {completed_bytes / (1024 ** 2):.1f} MiB",
                flush=True,
            )


if __name__ == "__main__":
    main()

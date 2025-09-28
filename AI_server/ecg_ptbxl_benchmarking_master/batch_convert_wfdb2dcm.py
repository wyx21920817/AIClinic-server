#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel batch conversion from WFDB to DICOM (flat directory layout like ICBEB records100)

Example:
  python batch_convert_wfdb2dcm.py \
    --records-dir data/ICBEB/records100 \
    --out-dir data/ICBEB/dcm_records100 \
    --workers 8
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def discover_records(records_dir: Path):
    """Return list of (record_name, wfdb_dir)."""
    hea_files = sorted(records_dir.glob('*.hea'))
    tasks = []
    for hea in hea_files:
        base = hea.stem  # e.g., 100, 1001
        tasks.append((base, records_dir))
    return tasks


def convert_one(python_exe: str, script_path: Path, record_name: str, wfdb_dir: Path, out_path: Path, retries: int = 1) -> tuple:
    """Invoke wfdb2dcm.py for one record, return (record_name, True/False, msg)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [python_exe, str(script_path), record_name, str(wfdb_dir), str(out_path)]
    attempt = 0
    last_err = ''
    while attempt <= retries:
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if res.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
                return record_name, True, ''
            last_err = res.stdout
        except Exception as e:
            last_err = str(e)
        attempt += 1
    return record_name, False, last_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--records-dir', required=True, help='Flat directory containing .hea/.dat, e.g., data/ICBEB/records100')
    parser.add_argument('--out-dir', required=True, help='Output DICOM directory, e.g., data/ICBEB/dcm_records100')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument('--python', default=sys.executable, help='Python interpreter path, default to current interpreter')
    parser.add_argument('--script', default='wfdb2dcm.py', help='Converter script path, default wfdb2dcm.py')
    parser.add_argument('--retries', type=int, default=1, help='Retry times on failure')
    args = parser.parse_args()

    records_dir = Path(args.records_dir)
    out_dir = Path(args.out_dir)
    script_path = Path(args.script)

    if not records_dir.exists():
        print(f"❌ Records directory not found: {records_dir}")
        sys.exit(1)
    if not script_path.exists():
        print(f"❌ Converter script not found: {script_path}")
        sys.exit(1)

    tasks = discover_records(records_dir)
    total = len(tasks)
    print(f"🔎 Found {total} records (.hea) to convert")
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    fail = 0
    futures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for record_name, wfdb_dir in tasks:
            out_path = out_dir / f"{record_name}.dcm"
            futures.append(
                ex.submit(convert_one, args.python, script_path, record_name, wfdb_dir, out_path, args.retries)
            )

        for i, fut in enumerate(as_completed(futures), start=1):
            record_name, success, msg = fut.result()
            if success:
                ok += 1
                if i % 50 == 0 or ok <= 5:
                    print(f"✅ [{i}/{total}] {record_name} converted")
            else:
                fail += 1
                print(f"⚠️  Convert failed: {record_name}\n{msg[:300]}\n---")

    print(f"\n📊 Done: success {ok} / {total}, failed {fail}")
    if fail == 0:
        print(f"📁 Output directory: {out_dir}")


if __name__ == '__main__':
    main()



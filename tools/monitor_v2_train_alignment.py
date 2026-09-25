"""Supervise one explicitly supplied V2 alignment command; no restart/shutdown.

Example: python tools/monitor_v2_train_alignment.py --run-dir /system/run \
 --data-dir /data/pool --control-dir /system/control -- python run_v2_train_alignment.py ...
The control directory must be outside the initially empty training directory.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def latest_step(path):
    try:
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - 65536)); lines = stream.read().splitlines()
        for line in reversed(lines):
            try: return json.loads(line)
            except (ValueError, UnicodeDecodeError): continue
    except FileNotFoundError: pass
    return None


def filesystem(path, reserve):
    while not path.exists(): path = path.parent
    usage = shutil.disk_usage(path); fs = os.statvfs(path)
    return {'path':str(path), 'filesystem_device':path.stat().st_dev,
            'free_bytes':usage.free, 'used_bytes':usage.used, 'total_bytes':usage.total,
            'free_inodes':fs.f_favail, 'reserve_bytes':reserve,
            'reserve_met':usage.free >= reserve, 'inodes_available':fs.f_favail >= 16}


def safe(path):
    path = Path(path).absolute()
    if '..' in path.parts or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('unsafe/symlink path')
    return path


def snapshot(run, data, process, started, gpu=True):
    now = time.monotonic()
    gpu_info = {'available': False}
    executable = shutil.which('nvidia-smi') if gpu else None
    if executable:
        try:
            probe = subprocess.run([executable, '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
            gpu_info = {'available': probe.returncode == 0, 'returncode': probe.returncode, 'rows': probe.stdout.strip().splitlines(), 'error': probe.stderr.strip()}
        except (OSError, subprocess.TimeoutExpired) as error: gpu_info = {'available': False, 'error': str(error)}
    return {'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'elapsed_seconds': now - started,
            'pid': process.pid, 'returncode': process.poll(),
            'filesystems': {'system':filesystem(run,1536*1024**2), 'data':filesystem(data,2*1024**3)},
            'gpu': gpu_info, 'latest_steps': {str(path.relative_to(run)): latest_step(path) for path in sorted(run.glob('seed_*/raw_v2_train/steps.jsonl'))},
            'completed_marker': (run / 'COMPLETED.json').exists(), 'failed_marker': (run / 'FAILED.json').exists()}


def supervise(args):
    run = safe(args.run_dir); control = safe(args.control_dir); data = safe(args.data_dir)
    if control == run or control.is_relative_to(run) or run.is_relative_to(control): raise ValueError('control and training directories must be separate, not nested')
    if not data.is_dir(): raise ValueError('existing pool/data directory required')
    if control.is_relative_to(data) or data.is_relative_to(control) or run.is_relative_to(data) or data.is_relative_to(run): raise ValueError('data/run/control roots must not overlap')
    if run.exists() and any(run.iterdir()): raise FileExistsError('training evidence directory is nonempty')
    if control.exists() and any(control.iterdir()): raise FileExistsError('supervisor evidence directory is nonempty')
    if float(args.interval) <= 0: raise ValueError('sampling interval must be positive')
    command = list(args.command)
    if command and command[0] == '--': command = command[1:]
    if not command: raise ValueError('an explicit child command is required')
    run.parent.mkdir(parents=True, exist_ok=True); control.mkdir(parents=True, exist_ok=True)
    atomic_json(control / 'launch.json', {'command': command, 'run_dir': str(run), 'data_dir':str(data), 'control_dir': str(control), 'interval_seconds': float(args.interval), 'automatic_restart': False, 'shutdown_allowed': False, 'monitor_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'dependencies':'Python standard library only', 'disk_monitoring':'observation only; runner owns per-stage fail-closed budgets'})
    started = time.monotonic(); process = None
    def record(value):
        atomic_json(control / 'status.json', value)
        with (control / 'status.jsonl').open('a') as stream: stream.write(json.dumps(value) + '\n')
    try:
        with (control / 'training.log').open('ab', buffering=0) as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            next_sample = started
            while True:
                code = process.poll(); now = time.monotonic()
                if code is not None or now >= next_sample:
                    value = snapshot(run, data, process, started, gpu=not getattr(args, 'no_gpu_probe', False))
                    value['status'] = 'running' if code is None else ('complete' if code == 0 and value['completed_marker'] and not value['failed_marker'] else 'failed')
                    record(value); next_sample = time.monotonic() + float(args.interval)
                    if code is not None:
                        atomic_json(control / ('COMPLETED.json' if value['status'] == 'complete' else 'FAILED.json'), value)
                        return 0 if value['status'] == 'complete' else (code if code and code > 0 else 1)
                # Exit detection is independent of the fifteen-minute sampling cadence.
                time.sleep(min(1.0, max(.01, next_sample - time.monotonic())))
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
        value = {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error), 'returncode': process.poll() if process is not None else None}
        record(value); atomic_json(control / 'FAILED.json', value)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True); parser.add_argument('--control-dir', required=True)
    parser.add_argument('--data-dir', required=True, help='existing new pool directory on data filesystem')
    parser.add_argument('--interval', type=float, default=900.)
    parser.add_argument('--no-gpu-probe', action='store_true', help='for CPU supervisor tests')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    return supervise(parser.parse_args(argv))

if __name__ == '__main__': raise SystemExit(main())

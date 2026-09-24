"""Supervise exactly one final training or test stage; never chain or retry.

Example: python tools/monitor_final_stage.py --stage training --run-dir /evidence/run \
 --control-dir /evidence/control -- python code/run_full_final_test100k.py train ...
The control directory must be outside the initially empty training directory.
"""
from __future__ import annotations
import argparse
import datetime
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


def snapshot(run, process, started, stage, gpu=True):
    now = time.monotonic()
    gpu_info = {'available': False}
    executable = shutil.which('nvidia-smi') if gpu else None
    if executable:
        try:
            probe = subprocess.run([executable, '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
            gpu_info = {'available': probe.returncode == 0, 'returncode': probe.returncode, 'rows': probe.stdout.strip().splitlines(), 'error': probe.stderr.strip()}
        except (OSError, subprocess.TimeoutExpired) as error: gpu_info = {'available': False, 'error': str(error)}
    usage = shutil.disk_usage(run if run.exists() else run.parent)
    return {'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'elapsed_seconds': now - started,
            'pid': process.pid, 'returncode': process.poll(), 'disk': {'free_bytes': usage.free, 'used_bytes': usage.used, 'total_bytes': usage.total},
            'gpu': gpu_info, 'latest_steps': {str(path.relative_to(run)): latest_step(path) for path in sorted(run.glob('*/steps.jsonl'))},
            'completed_marker': (run / ('TRAINING_COMPLETED.json' if stage == 'training' else 'TEST_COMPLETED.json')).exists(), 'stage': stage, 'failed_marker': (run / 'FAILED.json').exists()}


def supervise(args):
    if args.stage not in ('training', 'test'): raise ValueError('stage must be training or test')
    run = Path(args.run_dir).resolve(); control = Path(args.control_dir).resolve()
    if control == run or control.is_relative_to(run) or run.is_relative_to(control): raise ValueError('control and training directories must be separate, not nested')
    if run.exists() and any(run.iterdir()): raise FileExistsError('training evidence directory is nonempty')
    if control.exists() and any(control.iterdir()): raise FileExistsError('supervisor evidence directory is nonempty')
    if float(args.interval) <= 0: raise ValueError('sampling interval must be positive')
    command = list(args.command)
    if command and command[0] == '--': command = command[1:]
    if not command: raise ValueError('an explicit child command is required')
    run.parent.mkdir(parents=True, exist_ok=True); control.mkdir(parents=True, exist_ok=True)
    atomic_json(control / 'launch.json', {'command': command, 'run_dir': str(run), 'control_dir': str(control), 'interval_seconds': float(args.interval), 'stage': args.stage, 'automatic_restart': False, 'shutdown_allowed': False})
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
                    value = snapshot(run, process, started, args.stage, gpu=not getattr(args, 'no_gpu_probe', False))
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
    parser.add_argument('--stage', choices=['training', 'test'], required=True)
    parser.add_argument('--run-dir', required=True); parser.add_argument('--control-dir', required=True)
    parser.add_argument('--interval', type=float, default=900.)
    parser.add_argument('--no-gpu-probe', action='store_true', help='for CPU supervisor tests')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    return supervise(parser.parse_args(argv))

if __name__ == '__main__': raise SystemExit(main())

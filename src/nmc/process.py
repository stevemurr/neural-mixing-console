"""Stream a child process and enforce the configured host memory floor."""

import subprocess
import sys
import threading
from pathlib import Path


def _mem_available_gib() -> float:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024.0 * 1024.0)
    except OSError:
        pass
    return float("inf")


def run_logged(command, cwd, env, log_path, ram_floor):
    with log_path.open("w", buffering=1) as log:
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        stop = threading.Event()

        def watchdog():
            while not stop.wait(5):
                if proc.poll() is not None:
                    return
                available = _mem_available_gib()
                if available < ram_floor:
                    message = f"[watchdog] {available:.1f} GiB available, floor {ram_floor} GiB"
                    print(message, flush=True)
                    print(message, file=log, flush=True)
                    proc.terminate()
                    if not stop.wait(30) and proc.poll() is None:
                        proc.kill()
                    return

        monitor = threading.Thread(target=watchdog, daemon=True)
        if ram_floor:
            monitor.start()
        try:
            with proc.stdout as stream:
                for line in stream:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log.write(line)
            return proc.wait()
        finally:
            stop.set()
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            if ram_floor:
                monitor.join()

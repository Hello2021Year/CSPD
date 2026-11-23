"""Sample only the owned run's process tree; CPU percentages can exceed100%."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    import psutil
    p = argparse.ArgumentParser()
    p.add_argument('--parent', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    parent = psutil.Process(a.parent)
    tracked = {}
    with a.out.open('w', buffering=1) as stream:
        while parent.is_running():
            rows, alive = [], set()
            try:
                children = parent.children(recursive=True)
            except psutil.NoSuchProcess:
                break
            for child in children:
                if child.pid == os.getpid():
                    continue
                try:
                    key = (child.pid, child.create_time())
                    initialized = key in tracked
                    proc = tracked.setdefault(key, child)
                    cpu = proc.cpu_percent(None)
                    rows.append(dict(pid=proc.pid, ppid=proc.ppid(), name=proc.name(),
                        command=proc.cmdline(), cpu_percent=cpu if initialized else None,
                        rss_bytes=proc.memory_info().rss, threads=proc.num_threads()))
                    alive.add(key)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            tracked = {key: value for key, value in tracked.items() if key in alive}
            stream.write(json.dumps(dict(t=time.monotonic(), processes=rows))+'\n')
            time.sleep(1)


if __name__ == '__main__':
    main()

"""Run a command with a memory cap on its whole process tree (macOS ignores `ulimit -v`).

  python scripts/run_capped.py --cap-gb 10 --log data/run_global.log -- .venv/bin/python -u run.py ...

Every `--every` seconds, sums the resident memory (RSS) of the command and all its descendants via `ps`. If the total
exceeds the cap, the whole process group is terminated (SIGTERM, then SIGKILL after 10 s) and the reason is logged.
Summed RSS double-counts shared libraries, so the cap is conservative. The peak is logged on exit.
"""
import argparse
import os
import signal
import subprocess
import sys
import time


def tree_rss_kb(root: int) -> int:
    out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss="], capture_output=True, text=True).stdout
    children, rss = {}, {}
    for line in out.splitlines():
        pid, ppid, kb = (int(x) for x in line.split())
        children.setdefault(ppid, []).append(pid)
        rss[pid] = kb
    total, stack = 0, [root]
    while stack:
        pid = stack.pop()
        total += rss.get(pid, 0)
        stack.extend(children.get(pid, []))
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-gb", type=float, required=True)
    ap.add_argument("--every", type=float, default=5.0)
    ap.add_argument("--log", required=True)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    cap_kb, peak = args.cap_gb * 1024 ** 2, 0
    with open(args.log, "a", buffering=1) as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        log.write(f"memcap: pid={proc.pid} cap={args.cap_gb} GB\n")
        while proc.poll() is None:
            kb = tree_rss_kb(proc.pid)
            peak = max(peak, kb)
            if kb > cap_kb:
                log.write(f"memcap: KILLED - process tree RSS {kb / 1024 ** 2:.2f} GB > cap {args.cap_gb} GB\n")
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                break
            time.sleep(args.every)
        code = proc.wait()
        log.write(f"memcap: peak tree RSS {peak / 1024 ** 2:.2f} GB\nexit={code}\n")
    sys.exit(code)


if __name__ == "__main__":
    main()

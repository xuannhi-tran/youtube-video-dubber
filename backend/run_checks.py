"""Run every module's run_local_checks() (no GPU, no API keys, no network).

    python run_checks.py            # all modules
    python run_checks.py web llm    # only these

Each module runs in its own process, so one module's settings or stubs never
leak into another. Exit code 1 if any check fails.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def modules():
    return sorted(path.stem for path in HERE.glob("*.py")
                  if path.stem != "run_checks" and "def run_local_checks" in path.read_text(encoding="utf-8"))


def main():
    names = sys.argv[1:] or modules()
    failed = []
    for name in names:
        result = subprocess.run(
            [sys.executable, "-c", f"import {name}; {name}.run_local_checks()"],
            cwd=HERE, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        print(f"{'ok  ' if result.returncode == 0 else 'FAIL'} {name}", flush=True)
        if result.returncode != 0:
            failed.append(name)
            print("\n".join((result.stdout + result.stderr).strip().splitlines()[-20:]), flush=True)
    print(f"\n{len(names) - len(failed)}/{len(names)} modules passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

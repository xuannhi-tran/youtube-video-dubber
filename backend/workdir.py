"""Where one dubbing job keeps its files.

By default that is this folder, exactly as before. dub.py sets the DUB_WORKDIR
environment variable so every stage reads and writes inside the folder of one
video instead of next to the scripts.
"""

import os
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
WORKDIR = Path(os.environ.get("DUB_WORKDIR") or SCRIPT_DIR).resolve()


def replace_file(temporary, target, attempts=100, delay=0.1):
    """Move `temporary` over `target`; on Windows retry for a few seconds when it is refused.

    The move fails with "Access is denied" while another program (antivirus, a file preview,
    the other pipeline script) has the target open, which lasts a moment.
    """
    for attempt in range(attempts):
        try:
            temporary.replace(target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)


def run_local_checks():
    import tempfile

    class Flaky:
        """A path whose replace() is refused twice, like a file another program holds open."""
        def __init__(self, refusals):
            self.refusals, self.calls = refusals, 0

        def replace(self, target):
            self.calls += 1
            if self.calls <= self.refusals:
                raise PermissionError("[WinError 5] Access is denied")

    flaky = Flaky(2)
    replace_file(flaky, "x", delay=0)
    assert flaky.calls == 3
    stuck = Flaky(99)
    try:
        replace_file(stuck, "x", attempts=4, delay=0)
    except PermissionError:
        assert stuck.calls == 4
    else:
        raise AssertionError("a file that stays locked must still raise")
    with tempfile.TemporaryDirectory() as folder:
        source, target = Path(folder) / "a.tmp", Path(folder) / "a"
        source.write_text("new")
        target.write_text("old")
        replace_file(source, target)
        assert target.read_text() == "new" and not source.exists()


if __name__ == "__main__":
    run_local_checks()
    print("workdir local checks passed")

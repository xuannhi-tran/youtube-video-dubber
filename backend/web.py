"""A small local web page for dub.py: paste a link, get the dubbed video.

    python web.py                 # then open http://127.0.0.1:8765
    python web.py --port 9000 --no-browser

It only listens on this computer (127.0.0.1) and runs dub.py for you, one link
at a time; extra links wait in a queue. Needs no extra package. Everything
dub.py does (resume after an error, glossary, .env settings) works the same.
"""

import argparse
import itertools
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

SCRIPT_DIR = Path(__file__).resolve().parent
JOBS_DIR = SCRIPT_DIR / "dub_jobs_output"  # same folder dub.py uses
PAGE = SCRIPT_DIR / "web_ui.html"
DEFAULT_PORT = 8765
STAGE_COUNT = 5
LOG_LINES = 300
ALLOWED_HOSTS = {"127.0.0.1", "localhost"}
VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]+$")


# ---------------------------------------------------------
# Pure helpers (covered by run_local_checks)
# ---------------------------------------------------------

def validate_request(data):
    """Return (url, extra dub.py arguments); raise ValueError with a message for the page."""
    if not isinstance(data, dict):
        raise ValueError("Yêu cầu không hợp lệ.")
    url = str(data.get("url", "")).strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or len(url) > 500 or " " in url:
        raise ValueError("Link phải bắt đầu bằng http:// hoặc https://")
    arguments = []
    if data.get("subs"):
        arguments.append("--subs")
    volume = data.get("original_volume")
    if volume not in (None, ""):
        try:
            volume = float(volume)
        except (TypeError, ValueError):
            raise ValueError("Âm lượng gốc phải là số từ 0 đến 1.") from None
        if not 0 <= volume <= 1:
            raise ValueError("Âm lượng gốc phải từ 0 đến 1.")
        arguments += ["--original-volume", str(volume)]
    seconds = data.get("max_seconds")
    if seconds not in (None, ""):
        try:
            seconds = float(seconds)
        except (TypeError, ValueError):
            raise ValueError("Số giây thử phải là số dương.") from None
        if not seconds > 0:
            raise ValueError("Số giây thử phải là số dương.")
        arguments += ["--max-seconds", str(seconds)]
    return url, arguments


def read_progress(progress, line):
    """Update {'video_id', 'stage', 'stage_name'} from one line of dub.py output."""
    match = re.match(r"Video (\S+)\s*$", line)
    if match:
        progress["video_id"] = match.group(1)
    match = re.match(r"\[(\d+)/(\d+)\] (\w+):", line)
    if match and int(match.group(1)) >= progress["stage"]:
        progress["stage"], progress["stage_name"] = int(match.group(1)), match.group(3)
    return progress


def parse_range(header, size):
    """Return (start, end) inclusive for a 'bytes=a-b' header, or None when absent/invalid."""
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", (header or "").strip())
    if not match or size <= 0:
        return None
    first, last = match.groups()
    if first == "" and last == "":
        return None
    if first == "":
        count = int(last)
        return (max(0, size - count), size - 1) if count else None
    start = int(first)
    end = min(int(last), size - 1) if last else size - 1
    return (start, end) if start <= end and start < size else None


def host_allowed(header):
    host = (header or "").rsplit(":", 1)[0] if not (header or "").startswith("[") else ""
    return host in ALLOWED_HOSTS


def kill_tree(process):
    """Stop dub.py and the scripts it started."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def run_local_checks():
    url, args = validate_request({"url": " https://youtu.be/abc ", "subs": True,
                                  "original_volume": "0.05", "max_seconds": "120"})
    assert url == "https://youtu.be/abc"
    assert args == ["--subs", "--original-volume", "0.05", "--max-seconds", "120.0"]
    assert validate_request({"url": "https://x.com/v"})[1] == []
    for bad in (None, {}, {"url": "ftp://x"}, {"url": "-o evil"}, {"url": "https://a b"},
                {"url": "https://x.com", "original_volume": "2"},
                {"url": "https://x.com", "original_volume": "abc"},
                {"url": "https://x.com", "max_seconds": "0"},
                {"url": "https://x.com/" + "a" * 600}):
        try:
            validate_request(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")

    progress = {"video_id": None, "stage": 0, "stage_name": ""}
    for line in ("Video aB-1_x", "Job folder: x", "[1/5] download: d -- already done, skipping",
                 "[2/5] transcribe: t", "[3/5] speech: s -- at the same time"):
        read_progress(progress, line)
    assert progress == {"video_id": "aB-1_x", "stage": 3, "stage_name": "speech"}
    read_progress(progress, "[2/5] transcribe: again")  # never goes backwards
    assert progress["stage"] == 3

    assert parse_range("bytes=0-9", 100) == (0, 9)
    assert parse_range("bytes=90-", 100) == (90, 99)
    assert parse_range("bytes=-10", 100) == (90, 99)
    assert parse_range("bytes=50-999", 100) == (50, 99)
    assert parse_range("bytes=100-", 100) is None and parse_range("junk", 100) is None
    assert parse_range(None, 100) is None and parse_range("bytes=5-2", 100) is None

    assert host_allowed("127.0.0.1:8765") and host_allowed("localhost:1")
    assert not host_allowed("evil.com") and not host_allowed("evil.com:80") and not host_allowed(None)
    assert VIDEO_ID.match("aB-1_x") and not VIDEO_ID.match("../x")


# ---------------------------------------------------------
# Job queue
# ---------------------------------------------------------

class Jobs:
    """One worker thread runs the queued links one after another."""

    def __init__(self):
        self.lock = threading.Lock()
        self.ids = itertools.count(1)
        self.items = []  # newest last; status: waiting | running | done | failed | cancelled
        self.process = None
        threading.Thread(target=self.worker, daemon=True).start()

    def add(self, url, arguments):
        with self.lock:
            item = {"id": next(self.ids), "url": url, "arguments": arguments, "status": "waiting",
                    "video_id": None, "stage": 0, "stage_name": "", "log": [], "error": "",
                    "cancel": False, "started": None, "ended": None}
            self.items.append(item)
            return item["id"]

    def cancel(self, job_id):
        with self.lock:
            for item in self.items:
                if item["id"] != job_id:
                    continue
                if item["status"] == "waiting":
                    item["status"] = "cancelled"
                elif item["status"] == "running":
                    item["cancel"] = True
                    if self.process:
                        kill_tree(self.process)
                return True
        return False

    def snapshot(self):
        with self.lock:
            return [{key: value for key, value in item.items() if key not in ("arguments", "cancel")}
                    | {"log": item["log"][-40:]} for item in self.items[-20:]]

    def worker(self):
        while True:
            with self.lock:
                item = next((i for i in self.items if i["status"] == "waiting"), None)
                if item:
                    item["status"], item["started"] = "running", time.time()
            if item is None:
                time.sleep(0.5)
                continue
            self.run(item)

    def run(self, item):
        progress = {"video_id": None, "stage": 0, "stage_name": ""}
        command = [sys.executable, "-u", str(SCRIPT_DIR / "dub.py"), item["url"], *item["arguments"]]
        environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
        try:
            process = subprocess.Popen(
                command, cwd=SCRIPT_DIR, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                start_new_session=(os.name != "nt"),
            )
        except OSError as error:
            self.finish(item, "failed", f"Không chạy được dub.py: {error}")
            return
        with self.lock:
            self.process = process
        for raw in process.stdout:
            line = raw.rstrip()
            with self.lock:
                read_progress(progress, line)
                item.update(progress)
                item["log"].append(line[:300])
                del item["log"][:-LOG_LINES]
        code = process.wait()
        with self.lock:
            self.process = None
        video = JOBS_DIR / str(progress["video_id"]) / "dub_output" / "dubbed.mp4"
        if item["cancel"]:
            self.finish(item, "cancelled", "")
        elif code == 0 and progress["video_id"] and video.is_file():
            self.finish(item, "done", "")
        else:
            tail = [line for line in item["log"] if line.strip()][-3:]
            self.finish(item, "failed", " | ".join(tail) or f"dub.py dừng với mã {code}")

    def finish(self, item, status, error):
        with self.lock:
            item["status"], item["error"], item["ended"] = status, error, time.time()


def finished_videos():
    """Videos already on disk, newest first."""
    result = []
    if JOBS_DIR.is_dir():
        for folder in JOBS_DIR.iterdir():
            video = folder / "dub_output" / "dubbed.mp4"
            if VIDEO_ID.match(folder.name) and video.is_file():
                stat = video.stat()
                result.append({"video_id": folder.name, "size": stat.st_size, "time": stat.st_mtime})
    return sorted(result, key=lambda entry: entry["time"], reverse=True)


# ---------------------------------------------------------
# HTTP
# ---------------------------------------------------------

JOBS = None


class Handler(BaseHTTPRequestHandler):
    server_version = "dubweb"

    def log_message(self, *_):
        pass

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def allowed(self):
        if not host_allowed(self.headers.get("Host")):
            self.send_json({"error": "host not allowed"}, 403)
            return False
        return True

    def do_GET(self):
        if not self.allowed():
            return
        path = urlparse(self.path)
        if path.path == "/":
            body = PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path.path == "/api/state":
            self.send_json({"jobs": JOBS.snapshot(), "videos": finished_videos()})
        elif path.path.startswith("/video/"):
            self.send_video(path.path[len("/video/"):], "download" in path.query)
        else:
            self.send_json({"error": "not found"}, 404)

    def send_video(self, video_id, as_download):
        file = JOBS_DIR / video_id / "dub_output" / "dubbed.mp4"
        if not VIDEO_ID.match(video_id) or not file.is_file():
            self.send_json({"error": "not found"}, 404)
            return
        size = file.stat().st_size
        span = parse_range(self.headers.get("Range"), size)
        start, end = span if span else (0, size - 1)
        self.send_response(206 if span else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if span:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if as_download:
            self.send_header("Content-Disposition", f'attachment; filename="dubbed_{video_id}.mp4"')
        self.end_headers()
        try:
            with file.open("rb") as handle:
                handle.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = handle.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def read_json(self):
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).hostname not in ALLOWED_HOSTS:
            raise ValueError("origin not allowed")
        if "application/json" not in (self.headers.get("Content-Type") or ""):
            raise ValueError("JSON expected")
        length = int(self.headers.get("Content-Length") or 0)
        if length > 10_000:
            raise ValueError("request too large")
        return json.loads(self.rfile.read(length) or b"{}")

    def do_POST(self):
        if not self.allowed():
            return
        try:
            data = self.read_json()
        except (ValueError, json.JSONDecodeError) as error:
            self.send_json({"error": str(error)}, 400)
            return
        path = urlparse(self.path).path
        if path == "/api/jobs":
            try:
                url, arguments = validate_request(data)
            except ValueError as error:
                self.send_json({"error": str(error)}, 400)
                return
            self.send_json({"id": JOBS.add(url, arguments)})
        elif re.fullmatch(r"/api/jobs/\d+/cancel", path):
            ok = JOBS.cancel(int(path.split("/")[3]))
            self.send_json({"ok": ok}, 200 if ok else 404)
        else:
            self.send_json({"error": "not found"}, 404)


def main():
    global JOBS
    run_local_checks()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true", help="do not open the page automatically")
    args = parser.parse_args()
    JOBS = Jobs()
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as error:
        raise SystemExit(f"Cannot listen on port {args.port}: {error}. Try --port 9000.") from error
    address = f"http://127.0.0.1:{args.port}"
    print(f"Dubber web page: {address}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(address)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
        if JOBS.process:
            kill_tree(JOBS.process)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

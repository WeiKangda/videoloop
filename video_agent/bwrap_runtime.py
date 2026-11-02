"""
Sandbox without a Docker daemon: every command runs in bubblewrap.

It has DockerRuntime's interface and paths (/videos, /outputs, /testbed, /tmp), so
the agent and the memory code don't change. Files persist between commands because
those folders are bound from one per-run folder on the host; processes don't, so a
command left running in the background ends when its call returns. Only what is
listed in _argv is visible inside: the system folders, the Python environment and
the one video, not the dataset, and the environment holds no API keys.
"""

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from .docker_runtime import DockerRuntime
from .logging_utils import get_logger

logger = get_logger(__name__)

WRITABLE_DIRS = ("testbed", "outputs", "tmp", "models")
SYSTEM_DIRS = ("/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc")
# execve refuses a single argument over 128 KB; longer commands go through a script file.
MAX_INLINE_COMMAND = 100_000


class BwrapRuntime(DockerRuntime):
    """Runs commands in a bubblewrap sandbox instead of a Docker container."""

    video_mounted = True

    def __init__(
        self,
        video_path: Optional[str] = None,
        env_prefix: str = "",
        extra_ro_binds: Optional[List[str]] = None,
        network_disabled: bool = True,
        work_root: str = "",
        timeout: int = 300,
    ):
        # DockerRuntime.__init__ connects to the Docker daemon, so it is not called.
        self.video_path = os.path.abspath(video_path) if video_path else None
        self.env_prefix = env_prefix
        self.extra_ro_binds = extra_ro_binds or []
        self.network_disabled = network_disabled
        self.work_root = work_root
        self.timeout = timeout
        self.root: Optional[str] = None
        self.container = None
        self._script_count = 0

    def start(self) -> None:
        if self.root is not None:
            logger.warning("Sandbox already running")
            return
        bwrap = shutil.which("bwrap")
        if not bwrap:
            raise RuntimeError("bwrap is not installed")
        self._bwrap = bwrap
        if self.work_root:
            os.makedirs(self.work_root, exist_ok=True)
            self.root = os.path.realpath(self.work_root)
        else:
            self.root = os.path.realpath(tempfile.mkdtemp(prefix="videoloop_sandbox_"))
        for name in WRITABLE_DIRS:
            os.makedirs(os.path.join(self.root, name), exist_ok=True)
        self.container = "bwrap"  # DockerRuntime.transcribe_audio checks that this is set
        code, output = self.execute("true")
        if code != 0:
            self.stop()
            raise RuntimeError(f"bwrap can't start a sandbox: {output[:500]}")
        code, _ = self.execute("command -v ffmpeg && command -v ffprobe")
        if code != 0:
            path = self._path_entries()
            self.stop()
            raise RuntimeError(f"ffmpeg or ffprobe is not on the sandbox's PATH ({path}); use --env-prefix or --extra-ro-bind")
        logger.info(f"Sandbox started: {self.root}")

    def stop(self) -> None:
        if self.root is not None:
            shutil.rmtree(self.root, ignore_errors=True)
        self.root = None
        self.container = None

    def _path_entries(self) -> str:
        entries = ["/usr/local/bin", "/usr/bin", "/bin"]
        if self.env_prefix:
            entries.insert(0, os.path.join(self.env_prefix, "bin"))
        return ":".join(entries)

    def _argv(self) -> List[str]:
        c = [self._bwrap]
        for p in SYSTEM_DIRS:
            if os.path.exists(p):
                c += ["--ro-bind", p, p]
        c += ["--proc", "/proc", "--dev", "/dev"]
        for name in WRITABLE_DIRS:
            c += ["--bind", os.path.join(self.root, name), f"/{name}"]
        if self.env_prefix:
            c += ["--ro-bind", self.env_prefix, self.env_prefix]
        for spec in self.extra_ro_binds:
            src, _, dst = spec.partition(":")
            c += ["--ro-bind", os.path.expanduser(src), dst or src]
        if self.video_path:
            name = os.path.basename(self.video_path)
            c += ["--dir", "/videos", "--ro-bind", self.video_path, f"/videos/{name}"]
        c += ["--chdir", "/testbed", "--unshare-pid", "--die-with-parent"]
        if self.network_disabled:
            c += ["--unshare-net"]
        return c + ["--"]

    def _env(self) -> Dict[str, str]:
        env = {"PATH": self._path_entries(), "HOME": "/testbed", "TMPDIR": "/tmp"}
        for key in ("LANG", "LC_ALL", "TZ"):
            if key in os.environ:
                env[key] = os.environ[key]
        return env

    def execute(self, command: str, timeout: Optional[int] = None) -> Tuple[int, str]:
        """Run a command in the sandbox. Returns (exit_code, output), like DockerRuntime."""
        if self.root is None:
            raise RuntimeError("Sandbox not running. Call start() first.")
        if len(command) > MAX_INLINE_COMMAND:
            self._script_count += 1
            script = f".videoloop_cmd_{self._script_count}.sh"
            with open(os.path.join(self.root, "tmp", script), "w", encoding="utf-8") as f:
                f.write(command)
            shell = ["bash", "--norc", f"/tmp/{script}"]
        else:
            shell = ["bash", "--norc", "-c", command]
        # DockerRuntime never enforces a per-call timeout, so a short one must not cut
        # commands off here either; self.timeout only stops a command that hangs.
        limit = max(timeout or 0, self.timeout)
        try:
            p = subprocess.run(self._argv() + shell, capture_output=True, timeout=limit, env=self._env())
        except subprocess.TimeoutExpired as e:
            partial = (e.stdout or b"").decode("utf-8", errors="replace")
            return 124, (partial + f"\n[STDERR]\ncommand timed out after {limit}s").strip()
        except OSError as e:
            logger.error(f"Error executing command: {e}")
            return 1, str(e)
        output = p.stdout.decode("utf-8", errors="replace")
        if p.stderr:
            output += "\n[STDERR]\n" + p.stderr.decode("utf-8", errors="replace")
        return p.returncode, output.strip()

    def _host_path(self, sandbox_path: str) -> Optional[str]:
        """The host path of a file under a writable sandbox folder, or None.

        Resolved so that a symlink the agent made inside the sandbox can't reach a host file.
        """
        if self.root is None:
            return None
        parts = os.path.normpath(sandbox_path).strip("/").split("/", 1)
        if parts[0] not in WRITABLE_DIRS:
            return None
        real = os.path.realpath(os.path.join(self.root, *parts))
        allowed = os.path.join(self.root, parts[0])
        if real != allowed and not real.startswith(allowed + os.sep):
            return None
        return real

    def copy_to_container(self, local_path: str, container_path: str) -> bool:
        target = self._host_path(container_path)
        if target is None:
            logger.error(f"Can't copy into {container_path}")
            return False
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if os.path.isdir(local_path):
                shutil.copytree(local_path, target, dirs_exist_ok=True)
            else:
                shutil.copyfile(local_path, target)
            return True
        except OSError as e:
            logger.error(f"Error copying to sandbox: {e}")
            return False

    def copy_from_container(self, container_path: str, local_path: str) -> bool:
        """Copy out under the original name into local_path's folder, as DockerRuntime does."""
        source = self._host_path(container_path)
        if source is None or not os.path.exists(source):
            logger.error(f"Can't copy from {container_path}")
            return False
        target = os.path.join(os.path.dirname(local_path), os.path.basename(container_path))
        try:
            if os.path.isdir(source):
                shutil.copytree(source, target, dirs_exist_ok=True)
            else:
                shutil.copyfile(source, target)
            return True
        except OSError as e:
            logger.error(f"Error copying from sandbox: {e}")
            return False

    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        cmd = f"ffprobe -v quiet -print_format json -show_format -show_streams {shlex.quote(video_path)}"
        exit_code, output = self.execute(cmd)
        if exit_code != 0:
            return {"error": output}
        if "\n[STDERR]" in output:
            output = output.split("\n[STDERR]")[0]
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return {"error": "Failed to parse video info", "raw": output}

import os
import shutil
import subprocess
import launch

__dirname__ = os.path.dirname(__file__)


def get_pnpm():
    files = ["pnpm.exe", "pnpm.cmd", "pnpm"]
    for exe in files:
        exe = launch.which(exe)
        if exe is not None:
            return exe


def build_frontend():
    frontend_dir = os.path.join(__dirname__, "frontend")
    out_dir = os.path.join(__dirname__, "dist")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    pnpm = get_pnpm()

    subprocess.run(
        [pnpm, "i"],
        cwd=frontend_dir,
    )
    subprocess.run(
        [pnpm, "build", "--out-dir", out_dir],
        cwd=frontend_dir,
    )


def main():
    build_frontend()


if __name__ == "__main__":
    main()

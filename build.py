import os
import shutil
import subprocess

__dirname__ = os.path.dirname(__file__)


def build_frontend():
    frontend_dir = os.path.join(__dirname__, "frontend")
    out_dir = os.path.join(__dirname__, "dist")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    subprocess.run(
        ["pnpm", "i"],
        cwd=frontend_dir,
    )
    subprocess.run(
        ["pnpm", "build", "--out-dir", out_dir],
        cwd=frontend_dir,
    )


def main():
    build_frontend()


if __name__ == "__main__":
    main()

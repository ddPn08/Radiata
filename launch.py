import importlib.util
import os
import subprocess
import sys
import sys
import platform
import requests
from typing import Optional

python = sys.executable
git = os.environ.get("GIT", "git")
index_url = os.environ.get("INDEX_URL", "")
skip_install = False
__dirname__ = os.path.dirname(__file__)


def run(
    command: str,
    desc: Optional[str] = None,
    errdesc: Optional[str] = None,
    custom_env: Optional[str] = None,
):
    if desc is not None:
        print(desc)

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        env=os.environ if custom_env is None else custom_env,
    )

    if result.returncode != 0:
        message = f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}
stdout: {result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout)>0 else '<empty>'}
stderr: {result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr)>0 else '<empty>'}
"""
        raise RuntimeError(message)

    return result.stdout.decode(encoding="utf8", errors="ignore")


def download(url: str, dest: str):
    print(f"Downloading {url} to {dest}")
    data = requests.get(url).content
    with open(dest, mode="wb") as f:
        f.write(data)


def check_run(command: str):
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    return result.returncode == 0


def is_installed(package: str):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None


def run_pip(args: str, desc: Optional[str] = None):
    if skip_install:
        return

    index_url_line = f" --index-url {index_url}" if index_url != "" else ""
    return run(
        f'"{python}" -m pip {args} --prefer-binary{index_url_line}',
        desc=f"Installing {desc}",
        errdesc=f"Couldn't install {desc}",
    )


def run_python(code: str, desc: Optional[str] = None, errdesc: Optional[str] = None):
    return run(f'"{python}" -c "{code}"', desc, errdesc)


def extract_arg(args: list[str], name: str):
    return [x for x in args if x != name], name in args


def torch_version():
    try:
        import torch

        return torch.__version__
    except Exception:
        return None


def prepare_environment(args: list[str]):
    torch_command = os.environ.get(
        "TORCH_COMMAND",
        "pip install torch==1.12.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116",
    )
    tensorrt_command = os.environ.get(
        "TORCH_COMMAND",
        "pip install tensorrt==8.5.3.1",
    )
    requirements_file = os.environ.get("REQS_FILE", "requirements.txt")

    args, skip_install = extract_arg(args, "--skip-install")
    if skip_install:
        return

    args, reinstall_torch = extract_arg(args, "--reinstall-torch")
    args, reinstall_tensorrt = extract_arg(args, "--reinstall-tensorrt")

    if reinstall_torch or not is_installed("torch"):
        run(
            f'"{python}" -m {torch_command}',
            "Installing torch",
            "Couldn't install torch",
        )

    run_python(
        "import torch; assert torch.cuda.is_available(), 'Torch is not able to use GPU; add --skip-torch-cuda-test to COMMANDLINE_ARGS variable to disable this check'"
    )

    if reinstall_tensorrt or not is_installed("tensorrt"):
        run(
            f'"{python}" -m {tensorrt_command}',
            "Installing tensorrt",
            "Couldn't install tensorrt",
        )

    run(
        f"{python} -m pip install -r {requirements_file}",
        desc=f"Installing requirements",
        errdesc=f"Couldn't install requirements",
    )


def prepare_tensorrt_pluginlib():
    lib_dir = os.path.join(__dirname__, "lib", "trt", "lib")
    dest_path = os.path.join(
        lib_dir,
        "nvinfer_plugin.dll"
        if platform.system() == "Windows"
        else "libnvinfer_plugin.so",
    )

    if not os.path.exists(dest_path):
        url = (
            "https://github.com/ddPn08/Lsmith/releases/download/tensorrt-8.5.3.1/nvinfer_plugin.dll"
            if platform.system() == "Windows"
            else "https://github.com/ddPn08/Lsmith/releases/download/tensorrt-8.5.3.1/libnvinfer_plugin.so"
        )
        download(url, dest_path)

    if platform.system() == "Windows":
        os.environ["PATH"] = f'{lib_dir}{os.pathsep}{os.environ.get("PATH", "")}'
    else:
        os.environ["LD_PRELOAD"] = os.path.join(lib_dir, "libnvinfer_plugin.so")


if __name__ == "__main__":
    sys.argv = sys.argv[1:]
    main_args = os.environ.get("COMMANDLINE_ARGS", "").split(" ")
    uvicorn_args = os.environ.get("UVICORN_ARGS", "").split(" ")

    if "--" in sys.argv:
        index = sys.argv.index("--")
        uvicorn_args += sys.argv[:index]
        main_args += sys.argv[index + 1 :]
    else:
        uvicorn_args += sys.argv

    main_args = [x for x in main_args if x]
    uvicorn_args = [x for x in uvicorn_args if x]

    prepare_environment(main_args)
    prepare_tensorrt_pluginlib()

    env = os.environ.copy()
    env["COMMANDLINE_ARGS"] = " ".join(main_args)

    subprocess.run(
        [python, "-m", "uvicorn", "modules.main:app", *uvicorn_args], env=env
    )

from typing import Optional, List
import importlib.util
import os
import platform
import subprocess
import sys
import urllib.request as request

import build

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
    with request.urlopen(url) as res:
        data = res.read()
    with open(dest, mode="wb") as f:
        f.write(data)


def which(program):
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, _ = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


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


def extract_arg(args: List[str], name: str):
    return [x for x in args if x != name], name in args


def install_tensorrt(tensorrt_linux_command: str):
    if platform.system() == "Windows":
        libfile_path = which("nvinfer.dll")
        assert (
            libfile_path is not None
        ), "Could not find TensorRT. Please check if it is installed correctly."
        trt_dir = os.path.dirname(os.path.dirname(libfile_path))
        python_dir = os.path.join(trt_dir, "python")
        assert os.path.exists(
            python_dir
        ), "Couldn't find the python folder in TensorRT's directory. It may not have been installed correctly."
        key = f"{sys.version_info.major}{sys.version_info.minor}"
        for file in os.listdir(python_dir):
            if key in file and file.endswith(".whl"):
                filepath = os.path.join(python_dir, file)
                print("Installing tensorrt")
                run(f'{python} -m pip install "{filepath}"')
                return
        raise RuntimeError("Failed to install tensorrt.")
    else:
        run(f"{python} -m {tensorrt_linux_command}")


def prepare_environment(args: List[str]):
    torch_command = os.environ.get(
        "TORCH_COMMAND",
        "pip install torch==1.12.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116",
    )
    tensorrt_linux_command = os.environ.get(
        "TENSORRT_LINUX_COMMAND",
        "pip install tensorrt==8.5.3.1",
    )
    requirements_file = os.environ.get("REQS_FILE", "requirements.txt")

    args, skip_install = extract_arg(args, "--skip-install")
    if skip_install:
        return

    args, reinstall_torch = extract_arg(args, "--reinstall-torch")
    args, reinstall_tensorrt = extract_arg(args, "--reinstall-tensorrt")
    args, disable_build_frontend = extract_arg(args, "--disable-build-frontend")

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
        install_tensorrt(tensorrt_linux_command)

    run(
        f"{python} -m pip install -r {requirements_file}",
        desc=f"Installing requirements",
        errdesc=f"Couldn't install requirements",
    )

    if not disable_build_frontend:
        build.build_frontend()


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

    lib_dir = os.path.join(__dirname__, "lib", "trt", "lib")
    if platform.system() == "Windows":
        os.environ["PATH"] = f'{lib_dir}{os.pathsep}{os.environ.get("PATH", "")}'
    else:
        os.environ[
            "LD_PRELOAD"
        ] = f'{os.path.join(lib_dir, "libnvinfer_plugin.so")}{os.pathsep}{os.environ.get("LD_PRELOAD", "")}'


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

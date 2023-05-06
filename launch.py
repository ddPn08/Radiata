import importlib.util
import os
import platform
import shlex
import subprocess
import sys
from glob import glob

commandline_args = os.environ.get("COMMANDLINE_ARGS", "")
sys.argv += shlex.split(commandline_args)

python = sys.executable
git = os.environ.get("GIT", "git")
index_url = os.environ.get("INDEX_URL", "")
stored_commit_hash = None
skip_install = False


def run(command, desc=None, errdesc=None, custom_env=None):
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


def check_run(command):
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    return result.returncode == 0


def is_installed(package):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None


def commit_hash():
    global stored_commit_hash

    if stored_commit_hash is not None:
        return stored_commit_hash

    try:
        stored_commit_hash = run(f"{git} rev-parse HEAD").strip()
    except Exception:
        stored_commit_hash = "<none>"

    return stored_commit_hash


def run_pip(args, desc=None):
    if skip_install:
        return

    index_url_line = f" --index-url {index_url}" if index_url != "" else ""
    return run(
        f'"{python}" -m pip {args} --prefer-binary{index_url_line}',
        desc=f"Installing {desc}",
        errdesc=f"Couldn't install {desc}",
    )


def run_python(code, desc=None, errdesc=None):
    return run(f'"{python}" -c "{code}"', desc, errdesc)


def extract_arg(args, name):
    return [x for x in args if x != name], name in args


def install_plugin_deps():
    plugins = glob(
        os.path.join(os.path.dirname(__file__), "plugins", "*", "requirements.txt")
    )
    for plugin in plugins:
        dirname = os.path.dirname(plugin)
        dir_basename = os.path.basename(dirname)
        run(
            f'"{python}" -m pip install -r {plugin}',
            desc=f"Installing {dir_basename} plugin dependencies",
            errdesc=f"Couldn't install {dir_basename} plugin dependencies",
        )


def install_tensorrt():
    trt_version = "8.6.0"
    tensorrt_linux_command = os.environ.get(
        "TENSORRT_LINUX_COMMAND",
        f"pip install tensorrt=={trt_version}",
    )
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
                run(f'"{python}" -m pip install "{filepath}"')
        else:
            raise RuntimeError("Failed to install tensorrt.")
    else:
        run(f"{python} -m {tensorrt_linux_command}")

    run_python(
        f"import tensorrt; v = tensorrt.__version__ ; assert v == '{trt_version}', f'Incorrect version of TensorRT. Requires {trt_version} but {{v}} detected.'"
    )


def prepare_environment():
    commit = commit_hash()

    print(f"Python {sys.version}")
    print(f"Commit hash: {commit}")

    torch_command = os.environ.get(
        "TORCH_COMMAND",
        "pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117",
    )
    xformers_command = os.environ.get(
        "XFORMERS_COMMAND",
        "pip install xformers==0.0.16",
    )

    sys.argv, skip_install = extract_arg(sys.argv, "--skip-install")
    if skip_install:
        return

    sys.argv, reinstall_torch = extract_arg(sys.argv, "--reinstall-torch")
    sys.argv, reinstall_xformers = extract_arg(sys.argv, "--reinstall-xformers")
    tensorrt = "--tensorrt" in sys.argv

    if reinstall_torch or not is_installed("torch") or not is_installed("torchvision"):
        run(
            f'"{python}" -m {torch_command}',
            "Installing torch and torchvision",
            "Couldn't install torch",
        )

    if reinstall_xformers or not is_installed("xformers"):
        run(
            f'"{python}" -m {xformers_command}',
            "Installing xformers",
            "Couldn't install xformers",
        )

    run(
        f'"{python}" -m pip install -r requirements/base.txt',
        desc=f"Installing requirements",
        errdesc=f"Couldn't install requirements",
    )

    if tensorrt:
        run(
            f'"{python}" -m pip install -r requirements/tensorrt.txt',
            desc=f"Installing tensorrt requirements",
            errdesc=f"Couldn't install tensorrt requirements",
        )
        install_tensorrt()

    install_plugin_deps()


def start():
    os.environ["PATH"] = (
        os.path.join(os.path.dirname(__file__), "bin")
        + os.pathsep
        + os.environ.get("PATH", "")
    )
    os.environ["COMMANDLINE_ARGS"] = " ".join(sys.argv[1:])
    subprocess.run(
        [python, "webui.py"],
    )


if __name__ == "__main__":
    prepare_environment()
    start()

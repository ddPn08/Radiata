import importlib.util
import os
import subprocess
import sys
import sys

python = sys.executable
git = os.environ.get("GIT", "git")
index_url = os.environ.get("INDEX_URL", "")
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


def torch_version():
    try:
        import torch

        return torch.__version__
    except Exception:
        return None


def prepare_environment(args):
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


if __name__ == "__main__":
    sys.argv = sys.argv[1:]
    if "--" in sys.argv:
        index = sys.argv.index("--")
        uvicorn_args = sys.argv[:index]
        main_args = sys.argv[index + 1 :]
    else:
        uvicorn_args = sys.argv
        main_args = ""

    prepare_environment(main_args)

    env = os.environ.copy()
    if "COMMANDLINE_ARGS" in env:
        main_args += f' {env["COMMANDLINE_ARGS"]}'
    env["COMMANDLINE_ARGS"] = " ".join(main_args)

    subprocess.run(
        [python, "-m", "uvicorn", "modules.main:app", *uvicorn_args], env=env
    )

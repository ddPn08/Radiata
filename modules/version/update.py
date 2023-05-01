import importlib
import os
from glob import glob

from packaging.version import Version


def update(from_version: str, to_version: str):
    updaters_dir = os.path.join(os.path.dirname(__file__), "updaters")

    for updater in glob(os.path.join(updaters_dir, "*.py")):
        basename = os.path.splitext(os.path.basename(updater))[0]
        version_str = basename.replace("-", ".")
        version = Version(basename.replace("-", "."))

        if version <= Version(from_version):
            continue

        if version > Version(to_version):
            continue

        print(f"Running updater for v{version}...")

        importlib.import_module(f"modules.version.updaters.{basename}").run()

        yield version_str

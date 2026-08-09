"""py2app build script: bundles Saber into a standalone .app.

Usage:
    pip install -e ".[build]"
    python setup.py py2app
"""
from setuptools import setup

# py2app is only in the "build" extra (see pyproject.toml), not "dev" — a plain
# `pip install -e .` (used by CI, tests, everyday dev) must still be able to load
# this file, so the import has to be optional rather than top-level.
try:
    from py2app.build_app import py2app as py2app_command
except ImportError:
    py2app_command = None

APP = ["holo/app.py"]

cmdclass = {}
if py2app_command is not None:

    class Py2AppNoDeps(py2app_command):
        """py2app refuses to run if distribution.install_requires is set, since a
        standalone .app bundles its dependencies directly rather than pip-installing
        them. setuptools auto-populates install_requires from pyproject.toml's
        [project.dependencies] on every setup.py invocation (needed for `pip install
        -e .`), so clear it here right before py2app's own check runs.
        """

        def finalize_options(self) -> None:
            self.distribution.install_requires = None
            super().finalize_options()

    cmdclass["py2app"] = Py2AppNoDeps

OPTIONS = {
    "argv_emulation": False,
    "packages": ["holo", "rumps", "numpy", "scipy", "sklearn"],
    "plist": {
        "CFBundleName": "Saber",
        "CFBundleDisplayName": "Saber",
        "CFBundleIdentifier": "com.belal1908.saber",
        "CFBundleShortVersionString": "0.1.0",
        "CFBundleVersion": "0.1.0",
        "LSUIElement": True,  # menu-bar only, no Dock icon
        "NSMicrophoneUsageDescription": (
            "Saber listens for taps on the desk around your MacBook to trigger zone actions."
        ),
    },
}

setup(
    app=APP,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
    cmdclass=cmdclass,
)

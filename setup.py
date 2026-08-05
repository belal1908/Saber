"""py2app build script: bundles Saber into a standalone .app.

Usage:
    pip install -e ".[build]"
    python setup.py py2app
"""
from setuptools import setup

APP = ["holo/app.py"]
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
)

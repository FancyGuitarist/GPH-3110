from setuptools import setup

APP = ["powermeter_ui.py"]

OPTIONS = {
    "argv_emulation": False,
    "packages": ["numpy", "tkinter"],
    "includes": ["numpy", "tkinter"],
    "iconfile": "app_logo.icns",
    "plist": {
        "CFBundleName": "PowerMeterApp",
        "CFBundleDisplayName": "PowerMeterApp",
        "CFBundleShortVersionString": "0.1.0",
        "CFBundleVersion": "0.1.0",
        "CFBundleDevelopmentRegion": "en-CA",
        "CFBundleExecutable": "PowerMeterApp",
        "CFBundleIconFile": "app_logo.icns",
        "NSHumanReadableCopyright": "© 2025 Simon Ferland",
    },
}

setup(
    app=APP,
    options={"py2exe": OPTIONS},
    setup_requires=["py2exe"],
    install_requires=["numpy"],
)

from setuptools import setup

APP = ["powermeter_ui.py"]

DATA_FILES = [
    (
        "resources",
        [
            "resources/QuebecWattAppLogo.icns",
        ],
    ),
    ("packages", ["packages/powermeter_functions.py"]),
]

OPTIONS = {
    "argv_emulation": False,
    "packages": ["numpy", "tkinter"],
    "includes": ["numpy", "tkinter"],
    "iconfile": "resources/QuebecWattAppLogo.icns",
    "plist": {
        "CFBundleName": "PowerMeterApp",
        "CFBundleDisplayName": "PowerMeterApp",
        "CFBundleShortVersionString": "0.1.0",
        "CFBundleVersion": "0.1.0",
        "CFBundleDevelopmentRegion": "en-CA",
        "CFBundleExecutable": "PowerMeterApp",
        "CFBundleIconFile": "resources/QuebecWattAppLogo.icns",
        "NSHumanReadableCopyright": "© 2025 Simon Ferland",
    },
}

setup(
    app=APP,
    options={"py2app": OPTIONS},
    data_files=DATA_FILES,
    setup_requires=["py2app"],
    install_requires=["numpy"],
)

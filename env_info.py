"""Prints (and saves to results/env_info.txt) the machine description that the
report must contain (hardware + timing method). Works on Linux / Windows / macOS.

    python env_info.py
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path


def cpu_name() -> str:
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif sys.platform == "win32":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        elif sys.platform == "darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                           text=True).strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def ram_gb() -> str:
    try:
        if sys.platform.startswith("linux") or sys.platform == "darwin":
            pages = os.sysconf("SC_PHYS_PAGES")
            size = os.sysconf("SC_PAGE_SIZE")
            return f"{pages * size / 1024 ** 3:.1f} GB"
        if sys.platform == "win32":
            import ctypes

            class MEMSTAT(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MEMSTAT()
            m.dwLength = ctypes.sizeof(MEMSTAT)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return f"{m.ullTotalPhys / 1024 ** 3:.1f} GB"
    except Exception:
        pass
    return "unknown"


def lib_version(name: str) -> str:
    try:
        return __import__(name).__version__
    except Exception:
        return "not installed"


def main():
    lines = [
        f"OS            : {platform.platform()}",
        f"CPU           : {cpu_name()}",
        f"Logical cores : {os.cpu_count()}",
        f"RAM           : {ram_gb()}",
        f"Python        : {platform.python_version()} ({platform.python_implementation()})",
        f"numpy         : {lib_version('numpy')}",
        f"matplotlib    : {lib_version('matplotlib')}",
        f"scikit-fuzzy  : {lib_version('skfuzzy')}",
        f"scikit-learn  : {lib_version('sklearn')}",
        "Timing method : time.perf_counter() around each controller.compute() call "
        "(one decision), plotting and logging excluded.",
    ]
    text = "\n".join(lines)
    print(text)
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    (out / "env_info.txt").write_text(text + "\n", encoding="utf-8")
    print(f"\nSaved to {out / 'env_info.txt'}")


if __name__ == "__main__":
    main()

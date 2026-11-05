"""Recent llvmlite wheels keep their private MSVC runtime in llvmlite.libs."""

from PyInstaller.utils.hooks import (
    collect_delvewheel_libs_directory,
    collect_dynamic_libs,
)

datas, binaries = collect_delvewheel_libs_directory(
    "llvmlite", datas=[], binaries=collect_dynamic_libs("llvmlite")
)

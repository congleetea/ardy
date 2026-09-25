# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    @staticmethod
    def _pybind11_cmake_dir():
        """Path to the pybind11 CMake config of the current Python env, if usable.

        A system-wide pybind11 (e.g. 2.9.x shipped by distro packages) is too old
        for Python >= 3.11 and breaks the build, so prefer the one installed in
        the (build) environment and only accept it if it is new enough.
        """
        try:
            from pybind11 import __version__
            from pybind11 import get_cmake_dir
        except ImportError:
            return None

        try:
            major, minor = (int(p) for p in __version__.split(".")[:2])
        except ValueError:
            return None

        if (major, minor) < (2, 12):
            return None

        return str(get_cmake_dir())

    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as exc:
            raise RuntimeError("CMake must be installed to build this package") from exc

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DPython3_EXECUTABLE={sys.executable}",
        ]

        pybind11_cmake_dir = self._pybind11_cmake_dir()
        if pybind11_cmake_dir:
            cmake_args.append(f"-Dpybind11_DIR={pybind11_cmake_dir}")
        else:
            self.announce(
                "pybind11 >= 2.12 not found in the Python environment; CMake will try to fetch it.",
                level=2,
            )

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]
        cmake_args.append(f"-DCMAKE_BUILD_TYPE={cfg}")

        use_mingw = False
        mingw_bin = None

        if sys.platform == "win32":
            generator = os.environ.get("CMAKE_GENERATOR", "")
            if generator:
                cmake_args = ["-G", generator] + cmake_args
                if "mingw" in generator.lower():
                    use_mingw = True
                else:
                    cmake_args.append(f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}")
            else:
                try:
                    subprocess.check_output(["g++", "--version"], stderr=subprocess.STDOUT)
                    use_mingw = True
                    cmake_args = ["-G", "MinGW Makefiles"] + cmake_args
                    build_args = []
                except (OSError, subprocess.CalledProcessError):
                    cmake_args.append(f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}")

            if use_mingw:
                gxx_path = shutil.which("g++")
                if gxx_path:
                    mingw_bin = Path(gxx_path).parent
        else:
            build_args += ["--", "-j4"]

        env = os.environ.copy()
        env["CXXFLAGS"] = f'{env.get("CXXFLAGS", "")} -DVERSION_INFO=\\"{self.distribution.get_version()}\\"'

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)

        if use_mingw and mingw_bin is not None:
            runtime_libs = [
                "libstdc++-6.dll",
                "libgcc_s_seh-1.dll",
                "libwinpthread-1.dll",
            ]
            extdir_path = Path(extdir)
            extdir_path.mkdir(parents=True, exist_ok=True)
            for lib_name in runtime_libs:
                src_path = mingw_bin / lib_name
                if src_path.exists():
                    shutil.copy2(src_path, extdir_path / lib_name)
                else:
                    self.announce(
                        f"Warning: Expected MinGW runtime DLL '{lib_name}' not found next to g++ (looked in {mingw_bin}). "
                        "The built extension may fail to import if the DLL is not on PATH.",
                        level=3,
                    )


setup(
    packages=find_packages(include=["ardy", "ardy.*"]) + ["motion_correction"],
    package_dir={"motion_correction": "MotionCorrection/python/motion_correction"},
    ext_modules=[CMakeExtension("motion_correction._motion_correction", "MotionCorrection")],
    cmdclass={"build_ext": CMakeBuild},
)

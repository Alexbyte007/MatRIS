import glob
import os
import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

cuda_source = glob.glob("**/*.cu", recursive=True)
cpp_source = glob.glob("**/*.cpp", recursive=True)


def cuda_include_dirs():
    include_dirs = ["include"]
    repo_root = Path(__file__).resolve().parents[4]
    cutlass_root = repo_root / "external" / "cutlass"
    if cutlass_root.exists():
        include_dirs.extend(
            [
                str(cutlass_root / "include"),
                str(cutlass_root / "tools" / "util" / "include"),
            ]
        )

    candidates = []
    if CUDA_HOME:
        candidates.append(Path(CUDA_HOME) / "include")

    prefixes = [Path(sys.prefix)]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefixes.append(Path(conda_prefix))

    for conda_path in prefixes:
        candidates.extend(
            [
                conda_path / "include",
                conda_path / "targets" / "x86_64-linux" / "include",
            ]
        )

        site_packages = conda_path / "lib"
        candidates.extend(site_packages.glob("python*/site-packages/nvidia/*/include"))
        candidates.extend(site_packages.glob("python*/site-packages/triton/backends/nvidia/include"))

    for path in candidates:
        if path.exists():
            include_dirs.append(str(path))

    return include_dirs


def cuda_library_dirs():
    library_dirs = []

    prefixes = [Path(sys.prefix)]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefixes.append(Path(conda_prefix))

    candidates = []
    for conda_path in prefixes:
        candidates.append(conda_path / "lib")
        site_packages = conda_path / "lib"
        candidates.extend(site_packages.glob("python*/site-packages/nvidia/*/lib"))

    for path in candidates:
        if path.exists():
            library_dirs.append(str(path))

    link_dir = Path("build") / "cuda_lib"
    linked_dir_added = False
    for lib_name in ("cudart", "cublas"):
        for path in candidates:
            versioned_lib = path / f"lib{lib_name}.so.12"
            if not versioned_lib.exists():
                continue

            link_dir.mkdir(parents=True, exist_ok=True)
            link_path = link_dir / f"lib{lib_name}.so"
            if not link_path.exists():
                link_path.symlink_to(versioned_lib)
            if not linked_dir_added:
                library_dirs.insert(0, str(link_dir.resolve()))
                linked_dir_added = True
            break

    return library_dirs


setup(
    name="matris_op",
    ext_modules=[
        CUDAExtension(
            "matris_op",
            cuda_source + cpp_source,
            include_dirs=cuda_include_dirs(),
            library_dirs=cuda_library_dirs(),
            libraries=["cublas"],
            extra_compile_args={"nvcc": ["-O3", "-std=c++17"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)

#python setup.py build develop

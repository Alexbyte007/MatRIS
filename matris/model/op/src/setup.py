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
        # Do not add the broad conda include directory when pip torch is active:
        # it may contain stale conda libtorch headers and collide with
        # torch.utils.cpp_extension's site-packages/torch/include path.
        candidates.append(conda_path / "targets" / "x86_64-linux" / "include")

        site_packages = conda_path / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
        candidates.extend(site_packages.glob("nvidia/*/include"))
        candidates.extend(site_packages.glob("triton/backends/nvidia/include"))

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
        site_packages = conda_path / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
        candidates.extend(site_packages.glob("nvidia/*/lib"))
        # Keep the broad conda lib directory last. It can contain stale conda
        # libtorch/CUDA libraries after switching to pip torch, but is still a
        # useful fallback for non-torch libraries.
        candidates.append(conda_path / "lib")

    for path in candidates:
        if path.exists():
            library_dirs.append(str(path))

    for path in candidates:
        versioned_cudart = path / "libcudart.so.12"
        if not versioned_cudart.exists():
            continue

        link_dir = Path("build") / "cuda_lib"
        link_dir.mkdir(parents=True, exist_ok=True)
        link_path = link_dir / "libcudart.so"
        if not link_path.exists():
            link_path.symlink_to(versioned_cudart)
        library_dirs.insert(0, str(link_dir.resolve()))
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
            extra_compile_args={"nvcc": ["-O3", "-std=c++17"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)

#python setup.py build develop

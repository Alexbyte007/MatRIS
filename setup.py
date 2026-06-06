from setuptools import Extension, find_packages, setup
import numpy as np


def build_graph_extension():
    extension = Extension(
        "matris.graph.cygraph",
        ["matris/graph/cygraph.pyx"],
        include_dirs=[np.get_include()],
    )

    try:
        from Cython.Build import cythonize
    except ImportError:
        extension.sources = ["matris/graph/cygraph.c"]
        return [extension]

    return cythonize(
        [extension],
        compiler_directives={"language_level": 3},
    )


setup(
    packages=find_packages(include=["matris", "matris.*"]),
    ext_modules=build_graph_extension(),
)

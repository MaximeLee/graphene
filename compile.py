from setuptools import setup, find_namespace_packages, find_packages, Extension
import os, sys
from Cython.Build import cythonize
import numpy
import pythran
import subprocess

parallel_files = ["electron_electron.pyx"]
openmp_arg = '-fopenmp'

parallel = eval(sys.argv[1])
debug = eval(sys.argv[2])

# Recursively find all pyx/pxd files
extensions_name_path_to_moove_extensions = []
extension_modules = []
for root, _, files in os.walk('graphene'):
    for file in files:
        if file.endswith(".pyx"):
            path = os.path.join(root, file)
            name = os.path.splitext(file)[0]
            if parallel and file in parallel_files:
                _ext = Extension(name, [path], extra_compile_args=[openmp_arg], extra_link_args=[openmp_arg])
            else:
                _ext = path #Extension(f'extension{i}', [os.path.join(root, file)])

            extension_modules.append(_ext)

            extensions_name_path_to_moove_extensions.append((name, os.path.dirname(path)))

compiler_directives = {
    'boundscheck': False,
    'cdivision': True, 
    'initializedcheck': False
}

setup(
    ext_modules = cythonize(extension_modules, compiler_directives = compiler_directives, gdb_debug = debug),  # Compile all .pyx files in all discovered packages
    include_dirs = [numpy.get_include()],
    script_args = ['build_ext', '--inplace']
)

# moove extensions to their respective folder
if parallel:
    for name, path in extensions_name_path_to_moove_extensions:
        subprocess.call(f"mv {name}*so {path}", shell=True)

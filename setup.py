# Copyright 2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import numpy

with open("pennylane_lightning/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

requirements = [
    "numpy",
    "scipy",
    "networkx",
    "autograd",
    "pennylane>=0.9.0",
    "toml",
    "appdirs",
    "semantic_version==2.6",
]


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __str__(self):
        import pybind11
        return pybind11.get_include()


CFLAGS = ["-DXTENSOR_USE_XSIMD -mavx2 -ffast-math -lblas -llapack"]
LINKFLAGS = ["-l", "blas", "-l", "lapack"]

install_dir = '/home/antal/xanadu/lightning_deps/'

XTENSOR_INCLUDE = os.getcwd()

if sys.platform == "linux":
    prefix = os.getenv("CONDA_PREFIX","")
    prefix = prefix or os.getenv("PREFIX","")

    conda_dir = os.path.join(prefix, "lib")
else:
    conda_dir = ""

ext_modules = [
    Extension(
        'lightning_qubit_ops',
        # Sort input source files to ensure bit-for-bit reproducible builds
        # (https://github.com/pybind/python_example/pull/53)
        sorted(['pennylane_lightning/lightning_qubit_ops.cpp']),
        include_dirs=[
            # Path to pybind11 headers
            #get_pybind_include(),
            numpy.get_include(),
            "/usr/share/miniconda/lib",
            conda_dir,
            # os.path.join(XTENSOR_INCLUDE, 'pybind11/include/'),
            # os.path.join(XTENSOR_INCLUDE, 'xtensor/include/'),
            # os.path.join(XTENSOR_INCLUDE, 'xtensor-python/include/'),
            # os.path.join(XTENSOR_INCLUDE, 'xtensor-blas/include/'),
            # os.path.join(XTENSOR_INCLUDE, 'xtl/include/'),
        ],
        language='c++',
        extra_compile_args=CFLAGS,
        extra_link_args=LINKFLAGS,
    ),
]


# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    import os
    with tempfile.NamedTemporaryFile('w', suffix='.cpp', delete=False) as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        fname = f.name
    try:
        compiler.compile([fname], extra_postargs=[flagname])
    except setuptools.distutils.errors.CompileError:
        return False
    finally:
        try:
            os.remove(fname)
        except OSError:
            pass
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag.
    The newer version is prefered over c++11 (when it is available).
    """
    flags = ['-std=c++17', '-std=c++14', '-std=c++11']

    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    raise RuntimeError('Unsupported compiler -- at least C++11 support '
                       'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }

    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        opts.extend(["-O3", "-shared", "-w"])


        link_opts = self.l_opts.get(ct, [])
        link_opts += CFLAGS
        if ct == 'unix':
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')

        for ext in self.extensions:
            ext.define_macros = [('VERSION_INFO', '"{}"'.format(self.distribution.get_version()))]
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)


info = {
    'name': 'PennyLane-lightning',
    'version': version,
    'maintainer': 'Xanadu Inc.',
    'maintainer_email': 'software@xanadu.ai',
    'url': 'https://github.com/XanaduAI/pennylane-lightning',
    'license': 'Apache License 2.0',
    'packages': find_packages(where="."),
    'entry_points': {
        'pennylane.plugins': [
            'lightning.qubit = pennylane_lightning:LightningQubit',
            ],
        },
    'description': 'PennyLane is a Python quantum machine learning library by Xanadu Inc.',
    'long_description': open('README.rst').read(),
    'provides': ["pennylane_lightning"],
    'ext_modules': ext_modules,
    'install_requires': requirements,
    'setup_requires': ['pybind11>=2.5.0'],
    'cmdclass': {'build_ext': BuildExt},
    'command_options': {
        'build_sphinx': {
            'version': ('setup.py', version),
            'release': ('setup.py', version)}}
}

classifiers = [
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Operating System :: POSIX",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
    "Programming Language :: Python",
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3 :: Only',
    "Topic :: Scientific/Engineering :: Physics"
]

setup(classifiers=classifiers, **(info))

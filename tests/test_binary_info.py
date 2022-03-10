# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Test binary information of ``lightning.qubit``.
"""

try:
    from pennylane_lightning.lightning_qubit_ops import runtime_info, compile_info
except (ImportError, ModuleNotFoundError):
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


def test_runtime_info():
    m = runtime_info()
    for key in ["AVX", "AVX2", "AVX512F"]:
        assert key in m


def test_compile_info():
    m = compile_info()
    for key in ["cpu.arch", "compiler.name", "compiler.version", "AVX2", "AVX512F"]:
        assert key in m

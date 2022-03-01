# Copyright 2021 Xanadu Quantum Technologies Inc.

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
Unit tests for Measures in lightning.qubit.
"""
import numpy as np
import pennylane as qml
from pennylane.measure import (
    Variance,
    Expectation,
)

from pennylane.queuing import AnnotatedQueue

import pytest

try:
    from pennylane_lightning.lightning_qubit_ops import (
        MeasuresC64,
        MeasuresC128,
    )
except (ImportError, ModuleNotFoundError):
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


def test_no_measure(tol):
    """Test that failing to specify a measurement
    raises an exception"""
    dev = qml.device("lightning.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x, wires=0)
        return qml.PauliY(0)

    with pytest.raises(qml.QuantumFunctionError, match="must return either a single measurement"):
        circuit(0.65)


class TestProbs:
    """Test Probs in Lightning"""

    @pytest.fixture
    def dev(self):
        return qml.device("lightning.qubit", wires=2)

    def test_probs_dtype64(self, dev):
        """Test if probs changes the state dtype"""
        dev._state = np.array([1, 0]).astype(np.complex64)
        p = dev.probability(wires=[0, 1])

        assert dev._state.dtype == np.complex64
        assert np.allclose(p, [1, 1, 0, 0])

    def test_probs_dtype_error(self, dev):
        """Test if probs raise error with complex256"""
        dev._state = np.array([1, 0]).astype(np.complex256)

        with pytest.raises(TypeError, match="Unsupported complex Type:"):
            dev.probability(wires=[0, 1])

    @pytest.mark.parametrize("C", [np.complex64, np.complex128])
    def test_probs_H(self, tol, dev, C):
        """Test probs with Hadamard"""
        dev._state = dev._asarray(dev._state, C)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=1)
            return qml.probs(wires=[0, 1])

        assert np.allclose(circuit(), [0.5, 0.5, 0.0, 0.0], atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "cases",
        [
            [None, [0.9165164490394898, 0.0, 0.08348355096051052, 0.0]],
            [[], [0.9165164490394898, 0.0, 0.08348355096051052, 0.0]],
            [[0, 1], [0.9165164490394898, 0.0, 0.08348355096051052, 0.0]],
            [[1, 0], [0.9165164490394898, 0.08348355096051052, 0.0, 0.0]],
            [0, [0.9165164490394898, 0.08348355096051052]],
            [[0], [0.9165164490394898, 0.08348355096051052]],
        ],
    )
    @pytest.mark.parametrize("C", [np.complex64, np.complex128])
    def test_probs_tape_wire0(self, cases, tol, dev, C):
        """Test probs with a circuit on wires=[0]"""
        dev._state = dev._asarray(dev._state, C)

        x, y, z = [0.5, 0.3, -0.7]

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            return qml.probs(wires=cases[0])

        assert np.allclose(circuit(), cases[1], atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "cases",
        [
            [
                None,
                [
                    0.9178264236525453,
                    0.02096485729264079,
                    0.059841820910257436,
                    0.0013668981445561978,
                ],
            ],
            [
                [],
                [
                    0.9178264236525453,
                    0.02096485729264079,
                    0.059841820910257436,
                    0.0013668981445561978,
                ],
            ],
            [
                [0, 1],
                [
                    0.9178264236525453,
                    0.02096485729264079,
                    0.059841820910257436,
                    0.0013668981445561978,
                ],
            ],
            [
                [1, 0],
                [
                    0.9178264236525453,
                    0.059841820910257436,
                    0.02096485729264079,
                    0.0013668981445561978,
                ],
            ],
            [0, [0.938791280945186, 0.061208719054813635]],
            [[0], [0.938791280945186, 0.061208719054813635]],
        ],
    )
    @pytest.mark.parametrize("C", [np.complex64, np.complex128])
    def test_probs_tape_wire01(self, cases, tol, dev, C):
        """Test probs with a circuit on wires=[0,1]"""
        dev._state = dev._asarray(dev._state, C)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.5, wires=[0])
            qml.RY(0.3, wires=[1])
            return qml.probs(wires=cases[0])

        assert np.allclose(circuit(), cases[1], atol=tol, rtol=0)


class TestExpval:
    """Tests for the expval function"""

    @pytest.fixture
    def dev(self):
        return qml.device("lightning.qubit", wires=2)

    def test_expval_dtype64(self, dev):
        """Test if expval changes the state dtype"""
        dev._state = np.array([1, 0]).astype(np.complex64)
        e = dev.expval(qml.PauliX(0))

        assert dev._state.dtype == np.complex64
        assert np.allclose(e, 0.0)

    def test_expval_dtype_error(self, dev):
        """Test if expval raise error with complex256"""
        dev._state = np.array([1, 0]).astype(np.complex256)

        with pytest.raises(TypeError, match="Unsupported complex Type:"):
            dev.expval(qml.PauliX(0))

    @pytest.mark.parametrize(
        "cases",
        [
            [qml.PauliX(0), -0.041892271271228736],
            [qml.PauliX(1), 0.0],
            [qml.PauliY(0), -0.5516350865364075],
            [qml.PauliY(1), 0.0],
            [qml.PauliZ(0), 0.8330328980789793],
            [qml.PauliZ(1), 1.0],
        ],
    )
    @pytest.mark.parametrize("C", [np.complex64, np.complex128])
    def test_expval_qml_tape_wire0(self, cases, tol, dev, C):
        """Test expval with a circuit on wires=[0]"""
        dev._state = dev._asarray(dev._state, C)

        x, y, z = [0.5, 0.3, -0.7]

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            return qml.expval(cases[0])

        assert np.allclose(circuit(), cases[1], atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "cases",
        [
            [qml.PauliX(0), 0.0],
            [qml.PauliX(1), -0.19866933079506122],
            [qml.PauliY(0), -0.3894183423086505],
            [qml.PauliY(1), 0.0],
            [qml.PauliZ(0), 0.9210609940028852],
            [qml.PauliZ(1), 0.9800665778412417],
        ],
    )
    @pytest.mark.parametrize("C", [np.complex64, np.complex128])
    def test_expval_wire01(self, cases, tol, dev, C):
        """Test expval with a circuit on wires=[0,1]"""
        dev._state = dev._asarray(dev._state, C)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.RY(-0.2, wires=[1])
            return qml.expval(cases[0])

        assert np.allclose(circuit(), cases[1], atol=tol, rtol=0)

    @pytest.mark.parametrize("C", [np.complex64, np.complex128])
    def test_value(self, C, dev, tol):
        """Test that the expval interface works"""
        dev._state = dev._asarray(dev._state, C)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        x = 0.54
        res = circuit(x)
        expected = -np.sin(x)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("C", [np.complex64, np.complex128])
    def test_not_an_observable(self, C, dev):
        """Test that a qml.QuantumFunctionError is raised if the provided
        argument is not an observable"""
        dev._state = dev._asarray(dev._state, C)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.52, wires=0)
            return qml.expval(qml.CNOT(wires=[0, 1]))

        with pytest.raises(qml.QuantumFunctionError, match="CNOT is not an observable"):
            circuit()

    @pytest.mark.parametrize("C", [np.complex64, np.complex128])
    def test_observable_return_type_is_expectation(self, C, dev):
        """Test that the return type of the observable is :attr:`ObservableReturnTypes.Expectation`"""
        dev._state = dev._asarray(dev._state, C)

        @qml.qnode(dev)
        def circuit():
            res = qml.expval(qml.PauliZ(0))
            assert res.return_type is Expectation
            return res

        circuit()


class TestVar:
    """Tests for the var function"""

    @pytest.fixture
    def dev(self):
        return qml.device("lightning.qubit", wires=2)

    def test_var_dtype64(self, dev):
        """Test if var changes the state dtype"""
        dev._state = np.array([1, 0]).astype(np.complex64)
        v = dev.var(qml.PauliX(0))

        assert dev._state.dtype == np.complex64
        assert np.allclose(v, 1.0)

    def test_expval_dtype_error(self, dev):
        """Test if var raise error with complex256"""
        dev._state = np.array([1, 0]).astype(np.complex256)

        with pytest.raises(TypeError, match="Unsupported complex Type:"):
            dev.var(qml.PauliX(0))

    @pytest.mark.parametrize(
        "cases",
        [
            [qml.PauliX(0), 0.9982450376077382],
            [qml.PauliX(1), 1.0],
            [qml.PauliY(0), 0.6956987716741251],
            [qml.PauliY(1), 1.0],
            [qml.PauliZ(0), 0.3060561907181374],
            [qml.PauliZ(1), -4.440892098500626e-16],
        ],
    )
    @pytest.mark.parametrize("C", [np.complex64, np.complex128])
    def test_var_qml_tape_wire0(self, cases, tol, dev, C):
        """Test var with a circuit on wires=[0]"""
        dev._state = dev._asarray(dev._state, C)

        x, y, z = [0.5, 0.3, -0.7]

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            return qml.var(cases[0])

        assert np.allclose(circuit(), cases[1], atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "cases",
        [
            [qml.PauliX(0), 1.0],
            [qml.PauliX(1), 0.9605304970014426],
            [qml.PauliY(0), 0.8483533546735826],
            [qml.PauliY(1), 1.0],
            [qml.PauliZ(0), 0.15164664532641725],
            [qml.PauliZ(1), 0.03946950299855745],
        ],
    )
    @pytest.mark.parametrize("C", [np.complex64, np.complex128])
    def test_var_qml_tape_wire01(self, cases, tol, dev, C):
        """Test var with a circuit on wires=[0,1]"""
        dev._state = dev._asarray(dev._state, C)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.RY(-0.2, wires=[1])
            return qml.var(cases[0])

        assert np.allclose(circuit(), cases[1], atol=tol, rtol=0)

    @pytest.mark.parametrize("C", [np.complex64, np.complex128])
    def test_value(self, C, dev, tol):
        """Test that the var function works"""
        dev._state = dev._asarray(dev._state, C)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.var(qml.PauliZ(0))

        x = 0.54
        res = circuit(x)
        expected = np.sin(x) ** 2

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("C", [np.complex64, np.complex128])
    def test_not_an_observable(self, C, dev):
        """Test that a qml.QuantumFunctionError is raised if the provided
        argument is not an observable"""
        dev._state = dev._asarray(dev._state, C)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.52, wires=0)
            return qml.var(qml.CNOT(wires=[0, 1]))

        with pytest.raises(qml.QuantumFunctionError, match="CNOT is not an observable"):
            res = circuit()

    @pytest.mark.parametrize("C", [np.complex64, np.complex128])
    def test_observable_return_type_is_variance(self, C, dev):
        """Test that the return type of the observable is :attr:`ObservableReturnTypes.Variance`"""
        dev._state = dev._asarray(dev._state, C)

        @qml.qnode(dev)
        def circuit():
            res = qml.var(qml.PauliZ(0))
            assert res.return_type is Variance
            return res

        circuit()


@pytest.mark.parametrize("stat_func", [qml.expval, qml.var])
class TestBetaStatisticsError:
    """Tests for errors arising for the beta statistics functions"""

    def test_not_an_observable(self, stat_func):
        """Test that a qml.QuantumFunctionError is raised if the provided
        argument is not an observable"""
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.52, wires=0)
            return stat_func(qml.CNOT(wires=[0, 1]))

        with pytest.raises(qml.QuantumFunctionError, match="CNOT is not an observable"):
            circuit()


class TestWiresInExpval:
    """Test different Wires settings in Lightning's expval."""

    @pytest.mark.parametrize(
        "wires1, wires2",
        [
            ([2, 3, 0], [2, 3, 0]),
            ([0, 1], [0, 1]),
            ([0, 2, 3], [2, 0, 3]),
            (["a", "c", "d"], [2, 3, 0]),
            ([-1, -2, -3], ["q1", "ancilla", 2]),
            (["a", "c"], [3, 0]),
            ([-1, -2], ["ancilla", 2]),
            (["a"], ["nothing"]),
        ],
    )
    @pytest.mark.parametrize("C", [np.complex64, np.complex128])
    def test_wires_expval(self, wires1, wires2, C, tol):
        """Test that the expectation of a circuit is independent from the wire labels used."""
        dev1 = qml.device("lightning.qubit", wires=wires1)
        dev1._state = dev1._asarray(dev1._state, C)

        dev2 = qml.device("lightning.qubit", wires=wires2)
        dev2._state = dev2._asarray(dev2._state, C)

        n_wires = len(wires1)

        @qml.qnode(dev1)
        def circuit1():
            qml.RX(0.5, wires=wires1[0 % n_wires])
            qml.RY(2.0, wires=wires1[1 % n_wires])
            if n_wires > 1:
                qml.CNOT(wires=[wires1[0], wires1[1]])
            return [qml.expval(qml.PauliZ(wires=w)) for w in wires1]

        @qml.qnode(dev2)
        def circuit2():
            qml.RX(0.5, wires=wires2[0 % n_wires])
            qml.RY(2.0, wires=wires2[1 % n_wires])
            if n_wires > 1:
                qml.CNOT(wires=[wires2[0], wires2[1]])
            return [qml.expval(qml.PauliZ(wires=w)) for w in wires2]

        assert np.allclose(circuit1(), circuit2(), atol=tol)


class TestWiresInVar:
    """Test different Wires settings in Lightning's var."""

    @pytest.mark.parametrize(
        "wires1, wires2",
        [
            (["a", "c", "d"], [2, 3, 0]),
            ([-1, -2, -3], ["q1", "ancilla", 2]),
            (["a", "c"], [3, 0]),
            ([-1, -2], ["ancilla", 2]),
            (["a"], ["nothing"]),
        ],
    )
    @pytest.mark.parametrize("C", [np.complex64, np.complex128])
    def test_wires_var(self, wires1, wires2, C, tol):
        """Test that the expectation of a circuit is independent from the wire labels used."""
        dev1 = qml.device("lightning.qubit", wires=wires1)
        dev1._state = dev1._asarray(dev1._state, C)

        dev2 = qml.device("lightning.qubit", wires=wires2)
        dev2._state = dev2._asarray(dev2._state, C)

        n_wires = len(wires1)

        @qml.qnode(dev1)
        def circuit1():
            qml.RX(0.5, wires=wires1[0 % n_wires])
            qml.RY(2.0, wires=wires1[1 % n_wires])
            if n_wires > 1:
                qml.CNOT(wires=[wires1[0], wires1[1]])
            return [qml.var(qml.PauliZ(wires=w)) for w in wires1]

        @qml.qnode(dev2)
        def circuit2():
            qml.RX(0.5, wires=wires2[0 % n_wires])
            qml.RY(2.0, wires=wires2[1 % n_wires])
            if n_wires > 1:
                qml.CNOT(wires=[wires2[0], wires2[1]])
            return [qml.var(qml.PauliZ(wires=w)) for w in wires2]

        assert np.allclose(circuit1(), circuit2(), atol=tol)


class TestStatisticsQueuing:
    """Tests the statistics method in Lightning"""

    @pytest.fixture
    def dev(self):
        return qml.device("lightning.qubit", wires=2)

    @pytest.mark.parametrize(
        "obs, expected",
        [
            (qml.var(qml.PauliZ(0)), [0.0]),
            (qml.expval(qml.PauliZ(0)), [1.0]),
            (qml.probs(), [1.0, 0.0, 0.0, 0.0]),
        ],
    )
    def test_single_obs(self, dev, obs, expected):
        """Test statistics over single observable oven an initiated state"""
        res = dev.statistics([obs])
        assert np.allclose(res, expected)

    @pytest.mark.parametrize(
        "obs, expected",
        [
            ([qml.var(qml.PauliZ(0)), qml.var(qml.PauliZ(1))], [0.0]),
            ([qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))], [1.0, 1.0]),
            ([qml.probs(0), qml.probs(1)], [1.0, 0.0]),
            ([qml.var(qml.PauliZ(1)), qml.expval(qml.PauliX(1))], [0.0, 1.0]),
        ],
    )
    def test_obs_list(self, dev, obs, expected):
        """Test statistics of a list of observables oven an initiated state"""
        res = dev.statistics(obs)
        assert np.allclose(res, expected)


@pytest.mark.parametrize("stat_func", [qml.expval, qml.var, qml.sample])
class TestBetaStatisticsError:
    """Tests for errors arising for the beta statistics functions"""

    def test_not_an_observable(self, stat_func):
        """Test that a qml.QuantumFunctionError is raised if the provided
        argument is not an observable"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.52, wires=0)
            return stat_func(qml.CNOT(wires=[0, 1]))

        with pytest.raises(qml.QuantumFunctionError, match="CNOT is not an observable"):
            circuit()

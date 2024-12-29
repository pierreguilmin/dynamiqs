import timeit

import jax
import jax.numpy as jnp
import pytest

from dynamiqs import QArray, asqarray
from dynamiqs.time_qarray import (
    ConstantTimeQArray,
    SummedTimeQArray,
    constant,
    modulated,
    pwc,
    timecallable,
)

from .order import TEST_SHORT


def assert_equal(x, y):
    if isinstance(x, QArray):
        x = x.to_jax()
    if isinstance(y, QArray):
        y = y.to_jax()
    assert jnp.array_equal(x, y)


@pytest.mark.run(order=TEST_SHORT)
class TestConstantTimeQArray:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.x = constant(jnp.arange(4).reshape(2, 2))

    @pytest.mark.skip('broken test')
    def test_jit(self):
        # we don't test speed here, just that it works
        x = jax.jit(self.x)
        assert_equal(x(0.0), [0, 1, 2, 3])

    def test_call(self):
        assert_equal(self.x(0.0), [[0, 1], [2, 3]])
        assert_equal(self.x(1.0), [[0, 1], [2, 3]])

    def test_reshape(self):
        x = self.x.reshape(1, 2, 2)
        assert_equal(x(0.0), [[[0, 1], [2, 3]]])

    def test_broadcast(self):
        x = self.x.broadcast_to(2, 2, 2)
        assert_equal(x(0.0), [[[0, 1], [2, 3]], [[0, 1], [2, 3]]])

    def test_conj(self):
        x = constant([[0 + 0j, 1 + 1j], [2 + 2j, 3 + 3j]])
        x = x.conj()
        assert_equal(x(0.0), [[0 - 0j, 1 - 1j], [2 - 2j, 3 - 3j]])

    def test_neg(self):
        x = -self.x
        assert_equal(x(0.0), [[-0, -1], [-2, -3]])

    def test_mul(self):
        x = self.x * 2
        assert_equal(x(0.0), [[0, 2], [4, 6]])

    def test_rmul(self):
        x = 2 * self.x
        assert_equal(x(0.0), [[0, 2], [4, 6]])

    def test_add(self):
        # test type `ArrayLike`
        x = self.x + jnp.ones_like(self.x)
        assert isinstance(x, ConstantTimeQArray)
        assert_equal(x(0.0), [[1, 2], [3, 4]])

        # test type `ConstantTimeQArray`
        x = self.x + self.x
        assert isinstance(x, ConstantTimeQArray)
        assert_equal(x(0.0), [[0, 2], [4, 6]])

    def test_radd(self):
        # test type `ArrayLike`
        x = jnp.ones_like(self.x) + self.x
        assert isinstance(x, ConstantTimeQArray)
        assert_equal(x(0.0), [[1, 2], [3, 4]])


@pytest.mark.run(order=TEST_SHORT)
class TestCallableTimeQArray:
    @pytest.fixture(autouse=True)
    def _setup(self):
        f = lambda t: t * asqarray(jnp.arange(4).reshape(2, 2))
        self.x = timecallable(f)

    @pytest.mark.skip('broken test')
    def test_jit(self):
        x = jax.jit(self.x)
        assert_equal(x(0.0), [0, 0])
        assert_equal(x(1.0), [1, 2])

        t1 = timeit.timeit(lambda: x(1.0), number=1000)
        t2 = timeit.timeit(lambda: self.x(1.0), number=1000)
        assert t1 < t2

    def test_call(self):
        assert_equal(self.x(0.0), [[0, 0], [0, 0]])
        assert_equal(self.x(1.0), [[0, 1], [2, 3]])

    def test_reshape(self):
        x = self.x.reshape(1, 2, 2)
        assert_equal(x(0.0), [[[0, 0], [0, 0]]])
        assert_equal(x(1.0), [[[0, 1], [2, 3]]])

    def test_broadcast(self):
        x = self.x.broadcast_to(2, 2, 2)
        assert_equal(x(0.0), [[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
        assert_equal(x(1.0), [[[0, 1], [2, 3]], [[0, 1], [2, 3]]])

    def test_conj(self):
        f = lambda t: t * jnp.array([1 + 1j, 2 + 2j])
        x = timecallable(f)
        x = x.conj()
        assert_equal(x(1.0), [1 - 1j, 2 - 2j])

    def test_neg(self):
        x = -self.x
        assert_equal(x(0.0), [[0, 0], [0, 0]])
        assert_equal(x(1.0), [[0, -1], [-2, -3]])

    def test_mul(self):
        x = self.x * 2
        assert_equal(x(0.0), [[0, 0], [0, 0]])
        assert_equal(x(1.0), [[0, 2], [4, 6]])

    def test_rmul(self):
        x = 2 * self.x
        assert_equal(x(0.0), [[0, 0], [0, 0]])
        assert_equal(x(1.0), [[0, 2], [4, 6]])

    def test_add(self):
        # test type `ArrayLike`
        x = self.x + constant(jnp.ones_like(self.x))
        assert isinstance(x, SummedTimeQArray)
        assert_equal(x(0.0), [[1, 1], [1, 1]])
        assert_equal(x(1.0), [[1, 2], [3, 4]])

        # test type `ConstantTimeQArray`
        y = constant(jnp.arange(4).reshape(2, 2))
        x = self.x + y
        assert isinstance(x, SummedTimeQArray)
        assert_equal(x(0.0), [[0, 1], [2, 3]])
        assert_equal(x(1.0), [[0, 2], [4, 6]])

        # test type `CallableTimeQArray` (skipped for now)
        x = self.x + self.x
        assert isinstance(x, SummedTimeQArray)
        assert_equal(x(0.0), [[0, 0], [0, 0]])
        assert_equal(x(1.0), [[0, 2], [4, 6]])

    def test_radd(self):
        # test type `ArrayLike`
        x = jnp.ones_like(self.x) + self.x
        assert isinstance(x, SummedTimeQArray)
        assert_equal(x(0.0), [[1, 1], [1, 1]])
        assert_equal(x(1.0), [[1.0, 2.0], [3.0, 4.0]])

        # test type `ConstantTimeQArray`
        x = constant(jnp.ones_like(x)) + self.x
        assert isinstance(x, SummedTimeQArray)
        assert_equal(x(0.0), [[1, 1], [1, 1]])
        assert_equal(x(1.0), [[1, 2], [3, 4]])


@pytest.mark.run(order=TEST_SHORT)
class TestPWCTimeQArray:
    @pytest.fixture(autouse=True)
    def _setup(self):
        times = jnp.array([0, 1, 2, 3])
        values = jnp.array([1, 10 + 1j, 100])
        array = jnp.array([[1, 2], [3, 4]])

        self.x = pwc(times, values, array)  # shape at t: (2, 2)

    @pytest.mark.skip('broken test')
    def test_jit(self):
        x = jax.jit(self.x)
        assert_equal(x(-0.1), [[0, 0], [0, 0]])
        assert_equal(x(0.0), [[1, 2], [3, 4]])
        assert_equal(x(1.0), [[10 + 1j, 20 + 1j], [30 + 1j, 40 + 1j]])

        t1 = timeit.timeit(lambda: x(1.0), number=1000)
        t2 = timeit.timeit(lambda: self.x(1.0), number=1000)
        assert t1 < t2

    def test_call(self):
        assert_equal(self.x(-0.1), [[0, 0], [0, 0]])
        assert_equal(self.x(0.0), [[1, 2], [3, 4]])
        assert_equal(self.x(0.5), [[1, 2], [3, 4]])
        assert_equal(self.x(0.999), [[1, 2], [3, 4]])
        assert_equal(self.x(1.0), [[10 + 1j, 20 + 2j], [30 + 3j, 40 + 4j]])
        assert_equal(self.x(1.999), [[10 + 1j, 20 + 2j], [30 + 3j, 40 + 4j]])
        assert_equal(self.x(3.0), [[0, 0], [0, 0]])
        assert_equal(self.x(5.0), [[0, 0], [0, 0]])

    def test_reshape(self):
        x = self.x.reshape(1, 2, 2)
        assert_equal(x(-0.1), [[[0, 0], [0, 0]]])
        assert_equal(x(0.0), [[[1, 2], [3, 4]]])

    def test_broadcast(self):
        x = self.x.broadcast_to(2, 2, 2)
        assert_equal(x(-0.1), [[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
        assert_equal(x(0.0), [[[1, 2], [3, 4]], [[1, 2], [3, 4]]])

    def test_conj(self):
        x = self.x.conj()
        assert_equal(x(1.0), [[10 - 1j, 20 - 2j], [30 - 3j, 40 - 4j]])

    def test_neg(self):
        x = -self.x
        assert_equal(x(0.0), [[-1, -2], [-3, -4]])

    def test_mul(self):
        x = self.x * 2
        assert_equal(x(0.0), [[2, 4], [6, 8]])

    def test_rmul(self):
        x = 2 * self.x
        assert_equal(x(0.0), [[2, 4], [6, 8]])

    def test_add(self):
        # test type `ArrayLike`
        x = self.x + jnp.ones_like(self.x)
        assert isinstance(x, SummedTimeQArray)
        assert_equal(x(0.0), [[2, 3], [4, 5]])

        # test type `ConstantTimeQArray`
        y = constant(jnp.array([[1, 1], [1, 1]]))
        x = self.x + y
        assert isinstance(x, SummedTimeQArray)
        assert_equal(x(-0.1), [[1, 1], [1, 1]])
        assert_equal(x(0.0), [[2, 3], [4, 5]])

        # test type `PWCTimeQArray`
        x = self.x + self.x
        assert isinstance(x, SummedTimeQArray)
        assert_equal(x(0.0), [[2, 4], [6, 8]])

    def test_radd(self):
        # test type `ArrayLike`
        x = jnp.ones_like(self.x) + self.x
        assert isinstance(x, SummedTimeQArray)
        assert_equal(x(0.0), [[2, 3], [4, 5]])

        # test type `ConstantTimeQArray`
        y = constant(jnp.array([[1, 1], [1, 1]]))
        x = y + self.x
        assert isinstance(x, SummedTimeQArray)
        assert_equal(x(-0.1), [[1, 1], [1, 1]])
        assert_equal(x(0.0), [[2, 3], [4, 5]])


@pytest.mark.run(order=TEST_SHORT)
class TestModulatedTimeQArray:
    @pytest.fixture(autouse=True)
    def _setup(self):
        one = jnp.array(1.0)
        eps = lambda t: (0.5 * t + 1.0j) * one
        array = jnp.array([[1, 2], [3, 4]])

        self.x = modulated(eps, array)

    def test_call(self):
        assert_equal(self.x(0.0), [[1.0j, 2.0j], [3.0j, 4.0j]])
        assert_equal(self.x(2.0), [[1.0 + 1.0j, 2.0 + 2.0j], [3.0 + 3.0j, 4.0 + 4.0j]])

    def test_reshape(self):
        x = self.x.reshape(1, 2, 2)
        assert_equal(x(0.0), [[[1.0j, 2.0j], [3.0j, 4.0j]]])
        assert_equal(x(2.0), [[[1.0 + 1.0j, 2.0 + 2.0j], [3.0 + 3.0j, 4.0 + 4.0j]]])

    def test_broadcast(self):
        x = self.x.broadcast_to(2, 2, 2)
        assert_equal(
            x(0.0), [[[1.0j, 2.0j], [3.0j, 4.0j]], [[1.0j, 2.0j], [3.0j, 4.0j]]]
        )
        assert_equal(
            x(2.0),
            [
                [[1.0 + 1.0j, 2.0 + 2.0j], [3.0 + 3.0j, 4.0 + 4.0j]],
                [[1.0 + 1.0j, 2.0 + 2.0j], [3.0 + 3.0j, 4.0 + 4.0j]],
            ],
        )

    def test_conj(self):
        x = self.x.conj()
        assert_equal(x(0.0), [[-1.0j, -2.0j], [-3.0j, -4.0j]])

    def test_neg(self):
        x = -self.x
        assert_equal(x(0.0), [[-1.0j, -2.0j], [-3.0j, -4.0j]])

    def test_mul(self):
        x = self.x * 2
        assert_equal(x(0.0), [[2.0j, 4.0j], [6.0j, 8.0j]])

    def test_rmul(self):
        x = 2 * self.x
        assert_equal(x(0.0), [[2.0j, 4.0j], [6.0j, 8.0j]])

    def test_add(self):
        # test type `ArrayLike`
        x = self.x + jnp.ones_like(self.x)
        assert isinstance(x, SummedTimeQArray)
        assert_equal(x(0.0), [[1.0 + 1.0j, 1.0 + 2.0j], [1.0 + 3.0j, 1.0 + 4.0j]])

        # test type `ConstantTimeQArray`
        y = constant(jnp.array([[1, 1], [1, 1]]))
        x = self.x + y
        assert isinstance(x, SummedTimeQArray)
        assert_equal(x(0.0), [[1.0 + 1.0j, 1.0 + 2.0j], [1.0 + 3.0j, 1.0 + 4.0j]])

        # test type `ModulatedTimeQArray`
        x = self.x + self.x
        assert isinstance(x, SummedTimeQArray)
        assert_equal(x(0.0), [[2.0j, 4.0j], [6.0j, 8.0j]])

    def test_radd(self):
        # test type `ArrayLike`
        x = jnp.ones_like(self.x) + self.x
        assert isinstance(x, SummedTimeQArray)
        assert_equal(x(0.0), [[1.0 + 1.0j, 1.0 + 2.0j], [1.0 + 3.0j, 1.0 + 4.0j]])

        # test type `ConstantTimeQArray`
        y = constant(jnp.array([[1, 1], [1, 1]]))
        x = y + self.x
        assert isinstance(x, SummedTimeQArray)
        assert_equal(x(0.0), [[1.0 + 1.0j, 1.0 + 2.0j], [1.0 + 3.0j, 1.0 + 4.0j]])

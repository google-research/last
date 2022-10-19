# Copyright 2022 The LAST Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Semirings."""

from collections.abc import Sequence
import dataclasses
import functools
from typing import Any, Callable, Generic, Optional, TypeVar

import jax
import jax.numpy as jnp

# Types.
DType = Any
T = TypeVar('T')
S = TypeVar('S')


class Semiring(Generic[T]):
  """Base Semiring interface.

  See https://en.wikipedia.org/wiki/Semiring for what a semiring is. A Semiring
  object holds methods that implement the semiring operations. To simplify
  non-semiring operations on the semiring values, the semiring values are not
  typed: for most basic semirings, each value is a single ndarray; for some more
  complex semirings (e.g. Expectation), the values can be a tuple of ndarrays.

  Semiring is not an abstract base class because we allow operations to be
  unimplemented.

  Note: Reductions (prod & sum) can be tricky to implement right, here are two
  important things to watch out for:
  *   `axis` can be in the range [-rank, rank).
  *   The input can have 0-sized dimensions.
  """

  def zeros(self, shape: Sequence[int], dtype: Optional[DType] = None) -> T:
    """Semiring zeros in the given shape and dtype."""
    raise NotImplementedError

  def ones(self, shape: Sequence[int], dtype: Optional[DType] = None) -> T:
    """Semiring ones in the given shape and dtype."""
    raise NotImplementedError

  def times(self, a: T, b: T) -> T:
    """Semiring multiplication between two values."""
    raise NotImplementedError

  def plus(self, a: T, b: T) -> T:
    """Semiring addition between two values."""
    raise NotImplementedError

  def prod(self, a: T, axis: int) -> T:
    """Semiring multiplication along a single axis."""
    raise NotImplementedError

  def sum(self, a: T, axis: int) -> T:
    """Semiring addition along a single axis."""
    raise NotImplementedError


class _Real(Semiring[jnp.ndarray]):
  """Real semiring."""

  @staticmethod
  def zeros(shape: Sequence[int], dtype: Optional[DType] = None) -> jnp.ndarray:
    return jnp.zeros(shape, dtype)

  @staticmethod
  def ones(shape: Sequence[int], dtype: Optional[DType] = None) -> jnp.ndarray:
    return jnp.ones(shape, dtype)

  @staticmethod
  def times(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return a * b

  @staticmethod
  def plus(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return a + b

  @staticmethod
  def prod(a: jnp.ndarray, axis: int) -> jnp.ndarray:
    return jnp.prod(a, axis=axis)

  @staticmethod
  def sum(a: jnp.ndarray, axis: int) -> jnp.ndarray:
    return jnp.sum(a, axis=axis)


Real = _Real()


def _check_axis(a: jnp.ndarray, axis: int) -> None:
  if not isinstance(axis, int):
    raise ValueError(f'Only int axis is supported, got axis={axis!r}')
  if not -a.ndim <= axis < a.ndim:
    raise ValueError(
        f'Invalid reduction axis={axis!r} for input shape {a.shape}')


class _Log(Semiring[jnp.ndarray]):
  """Log semiring."""

  @staticmethod
  def zeros(shape: Sequence[int], dtype: Optional[DType] = None) -> jnp.ndarray:
    return jnp.full(shape, -jnp.inf, dtype)

  @staticmethod
  def ones(shape: Sequence[int], dtype: Optional[DType] = None) -> jnp.ndarray:
    return jnp.zeros(shape, dtype)

  @staticmethod
  def times(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return a + b

  @staticmethod
  def plus(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return _logaddexp(a, b)

  @staticmethod
  def prod(a: jnp.ndarray, axis: int) -> jnp.ndarray:
    return jnp.sum(a, axis=axis)

  @classmethod
  def sum(cls, a: jnp.ndarray, axis: int) -> jnp.ndarray:
    _check_axis(a, axis)
    # Special handling is needed because jnp.max (used in _logsumexp) doesn't
    # support reduction on 0-sized dimensions.
    if a.size > 0:
      return _logsumexp(a, axis=axis)
    # Summing empty input should result in zeros.
    if axis < 0:
      axis += a.ndim
    result_shape = a.shape[:axis] + a.shape[axis + 1:]
    return cls.zeros(result_shape, a.dtype)


# Specialized log{add,sum}exp with safe gradients.
#
# Scenarios:
# -   All operands are finite: As expected.
# -   All operands are -inf: Sum should be -inf. Gradient should be 0.
# -   All operands are +inf: Sum should be +inf. Gradient should be NaN.
# -   Mixed finite & -inf operands: Sum as expected. Gradient should be 0 for
#     -inf; non-0 for others.
# -   Mixed finite & +inf operands: Sum should +inf. Gradient should be NaN for
#     +inf; 0 for others.
# -   Mixed -inf & +inf operands: Sum should be +inf. Gradient should be NaN for
#     +inf; 0 for -inf.
# -   Mixed finite, -inf & +inf operands: Sum should be +inf. Gradient should be
#     NaN for +inf; 0 for others.
#
# The different treatment of -inf & +inf comes from their different sources.
# -   +inf is an indicator of a true error, e.g. an overflow somewhere. It's
#     thus desirabled to not silence such issues.
# -   -inf often arises from perfectly legitimate computations such as
#     `logaddexp(-inf, -inf + x)`, where `x` should not receive a NaN gradient.


@jax.custom_vjp
def _logaddexp(a, b):
  return _logaddexp_fwd(a, b)[0]


def _logaddexp_fwd(a, b):
  c = jnp.maximum(a, b)
  safe = jnp.isfinite(c)
  c = jnp.where(safe, c, 0)
  ea = jnp.exp(a - c)
  eb = jnp.exp(b - c)
  z = ea + eb
  return c + jnp.log(z), (ea, eb, z)


def _logaddexp_bwd(res, g):
  ea, eb, z = res
  safe = z != 0
  z = jnp.where(safe, z, 1)
  scale = g / z
  return scale * ea, scale * eb


_logaddexp.defvjp(_logaddexp_fwd, _logaddexp_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def _logsumexp(a, axis):
  return _logsumexp_fwd(a, axis)[0]


def _logsumexp_fwd(a, axis):
  c = jnp.max(a, axis=axis, keepdims=True)
  safe = jnp.isfinite(c)
  c = jnp.where(safe, c, 0)
  e = jnp.exp(a - c)
  z = jnp.sum(e, axis=axis, keepdims=True)
  r = jnp.squeeze(c, axis=axis) + jnp.log(jnp.squeeze(z, axis=axis))
  return r, (e, z)


def _logsumexp_bwd(axis, res, g):
  e, z = res
  safe = z != 0
  z = jnp.where(safe, z, 1)
  g = jnp.expand_dims(g, axis=axis)
  # g & z are smaller than e, doing the division between g & z instead e & z is
  # thus faster.
  return (g / z * e,)


_logsumexp.defvjp(_logsumexp_fwd, _logsumexp_bwd)

Log = _Log()


class _MaxTropical(Semiring):
  """Max tropical semiring.

  The gradients of `plus` and `sum` is guaranteed to be non-zero on exactly 1
  input element, even in the event of a tie.
  """

  @staticmethod
  def zeros(shape: Sequence[int], dtype: Optional[DType] = None) -> jnp.ndarray:
    return jnp.full(shape, -jnp.inf, dtype)

  @staticmethod
  def ones(shape: Sequence[int], dtype: Optional[DType] = None) -> jnp.ndarray:
    return jnp.zeros(shape, dtype)

  @staticmethod
  def times(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return a + b

  @staticmethod
  def plus(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return _maximum(a, b)

  @staticmethod
  def prod(a: jnp.ndarray, axis: int) -> jnp.ndarray:
    return jnp.sum(a, axis=axis)

  @classmethod
  def sum(cls, a: jnp.ndarray, axis: int) -> jnp.ndarray:
    _check_axis(a, axis)
    # Special handling is needed because jnp.max doesn't support reduction on
    # 0-sized dimensions.
    if a.size > 0:
      return _max(a, axis=axis)
    # Summing empty input should result in zeros.
    if axis < 0:
      axis += a.ndim
    result_shape = a.shape[:axis] + a.shape[axis + 1:]
    return cls.zeros(result_shape, a.dtype)


MaxTropical = _MaxTropical()


@jax.custom_vjp
def _maximum(a, b):
  return _maximum_fwd(a, b)[0]


def _maximum_fwd(a, b):
  return jnp.maximum(a, b), a >= b


def _maximum_bwd(res, g):
  choose_a = res
  return g * choose_a, g * (1 - choose_a)


_maximum.defvjp(_maximum_fwd, _maximum_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def _max(a, axis):
  return _max_fwd(a, axis)[0]


def _max_fwd(a, axis):
  argmax = jnp.argmax(a, axis)
  width = a.shape[axis]
  return jnp.max(a, axis), (argmax, width)


def _max_bwd(axis, res, g):
  argmax, width = res
  mask = jax.nn.one_hot(argmax, width, axis=axis, dtype=g.dtype)
  g = jnp.expand_dims(g, axis=axis)
  return (g * mask,)


_max.defvjp(_max_fwd, _max_bwd)


@dataclasses.dataclass(frozen=True)
class Expectation(Generic[T, S], Semiring[tuple[T, S]]):
  """Jason Eisner's expectation semiring.

  In most cases, use LogLogExpectation below directly.

  See https://www.cs.jhu.edu/~jason/papers/eisner.fsmnlp01.pdf for reference.

  Each semiring value is a tuple (w, x):
  -   w: The weight of this tuple, expressed in the self.w semiring.
  -   x: The weighted sum for some corresponding weighted values, expressed in
      the self.x semiring.

  To create a semiring value from a weight-value pair, use
  `self.weighted()`. See `ExpectationTest.test_entropy` for an
  example of using the expectation semiring to compute the entropy of
  probability distributions.

  Attributes:
    w: Semiring for representing weights.
    x: Semiring for representing weighted sums.
    w_to_x: Function to convert a value from semiring `w` to semiring `x`.
  """
  w: Semiring[T]
  x: Semiring[S]
  w_to_x: Callable[[T], S]

  def weighted(self, w: T, v: S) -> tuple[T, S]:
    return w, self.x.times(self.w_to_x(w), v)

  def zeros(self,
            shape: Sequence[int],
            dtype: Optional[DType] = None) -> tuple[T, S]:
    return self.w.zeros(shape, dtype), self.x.zeros(shape, dtype)

  def ones(self,
           shape: Sequence[int],
           dtype: Optional[DType] = None) -> tuple[T, S]:
    return self.w.ones(shape, dtype), self.x.zeros(shape, dtype)

  def times(self, a: tuple[T, S], b: tuple[T, S]) -> tuple[T, S]:
    w_a, x_a = a
    w_b, x_b = b
    w = self.w.times(w_a, w_b)
    x = self.x.plus(
        self.x.times(self.w_to_x(w_a), x_b),
        self.x.times(self.w_to_x(w_b), x_a))
    return w, x

  def plus(self, a: tuple[T, S], b: tuple[T, S]) -> tuple[T, S]:
    w_a, x_a = a
    w_b, x_b = b
    w = self.w.plus(w_a, w_b)
    x = self.x.plus(x_a, x_b)
    return w, x

  def sum(self, a: tuple[T, S], axis: int) -> tuple[T, S]:
    w, x = a
    w = self.w.sum(w, axis)
    x = self.x.sum(x, axis)
    return w, x


# Expectation semiring with weight and weighted sum represented both using the
# Log semiring. Therefore only summation on non-negative value is allowed.
LogLogExpectation = Expectation(w=Log, x=Log, w_to_x=lambda x: x)

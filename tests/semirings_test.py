# Copyright 2025 The LAST Authors.
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

"""Tests for semirings."""

from absl.testing import absltest

import jax
import jax.numpy as jnp
from last import semirings
import numpy.testing as npt


def zero_and_one_test(semiring):
  jax.config.update('jax_debug_nans', True)
  try:
    one = semiring.ones([3])
    zero = semiring.zeros([3])
    xs = jnp.array([1., 2., 3.])

    for args in [(one, xs), (xs, one)]:
      npt.assert_array_equal(semiring.times(*args), xs)
      npt.assert_array_equal(semiring.prod(jnp.stack(args), axis=0), xs)

    for args in [(zero, xs), (xs, zero)]:
      npt.assert_array_equal(semiring.plus(*args), xs)
      npt.assert_array_equal(semiring.sum(jnp.stack(args), axis=0), xs)

    npt.assert_array_equal(
        semiring.times(semiring.ones((1, 2)), semiring.zeros((3, 1))),
        semiring.zeros((3, 2)))
    npt.assert_array_equal(
        semiring.times(semiring.zeros((1, 2)), semiring.ones((3, 1))),
        semiring.zeros((3, 2)))
    npt.assert_array_equal(
        semiring.times(semiring.ones((1, 2)), semiring.ones((3, 1))),
        semiring.ones((3, 2)))
    npt.assert_array_equal(
        semiring.times(semiring.zeros((1, 2)), semiring.zeros((3, 1))),
        semiring.zeros((3, 2)))

    npt.assert_array_equal(
        semiring.plus(semiring.ones((1, 2)), semiring.zeros((3, 1))),
        semiring.ones((3, 2)))
    npt.assert_array_equal(
        semiring.plus(semiring.zeros((1, 2)), semiring.ones((3, 1))),
        semiring.ones((3, 2)))
    npt.assert_array_equal(
        semiring.plus(semiring.zeros((1, 2)), semiring.zeros((3, 1))),
        semiring.zeros((3, 2)))

    npt.assert_array_equal(
        semiring.sum(jnp.zeros([3, 0]), axis=0), jnp.zeros([0]))
    npt.assert_array_equal(
        semiring.prod(jnp.zeros([3, 0]), axis=0), jnp.zeros([0]))

    npt.assert_array_equal(semiring.sum(jnp.zeros([3, 0]), axis=1), zero)
    npt.assert_array_equal(semiring.prod(jnp.zeros([3, 0]), axis=1), one)
  finally:
    jax.config.update('jax_debug_nans', False)


def binary_op_broadcasting_test(semiring):

  def expected(op, x, y):
    expected_z, expected_vjp_fn = jax.vjp(
        lambda x, y: op(*jnp.broadcast_arrays(x, y)), x, y
    )
    expected_dx, expected_dy = expected_vjp_fn(jnp.ones_like(expected_z))
    return expected_z, expected_dx, expected_dy

  for op in [semiring.times, semiring.plus]:
    for shapes in [
        ([], [2]),
        ([1], [2]),
        ([1, 2], [3, 2]),
        ([2, 1], [2, 3]),
        ([3], [2, 3]),
    ]:
      for shape_x, shape_y in [shapes, shapes[::-1]]:
        err_msg = f'op={op} shapes={(shape_x, shape_y)}'
        x = semiring.ones(shape_x)
        y = semiring.ones(shape_y)
        z, vjp_fn = jax.vjp(op, x, y)
        dx, dy = vjp_fn(jnp.ones_like(z))
        expected_z, expected_dx, expected_dy = expected(op, x, y)
        npt.assert_allclose(z, expected_z, err_msg=err_msg)
        npt.assert_allclose(dx, expected_dx, err_msg=err_msg)
        npt.assert_allclose(dy, expected_dy, err_msg=err_msg)


class SemiringTest(absltest.TestCase):

  def test_value_shape(self):
    self.assertEqual(semirings.value_shape(jnp.zeros([1, 2])), (1, 2))
    self.assertEqual(
        semirings.value_shape({'a': jnp.zeros([1, 2]), 'b': jnp.ones([1, 2])}),
        (1, 2),
    )
    with self.assertRaisesRegex(
        ValueError, 'No common shape can be derived for an empty PyTree'
    ):
      semirings.value_shape(None)
    with self.assertRaisesRegex(
        ValueError,
        'A semiring value must consist of ndarrays of a common shape',
    ):
      semirings.value_shape({'a': jnp.zeros([1, 2]), 'b': jnp.ones([2])})


class RealTest(absltest.TestCase):

  def test_basics(self):
    npt.assert_array_equal(semirings.Real.times(jnp.array(2), jnp.array(3)), 6)
    npt.assert_array_equal(semirings.Real.prod(jnp.array([2, 3]), axis=0), 6)
    npt.assert_array_equal(semirings.Real.plus(jnp.array(2), jnp.array(3)), 5)
    npt.assert_array_equal(semirings.Real.sum(jnp.array([2, 3]), axis=0), 5)
    zero_and_one_test(semirings.Real)
    binary_op_broadcasting_test(semirings.Real)


def check_sum_axis(self, semiring):
  """Checks that semiring sum handles axes correctly."""
  xs = jnp.arange(2 * 3 * 4 * 5, dtype=jnp.float32).reshape([2, 3, 4, 5])

  with self.subTest('forward'):
    self.assertEqual(semiring.sum(xs, axis=0).shape, (3, 4, 5))
    self.assertEqual(semiring.sum(xs, axis=1).shape, (2, 4, 5))
    self.assertEqual(semiring.sum(xs, axis=2).shape, (2, 3, 5))
    self.assertEqual(semiring.sum(xs, axis=3).shape, (2, 3, 4))
    self.assertEqual(semiring.sum(xs, axis=-1).shape, (2, 3, 4))
    self.assertEqual(semiring.sum(xs, axis=-2).shape, (2, 3, 5))
    self.assertEqual(semiring.sum(xs, axis=-3).shape, (2, 4, 5))
    self.assertEqual(semiring.sum(xs, axis=-4).shape, (3, 4, 5))
    with self.assertRaisesRegex(ValueError, 'Invalid reduction axis'):
      semiring.sum(xs, axis=4)
    with self.assertRaisesRegex(ValueError, 'Invalid reduction axis'):
      semiring.sum(xs, axis=-5)
    with self.assertRaisesRegex(ValueError, 'Only int axis'):
      semiring.sum(xs, axis=None)  # type: ignore

  with self.subTest('backward'):

    @jax.grad
    def f(xs, axis):
      zs = semiring.sum(xs, axis=axis)
      while zs.shape:
        zs = jnp.sum(zs, axis=0)
      return zs

    for axis in range(-4, 4):
      self.assertEqual(f(xs, axis=axis).shape, xs.shape)


def check_sum_zero_sized(self, semiring):
  """Checks that semiring sum handles zero-sized dimensions correctly."""
  xs = jnp.zeros([0, 2])

  npt.assert_array_equal(semiring.sum(xs, axis=0), semiring.zeros([2]))
  npt.assert_array_equal(semiring.sum(xs, axis=-2), semiring.zeros([2]))

  self.assertEqual(semiring.sum(xs, axis=1).shape, (0,))
  self.assertEqual(semiring.sum(xs, axis=-1).shape, (0,))


class LogTest(absltest.TestCase):

  def test_basics(self):
    npt.assert_array_equal(semirings.Log.times(jnp.array(2), jnp.array(3)), 5)
    self.assertEqual(semirings.Log.prod(jnp.array([2, 3]), axis=0), 5)
    npt.assert_allclose(
        semirings.Log.plus(jnp.array(2), jnp.array(3)), 3.31326169
    )
    npt.assert_allclose(
        semirings.Log.sum(jnp.array([2, 3]), axis=0), 3.31326169)
    zero_and_one_test(semirings.Log)
    binary_op_broadcasting_test(semirings.Log)

  def test_times_safety(self):
    inf = jnp.array(jnp.inf)
    self.assertTrue(jnp.isnan(semirings.Log.times(-inf, inf)))
    self.assertTrue(jnp.isnan(semirings.Log.times(inf, -inf)))
    npt.assert_array_equal(semirings.Log.times(inf, jnp.array(1)), inf)
    npt.assert_array_equal(semirings.Log.times(jnp.array(1), inf), inf)

  def test_prod_safety(self):
    inf = jnp.array(jnp.inf)
    self.assertTrue(
        jnp.isnan(semirings.Log.prod(jnp.array([-inf, inf]), axis=0)))
    self.assertTrue(
        jnp.isnan(semirings.Log.prod(jnp.array([inf, -inf]), axis=0)))
    npt.assert_array_equal(semirings.Log.prod(jnp.array([inf, 1]), axis=0), inf)
    npt.assert_array_equal(semirings.Log.prod(jnp.array([1, inf]), axis=0), inf)

  def test_plus_safety(self):
    inf = jnp.array(jnp.inf)
    npt.assert_array_equal(semirings.Log.plus(-inf, inf), inf)
    npt.assert_array_equal(semirings.Log.plus(inf, -inf), inf)
    npt.assert_array_equal(semirings.Log.plus(inf, jnp.array(1)), inf)
    npt.assert_array_equal(semirings.Log.plus(jnp.array(1), inf), inf)

  def test_sum_safety(self):
    inf = jnp.array(jnp.inf)
    npt.assert_array_equal(
        semirings.Log.sum(jnp.array([-inf, inf]), axis=0), inf)
    npt.assert_array_equal(
        semirings.Log.sum(jnp.array([inf, -inf]), axis=0), inf)
    npt.assert_array_equal(semirings.Log.sum(jnp.array([inf, 1]), axis=0), inf)
    npt.assert_array_equal(semirings.Log.sum(jnp.array([1, inf]), axis=0), inf)
    npt.assert_array_equal(
        semirings.Log.sum(jnp.array([1, inf, -inf]), axis=0), inf)

  def test_log_plus_grad(self):
    inf = jnp.array(jnp.inf)
    plus_grad = jax.grad(lambda xy: semirings.Log.plus(*xy))
    for x, y, dx, dy in [
        (1., 1., 0.5, 0.5),
        (1., 2., 0.2689414213699951, 0.7310585786300049),
        (2., 1., 0.7310585786300049, 0.2689414213699951),
        (-inf, -inf, 0., 0.),
        (1., -inf, 1., 0.),
        (-inf, 1., 0., 1.),
    ]:
      with self.subTest(f'x={x},y={y}'):
        dx_, dy_ = plus_grad((x, y))
        npt.assert_allclose(dx_, dx)
        npt.assert_allclose(dy_, dy)
    for x, y in [(inf, inf), (1., inf), (inf, 1.), (inf, -inf), (-inf, inf)]:
      with self.subTest(f'x={x},y={y}'):
        dx_, dy_ = plus_grad((x, y))
        if x == inf:
          self.assertTrue(jnp.isnan(dx_))
        else:
          self.assertEqual(dx_, 0)
        if y == inf:
          self.assertTrue(jnp.isnan(dy_))
        else:
          self.assertEqual(dy_, 0)
    with self.subTest('plus & times'):

      @jax.grad
      def f(x):
        return semirings.Log.plus(-jnp.inf, semirings.Log.times(-jnp.inf, x))  # pytype: disable=wrong-arg-types  # jnp-type

      self.assertEqual(f(1.), 0.)

  def test_log_sum_grad(self):
    inf = jnp.array(jnp.inf)
    sum_grad = jax.grad(lambda xs: jnp.sum(semirings.Log.sum(xs, axis=0)))
    for xs, dxs in [
        ([1., 1.], [0.5, 0.5]),
        ([1., 2.], [0.2689414213699951, 0.7310585786300049]),
        ([2., 1.], [0.7310585786300049, 0.2689414213699951]),
        ([-inf, -inf], [0., 0.]),
        ([1., -inf], [1., 0.]),
        ([-inf, 1.], [0., 1.]),
        ([-inf, 1., 2.], [0, 0.2689414213699951, 0.7310585786300049]),
    ]:
      with self.subTest(f'xs={xs}'):
        dxs_ = sum_grad(jnp.array(xs))
        npt.assert_allclose(dxs_, dxs)
    for xs in [(inf, inf), (1, inf), (inf, 1), (inf, -inf), (-inf, inf),
               (inf, -inf, 1)]:
      with self.subTest(f'xs={xs}'):
        xs = jnp.array(xs)
        dxs_ = sum_grad(xs)
        npt.assert_array_equal(jnp.isnan(dxs_), xs == inf)
        npt.assert_array_equal(jnp.where(xs != inf, dxs_, 0), 0)

    with self.subTest('sum & prod'):

      @jax.grad
      def f(x):
        return semirings.Log.sum(
            jnp.stack([
                -jnp.inf,
                semirings.Log.prod(jnp.stack([-jnp.inf, x]), axis=0)
            ]),
            axis=0)

      self.assertEqual(f(1.), 0.)

  def test_log_sum_axis(self):
    check_sum_axis(self, semirings.Log)

  def test_log_sum_zero_sized(self):
    check_sum_zero_sized(self, semirings.Log)


class MaxTropicalTest(absltest.TestCase):

  def test_basics(self):
    npt.assert_array_equal(
        semirings.MaxTropical.times(jnp.array(2), jnp.array(3)), 5
    )
    npt.assert_array_equal(
        semirings.MaxTropical.prod(jnp.array([2, 3]), axis=0), 5
    )
    npt.assert_array_equal(
        semirings.MaxTropical.plus(jnp.array(2), jnp.array(3)), 3
    )
    npt.assert_array_equal(
        semirings.MaxTropical.sum(jnp.array([2, 3]), axis=0), 3
    )
    zero_and_one_test(semirings.MaxTropical)
    binary_op_broadcasting_test(semirings.MaxTropical)

  def test_plus_grad(self):
    npt.assert_array_equal(
        jax.grad(lambda a: jnp.sum(semirings.MaxTropical.plus(a[0], a[1])))(
            [jnp.array([1., 2., 3.]),
             jnp.array([0., 2., 4.])]), [[1., 1., 0.], [0., 0., 1.]])

  def test_sum_grad(self):
    npt.assert_array_equal(
        jax.grad(lambda a: jnp.sum(semirings.MaxTropical.sum(a, axis=0)))(
            jnp.array([[1., 2., 3.], [0., 2., 4.]])),
        [[1., 1., 0.], [0., 0., 1.]])
    npt.assert_array_equal(
        jax.grad(lambda a: jnp.sum(semirings.MaxTropical.sum(a, axis=1)))(
            jnp.array([[1., 2., 3.], [0., 2., 4.]]).T),
        jnp.array([[1., 1., 0.], [0., 0., 1.]]).T)

  def test_sum_axis(self):
    check_sum_axis(self, semirings.MaxTropical)

  def test_sum_zero_sized(self):
    check_sum_zero_sized(self, semirings.MaxTropical)


class ExpectationTest(absltest.TestCase):

  def test_basics(self):
    one = semirings.LogLogExpectation.ones([])
    zero = semirings.LogLogExpectation.zeros([])
    for wx in [
        semirings.LogLogExpectation.weighted(jnp.array(1.), jnp.array(2.)), one,
        zero
    ]:
      with self.subTest(str(wx)):
        jax.tree_util.tree_map(npt.assert_array_equal,
                               semirings.LogLogExpectation.times(wx, one), wx)
        jax.tree_util.tree_map(npt.assert_array_equal,
                               semirings.LogLogExpectation.times(one, wx), wx)
        jax.tree_util.tree_map(npt.assert_array_equal,
                               semirings.LogLogExpectation.plus(wx, zero), wx)
        jax.tree_util.tree_map(npt.assert_array_equal,
                               semirings.LogLogExpectation.plus(zero, wx), wx)

  def test_shape_dtypes(self):
    one = semirings.LogLogExpectation.ones([1, 2], (jnp.float32, jnp.bfloat16))
    self.assertEqual(semirings.value_shape(one), (1, 2))
    self.assertEqual(semirings.value_dtype(one), (jnp.float32, jnp.bfloat16))
    zero = semirings.LogLogExpectation.zeros([], (jnp.bfloat16, jnp.float32))
    self.assertEqual(semirings.value_shape(zero), ())
    self.assertEqual(semirings.value_dtype(zero), (jnp.bfloat16, jnp.float32))

  def test_weighted(self):
    w, x = semirings.LogLogExpectation.weighted(
        jnp.log(jnp.array([0, 1, 2])), jnp.log(jnp.array([3, 4, 5])))
    npt.assert_allclose(jnp.exp(w), [0, 1, 2])
    npt.assert_allclose(jnp.exp(x), [0 * 3, 1 * 4, 2 * 5])

  def test_weighted_safety(self):
    w = jnp.array(float('-inf'))
    v = jnp.array(float('inf'))
    w, x = semirings.LogLogExpectation.weighted(w, v)
    npt.assert_array_equal(w, float('-inf'))
    npt.assert_array_equal(x, float('-inf'))

  def test_sum(self):
    w, x = semirings.LogLogExpectation.sum(
        semirings.LogLogExpectation.weighted(
            jnp.log(jnp.array([[0, 1], [2, 3]])),
            jnp.log(jnp.array([[4, 5], [6, 7]]))),
        axis=1)
    npt.assert_allclose(jnp.exp(w), [0 + 1, 2 + 3])
    npt.assert_allclose(jnp.exp(x), [0 * 4 + 1 * 5, 2 * 6 + 3 * 7], rtol=1e-6)

  def test_entropy(self):
    probs = jnp.array([0.25, 0.25, 0.5])
    log_probs = jnp.log(probs)
    wx = semirings.LogLogExpectation.weighted(log_probs, jnp.log(-log_probs))
    log_z, log_sum = semirings.LogLogExpectation.sum(wx, axis=0)
    npt.assert_allclose(log_z, 0)
    entropy = jnp.exp(log_sum)
    npt.assert_allclose(entropy, -jnp.sum(probs * log_probs))

    new_probs = jnp.array([0.25, 0.5, 0.25])
    new_log_probs = jnp.log(new_probs)
    new_wx = semirings.LogLogExpectation.weighted(new_log_probs,
                                                  jnp.log(-new_log_probs))
    log_z, log_sum = semirings.LogLogExpectation.sum(
        semirings.LogLogExpectation.times(wx, new_wx), axis=0)
    npt.assert_allclose(jnp.exp(log_z), jnp.sum(probs * new_probs))
    entropy = log_z + jnp.exp(log_sum - log_z)
    npt.assert_allclose(
        entropy, -jnp.sum(probs * new_probs * jnp.exp(-log_z) *
                          (log_probs + new_log_probs - log_z)))


class CartesianTest(absltest.TestCase):

  def test_basics(self):
    semiring = semirings.Cartesian(semirings.Real, semirings.MaxTropical)
    one = semiring.ones([])
    zero = semiring.zeros([])
    for wx in [(jnp.array(1.0), jnp.array(2.0)), one, zero]:
      with self.subTest(str(wx)):
        jax.tree_util.tree_map(
            npt.assert_array_equal, semiring.times(wx, one), wx
        )
        jax.tree_util.tree_map(
            npt.assert_array_equal, semiring.times(one, wx), wx
        )
        jax.tree_util.tree_map(
            npt.assert_array_equal, semiring.plus(wx, zero), wx
        )
        jax.tree_util.tree_map(
            npt.assert_array_equal, semiring.plus(zero, wx), wx
        )

  def test_shape_dtypes(self):
    semiring = semirings.Cartesian(semirings.Real, semirings.MaxTropical)
    one = semiring.ones([1, 2], (jnp.float32, jnp.bfloat16))
    self.assertEqual(semirings.value_shape(one), (1, 2))
    self.assertEqual(semirings.value_dtype(one), (jnp.float32, jnp.bfloat16))
    zero = semiring.zeros([], (jnp.bfloat16, jnp.float32))
    self.assertEqual(semirings.value_shape(zero), ())
    self.assertEqual(semirings.value_dtype(zero), (jnp.bfloat16, jnp.float32))

  def test_arithmetics(self):
    semiring = semirings.Cartesian(semirings.Real, semirings.MaxTropical)
    a = (jnp.array(2.0), jnp.array(1.0))
    b = (jnp.array(3.0), jnp.array(4.0))
    c = (jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))

    with self.subTest('times'):
      a_times_b = semiring.times(a, b)
      self.assertIsInstance(a_times_b, tuple)
      self.assertLen(a_times_b, 2)
      npt.assert_array_equal(a_times_b[0], 6.0)
      npt.assert_array_equal(a_times_b[1], 5.0)

    with self.subTest('plus'):
      a_plus_b = semiring.plus(a, b)
      self.assertIsInstance(a_plus_b, tuple)
      self.assertLen(a_plus_b, 2)
      npt.assert_array_equal(a_plus_b[0], 5.0)
      npt.assert_array_equal(a_plus_b[1], 4.0)

    with self.subTest('sum'):
      sum_c = semiring.sum(c, axis=0)
      self.assertIsInstance(sum_c, tuple)
      self.assertLen(sum_c, 2)
      npt.assert_array_equal(sum_c[0], 3.0)
      npt.assert_array_equal(sum_c[1], 4.0)

    with self.subTest('prod'):
      prod_c = semiring.prod(c, axis=0)
      self.assertIsInstance(prod_c, tuple)
      self.assertLen(prod_c, 2)
      npt.assert_array_equal(prod_c[0], 2.0)
      npt.assert_array_equal(prod_c[1], 7.0)


if __name__ == '__main__':
  absltest.main()

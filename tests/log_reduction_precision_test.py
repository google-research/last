# Copyright 2026 The LAST Authors.
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
"""Regression tests for wide low-precision log-semiring reductions."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from last import contexts
from last import semirings
import numpy as np


class LogReductionPrecisionTest(parameterized.TestCase):

  @parameterized.product(size=(65536, 131072), axis=(0, -1))
  def test_wide_float16_reduction_and_gradients(self, size, axis):
    x = jnp.broadcast_to(
        jnp.array([0., -8.], dtype=jnp.float16)[:, None], (2, size)
    )
    if axis == 0:
      x = x.T

    def reduce(values):
      return semirings.Log.sum(values, axis=axis)

    expected = np.log(float(size)) + np.array([0., -8.])
    for fn in (reduce, jax.jit(reduce)):
      result, vjp = jax.vjp(fn, x)
      self.assertEqual(result.dtype, x.dtype)
      np.testing.assert_allclose(result, expected, rtol=5e-4, atol=4e-3)
      gradient, = vjp(jnp.array([1., 2.], dtype=x.dtype))
      reference = np.broadcast_to(np.array([1., 2.])[:, None] / size, (2, size))
      if axis == 0:
        reference = reference.T
      self.assertEqual(gradient.dtype, x.dtype)
      np.testing.assert_allclose(gradient, reference, rtol=1e-3, atol=0.)

  @parameterized.parameters(jnp.float16, jnp.bfloat16, jnp.float32)
  def test_finite_and_zero_weight_gradients(self, dtype):
    x = jnp.array(
        [[0., -jnp.inf, 0.], [-jnp.inf, -jnp.inf, -jnp.inf]], dtype=dtype
    )
    result, vjp = jax.vjp(lambda a: semirings.Log.sum(a, axis=-1), x)
    self.assertEqual(result.dtype, dtype)
    np.testing.assert_allclose(result[0], np.log(2.), rtol=4e-3)
    self.assertEqual(float(result[1]), -np.inf)
    gradient, = vjp(jnp.ones(2, dtype=dtype))
    np.testing.assert_array_equal(gradient, [[.5, 0., .5], [0., 0., 0.]])

  def test_full_ngram_forward_reduction(self):
    context = contexts.FullNGram(vocab_size=65536, context_size=0)
    weights = jnp.zeros(context.shape(), dtype=jnp.float16)

    def forward(values):
      return context.forward_reduce(values, semirings.Log)

    result, vjp = jax.vjp(jax.jit(forward), weights)
    self.assertEqual(result.shape, (1,))
    np.testing.assert_allclose(result, [np.log(65536.)], rtol=5e-4)
    gradient, = vjp(jnp.ones_like(result))
    np.testing.assert_array_equal(gradient, np.full(weights.shape, 1. / 65536))


if __name__ == '__main__':
  absltest.main()

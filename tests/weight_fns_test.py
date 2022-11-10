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

"""Tests for weight_fns."""

from absl.testing import absltest
from flax import linen as nn
import jax
import jax.numpy as jnp
from last import weight_fns
import numpy.testing as npt


class WeightFnTest(absltest.TestCase):

  def test_hat_normalize(self):
    blank = jnp.array([2., 7.])
    lexical = jnp.array([[0., 1.], [3., 5.]])
    expect_blank = jnp.array([-0.126928, -0.000912])
    expect_lexical = jnp.array([[-3.44019, -2.44019], [-9.12784, -7.12784]])
    actual_blank, actual_lexical = weight_fns.hat_normalize(blank, lexical)
    npt.assert_allclose(actual_blank, expect_blank, rtol=1e-3, atol=1e-6)
    npt.assert_allclose(actual_lexical, expect_lexical, rtol=1e-6)

  def test_log_softmax_normalize(self):
    blank = jnp.array([2., 7.])
    lexical = jnp.array([[0., 1.], [3., 5.]])
    expect_blank = jnp.array([-0.407606, -0.142932])
    expect_lexical = jnp.array([[-2.407606, -1.407606], [-4.142932, -2.142932]])
    actual_blank, actual_lexical = weight_fns.log_softmax_normalize(
        blank, lexical)
    npt.assert_allclose(actual_blank, expect_blank, rtol=1e-3, atol=1e-6)
    npt.assert_allclose(actual_lexical, expect_lexical, rtol=1e-6)

  def test_NullCacher(self):
    cacher = weight_fns.NullCacher()
    cache, params = cacher.init_with_output(jax.random.PRNGKey(0))
    self.assertIsNone(cache)
    self.assertDictEqual(params.unfreeze(), {})

  def test_TableWeightFn(self):
    with self.subTest('batch ndim = 0'):
      table = jnp.arange(5 * 4 * 3).reshape([5, 4, 3])
      weight_fn = weight_fns.TableWeightFn(table)

      frame = jnp.array([1., 2.])
      (blank, lexical), params = weight_fn.init_with_output(
          jax.random.PRNGKey(0), None, frame)
      self.assertDictEqual(params.unfreeze(), {})
      npt.assert_array_equal(blank, table[1, :, 0])
      npt.assert_array_equal(lexical, table[1, :, 1:])

      state = jnp.array(3)
      blank, lexical = weight_fn.apply(params, None, frame, state)
      npt.assert_array_equal(blank, table[1, 3, 0])
      npt.assert_array_equal(lexical, table[1, 3, 1:])

      with self.assertRaisesRegex(
          ValueError, r'frame should have batch_dims=\(\) but got \(1,\)'):
        weight_fn.apply(params, None, frame[jnp.newaxis])

    with self.subTest('batch ndim = 1'):
      table = jnp.arange(2 * 5 * 4 * 3).reshape([2, 5, 4, 3])
      weight_fn = weight_fns.TableWeightFn(table)

      frame = jnp.array([[1., 2.], [4., 3.]])
      (blank, lexical), params = weight_fn.init_with_output(
          jax.random.PRNGKey(0), None, frame)
      self.assertDictEqual(params.unfreeze(), {})
      npt.assert_array_equal(blank, [table[0, 1, :, 0], table[1, 4, :, 0]])
      npt.assert_array_equal(lexical, [table[0, 1, :, 1:], table[1, 4, :, 1:]])

      state = jnp.array([3, 2])
      blank, lexical = weight_fn.apply(params, None, frame, state)
      npt.assert_array_equal(blank, [table[0, 1, 3, 0], table[1, 4, 2, 0]])
      npt.assert_array_equal(lexical, [table[0, 1, 3, 1:], table[1, 4, 2, 1:]])

      with self.assertRaisesRegex(
          ValueError, r'frame should have batch_dims=\(2,\) but got \(1, 2\)'):
        weight_fn.apply(params, None, frame[jnp.newaxis])


class LocallyNormalizedWeightFnTest(absltest.TestCase):

  def test_call(self):
    weight_fn = weight_fns.LocallyNormalizedWeightFn(
        weight_fns.JointWeightFn(vocab_size=3, hidden_size=8))
    rngs = jax.random.split(jax.random.PRNGKey(0), 5)
    frame = jax.random.uniform(rngs[0], (2, 4))
    cache = jax.random.uniform(rngs[1], (6, 5))  # context embeddings.
    params = weight_fn.init(rngs[2], cache, frame)

    with self.subTest('all context states'):
      blank, lexical = weight_fn.apply(params, cache, frame)
      npt.assert_equal(blank.shape, (2, 6))
      npt.assert_equal(lexical.shape, (2, 6, 3))
      npt.assert_allclose(
          jnp.exp(blank) + jnp.sum(jnp.exp(lexical), axis=-1),
          jnp.ones_like(blank),
          rtol=1e-4)

    with self.subTest('per-state'):
      state = jnp.array([2, 4])
      blank_per_state, lexical_per_state = weight_fn.apply(
          params, cache, frame, state)
      npt.assert_allclose(
          blank_per_state, blank[jnp.array([0, 1]), state], rtol=1e-6)
      npt.assert_allclose(
          lexical_per_state, lexical[jnp.array([0, 1]), state], rtol=1e-6)


class JointWeightFnTest(absltest.TestCase):

  def test_call(self):
    weight_fn = weight_fns.JointWeightFn(vocab_size=3, hidden_size=8)
    rngs = jax.random.split(jax.random.PRNGKey(0), 5)
    frame = jax.random.uniform(rngs[0], (2, 4))
    cache = jax.random.uniform(rngs[1], (6, 5))  # context embeddings.
    params = weight_fn.init(rngs[2], cache, frame)

    with self.subTest('all context states'):
      blank, lexical = weight_fn.apply(params, cache, frame)
      npt.assert_equal(blank.shape, (2, 6))
      npt.assert_equal(lexical.shape, (2, 6, 3))

    with self.subTest('per-state'):
      state = jnp.array([2, 4])
      blank_per_state, lexical_per_state = weight_fn.apply(
          params, cache, frame, state)
      npt.assert_allclose(
          blank_per_state,
          blank[jnp.array([0, 1]), state],
          rtol=1e-6,
          atol=1e-6)
      npt.assert_allclose(
          lexical_per_state,
          lexical[jnp.array([0, 1]), state],
          rtol=1e-6,
          atol=1e-6)

  def test_SharedEmbCacher(self):
    cacher = weight_fns.SharedEmbCacher(num_context_states=4, embedding_size=5)
    params = cacher.init(jax.random.PRNGKey(0))
    npt.assert_equal(params['params']['context_embeddings'].shape, (4, 5))
    npt.assert_array_equal(
        cacher.apply(params), params['params']['context_embeddings'])

  def test_SharedRNNCacher(self):

    pad = -2.
    start = -1.

    class FakeRNNCell(nn.recurrent.RNNCellBase):
      """Test RNN cell that remembers past inputs."""

      def __call__(self, carry, inputs):
        carry = jnp.concatenate([carry[..., 1:], inputs[..., :1]], axis=-1)
        return carry, carry

      @staticmethod
      def initialize_carry(rng, batch_dims, size):
        del rng
        return jnp.full((*batch_dims, size), pad)

    cacher = weight_fns.SharedRNNCacher(
        vocab_size=3, context_size=2, rnn_size=4, rnn_cell=FakeRNNCell())
    params = {
        'params': {
            'Embed_0': {
                'embedding':
                    jnp.broadcast_to(
                        jnp.array([start, 1., 2., 3.])[:, jnp.newaxis], (4, 4))
            }
        }
    }
    npt.assert_array_equal(
        cacher.apply(params),
        [
            # Start.
            [pad, pad, pad, start],
            # Unigrams.
            [pad, pad, start, 1],
            [pad, pad, start, 2],
            [pad, pad, start, 3],
            # Bigrams.
            [pad, start, 1, 1],
            [pad, start, 1, 2],
            [pad, start, 1, 3],
            [pad, start, 2, 1],
            [pad, start, 2, 2],
            [pad, start, 2, 3],
            [pad, start, 3, 1],
            [pad, start, 3, 2],
            [pad, start, 3, 3],
        ])


if __name__ == '__main__':
  absltest.main()

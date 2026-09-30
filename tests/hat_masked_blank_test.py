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
"""HAT normalization with disabled blank transitions."""

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
import jax
import jax.numpy as jnp
import last
import numpy as np


class _BlankMaskedJointWeightFn(last.weight_fns.WeightFn):

  @nn.compact
  def __call__(self, cache, frame, state=None):
    joint = last.weight_fns.JointWeightFn(vocab_size=2, hidden_size=4)
    blank, lexical = joint(cache, frame, state)
    return jnp.full_like(blank, -jnp.inf), lexical


def _lattice(normalize):
  return last.RecognitionLattice(
      context=last.contexts.FullNGram(vocab_size=2, context_size=0),
      alignment=last.alignments.FrameDependent(),
      weight_fn_cacher_factory=lambda _: last.weight_fns.SharedEmbCacher(1, 4),
      weight_fn_factory=lambda _: last.weight_fns.LocallyNormalizedWeightFn(
          _BlankMaskedJointWeightFn(), normalize=normalize
      )
  )


class HatMaskedBlankTest(parameterized.TestCase):

  @parameterized.product(
      dtype=(jnp.float16, jnp.bfloat16, jnp.float32), jit=(False, True)
  )
  def test_normalization_and_gradients(self, dtype, jit):
    blank = jnp.array([-jnp.inf, 0., jnp.inf], dtype=dtype)
    lexical = jnp.broadcast_to(jnp.array([-.5, 0., 1.25], dtype=dtype), (3, 3))
    fn = last.weight_fns.hat_normalize
    if jit:
      fn = jax.jit(fn)
    # Check the intermediate subtraction too, not just the returned arrays.
    with jax.debug_nans(True):
      (actual_blank, actual_lexical), vjp = jax.vjp(fn, blank, lexical)
      grad_blank, grad_lexical = vjp(
          (jnp.ones_like(blank), jnp.ones_like(lexical))
      )
    b = np.asarray(blank, dtype=np.float64)
    l = np.asarray(lexical, dtype=np.float64)
    expected_blank = -np.logaddexp(0., -b)
    log_softmax = l - np.logaddexp.reduce(l, axis=-1, keepdims=True)
    expected_lexical = log_softmax - np.logaddexp(0., b)[:, None]
    rtol = 2e-2 if dtype == jnp.bfloat16 else 2e-3
    np.testing.assert_allclose(
        actual_blank, expected_blank, rtol=rtol, atol=1e-6
    )
    np.testing.assert_allclose(
        actual_lexical, expected_lexical, rtol=rtol, atol=1e-6
    )
    np.testing.assert_allclose(
        np.exp(np.asarray(actual_blank, dtype=float)) +
        np.exp(np.asarray(actual_lexical, dtype=float)).sum(axis=-1),
        1.,
        rtol=rtol
    )
    expected_grad_blank = 1. - 4. * np.exp(expected_blank)
    expected_grad_lexical = 1. - 3. * np.exp(log_softmax)
    np.testing.assert_allclose(
        grad_blank, expected_grad_blank, rtol=rtol, atol=1e-3
    )
    np.testing.assert_allclose(
        grad_lexical, expected_grad_lexical, rtol=rtol, atol=1e-3
    )
    self.assertEqual(actual_blank.dtype, dtype)
    self.assertEqual(actual_lexical.dtype, dtype)

  @parameterized.parameters(jnp.float16, jnp.bfloat16, jnp.float32)
  def test_finite_results_and_gradients_are_unchanged(self, dtype):
    blank = jnp.linspace(-100., 100., 17).astype(dtype)
    lexical = jax.random.normal(jax.random.PRNGKey(1), (17, 3)).astype(dtype)

    def previous(b, l):
      normalized_blank = nn.log_sigmoid(b)
      z = b - normalized_blank
      return normalized_blank, nn.log_softmax(l) - z[..., None]

    for transform in (lambda fn: fn, jax.jit):
      expected, expected_vjp = jax.vjp(transform(previous), blank, lexical)
      actual, actual_vjp = jax.vjp(
          transform(last.weight_fns.hat_normalize), blank, lexical
      )
      cotangent = (jnp.ones_like(blank), jnp.ones_like(lexical))
      for got, want in zip(
          actual + actual_vjp(cotangent), expected + expected_vjp(cotangent)
      ):
        np.testing.assert_array_equal(got, want)

  @parameterized.parameters(False, True)
  def test_masked_blank_lattice_loss_and_gradients(self, jit):
    # With blanks disabled the remaining logits need only a lexical softmax.
    reference = _lattice(lambda b, l: (b, jax.nn.log_softmax(l)))
    actual = _lattice(last.weight_fns.hat_normalize)
    data = dict(
        frames=jnp.arange(12, dtype=jnp.float32).reshape(2, 2, 3) / 10,
        num_frames=jnp.array([2, 1]),
        labels=jnp.array([[1, 2], [2, 0]]),
        num_labels=jnp.array([2, 1])
    )
    variables = reference.init(jax.random.PRNGKey(0), **data)

    def value_and_grad(lattice):
      fn = jax.value_and_grad(
          lambda params: jnp.sum(lattice.apply(params, **data))
      )
      return (jax.jit(fn) if jit else fn)(variables)

    expected_loss, expected_grads = value_and_grad(reference)
    actual_loss, actual_grads = value_and_grad(actual)
    np.testing.assert_allclose(actual_loss, expected_loss, rtol=1e-6)
    for got, want in zip(
        jax.tree_util.tree_leaves(actual_grads),
        jax.tree_util.tree_leaves(expected_grads)
    ):
      np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-7)


if __name__ == '__main__':
  absltest.main()

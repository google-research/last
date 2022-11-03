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

"""Tests for alignments."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
from last import alignments
from last import contexts
from last import semirings
import numpy.testing as npt


class AlignmentsTest(absltest.TestCase):

  def test_shift_down(self):
    npt.assert_array_equal(
        alignments.shift_down(jnp.array([1, 2, 3]), semirings.Real), [0, 1, 2])
    npt.assert_array_equal(
        alignments.shift_down(
            jnp.array([[1, 2, 3], [4, 5, 6]]), semirings.Real),
        [[0, 1, 2], [0, 4, 5]])
    npt.assert_array_equal(
        alignments.shift_down(
            jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.float32),
            semirings.Log), [[-jnp.inf, 1, 2], [-jnp.inf, 4, 5]])


class FrameDependentTest(absltest.TestCase):

  def test_topology(self):
    alignment = alignments.FrameDependent()
    self.assertEqual(alignment.num_states(), 1)
    self.assertEqual(alignment.start(), 0)
    self.assertEqual(alignment.blank_next(0), 0)
    self.assertEqual(alignment.lexical_next(0), 0)
    self.assertListEqual(alignment.topological_visit(), [0])

  def test_forward(self):
    context = contexts.FullNGram(vocab_size=2, context_size=1)
    alignment = alignments.FrameDependent()
    rngs = jax.random.split(jax.random.PRNGKey(0), 3)
    alpha = jax.random.uniform(rngs[0], [3])
    blank = jax.random.uniform(rngs[1], [3])
    lexical = jax.random.uniform(rngs[2], [3, 2])

    # Single.
    next_alpha = alignment.forward(
        alpha=alpha,
        blank=[blank],
        lexical=[lexical],
        context=context,
        semiring=semirings.Real)
    npt.assert_allclose(next_alpha, [
        alpha[0] * blank[0],
        alpha[1] * blank[1] + jnp.sum(alpha * lexical[:, 0]),
        alpha[2] * blank[2] + jnp.sum(alpha * lexical[:, 1]),
    ])

    # Batched.
    batched_next_alpha = alignment.forward(
        alpha=alpha[jnp.newaxis],
        blank=[blank[jnp.newaxis]],
        lexical=[lexical[jnp.newaxis]],
        context=context,
        semiring=semirings.Real)
    npt.assert_allclose(batched_next_alpha, next_alpha[jnp.newaxis])

    # Wrong number of weights.
    with self.assertRaisesRegex(ValueError, 'blank should be'):
      alignment.forward(
          alpha=alpha,
          blank=[blank, blank],
          lexical=[lexical],
          context=context,
          semiring=semirings.Real)
    with self.assertRaisesRegex(ValueError, 'lexical should be'):
      alignment.forward(
          alpha=alpha,
          blank=[blank],
          lexical=[lexical, lexical],
          context=context,
          semiring=semirings.Real)

  def test_backward(self):
    context = contexts.FullNGram(vocab_size=2, context_size=1)
    alignment = alignments.FrameDependent()
    rngs = jax.random.split(jax.random.PRNGKey(0), 5)
    alpha = jax.random.uniform(rngs[0], [3])
    blank = jax.random.uniform(rngs[1], [3])
    lexical = jax.random.uniform(rngs[2], [3, 2])
    beta = jax.random.uniform(rngs[3], [3])
    z = jax.random.uniform(rngs[4], [])

    # backward() always uses the log semiring.

    # Single.
    log_next_beta, [blank_marginal], [lexical_marginal] = (
        alignment.backward(
            alpha=jnp.log(alpha),
            blank=[jnp.log(blank)],
            lexical=[jnp.log(lexical)],
            beta=jnp.log(beta),
            log_z=jnp.log(z),
            context=context))
    next_beta = jnp.exp(log_next_beta)
    npt.assert_allclose(
        next_beta, [
            blank[0] * beta[0] + lexical[0, 0] * beta[1] +
            lexical[0, 1] * beta[2],
            blank[1] * beta[1] + lexical[1, 0] * beta[1] +
            lexical[1, 1] * beta[2],
            blank[2] * beta[2] + lexical[2, 0] * beta[1] +
            lexical[2, 1] * beta[2],
        ],
        rtol=1e-4)
    npt.assert_allclose(blank_marginal, alpha * blank * beta / z, rtol=1e-4)
    npt.assert_allclose(
        lexical_marginal, [
            [
                alpha[0] * lexical[0, 0] * beta[1] / z,
                alpha[0] * lexical[0, 1] * beta[2] / z
            ],
            [
                alpha[1] * lexical[1, 0] * beta[1] / z,
                alpha[1] * lexical[1, 1] * beta[2] / z
            ],
            [
                alpha[2] * lexical[2, 0] * beta[1] / z,
                alpha[2] * lexical[2, 1] * beta[2] / z
            ],
        ],
        rtol=1e-4)

    # Batched.
    batched_log_next_beta, _, _ = (
        alignment.backward(
            alpha=jnp.log(alpha)[jnp.newaxis],
            blank=[jnp.log(blank)[jnp.newaxis]],
            lexical=[jnp.log(lexical)[jnp.newaxis]],
            beta=jnp.log(beta)[jnp.newaxis],
            log_z=jnp.log(z)[jnp.newaxis],
            context=context))
    npt.assert_allclose(batched_log_next_beta, log_next_beta[jnp.newaxis])

    # Wrong number of weights.
    with self.assertRaisesRegex(ValueError, 'blank should be'):
      alignment.backward(
          alpha=alpha,
          blank=[blank, blank],
          lexical=[lexical],
          beta=beta,
          log_z=z,
          context=context)
    with self.assertRaisesRegex(ValueError, 'lexical should be'):
      alignment.backward(
          alpha=alpha,
          blank=[blank],
          lexical=[lexical, lexical],
          beta=beta,
          log_z=z,
          context=context)

  def test_string_forward(self):
    alignment = alignments.FrameDependent()
    rngs = jax.random.split(jax.random.PRNGKey(0), 3)
    alpha = jax.random.uniform(rngs[0], [4])
    blank = jax.random.uniform(rngs[1], [4])
    lexical = jax.random.uniform(rngs[2], [4])

    # Single.
    next_alpha = alignment.string_forward(
        alpha=alpha, blank=[blank], lexical=[lexical], semiring=semirings.Real)
    npt.assert_allclose(next_alpha, [
        alpha[0] * blank[0],
        alpha[1] * blank[1] + alpha[0] * lexical[0],
        alpha[2] * blank[2] + alpha[1] * lexical[1],
        alpha[3] * blank[3] + alpha[2] * lexical[2],
    ])

    # Batched.
    batched_next_alpha = alignment.string_forward(
        alpha=alpha[jnp.newaxis],
        blank=[blank[jnp.newaxis]],
        lexical=[lexical[jnp.newaxis]],
        semiring=semirings.Real)
    npt.assert_allclose(batched_next_alpha, next_alpha[jnp.newaxis])

    # Wrong number of weights.
    with self.assertRaisesRegex(ValueError, 'blank should be'):
      alignment.string_forward(
          alpha=alpha,
          blank=[blank, blank],
          lexical=[lexical],
          semiring=semirings.Real)
    with self.assertRaisesRegex(ValueError, 'lexical should be'):
      alignment.string_forward(
          alpha=alpha,
          blank=[blank],
          lexical=[lexical, lexical],
          semiring=semirings.Real)


class FrameLabelDependentTest(absltest.TestCase):

  def test_topology(self):
    alignment = alignments.FrameLabelDependent(max_expansions=2)
    self.assertEqual(alignment.num_states(), 3)
    self.assertEqual(alignment.start(), 0)
    self.assertEqual(alignment.blank_next(0), 0)
    self.assertEqual(alignment.blank_next(1), 0)
    self.assertEqual(alignment.blank_next(2), 0)
    self.assertEqual(alignment.lexical_next(0), 1)
    self.assertEqual(alignment.lexical_next(1), 2)
    self.assertIsNone(alignment.lexical_next(2))
    self.assertListEqual(alignment.topological_visit(), [0, 1, 2])

  # All possible paths. Useful for creating unit tests.
  #
  # alpha[0] * blank[0][0] * beta[0]
  # alpha[0] * lexical[0][0, 0] * blank[1][1] * beta[1]
  # alpha[0] * lexical[0][0, 0] * lexical[1][1, 0] * blank[2][1] * beta[1]
  # alpha[0] * lexical[0][0, 0] * lexical[1][1, 1] * blank[2][2] * beta[2]
  # alpha[0] * lexical[0][0, 1] * blank[1][2] * beta[2]
  # alpha[0] * lexical[0][0, 1] * lexical[1][2, 0] * blank[2][1] * beta[1]
  # alpha[0] * lexical[0][0, 1] * lexical[1][2, 1] * blank[2][2] * beta[2]

  # alpha[1] * blank[0][1] * beta[1]
  # alpha[1] * lexical[0][1, 0] * blank[1][1] * beta[1]
  # alpha[1] * lexical[0][1, 0] * lexical[1][1, 0] * blank[2][1] * beta[1]
  # alpha[1] * lexical[0][1, 0] * lexical[1][1, 1] * blank[2][2] * beta[2]
  # alpha[1] * lexical[0][1, 1] * blank[1][2] * beta[2]
  # alpha[1] * lexical[0][1, 1] * lexical[1][2, 0] * blank[2][1] * beta[1]
  # alpha[1] * lexical[0][1, 1] * lexical[1][2, 1] * blank[2][2] * beta[2]

  # alpha[2] * blank[0][2] * beta[2]
  # alpha[2] * lexical[0][2, 0] * blank[1][1] * beta[1]
  # alpha[2] * lexical[0][2, 0] * lexical[1][1, 0] * blank[2][1] * beta[1]
  # alpha[2] * lexical[0][2, 0] * lexical[1][1, 1] * blank[2][2] * beta[2]
  # alpha[2] * lexical[0][2, 1] * blank[1][2] * beta[2]
  # alpha[2] * lexical[0][2, 1] * lexical[1][2, 0] * blank[2][1] * beta[1]
  # alpha[2] * lexical[0][2, 1] * lexical[1][2, 1] * blank[2][2] * beta[2]

  def test_forward(self):
    context = contexts.FullNGram(vocab_size=2, context_size=1)
    alignment = alignments.FrameLabelDependent(max_expansions=2)
    rngs = jax.random.split(jax.random.PRNGKey(0), 4)
    alpha = jax.random.uniform(rngs[0], [3])
    blank = list(jax.random.uniform(rngs[1], [3, 3]))
    lexical = list(jax.random.uniform(rngs[2], [3, 3, 2]))

    # Single.
    next_alpha = alignment.forward(
        alpha=alpha,
        blank=blank,
        lexical=lexical,
        context=context,
        semiring=semirings.Real)
    npt.assert_allclose(next_alpha, [
        alpha[0] * blank[0][0],
        alpha[0] * lexical[0][0, 0] * blank[1][1] +
        alpha[0] * lexical[0][0, 0] * lexical[1][1, 0] * blank[2][1] +
        alpha[0] * lexical[0][0, 1] * lexical[1][2, 0] * blank[2][1] +
        alpha[1] * blank[0][1] + alpha[1] * lexical[0][1, 0] * blank[1][1] +
        alpha[1] * lexical[0][1, 0] * lexical[1][1, 0] * blank[2][1] +
        alpha[1] * lexical[0][1, 1] * lexical[1][2, 0] * blank[2][1] +
        alpha[2] * lexical[0][2, 0] * blank[1][1] +
        alpha[2] * lexical[0][2, 0] * lexical[1][1, 0] * blank[2][1] +
        alpha[2] * lexical[0][2, 1] * lexical[1][2, 0] * blank[2][1],
        alpha[0] * lexical[0][0, 0] * lexical[1][1, 1] * blank[2][2] +
        alpha[0] * lexical[0][0, 1] * blank[1][2] +
        alpha[0] * lexical[0][0, 1] * lexical[1][2, 1] * blank[2][2] +
        alpha[1] * lexical[0][1, 0] * lexical[1][1, 1] * blank[2][2] +
        alpha[1] * lexical[0][1, 1] * blank[1][2] +
        alpha[1] * lexical[0][1, 1] * lexical[1][2, 1] * blank[2][2] +
        alpha[2] * blank[0][2] +
        alpha[2] * lexical[0][2, 0] * lexical[1][1, 1] * blank[2][2] +
        alpha[2] * lexical[0][2, 1] * blank[1][2] +
        alpha[2] * lexical[0][2, 1] * lexical[1][2, 1] * blank[2][2],
    ])

    # Batched.
    batched_next_alpha = alignment.forward(
        alpha=alpha[jnp.newaxis],
        blank=[i[jnp.newaxis] for i in blank],
        lexical=[i[jnp.newaxis] for i in lexical],
        context=context,
        semiring=semirings.Real)
    npt.assert_allclose(batched_next_alpha, next_alpha[jnp.newaxis])

    # Wrong number of weights.
    with self.assertRaisesRegex(ValueError, 'blank should be'):
      alignment.forward(
          alpha=alpha,
          blank=blank + blank,
          lexical=lexical,
          context=context,
          semiring=semirings.Real)
    with self.assertRaisesRegex(ValueError, 'lexical should be'):
      alignment.forward(
          alpha=alpha,
          blank=blank,
          lexical=lexical + lexical,
          context=context,
          semiring=semirings.Real)

  def test_backward(self):
    context = contexts.FullNGram(vocab_size=2, context_size=1)
    alignment = alignments.FrameLabelDependent(max_expansions=2)
    rngs = jax.random.split(jax.random.PRNGKey(0), 5)
    alpha = jax.random.uniform(rngs[0], [3])
    blank = list(jax.random.uniform(rngs[1], [3, 3]))
    lexical = list(jax.random.uniform(rngs[2], [3, 3, 2]))
    beta = jax.random.uniform(rngs[3], [3])
    z = jax.random.uniform(rngs[4], [])

    # backward() always uses the log semiring.

    # Single.
    log_next_beta, blank_marginals, lexical_marginals = (
        alignment.backward(
            alpha=jnp.log(alpha),
            blank=[jnp.log(i) for i in blank],
            lexical=[jnp.log(i) for i in lexical],
            beta=jnp.log(beta),
            log_z=jnp.log(z),
            context=context))
    next_beta = jnp.exp(log_next_beta)
    npt.assert_allclose(
        next_beta, [
            blank[0][0] * beta[0] + lexical[0][0, 0] * blank[1][1] * beta[1] +
            lexical[0][0, 0] * lexical[1][1, 0] * blank[2][1] * beta[1] +
            lexical[0][0, 0] * lexical[1][1, 1] * blank[2][2] * beta[2] +
            lexical[0][0, 1] * blank[1][2] * beta[2] +
            lexical[0][0, 1] * lexical[1][2, 0] * blank[2][1] * beta[1] +
            lexical[0][0, 1] * lexical[1][2, 1] * blank[2][2] * beta[2],
            blank[0][1] * beta[1] + lexical[0][1, 0] * blank[1][1] * beta[1] +
            lexical[0][1, 0] * lexical[1][1, 0] * blank[2][1] * beta[1] +
            lexical[0][1, 0] * lexical[1][1, 1] * blank[2][2] * beta[2] +
            lexical[0][1, 1] * blank[1][2] * beta[2] +
            lexical[0][1, 1] * lexical[1][2, 0] * blank[2][1] * beta[1] +
            lexical[0][1, 1] * lexical[1][2, 1] * blank[2][2] * beta[2],
            blank[0][2] * beta[2] + lexical[0][2, 0] * blank[1][1] * beta[1] +
            lexical[0][2, 0] * lexical[1][1, 0] * blank[2][1] * beta[1] +
            lexical[0][2, 0] * lexical[1][1, 1] * blank[2][2] * beta[2] +
            lexical[0][2, 1] * blank[1][2] * beta[2] +
            lexical[0][2, 1] * lexical[1][2, 0] * blank[2][1] * beta[1] +
            lexical[0][2, 1] * lexical[1][2, 1] * blank[2][2] * beta[2],
        ],
        rtol=1e-4)
    npt.assert_allclose(
        blank_marginals,
        jnp.array([
            [
                alpha[0] * blank[0][0] * beta[0],
                alpha[1] * blank[0][1] * beta[1],
                alpha[2] * blank[0][2] * beta[2],
            ],
            [
                0,
                alpha[0] * lexical[0][0, 0] * blank[1][1] * beta[1] +
                alpha[1] * lexical[0][1, 0] * blank[1][1] * beta[1] +
                alpha[2] * lexical[0][2, 0] * blank[1][1] * beta[1],
                alpha[0] * lexical[0][0, 1] * blank[1][2] * beta[2] +
                alpha[1] * lexical[0][1, 1] * blank[1][2] * beta[2] +
                alpha[2] * lexical[0][2, 1] * blank[1][2] * beta[2],
            ],
            [
                0,
                alpha[0] * lexical[0][0, 0] * lexical[1][1, 0] * blank[2][1] *
                beta[1] + alpha[0] * lexical[0][0, 1] * lexical[1][2, 0] *
                blank[2][1] * beta[1] + alpha[1] * lexical[0][1, 0] *
                lexical[1][1, 0] * blank[2][1] * beta[1] + alpha[1] *
                lexical[0][1, 1] * lexical[1][2, 0] * blank[2][1] * beta[1] +
                alpha[2] * lexical[0][2, 0] * lexical[1][1, 0] * blank[2][1] *
                beta[1] + alpha[2] * lexical[0][2, 1] * lexical[1][2, 0] *
                blank[2][1] * beta[1],
                alpha[0] * lexical[0][0, 0] * lexical[1][1, 1] * blank[2][2] *
                beta[2] + alpha[0] * lexical[0][0, 1] * lexical[1][2, 1] *
                blank[2][2] * beta[2] + alpha[1] * lexical[0][1, 0] *
                lexical[1][1, 1] * blank[2][2] * beta[2] + alpha[1] *
                lexical[0][1, 1] * lexical[1][2, 1] * blank[2][2] * beta[2] +
                alpha[2] * lexical[0][2, 0] * lexical[1][1, 1] * blank[2][2] *
                beta[2] + alpha[2] * lexical[0][2, 1] * lexical[1][2, 1] *
                blank[2][2] * beta[2],
            ],
        ]) / z,
        rtol=1e-4)
    npt.assert_allclose(
        lexical_marginals,
        jnp.array([
            [
                [
                    alpha[0] * lexical[0][0, 0] * blank[1][1] * beta[1] +
                    alpha[0] * lexical[0][0, 0] * lexical[1][1, 0] *
                    blank[2][1] * beta[1] + alpha[0] * lexical[0][0, 0] *
                    lexical[1][1, 1] * blank[2][2] * beta[2],
                    alpha[0] * lexical[0][0, 1] * blank[1][2] * beta[2] +
                    alpha[0] * lexical[0][0, 1] * lexical[1][2, 0] *
                    blank[2][1] * beta[1] + alpha[0] * lexical[0][0, 1] *
                    lexical[1][2, 1] * blank[2][2] * beta[2],
                ],
                [
                    alpha[1] * lexical[0][1, 0] * blank[1][1] * beta[1] +
                    alpha[1] * lexical[0][1, 0] * lexical[1][1, 0] *
                    blank[2][1] * beta[1] + alpha[1] * lexical[0][1, 0] *
                    lexical[1][1, 1] * blank[2][2] * beta[2],
                    alpha[1] * lexical[0][1, 1] * blank[1][2] * beta[2] +
                    alpha[1] * lexical[0][1, 1] * lexical[1][2, 0] *
                    blank[2][1] * beta[1] + alpha[1] * lexical[0][1, 1] *
                    lexical[1][2, 1] * blank[2][2] * beta[2],
                ],
                [
                    alpha[2] * lexical[0][2, 0] * blank[1][1] * beta[1] +
                    alpha[2] * lexical[0][2, 0] * lexical[1][1, 0] *
                    blank[2][1] * beta[1] + alpha[2] * lexical[0][2, 0] *
                    lexical[1][1, 1] * blank[2][2] * beta[2],
                    alpha[2] * lexical[0][2, 1] * blank[1][2] * beta[2] +
                    alpha[2] * lexical[0][2, 1] * lexical[1][2, 0] *
                    blank[2][1] * beta[1] + alpha[2] * lexical[0][2, 1] *
                    lexical[1][2, 1] * blank[2][2] * beta[2],
                ],
            ],
            [
                [0, 0],
                [
                    alpha[0] * lexical[0][0, 0] * lexical[1][1, 0] *
                    blank[2][1] * beta[1] + alpha[1] * lexical[0][1, 0] *
                    lexical[1][1, 0] * blank[2][1] * beta[1] + alpha[2] *
                    lexical[0][2, 0] * lexical[1][1, 0] * blank[2][1] * beta[1],
                    alpha[0] * lexical[0][0, 0] * lexical[1][1, 1] *
                    blank[2][2] * beta[2] + alpha[1] * lexical[0][1, 0] *
                    lexical[1][1, 1] * blank[2][2] * beta[2] + alpha[2] *
                    lexical[0][2, 0] * lexical[1][1, 1] * blank[2][2] * beta[2],
                ],
                [
                    alpha[0] * lexical[0][0, 1] * lexical[1][2, 0] *
                    blank[2][1] * beta[1] + alpha[1] * lexical[0][1, 1] *
                    lexical[1][2, 0] * blank[2][1] * beta[1] + alpha[2] *
                    lexical[0][2, 1] * lexical[1][2, 0] * blank[2][1] * beta[1],
                    alpha[0] * lexical[0][0, 1] * lexical[1][2, 1] *
                    blank[2][2] * beta[2] + alpha[1] * lexical[0][1, 1] *
                    lexical[1][2, 1] * blank[2][2] * beta[2] + alpha[2] *
                    lexical[0][2, 1] * lexical[1][2, 1] * blank[2][2] * beta[2],
                ],
            ],
            [
                [0, 0],
                [0, 0],
                [0, 0],
            ],
        ]) / z,
        rtol=1e-4)

    # Batched.
    batched_log_next_beta, _, _ = (
        alignment.backward(
            alpha=jnp.log(alpha)[jnp.newaxis],
            blank=[jnp.log(i)[jnp.newaxis] for i in blank],
            lexical=[jnp.log(i)[jnp.newaxis] for i in lexical],
            beta=jnp.log(beta)[jnp.newaxis],
            log_z=jnp.log(z)[jnp.newaxis],
            context=context))
    npt.assert_allclose(batched_log_next_beta, log_next_beta[jnp.newaxis])

    # Wrong number of weights.
    with self.assertRaisesRegex(ValueError, 'blank should be'):
      alignment.backward(
          alpha=alpha,
          blank=blank + blank,
          lexical=lexical,
          beta=beta,
          log_z=z,
          context=context)
    with self.assertRaisesRegex(ValueError, 'lexical should be'):
      alignment.backward(
          alpha=alpha,
          blank=blank,
          lexical=lexical + lexical,
          beta=beta,
          log_z=z,
          context=context)

  def test_string_forward(self):
    alignment = alignments.FrameLabelDependent(max_expansions=2)
    rngs = jax.random.split(jax.random.PRNGKey(0), 3)
    alpha = jax.random.uniform(rngs[0], [4])
    blank = list(jax.random.uniform(rngs[1], [3, 4]))
    lexical = list(jax.random.uniform(rngs[2], [3, 4]))

    # Single.
    next_alpha = alignment.string_forward(
        alpha=alpha, blank=blank, lexical=lexical, semiring=semirings.Real)
    npt.assert_allclose(next_alpha, [
        alpha[0] * blank[0][0],
        alpha[1] * blank[0][1] + alpha[0] * lexical[0][0] * blank[1][1],
        alpha[2] * blank[0][2] + alpha[1] * lexical[0][1] * blank[1][2] +
        alpha[0] * lexical[0][0] * lexical[1][1] * blank[2][2],
        alpha[3] * blank[0][3] + alpha[2] * lexical[0][2] * blank[1][3] +
        alpha[1] * lexical[0][1] * lexical[1][2] * blank[2][3],
    ])

    # Batched.
    batched_next_alpha = alignment.string_forward(
        alpha=alpha[jnp.newaxis],
        blank=[i[jnp.newaxis] for i in blank],
        lexical=[i[jnp.newaxis] for i in lexical],
        semiring=semirings.Real)
    npt.assert_allclose(batched_next_alpha, next_alpha[jnp.newaxis])

    # Wrong number of weights.
    with self.assertRaisesRegex(ValueError, 'blank should be'):
      alignment.string_forward(
          alpha=alpha,
          blank=blank + blank,
          lexical=lexical,
          semiring=semirings.Real)
    with self.assertRaisesRegex(ValueError, 'lexical should be'):
      alignment.string_forward(
          alpha=alpha,
          blank=blank,
          lexical=lexical + lexical,
          semiring=semirings.Real)


if __name__ == '__main__':
  absltest.main()

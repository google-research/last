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

"""Tests for lattices."""

import functools

from absl.testing import absltest
import jax
import jax.numpy as jnp
import last
import numpy.testing as npt


def weight_fn_cacher_factory(context: last.contexts.FullNGram):
  return last.weight_fns.SharedRNNCacher(
      vocab_size=context.vocab_size,
      context_size=context.context_size,
      rnn_size=24)


def weight_fn_factory(context: last.contexts.ContextDependency):
  _, vocab_size = context.shape()
  return last.weight_fns.JointWeightFn(vocab_size=vocab_size, hidden_size=16)


class RecognitionLatticeBasicsTest(absltest.TestCase):

  def test_build_cache(self):
    vocab_size = 2
    context_size = 1
    lattice = last.RecognitionLattice(
        context=last.contexts.FullNGram(
            vocab_size=vocab_size, context_size=context_size),
        alignment=last.alignments.FrameDependent(),
        weight_fn_cacher_factory=weight_fn_cacher_factory,
        weight_fn_factory=weight_fn_factory)
    frames = jax.random.normal(jax.random.PRNGKey(0), [4, 6, 8])
    num_frames = jnp.array([6, 3, 2, 0])
    labels = jnp.array([[1, 1, 1, 1], [2, 2, 2, 2], [1, 2, 1, 2], [2, 1, 2, 1]])
    num_labels = jnp.array([4, 3, 1, 0])
    loss, params = lattice.init_with_output(
        jax.random.PRNGKey(0),
        frames=frames,
        num_frames=num_frames,
        labels=labels,
        num_labels=num_labels)
    with self.subTest('same cache'):
      cache = lattice.apply(params, method=lattice.build_cache)
      loss_with_same_cache = lattice.apply(
          params,
          frames=frames,
          num_frames=num_frames,
          labels=labels,
          num_labels=num_labels,
          cache=cache)
      npt.assert_array_equal(loss_with_same_cache, loss)
    with self.subTest('different cache'):
      # This makes sure we are using the cache when supplied.
      loss_with_different_cache = lattice.apply(
          params,
          frames=frames,
          num_frames=num_frames,
          labels=labels,
          num_labels=num_labels,
          cache=jax.tree_util.tree_map(lambda x: x + 1, cache))
      self.assertTrue(
          (loss_with_different_cache != loss).any(),
          msg=f'Should be not equal: loss={loss!r}, '
          f'loss_with_different_cache={loss_with_different_cache!r}')

  def test_call(self):
    vocab_size = 2
    context_size = 1
    lattice = last.RecognitionLattice(
        context=last.contexts.FullNGram(
            vocab_size=vocab_size, context_size=context_size),
        alignment=last.alignments.FrameDependent(),
        weight_fn_cacher_factory=weight_fn_cacher_factory,
        weight_fn_factory=weight_fn_factory)
    frames = jax.random.normal(jax.random.PRNGKey(0), [4, 6, 8])
    num_frames = jnp.array([6, 3, 2, 1])
    labels = jnp.array([[1, 1, 1, 1], [2, 2, 2, 2], [1, 2, 1, 2], [2, 1, 2, 1]])
    num_labels = jnp.array([4, 3, 1, 2])
    loss, params = lattice.init_with_output(
        jax.random.PRNGKey(0),
        frames=frames,
        num_frames=num_frames,
        labels=labels,
        num_labels=num_labels)
    npt.assert_array_equal(jnp.isfinite(loss), [True, True, True, False])

    with self.subTest('padded inputs'):
      loss_with_padded_inputs = lattice.apply(
          params,
          frames=jnp.pad(frames, [(0, 0), (0, 1), (0, 0)]),
          num_frames=num_frames,
          labels=jnp.pad(labels, [(0, 0), (0, 2)]),
          num_labels=num_labels)
      npt.assert_allclose(loss_with_padded_inputs, loss)

    with self.subTest('invalid shapes'):
      with self.assertRaisesRegex(
          ValueError, 'frames and num_frames have different batch_dims'):
        lattice.apply(
            params,
            frames=frames[:1],
            num_frames=num_frames,
            labels=labels,
            num_labels=num_labels)
      with self.assertRaisesRegex(
          ValueError, 'labels and num_frames have different batch_dims'):
        lattice.apply(
            params,
            frames=frames,
            num_frames=num_frames,
            labels=labels[:1],
            num_labels=num_labels)
      with self.assertRaisesRegex(
          ValueError, 'num_labels and num_frames have different batch_dims'):
        lattice.apply(
            params,
            frames=frames,
            num_frames=num_frames,
            labels=labels,
            num_labels=num_labels[:1])

  def test_shortest_path(self):
    vocab_size = 2
    context_size = 1
    lattice = last.RecognitionLattice(
        context=last.contexts.FullNGram(
            vocab_size=vocab_size, context_size=context_size),
        alignment=last.alignments.FrameDependent(),
        weight_fn_cacher_factory=weight_fn_cacher_factory,
        weight_fn_factory=weight_fn_factory)
    frames = jax.random.normal(jax.random.PRNGKey(0), [4, 6, 8])
    num_frames = jnp.array([6, 3, 2, 0])
    (alignment_labels, num_alignment_labels, path_weights), params = (
        lattice.init_with_output(
            jax.random.PRNGKey(0),
            frames,
            num_frames,
            method=lattice.shortest_path))
    with self.subTest('reasonable outputs'):
      npt.assert_array_equal(num_alignment_labels, [6, 3, 2, 0])
      is_padding = jnp.arange(6) >= num_frames[:, jnp.newaxis]
      npt.assert_array_equal(
          jnp.where(is_padding, alignment_labels, -1), [
              [-1, -1, -1, -1, -1, -1],
              [-1, -1, -1, 0, 0, 0],
              [-1, -1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
          ])
      npt.assert_array_equal(
          alignment_labels >= 0,
          jnp.ones([4, 6], dtype=bool),
          err_msg=f'alignment_labels={alignment_labels!r}')
      npt.assert_array_equal(
          alignment_labels <= vocab_size,
          jnp.ones([4, 6], dtype=bool),
          err_msg=f'alignment_labels={alignment_labels!r}')
      npt.assert_array_equal(
          jnp.isfinite(path_weights), [True, True, True, True],
          err_msg=f'path_weights={path_weights!r}')
      npt.assert_array_equal(
          path_weights == 0, [False, False, False, True],
          err_msg=f'path_weights={path_weights!r}')

    with self.subTest('padded inputs'):
      (_, _, path_weights_with_padded_inputs) = lattice.apply(
          params,
          jnp.pad(frames, [(0, 0), (0, 1), (0, 0)]),
          num_frames,
          method=lattice.shortest_path)
      npt.assert_allclose(path_weights_with_padded_inputs, path_weights)

    with self.subTest('invalid shapes'):
      with self.assertRaisesRegex(
          ValueError, 'frames and num_frames have different batch_dims'):
        lattice.apply(
            params, frames[:1], num_frames, method=lattice.shortest_path)

  def test_frame_label_dependent(self):
    vocab_size = 2
    context_size = 1
    lattice = last.RecognitionLattice(
        context=last.contexts.FullNGram(
            vocab_size=vocab_size, context_size=context_size),
        alignment=last.alignments.FrameLabelDependent(max_expansions=2),
        weight_fn_cacher_factory=weight_fn_cacher_factory,
        weight_fn_factory=weight_fn_factory)
    frames = jax.random.normal(jax.random.PRNGKey(0), [4, 6, 8])
    num_frames = jnp.array([6, 3, 2, 1])
    labels = jnp.array([[1, 1, 1, 1], [2, 2, 2, 2], [1, 2, 1, 2], [2, 1, 2, 1]])
    num_labels = jnp.array([4, 3, 4, 3])
    with self.subTest('loss'):
      loss, params = lattice.init_with_output(
          jax.random.PRNGKey(0),
          frames=frames,
          num_frames=num_frames,
          labels=labels,
          num_labels=num_labels)
      npt.assert_array_equal(jnp.isfinite(loss), [True, True, True, False])
    with self.subTest('shortest_path'):
      alignment_labels, num_alignment_labels, path_weights = (
          lattice.apply(
              params, frames, num_frames, method=lattice.shortest_path))
      npt.assert_array_equal(num_alignment_labels, 3 * num_frames)
      is_padding = jnp.arange(18) >= num_alignment_labels[:, jnp.newaxis]
      npt.assert_array_equal(
          jnp.where(is_padding, alignment_labels, -1), [
              [-1] * 18,
              [-1] * 9 + [0] * 9,
              [-1] * 6 + [0] * 12,
              [-1] * 3 + [0] * 15,
          ])
      # Every third label is 0.
      npt.assert_array_equal(
          alignment_labels.reshape([4, 6, 3])[..., -1], jnp.zeros([4, 6]))
      npt.assert_array_equal(
          alignment_labels >= 0,
          jnp.ones([4, 18], dtype=bool),
          err_msg=f'alignment_labels={alignment_labels!r}')
      npt.assert_array_equal(
          alignment_labels <= vocab_size,
          jnp.ones([4, 18], dtype=bool),
          err_msg=f'alignment_labels={alignment_labels!r}')
      npt.assert_array_equal(
          jnp.isfinite(path_weights), [True, True, True, True],
          err_msg=f'path_weights={path_weights!r}')


class RecognitionLatticeCorrectnessTest(absltest.TestCase):
  """Tests the correctness of various RecognitionLattice operations."""

  def test_frame_dependent(self):
    batch_size = 3
    max_num_frames = 2
    vocab_size = 2
    context_size = 1
    num_context_states = 3

    frames = jnp.broadcast_to(
        jnp.expand_dims(
            jnp.arange(max_num_frames, dtype=jnp.float32), axis=[0, 2]),
        [batch_size, max_num_frames, 1])
    num_frames = jnp.array([2, 1, 0])

    weight_table = 1 + jnp.arange(
        batch_size * max_num_frames * num_context_states * (1 + vocab_size),
        dtype=jnp.float32).reshape(
            [batch_size, max_num_frames, num_context_states, 1 + vocab_size])
    # Alternate the signs over the frame time dimension so that we get some
    # interesting shortest paths.
    weight_table *= jnp.expand_dims(
        jnp.array([[-1, 1], [1, -1], [1, 1]]), axis=[2, 3])

    lattice = last.RecognitionLattice(
        context=last.contexts.FullNGram(
            vocab_size=vocab_size, context_size=context_size),
        alignment=last.alignments.FrameDependent(),
        weight_fn_factory=lambda _: last.weight_fns.TableWeightFn(weight_table),
        weight_fn_cacher_factory=lambda _: last.weight_fns.NullCacher())
    # For easier application of methods.
    lattice = lattice.bind({})

    # Forward, i.e. shortest distance.
    for semiring_name, expected in [
        ('MaxTropical', [-3 + 18, 21, 0]),
        ('Real',
         [(-1) * (10 + 11 + 12) + (-2) * (13 + 14 + 15) + (-3) * (16 + 17 + 18),
          19 + 20 + 21, 1]),
        ('Log', [
            jax.nn.logsumexp(
                jnp.array([
                    -1 + 10, -1 + 11, -1 + 12, -2 + 13, -2 + 14, -2 + 15,
                    -3 + 16, -3 + 17, -3 + 18
                ])),
            jax.nn.logsumexp(jnp.array([19, 20, 21])), 0.
        ])
    ]:
      semiring = getattr(last.semirings, semiring_name)
      with self.subTest(f'forward/{semiring_name}'):
        npt.assert_allclose(
            lattice._forward(
                cache=None,
                frames=frames,
                num_frames=num_frames,
                semiring=semiring)[0], expected)

    with self.subTest('shortest_path'):
      alignment_labels, num_alignment_labels, path_weights = (
          lattice.shortest_path(
              frames=frames, num_frames=num_frames, cache=None))
      npt.assert_array_equal(num_alignment_labels, num_frames)
      npt.assert_allclose(path_weights, [-3 + 18, 21, 0])
      npt.assert_array_equal(alignment_labels, [
          [2, 2],
          [2, 0],
          [0, 0],
      ])

    # String forward, i.e. shortest distance after intersection with a string.
    labels = jnp.array([[1, 2, 0], [2, 1, 0], [1, 2, 0]])
    num_labels = jnp.array([1, 1, 0])
    for semiring_name, expected in [
        ('MaxTropical', [-2 + 13, 21, 0]),
        ('Real', [(-1) * 11 + (-2) * 13, 21, 1]),
        ('Log', [jax.nn.logsumexp(jnp.array([-1 + 11, -2 + 13])), 21., 0.])
    ]:
      semiring = getattr(last.semirings, semiring_name)
      with self.subTest(f'string_forward/{semiring_name}'):
        npt.assert_allclose(
            lattice._string_forward(
                cache=None,
                frames=frames,
                num_frames=num_frames,
                labels=labels,
                num_labels=num_labels,
                semiring=semiring), expected)
      with self.subTest(f'string_forward non-reachable/{semiring_name}'):
        npt.assert_array_equal(
            lattice._string_forward(
                cache=None,
                frames=frames,
                num_frames=num_frames,
                labels=labels,
                num_labels=jnp.array([3, 2, 1]),
                semiring=semiring), semiring.zeros([3]))

    with self.subTest('call'):
      log_loss = lattice(
          frames=frames,
          num_frames=num_frames,
          labels=labels,
          num_labels=num_labels,
          cache=None)
      npt.assert_allclose(
          log_loss, [
              jax.nn.logsumexp(
                  jnp.array([
                      -1 + 10, -1 + 11, -1 + 12, -2 + 13, -2 + 14, -2 + 15,
                      -3 + 16, -3 + 17, -3 + 18
                  ])) - jax.nn.logsumexp(jnp.array([-1 + 11, -2 + 13])),
              jax.nn.logsumexp(jnp.array([19, 20, 21])) - 21., 0.
          ],
          rtol=1e-6)

  # Tests for _backward().

  def test_arc_marginals(self):
    # Test _backward() by computing arc marginals. This is a bit easier to debug
    # than the full-on forward-backward.
    vocab_size = 2
    context_size = 1
    lattice = last.RecognitionLattice(
        context=last.contexts.FullNGram(
            vocab_size=vocab_size, context_size=context_size),
        alignment=last.alignments.FrameDependent(),
        weight_fn_cacher_factory=weight_fn_cacher_factory,
        weight_fn_factory=weight_fn_factory)
    frames = jax.random.uniform(jax.random.PRNGKey(0), [4, 6, 8])
    num_frames = jnp.array([6, 3, 2, 0])
    params = lattice.init(
        jax.random.PRNGKey(0), frames, num_frames, method=lattice.shortest_path)
    # For easier application of methods.
    lattice = lattice.bind(params)
    del params
    cache = lattice.build_cache()

    # Compute expected marginals using autodiff.
    def forward(masks):
      blank_mask, lexical_mask = masks
      log_z, _ = lattice._forward(
          cache=cache,
          frames=frames,
          num_frames=num_frames,
          semiring=last.semirings.Log,
          blank_mask=[blank_mask],
          lexical_mask=[lexical_mask])
      return jnp.sum(log_z)

    num_context_states, _ = lattice.context.shape()
    blank_mask = jnp.zeros([*frames.shape[:-1], num_context_states])
    lexical_mask = jnp.zeros(
        [*frames.shape[:-1], num_context_states, vocab_size])
    expected_marginals = jax.grad(forward)((blank_mask, lexical_mask))

    # Compute marginals using _backward().
    def arc_marginals(frames, num_frames):

      def arc_marginals_callback(weight_vjp_fn, carry, blank_marginal,
                                 lexical_marginals):
        del weight_vjp_fn
        del carry
        next_carry = None
        outputs = (blank_marginal, lexical_marginals)
        return next_carry, outputs

      log_z, alpha_0_to_T_minus_1 = lattice._forward(  # pylint: disable=invalid-name
          cache=cache,
          frames=frames,
          num_frames=num_frames,
          semiring=last.semirings.Log)
      _, (blank_marginal, lexical_marginals) = lattice._backward(
          cache=cache,
          frames=frames,
          num_frames=num_frames,
          log_z=log_z,
          alpha_0_to_T_minus_1=alpha_0_to_T_minus_1,
          init_callback_carry=None,
          callback=arc_marginals_callback)
      return blank_marginal, lexical_marginals

    actual_marginals = arc_marginals(frames, num_frames)
    jax.tree_util.tree_map(
        functools.partial(npt.assert_allclose, rtol=1e-3), actual_marginals,
        expected_marginals)

  def test_forward_backward(self):
    vocab_size = 2
    context_size = 1
    lattice = last.RecognitionLattice(
        context=last.contexts.FullNGram(
            vocab_size=vocab_size, context_size=context_size),
        alignment=last.alignments.FrameDependent(),
        weight_fn_cacher_factory=weight_fn_cacher_factory,
        weight_fn_factory=weight_fn_factory)
    frames = jax.random.uniform(jax.random.PRNGKey(0), [4, 6, 8])
    num_frames = jnp.array([6, 3, 2, 0])
    params = lattice.init(
        jax.random.PRNGKey(0), frames, num_frames, method=lattice.shortest_path)

    def forward(params, frames):
      cache = lattice.apply(params, method=lattice.build_cache)
      log_z, _ = lattice.apply(
          params,
          cache=cache,
          frames=frames,
          num_frames=num_frames,
          semiring=last.semirings.Log,
          method=lattice._forward)
      return log_z

    expected_log_z, expected_vjp_fn = jax.vjp(forward, params, frames)

    def forward_backward(params, frames):
      cache = lattice.apply(params, method=lattice.build_cache)
      return lattice.apply(
          params,
          cache=cache,
          frames=frames,
          num_frames=num_frames,
          method=lattice._forward_backward)

    actual_log_z, actual_vjp_fn = jax.vjp(forward_backward, params, frames)
    npt.assert_allclose(actual_log_z, expected_log_z)

    for g in [
        jnp.ones_like(expected_log_z),
        jax.random.uniform(jax.random.PRNGKey(0), expected_log_z.shape)
    ]:
      expected_grads = expected_vjp_fn(g)
      actual_grads = actual_vjp_fn(g)
      jax.tree_util.tree_map(
          functools.partial(npt.assert_allclose, rtol=1e-3, atol=1e-6),
          actual_grads, expected_grads)


if __name__ == '__main__':
  absltest.main()

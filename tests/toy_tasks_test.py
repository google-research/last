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

"""Tests for last.RecognitionLattice using toy tasks.

Toy tasks are designed to test the limits of model expressiveness and validate
that training & inference both work in close to practical settings. Thus these
are light weight integration tests.
"""

from collections.abc import Iterator
import pprint
from typing import Any

from absl import logging
from absl.testing import absltest
import jax
import jax.numpy as jnp
import last
import numpy as np
import numpy.testing as npt
import optax


def run_toy_task(data_iter: Iterator[Any],
                 lattice: last.RecognitionLattice,
                 num_steps: int = 200) -> dict[str, Any]:
  # Each batch is (ilabels, frames, num_frames, labels, num_labels).
  test_batch = next(data_iter)

  params = jax.jit(lattice.init)(jax.random.PRNGKey(0), *test_batch[1:])
  optimizer = optax.adam(1e-2)
  opt_state = optimizer.init(params)

  @jax.jit
  def train_step(params, opt_state, batch):
    _, frames, num_frames, labels, num_labels = batch

    def loss_fn(params):
      return jnp.mean(
          lattice.apply(
              params,
              frames=frames,
              num_frames=num_frames,
              labels=labels,
              num_labels=num_labels))

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return loss, params, opt_state

  @jax.jit
  def eval_step(params, batch):
    ilabels, frames, num_frames, labels, num_labels = batch
    alignment_labels, _, path_weights = lattice.apply(
        params,
        frames=frames,
        num_frames=num_frames,
        method=lattice.shortest_path)
    sequence_loss = lattice.apply(params, *batch[1:])

    def masked(labels, num_labels):
      mask = jnp.arange(labels.shape[-1]) < num_labels[..., jnp.newaxis]
      return jnp.where(mask, labels, 0)

    return {
        'ilabels': masked(ilabels, num_frames),
        'alignment_labels': alignment_labels,
        'labels': masked(labels, num_labels),
        'path_weights': path_weights,
        'sequence_loss': sequence_loss
    }

  for i in range(num_steps):
    loss, params, opt_state = train_step(params, opt_state, next(data_iter))
    logging.info('step %d loss %s', i, jax.device_get(loss))
  eval_result = jax.device_get(eval_step(params, test_batch))
  logging.info('eval_result:\n%s', pprint.pformat(eval_result))
  return eval_result


# Labels are in the range [1, VOCAB_SIZE].
VOCAB_SIZE = 8


def gen_exact_copy(seed=0, batch_size=16, min_length=8, max_length=12):
  rng = np.random.RandomState(seed)
  embed = rng.normal(size=(VOCAB_SIZE, 64)).astype(np.float32)
  while True:
    labels = 1 + rng.randint(
        VOCAB_SIZE, size=(batch_size, max_length)).astype(np.int32)
    num_labels = rng.randint(
        min_length, max_length + 1, size=(batch_size,)).astype(np.int32)
    frames = embed[labels - 1]
    yield labels, frames, num_labels, labels, num_labels


def gen_odd(seed=0, batch_size=16, min_length=8, max_length=12):

  def translate(ilabels):
    olabels = []
    for i in ilabels:
      if i % 2:
        olabels.append(i)
    return olabels

  npt.assert_equal(translate([1, 2, 3, 4, 5]), [1, 3, 5])

  rng = np.random.RandomState(seed)
  embed = rng.normal(size=(VOCAB_SIZE, 64)).astype(np.float32)
  while True:
    ilabels = 1 + rng.randint(
        VOCAB_SIZE, size=(batch_size, max_length)).astype(np.int32)
    frames = embed[ilabels - 1]
    num_frames = rng.randint(
        min_length, max_length + 1, size=(batch_size,)).astype(np.int32)

    labels = np.zeros_like(ilabels)
    num_labels = np.zeros((batch_size,), np.int32)
    for i in range(batch_size):
      labels_i = translate(ilabels[i, :num_frames[i]])
      num_labels[i] = len(labels_i)
      labels[i, :num_labels[i]] = labels_i

    yield ilabels, frames, num_frames, labels, num_labels


def gen_odd_bigram(seed=0, batch_size=16, min_length=8, max_length=12):

  def translate(ilabels):
    olabels = []
    last_olabel = 0
    for i in ilabels:
      if i % 2:
        olabels.append(1 + (i + last_olabel) % VOCAB_SIZE)
        last_olabel = olabels[-1]
    return olabels

  npt.assert_equal(translate([1, 2, 3, 4, 5]), [2, 6, 4])

  rng = np.random.RandomState(seed)
  embed = rng.normal(size=(VOCAB_SIZE, 64)).astype(np.float32)
  while True:
    ilabels = 1 + rng.randint(
        VOCAB_SIZE, size=(batch_size, max_length)).astype(np.int32)
    frames = embed[ilabels - 1]
    num_frames = rng.randint(
        min_length, max_length + 1, size=(batch_size,)).astype(np.int32)

    labels = np.zeros_like(ilabels)
    num_labels = np.zeros((batch_size,), np.int32)
    for i in range(batch_size):
      labels_i = translate(ilabels[i, :num_frames[i]])
      num_labels[i] = len(labels_i)
      labels[i, :num_labels[i]] = labels_i
    yield ilabels, frames, num_frames, labels, num_labels


def gen_next_label(seed=0, batch_size=16, min_length=8, max_length=12):

  def translate(ilabels):
    olabels = list(ilabels[1:])
    olabels.append(ilabels[-1])
    return olabels

  npt.assert_equal(translate([1, 2, 3, 4, 5]), [2, 3, 4, 5, 5])

  rng = np.random.RandomState(seed)
  embed = rng.normal(size=(VOCAB_SIZE, 64)).astype(np.float32)
  while True:
    ilabels = 1 + rng.randint(
        1, VOCAB_SIZE, size=(batch_size, max_length)).astype(np.int32)
    num_frames = rng.randint(
        min_length, max_length + 1, size=(batch_size,)).astype(np.int32)
    # Explicit eos marker using 1.
    for i in range(batch_size):
      ilabels[i, num_frames[i] - 1:] = 1
    frames = embed[ilabels - 1]

    labels = np.zeros_like(ilabels)
    num_labels = np.zeros((batch_size,), np.int32)
    for i in range(batch_size):
      labels_i = translate(ilabels[i, :num_frames[i]])
      num_labels[i] = len(labels_i)
      labels[i, :num_labels[i]] = labels_i

    yield ilabels, frames, num_frames, labels, num_labels


def gen_next_label_bigram(seed=0,
                          batch_size=16,
                          min_length=8,
                          max_length=12,
                          stack_input=True):

  def translate(ilabels):
    olabels = []
    for i in range(1, len(ilabels)):
      olabels.append(1 + (ilabels[i - 1] + ilabels[i]) % VOCAB_SIZE)
    olabels.append(ilabels[-1])
    return olabels

  npt.assert_equal(translate([1, 2, 3, 4, 5]), [4, 6, 8, 2, 5])

  rng = np.random.RandomState(seed)
  embed = rng.normal(size=(VOCAB_SIZE, 64)).astype(np.float32)
  while True:
    ilabels = 1 + rng.randint(
        1, VOCAB_SIZE, size=(batch_size, max_length)).astype(np.int32)
    num_frames = rng.randint(
        min_length, max_length + 1, size=(batch_size,)).astype(np.int32)
    # Explicit eos marker using 1.
    for i in range(batch_size):
      ilabels[i, num_frames[i] - 1:] = 1
    frames = embed[ilabels - 1]
    if stack_input:
      # Make frame t-1 available at time t.
      frames = np.concatenate([np.roll(frames, 1, axis=1), frames], axis=2)

    labels = np.zeros_like(ilabels)
    num_labels = np.zeros((batch_size,), np.int32)
    for i in range(batch_size):
      labels_i = translate(ilabels[i, :num_frames[i]])
      num_labels[i] = len(labels_i)
      labels[i, :num_labels[i]] = labels_i

    yield ilabels, frames, num_frames, labels, num_labels


# This task requires remembering blank outputs, and thus can't be handled
# by frame-dependent models.
def gen_next_even(seed=0, batch_size=16, min_length=8, max_length=12):

  def translate(ilabels):
    """Copies ilabels[i] if ilabels[i + 1] is even."""
    olabels = []
    for i in range(1, len(ilabels)):
      if ilabels[i] % 2 == 0:
        olabels.append(ilabels[i - 1])
    return olabels

  npt.assert_equal(translate([1, 1, 2, 3, 5, 8, 0]), [1, 5, 8])
  npt.assert_equal(translate([0, 1, 2, 3, 4, 5, 6]), [1, 3, 5])
  npt.assert_equal(translate([0, 2, 4, 6]), [0, 2, 4])

  rng = np.random.RandomState(seed)
  embed = rng.normal(size=(VOCAB_SIZE, 64)).astype(np.float32)
  while True:
    ilabels = 1 + rng.randint(
        1, VOCAB_SIZE, size=(batch_size, max_length)).astype(np.int32)
    num_frames = rng.randint(
        min_length, max_length + 1, size=(batch_size,)).astype(np.int32)
    frames = embed[ilabels - 1]

    labels = np.zeros_like(ilabels)
    num_labels = np.zeros((batch_size,), np.int32)
    for i in range(batch_size):
      labels_i = translate(ilabels[i, :num_frames[i]])
      num_labels[i] = len(labels_i)
      labels[i, :num_labels[i]] = labels_i

    yield ilabels, frames, num_frames, labels, num_labels


def gen_copy_no_repeat(seed=0, batch_size=16, min_length=8, max_length=12):

  def translate(ilabels):
    """Copies unless same as the previous label."""
    olabels = []
    for i in ilabels:
      if not olabels or i != olabels[-1]:
        olabels.append(i)
    return olabels

  npt.assert_equal(translate([1, 2, 3, 3, 4]), [1, 2, 3, 4])

  rng = np.random.RandomState(seed)
  embed = rng.normal(size=(VOCAB_SIZE, 64)).astype(np.float32)
  while True:
    ilabels = 1 + rng.randint(
        VOCAB_SIZE, size=(batch_size, max_length)).astype(np.int32)
    num_frames = rng.randint(
        min_length, max_length + 1, size=(batch_size,)).astype(np.int32)
    frames = embed[ilabels - 1]

    labels = np.zeros_like(ilabels)
    num_labels = np.zeros((batch_size,), np.int32)
    for i in range(batch_size):
      labels_i = translate(ilabels[i, :num_frames[i]])
      num_labels[i] = len(labels_i)
      labels[i, :num_labels[i]] = labels_i

    yield ilabels, frames, num_frames, labels, num_labels


def gen_odd_twice_no_repeat(seed=0, batch_size=16, min_length=8, max_length=12):

  def translate(ilabels):
    """Copies odd inputs twice unless same as the previous output."""
    olabels = []
    for i in ilabels:
      if i % 2 != 0 and (not olabels or i != olabels[-1]):
        olabels.append(i)
        olabels.append(i)
    return olabels

  npt.assert_equal(translate([1, 2, 3, 3, 5]), [1, 1, 3, 3, 5, 5])

  rng = np.random.RandomState(seed)
  embed = rng.normal(size=(VOCAB_SIZE, 64)).astype(np.float32)
  while True:
    ilabels = 1 + rng.randint(
        VOCAB_SIZE, size=(batch_size, max_length)).astype(np.int32)
    num_frames = rng.randint(
        min_length, max_length + 1, size=(batch_size,)).astype(np.int32)
    frames = embed[ilabels - 1]

    labels = np.zeros((batch_size, max_length * 2), np.int32)
    num_labels = np.zeros((batch_size,), np.int32)
    for i in range(batch_size):
      labels_i = translate(ilabels[i, :num_frames[i]])
      num_labels[i] = len(labels_i)
      labels[i, :num_labels[i]] = labels_i

    yield ilabels, frames, num_frames, labels, num_labels


def model_factory(
    context_size: int,
    locally_normalize: bool,
    alignment: last.alignments.TimeSyncAlignmentLattice = last.alignments
    .FrameDependent()
) -> last.RecognitionLattice:

  def weight_fn_factory(context: last.contexts.FullNGram):
    weight_fn = last.weight_fns.JointWeightFn(
        vocab_size=context.vocab_size, hidden_size=128)
    if locally_normalize:
      weight_fn = last.weight_fns.LocallyNormalizedWeightFn(weight_fn)
    return weight_fn

  def weight_fn_cacher_factory(context: last.contexts.FullNGram):
    return last.weight_fns.SharedRNNCacher(
        vocab_size=context.vocab_size,
        context_size=context.context_size,
        rnn_size=128)

  return last.RecognitionLattice(
      context=last.contexts.FullNGram(
          vocab_size=VOCAB_SIZE, context_size=context_size),
      alignment=alignment,
      weight_fn_cacher_factory=weight_fn_cacher_factory,
      weight_fn_factory=weight_fn_factory)


def assert_same_labels(a, b):
  for i in range(max(len(a), len(b))):
    npt.assert_array_equal(
        a[i][np.nonzero(a[i])],
        b[i][np.nonzero(b[i])],
        err_msg=f'row index = {i}')


class FrameDependentToyTasksTest(absltest.TestCase):

  def test_local_context_0_exact_copy(self):
    model = model_factory(context_size=0, locally_normalize=True)
    eval_result = run_toy_task(iter(gen_exact_copy()), model)
    npt.assert_array_less(eval_result['sequence_loss'].mean(), 1e-2)
    assert_same_labels(eval_result['alignment_labels'], eval_result['labels'])

  def test_local_context_0_odd(self):
    model = model_factory(context_size=0, locally_normalize=True)
    eval_result = run_toy_task(iter(gen_odd()), model)
    npt.assert_array_less(eval_result['sequence_loss'].mean(), 1e-2)
    assert_same_labels(eval_result['alignment_labels'], eval_result['labels'])

  def test_local_context_0_odd_bigram(self):
    model = model_factory(context_size=0, locally_normalize=True)
    eval_result = run_toy_task(iter(gen_odd_bigram()), model, 1000)
    npt.assert_array_less(1, eval_result['sequence_loss'].mean())

  def test_local_context_1_exact_copy(self):
    model = model_factory(context_size=1, locally_normalize=True)
    eval_result = run_toy_task(iter(gen_exact_copy()), model)
    npt.assert_array_less(eval_result['sequence_loss'].mean(), 1e-2)
    assert_same_labels(eval_result['alignment_labels'], eval_result['labels'])

  def test_local_context_1_odd(self):
    model = model_factory(context_size=1, locally_normalize=True)
    eval_result = run_toy_task(iter(gen_odd()), model)
    npt.assert_array_less(eval_result['sequence_loss'].mean(), 1e-2)
    assert_same_labels(eval_result['alignment_labels'], eval_result['labels'])

  def test_local_context_1_odd_bigram(self):
    model = model_factory(context_size=1, locally_normalize=True)
    eval_result = run_toy_task(iter(gen_odd_bigram()), model, 400)
    npt.assert_array_less(eval_result['sequence_loss'].mean(), 1e-2)
    assert_same_labels(eval_result['alignment_labels'], eval_result['labels'])

  def test_local_context_1_next_label(self):
    model = model_factory(context_size=1, locally_normalize=True)
    eval_result = run_toy_task(iter(gen_next_label()), model, 1000)
    npt.assert_array_less(1, eval_result['sequence_loss'].mean())

  def test_local_context_1_next_even(self):
    model = model_factory(context_size=1, locally_normalize=True)
    eval_result = run_toy_task(iter(gen_next_even()), model, 2000)
    npt.assert_array_less(1, eval_result['sequence_loss'].mean())

  def test_local_context_2_odd_twice_no_repeat(self):
    model = model_factory(context_size=2, locally_normalize=True)
    eval_result = run_toy_task(iter(gen_odd_twice_no_repeat()), model)
    npt.assert_array_equal(jnp.inf, eval_result['sequence_loss'].mean())

  def test_global_context_0_exact_copy(self):
    model = model_factory(context_size=0, locally_normalize=False)
    eval_result = run_toy_task(iter(gen_exact_copy()), model)
    npt.assert_array_less(eval_result['sequence_loss'].mean(), 2e-2)
    assert_same_labels(eval_result['alignment_labels'], eval_result['labels'])

  def test_global_context_0_odd(self):
    model = model_factory(context_size=0, locally_normalize=False)
    eval_result = run_toy_task(iter(gen_odd()), model)
    npt.assert_array_less(eval_result['sequence_loss'].mean(), 1e-2)
    assert_same_labels(eval_result['alignment_labels'], eval_result['labels'])

  def test_global_context_0_odd_bigram(self):
    model = model_factory(context_size=0, locally_normalize=False)
    eval_result = run_toy_task(iter(gen_odd_bigram()), model, 1000)
    npt.assert_array_less(1, eval_result['sequence_loss'].mean())

  def test_global_context_1_exact_copy(self):
    model = model_factory(context_size=1, locally_normalize=False)
    eval_result = run_toy_task(iter(gen_exact_copy()), model)
    npt.assert_array_less(eval_result['sequence_loss'].mean(), 1e-2)
    assert_same_labels(eval_result['alignment_labels'], eval_result['labels'])

  def test_global_context_1_odd(self):
    model = model_factory(context_size=1, locally_normalize=False)
    eval_result = run_toy_task(iter(gen_odd()), model)
    npt.assert_array_less(eval_result['sequence_loss'].mean(), 1e-2)
    assert_same_labels(eval_result['alignment_labels'], eval_result['labels'])

  def test_global_context_1_odd_bigram(self):
    model = model_factory(context_size=1, locally_normalize=False)
    eval_result = run_toy_task(iter(gen_odd_bigram()), model, 400)
    npt.assert_array_less(eval_result['sequence_loss'].mean(), 1e-2)
    assert_same_labels(eval_result['alignment_labels'], eval_result['labels'])

  def test_global_context_1_next_label(self):
    model = model_factory(context_size=1, locally_normalize=False)
    eval_result = run_toy_task(iter(gen_next_label()), model)
    npt.assert_array_less(eval_result['sequence_loss'].mean(), 1e-2)
    assert_same_labels(eval_result['alignment_labels'], eval_result['labels'])

  def test_global_context_1_next_label_bigram(self):
    model = model_factory(context_size=1, locally_normalize=False)
    eval_result = run_toy_task(iter(gen_next_label_bigram()), model, 1000)
    npt.assert_array_less(eval_result['sequence_loss'].mean(), 1e-2)
    assert_same_labels(eval_result['alignment_labels'], eval_result['labels'])

  def test_global_context_1_next_label_bigram_no_stack_input(self):
    model = model_factory(context_size=1, locally_normalize=False)
    eval_result = run_toy_task(
        iter(gen_next_label_bigram(stack_input=False)), model, 2000)
    npt.assert_array_less(1, eval_result['sequence_loss'].mean())

  def test_global_context_1_next_even(self):
    model = model_factory(context_size=1, locally_normalize=False)
    eval_result = run_toy_task(iter(gen_next_even()), model, 2000)
    npt.assert_array_less(1, eval_result['sequence_loss'].mean())


class FrameLabelDependentToyTasksTest(absltest.TestCase):

  def test_local_context_1_max_expansions_1_exact_copy(self):
    model = model_factory(
        context_size=1,
        locally_normalize=True,
        alignment=last.alignments.FrameLabelDependent(1))
    eval_result = run_toy_task(iter(gen_exact_copy()), model)
    npt.assert_array_less(1, eval_result['sequence_loss'].mean())

  def test_local_context_1_max_expansions_1_copy_no_repeat(self):
    model = model_factory(
        context_size=1,
        locally_normalize=True,
        alignment=last.alignments.FrameLabelDependent(1))
    eval_result = run_toy_task(iter(gen_copy_no_repeat()), model)
    npt.assert_array_less(eval_result['sequence_loss'].mean(), 1e-2)
    assert_same_labels(eval_result['alignment_labels'], eval_result['labels'])

  def test_local_context_1_max_expansions_2_odd_twice_no_repeat(self):
    model = model_factory(
        context_size=1,
        locally_normalize=True,
        alignment=last.alignments.FrameLabelDependent(2))
    eval_result = run_toy_task(iter(gen_odd_twice_no_repeat()), model)
    npt.assert_array_less(1, eval_result['sequence_loss'].mean())

  def test_local_context_2_max_expansions_2_odd_twice_no_repeat(self):
    model = model_factory(
        context_size=2,
        locally_normalize=True,
        alignment=last.alignments.FrameLabelDependent(2))
    eval_result = run_toy_task(iter(gen_odd_twice_no_repeat()), model)
    npt.assert_array_less(eval_result['sequence_loss'].mean(), 5e-2)
    assert_same_labels(eval_result['alignment_labels'], eval_result['labels'])


if __name__ == '__main__':
  absltest.main()

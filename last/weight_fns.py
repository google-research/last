# Copyright 2023 The LAST Authors.
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

"""Weight functions."""

import abc
from typing import Callable, Generic, Optional, TypeVar

import einops
from flax import linen as nn
import flax.struct
import jax
import jax.numpy as jnp

# Weight functions are the only components in GNAT with trainable parameters. We
# implement weight functions in two parts: WeightFn and WeightFnCacher.
#
# A WeightFn is a neural network that computes the arc weights for a given
# frame. Sometimes it requires static data that doesn't depend on the frames but
# is expensive to compute (e.g. the context embeddings of the shared-rnn weight
# function). We avoid unnecessarily recomputing such static data by off-loading
# the computation of static data to a separate WeightFnCacher (e.g.
# SharedRNNCacher).
#
# This way, whenever we know the static data doesn't change (e.g. when the
# underlying model parameters don't change such as during inference), we can
# reuse the result from WeightFnCacher as cache.

T = TypeVar('T')


class WeightFn(nn.Module, Generic[T], abc.ABC):
  """Interface for weight functions.

  A weight function is a neural network that computes the arc weights from all
  or some context states for a given frame. A WeightFn is used in pair with a
  WeightFnCacher that produces the static data cache, e.g. JointWeightFn can be
  used with SharedEmbCacher or SharedRNNCacher.
  """

  @abc.abstractmethod
  def __call__(
      self,
      cache: T,
      frame: jnp.ndarray,
      state: Optional[jnp.ndarray] = None) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Computes arc weights for a given frame.

    Args:
      cache: Cached data from the corresponding WeightFnCacher.
      frame: [batch_dims..., feature_size] input frame.
      state: None or int32 array broadcastable to [batch_dims...]. If None,
        compute arc weights for all context states. Otherwise, compute arc
        weights for the specified context state.

    Returns:
      (blank, lexical) tuple.

      If state is None:
      - blank: [batch_dims..., num_context_states] weights for blank arcs.
        blank[..., p] is the weight of producing blank from context state p.
      - lexical: [batch_dims..., num_context_states, vocab_size] weights for
        lexical arcs. lexical[..., p, y] is the weight of producing label y from
        context state p.

      If state is not None:
      - blank: [batch_dims...] weights for blank arcs from the corresponding
        `state`.
      - lexical: [batch_dims..., vocab_size] weights for lexical arcs.
        lexical[..., y] is the weight of producing label y from the
        corresponding `state`.
    """
    raise NotImplementedError


class WeightFnCacher(nn.Module, Generic[T], abc.ABC):
  """Interface for weight function cachers.

  A weight function cacher prepares static data that may require expensive
  computational work. For example: the context state embeddings used by
  JointWeightFn can be from running an RNN on n-gram label sequences
  """

  @abc.abstractmethod
  def __call__(self) -> T:
    """Builds the cached data."""


def hat_normalize(blank: jnp.ndarray,
                  lexical: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Local normalization used in the Hybrid Autoregressive Transducer (HAT) paper.

  The sigmoid of the blank weight is directly interpreted as the probability of
  blank. The lexical probability is then normalized with a log-softmax.

  Args:
    blank: [batch_dims...] blank weight.
    lexical: [batch_dims..., vocab_size] lexical weights.

  Returns:
    Normalized (blank, lexical) weights.
  """
  # Outside normalizer.
  z = jnp.log(1 + jnp.exp(blank))
  normalized_blank = blank - z
  normalized_lexical = nn.log_softmax(lexical) - z[..., jnp.newaxis]
  return normalized_blank, normalized_lexical


def log_softmax_normalize(
    blank: jnp.ndarray,
    lexical: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Standard log-softmax local normalization.

  Weights are concatenated and then normalized together.

  Args:
    blank: [batch_dims...] blank weight.
    lexical: [batch_dims..., vocab_size] lexical weights.

  Returns:
    Normalized (blank, lexical) weights.
  """
  all_weights = jnp.concatenate([blank[..., jnp.newaxis], lexical], axis=-1)
  all_weights = nn.log_softmax(all_weights)
  return all_weights[..., 0], all_weights[..., 1:]


class LocallyNormalizedWeightFn(WeightFn[T]):
  """Wrapper for turning any weight function into a locally normalized one.

  This is the recommended way of obtaining a locally normalized weight function.
  Algorithms such as those that computes the sequence log-loss may rely on a
  weight function being of this type to eliminate unnecessary denominator
  computation.

  It is thus also important for the normalize function to be mathematically
  correct: let (blank, lexical) be the pair of weights produced by the normalize
  function, then `jnp.exp(blank) + jnp.sum(jnp.exp(lexical), axis=-1)` should be
  approximately equal to 1.

  Attributes:
    weight_fn: Underlying weight function.
    normalize: Callable that produces normalized log-probabilities from (blank,
      lexical) weights, e.g. hat_normalize() or log_softmax_normalize().
  """
  weight_fn: WeightFn[T]
  normalize: Callable[[jnp.ndarray, jnp.ndarray],
                      tuple[jnp.ndarray, jnp.ndarray]] = hat_normalize

  def __call__(
      self,
      cache: T,
      frame: jnp.ndarray,
      state: Optional[jnp.ndarray] = None) -> tuple[jnp.ndarray, jnp.ndarray]:
    blank, lexical = self.weight_fn(cache, frame, state)
    # pylint: disable=too-many-function-args
    return self.normalize(blank, lexical)


class JointWeightFn(WeightFn[jnp.ndarray]):
  r"""Common implementation of both the shared-emb and shared-rnn weight functions.

  To use shared-emb weight functions, pair this with a SharedEmbCacher. To use
  shared-rnn weight functions, pair this with a SharedRNNCacher. More generally,
  this weight function works with any WeightFnCacher that produces a
  [num_context_states, embedding_size] context embedding table.

  Attributes:
    vocab_size: Size of the lexical output vocabulary (not including the blank),
      i.e. $|\Sigma|$.
    hidden_size: Hidden layer size.
  """

  vocab_size: int
  hidden_size: int

  @nn.compact
  def __call__(
      self,
      cache: jnp.ndarray,
      frame: jnp.ndarray,
      state: Optional[jnp.ndarray] = None) -> tuple[jnp.ndarray, jnp.ndarray]:
    context_embeddings = cache
    if state is None:
      frame = frame[..., jnp.newaxis, :]
    else:
      context_embeddings = context_embeddings[state]
    # TODO(wuke): Projections empirically helps WER. But doing projection here
    # isn't the most efficient although simpler: context embeddings can be
    # projected in the cacher (and is possibly only necessary for
    # SharedRNNCacher); frame can be projected in the encoder.
    projected_context_embeddings = nn.Dense(
        self.hidden_size, name='project_context_embeddings', use_bias=False)(
            context_embeddings)
    projected_frame = nn.Dense(
        self.hidden_size, name='project_frame', use_bias=False)(
            frame)
    joint = nn.tanh(
        self.param('joint_bias', nn.initializers.zeros, (self.hidden_size,)) +
        projected_context_embeddings + projected_frame)
    blank = jnp.squeeze(nn.Dense(1)(joint), axis=-1)
    lexical = nn.Dense(self.vocab_size)(joint)
    return blank, lexical


class SharedEmbCacher(WeightFnCacher[jnp.ndarray]):
  """A randomly initialized, independent context embedding table.

  The result context embedding table can be used with JointWeightFn.
  """
  num_context_states: int
  embedding_size: int

  @nn.compact
  def __call__(self) -> jnp.ndarray:
    return self.param('context_embeddings', nn.linear.default_embed_init,
                      (self.num_context_states, self.embedding_size))


class SharedRNNCacher(WeightFnCacher[jnp.ndarray]):
  """Builds a context embedding table by running n-gram context labels through an RNN.

  This is usually used with last.contexts.FullNGram, where num_context_states =
  sum(vocab_size**i for i in range(context_size + 1)). The result context
  embedding table can be used with JointWeightFn.
  """
  vocab_size: int
  context_size: int
  rnn_size: int
  rnn_embedding_size: int
  # TODO(wuke): Use LSTM with layer norm.
  rnn_cell: nn.recurrent.RNNCellBase = flax.struct.field(
      default_factory=nn.OptimizedLSTMCell)

  @nn.compact
  def __call__(self) -> jnp.ndarray:

    def tile_rnn_state(state):
      return einops.repeat(state, 'n ... -> (n v) ...', v=self.vocab_size)

    # Label 0 is an extra label for seeding the start state.
    # Labels [1, vocab_size] are the actual lexical labels.
    embed = nn.Embed(self.vocab_size + 1, self.rnn_embedding_size)
    # TODO(wuke): Proper rng handling.
    dummy_rng = jax.random.PRNGKey(0)
    rnn_states, start_embedding = self.rnn_cell(
        self.rnn_cell.initialize_carry(dummy_rng, (1,), self.rnn_size),
        embed(jnp.array([0])))
    parts = [start_embedding]
    inputs = embed(jnp.arange(1, self.vocab_size + 1))
    for _ in range(self.context_size):
      rnn_states, embeddings = self.rnn_cell(
          jax.tree_util.tree_map(tile_rnn_state, rnn_states), inputs)
      inputs = einops.repeat(inputs, 'n ... -> (v n) ...', v=self.vocab_size)
      parts.append(embeddings)
    return jnp.concatenate(parts, axis=0)


class NullCacher(WeightFnCacher[type(None)]):
  """A cacher that simply returns None.

  Mainly used with TableWeightFn for unit testing.
  """

  @nn.compact
  def __call__(self) -> None:
    return None


class TableWeightFn(WeightFn[type(None)]):
  """Weight function that looks up a fixed table, useful for testing.

  Attributes:
    table: [batch_dims..., input_vocab_size, num_context_states, 1 + vocab_size]
      arc weight table. For each input frame, we simply cast the 0-th element
      into an integer "input label" and look up the corresponding weights. The
      weights of blank arcs are stored at `table[..., 0]`, and the weights of
      lexical arcs at `table[..., 1:]`.
  """
  table: jnp.ndarray

  @nn.compact
  def __call__(
      self,
      cache: None,
      frame: jnp.ndarray,
      state: Optional[jnp.ndarray] = None) -> tuple[jnp.ndarray, jnp.ndarray]:
    del cache

    *batch_dims, input_vocab_size, num_context_states, _ = self.table.shape
    if frame.shape[:-1] != tuple(batch_dims):
      raise ValueError(f'frame should have batch_dims={tuple(batch_dims)} but '
                       f'got {frame.shape[:-1]}')

    frame_mask = nn.one_hot(frame[..., 0].astype(jnp.int32), input_vocab_size)
    weights = jnp.einsum('...xcy,...x->...cy', self.table, frame_mask)

    if state is not None:
      state = jnp.broadcast_to(state, batch_dims)
      state_mask = nn.one_hot(state, num_context_states)
      weights = jnp.einsum('...cy,...c->...y', weights, state_mask)

    blank = weights[..., 0]
    lexical = weights[..., 1:]
    return blank, lexical

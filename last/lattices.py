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

"""Recognition lattice."""

from collections.abc import Callable, Sequence
import functools
from typing import Any, Generic, Optional, TypeVar

import flax.linen as nn
import jax
import jax.numpy as jnp

from last import alignments
from last import contexts
from last import semirings
from last import weight_fns

DType = Any
T = TypeVar('T')


class RecognitionLattice(nn.Module, Generic[T]):
  """Recognition lattice in GNAT-style formulation and operations over it.

  A RecognitionLattice provides operations used in training and inference, such
  as computing the negative-log-probability loss, or finding the highest scoring
  alignment path.

  Following the formulation in GNAT, we combines the three modelling components
  to define a RecognitionLattice:
  -   Context dependency: The finite automaton that models output history. See
      last.contexts.ContextDependency for details.
  -   Alignment lattice: The finite automaton that models the alignment between
      input frames and output labels. See
      last.alignments.TimeSyncAlignmentLattice for details.
  -   Weight function: The neural network that produces arc weights from any
      context state given an input frame. See last.weight_fns for details.

  Given a sequence of `T` input frames, its recognition lattice is the following
  finite automaton:
  -   States: The states are pairs of an alignment state and a context state.
      For any alignment state (t, a) (t is the index of the current frame) and
      context state c, there is a state (t, a, c) in the recognition lattice.
  -   Start state: (0, s_a, s_c), where s_a is the start state in the
      frame-local alignment lattice (see
      last.contexts.TimeSyncAlignmentLattice), and s_c is the start state in the
      context dependency.
  -   Final states: (T, s_a, c) for any context state c.
  -   Arcs:
      -   Blank arcs: For any blank arc `(t, a) -> (t', a')` in the alignment
          lattice, and any context state c, there is an arc
          `(t, a, c) --blank-> (t', a', c)` in the recognition lattice.
      -   Lexical arcs: For any lexical arc `(t, a) -> (t', a')` in the
          alignment lattice, and any arc `c --y-> c'` in the context dependency,
          there is an arc `(t, a, c) --y-> (t', a', c')` in the recognition
          lattice.
  -   Arc weights: For each state (t, a, c), the weight function receives the
      t-th frame and the context state c, and produces arc weights for blank
      and lexical arcs. Notably while in principle weight functions can also
      depend on the alignment state, in practice we haven't yet encountered a
      compelling reason for such weight functions. Thus for the sake of
      simplicity, weight functions currently only depend on the frame and the
      context state.

  The arc weights can be used to model the conditional distribution of the paths
  in the recognition lattice, especially those that lead to the reference label
  sequence. A RecognitionLattice can be either a locally normalized model, or
  a globally normalized model,
  -   A locally normalized model uses last.weight_fns.LocallyNormalizedWeightFn,
      where the arc weights from the same recognition lattice state add up to 1
      after taking an exponential. The probability of
      P(alignment labels | frames) is simply the product of arc weights on the
      alignment path after exponential.
  -   A globally normalized model uses a WeightFn that is not a subclass of
      last.weight_fns.LocallyNormalizedWeightFn. To obtain a probabilistic
      interpretation, we normalize the path weights with the sum of the
      exponentiated weights of all possible paths in the recognition lattice.

  Globally normalized models are more expensive to train, but they have various
  advantages. See the GNAT paper for more details.

  Attributes:
    context: Context dependency.
    alignment: Alignment lattice.
    weight_fn_cacher_factory: Callable that builds a WeightFnCacher given the
      context dependency.
    weight_fn_factory: Callable that builds a WeightFn given the context
      dependency.
  """

  context: contexts.ContextDependency
  alignment: alignments.TimeSyncAlignmentLattice

  # We use factories instead of fields of nn.Module because of restrictions from
  # nn.custom_vjp (needed for _forward_backward). The important thing though is
  # that weight_fn and weight_fn_cacher have to be child modules, so that the
  # variable ownership is clear for nn.custom_vjp.
  weight_fn_cacher_factory: Callable[[contexts.ContextDependency],
                                     weight_fns.WeightFnCacher[T]]
  weight_fn_factory: Callable[[contexts.ContextDependency],
                              weight_fns.WeightFn[T]]

  def setup(self):
    self.weight_fn_cacher = self.weight_fn_cacher_factory(self.context)
    self.weight_fn = self.weight_fn_factory(self.context)

  def build_cache(self) -> T:
    """Builds the weight function cache.

    Weight functions are implemented as a pair of WeightFn and WeightFnCacher to
    avoid unnecessary recomputation (see last.weight_fns for more details).
    build_cache() builds the cached static data that can be used in other public
    methods.

    Returns:
      Cached data.
    """
    return self.weight_fn_cacher()

  # Should a public method take an optional cache argument?
  #
  # -   If the method operates on a sequence: cache is optional. The method
  #     should build the cache itself if necessary.
  # -   If the method operates on a single frame: cache should be required. The
  #     method should NOT build the cache itself. Methods operating on a single
  #     frame are expected to be called repeatedly. We want to prevent the users
  #     from making the mistake of recomputing the cache repeatedly.

  def __call__(self,
               frames: jnp.ndarray,
               num_frames: jnp.ndarray,
               labels: jnp.ndarray,
               num_labels: jnp.ndarray,
               cache: Optional[T] = None) -> jnp.ndarray:
    """Compute the negative sequence log-probability loss.

    There can be multiple alignment paths from the input frames to the output
    labels. The conditional probability P(labels | frames) is thus the sum
    of probabilities P(alignment labels | frames) for all possible alignments
    that produce the given label sequence. Interpreting the arc weights as
    (possibly unnormalized) log-probabilities, this function computes
    -log P(labels | frames) for both locally and globally normalized models.

    Args:
      frames: [batch_dims..., max_num_frames, feature_size] padded frame
        sequences.
      num_frames: [batch_dims...] number of frames.
      labels: [batch_dims..., max_num_labels] padded label sequences.
      num_labels: [batch_dims...] number of labels.
      cache: Optional weight function cache data.

    Returns:
      [batch_dims...] negative sequence log-prob loss.
    """
    batch_dims = num_frames.shape
    if frames.shape[:-2] != batch_dims:
      raise ValueError('frames and num_frames have different batch_dims: '
                       f'{frames.shape[:-2]} vs {batch_dims}')
    if labels.shape[:-1] != batch_dims:
      raise ValueError('labels and num_frames have different batch_dims: '
                       f'{labels.shape[:-1]} vs {batch_dims}')
    if num_labels.shape != batch_dims:
      raise ValueError('num_labels and num_frames have different batch_dims: '
                       f'{num_labels.shape} vs {batch_dims}')

    semiring = semirings.Log
    if cache is None:
      cache = self.weight_fn_cacher()
    numerator = self._string_forward(
        cache=cache,
        frames=frames,
        num_frames=num_frames,
        labels=labels,
        num_labels=num_labels,
        semiring=semiring)
    if isinstance(self.weight_fn, weight_fns.LocallyNormalizedWeightFn):
      return -numerator
    denominator, _ = self._forward(
        cache=cache, frames=frames, num_frames=num_frames, semiring=semiring)
    return denominator - numerator

  def shortest_path(
      self,
      frames: jnp.ndarray,
      num_frames: jnp.ndarray,
      cache: Optional[T] = None
  ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Computes the shortest path in the recognition lattice.

    The shortest path is the path with the highest score, in other words, the
    "shoretst" path under the max-tropical semiring.

    Args:
      frames: [batch_dims..., max_num_frames, feature_size] padded frame
        sequences.
      num_frames: [batch_dims...] number of frames.
      cache: Optional weight function cache data.

    Returns:
      (alignment_labels, num_alignment_labels, path_weights) tuple,
      -   alignment_labels: [batch_dims..., max_num_alignment_labels] padded
          alignment labels, either blank (0) or lexical (1 to vocab_size).
      -   num_alignment_labels: [batch_dims...] number of alignment labels.
      -   path_weights: [batch_dims...] path weights.
    """
    batch_dims = num_frames.shape
    if frames.shape[:-2] != batch_dims:
      raise ValueError('frames and num_frames have different batch_dims: '
                       f'{frames.shape[:-2]} vs {batch_dims}')
    max_num_frames = frames.shape[-2]
    num_alignment_states = self.alignment.num_states()

    if cache is None:
      cache = self.weight_fn_cacher()

    # Find shortest path by differentiating shortest distance under the tropical
    # semiring.
    def forward(lattice: 'RecognitionLattice', lexical_mask: jnp.ndarray):
      path_weights, _ = lattice._forward(  # pylint: disable=protected-access
          cache=cache,
          frames=frames,
          num_frames=num_frames,
          semiring=semirings.MaxTropical,
          # _forward expects lexical_mask to be a length num_alignment_states
          # Sequence whose elements are broadcastable to
          # [batch_dims..., max_num_frames, num_context_states, vocab_size].
          lexical_mask=[
              lexical_mask[..., i, jnp.newaxis, :]
              for i in range(num_alignment_states)
          ])
      return path_weights

    _, vocab_size = self.context.shape()
    lexical_mask = jnp.zeros(
        [*batch_dims, max_num_frames, num_alignment_states, vocab_size])
    path_weights, vjp_fn = nn.vjp(
        forward, self, lexical_mask, vjp_variables=False)
    _, viterbi_lexical_mask = vjp_fn(jnp.ones_like(path_weights))
    is_blank = jnp.all(viterbi_lexical_mask == 0, axis=-1)
    alignment_labels = jnp.where(is_blank, 0,
                                 1 + jnp.argmax(viterbi_lexical_mask, axis=-1))
    # Flatten into [batch_dims..., max_num_frames * num_alignment_states]
    alignment_labels = alignment_labels.reshape([*batch_dims, -1])
    num_alignment_labels = num_alignment_states * num_frames
    return alignment_labels, num_alignment_labels, path_weights

  # Private methods.

  # TODO(wuke): Add support for lexical_mask for forced decoding.
  # TODO(wuke): Add support for composite semirings (e.g. expectation).
  def _string_forward(self, cache: T, frames: jnp.ndarray,
                      num_frames: jnp.ndarray, labels: jnp.ndarray,
                      num_labels: jnp.ndarray,
                      semiring: semirings.Semiring[jnp.ndarray]) -> jnp.ndarray:
    """Shortest distance on the intersection of the recognition lattice and an output string (the label sequence) computed using the forward algorithm.

    Args:
      cache: Weight function cache data.
      frames: [batch_dims..., max_num_frames, feature_size] padded frame
        sequences.
      num_frames: [batch_dims...] number of frames.
      labels: [batch_dims..., max_num_labels] padded label sequence.
      num_labels: [batch_dims...] number of labels.
      semiring: Semiring to use for shortest distance computation.

    Returns:
      [batch_dims...] shortest distance.
    """
    batch_dims = num_frames.shape
    if frames.shape[:-2] != batch_dims:
      raise ValueError('frames and num_frames have different batch_dims: '
                       f'{frames.shape[:-2]} vs {batch_dims}')
    if labels.shape[:-1] != batch_dims:
      raise ValueError('labels and num_frames have different batch_dims: '
                       f'{labels.shape[:-1]} vs {batch_dims}')
    if num_labels.shape != batch_dims:
      raise ValueError('num_labels and num_frames have different batch_dims: '
                       f'{num_labels.shape} vs {batch_dims}')

    # Calculate arc weights for all visited context states.
    #
    # We can't fit into memory all
    # O(batch_size * max_num_frames * (max_num_labels + 1) * (vocab_size + 1))
    # arcs, thus we use a scan loop over the (max_num_labels + 1) axis to
    # produce just the O(batch_size * max_num_frames * (max_num_labels + 1))
    # arcs actually needed later. This is better than scanning over the
    # max_num_frames axis because weight_fn can be vectorized over multiple
    # frames for the same state (weight_fn with states often involves
    # gathering).

    # Before vmaps
    # - frame is [batch_dims..., hidden_size]
    # - state is [batch_dims...]
    # Results are ([batch_dims...], [batch_dims..., vocab_size]).
    compute_weights = (
        lambda weight_fn, frame, state: weight_fn(cache, frame, state))
    # Add time dimension on frame
    # - frame is [batch_dims..., max_num_frames, hidden_size]
    # - state is [batch_dims...]
    # Results are ([batch_dims..., max_num_frames],
    # [batch_dims..., max_num_frames, vocab_size]).
    compute_weights = nn.vmap(
        compute_weights,
        variable_axes={'params': None},
        split_rngs={'params': False},
        in_axes=(-2, None),
        out_axes=(-1, -2))

    def gather_weight(weights, y):
      # weights: [batch_dims..., max_num_frames, vocab_size]
      # y: [batch_dims..., max_num_frames]
      # weights are for labels [1, vocab_size], so y-1 are the corresponding
      # indicies. one_hot(-1) is safe (all zeros).
      mask = nn.one_hot(y - 1, weights.shape[-1])
      return jnp.einsum('...TV,...V->...T', weights, mask)

    def weight_step(weight_fn, carry, inputs):
      del carry
      state, next_label = inputs
      blank_weight, lexical_weights = compute_weights(weight_fn, frames, state)
      lexical_weight = gather_weight(lexical_weights, next_label)
      return None, (blank_weight, lexical_weight)

    # prevent_cse is not needed in loops. Turning it off allows the compiler to
    # better optimize the loop step.
    weight_step = nn.remat(weight_step, prevent_cse=False)

    # [batch_dims..., max_num_labels + 1]
    context_states = self.context.walk_states(labels)
    context_next_labels = jnp.concatenate(
        [labels, jnp.ones_like(labels[..., :1])], axis=-1)
    # [batch_dims..., max_num_frames, max_num_labels+1]
    _, (blank_weight, lexical_weight) = nn.scan(
        weight_step,
        variable_broadcast='params',
        split_rngs={'params': False},
        in_axes=len(batch_dims),
        out_axes=len(batch_dims) + 1)(self.weight_fn, None,
                                      (context_states, context_next_labels))

    # Dynamic program for summing up all alignment paths. Actual work is done by
    # alignment.string_forward(). This function mostly takes care of padding
    # frames.
    def shortest_distance_step(carry, inputs):
      # alpha: [batch_dims..., max_num_labels + 1]
      t, alpha = carry
      # blank, lexical: [batch_dims..., max_num_labels + 1]
      blank, lexical = inputs
      # We current only support alignment-state invariant weights.
      blank = [blank for _ in range(self.alignment.num_states())]
      lexical = [lexical for _ in range(self.alignment.num_states())]
      next_alpha = self.alignment.string_forward(
          alpha=alpha, blank=blank, lexical=lexical, semiring=semiring)
      is_padding = (t >= num_frames)[..., jnp.newaxis]
      next_alpha = jnp.where(is_padding, alpha, next_alpha)
      return (t + 1, next_alpha), None

    num_alpha_states = labels.shape[-1] + 1
    init_alpha = _init_context_state_weights(
        batch_dims=batch_dims,
        dtype=lexical_weight.dtype,
        num_states=num_alpha_states,
        start=0,
        semiring=semiring)
    (_, alpha), _ = jax.lax.scan(
        shortest_distance_step, (0, init_alpha),
        jax.tree_util.tree_map(
            functools.partial(_to_time_major, num_batch_dims=len(batch_dims)),
            (blank_weight, lexical_weight)))
    is_final = num_labels[..., jnp.newaxis] == jnp.arange(num_alpha_states)
    return semiring.sum(
        jnp.where(is_final, alpha, semiring.zeros([], alpha.dtype)), axis=-1)

  # TODO(wuke): Find a way to create a public shortest_distance() method whose
  # interface isn't overly complex.
  def _forward(
      self,
      cache: T,
      frames: jnp.ndarray,
      num_frames: jnp.ndarray,
      semiring: semirings.Semiring[jnp.ndarray],
      blank_mask: Optional[Sequence[jnp.ndarray]] = None,
      lexical_mask: Optional[Sequence[jnp.ndarray]] = None,
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Shortest distance on the recognition lattice computed using the forward algorithm.

    It is often useful to differentiate through shortest distance with respect
    to arc weights. For example, under the log semiring, that gives us arc
    marginals; whereas under the tropical semiring, that gives us shortest path.
    To make that possible while the arc weights are being computed on the fly,
    we can pass in zero-valued masks. The masks are added onto arc weights, and
    because d f(x + y) / dy | y=0 = d f(x) / dx, we can get gradients with
    respect to arc weights (i.e. x) by differentiating over the masks (i.e. y).

    Args:
      cache: Weight function cache data.
      frames: [batch_dims..., max_num_frames, feature_size] padded frame
        sequences.
      num_frames: [batch_dims...] number of frames.
      semiring: Semiring to use for shortest distance computation.
      blank_mask: Optional length num_alignment_states sequence whose elements
        are broadcastable to [batch_dims..., max_num_frames,
        num_context_states].
      lexical_mask: Optional length num_alignment_states sequence whose elements
        are broadcastable to [batch_dims..., max_num_frames, num_context_states,
        vocab_size].

    Returns:
      (shortest_distance, alpha_0_to_T_minus_1) tuple,
      -   shortest_distance: [batch_dims...] shortest distance.
      -   alpha_0_to_T_minus_1: [batch_dims..., max_num_frames,
          num_context_states] forward weights after observing 0 to T-1 frames
          (T is the number of frames in the sequence).
    """
    batch_dims = num_frames.shape
    if frames.shape[:-2] != batch_dims:
      raise ValueError('frames and num_frames have different batch_dims: '
                       f'{frames.shape[:-2]} vs {batch_dims}')
    if blank_mask is not None and len(
        blank_mask) != self.alignment.num_states():
      raise ValueError(
          'The length of blank_mask should be equal to '
          f'{self.alignment.num_states()} (the number of alignment states), '
          f'but is {len(blank_mask)}')
    if lexical_mask is not None and len(
        lexical_mask) != self.alignment.num_states():
      raise ValueError(
          'The length of lexical_mask should be equal to '
          f'{self.alignment.num_states()} (the number of alignment states), '
          f'but is {len(lexical_mask)}')

    # Dynamic program for summing up all alignment paths.
    def step(weight_fn, carry, inputs):
      # alpha: [batch_dims..., num_context_states]
      t, alpha = carry
      # frame: [batch_dims..., hidden_size]
      # blank_mask: None or [batch_dims...]
      # lexical_mask: None or broadcastable to
      #   [batch_dims..., num_alignment_states, vocab_size]
      frame, blank_mask, lexical_mask = inputs
      # blank: [batch_dims..., num_context_states]
      # lexical: [batch_dims..., num_context_states, vocab_size]
      blank, lexical = weight_fn(cache, frame)
      # We currently only support alignment-state-invariant weights.
      blank = [blank for _ in range(self.alignment.num_states())]
      lexical = [lexical for _ in range(self.alignment.num_states())]
      if blank_mask is not None:
        blank = [b + m for b, m in zip(blank, blank_mask)]
      if lexical_mask is not None:
        lexical = [l + m for l, m in zip(lexical, lexical_mask)]
      next_alpha = self.alignment.forward(
          alpha=alpha,
          blank=blank,
          lexical=lexical,
          context=self.context,
          semiring=semiring)
      is_padding = (t >= num_frames)[..., jnp.newaxis]
      next_alpha = jnp.where(is_padding, alpha, next_alpha)
      return (t + 1, next_alpha), alpha

    # Reduce memory footprint when using autodiff.
    #
    # For the log semiring, this is not as fast or memory efficient as forward-
    # backward, but still better than the defaults (i.e. no remat or no saving
    # intermediates at all).
    #
    # For the tropical semiring, this should be equivalent to no remat.
    def save_small(prim, *args, **params):
      y, _ = prim.abstract_eval(*args, **params)
      greater_than_1_dims = len([None for i in y.shape if i > 1])
      save = greater_than_1_dims <= (len(batch_dims) + 1)
      return save

    # prevent_cse is not needed in loops. Turning it off allows the compiler to
    # better optimize the loop step.
    step = nn.remat(step, prevent_cse=False, policy=save_small)

    init_t = jnp.array(0)
    init_alpha = _init_context_state_weights(
        batch_dims=batch_dims,
        # TODO(wuke): Find a way to do this with jax.eval_shape.
        dtype=self.weight_fn(cache, frames[..., 0, :])[0].dtype,
        num_states=self.context.shape()[0],
        start=self.context.start(),
        semiring=semiring)
    init_carry = (init_t, init_alpha)

    inputs = (frames, blank_mask, lexical_mask)
    (_, alpha_T), alpha_0_to_T_minus_1 = nn.scan(  # pylint: disable=invalid-name
        step,
        variable_broadcast='params',
        split_rngs={'params': False},
        in_axes=len(batch_dims),
        out_axes=len(batch_dims))(self.weight_fn, init_carry, inputs)

    return semiring.sum(alpha_T, axis=-1), alpha_0_to_T_minus_1


def _init_context_state_weights(
    batch_dims: Sequence[int], dtype: DType, num_states: int, start: int,
    semiring: semirings.Semiring[jnp.ndarray]) -> jnp.ndarray:
  is_start = jnp.arange(num_states) == start
  weights = jnp.where(is_start, semiring.ones([], dtype),
                      semiring.zeros([], dtype))
  return jnp.broadcast_to(weights, (*batch_dims, num_states))


def _to_time_major(x: jnp.ndarray, num_batch_dims: int) -> jnp.ndarray:
  # [batch_dims..., time, ...] -> [time, batch_dims..., ...]
  axes = [
      num_batch_dims,
      *range(num_batch_dims),
      *range(num_batch_dims + 1, x.ndim),
  ]
  return jnp.transpose(x, axes)


def _to_batch_major(x: jnp.ndarray, num_batch_dims: int) -> jnp.ndarray:
  # [time, batch_dims..., ...] -> [batch_dims..., time, ...]
  axes = [
      *range(1, num_batch_dims + 1),
      0,
      *range(num_batch_dims + 1, x.ndim),
  ]
  return jnp.transpose(x, axes)

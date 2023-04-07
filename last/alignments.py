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

"""Alignment lattices."""

import abc
from collections.abc import Sequence
from typing import Optional

import flax.struct
import jax.numpy as jnp

from last import contexts
from last import semirings


# TODO(wuke): Improve documentation.
class TimeSyncAlignmentLattice(abc.ABC):
  r"""Interface for time synchronous alignment lattices.

  Frame-dependent and k-constrained label-frame-dependent alignment lattices are
  examples of time synchronous alignment lattices. See Sections 3 and 4 of the
  GNAT paper for more details.

  The alignment lattice is intersected with the context dependency to form the
  topology of a recognition lattice. class last.RecognitionLattice carries out
  this intersection on the fly with the help of methods of
  TimeSyncAlignmentLattice.

  To describe time synchronous alignment lattices, we first introduce the notion
  of a frame-local alignment lattice. A frame-local alignment lattice is an
  acyclic deterministic finite automaton with 2 input labels, "lexical" and
  "blank" with a single final state f.

  Let Q be the set of states in the frame-local alignment and E be the set of
  arcs, a time synchronous alignment lattice is then the same frame-local
  alignment lattice repeated num_frames times,
  -   The states are {(t, a) | 0 <= t < num_frames, a \in Q - {f}} U
      {(num_frames, s)};
  -   The start state is (0, s);
  -   The final state is (T, s);
  -   For any arc (a, y, b), b != f, in E, there is an arc ((t, a), y, (t, b));
  -   For any arc (a, y, f) in E, there is an arc ((t, a), y, (t + 1, s)).
  """

  @abc.abstractmethod
  def num_states(self) -> int:
    """Number of non-final frame-local alignment states, `num_alignment_states`.
    """

  @abc.abstractmethod
  def start(self) -> int:
    """Start state of the frame-local alignment lattice."""

  @abc.abstractmethod
  def blank_next(self, state: int) -> Optional[int]:
    """Next alignment state id when taking the blank arc.

    Args:
      state: A state id in the range [0, num_alignment_states).

    Returns:
      None if there is no blank arc leaving `state`.
      The start state id if the blank arc leads to the final state.
      Otherwise, an ordinary state id in the range [0, num_alignment_states).
    """

  @abc.abstractmethod
  def lexical_next(self, state: int) -> Optional[int]:
    """Next alignment state id when taking the lexical arc.

    Args:
      state: A state id in the range [0, num_alignment_states).

    Returns:
      None if there is no blank arc leaving `state`.
      The start state id if the blank arc leads to the final state.
      Otherwise, an ordinary state id in the range [0, num_alignment_states).
    """

  @abc.abstractmethod
  def topological_visit(self) -> list[int]:
    """Produces non-final frame-local alignment state ids in a topological order.
    """

  @abc.abstractmethod
  def forward(self, alpha: jnp.ndarray, blank: Sequence[jnp.ndarray],
              lexical: Sequence[jnp.ndarray],
              context: contexts.ContextDependency,
              semiring: semirings.Semiring[jnp.ndarray]) -> jnp.ndarray:
    """Processes one frame in the recognition lattice forward algorithm.

    On a recognition lattice, the forward algorithm computes the shortest
    distance, i.e. the sum of path weights reaching all final states.
    The shortest distance can be computed frame by frame with the help of
    TimeSyncAlignmentLattice.forward:
    1.  At time t, after observing the previous frame (t-1), we know the forward
        weights of states (t, s, c) for all context states c (`alpha`).
    2.  We also know all the arc weights from states (t, a, c), for all non-
        final frame-local alignment states a and all context states c (`blank`
        and `lexical`).
    3.  We also know the context dependency.
    4.  With the information above, TimeSyncAlignmentLattice.forward computes
        the forward weights of states (t+1, s, c) for all context states c.

    Args:
      alpha: [batch_dims..., num_context_states] forward weights after observing
        the previous frame.
      blank: length num_alignment_states Sequence of [batch_dims...,
        num_context_states] blank weights for the current frame, one for each
        frame-local alignment state.
      lexical: length num_alignment_states Sequence of [batch_dims...,
        num_context_states, vocab_size] lexical weights for the current frame,
        one for each frame-local alignment state.
      context: Context dependency.
      semiring: Semiring.

    Returns:
      [batch_dims..., num_context_states] forward weights after observing the
      current frame.
    """

  @abc.abstractmethod
  def backward(
      self, alpha: jnp.ndarray, blank: Sequence[jnp.ndarray],
      lexical: Sequence[jnp.ndarray], beta: jnp.ndarray, log_z: jnp.ndarray,
      context: jnp.ndarray
  ) -> tuple[jnp.ndarray, list[jnp.ndarray], list[jnp.ndarray]]:
    """Processes one frame in the recognition lattice backward algorithm.

    On a recognition lattice, the backward algorithm computes the arc marginals
    under last.semirings.Log (the marginal probability of taking each lexical or
    blank arc), as wells the sum of path weights from a recognition lattice
    state to all final states (backward weights). The marginals and backward
    weights can be computed frame by frame with the help of
    TimeSyncAlignmentLattice.backward:
    1.  At time t, from the forward algorithm's results, we know the forward
        weights of states (t, s, c) for all context states c (`alpha`) and the
        overall shortest distance (sum of weights of all accepting paths,
        `log_z`).
    2.  We also know all the arc weights from states (t, a, c), for all non-
        final frame-local alignment states a and all context states c (`blank`
        and `lexical`).
    3.  At time t, after observing frame (t+1), we know the backward weights of
        states (t + 1, s, c) for all context states c (`beta`).
    3.  We also know the context dependency.
    4.  With the information above, TimeSyncAlignmentLattice.forward computes
        the backward weights of states (t, s, c) for all context states c.

    Args:
      alpha: [batch_dims..., num_context_states] forward weights after observing
        the previous frame.
      blank: length num_alignment_states Sequence of [batch_dims...,
        num_context_states] blank weights for the current frame, one for each
        frame-local alignment state.
      lexical: length num_alignment_states Sequence of [batch_dims...,
        num_context_states, vocab_size] lexical weights for the current frame,
        one for each frame-local alignment state.
      beta: [batch_dims..., num_context_states] backward weights after observing
        the next frame.
      log_z: [batch_dims...] denominator, i.e. the sum of weights of all
        accepting paths.
      context: Context dependency.

    Returns:
      (next_beta, blank_marginal, lexical_marginal):
      -   next_beta: [batch_dims..., num_context_states] backward weights after
          observing the current frame.
      -   blank_marginal: length num_alignment_states list of [batch_dims...,
          num_context_states] marginals of blank arcs, one for each frame-local
          alignment state.
      -   lexical_marginal: length num_alignment_states list of [batch_dims...,
          num_context_states, vocab_size] marginals of lexical arcs, one for
          each frame-local alignment state.
    """

  @abc.abstractmethod
  def string_forward(self, alpha: jnp.ndarray, blank: Sequence[jnp.ndarray],
                     lexical: Sequence[jnp.ndarray],
                     semiring: semirings.Semiring[jnp.ndarray]) -> jnp.ndarray:
    """Processes one frame in the recognition lattice forward algorithm after the intersection with an output string.

    Because the recognition lattice's topology is the intersection of an
    alignment lattice and the context dependency, the intersection between
    the recognition lattice and an output string is thus equivalent to
    first intersecting the context dependency with the output string (i.e.
    last.contexts.ContextDependency.walk_states), and then intersecting the
    alignment lattice with the result.

    On this intersected lattice, the forward algorithm computes the shortest
    distance, i.e. the sum of path weights reaching all final states.
    The shortest distance can be computed frame by frame with the help of
    TimeSyncAlignmentLattice.string_forward:
    1.  At time t, after observing the previous frame (t-1), we know the forward
        weights of states (t, s, c) for all context states c after the first
        intersection. This is passed to TimeSyncAlignmentLattice.forward as
        `alpha`. Note the first intersection results in a string acceptor, with
        only `output_length + 1` context states.
    2.  We also know all the arc weights from states (t, a, c), for all non-
        final frame-local alignment states a and all context states c after the
        first intersection. These are passed to TimeSyncAlignmentLattice.forward
        as `blank` and `lexical`.
    3.  We no longer need to know the context dependency because the result of
        the first intersection is a simple chain.
    4.  With the information above, TimeSyncAlignmentLattice.string_forward
        computes the forward weights of state (t+1, s, c) for all context states
        c after the first intersection.

    Args:
      alpha: [batch_dims..., output_length + 1] forward weights after observing
        the previous frame.
      blank: length num_alignment_states Sequence of [batch_dims...,
        output_length + 1] blank weights for the current frame, one for each
        frame-local alignment state.
      lexical: length num_alignment_states Sequence of [batch_dims...,
        output_length + 1] lexical weights for the current frame, one for each
        frame local alignment state.
      semiring: Semiring.

    Returns:
      [batch_dims..., output_length + 1] forward weights after observing the
      current frame.
    """


def shift_down(x: jnp.ndarray,
               semiring: semirings.Semiring[jnp.ndarray]) -> jnp.ndarray:
  """Shifts values down by 1 position.

  This is a useful helper function for implementing string_forward().

  Args:
    x: [batch_dims..., N] input values.
    semiring: Semiring to use for filling in zero values.

  Returns:
    [batch_dims..., N] output values, where output[..., i + 1] = x[..., i] and
    output[..., 0] = semiring zero.
  """
  return jnp.concatenate(
      [semiring.zeros((*x.shape[:-1], 1), x.dtype), x[..., :-1]], axis=-1)


def check_num_weights(alignment: TimeSyncAlignmentLattice,
                      blank: Sequence[jnp.ndarray],
                      lexical: Sequence[jnp.ndarray]):
  """Ensures that there are correct numbers of weight arrays."""
  num_states = alignment.num_states()
  if len(blank) != num_states:
    raise ValueError(
        f'blank should be a length {num_states} sequence of ndarrays, '
        f'but got length {len(blank)}')
  if len(lexical) != num_states:
    raise ValueError(
        f'lexical should be a length {num_states} sequence of ndarrays, '
        f'but got length {len(lexical)}')


class FrameDependent(TimeSyncAlignmentLattice):
  """Frame dependent alignment lattice.

  Each frame is aligned to either a lexical label or a blank label.
  """

  def num_states(self) -> int:
    return 1

  def start(self) -> int:
    return 0

  def blank_next(self, state: int) -> Optional[int]:
    return 0

  def lexical_next(self, state: int) -> Optional[int]:
    return 0

  def topological_visit(self) -> list[int]:
    return [0]

  def forward(self, alpha: jnp.ndarray, blank: Sequence[jnp.ndarray],
              lexical: Sequence[jnp.ndarray],
              context: contexts.ContextDependency,
              semiring: semirings.Semiring[jnp.ndarray]) -> jnp.ndarray:
    check_num_weights(self, blank, lexical)
    # alpha: [batch_dims..., num_context_states]
    # blank[0]: [batch_dims..., num_context_states]
    # lexical[0]: [batch_dims..., num_context_states, vocab_size]
    return semiring.plus(
        semiring.times(alpha, blank[0]),
        context.forward_reduce(
            semiring.times(alpha[..., jnp.newaxis], lexical[0]), semiring))

  def backward(
      self, alpha: jnp.ndarray, blank: Sequence[jnp.ndarray],
      lexical: Sequence[jnp.ndarray], beta: jnp.ndarray, log_z: jnp.ndarray,
      context: contexts.ContextDependency
  ) -> tuple[jnp.ndarray, list[jnp.ndarray], list[jnp.ndarray]]:
    check_num_weights(self, blank, lexical)
    # alpha: [batch_dims..., num_context_states]
    # blank: [batch_dims..., num_context_states]
    # lexical: [batch_dims..., num_context_states, vocab_size]
    # beta: [batch_dims..., num_context_states]
    # log_z: [batch_dims...]
    blank_beta = blank[0] + beta
    lexical_beta = lexical[0] + context.backward_broadcast(beta)
    log_scale = alpha - log_z[..., jnp.newaxis]
    blank_marginal = jnp.exp(blank_beta + log_scale)
    lexical_marginal = jnp.exp(lexical_beta + log_scale[..., jnp.newaxis])
    next_beta = semirings.Log.plus(blank_beta,
                                   semirings.Log.sum(lexical_beta, axis=-1))
    return next_beta, [blank_marginal], [lexical_marginal]

  def string_forward(self, alpha: jnp.ndarray, blank: Sequence[jnp.ndarray],
                     lexical: Sequence[jnp.ndarray],
                     semiring: semirings.Semiring[jnp.ndarray]) -> jnp.ndarray:
    check_num_weights(self, blank, lexical)
    # alpha: [batch_dims..., output_length + 1]
    # blank: [batch_dims..., output_length + 1]
    # lexical: [batch_dims..., output_length + 1]
    return semiring.plus(
        semiring.times(alpha, blank[0]),
        shift_down(semiring.times(alpha, lexical[0]), semiring))


@flax.struct.dataclass
class FrameLabelDependent(TimeSyncAlignmentLattice):
  """k-constrained frame-label-dependent alignment lattice.

  Each frame is aligned to up to k lexical labels followed by a blank label.

  Attributes:
    max_expansions: The maximum number of lexical labels allowed per frame.
  """

  max_expansions: int

  def num_states(self) -> int:
    return self.max_expansions + 1

  def start(self) -> int:
    return 0

  def blank_next(self, state: int) -> Optional[int]:
    return 0

  def lexical_next(self, state: int) -> Optional[int]:
    next_state = state + 1
    if next_state <= self.max_expansions:
      return next_state
    else:
      return None

  def topological_visit(self) -> list[int]:
    return list(range(self.max_expansions + 1))

  def forward(self, alpha: jnp.ndarray, blank: Sequence[jnp.ndarray],
              lexical: Sequence[jnp.ndarray],
              context: contexts.ContextDependency,
              semiring: semirings.Semiring[jnp.ndarray]) -> jnp.ndarray:
    check_num_weights(self, blank, lexical)
    # alpha: [batch_dims..., num_context_states]
    # blank[i]: [batch_dims..., num_context_states]
    # lexical[i]: [batch_dims..., num_context_states, vocab_size]
    terminated = [semiring.times(alpha, blank[0])]
    last = alpha
    for i in range(self.max_expansions):
      last = context.forward_reduce(
          semiring.times(last[..., jnp.newaxis], lexical[i]), semiring)
      terminated.append(semiring.times(last, blank[i + 1]))
    return semiring.sum(jnp.stack(terminated), axis=0)

  def backward(
      self, alpha: jnp.ndarray, blank: Sequence[jnp.ndarray],
      lexical: Sequence[jnp.ndarray], beta: jnp.ndarray, log_z: jnp.ndarray,
      context: contexts.ContextDependency
  ) -> tuple[jnp.ndarray, list[jnp.ndarray], list[jnp.ndarray]]:
    check_num_weights(self, blank, lexical)
    # alpha: [batch_dims..., num_context_states]
    # blank[i]: [batch_dims..., num_context_states]
    # lexical[i]: [batch_dims..., num_context_states, vocab_size]
    # beta: [batch_dims..., num_context_states]
    # log_z: [batch_dims...]

    lexical_alphas = [alpha]
    last = alpha
    for i in range(self.max_expansions):
      last = context.forward_reduce(last[..., jnp.newaxis] + lexical[i],
                                    semirings.Log)
      lexical_alphas.append(last)

    # Corresponding backward paths.
    blank_marginals = []
    blank_log_scale = beta - log_z[..., jnp.newaxis]
    for i in range(self.max_expansions + 1):
      blank_marginals.append(
          jnp.exp(lexical_alphas[i] + blank[i] + blank_log_scale))

    next_beta = blank[self.max_expansions] + beta
    lexical_marginals = []
    for i in range(self.max_expansions):
      j = self.max_expansions - 1 - i
      lexical_beta = (lexical[j] + context.backward_broadcast(next_beta))
      log_scale = (lexical_alphas[j] - log_z[..., jnp.newaxis])
      lexical_marginals.append(
          jnp.exp(lexical_beta + log_scale[..., jnp.newaxis]))
      next_beta = semirings.Log.plus(blank[j] + beta,
                                     semirings.Log.sum(lexical_beta, axis=-1))
    lexical_marginals.reverse()
    lexical_marginals.append(jnp.zeros_like(lexical[self.max_expansions]))
    return next_beta, blank_marginals, lexical_marginals

  def string_forward(self, alpha: jnp.ndarray, blank: Sequence[jnp.ndarray],
                     lexical: Sequence[jnp.ndarray],
                     semiring: semirings.Semiring[jnp.ndarray]) -> jnp.ndarray:
    check_num_weights(self, blank, lexical)
    # alpha: [batch_dims..., output_length + 1]
    # blank[i]: [batch_dims..., output_length + 1]
    # lexical[i]: [batch_dims..., output_length + 1]
    terminated = [semiring.times(alpha, blank[0])]
    last = alpha
    for i in range(self.max_expansions):
      last = shift_down(semiring.times(last, lexical[i]), semiring)
      terminated.append(semiring.times(last, blank[i + 1]))
    return semiring.sum(jnp.stack(terminated), axis=0)

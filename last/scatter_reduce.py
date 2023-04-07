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

"""A custom JAX primitive for emitting XLA scatter ops with a custom reducer.

Temporarily here while https://github.com/google/jax/pull/12004 is under review.
"""

# pylint: skip-file
# pytype: skip-file
from jax._src.lax.slicing import *
from jax._src.lax.slicing import _argnum_weak_type, _scatter_dtype_rule, _scatter_lower, _scatter_shape_rule


def scatter_reduce(
    operand: Array,
    scatter_indices: Array,
    updates: Array,
    computation: Callable[[Array, Array], Array],
    dimension_numbers: ScatterDimensionNumbers,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: Optional[Union[str, GatherScatterMode]] = None) -> Array:
  """Scatter-reduce operator.

  Wraps `XLA's Scatter operator
  <https://www.tensorflow.org/xla/operation_semantics#scatter>`_, where the
  custom `computation` function is used to combine updates and values from
  `operand`.

  The semantics of scatter are complicated, and its API might change in the
  future. For most use cases, you should prefer the
  :attr:`jax.numpy.ndarray.at` property on JAX arrays which uses
  the familiar NumPy indexing syntax.

  Args:
    operand: an array to which the scatter should be applied
    scatter_indices: an array that gives the indices in `operand` to which each
      update in `updates` should be applied.
    updates: the updates that should be scattered onto `operand`.
    computation: the reduction function for computing a new value given a
      current value from `operand` and a corresponding update value from
      `updates`.
    dimension_numbers: a `lax.ScatterDimensionNumbers` object that describes how
      dimensions of `operand`, `start_indices`, `updates` and the output relate.
    indices_are_sorted: whether `scatter_indices` is known to be sorted. If
      true, may improve performance on some backends.
    unique_indices: whether the indices to be updated in ``operand`` are
      guaranteed to not overlap with each other. If true, may improve
      performance on some backends.
    mode: how to handle indices that are out of bounds: when set to 'clip',
      indices are clamped so that the slice is within bounds, and when set to
      'fill' or 'drop' out-of-bounds updates are dropped. The behavior for
      out-of-bounds indices when set to 'promise_in_bounds' is
      implementation-defined.

  Returns:
    An array containing the sum of `operand` and the scattered updates.
  """
  jaxpr, consts = lax._reduction_jaxpr(computation,
                                       lax._abstractify(lax._const(operand, 0)))
  return scatter_reduce_p.bind(
      operand,
      scatter_indices,
      updates,
      update_jaxpr=jaxpr,
      update_consts=consts,
      dimension_numbers=dimension_numbers,
      indices_are_sorted=indices_are_sorted,
      unique_indices=unique_indices,
      mode=GatherScatterMode.from_any(mode))


# TODO(wuke): Implement _scatter_reduce_jvp_rule.
def _scatter_reduce_batching_rule(batched_args, batch_dims, *, update_jaxpr,
                                  update_consts, dimension_numbers,
                                  indices_are_sorted, unique_indices, mode):

  def scatter_op(operand, scatter_indices, updates, dimension_numbers, *,
                 indices_are_sorted, unique_indices, mode):
    return scatter_reduce_p.bind(
        operand,
        scatter_indices,
        updates,
        update_jaxpr=update_jaxpr,
        update_consts=update_consts,
        dimension_numbers=dimension_numbers,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
        mode=GatherScatterMode.from_any(mode))

  return _scatter_batching_rule(
      scatter_op,
      batched_args,
      batch_dims,
      update_jaxpr=update_jaxpr,
      update_consts=update_consts,
      dimension_numbers=dimension_numbers,
      indices_are_sorted=indices_are_sorted,
      unique_indices=unique_indices,
      mode=mode)


scatter_reduce_p = standard_primitive(
    _scatter_shape_rule,
    _scatter_dtype_rule,
    "last-scatter-reduce",  # Prepend "last" to avoid future name collision.
    weak_type_rule=_argnum_weak_type(0))
batching.primitive_batchers[scatter_reduce_p] = _scatter_reduce_batching_rule

mlir.register_lowering(scatter_reduce_p, _scatter_lower)

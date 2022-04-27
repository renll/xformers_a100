# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional

import torch
import triton
import triton.language as tl
from triton.ops.matmul import get_configs_io_bound
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time

from xformers.triton.sum_strided import sum_2d_dim_0


# fmt: off
@triton.heuristics({
    'EVEN_N': lambda args: args["N"] % (args['BLOCK_N']) == 0,
})
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 32}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_N": 64}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_N": 128}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_N": 256}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_N": 512}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_N": 1024}, num_stages=3, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def kernel_bw_act(
    # Pointers to matrices
    GRAD_ACT, GRAD_OUT, ACT_INPUTS,
    # Matrix dimensions
    N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_gom, stride_aim,
    # Meta-parameters
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
    ACTIVATION_GRAD: tl.constexpr,
):
    # fmt: on

    """
    Go over all the activation inputs, compute the corresponding gradient
    """

    # this kernel is relatively simple in terms of scheduling:
    # - per row (pid_m)
    # - each program a given chunk on the col axis,
    # since it's more effective memory and occupancy wise
    pid_m, pid_n = tl.program_id(axis=0), tl.program_id(axis=1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # the memory addresses of elements in the first block of
    # A and W can be computed using numpy-style broadcasting
    act_input_ptrs = ACT_INPUTS + pid_m * stride_aim + rn

    # compute the gradient which is related to this activation
    if EVEN_N:
        act_in = tl.load(act_input_ptrs)
    else:
        act_in = tl.load(act_input_ptrs, mask=rn < N, other=0.0)

    grad_act = ACTIVATION_GRAD(act_in)

    # now read the incoming gradient, the backpropagated one is the multiple of both
    grad_out_ptrs = GRAD_OUT + pid_m * stride_gom + rn
    if EVEN_N:
        grad_out = tl.load(grad_out_ptrs)
    else:
        grad_out = tl.load(grad_out_ptrs, mask=rn < N)

    grad_act *= grad_out

    # write back result
    grad_act_ptrs = GRAD_ACT + pid_m * stride_gom + rn
    tl.store(grad_act_ptrs, grad_act, mask=rn < N)


# fmt: off
@triton.heuristics({
    'EVEN_BLOCKS': lambda args:
        args["K"] % (args['BLOCK_K']) == 0
        and args["M"] % (args['BLOCK_M']) == 0
        and args["N"] % (args['BLOCK_N']) == 0,
})
@triton.autotune(
    configs=[
        # basic configs for compute-bound matmuls
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=5, num_warps=2),
        # good for int8
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=5, num_warps=2),
    ] + get_configs_io_bound(),
    key=["M", "N", "K"],
    prune_configs_by={
        'early_config_prune': early_config_prune,
        'perf_model': estimate_matmul_time,
        'top_k': 10
    },
)
@triton.jit
def kernel_matmul_transpose(
    C, A, B,
    M, N, K,
    stride_on, stride_am, stride_bm,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr, GROUP_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_BLOCKS: tl.constexpr,
    SPLIT_K: tl.constexpr  # not being used, here for config compatibility
):
    # fmt: on

    """
    Kernel for computing Out = A^T x B

    - A has shape (M, N)
    - B has shape (M, K)
    - Out has shape (N, K)

    This kernel will consolidate over M
    """

    # programs are grouped together to improve L2 hit rate
    # the logic is that we'll consolidate over K. If the programs were not grouped,
    # then multiple cols/rows in the result would end up pulling in the same row and lines
    # from the inputs. By grouping the computation we ensure some data reuse, which the hardware
    # covers via the L2 cache
    pid = tl.program_id(axis=0)

    num_pid_n = tl.cdiv(N, BLOCK_N)  # number of program ids along the M axis
    num_pid_k = tl.cdiv(K, BLOCK_K)  # number of programs ids along the N axis
    num_pid_in_group = GROUP_N * num_pid_k  # number of programs in group
    group_id = pid // num_pid_in_group  # id of the group this program is in
    first_pid_n = group_id * GROUP_N  # row-id of the first program in the group
    GROUP_N = min(
        num_pid_n - first_pid_n, GROUP_N
    )

    # *within groups*, programs are ordered in a column-major order
    # row-id /col-id of the program in the *launch grid*
    pid_n = first_pid_n + (pid % GROUP_N)
    pid_k = (pid % num_pid_in_group) // GROUP_N

    # now compute the block that each program will go through
    # rm (resp. rn) denotes a range of indices
    # for rows (resp. col) of C
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    rm = tl.arange(0, BLOCK_M)

    # the memory addresses of elements can follow numpy broadcasting
    a_ptrs = A + rn[:, None]    # we transpose on the fly
    b_ptrs = B + rk[None, :]

    # initialize and iteratively update accumulator
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

    # block level matrix multiplication.
    # We fetch a block memory block from both inputs, matmul and accumulate, then repeat
    if EVEN_BLOCKS:
        other = None
        mask_rn = None
        mask_rk = None
    else:
        other = 0.0
        mask_rn = rn < N
        mask_rk = rk < K

    for i in range(0, M, BLOCK_M):
        rm = tl.arange(0, BLOCK_M) + i

        if EVEN_BLOCKS:
            mask_a = None
            mask_b = None
        else:
            mask_a = ((rm[None, :] < M) & mask_rn[:, None])         # type: ignore
            mask_b = ((mask_rk[None, :] < K) & rm[:, None] < M)     # type: ignore

        a = tl.load(a_ptrs + rm[None, :] * stride_am, mask=mask_a, other=other)
        b = tl.load(b_ptrs + rm[:, None] * stride_bm, mask=mask_b, other=other)

        acc += tl.dot(a, b)

    # write back result
    out_ptrs = C + rn[:, None] * stride_on + rk[None, :]
    if EVEN_BLOCKS:
        mask_out = None
    else:
        mask_out = mask_rn[:, None] & mask_rk[None, :]      # type: ignore
    tl.store(out_ptrs, acc, mask=mask_out)


def fused_matmul_backward(
    grad_out: torch.Tensor,
    inputs: torch.Tensor,
    act_in: Optional[torch.Tensor],
    weight: torch.Tensor,
    trainable_weight: bool,
    trainable_bias: bool,
    activation_grad=None,
):
    """
    Compute grad_in = activation^-1(grad_out) @ weight.transpose()

    .. note: The weight buffer is transposed on the fly
    .. note: Activation gradient needs to be a Triton kernel
    """

    # Make sure that we don't have to handle the stride over cols
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()

    grad_out_ = grad_out if grad_out.ndim == 2 else grad_out.flatten(0, 1)

    assert grad_out_.shape[1] == weight.shape[0], "Incompatible dimensions in between grad_out and weight"

    M, N = grad_out_.shape
    N, K = weight.shape

    # Compute the gradient for the activation
    if activation_grad is not None:
        grad_act = torch.empty_like(grad_out_)

        # Some activations do not require their inputs to
        # know of their grad, the downstream grad is enough
        if act_in is None:
            act_in = grad_out_

        grid = lambda META: (M, triton.cdiv(N, META["BLOCK_N"])) # noqa

        # fmt: off
        kernel_bw_act[grid](
            grad_act, grad_out_, act_in,            # data ptrs
            N,                                      # shapes
            grad_act.stride(0), act_in.stride(0),   # strides
            ACTIVATION_GRAD=activation_grad,        # optional fused activation
        )
        # fmt: on

        # Backpropagation going up, the reference gradient is now
        # just before the activation
        grad_out_ = grad_act

    # Compute the gradient for the weight
    if trainable_weight:
        inputs_ = inputs if inputs.ndim == 2 else inputs.flatten(0, 1)

        if True:
            grid_ = lambda META: (triton.cdiv(N, META["BLOCK_N"]) * triton.cdiv(K, META["BLOCK_K"]),) # noqa

            grad_weight = torch.empty_like(weight)

            # fmt: off
            kernel_matmul_transpose[grid_](
                grad_weight, grad_out_, inputs_,        # data ptrs
                M, N, K,                                # shapes
                grad_weight.stride(0),
                grad_out_.stride(0),
                inputs_.stride(0),
                GROUP_N=8,
            )
            # fmt: on
        else:
            grad_weight = grad_out_.transpose(0, 1) @ inputs_

    # Epilogue, could probably be better handled
    grad_in = grad_out_ @ weight
    grad_bias = sum_2d_dim_0(grad_out_) if trainable_bias else None

    return grad_in.reshape_as(inputs), grad_weight if trainable_weight else None, grad_bias

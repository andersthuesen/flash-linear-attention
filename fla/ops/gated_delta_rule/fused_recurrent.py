# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang


import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp
from fla.utils import input_guard


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_GV": lambda args: args["gv"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        # new: store per-timestep hidden states if hs is provided
        "STORE_STATES": lambda args: args["hs"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    g,
    gk,
    gv,
    beta,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    T,
    hs,  # [B, T, HV, K, V] (or packed for varlen)
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    STORE_STATES: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T_local = eos - bos
    else:
        bos = i_n * T
        eos = i_n * T + T
        T_local = T

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if USE_G:
        p_g = g + bos * HV + i_hv
    if USE_GK:
        p_gk = gk + (bos * HV + i_hv) * K + o_k
    if USE_GV:
        p_gv = gv + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + bos * HV + i_hv
    else:
        p_beta = beta + (bos * HV + i_hv) * V + o_v

    p_o = o + (bos * HV + i_hv) * V + o_v

    if STORE_STATES:
        # hs layout matches [B, T, HV, K, V] / flattened
        p_hs = hs + ((bos * HV + i_hv) * K + o_k[:, None]) * V + o_v[None, :]

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    # time loop
    for _ in range(0, T_local):
        # store pre-gate H_t if requested
        if STORE_STATES:
            tl.store(p_hs, b_h.to(tl.float32), mask=mask_h)
            # advance hs pointer in time
            p_hs += HV * K * V

        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta).to(tl.float32)
        else:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)

        # [BK, BV]
        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            b_h *= exp(b_g)

        if USE_GK:
            b_gk = tl.load(p_gk, mask=mask_k, other=0).to(tl.float32)
            b_h *= exp(b_gk[:, None])

        if USE_GV:
            b_gv = tl.load(p_gv, mask=mask_v, other=0).to(tl.float32)
            b_h *= exp(b_gv[None, :])

        # delta rule
        b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
        b_h += b_k[:, None] * b_v

        # [BV]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # advance to next time step
        p_q += H * K
        p_k += H * K
        p_v += HV * V
        if USE_G:
            p_g += HV
        if USE_GK:
            p_gk += HV * K
        if USE_GV:
            p_gv += HV * V
        p_beta += HV * (1 if IS_BETA_HEADWISE else V)
        p_o += HV * V

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def fused_recurrent_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = triton.next_power_of_2(K)
    BV = min(8, triton.next_power_of_2(V)) if gv is None else triton.next_power_of_2(V)
    NV = triton.cdiv(V, BV)

    o = torch.empty_like(v)
    final_state = q.new_empty(N, HV, K, V, dtype=torch.float32) if output_final_state else None

    # store all hidden states (float32) for backward
    hs = q.new_empty(B, T, HV, K, V, dtype=torch.float32)

    grid = (NV, N * HV)
    fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        gk=gk,
        gv=gv,
        beta=beta,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        hs=hs,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        IS_BETA_HEADWISE=beta.ndim != v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        num_warps=1,
        num_stages=3,
    )
    return o, final_state, hs


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_GV": lambda args: args["gv"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def fused_recurrent_gated_delta_rule_bwd_kernel(
    q,
    k,
    v,
    g,
    gk,
    gv,
    beta,
    h0,
    ht,
    cu_seqlens,
    scale,
    T,
    hs,  # stored pre-gate H_t: [B, T, HV, K, V] (float32)
    do,  # grad output: [B, T, HV, V]
    dht,  # grad final state: [N, HV, K, V] or None
    dq,
    dk,
    dv,
    dg,
    dgk,
    dgv,
    dbeta,
    dh0,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T_local = eos - bos
    else:
        bos = i_n * T
        eos = i_n * T + T
        T_local = T

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    # pointers for last time step (t = T_local - 1)
    last_t = T_local - 1
    t_idx = bos + last_t

    p_q = q + (t_idx * H + i_h) * K + o_k
    p_k = k + (t_idx * H + i_h) * K + o_k
    p_v = v + (t_idx * HV + i_hv) * V + o_v
    if USE_G:
        p_g = g + t_idx * HV + i_hv
    if USE_GK:
        p_gk = gk + (t_idx * HV + i_hv) * K + o_k
    if USE_GV:
        p_gv = gv + (t_idx * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + t_idx * HV + i_hv
    else:
        p_beta = beta + (t_idx * HV + i_hv) * V + o_v

    p_do = do + (t_idx * HV + i_hv) * V + o_v

    # hs pointer (pre-gate H_t)
    p_hs = hs + ((t_idx * HV + i_hv) * K + o_k[:, None]) * V + o_v[None, :]

    # grad pointers (start at last t too)
    p_dq = dq + (t_idx * H + i_h) * K + o_k
    p_dk = dk + (t_idx * H + i_h) * K + o_k
    p_dv = dv + (t_idx * HV + i_hv) * V + o_v
    if USE_G:
        p_dg = dg + t_idx * HV + i_hv
    if USE_GK:
        p_dgk = dgk + (t_idx * HV + i_hv) * K + o_k
    if USE_GV:
        p_dgv = dgv + (t_idx * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_dbeta = dbeta + t_idx * HV + i_hv
    else:
        p_dbeta = dbeta + (t_idx * HV + i_hv) * V + o_v

    # dH_{t+1} accumulator (float32)
    b_dh_next = tl.zeros([BK, BV], dtype=tl.float32)
    if STORE_FINAL_STATE and dht is not None:
        p_dht = dht + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        b_dh_next += tl.load(p_dht, mask=mask_h, other=0).to(tl.float32)

    # reverse time loop
    for _ in range(0, T_local):
        # load H_t (pre-gate)
        b_h_t = tl.load(p_hs, mask=mask_h, other=0).to(tl.float32)

        # load q,k,v,beta,g,gk,gv for this time step
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta).to(tl.float32)  # scalar
        else:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)

        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
        if USE_GK:
            b_gk = tl.load(p_gk, mask=mask_k, other=0).to(tl.float32)
        if USE_GV:
            b_gv = tl.load(p_gv, mask=mask_v, other=0).to(tl.float32)

        # re-build gated H_tilde and intermediates
        b_h1 = b_h_t
        if USE_G:
            e_g = exp(b_g)
            b_h1 = b_h1 * e_g
        if USE_GK:
            e_gk = exp(b_gk)
            b_h2 = b_h1 * e_gk[:, None]
        else:
            b_h2 = b_h1
        if USE_GV:
            e_gv = exp(b_gv)
            b_h3 = b_h2 * e_gv[None, :]
        else:
            b_h3 = b_h2
        b_h_tilde = b_h3  # [BK, BV]

        # v_hat = H_tilde^T k
        b_v_hat = tl.sum(b_h_tilde * b_k[:, None], 0)  # [BV]
        b_e = b_v - b_v_hat  # [BV]

        # dv = beta * e
        if IS_BETA_HEADWISE:
            b_dv = b_beta * b_e  # [BV]
        else:
            b_dv = b_beta * b_e  # [BV]

        # H_next = H_tilde + k * dv^T
        b_h_next = b_h_tilde + b_k[:, None] * b_dv[None, :]

        # --- gradients ---

        # contribution from o_t = H_next^T (q_t * scale)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)  # [BV]

        # q_scaled = q * scale   (this is what forward actually used)
        b_q_scaled = b_q * scale

        # dH_next_local = q_scaled * do^T
        b_dh_local = b_q_scaled[:, None] * b_do[None, :]  # [BK, BV]
        b_dh_next += b_dh_local

        # dq = scale * (H_next @ do)
        b_dq = scale * tl.sum(b_h_next * b_do[None, :], 1)  # [BK]

        # H_next = H_tilde + k * dv^T
        # dH_tilde starts from dH_next
        b_dh_tilde = b_dh_next

        # dk from update term
        # dk_i += sum_j dH_next[i,j] * dv[j]
        b_dk = tl.sum(b_dh_next * b_dv[None, :], 1)  # [BK]

        # ddv_j += sum_i dH_next[i,j] * k[i]
        b_ddv = tl.sum(b_dh_next * b_k[:, None], 0)  # [BV]

        # dv = beta * e
        if IS_BETA_HEADWISE:
            # scalar beta per head
            b_dbeta = tl.sum(b_ddv * b_e)  # scalar
            b_de = b_ddv * b_beta  # [BV]
        else:
            b_dbeta_vec = b_ddv * b_e  # [BV]
            b_dbeta = b_dbeta_vec
            b_de = b_ddv * b_beta  # [BV]

        # e = v - v_hat
        b_dv_input = b_de  # [BV]
        b_dvhat = -b_de  # [BV]

        # v_hat = H_tilde^T k
        # dH_tilde += k * dvhat^T
        b_dh_tilde += b_k[:, None] * b_dvhat[None, :]
        # dk extra from v_hat
        b_dk += tl.sum(b_h_tilde * b_dvhat[None, :], 1)

        # now back through gating H_tilde = H_t * gates

        # start from dH3 = b_dh_tilde
        b_dh3 = b_dh_tilde

        # GV
        if USE_GV:
            # dgv_j += sum_i dH3[i,j] * H3[i,j]
            b_dgv = tl.sum(b_dh3 * b_h3, 0)  # [BV]
            # dH2 = dH3 * exp(gv)
            b_dh2 = b_dh3 * e_gv[None, :]
        else:
            b_dgv = tl.zeros([BV], dtype=tl.float32)
            b_dh2 = b_dh3

        # GK
        if USE_GK:
            # dgk_i += sum_j dH2[i,j] * H2[i,j]
            b_dgk = tl.sum(b_dh2 * b_h2, 1)  # [BK]
            # dH1 = dH2 * exp(gk)
            b_dh1 = b_dh2 * e_gk[:, None]
        else:
            b_dgk = tl.zeros([BK], dtype=tl.float32)
            b_dh1 = b_dh2

        # G (scalar)
        if USE_G:
            # dg += sum_{i,j} dH1[i,j] * H1[i,j]
            b_dg = tl.sum(b_dh1 * b_h1)
            # dH_t = dH1 * exp(g)
            b_dh_t = b_dh1 * e_g
        else:
            b_dg = tl.zeros((), dtype=tl.float32)
            b_dh_t = b_dh1

        # write grads (using atomics where needed across V-tiles)

        # dq, dk: need atomic add across NV tiles
        tl.atomic_add(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=mask_k)
        tl.atomic_add(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask_k)

        # dv: disjoint across tiles
        tl.store(p_dv, b_dv_input.to(p_dv.dtype.element_ty), mask=mask_v)

        # g: scalar per (t, head) shared across tiles
        if USE_G:
            tl.atomic_add(p_dg, b_dg.to(p_dg.dtype.element_ty))

        # gk: per-K shared across tiles
        if USE_GK:
            tl.atomic_add(p_dgk, b_dgk.to(p_dgk.dtype.element_ty), mask=mask_k)

        # gv: per-V, disjoint across tiles
        if USE_GV:
            tl.store(p_dgv, b_dgv.to(p_dgv.dtype.element_ty), mask=mask_v)

        # beta
        if IS_BETA_HEADWISE:
            tl.atomic_add(p_dbeta, b_dbeta.to(p_dbeta.dtype.element_ty))
        else:
            tl.store(p_dbeta, b_dbeta.to(p_dbeta.dtype.element_ty), mask=mask_v)

        # prepare dH_next = dH_t for previous time step
        b_dh_next = b_dh_t

        # move pointers one step back in time
        p_q -= H * K
        p_k -= H * K
        p_v -= HV * V
        p_do -= HV * V
        p_hs -= HV * K * V
        p_dq -= H * K
        p_dk -= H * K
        p_dv -= HV * V
        if USE_G:
            p_g -= HV
            p_dg -= HV
        if USE_GK:
            p_gk -= HV * K
            p_dgk -= HV * K
        if USE_GV:
            p_gv -= HV * V
            p_dgv -= HV * V
        step_beta = HV * (1 if IS_BETA_HEADWISE else V)
        p_beta -= step_beta
        p_dbeta -= step_beta

    # gradient wrt initial state h0 comes from dH_0
    if USE_INITIAL_STATE and dh0 is not None:
        p_dh0 = dh0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_dh0, b_dh_next.to(p_dh0.dtype.element_ty), mask=mask_h)


class FusedRecurrentFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None = None,
        gk: torch.Tensor | None = None,
        gv: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        scale: float = None,
        initial_state: torch.Tensor = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
    ):
        if use_qk_l2norm_in_kernel:
            raise NotImplementedError("Backward with USE_QK_L2NORM_IN_KERNEL=True is not implemented.")

        o, final_state, hs = fused_recurrent_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            gk=gk,
            gv=gv,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
        )

        # save tensors for backward
        ctx.save_for_backward(q, k, v, g, gk, gv, beta, initial_state, final_state, hs, cu_seqlens)
        ctx.scale = scale
        ctx.output_final_state = output_final_state
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel

        return o, final_state

    @staticmethod
    @input_guard
    def backward(ctx, do, dht):
        q, k, v, g, gk, gv, beta, h0, ht, hs, cu_seqlens = ctx.saved_tensors
        scale = ctx.scale
        output_final_state = ctx.output_final_state
        use_qk_l2norm_in_kernel = ctx.use_qk_l2norm_in_kernel

        B, T, H, K = k.shape
        HV = v.shape[2]
        V = v.shape[-1]
        N = B if cu_seqlens is None else len(cu_seqlens) - 1

        BK = triton.next_power_of_2(K)
        BV = min(8, triton.next_power_of_2(V)) if gv is None else triton.next_power_of_2(V)
        NV = triton.cdiv(V, BV)

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        dg = torch.zeros_like(g) if g is not None else None
        dgk = torch.zeros_like(gk) if gk is not None else None
        dgv = torch.zeros_like(gv) if gv is not None else None
        dbeta = torch.zeros_like(beta)
        dh0 = torch.zeros_like(h0) if h0 is not None else None

        grid = (NV, N * HV)
        fused_recurrent_gated_delta_rule_bwd_kernel[grid](
            q=q,
            k=k,
            v=v,
            g=g,
            gk=gk,
            gv=gv,
            beta=beta,
            h0=h0,
            ht=ht,
            cu_seqlens=cu_seqlens,
            scale=scale,
            T=T,
            hs=hs,
            do=do,
            dht=dht if output_final_state else None,
            dq=dq,
            dk=dk,
            dv=dv,
            dg=dg,
            dgk=dgk,
            dgv=dgv,
            dbeta=dbeta,
            dh0=dh0,
            B=B,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BK=BK,
            BV=BV,
            IS_BETA_HEADWISE=beta.ndim != v.ndim,
            num_warps=1,
            num_stages=3,
        )

        # gradients in same order as forward args
        dscale = None  # not computing grad for scale
        # outputs_final_state & use_qk_l2norm & cu_seqlens have no grads
        return dq, dk, dv, dg, dgk, dgv, dbeta, dscale, dh0, None, None, None


def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, HV, V]`.
            GVA is applied if `HV > H`.
        g (torch.Tensor):
            g (decays) of shape `[B, T, HV]`. Default: `None`.
        gk (torch.Tensor):
            gk (decays) of shape `[B, T, HV, K]`. Default: `None`.
        gv (torch.Tensor):
            gv (decays) of shape `[B, T, HV, V]`. Default: `None`.
        beta (torch.Tensor):
            betas of shape `[B, T, HV]`.
        scale (Optional[float]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, HV, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, HV, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (Optional[bool]):
            Whether to use L2 normalization in the kernel. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HV, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, HV, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, HV, K, V = 4, 2048, 4, 8, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, HV, V, device='cuda')
        >>> g = F.logsigmoid(torch.rand(B, T, HV, device='cuda'))
        >>> beta = torch.rand(B, T, HV, device='cuda').sigmoid()
        >>> h0 = torch.randn(B, HV, K, V, device='cuda')
        >>> o, ht = fused_gated_recurrent_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, g, beta))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o, ht = fused_gated_recurrent_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if beta is None:
        beta = torch.ones_like(q[..., 0])

    o, final_state = FusedRecurrentFunction.apply(
        q,
        k,
        v,
        g,
        gk,
        gv,
        beta,
        scale,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        cu_seqlens,
    )
    return o, final_state

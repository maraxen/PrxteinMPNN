"""Stacked wave-parallel autoregressive sampler for ProteinMPNN (exact strategy).

Implemented via jax.vmap over states with full attention.

Extracted from :mod:`prxteinmpnn.model.mpnn` (Phase 5e-cont) to shrink ``mpnn.py`` without
changing JIT boundaries or public method names on the model class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from prxteinmpnn.model._shared import combine_logits_multistate_idx
from prxteinmpnn.model.encoder import encoder_forward_with_int_neighbors, pack_encoder_context
from prxteinmpnn.utils.concatenate import concatenate_neighbor_nodes
from prxteinmpnn.utils.ste import straight_through_estimator

if TYPE_CHECKING:
  from prxteinmpnn.model.mpnn import PrxteinMPNN  # prxteinmpnn-internal-import
  from prxteinmpnn.model_inputs import ARLogitTransformFn
  from prxteinmpnn.utils.types import (
    Float,
    Int,
    Logits,
    OneHotProteinSequence,
    PRNGKeyArray,
  )


def run_sample_ar_exact(  # noqa: PLR0915
  model: PrxteinMPNN,
  prng_key: PRNGKeyArray,
  coords_stack: jax.Array,
  mask_stack: jax.Array,
  residue_index_stack: jax.Array,
  chain_index_stack: jax.Array,
  autoregressive_mask_stack: jax.Array,
  tie_group_map_stack: jax.Array | None,
  bias_stack: jax.Array,
  temperature: Float | float,
  multi_state_strategy_idx: Int,
  state_weights: jnp.ndarray | None,
  fixed_mask_stack: jax.Array,
  fixed_tokens_stack: jax.Array,
  wave_group_ids_local: jax.Array,
  wave_group_positions_local: jax.Array,
  wave_group_valid_local: jax.Array,
  wave_position_valid_local: jax.Array,
  ar_logit_transform_fn: "ARLogitTransformFn | None" = None,
) -> tuple[OneHotProteinSequence, Logits]:
  """Stacked-graph wave-parallel sampler for ProteinMPNN (exact strategy).

  Outputs one-hot sequences and logits with shape ``(n_states, n_pad, ·)``.
  """
  del tie_group_map_stack
  pk_cur, fk = jax.random.split(prng_key)
  temp = jnp.asarray(temperature, dtype=jnp.float32)
  feat_keys = jax.random.split(fk, coords_stack.shape[0])

  def encode_one(coords, ma, ri, ci, k):  # noqa: ANN001, ANN202
    ef, nei, nf, _ = model.features(
      k,
      coords,
      ma,
      ri,
      ci,
      jnp.asarray(0.0, jnp.float32),
      structure_mapping=None,
      initial_node_features=None,
      rbf_features=None,
      neighbor_indices=None,
    )
    return encoder_forward_with_int_neighbors(
      model.encoder,
      ef,
      nei,
      ma,
      nf,
      inference=True,
      key=k,
    )

  node_b, edge_b, nei_b = jax.vmap(encode_one)(
    coords_stack,
    mask_stack,
    residue_index_stack,
    chain_index_stack,
    feat_keys,
  )

  attn_b = jax.vmap(
    lambda am, nei_i: jnp.take_along_axis(am, nei_i.astype(jnp.int32), axis=1),
  )(autoregressive_mask_stack, nei_b)
  mask_bw = mask_stack[:, :, jnp.newaxis] * attn_b
  mask_fw = mask_stack[:, :, jnp.newaxis] * (jnp.float32(1.0) - jnp.asarray(attn_b, jnp.float32))
  encoder_stack = jax.vmap(pack_encoder_context)(node_b, edge_b, nei_b, mask_fw)

  log_dtype = model.w_out.weight.dtype
  num_dec = len(model.decoder.layers) + 1
  d_hid = node_b.shape[-1]
  s_dim, p_pad = int(coords_stack.shape[0]), int(coords_stack.shape[1])

  ah = jnp.zeros((s_dim, num_dec, p_pad, d_hid), dtype=log_dtype)
  ah = ah.at[:, 0, :, :].set(node_b.astype(log_dtype))

  fixed_b = fixed_mask_stack.astype(jnp.bool_)
  fixed_toks = fixed_tokens_stack.astype(jnp.int32)
  oh_fix = jax.nn.one_hot(fixed_toks, model.w_s_embed.num_embeddings, dtype=log_dtype)
  seq_z = jnp.zeros((s_dim, p_pad, model.w_s_embed.num_embeddings), dtype=log_dtype)
  seq_carry = jnp.where(fixed_b[..., jnp.newaxis], oh_fix, seq_z)
  emb_w = model.w_s_embed.weight.astype(log_dtype)
  se = seq_carry @ emb_w

  logits_acc = jnp.zeros((s_dim, p_pad, model.w_out.out_features), dtype=log_dtype)

  nw_i = jnp.int32(wave_group_valid_local.shape[0])
  nslot_i = jnp.int32(wave_group_valid_local.shape[1])
  max_gs_tr = jnp.int32(wave_position_valid_local.shape[-1])
  max_gs = int(wave_position_valid_local.shape[-1])
  strat_idx = jnp.asarray(multi_state_strategy_idx, dtype=jnp.int32)
  ms_temp = jnp.asarray(1.0, dtype=jnp.float32)
  if state_weights is None:
    sw_use = jnp.ones((s_dim,), dtype=jnp.float32) / jnp.float32(max(s_dim, 1))
  else:
    sw_use = jnp.asarray(state_weights, dtype=jnp.float32)
  row_state_map = jnp.arange(max_gs, dtype=jnp.int32)

  def decode_site(
    ah_s: jax.Array,
    se_s: jax.Array,
    lid_s: jax.Array,
    dk: jax.Array,
    *,
    eb: jax.Array,
    nei: jax.Array,
    mw_atom: jax.Array,
    ecx: jax.Array,
    bw: jax.Array,
  ) -> tuple[jax.Array, jax.Array]:
    ecx_row = jax.lax.dynamic_index_in_dim(ecx, lid_s, axis=0).squeeze(0)
    nei_row = jax.lax.dynamic_index_in_dim(nei, lid_s, axis=0).squeeze(0)
    mask_pos = jax.lax.dynamic_index_in_dim(mw_atom, lid_s, axis=0).squeeze(0)
    bw_row = jax.lax.dynamic_index_in_dim(bw, lid_s, axis=0).squeeze(0)
    edge_slice = jax.lax.dynamic_index_in_dim(eb, lid_s, axis=0).squeeze(0)
    seq_nf = concatenate_neighbor_nodes(se_s, edge_slice, nei_row)
    sk_l = jax.random.split(dk, len(model.decoder.layers))
    ah_cur = ah_s
    for lyr_i, layer in enumerate(model.decoder.layers):
      h_in = jax.lax.dynamic_index_in_dim(ah_cur[lyr_i], lid_s, axis=0).squeeze(0)
      dctx = concatenate_neighbor_nodes(ah_cur[lyr_i], seq_nf, nei_row)
      dcomb = bw_row[..., None] * dctx + ecx_row
      h_out = layer(
        jnp.expand_dims(h_in, 0),
        jnp.expand_dims(dcomb, 0),
        mask=mask_pos,
        key=sk_l[lyr_i],
        inference=False,
      )
      ah_cur = ah_cur.at[lyr_i + 1, lid_s].set(jnp.squeeze(h_out, axis=0))
    lg = model.w_out(jax.lax.dynamic_index_in_dim(ah_cur[-1], lid_s, axis=0).squeeze(0))
    return ah_cur, lg

  def outer_wave(w_ix: jax.Array, cw: tuple) -> tuple:  # noqa: PLR0915
    wi_here = jnp.int32(w_ix)
    pk_w, ah_w, se_w, seq_w, log_w = cw

    def inner_slot(mi_xx: jax.Array, cms: tuple) -> tuple:  # noqa: PLR0915
      mi_here = jnp.int32(mi_xx)
      pk_mi, ah_mi, se_mi, seq_mi, logits_mi = cms

      lane_wgv = jax.lax.dynamic_index_in_dim(wave_group_valid_local, wi_here, axis=0).squeeze(0)
      wgv_here = jax.lax.dynamic_index_in_dim(lane_wgv, mi_here, axis=0).squeeze()
      lane_gid = jax.lax.dynamic_index_in_dim(wave_group_ids_local, wi_here, axis=0).squeeze(0)
      gid_here = jax.lax.dynamic_index_in_dim(lane_gid, mi_here, axis=0).squeeze()
      wl_plane = jax.lax.dynamic_index_in_dim(
        wave_group_positions_local,
        wi_here,
        axis=0,
      ).squeeze(0)
      wl_lane = jax.lax.dynamic_index_in_dim(wl_plane, mi_here, axis=0).squeeze(0)
      pv_plane = jax.lax.dynamic_index_in_dim(wave_position_valid_local, wi_here, axis=0).squeeze(
        0,
      )
      pv_lane = jax.lax.dynamic_index_in_dim(pv_plane, mi_here, axis=0).squeeze(0)
      active_slot = jnp.logical_and(wgv_here.astype(jnp.bool_), gid_here >= jnp.int32(0))
      has_member = jnp.any(pv_lane.astype(jnp.bool_))
      first_g = jnp.argmax(pv_lane.astype(jnp.int32))
      lid_pick = jax.lax.dynamic_index_in_dim(wl_lane, first_g).astype(jnp.int32)
      cand_lid = jnp.reshape(lid_pick, ())
      lid_here = jax.lax.select(
        jnp.logical_and(active_slot.squeeze(), has_member.squeeze()),
        cand_lid,
        jnp.int32(0),
      )

      def skip_slot(__: None) -> tuple:
        del __
        return pk_mi, ah_mi, se_mi, seq_mi, logits_mi

      def do_slot(__: None) -> tuple:  # noqa: PLR0915
        del __
        logits_rows_init = jnp.zeros((max_gs, model.w_out.out_features), dtype=log_dtype)
        cmask_init = jnp.zeros((max_gs,), dtype=jnp.bool_)

        def body_g(g: jax.Array, carry_g: tuple) -> tuple:
          pk_g, ah_g, se_g, lrows_g, cg_g = carry_g
          g_idx = jnp.int32(g)
          pv_g = jax.lax.dynamic_index_in_dim(pv_lane, g_idx, axis=0).squeeze(axis=0)
          si = g_idx
          mask_here = jax.lax.dynamic_index_in_dim(
            jax.lax.dynamic_index_in_dim(mask_stack.astype(jnp.float32), si, axis=0).squeeze(
              axis=0,
            ),
            lid_here,
            axis=0,
          ).squeeze()
          go_decode = jnp.logical_and(
            jnp.logical_and(pv_g.astype(jnp.bool_), active_slot.squeeze()),
            mask_here > jnp.float32(0.0),
          )

          def do_dec(___: None) -> tuple:
            del ___
            pk_dec, dk = jax.random.split(pk_g)
            ah_si = jax.lax.dynamic_index_in_dim(ah_g, si, axis=0).squeeze(axis=0)
            se_si = jax.lax.dynamic_index_in_dim(se_g, si, axis=0).squeeze(axis=0)
            eb_si = jax.lax.dynamic_index_in_dim(edge_b, si, axis=0).squeeze(axis=0)
            nei_si = jax.lax.dynamic_index_in_dim(nei_b, si, axis=0).squeeze(axis=0)
            mw_si = jax.lax.dynamic_index_in_dim(
              mask_stack.astype(jnp.float32),
              si,
              axis=0,
            ).squeeze(axis=0)
            ecx_si = jax.lax.dynamic_index_in_dim(encoder_stack, si, axis=0).squeeze(axis=0)
            bw_si = jax.lax.dynamic_index_in_dim(mask_bw, si, axis=0).squeeze(axis=0)
            ah_new, logits_row = decode_site(
              ah_si,
              se_si,
              lid_here,
              dk,
              eb=eb_si,
              nei=nei_si,
              mw_atom=mw_si,
              ecx=ecx_si,
              bw=bw_si,
            )
            ah_upd = ah_g.at[si].set(ah_new)
            cg_upd = cg_g.at[g_idx].set(True)
            lr_upd = lrows_g.at[g_idx].set(logits_row.astype(log_dtype))
            return pk_dec, ah_upd, se_g, lr_upd, cg_upd

          return jax.lax.cond(
            go_decode.squeeze(),
            do_dec,
            lambda _: (pk_g, ah_g, se_g, lrows_g, cg_g),
            None,
          )

        pk_in, ah_in = pk_mi, ah_mi
        pk_step, ah_step, _, lrows_fin, cmask_fin = jax.lax.fori_loop(
          jnp.int32(0),
          max_gs_tr,
          body_g,
          (pk_in, ah_in, se_mi, logits_rows_init, cmask_init),
        )
        any_cmem = jnp.any(cmask_fin)

        def noop(__: None) -> tuple:
          del __
          return pk_step, ah_step, se_mi, seq_mi, logits_mi

        def contrib(__: None) -> tuple:  # noqa: PLR0915
          del __
          if ar_logit_transform_fn is not None:
            comb_vec = ar_logit_transform_fn(
              lrows_fin * cmask_fin[:, jnp.newaxis].astype(lrows_fin.dtype),
              row_state_map,
              sw_use,
            ).astype(log_dtype)
          else:
            combined = combine_logits_multistate_idx(
              lrows_fin,
              cmask_fin.astype(jnp.bool_),
              strat_idx,
              ms_temp,
              sw_use,
              row_state_map,
            )
            comb_vec = jnp.squeeze(combined, axis=0).astype(log_dtype)

          def gb_body(g_i: jax.Array, acc_gb: tuple) -> tuple:
            num_acc, den_acc = acc_gb
            gi = jnp.int32(g_i)
            take = jax.lax.dynamic_index_in_dim(cmask_fin.astype(jnp.bool_), gi).squeeze(axis=0)
            bw_row_here = jax.lax.dynamic_index_in_dim(
              jax.lax.dynamic_index_in_dim(bias_stack, gi, axis=0).squeeze(axis=0),
              lid_here,
              axis=0,
            ).squeeze()
            ww = jax.lax.dynamic_index_in_dim(sw_use, gi).squeeze(axis=0)
            inc_num = jax.lax.select(take, bw_row_here * ww, jnp.zeros_like(bw_row_here))
            inc_den = jax.lax.select(take, ww, jnp.float32(0.0))
            return (num_acc + inc_num.astype(log_dtype), den_acc + inc_den)

          num_b_v, den_b_v = jax.lax.fori_loop(
            jnp.int32(0),
            max_gs_tr,
            gb_body,
            (jnp.zeros((model.w_out.out_features,), dtype=log_dtype), jnp.float32(0.0)),
          )
          group_bias_flat = num_b_v / jnp.maximum(den_b_v, jnp.float32(1e-8))
          logits_wb = comb_vec + group_bias_flat

          def orb_fix(g_fix: jax.Array, hv_fix: jax.Array) -> jax.Array:
            gx = jnp.int32(g_fix)
            take_f = jax.lax.dynamic_index_in_dim(cmask_fin.astype(jnp.bool_), gx).squeeze(axis=0)
            fx_slot = jax.lax.dynamic_index_in_dim(
              jax.lax.dynamic_index_in_dim(fixed_b.astype(jnp.bool_), gx, axis=0).squeeze(axis=0),
              lid_here,
              axis=0,
            ).squeeze(axis=0)
            return hv_fix | jnp.logical_and(take_f, fx_slot)

          init_false = jnp.zeros((), dtype=jnp.bool_)
          has_any_fixed = jax.lax.fori_loop(jnp.int32(0), max_gs_tr, orb_fix, init_false)

          pk_samp, dk_s = jax.random.split(pk_step)

          def samp_vec(*_unused: object) -> jax.Array:
            del _unused
            lo_row = jnp.expand_dims(logits_wb, axis=0)
            samp_logits = lo_row / temp + jax.random.gumbel(
              dk_s,
              shape=lo_row.shape,
              dtype=log_dtype,
            )
            one_hot_sample = straight_through_estimator(samp_logits[..., :20])
            pad_b = jnp.zeros_like(one_hot_sample[..., :1])
            oh = jnp.concatenate([one_hot_sample, pad_b], axis=-1)
            return oh[0, :]

          def fixed_vec(*_unused2: object) -> jax.Array:
            del _unused2

            def fv_body(g_ff: jax.Array, best_tok: jax.Array) -> jax.Array:
              gx_ff = jnp.int32(g_ff)
              cand = jax.lax.dynamic_index_in_dim(cmask_fin.astype(jnp.bool_), gx_ff).squeeze(
                axis=0,
              )
              tok_g = jax.lax.dynamic_index_in_dim(
                jax.lax.dynamic_index_in_dim(fixed_toks.astype(jnp.int32), gx_ff, axis=0).squeeze(
                  axis=0,
                ),
                lid_here,
                axis=0,
              ).squeeze(axis=0)
              best_i = jax.lax.select(cand, jnp.maximum(best_tok, tok_g), best_tok)
              return best_i.astype(jnp.int32)

            fixed_idx = jax.lax.fori_loop(jnp.int32(0), max_gs_tr, fv_body, jnp.int32(-1))
            return jax.nn.one_hot(fixed_idx, model.w_s_embed.num_embeddings, dtype=log_dtype)

          oh_broadcast = jax.lax.cond(has_any_fixed, fixed_vec, samp_vec)
          pk_post = jax.lax.select(has_any_fixed, pk_step, pk_samp)

          emb_new_vec = jnp.matmul(
            oh_broadcast.astype(log_dtype),
            emb_w.astype(log_dtype),
          )

          def upd_si(si_ix: jax.Array, bags: tuple) -> tuple:
            seq_lp, logits_lp, emb_lp = bags
            si_j = jnp.int32(si_ix)
            in_rng = jnp.logical_and(si_j >= jnp.int32(0), si_j < max_gs_tr)
            grp_mem = jax.lax.select(
              in_rng.squeeze(),
              jax.lax.dynamic_index_in_dim(cmask_fin.astype(jnp.bool_), si_j, axis=0).squeeze(
                axis=0,
              ),
              jnp.zeros((), dtype=jnp.bool_),
            )

            row_seq = seq_lp[si_j]
            new_seq_row = jax.lax.select(
              grp_mem,
              row_seq.at[lid_here].set(oh_broadcast.astype(row_seq.dtype)),
              row_seq,
            )
            seq_lp2 = seq_lp.at[si_j].set(new_seq_row)

            row_logits = logits_lp[si_j]
            li_next = jax.lax.select(grp_mem, row_logits.at[lid_here].set(comb_vec), row_logits)
            logits_lp2 = logits_lp.at[si_j].set(li_next)

            row_emb = emb_lp[si_j]
            em_next = jax.lax.select(
              grp_mem,
              row_emb.at[lid_here].set(emb_new_vec.astype(row_emb.dtype)),
              row_emb,
            )
            emb_lp2 = emb_lp.at[si_j].set(em_next)
            return seq_lp2, logits_lp2, emb_lp2

          seq_nf, logits_nf, se_nf = jax.lax.fori_loop(
            jnp.int32(0),
            jnp.int32(s_dim),
            upd_si,
            (seq_mi, logits_mi, se_mi),
          )
          return pk_post, ah_step, se_nf, seq_nf, logits_nf

        return jax.lax.cond(any_cmem, contrib, noop, None)

      active_decode = jnp.logical_and(active_slot.squeeze(), has_member)
      return jax.lax.cond(active_decode, do_slot, skip_slot, None)

    return jax.lax.fori_loop(jnp.int32(0), nslot_i, inner_slot, (pk_w, ah_w, se_w, seq_w, log_w))

  _pk_out, _ah_out, _se_out, seq_out, log_out = jax.lax.fori_loop(
    jnp.int32(0),
    nw_i,
    outer_wave,
    (pk_cur, ah, se, seq_carry, logits_acc),
  )
  del _pk_out
  return seq_out.astype(log_dtype), log_out.astype(log_dtype)

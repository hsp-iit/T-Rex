"""
Qwen3-VL MoT mid-training with tactile + flare — same model + training loop as
train_qwen3vl_flare.py, but the dataloader reads the EgoDex-style pretrain
manifest format produced by
  data/midtrain/scripts/gen_pretrain_mecka_parallel.py

Per episode the dataset expects:
  <ep>/pretrain.hdf5   states[N,62], action_chunks[N,16,62],
                       tracking_error[N-1,56], tactile_f6[N,10,6],
                       attrs(language, num_frames, fps, ...)
  <ep>/raw.h5          symlink to original episode_XXXX.h5 (read at training
                       time for tactile_deform[N,5,240,240] — too big to copy)
  <ep>/ego_view.mp4    head/slow camera (cropped or symlinked by preprocess)
  <ep>/left_wrist.mp4  fast camera 1
  <ep>/right_wrist.mp4 fast camera 2

The collate output dict is API-compatible with SftDataset.collate_fn so the
rest of the cascaded training loop and the FLARE prediction loss are
unchanged.

CLI:
  --data_root  ROOT   directory containing batch_*/pretrain_manifest.json
                      (replaces --data_path JSON from train_qwen3vl_flare.py)
"""

import os, sys

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import glob
import json
import torch
import logging
import argparse
import shutil
import itertools
import time
import math
import re
import wandb
import PIL.Image
import numpy as np
import h5py

from typing import List, Dict, Optional
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler as _DistributedSampler
from torch.optim.lr_scheduler import LambdaLR
from accelerate import Accelerator, DataLoaderConfiguration
from transformers import AutoProcessor, set_seed

from qwen_vla import Qwen3VLVLAModel, extend_position_ids_for_flare, split_slow_fast_embeds
from qwen_vla.flow_matching import validate_split_config, select_handover, tactile_training_pair
from qwen_vla.contract import architecture_from_training_args, load_with_contract, validate_variant
import cv2

logger = logging.getLogger(__name__)


def reinit_tactile_expert(model):
    """Fresh tactile expert (no midtrain checkpoint to resume it from).

    Matrices: Xavier. RMSNorm gains: ones (a zero gain zeroes the norm's output and
    switches the whole tactile branch off at step 0). Other 1-D params (biases): zeros.
    Same rule as upstream T-Rex scripts/train.py (PR #17, fix/tactile-norm-init).
    """
    for name, param in model.named_parameters():
        if "_tactile" in name:
            if param.ndim >= 2:
                nn.init.xavier_uniform_(param)
            elif "norm" in name:
                nn.init.ones_(param)  # RMSNorm gain: zero would kill its branch
            elif param.ndim == 1:
                nn.init.zeros_(param)


def tactile_flow_loss(pred, target, mask=None, slot_mask=None):
    """Tactile-expert MSE. With a per-sample mask (TAPTIC sdk_force), samples
    without a valid native tactile observation (legacy 10 Hz episodes, stale or
    missing reads) contribute nothing; an all-invalid batch gives a zero loss
    that keeps the graph connected.

    slot_mask [B, chunk] (delayed tactile conditioning): only action slots whose
    recorded time has not elapsed at the fast tick F are supervised. The noisy
    chunk keeps all slots (same layout as inference); the loss is the mean over
    supervised (sample, slot) pairs of the per-slot MSE over action dims. With
    slot_mask=None the original formula is used unchanged (bitwise)."""
    if slot_mask is None:
        if mask is None:
            return nn.MSELoss()(pred, target)
        per = (pred.float() - target.float()).pow(2).mean(dim=tuple(range(1, pred.dim())))
        m = mask.to(device=per.device, dtype=per.dtype)
        return (per * m).sum() / m.sum().clamp_min(1.0)
    err = (pred.float() - target.float()).pow(2).mean(dim=-1)                 # [B, chunk]
    m = slot_mask.to(device=err.device, dtype=torch.bool)
    if mask is not None:
        m = m & mask.to(device=err.device, dtype=torch.bool)[:, None]
    m = m.to(err.dtype)
    return (err * m).sum() / m.sum().clamp_min(1.0)


def _tactile_feature_dim(args):
    """Per-finger tactile vector width as built by the TAPTIC loader (saved in training_args.json)."""
    from qwen_vla.taptic_dataset import tactile_feature_dim
    return tactile_feature_dim(args)


def _masked_mse(pred, target, mask):
    """Mean squared error over mask entries; an empty mask gives a connected zero."""
    m = mask.to(device=pred.device, dtype=torch.float32)
    err = (pred.float() - target.to(device=pred.device, dtype=torch.float32)) ** 2
    return (err * m).sum() / m.sum().clamp_min(1.0)


def _risk_dropout(valid, p, training):
    """Risk dropout (OURS): mark a valid risk measurement missing with probability p per sample,
    training only, so the learned 'missing' embedding is not a proxy for the legacy tasks.
    No RNG draw at all when disabled (p == 0 or validation)."""
    if valid is None or not training or not p:
        return valid
    keep = torch.rand(valid.shape[0], device=valid.device) >= p
    return valid.to(torch.bool) & keep


def _future_risk_terms(pred, batch, after_key, base_risk, base_valid):
    """Loss of a per-slot future-risk prediction [B, 16] and (validation) sums for the skill
    against persistence (predict the current risk for every future slot)."""
    tgt, ok = batch['risk_future'], batch['risk_future_valid'].to(torch.bool)
    mask = ok & batch[after_key].to(device=ok.device, dtype=torch.bool)
    loss = _masked_mse(pred, tgt, mask)
    stats = None
    if base_risk is not None:
        with torch.no_grad():
            m = (mask & base_valid.to(device=mask.device, dtype=torch.bool)[:, None]).to(pred.device)
            t = tgt.to(device=pred.device, dtype=torch.float32)
            persist = base_risk.to(device=pred.device, dtype=torch.float32)[:, None].expand_as(t)
            stats = torch.stack([((pred.float() - t) ** 2 * m).sum(), ((persist - t) ** 2 * m).sum(),
                                 m.float().sum()])
    return loss, stats


def _risk_inputs(raw_model, batch, tag):
    """(risk, valid) for the model's risk token, or (None, None) for a risk-free
    model (risk features may still be in the batch for logging / external gate)."""
    if not getattr(raw_model, 'use_risk_tokens', False):
        return None, None
    if batch.get(f'risk_{tag}') is None:
        raise ValueError(f'risk-token model but the batch has no risk_{tag} (pass --risk_stats)')
    return batch[f'risk_{tag}'], batch[f'risk_{tag}_valid']


def cascaded_losses(model, raw_model, batch, args, handover, *, is_stage1, use_flare, K,
                    training, merge_default=2):
    """Action loss (single grad forward, tau ~ full [0,1]) + tactile loss at
    tau in (0, tau_split] from a no-grad Action handover cache. Shared by the
    training and validation loops; statement and RNG order are the original
    loops' (randn for the cache noise, then — training only — the dropout draw).

    Risk tokens: r_S conditions the Action forward and the no-grad cache; r_F
    conditions the tactile step (marked missing when tactile dropout fires)."""
    # viene eseguito il vision encoder
    # input_ids [B,L] -> text embeddings
    #                                   ===> input_embeds [B,L,H]
    # pixel valuse -> vision encoder
    inputs_embeds = raw_model.prepare_inputs_embeds(
        input_ids=batch["input_ids"],
        pixel_values=batch.get("pixel_values"),
        image_grid_thw=batch.get("image_grid_thw"))
    n_slow_imgs = batch["n_slow_images"]
    grid_thw = batch.get("image_grid_thw")
    B = inputs_embeds.shape[0]
    if grid_thw is not None and grid_thw.shape[0] > n_slow_imgs:
        merge = getattr(raw_model.visual, "spatial_merge_size", merge_default)
        n_imgs_per_sample = grid_thw.shape[0] // B
        n_slow_img_tokens = sum(
            int(g[0] * (g[1] // merge) * (g[2] // merge))
            for g in grid_thw[:n_imgs_per_sample][:n_slow_imgs])
        # SEPARAZIONE DELLE DUE CAMERE
        slow_embeds, fast_embeds = split_slow_fast_embeds(
            inputs_embeds, batch["input_ids"], raw_model.image_token_id, n_slow_img_tokens)
    else:
        slow_embeds = inputs_embeds
        fast_embeds = inputs_embeds[:, :0]
    L_slow = slow_embeds.shape[1]
    # Quindi è importante distinguere:
    # - inputs_embeds: contenuto dei token;
    # - position_ids: informazioni utilizzate per costruire le rotazioni posizionali di query e key.
    # I position_ids non vengono concatenati come normali token al tensore inputs_embeds.
    pos_ids, _ = raw_model.get_rope_index(
        input_ids=batch["input_ids"], image_grid_thw=batch.get("image_grid_thw"),
        attention_mask=batch["attention_mask"])
    pos_ids = pos_ids[:, :, :L_slow]
    if use_flare:
        flare_q = raw_model.flare_queries.expand(B, -1, -1).to(device=slow_embeds.device, dtype=slow_embeds.dtype)
        # I 32 token flare attraversano il Latent Expert insieme al contesto visivo e linguistico.
        slow_embeds_ext = torch.cat([slow_embeds, flare_q], dim=1)
        pos_ids = extend_position_ids_for_flare(pos_ids, K)
        L_latent = slow_embeds_ext.shape[1]  # L_slow + K
    else:
        slow_embeds_ext = slow_embeds
        L_latent = L_slow
    if args.use_robot_state and batch["state_raw"] is not None:
        state_vec = batch["state_raw"].to(slow_embeds.device, dtype=slow_embeds.dtype)
        state_embeds = raw_model.state_embedder(state_vec).unsqueeze(1)
    else:
        # Se use_robot_state=0, come nella configurazione che stiamo considerando
        # la rete non riceve un token di stato robotico esplicito.
        # state_embeds.shape = [B, 0, H]
        # n_state = 0
        state_embeds = torch.empty((B, 0, slow_embeds.shape[2]), device=slow_embeds.device, dtype=slow_embeds.dtype)
    n_state = state_embeds.shape[1]
    # r_S: slow-tick risk token, Action-routed, before the timestep token.
    risk_S, risk_S_valid = _risk_inputs(raw_model, batch, 'S')
    risk_S_valid = _risk_dropout(risk_S_valid, getattr(args, 'risk_dropout', 0.), training)
    risk_S_tok = raw_model.risk_token(risk_S, risk_S_valid, 'action', B, slow_embeds.device, slow_embeds.dtype)
    n_risk = risk_S_tok.shape[1]
    # Prima dell'embedder batch["noisy_actions"].shape = [B, 16, D]
    # Dopo batch["noisy_actions"].shape = [B, 16, H]
    noisy_actions = raw_model.x_embedder(batch["noisy_actions"].to(slow_embeds.dtype))
    # Prima dell'embedder batch["timesteps"].shape = [B]
    # Dopo batch["timesteps"].shape = [B, H]
    # Dopo unsqueeze(1) timesteps.shape = [B, 1, H]
    timesteps = raw_model.t_embedder(batch["timesteps"].to(slow_embeds.dtype)).unsqueeze(1)
    chunk = args.action_chunk
    target = batch["target"].to(slow_embeds.dtype)
    n_fast = fast_embeds.shape[1]
    has_any_tac = bool(args.use_tactile_vec or args.use_tactile_deform)

    # Cascaded: shared flow split between action + tactile experts.
    # L_flow — action expert single-step (full τ ∈ [0,1] training).
    # Questa singola istruzione costruisce la sequenza finale in ingresso al MoT:
    # [latent | fast | state | (risk_S) | t | x_t] — one forward with gradient.
    parts = [slow_embeds_ext, fast_embeds, state_embeds] + ([risk_S_tok] if n_risk else []) + [timesteps, noisy_actions]
    full_embeds = torch.cat(parts, dim=1)
    L_total = full_embeds.shape[1]
    # Chiamata al transformer
    # Questa è la prima vera chiamata ai 28 layer MoT.
    # latent_indexes = [0, ..., L_latent-1]
    # action_indexes = [L_latent, ..., L_total-1]   (include risk_S quando presente)
    # tactile_indexes = []
    outputs = model.model(
        inputs_embeds=full_embeds, position_ids=pos_ids, attention_mask=batch["attention_mask"],
        # use_cache=False: in questo forward action del training non viene costruita
        # una KV cache destinata al Tactile Expert. Stiamo addestrando il campo di
        # velocità a un tempo tau campionato, con un singolo forward con gradiente.
        # La cache per il training tattile viene costruita dopo, con un'altra funzione.
        use_cache=False, output_hidden_states=use_flare,
        latent_indexes=torch.arange(0, L_latent, device=full_embeds.device),
        action_indexes=torch.arange(L_latent, L_total, device=full_embeds.device),
        tactile_indexes=torch.arange(0, 0, device=full_embeds.device))
    hidden = outputs.last_hidden_state
    # Calcola la posizione del primo token corrispondente alle azioni rumorose
    # (dopo latent, fast, state, risk_S e il token del tempo).
    act_pred_start = L_latent + n_fast + n_state + n_risk + 1
    # Estrae  hidden[:, act_pred_start:act_pred_start + 16, :]  ->  v_act.shape = [B, 16, D]
    v_act = raw_model.final_layer(hidden[:, act_pred_start: act_pred_start + chunk, :])
    # il calcolo di loss_act non aggiorna ancora alcun peso.
    # Il grafo di autograd è stato costruito, ma il backward avverrà dopo
    # che il codice avrà calcolato anche le altre loss.
    loss_act = nn.MSELoss()(v_act, target)

    res = {'loss_act': loss_act, 'loss_tac': 0.0, 'outputs': outputs, 'hidden': hidden, 'L_slow': L_slow,
           'B': B, 'r_target_norm': None, 'tactile_dropped': False, 'n_risk': n_risk,
           'loss_risk_head': None, 'loss_risk_probe': None, 'risk_head_stats': None, 'risk_probe_stats': None}
    if getattr(raw_model, 'risk_future_probe', False):
        # Linear probe on DETACHED Action-expert hidden states: trains only the probe.
        h_probe = hidden[:, act_pred_start: act_pred_start + chunk, :].detach()
        pred_probe = raw_model.future_risk_probe(h_probe).squeeze(-1)
        res['loss_risk_probe'], res['risk_probe_stats'] = _future_risk_terms(
            pred_probe, batch, 'risk_future_after_S', batch.get('risk_S'), batch.get('risk_S_valid'))
    if has_any_tac and not is_stage1:
        fe = fast_embeds if n_fast > 0 else None
        se = state_embeds if n_state > 0 else None
        # Questa istruzione genera nuovo rumore gaussiano.
        # questo rumore è indipendente da quello utilizzato per calcolare loss_act.
        ahat_noise = torch.randn_like(batch["noisy_actions"])
        with torch.no_grad():
            # cached_kv at tau_split, conditioned on the SAME r_S (matches inference).
            # Qui gli autori conservano la KV cache e il numero di token action.
            # Non conservano x_t per la loss tattile: nel training il Tactile Expert viene
            # supervisionato con un altro punto rumoroso, costruito dalle azioni ground truth.
            x_split, cached_kv, n_action_in_cache, _ = raw_model.forward_flow_action_partial(
                inputs_embeds=slow_embeds_ext, position_ids=pos_ids, noise=ahat_noise,
                attention_mask=batch["attention_mask"], state_embeds=se, fast_embeds=fe,
                num_steps_total=args.cascaded_total_steps, split_step=handover.action_steps,
                flow_time_precision=args.flow_time_precision, refresh_clean_kv=True,
                risk=risk_S, risk_valid=risk_S_valid)
        # INIZIO DELLA PARTE TATTILE
        # Sono state già eseguite due operazioni:
        # - 1. Un forward dell'Action Expert con gradiente per calcolare loss_act.
        # - 2. Un forward dell'Action Expert senza gradiente per costruire la KV cache al passo τ_split.
        # L_flow_tactile — tactile expert as velocity predictor at τ ∈ [0, τ_split]
        # AZIONI CORRETTE: sono le azioni provenienti dalle dimostrazioni, dopo la normalizzazione.
        # a in [B,16,D]
        norm_actions_gt = batch["norm_actions"].to(slow_embeds.dtype)
        # Nuovo rumore gaussiano (collate_fn())
        eps_r = batch["eps_r"].to(slow_embeds.dtype)
        x_tau, tau_t, v_target_r = tactile_training_pair(
            norm_actions_gt, eps_r, batch['time_r'], handover, args.flow_time_precision)
        # Self-handover (training only, --tactile_self_handover p): for a random fraction p of the
        # samples the pair starts at the Action expert's OWN state x_split at tau_split (the no-grad
        # trajectory above, same weights) and follows the straight path x_split -> demo:
        #   x_tau = a + (tau / tau_split) (x_split - a),   v = (x_split - a) / tau_split.
        # At inference the Tactile expert always continues from x_split; with ground-truth pairs only,
        # its input is centred on the demo and it never learns to correct the Action expert's error.
        p_self = float(getattr(args, 'tactile_self_handover', 0.) or 0.) if training else 0.
        res['self_handover_frac'] = 0.
        if p_self > 0:
            sel = torch.rand(B, device=x_tau.device) < p_self
            a_f, xs_f = norm_actions_gt.float(), x_split.float()
            s_f = (tau_t.float() / handover.tau)[:, None, None]
            m3 = sel[:, None, None]
            x_tau = torch.where(m3, (a_f + s_f * (xs_f - a_f)).to(x_tau.dtype), x_tau)
            v_target_r = torch.where(m3, ((xs_f - a_f) / handover.tau).to(v_target_r.dtype), v_target_r)
            res['self_handover_frac'] = float(sel.float().mean())
        res['r_target_norm'] = v_target_r.float().abs().mean().detach()
        dropout = getattr(args, 'cascaded_tactile_dropout', 0.) if training else 0.
        # Tactile dropout — zero the values, not the tensor shape.
        # Con il valore predefinito 0.1, ogni training step ha una probabilità del 10% di azzerare gli input tattili interessati.
        drop = (dropout > 0 and torch.rand((), device=full_embeds.device).item() < dropout)
        tac_f6_in = batch.get("tactile_f6s_delayed")
        tac_def_in = batch.get("tactile_deforms_delayed")
        tac_codes_in = batch.get("tactile_codes")
        tac_hist_in = batch.get("tactile_f6_history")
        risk_F, risk_F_valid = _risk_inputs(raw_model, batch, 'F')
        if drop:
            tac_f6_in = torch.zeros_like(tac_f6_in) if tac_f6_in is not None else None
            tac_def_in = torch.zeros_like(tac_def_in) if tac_def_in is not None else None
            tac_codes_in = torch.zeros_like(tac_codes_in) if tac_codes_in is not None else None
            tac_hist_in = torch.zeros_like(tac_hist_in) if tac_hist_in is not None else None
            # No tactile signal -> no fast-tick risk either: explicit "missing", not r=0.
            risk_F_valid = torch.zeros_like(risk_F_valid) if risk_F_valid is not None else None
        risk_F_valid = _risk_dropout(risk_F_valid, getattr(args, 'risk_dropout', 0.), training)
        res['tactile_dropped'] = bool(drop)
        # Calcolo della loss e chiamata a tactile_flow_train_step
        # Quindi tactile_flow_train_step non restituisce direttamente un chunk pulito.
        # Restituisce la velocità predetta dal Tactile Expert nel punto di flow campionato.
        # L'aggiornamento con Eulero si farà durante l'inferenza, nella funzione tactile_flow_continue
        head = getattr(raw_model, 'risk_future_head', False)
        out_t = raw_model.tactile_flow_train_step(
            cached_kv=cached_kv, latent_position_ids=pos_ids, n_action_in_cache=n_action_in_cache,
            x_tau=x_tau, tau=tau_t, flow_time_precision=args.flow_time_precision,
            tactile_f6=tac_f6_in, tactile_deform=tac_def_in, tactile_codes=tac_codes_in,
            tactile_f6_history=tac_hist_in, risk=risk_F, risk_valid=risk_F_valid, return_hidden=head)
        v_pred_r = out_t[0] if head else out_t
        if head:
            # Future risk from the Tactile expert's chunk tokens (aux loss, trains the tactile path).
            pred_head = raw_model.future_risk_head(out_t[1]).squeeze(-1)
            res['loss_risk_head'], res['risk_head_stats'] = _future_risk_terms(
                pred_head, batch, 'risk_future_after_F', batch.get('risk_F'), batch.get('risk_F_valid'))
        res['loss_tac'] = tactile_flow_loss(v_pred_r, v_target_r, batch.get("tactile_loss_mask"),
                                            batch.get("tactile_slot_mask"))
        # Contact weighting (training only, --tactile_force_contact_weight w): the hand dims (9:21) of
        # samples flagged force_contact (force task, fingertip |F| above threshold at the fast tick)
        # count w times in the tactile loss. The unweighted value is kept for logging.
        w_fc = float(getattr(args, 'tactile_force_contact_weight', 1.) or 1.) if training else 1.
        res['loss_tac_plain'] = res['loss_tac']
        res['force_contact_frac'] = None
        sw = None          # per-(sample, dim) sqrt-weights of the tactile loss; None = unweighted (baseline)
        if batch.get('force_contact') is not None:
            fc = batch['force_contact'].to(device=v_pred_r.device, dtype=torch.bool)
            res['force_contact_frac'] = float(fc.float().mean())
            if w_fc != 1.:
                sw = torch.ones(v_pred_r.shape[0], 1, v_pred_r.shape[-1], device=v_pred_r.device, dtype=torch.float32)
                sw[fc, :, 9:] = w_fc ** 0.5
        # Event weighting (training only, --tactile_event_weight w; taptic/touch_events.py): samples flagged
        # touch_event (an event of --tactile_event_types inside --tactile_event_window_ms, a task in
        # --tactile_event_tasks, valid native touch) count w times on EVERY action dim. With
        # --tactile_event_normalize 1 the per-sample weights are rescaled to mean 1 over the supervised samples
        # of the batch, so the loss scale does not depend on how many events a batch holds. Combines
        # multiplicatively with contact weighting.
        w_ev = float(getattr(args, 'tactile_event_weight', 1.) or 1.) if training else 1.
        res['touch_event_frac'] = None
        if batch.get('touch_event') is not None:
            ev = batch['touch_event'].to(device=v_pred_r.device, dtype=torch.bool)
            res['touch_event_frac'] = float(ev.float().mean())
            if w_ev != 1.:
                f = torch.where(ev, torch.tensor(w_ev, device=ev.device), torch.tensor(1., device=ev.device)).float()
                if int(getattr(args, 'tactile_event_normalize', 1)):
                    m = batch.get("tactile_loss_mask")
                    m = (torch.ones_like(ev) if m is None else m.to(device=ev.device, dtype=torch.bool))
                    if m.any():
                        f = f / f[m].mean()
                s = f.sqrt()[:, None, None].expand(-1, 1, v_pred_r.shape[-1])
                sw = s if sw is None else sw * s
        if sw is not None:
            res['loss_tac'] = tactile_flow_loss(v_pred_r.float() * sw, v_target_r.float() * sw,
                                                batch.get("tactile_loss_mask"), batch.get("tactile_slot_mask"))
    return res


def batch_diagnostics(batch):
    """Scalar logging of risk / delay inputs (no effect on training)."""
    d = {}
    for tag in ('S', 'F'):
        r, v = batch.get(f'risk_{tag}'), batch.get(f'risk_{tag}_valid')
        if r is not None:
            vf = v.float()
            d[f'risk/{tag}_valid_frac'] = float(vf.mean())
            d[f'risk/{tag}_mean'] = float((r.float() * vf).sum() / vf.sum().clamp_min(1))
            f = batch.get(f'risk_features_{tag}') or {}
            for k in ('M', 'R', 'V', 'R_pos', 'R_neg'):
                if k in f:
                    kv = f[(k[0] if k.startswith('R_') else k) + '_valid'].float()
                    d[f'risk/{tag}_{k}_mean'] = float((f[k].float() * kv).sum() / kv.sum().clamp_min(1))
    if batch.get('tactile_delay_ns') is not None:
        d['tactile/delay_ms_mean'] = float(batch['tactile_delay_ns'].float().mean() / 1e6)
    if batch.get('tactile_slot_mask') is not None:
        d['tactile/supervised_slot_frac'] = float(batch['tactile_slot_mask'].float().mean())
    return d
logging.basicConfig(level="INFO")


def _rot6d_to_mat(rot6d):
    """6D rotation (first two columns) → 3×3 rotation matrix."""
    col1 = rot6d[:3]
    col2 = rot6d[3:6]
    col3 = np.cross(col1, col2)
    return np.column_stack([col1, col2, col3])


def _arm9d_to_axis_angle(arm_9d):
    """[trans(3), rot6d(6)] → [trans(3), axis_angle(3)]."""
    R = _rot6d_to_mat(arm_9d[3:9])
    aa, _ = cv2.Rodrigues(R)
    return np.concatenate([arm_9d[:3], aa.flatten()])


def _axis_angle_to_arm9d(arm_aa):
    """[trans(3), axis_angle(3)] → [trans(3), rot6d(6)]."""
    R, _ = cv2.Rodrigues(arm_aa[3:6])
    return np.concatenate([arm_aa[:3], R[:, 0], R[:, 1]])


def add_tracking_error_noise(state, te_mean, te_std, action_dim):
    """Add tracking-error noise to robot state.

    Supports single-arm (action_dim=31, te=28D) and bimanual (action_dim=62, te=56D).
    Per arm (28D tracking error = 3 xyz + 3 axis-angle + 22 hand):
      1. Convert arm 9D [trans, rot6d] → [trans, axis_angle]
      2. Sample noise ~ N(te_mean, te_std)
      3. Add noise
      4. Convert back to [trans, rot6d]
    """
    noisy = state.copy()
    n_arms = action_dim // 31
    for arm_idx in range(n_arms):
        offset = arm_idx * 31
        arm_9d = state[offset:offset + 9]
        hand_22d = state[offset + 9:offset + 31]

        arm_aa = _arm9d_to_axis_angle(arm_9d)  # (6,)

        te_off = arm_idx * 28
        noise_arm = np.random.normal(te_mean[te_off:te_off + 6],
                                     te_std[te_off:te_off + 6]).astype(np.float32)
        noise_hand = np.random.normal(te_mean[te_off + 6:te_off + 28],
                                      te_std[te_off + 6:te_off + 28]).astype(np.float32)

        noisy_arm_9d = _axis_angle_to_arm9d(arm_aa + noise_arm)
        noisy[offset:offset + 9] = noisy_arm_9d
        noisy[offset + 9:offset + 31] = hand_22d + noise_hand
    return noisy


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                                     min_lr_ratio=0.0, num_cycles=0.5):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * 2 * num_cycles * progress))
        return (1 - min_lr_ratio) * cosine + min_lr_ratio
    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)


# ────────────────────────────────────────────────────────────────────────────
# Mid-training pretrain-format dataset (replaces SftDataset).
#
# Reads <data_root>/*/pretrain_manifest.json — per episode, opens
#   pretrain.hdf5  (states, action_chunks, tactile_f6, tracking_error, lang)
#   raw.h5         (tactile_deform — symlink to original episode HDF5)
#   ego_view.mp4 + left_wrist.mp4 + right_wrist.mp4 (decoded once per episode)
# and serves frames from a per-worker cache. EpisodeGroupedSampler keeps the
# cache warm by emitting each episode's frames contiguously to a given rank.
# ────────────────────────────────────────────────────────────────────────────


class EpisodeGroupedSampler(_DistributedSampler):
    """Distributed sampler that emits each episode's frames contiguously so
    that the per-worker episode cache stays warm. Episode *order* is shuffled
    each epoch; frame order *within* an episode is sequential. Subclasses
    DistributedSampler so accelerate/DeepSpeed don't replace it.
    """

    def __init__(self, dataset, num_replicas=None, rank=None,
                 shuffle=True, seed=0, drop_last=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank,
                         shuffle=shuffle, seed=seed, drop_last=drop_last)
        self._cum_frames = dataset._cum_frames.copy()
        self._num_episodes = dataset._num_episodes
        self._frame_counts = np.diff(self._cum_frames, prepend=0)
        self._orig_starts = np.zeros(self._num_episodes, dtype=np.int64)
        if self._num_episodes > 1:
            self._orig_starts[1:] = self._cum_frames[:-1]

    # Restartable training (docs/TOUCH_MODULES.md): drop the first N indices of the next epoch on this rank (the
    # samples an interrupted run already trained on), once; the order of the rest is unchanged.
    skip_samples = 0

    def _skip(self, it):
        n, self.skip_samples = int(self.skip_samples), 0
        return itertools.islice(it, n, None) if n else it

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            ep_perm = torch.randperm(self._num_episodes, generator=g).numpy()
        else:
            ep_perm = np.arange(self._num_episodes)

        shuffled_fc = self._frame_counts[ep_perm]
        shuffled_cum = np.cumsum(shuffled_fc)
        total_frames = int(shuffled_cum[-1])

        chunk = self.num_samples
        rank_start = self.rank * chunk
        rank_end = rank_start + chunk
        actual_end = min(rank_end, total_frames)

        ep_first = int(np.searchsorted(shuffled_cum, rank_start, side='right'))
        ep_last = int(np.searchsorted(shuffled_cum, actual_end, side='right'))
        ep_last = min(ep_last, self._num_episodes - 1)

        parts = []
        for i in range(ep_first, ep_last + 1):
            ep_idx = int(ep_perm[i])
            shuf_start = int(shuffled_cum[i - 1]) if i > 0 else 0
            t_start = max(0, rank_start - shuf_start)
            t_end = min(int(self._frame_counts[ep_idx]), actual_end - shuf_start)
            if t_start < t_end:
                base = int(self._orig_starts[ep_idx])
                parts.append(np.arange(base + t_start, base + t_end, dtype=np.int64))

        indices = np.concatenate(parts) if parts else np.array([], dtype=np.int64)

        if len(indices) < chunk:
            needed = chunk - len(indices)
            wrap_parts = []
            remaining = needed
            for i in range(self._num_episodes):
                if remaining <= 0:
                    break
                ep_idx = int(ep_perm[i])
                base = int(self._orig_starts[ep_idx])
                n = min(int(self._frame_counts[ep_idx]), remaining)
                wrap_parts.append(np.arange(base, base + n, dtype=np.int64))
                remaining -= n
            if wrap_parts:
                indices = np.concatenate([indices] + wrap_parts)

        return self._skip(iter(indices[:chunk].tolist()))


class RiskOversampledEpisodeSampler(EpisodeGroupedSampler):
    """EpisodeGroupedSampler with compute-neutral high-risk oversampling (OURS).

    Each epoch every high-risk row (r_S valid and >= threshold) appears `factor` times and the
    same number of extra slots is freed by dropping randomly chosen other rows (a fresh draw per
    epoch), so the epoch length, the number of optimizer steps and the LR schedule are those of
    the plain sampler. Episode order and the rank split follow EpisodeGroupedSampler (frames stay
    grouped by episode, duplicates adjacent, so the per-worker episode cache stays warm).
    """

    def __init__(self, dataset, high_mask, factor, **kw):
        super().__init__(dataset, **kw)
        self.high = np.asarray(high_mask, bool)
        self.factor = int(factor)
        if len(self.high) != int(self._cum_frames[-1]):
            raise ValueError('high-risk mask does not match the dataset rows')
        n_high = int(self.high.sum())
        self.n_drop = (self.factor - 1) * n_high
        if self.factor < 2 or self.n_drop > int((~self.high).sum()):
            raise ValueError(f'cannot oversample {n_high} high-risk rows x{self.factor} at constant epoch length')
        self.high_share = self.factor * n_high / len(self.high)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        ep_perm = (torch.randperm(self._num_episodes, generator=g).numpy() if self.shuffle
                   else np.arange(self._num_episodes))
        mult = np.where(self.high, self.factor, 1).astype(np.int64)
        low = np.flatnonzero(~self.high)
        drop = low[torch.randperm(len(low), generator=g)[:self.n_drop].numpy()]
        mult[drop] = 0
        parts = []
        for ep_idx in ep_perm:
            a = int(self._orig_starts[ep_idx])
            b = a + int(self._frame_counts[ep_idx])
            parts.append(np.repeat(np.arange(a, b, dtype=np.int64), mult[a:b]))
        order = np.concatenate(parts)
        chunk = self.num_samples
        lo = self.rank * chunk
        idx = order[lo:lo + chunk]
        if len(idx) < chunk:                              # wrap like the parent sampler
            idx = np.concatenate([idx, order[:chunk - len(idx)]])
        return self._skip(iter(idx.tolist()))


class MidtrainTacFlareDataset(Dataset):
    """Pretrain-format mid-training dataset with tactile + bimanual wrist views.

    Drop-in replacement for SftDataset: same constructor signature,
    `create_val_split`, `collate_fn`, and `stats_data` attribute.
    """

    def __init__(self, config, processor, accelerator):
        self.config = config
        self.processor = processor
        self.accelerator = accelerator

        manifest_paths = sorted(
            glob.glob(os.path.join(config.data_root, "*", "pretrain_manifest.json"))
        )
        if not manifest_paths:
            raise FileNotFoundError(
                f"No pretrain_manifest.json found under {config.data_root}/*/")

        _episode_dirs, _frame_counts = [], []
        _ep_crop_boxes = []   # parallel to _episode_dirs; (y0,y1,x0,x1) | None
        all_state_q01,  all_state_q99  = [], []
        all_tacf6_q01,  all_tacf6_q99  = [], []
        all_te_mean,    all_te_std     = [], []

        for mp in manifest_paths:
            with open(mp, "r") as f:
                manifest = json.load(f)
            # Per-batch head_crop_box. Manifests written before the
            # dataloader-crop change (e.g. the already-processed NV data)
            # don't have this field — treat as no-crop.
            cb = manifest.get("head_crop_box", None)
            crop_tuple = tuple(int(x) for x in cb) if cb else None
            for ep in manifest["episodes"]:
                _episode_dirs.append(ep["episode_dir"])
                _frame_counts.append(ep["num_frames"])
                _ep_crop_boxes.append(crop_tuple)
            stats = manifest.get("statistics", {})
            # NOTE: action stats from manifests are per-dim (shape [action_dim]),
            # which forces every chunk position to share one normalization range.
            # The post-train pipeline writes per-(step, dim) stats (shape
            # [action_chunk, action_dim]) so each chunk position gets its own
            # range — that matters because deltabase actions grow linearly with
            # chunk position.  We ignore manifest action stats and re-compute
            # per-(step, dim) directly from action_chunks below so the schema
            # matches post-train and resume from a midtrain ckpt is consistent.
            if "state" in stats:
                all_state_q01.append(np.array(stats["state"]["q01"], dtype=np.float32))
                all_state_q99.append(np.array(stats["state"]["q99"], dtype=np.float32))
            if "tactile_f6" in stats and stats["tactile_f6"]:
                all_tacf6_q01.append(np.array(stats["tactile_f6"]["q01"], dtype=np.float32))
                all_tacf6_q99.append(np.array(stats["tactile_f6"]["q99"], dtype=np.float32))
            te_block = stats.get("tracking_error", {})
            if te_block:
                all_te_mean.append(np.array(te_block["mean"], dtype=np.float32))
                all_te_std.append(np.array(te_block["std"], dtype=np.float32))

        accelerator.print(f"[Midtrain] Loaded {len(manifest_paths)} batch manifests")

        # ── Per-(step, dim) action stats + per-dim tactile_f6 stats ─────────
        # Computed once over the raw episode data and cached as a post-train-
        # compatible JSON.  Replaces the manifest pool for action (always)
        # and for tactile_f6 (when the cache has it) so the trainer's
        # normalization always reflects the data it actually reads.
        (self.action_min, self.action_max,
         _cached_tacf6_min, _cached_tacf6_max) = (
            self._compute_or_load_normalization_stats(
                _episode_dirs, accelerator))

        # State stats remain per-dim — state is a single observation, not a
        # chunk, so per-step normalization doesn't apply.
        self.state_min  = np.min(np.stack(all_state_q01),  axis=0)
        self.state_max  = np.max(np.stack(all_state_q99),  axis=0)
        if _cached_tacf6_min is not None:
            self.tacf6_min = _cached_tacf6_min
            self.tacf6_max = _cached_tacf6_max
            accelerator.print(
                f"[Midtrain] tactile_f6 stats from raw-data cache: "
                f"q99-q01 mean={(self.tacf6_max - self.tacf6_min).mean():.4f}")
        elif all_tacf6_q01:
            self.tacf6_min = np.min(np.stack(all_tacf6_q01), axis=0)
            self.tacf6_max = np.max(np.stack(all_tacf6_q99), axis=0)
            accelerator.print(
                f"[Midtrain] tactile_f6 stats from manifest pool (fallback): "
                f"q99-q01 mean={(self.tacf6_max - self.tacf6_min).mean():.4f}")
        else:
            # Fallback: pull tactile_f6 q01/q99 from a VQ-VAE checkpoint when
            # the source manifests don't include them.  Without this, the
            # default [-1, 1] range would clip real F6 values (which reach
            # ±20+) and the tactile expert would only see a near-binary
            # contact / no-contact signal.
            stats_loaded = False
            ckpt_path = (getattr(config, "tactile_f6_stats_ckpt", "")
                         or getattr(config, "vqvae_ckpt", ""))
            if ckpt_path and os.path.isfile(ckpt_path):
                try:
                    import torch as _torch
                    blob = _torch.load(ckpt_path, map_location="cpu",
                                       weights_only=False)
                    if isinstance(blob, dict) and "stats" in blob:
                        st = blob["stats"]
                        self.tacf6_min = np.asarray(st["tacf6_min"], dtype=np.float32)
                        self.tacf6_max = np.asarray(st["tacf6_max"], dtype=np.float32)
                        stats_loaded = True
                        accelerator.print(
                            f"[Midtrain] tactile_f6 stats loaded from VQ-VAE "
                            f"ckpt: {ckpt_path}")
                except Exception as e:
                    accelerator.print(
                        f"[Midtrain] WARN: failed to load tactile_f6 stats "
                        f"from {ckpt_path}: {e}")
            if not stats_loaded:
                self.tacf6_min = np.full(60, -1.0, dtype=np.float32)
                self.tacf6_max = np.full(60, +1.0, dtype=np.float32)
                accelerator.print(
                    "[Midtrain] WARN: no tactile_f6 stats in manifests AND no "
                    "VQ-VAE ckpt provided — using default [-1, 1] which WILL "
                    "saturate real F6 values.  Pass --tactile_f6_stats_ckpt "
                    "or --vqvae_ckpt with a checkpoint that has stats baked in.")

        te_dim = (config.action_dim // 31) * 28
        if all_te_mean:
            self.te_mean = np.mean(np.stack(all_te_mean), axis=0)
            self.te_std  = np.mean(np.stack(all_te_std),  axis=0)
        else:
            self.te_mean = np.zeros(te_dim, dtype=np.float32)
            self.te_std  = np.zeros(te_dim, dtype=np.float32)

        self.action_mask = np.ones(config.action_dim, dtype=bool)
        self.state_mask  = np.ones(config.action_dim, dtype=bool)
        self.tacf6_mask  = np.ones(60, dtype=bool)

        # `stats_data` is consumed by save_checkpoint to produce stats_data.json.
        # Keyed under "midtrain" so eval scripts can `next(iter(...))` like before.
        self.stats_data = {
            "midtrain": {
                "action": {
                    "mask": self.action_mask.tolist(),
                    "q01": self.action_min.tolist(),
                    "q99": self.action_max.tolist(),
                },
                "state": {
                    "mask": self.state_mask.tolist(),
                    "q01": self.state_min.tolist(),
                    "q99": self.state_max.tolist(),
                },
                "tactile_f6": {
                    "mask": self.tacf6_mask.tolist(),
                    "q01": self.tacf6_min.tolist(),
                    "q99": self.tacf6_max.tolist(),
                },
                "tracking_error": {
                    "mean": self.te_mean.tolist(),
                    "std":  self.te_std.tolist(),
                },
            }
        }

        self.image_size = tuple(config.image_size) if config.image_size else None
        self.use_flare = bool(getattr(config, "use_flare", 0))
        self.n_flare_steps = getattr(config, "n_flare_steps", 0) if self.use_flare else 0
        self.flare_frame_stride = getattr(config, "flare_frame_stride", 4)

        self._episode_dirs = _episode_dirs
        self._ep_crop_boxes = _ep_crop_boxes
        _frame_counts = np.array(_frame_counts, dtype=np.int64)
        self._cum_frames = np.cumsum(_frame_counts)
        self._total_transitions = int(self._cum_frames[-1])
        self._num_episodes = len(_episode_dirs)

        accelerator.print(
            f"[Midtrain] {self._num_episodes} episodes, "
            f"{self._total_transitions} transitions, "
            f"flare={self.n_flare_steps}×{self.flare_frame_stride}")

    @staticmethod
    def _read_episode_tactile_f6(ep_dir):
        """Load tactile_f6 [N, 10, 6] for one episode with the dual-source
        fallback used by `_load_episode_cache`: try pretrain.hdf5's
        `tactile_f6` first, then raw.h5's per-hand keys.  Returns None when
        neither source is available.
        """
        ph5 = os.path.join(ep_dir, "pretrain.hdf5")
        if os.path.isfile(ph5):
            try:
                with h5py.File(ph5, "r") as f:
                    if "tactile_f6" in f:
                        return f["tactile_f6"][:].astype(np.float32, copy=False)
            except Exception:
                pass
        rh5 = os.path.join(ep_dir, "raw.h5")
        if os.path.isfile(rh5):
            try:
                with h5py.File(rh5, "r") as f:
                    if ("left_hand_tactile_f6" in f
                            and "right_hand_tactile_f6" in f):
                        l = f["left_hand_tactile_f6"][:]
                        r = f["right_hand_tactile_f6"][:]
                        return np.concatenate(
                            [l, r], axis=1).astype(np.float32, copy=False)
            except Exception:
                pass
        return None

    def _compute_or_load_normalization_stats(self, episode_dirs, accelerator):
        """Compute action + tactile_f6 normalization stats from raw episode
        data, cached as a post-train-compatible JSON file.

        Both action and tactile_f6 stats are pooled across the full midtrain
        mix.  Schema (cp-droppable as a post-train `_statistics.json`):

            {
              "midtrain": {
                "action": {
                  "mask": [D],         # all-ones
                  "q01":  [C][D],      # per-(chunk_step, dim)
                  "q99":  [C][D]
                },
                "tactile_f6": {        # omitted if no episode has tactile data
                  "mask": [60],        # all-ones
                  "q01":  [60],
                  "q99":  [60]
                }
              }
            }

        Computing fresh closes the gap between manifest stats (which may pool
        a stale or non-representative quantile per source) and what the
        trainer actually reads — same fix as for action stats.

        Delete the cache file when the data_root episode set changes.
        """
        action_chunk = int(self.config.action_chunk)
        action_dim   = int(self.config.action_dim)
        cache_path   = os.path.join(
            self.config.data_root,
            f"midtrain_statistics_c{action_chunk}_d{action_dim}.json")

        if not os.path.isfile(cache_path):
            if accelerator.is_main_process:
                accelerator.print(
                    f"[Midtrain] computing per-(step, dim) action stats + "
                    f"per-dim tactile_f6 stats over {len(episode_dirs)} "
                    f"episodes → {cache_path} "
                    f"(one-time; cached for next launch)")
                from tqdm import tqdm as _tqdm
                chunks_list = []
                tac_list = []
                for ep_dir in _tqdm(episode_dirs, desc="midtrain stats"):
                    ph5 = os.path.join(ep_dir, "pretrain.hdf5")
                    if os.path.isfile(ph5):
                        try:
                            with h5py.File(ph5, "r") as f:
                                if "action_chunks" in f:
                                    chunks_list.append(
                                        f["action_chunks"][:].astype(
                                            np.float32, copy=False))
                        except Exception as e:
                            accelerator.print(
                                f"[Midtrain] WARN: skip action_chunks "
                                f"in {ph5}: {e}")
                    tac = self._read_episode_tactile_f6(ep_dir)
                    if tac is not None:
                        tac_list.append(tac)

                if not chunks_list:
                    raise RuntimeError(
                        "[Midtrain] no action_chunks found in any episode "
                        "HDF5; cannot compute per-(step, dim) action stats.")
                A = np.concatenate(chunks_list, axis=0)        # [N, C, D]
                accelerator.print(
                    f"[Midtrain] action quantile over {A.shape[0]} chunks "
                    f"(shape={tuple(A.shape)})")
                q01_a = np.quantile(A, 0.01, axis=0).astype(np.float32)
                q99_a = np.quantile(A, 0.99, axis=0).astype(np.float32)
                del A, chunks_list

                tactile_block = None
                if tac_list:
                    T = np.concatenate(tac_list, axis=0)       # [Nt, 10, 6]
                    T_flat = T.reshape(-1, 60)
                    accelerator.print(
                        f"[Midtrain] tactile_f6 quantile over {T_flat.shape[0]} "
                        f"frames (raw shape={tuple(T.shape)})")
                    q01_t = np.quantile(T_flat, 0.01, axis=0).astype(np.float32)
                    q99_t = np.quantile(T_flat, 0.99, axis=0).astype(np.float32)
                    tactile_block = {
                        "mask": np.ones(60, dtype=bool).tolist(),
                        "q01":  q01_t.tolist(),
                        "q99":  q99_t.tolist(),
                    }
                    del T, T_flat, tac_list
                else:
                    accelerator.print(
                        "[Midtrain] no tactile_f6 found in any episode; "
                        "tactile_f6 stats will fall back to manifests / "
                        "VQ-VAE ckpt at load time")

                blob = {
                    "midtrain": {
                        "action": {
                            "mask": np.ones(action_dim, dtype=bool).tolist(),
                            "q01":  q01_a.tolist(),
                            "q99":  q99_a.tolist(),
                        }
                    }
                }
                if tactile_block is not None:
                    blob["midtrain"]["tactile_f6"] = tactile_block

                tmp = cache_path + ".tmp"
                with open(tmp, "w") as _f:
                    json.dump(blob, _f)
                os.replace(tmp, cache_path)
                accelerator.print(
                    f"[Midtrain] wrote midtrain stats "
                    f"(action={tuple(q01_a.shape)}, "
                    f"tactile_f6="
                    f"{'(60,)' if tactile_block is not None else 'absent'}) "
                    f"→ {cache_path}")
            accelerator.wait_for_everyone()

        with open(cache_path, "r") as _f:
            blob = json.load(_f)
        ds_key = next(iter(blob))
        action_min = np.asarray(blob[ds_key]["action"]["q01"], dtype=np.float32)
        action_max = np.asarray(blob[ds_key]["action"]["q99"], dtype=np.float32)
        expected = (action_chunk, action_dim)
        if action_min.shape != expected or action_max.shape != expected:
            raise ValueError(
                f"[Midtrain] cached action stats shape {action_min.shape} != "
                f"expected {expected}; delete {cache_path} and re-run.")
        tactile_block = blob[ds_key].get("tactile_f6")
        if tactile_block is not None:
            tacf6_min = np.asarray(tactile_block["q01"], dtype=np.float32)
            tacf6_max = np.asarray(tactile_block["q99"], dtype=np.float32)
            if tacf6_min.shape != (60,) or tacf6_max.shape != (60,):
                raise ValueError(
                    f"[Midtrain] cached tactile_f6 stats shape "
                    f"{tacf6_min.shape} != expected (60,); "
                    f"delete {cache_path} and re-run.")
        else:
            tacf6_min = tacf6_max = None
        return action_min, action_max, tacf6_min, tacf6_max

    def __len__(self):
        return self._total_transitions

    def create_val_split(self, val_ratio=0.05, seed=42):
        import copy
        rng = np.random.RandomState(seed)
        n_val = max(1, int(self._num_episodes * val_ratio))
        perm = rng.permutation(self._num_episodes)
        val_eps = sorted(perm[:n_val].tolist())
        train_eps = sorted(perm[n_val:].tolist())

        frame_counts = np.diff(self._cum_frames, prepend=0)

        val_ds = copy.copy(self)
        val_ds._episode_dirs = [self._episode_dirs[i] for i in val_eps]
        val_ds._ep_crop_boxes = [self._ep_crop_boxes[i] for i in val_eps]
        val_fc = frame_counts[val_eps]
        val_ds._cum_frames = np.cumsum(val_fc)
        val_ds._total_transitions = int(val_ds._cum_frames[-1])
        val_ds._num_episodes = len(val_eps)

        self._episode_dirs = [self._episode_dirs[i] for i in train_eps]
        self._ep_crop_boxes = [self._ep_crop_boxes[i] for i in train_eps]
        train_fc = frame_counts[train_eps]
        self._cum_frames = np.cumsum(train_fc)
        self._total_transitions = int(self._cum_frames[-1])
        self._num_episodes = len(train_eps)

        self.accelerator.print(
            f"[Midtrain] Train/Val split: {self._num_episodes} train eps "
            f"({self._total_transitions} frames), "
            f"{val_ds._num_episodes} val eps "
            f"({val_ds._total_transitions} frames)")
        return val_ds

    @staticmethod
    def _normalize(values, mask, vmin, vmax):
        return np.where(
            mask,
            np.clip(2 * (values - vmin) / (vmax - vmin + 1e-8) - 1, -1, 1),
            values,
        )

    @staticmethod
    def _decode_video(path: str, crop_box=None) -> list:
        """Decode all frames into a list of HWC uint8 RGB arrays. When
        `crop_box=(y0,y1,x0,x1)` is given, slice each frame *during* decode
        so the per-worker cache only holds cropped frames (this is how BKL
        head videos get cropped at training time without the preprocess
        re-encoding the mp4 to disk).
        """
        cap = cv2.VideoCapture(path)
        frames = []
        if crop_box is not None:
            y0, y1, x0, x1 = crop_box
        while True:
            ret, fr = cap.read()
            if not ret:
                break
            if crop_box is not None:
                fr = fr[y0:y1, x0:x1]
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    def _load_episode_cache(self, ep_idx: int):
        """Bulk-read everything we need for one episode. Called once per
        episode per worker; previous episode's cache is freed when ep_idx
        changes (Python GC reclaims the large numpy arrays/frame lists).
        """
        ep_dir = self._episode_dirs[ep_idx]
        self._cache_ep_idx = ep_idx
        self._cache_h5 = None
        self._cache_tac = None
        self._cache_codes = None
        self._cache_videos = {"head": None, "wrist_l": None, "wrist_r": None}

        ph5 = os.path.join(ep_dir, "pretrain.hdf5")
        if os.path.isfile(ph5):
            try:
                with h5py.File(ph5, "r") as f:
                    self._cache_h5 = {
                        "states": f["states"][:],
                        "action_chunks": f["action_chunks"][:],
                        "tactile_f6": f["tactile_f6"][:] if "tactile_f6" in f else None,
                        "language": str(f.attrs.get("language", "")),
                    }
            except Exception:
                pass

        # tactile_f6 fallback: newer in-lab data layout stores tactile in
        # raw.h5 as separate per-hand keys (left_hand_tactile_f6 [N,5,6] +
        # right_hand_tactile_f6 [N,5,6]) rather than as a concatenated
        # tactile_f6 [N,10,6] in pretrain.hdf5.  If we didn't get it from
        # pretrain.hdf5, build it from raw.h5 here — otherwise the trainer
        # silently runs with all-zero tactile_f6.
        if (self._cache_h5 is not None
                and self._cache_h5.get("tactile_f6") is None):
            rh5_for_f6 = os.path.join(ep_dir, "raw.h5")
            if os.path.isfile(rh5_for_f6):
                try:
                    with h5py.File(rh5_for_f6, "r") as f:
                        if ("left_hand_tactile_f6" in f
                                and "right_hand_tactile_f6" in f):
                            l = f["left_hand_tactile_f6"][:]    # [N, 5, 6]
                            r = f["right_hand_tactile_f6"][:]
                            self._cache_h5["tactile_f6"] = np.concatenate(
                                [l, r], axis=1).astype(np.float32, copy=False)
                except Exception:
                    pass

        # tactile_codes from a sibling HDF5 produced by tactile_vqvae.extract_codes
        # — only load when the model actually uses them.
        if getattr(self.config, "use_tactile_code", 0):
            codes_h5 = os.path.join(
                ep_dir,
                getattr(self.config, "vqvae_codes_h5_name", "tactile_codes.h5"))
            if os.path.isfile(codes_h5):
                try:
                    with h5py.File(codes_h5, "r") as f:
                        # codes_per_frame shape: [N, 2] (hand) or [N, 2, 5] (finger)
                        self._cache_codes = f["codes_per_frame"][:]
                        # Warn once per worker if alignment != historical, since
                        # post-train and real-world inference both use historical
                        # rolling windows; chunk alignment introduces a train/
                        # test signal mismatch.
                        align = f.attrs.get("alignment", "chunk")
                        if isinstance(align, bytes):
                            align = align.decode("utf-8", "ignore")
                        if (align != "historical"
                                and not getattr(self, "_warned_alignment", False)):
                            print(f"[Midtrain WARNING] tactile_codes.h5 was "
                                  f"extracted with alignment={align!r}; "
                                  f"post-train and inference both use "
                                  f"'historical'.  Re-run extract_codes with "
                                  f"--alignment historical for matched signals.",
                                  flush=True)
                            self._warned_alignment = True
                except Exception:
                    self._cache_codes = None

        # tactile_deform from raw.h5 — only load when the model actually uses it.
        if self.config.use_tactile_deform:
            rh5 = os.path.join(ep_dir, "raw.h5")
            if os.path.isfile(rh5):
                try:
                    with h5py.File(rh5, "r") as f:
                        if ("left_hand_tactile_deform" in f
                                and "right_hand_tactile_deform" in f):
                            l_def = f["left_hand_tactile_deform"][:]   # [N,5,H,W] uint8
                            r_def = f["right_hand_tactile_deform"][:]
                            self._cache_tac = np.concatenate([l_def, r_def], axis=1)
                except Exception:
                    pass

        head_crop = self._ep_crop_boxes[ep_idx]   # None or (y0,y1,x0,x1)
        for vname, fname, cb in [("head", "ego_view.mp4", head_crop),
                                 ("wrist_l", "left_wrist.mp4", None),
                                 ("wrist_r", "right_wrist.mp4", None)]:
            vp = os.path.join(ep_dir, fname)
            if os.path.isfile(vp):
                try:
                    self._cache_videos[vname] = self._decode_video(vp, crop_box=cb)
                except Exception:
                    pass

    def __getitem__(self, idx: int) -> Dict:
        ep_idx = int(np.searchsorted(self._cum_frames, idx, side="right"))
        frame_t = int(idx - (int(self._cum_frames[ep_idx - 1]) if ep_idx > 0 else 0))

        fb_frame = np.zeros((288, 384, 3), dtype=np.uint8)
        fb_def = np.zeros((10, 240, 240), dtype=np.uint8)
        fb_f6 = np.zeros((10, 6), dtype=np.float32)
        fb_codes_len = (2 * int(getattr(self.config, "n_fingers", 5))
                        if getattr(self.config, "tactile_code_per_finger", 0)
                        else 2)
        fb_codes = (np.zeros(fb_codes_len, dtype=np.int64)
                    if getattr(self.config, "use_tactile_code", 0) else None)
        fallback = {
            "head": fb_frame, "wrist_l": fb_frame.copy(), "wrist_r": fb_frame.copy(),
            "state": np.zeros(self.config.action_dim, dtype=np.float32),
            "tactile_codes": fb_codes,
            "action_chunk": np.zeros(
                (self.config.action_chunk, self.config.action_dim), dtype=np.float32),
            "tactile_f6": fb_f6, "tactile_deform": fb_def,
            "language": "",
            "flare_frames": [fb_frame.copy() for _ in range(self.n_flare_steps)],
        }

        if not hasattr(self, "_cache_ep_idx") or self._cache_ep_idx != ep_idx:
            self._load_episode_cache(ep_idx)

        if self._cache_h5 is None:
            return fallback
        try:
            state = self._cache_h5["states"][frame_t]
            action_chunk = self._cache_h5["action_chunks"][frame_t]
            language = self._cache_h5["language"]
        except (IndexError, KeyError):
            return fallback

        tactile_f6 = (self._cache_h5["tactile_f6"][frame_t]
                      if self._cache_h5.get("tactile_f6") is not None
                      else fb_f6)
        tactile_deform = (self._cache_tac[frame_t]
                          if (self._cache_tac is not None
                              and frame_t < len(self._cache_tac))
                          else fb_def)

        # Delayed tactile observations at frame_t + k where
        # k ~ Uniform({0, 4, 8, 12}).  Used as the "fresh tactile" input to
        # the tactile expert during cascaded training (simulates the
        # inference pattern: slow chunk anchored at t=0, tactile fast tick
        # fired at t+k).
        delay_offsets = getattr(self.config, "tactile_delay_offsets",
                                (0, 4, 8, 12))
        rng = np.random.default_rng()
        delay_k = int(rng.choice(delay_offsets))
        # Clamp to episode end
        ep_len_tac_f6 = (len(self._cache_h5["tactile_f6"])
                        if self._cache_h5.get("tactile_f6") is not None else 0)
        ep_len_tac_def = (len(self._cache_tac)
                          if self._cache_tac is not None else 0)
        delayed_t_f6  = min(frame_t + delay_k, max(ep_len_tac_f6 - 1, 0))
        delayed_t_def = min(frame_t + delay_k, max(ep_len_tac_def - 1, 0))
        tactile_f6_delayed = (self._cache_h5["tactile_f6"][delayed_t_f6]
                              if ep_len_tac_f6 > 0 else fb_f6)
        tactile_deform_delayed = (self._cache_tac[delayed_t_def]
                                  if ep_len_tac_def > 0 else fb_def)

        # tactile_codes at the delayed frame — flatten across (hand[, finger]).
        tactile_codes = None
        if getattr(self.config, "use_tactile_code", 0):
            if self._cache_codes is not None:
                t_clamp = min(delayed_t_f6, self._cache_codes.shape[0] - 1)
                tactile_codes = self._cache_codes[t_clamp].reshape(-1).astype(
                    np.int64, copy=False)
            else:
                # Fallback: zeros of the expected length.  Length is 2 (hand)
                # or 2*n_fingers (finger).  We don't know granularity here, so
                # use the configured codebook layout; default is 2.
                expected = (2 * int(getattr(self.config, "n_fingers", 5))
                            if getattr(self.config, "tactile_code_per_finger", 0)
                            else 2)
                tactile_codes = np.zeros(expected, dtype=np.int64)

        # Raw F6 history window for the embedded VQ-VAE (encoded on-the-fly in
        # the model).  Ends at the delayed/observed frame and is left-padded
        # with the episode's first frame — same alignment as extract_codes'
        # historical mode and the inference rolling buffer.
        tactile_f6_history = None
        if getattr(self.config, "use_tactile_vqvae", 0):
            W = int(getattr(self.config, "vqvae_window", 16))
            if ep_len_tac_f6 > 0:
                f6_series = self._cache_h5["tactile_f6"]          # [N, 10, 6]
                end = delayed_t_f6
                start = end - (W - 1)
                if start >= 0:
                    win = f6_series[start: end + 1]
                else:
                    head = np.repeat(f6_series[0:1], -start, axis=0)
                    win = np.concatenate([head, f6_series[: end + 1]], axis=0)
                tactile_f6_history = np.asarray(win, dtype=np.float32).reshape(W, 10, 6)
            else:
                tactile_f6_history = np.repeat(
                    np.asarray(fb_f6, dtype=np.float32).reshape(1, 10, 6), W, axis=0)

        head_frames    = self._cache_videos.get("head")    or []
        wrist_l_frames = self._cache_videos.get("wrist_l") or []
        wrist_r_frames = self._cache_videos.get("wrist_r") or []

        def _pick(seq, t):
            if seq and t < len(seq):
                return seq[t]
            return seq[-1] if seq else fb_frame

        head_frame    = _pick(head_frames,    frame_t)
        wrist_l_frame = _pick(wrist_l_frames, frame_t)
        wrist_r_frame = _pick(wrist_r_frames, frame_t)

        # FLARE futures from the head camera (matches the JSON pipeline:
        # flare images come from `input_image_slow`).
        flare_frames = []
        n_head = len(head_frames)
        for k in range(self.n_flare_steps):
            ft = frame_t + (k + 1) * self.flare_frame_stride
            if n_head > 0:
                flare_frames.append(head_frames[min(ft, n_head - 1)])
            else:
                flare_frames.append(head_frame.copy()
                                    if head_frame is not fb_frame
                                    else fb_frame.copy())

        return {
            "head": head_frame,
            "wrist_l": wrist_l_frame,
            "wrist_r": wrist_r_frame,
            "state": state,
            "action_chunk": action_chunk,
            "tactile_f6": tactile_f6,
            "tactile_deform": tactile_deform,
            "tactile_f6_delayed": tactile_f6_delayed,
            "tactile_deform_delayed": tactile_deform_delayed,
            "tactile_codes": tactile_codes,        # None or [K] int64
            "tactile_f6_history": tactile_f6_history,  # None or [W, 10, 6] raw
            "delay_k": delay_k,
            "language": language,
            "flare_frames": flare_frames,
        }

    def collate_fn(self, batch: List[Dict]) -> Dict:
        cfg = self.config
        B = len(batch)

        # ── action chunk (already 16×D from preprocessing) ──
        actions = np.stack([x["action_chunk"] for x in batch], axis=0).astype(np.float32)
        norm_actions = self._normalize(
            actions, self.action_mask, self.action_min, self.action_max)
        norm_actions = torch.tensor(norm_actions, dtype=torch.bfloat16)

        # ── flow-matching noise ──
        d = torch.distributions.Beta(
            torch.tensor(1.5, dtype=torch.float32),
            torch.tensor(1.0, dtype=torch.float32),
        )
        time = (d.sample((B,)) * 0.999 + 0.001).to(torch.bfloat16)
        t_ = time[:, None, None]
        noise = torch.randn_like(norm_actions)
        x_t = t_ * noise + (1 - t_) * norm_actions
        u_t = noise - norm_actions

        # ── tactile_f6 ──
        norm_tacf6 = None
        if cfg.use_tactile_vec:
            tacf6 = np.stack([x["tactile_f6"] for x in batch], axis=0).astype(np.float32)
            tacf6_flat = tacf6.reshape(B, -1)  # [B, 60]
            tacf6_flat = self._normalize(
                tacf6_flat, self.tacf6_mask, self.tacf6_min, self.tacf6_max)
            norm_tacf6 = torch.tensor(
                tacf6_flat.reshape(B, -1, 6), dtype=torch.bfloat16)

        # ── tactile_deform [B,10,H,W] uint8 → [B,10,1,H,W] float ──
        deforms_tensor = None
        if cfg.use_tactile_deform:
            deforms = np.stack(
                [x["tactile_deform"] for x in batch], axis=0)   # [B,10,H,W] uint8
            deforms_tensor = torch.tensor(
                deforms.astype(np.float32) / 255.0, dtype=torch.float32
            ).unsqueeze(2)   # [B,10,1,H,W]

        # ── delayed tactile (Paradigm C) ──
        norm_tacf6_delayed = None
        if cfg.use_tactile_vec:
            tacf6_d = np.stack([x["tactile_f6_delayed"] for x in batch],
                               axis=0).astype(np.float32)
            tacf6_d_flat = tacf6_d.reshape(B, -1)
            tacf6_d_flat = self._normalize(
                tacf6_d_flat, self.tacf6_mask, self.tacf6_min, self.tacf6_max)
            norm_tacf6_delayed = torch.tensor(
                tacf6_d_flat.reshape(B, -1, 6), dtype=torch.bfloat16)
        deforms_delayed_tensor = None
        if cfg.use_tactile_deform:
            deforms_d = np.stack(
                [x["tactile_deform_delayed"] for x in batch], axis=0)
            deforms_delayed_tensor = torch.tensor(
                deforms_d.astype(np.float32) / 255.0, dtype=torch.float32
            ).unsqueeze(2)
        delay_k_tensor = torch.tensor(
            [x["delay_k"] for x in batch], dtype=torch.long)

        # ── tactile-expert flow noise sampling (cascaded τ ∈ [0, τ_split]) ──
        # Same Beta(1.5, 1.0) prior as the action flow; the trainer rescales
        # time_r ∈ (0, 1] to (0, τ_split] before passing it into the tactile
        # expert's flow-matching loss.

        # Questa parte è importante perché introduce una nuova coppia tempo-rumore
        # indipendente da quella usata per la loss action.
        time_r = (d.sample((B,)) * 0.999 + 0.001).to(torch.bfloat16) # [B]
        eps_r = torch.randn_like(norm_actions) # [B,16,D]

        # ── state (with tracking-error noise when use_robot_state=1) ──
        state_raw_list = []
        if cfg.use_robot_state:
            for x in batch:
                s = np.array(x["state"], dtype=np.float32)
                s = add_tracking_error_noise(
                    s, self.te_mean, self.te_std, cfg.action_dim)
                ns = self._normalize(s, self.state_mask, self.state_min, self.state_max)
                state_raw_list.append(torch.tensor(ns, dtype=torch.bfloat16))
        state_raw = torch.stack(state_raw_list) if state_raw_list else None

        # ── chat template: [head (slow) | text | wrist_r, wrist_l (fast)] ──
        # Order matches gen_json_tac_deltabase_eef_bimanual_parallel.py:
        #   img_fast_list = [fast_r_img_paths[i], fast_l_img_paths[i]]
        all_input_ids, all_pixel_values, all_grid_thw = [], [], []
        n_slow_images = 1

        for x in batch:
            pil_head = PIL.Image.fromarray(x["head"])
            pil_wr = PIL.Image.fromarray(x["wrist_r"])
            pil_wl = PIL.Image.fromarray(x["wrist_l"])
            if self.image_size is not None:
                pil_head = pil_head.resize(self.image_size, PIL.Image.LANCZOS)
                pil_wr = pil_wr.resize(self.image_size, PIL.Image.LANCZOS)
                pil_wl = pil_wl.resize(self.image_size, PIL.Image.LANCZOS)

            content = [
                {"type": "image"},                                  # slow head
                {"type": "text", "text": x.get("language", "")},
                {"type": "image"},                                  # fast right wrist
                {"type": "image"},                                  # fast left wrist
            ]
            messages = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inp = self.processor(
                text=text, images=[pil_head, pil_wr, pil_wl],
                return_tensors="pt", padding=False,
            )
            all_input_ids.append(inp.input_ids[0])
            if "pixel_values" in inp and inp.pixel_values is not None:
                all_pixel_values.append(inp.pixel_values)
                all_grid_thw.append(inp.image_grid_thw)

        # ── FLARE future frames (head only) ──
        flare_pixel_values = None
        flare_grid_thw = None
        if self.n_flare_steps > 0:
            flare_pil_imgs = []
            for x in batch:
                for ff in x["flare_frames"]:
                    pil = PIL.Image.fromarray(ff)
                    if self.image_size is not None:
                        pil = pil.resize(self.image_size, PIL.Image.LANCZOS)
                    flare_pil_imgs.append(pil)
            flare_inp = self.processor.image_processor(
                flare_pil_imgs, return_tensors="pt")
            flare_pixel_values = flare_inp.pixel_values.to(torch.bfloat16)
            flare_grid_thw = flare_inp.image_grid_thw

        # ── padding + stacking (mirrors SftDataset.collate_fn) ──
        pad_id = self.processor.tokenizer.pad_token_id or 0
        max_len = max(ids.shape[0] for ids in all_input_ids)
        padded_ids, attention_ms = [], []
        for ids in all_input_ids:
            pad_len = max_len - ids.shape[0]
            padded_ids.append(F.pad(ids, (pad_len, 0), value=pad_id))
            attn = torch.ones(max_len, dtype=torch.long)
            if pad_len > 0:
                attn[:pad_len] = 0
            attention_ms.append(attn)

        input_ids = torch.stack(padded_ids)
        attention_mask = torch.stack(attention_ms)
        pixel_values = (torch.cat(all_pixel_values, dim=0)
                        if all_pixel_values else None)
        image_grid_thw = (torch.cat(all_grid_thw, dim=0)
                          if all_grid_thw else None)

        # Tactile VQ-VAE codes — stack to [B, K] int64 (K=2 hand or K=10 finger).
        tactile_codes_tensor = None
        tactile_f6_history_tensor = None
        if getattr(cfg, "use_tactile_vqvae", 0):
            hist_list = [x.get("tactile_f6_history") for x in batch]
            if all(h is not None for h in hist_list):
                tactile_f6_history_tensor = torch.from_numpy(
                    np.stack(hist_list, axis=0).astype(np.float32))  # [B, W, 10, 6]
        elif getattr(cfg, "use_tactile_code", 0):
            codes_list = [x.get("tactile_codes") for x in batch]
            if all(c is not None for c in codes_list):
                tactile_codes_tensor = torch.from_numpy(
                    np.stack(codes_list, axis=0).astype(np.int64))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "n_slow_images": n_slow_images,
            "noisy_actions": x_t,
            "target": u_t,
            "timesteps": time,
            "norm_actions": norm_actions,        # ground-truth A_demo
            "tactile_f6s": norm_tacf6,
            "tactile_deforms": deforms_tensor,
            "tactile_f6s_delayed": norm_tacf6_delayed,
            "tactile_deforms_delayed": deforms_delayed_tensor,
            "tactile_codes": tactile_codes_tensor,
            "tactile_f6_history": tactile_f6_history_tensor,  # [B, W, 10, 6] raw or None
            "delay_k": delay_k_tensor,
            "time_r": time_r,                    # τ_r ∈ (0,1] for tactile flow
            "eps_r": eps_r,                      # ε_r noise for tactile flow
            "state_raw": state_raw,
            "flare_pixel_values": flare_pixel_values,
            "flare_grid_thw": flare_grid_thw,
        }


def _sha256(path):
    import hashlib
    if not path or not os.path.isfile(path):
        return None
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def _risk_definition(args):
    if not getattr(args, "risk_stats", ""):
        return None
    from taptic.risk import RiskConfig
    return RiskConfig.from_args(args).to_dict()


def resume_state_dir(args):
    return getattr(args, "resume_state_dir", "") or os.path.join(args.output_dir, "resume_state")


def save_resume_state(accelerator, args, lr_scheduler, progress):
    """Restartable training (docs/TOUCH_MODULES.md): everything needed to continue this run where it stopped --
    model and optimizer (the DeepSpeed engine incl. its ZeRO partitions, via accelerate), RNG states, the LR
    scheduler, the step counters and the position in the epoch. Written to <dir>.tmp and swapped in, so a crash
    during the save keeps the previous state. All ranks must call it."""
    final = resume_state_dir(args)
    tmp, old = final + ".tmp", final + ".old"
    if accelerator.is_main_process:
        shutil.rmtree(tmp, ignore_errors=True)
    accelerator.wait_for_everyone()
    t0 = time.time()
    accelerator.save_state(tmp)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save(lr_scheduler.state_dict(), os.path.join(tmp, "lr_scheduler.pt"))
        with open(os.path.join(tmp, "progress.json"), "w") as f:
            json.dump(progress, f, indent=1)
        shutil.rmtree(old, ignore_errors=True)
        if os.path.isdir(final):
            os.rename(final, old)
        os.rename(tmp, final)
        shutil.rmtree(old, ignore_errors=True)
        accelerator.print(f"[resume] state saved at global_step {progress['global_step']} "
                          f"(epoch {progress['epoch']}, {progress['batches_done_in_epoch']} batches) in "
                          f"{time.time() - t0:.0f} s -> {final}")
    accelerator.wait_for_everyone()


def load_resume_state(accelerator, args, lr_scheduler, fingerprint):
    """Counterpart of save_resume_state. --resume_from_state auto: continue from <output_dir>/resume_state when it
    exists (None otherwise: fresh start); a directory: that state (must exist). The saved run must have the same
    structure (world size, batch, accumulation, epochs, steps, seed), else the data position would be wrong."""
    d = args.resume_from_state
    if d == "auto":
        d = resume_state_dir(args)
        if not os.path.isfile(os.path.join(d, "progress.json")):
            accelerator.print(f"[resume] no saved state in {d}: starting from the beginning")
            return None
    if not os.path.isfile(os.path.join(d, "progress.json")):
        raise FileNotFoundError(f"--resume_from_state {d}: no progress.json")
    with open(os.path.join(d, "progress.json")) as f:
        prog = json.load(f)
    saved = prog.get("fingerprint", {})
    bad = {k: (saved.get(k), v) for k, v in fingerprint.items() if saved.get(k) != v}
    if bad:
        raise ValueError(f"[resume] saved state {d} does not match this run (saved, now): {bad}")
    accelerator.load_state(d)
    lr_scheduler.load_state_dict(torch.load(os.path.join(d, "lr_scheduler.pt"), map_location="cpu"))
    accelerator.print(f"[resume] restored {d}: epoch {prog['epoch']}, global_step {prog['global_step']}, "
                      f"{prog['batches_done_in_epoch']} batches of that epoch done (saved {prog.get('saved_at')})")
    return prog


def save_checkpoint(model, processor, accelerator, args, epoch, global_step, stats_data):
    save_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}-{global_step}")

    if accelerator.is_main_process:
        ckpts = [f for f in os.listdir(args.output_dir) if f.startswith("checkpoint-")]
        if args.max_ckpts > 0 and len(ckpts) >= args.max_ckpts:
            oldest = min(ckpts, key=lambda f: os.path.getctime(os.path.join(args.output_dir, f)))
            shutil.rmtree(os.path.join(args.output_dir, oldest))

        os.makedirs(save_dir, exist_ok=True)

        sd = accelerator.get_state_dict(model)
        torch.save(sd, os.path.join(save_dir, "model.pt"))

        processor.save_pretrained(os.path.join(save_dir, "processor"))

        src_config = os.path.join(args.model_path, "config.json")
        if os.path.exists(src_config):
            shutil.copy(src_config, os.path.join(save_dir, "config.json"))

        with open(os.path.join(save_dir, "training_args.json"), "w") as f:
            json.dump({
                **({"data_format": "taptic",
                    "taptic_tactile_input": getattr(args, "taptic_tactile_input", "taxels"),
                    "tacf6_dim": _tactile_feature_dim(args),
                    "tactile_history_span_ms": float(getattr(args, "tactile_history_span_ms", 0.)),
                    "tactile_history_points": int(getattr(args, "tactile_history_points", 8)),
                    "tactile_self_handover": float(getattr(args, "tactile_self_handover", 0.)),
                    "tactile_force_contact_weight": float(getattr(args, "tactile_force_contact_weight", 1.)),
                    "tactile_force_tasks": list(getattr(args, "tactile_force_tasks", [])),
                    "force_contact_threshold": float(getattr(args, "force_contact_threshold", 2.)),
                    "tactile_event_weight": float(getattr(args, "tactile_event_weight", 1.)),
                    "tactile_event_types": list(getattr(args, "tactile_event_types", [])),
                    "tactile_event_window_ms": list(getattr(args, "tactile_event_window_ms", [200., 1000.])),
                    "tactile_event_tasks": list(getattr(args, "tactile_event_tasks", [])),
                    "tactile_event_labels": list(getattr(args, "tactile_event_labels", [])),
                    "tactile_event_jump_quantile": float(getattr(args, "tactile_event_jump_quantile", .97)),
                    "tactile_event_normalize": int(getattr(args, "tactile_event_normalize", 1)),
                    "taptic_exclude_sources": list(getattr(args, "taptic_exclude_sources", [])),
                    "taptic_vqvae_source_identity": getattr(args, "taptic_vqvae_source_identity", ""),
                    "tactile_async_touch": int(getattr(args, "tactile_async_touch", 0)),
                    "resume_state_every": int(getattr(args, "resume_state_every", 0)),
                    "resumed_from": getattr(args, "resumed_from", None),
                    "lerobot_root": getattr(args, "lerobot_root", ""),
                    "tactile_layout": ("thumb..little x 120 taxels x xyz"
                                       if getattr(args, "taptic_tactile_input", "taxels") == "taxels"
                                       else "thumb..little x SDK force xyz; VQ history native ~76 Hz"),
                    "tactile_delay_offsets": [0], "taptic_stats": args.taptic_stats,
                    "tactile_delay_ms": list(getattr(args, "tactile_delay_ms", [0])),
                    "risk_stats": getattr(args, "risk_stats", ""),
                    "risk_stats_sha256": _sha256(getattr(args, "risk_stats", "")),
                    "risk_definition": _risk_definition(args)}
                   if getattr(args, "data_format", "json") == "taptic" else {}),
                "variant": getattr(args, "variant", ""),
                "architecture": accelerator.unwrap_model(model).architecture_contract(),
                "use_risk_tokens": int(getattr(args, "use_risk_tokens", 0)),
                "risk_n_frequencies": getattr(args, "risk_n_frequencies", 8),
                "risk_future_head_weight": float(getattr(args, "risk_future_head_weight", 0.)),
                "risk_future_probe": int(getattr(args, "risk_future_probe", 0)),
                "risk_dropout": float(getattr(args, "risk_dropout", 0.)),
                "risk_oversample": float(getattr(args, "risk_oversample", 1.)),
                "risk_oversample_report": getattr(args, "risk_oversample_report", None),
                "resume_checkpoint": getattr(args, "resume_checkpoint", ""),
                "resume_load_report": getattr(args, "resume_load_report", None),
                "model_path": args.model_path,
                "action_dim": args.action_dim,
                "action_chunk": args.action_chunk,
                "use_robot_state": args.use_robot_state,
                "use_tactile_deform": args.use_tactile_deform,
                "use_tactile_vec": getattr(args, "use_tactile_vec", 0),
                "tactile_intermediate_size": getattr(args, "tactile_intermediate_size", 0),
                "training_stage": args.training_stage,
                "use_flare": args.use_flare,
                "n_flare_tokens_per_frame": args.n_flare_tokens_per_frame,
                "n_flare_steps": args.n_flare_steps,
                "flare_layer_index": args.flare_layer_index,
                "use_tactile_code": getattr(args, "use_tactile_code", 0),
                "vqvae_codebook_size": getattr(args, "vqvae_codebook_size", 64),
                "tactile_code_per_finger": getattr(args, "tactile_code_per_finger", 0),
                "use_tactile_vqvae": getattr(args, "use_tactile_vqvae", 0),
                "vqvae_config": getattr(args, "vqvae_config_dict", None),
                "paradigm": "cascaded",
                "cascaded_total_steps": getattr(args, "cascaded_total_steps", 10),
                "cascaded_split_step":  getattr(args, "cascaded_split_step", 6),
                "cascaded_split_mode": args.cascaded_split_mode,
                "cascaded_splits": args.cascaded_splits,
                "flow_time_precision": args.flow_time_precision,
                "tactile_stale_ms": args.tactile_stale_ms,
                "tactile_legacy_stale_ms": args.tactile_legacy_stale_ms,
                "tactile_gap_ms": args.tactile_gap_ms,
                "flare_frame_stride": getattr(args, "flare_frame_stride", 2),
            }, f, indent=2)

        with open(os.path.join(save_dir, "stats_data.json"), "w") as f:
            json.dump(stats_data, f, indent=2)

    accelerator.wait_for_everyone()
    logger.info(f"Checkpoint {epoch}-{global_step} saved.")


class TrainingMetrics:
    def __init__(self, device):
        self.n_step       = 0
        self.action_loss  = torch.tensor(0.0, device=device)
        self.tactile_loss = torch.tensor(0.0, device=device)
        self.flare_loss  = torch.tensor(0.0, device=device)
        self.total_loss   = torch.tensor(0.0, device=device)
        self.world_size   = dist.get_world_size()

    def update(self, total, action, tactile=0.0, flare=0.0):
        self.n_step += 1
        self.total_loss   += total.item() if torch.is_tensor(total) else total
        self.action_loss  += action.item() if torch.is_tensor(action) else action
        self.tactile_loss += tactile.item() if torch.is_tensor(tactile) else tactile
        self.flare_loss  += flare.item() if torch.is_tensor(flare) else flare

    def get_metric(self, reset=True):
        for t in [self.total_loss, self.action_loss, self.tactile_loss, self.flare_loss]:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        denom = self.world_size * max(self.n_step, 1)
        m = {
            "total_loss":   self.total_loss.item() / denom,
            "action_loss":  self.action_loss.item() / denom,
            "tactile_loss": self.tactile_loss.item() / denom,
            "flare_loss":  self.flare_loss.item() / denom,
        }
        if reset:
            self.n_step = 0
            for t in [self.total_loss, self.action_loss, self.tactile_loss, self.flare_loss]:
                t.fill_(0)
        return m


@torch.no_grad()
def run_validation(model, val_dataloader, accelerator, args,
                   is_stage1, use_flare, K, T_per_frame, flare_layer_idx):
    """Run validation and return averaged metrics (action + tactile + flare)."""
    model.eval()
    device = torch.cuda.current_device()
    val_act = torch.tensor(0.0, device=device)
    val_tac = torch.tensor(0.0, device=device)
    val_flare = torch.tensor(0.0, device=device)
    risk_sums = {k: torch.zeros(3, device=device) for k in ('head', 'probe')}
    risk_loss = {k: torch.tensor(0.0, device=device) for k in ('head', 'probe')}
    n_val = 0
    max_batches = getattr(args, "max_val_batches", 50)

    for i, batch in enumerate(val_dataloader):
        if i >= max_batches:
            break
        raw_model = accelerator.unwrap_model(model)
        handover = select_handover(args, device)
        # Same computation as a training step, without tactile dropout / FLARE.
        res = cascaded_losses(model, raw_model, batch, args, handover, is_stage1=is_stage1,
                              use_flare=use_flare, K=K, training=False)
        loss_act, loss_tac = res['loss_act'], res['loss_tac']

        val_act += loss_act.item()
        val_tac += (loss_tac.item() if torch.is_tensor(loss_tac) else loss_tac)
        for k in ('head', 'probe'):
            if res.get(f'risk_{k}_stats') is not None:
                risk_sums[k] += res[f'risk_{k}_stats'].to(device)
            if res.get(f'loss_risk_{k}') is not None:
                risk_loss[k] += res[f'loss_risk_{k}'].detach().float()
        n_val += 1

    for t in [val_act, val_tac, val_flare] + list(risk_sums.values()) + list(risk_loss.values()):
        if dist.is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    ws = dist.get_world_size() if dist.is_initialized() else 1
    denom = ws * max(n_val, 1)

    model.train()
    out = {
        "val/action_loss": val_act.item() / denom,
        "val/tactile_loss": val_tac.item() / denom,
    }
    raw = accelerator.unwrap_model(model)
    for k, flag in (('head', 'risk_future_head'), ('probe', 'risk_future_probe')):
        if getattr(raw, flag, False):
            sse_m, sse_p, n = (float(x) for x in risk_sums[k])
            out[f"val/risk_{k}_loss"] = risk_loss[k].item() / denom
            if n > 0:
                # skill > 0: better than repeating the current risk for every future slot
                out[f"val/risk_{k}_mse"] = sse_m / n
                out[f"val/risk_{k}_persist_mse"] = sse_p / n
                out[f"val/risk_{k}_skill"] = 1. - sse_m / max(sse_p, 1e-12)
    return out


def train(args):
    validate_split_config(args)
    validate_variant(args)
    if getattr(args, "data_format", "json") == "taptic":
        from qwen_vla.taptic_dataset import validate_config
        validate_config(args)
    # dispatch_batches=False — required because we hand-shard via
    # EpisodeGroupedSampler. With the default (None), accelerate may wrap our
    # batch_sampler in BatchSamplerShard and shard a SECOND time → each rank
    # sees 1/world_size² of the data (the "5925 → 93 steps" bug at ws=64).

    #mixed_precision="bf16"
    #Indica che il training utilizza precisione mista BF16 gestita da Accelerate.

    #BF16 usa 16 bit per rappresentare numeri floating-point.
    #Riduce memoria e costo computazionale rispetto a un training interamente in FP32, pur conservando un intervallo di esponenti ampio.
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_config=DataLoaderConfiguration(
            dispatch_batches=False, even_batches=True),
    )
    set_seed(args.seed)

    if accelerator.is_main_process:
        wandb.init(project=args.experiment_name, 
                   name=args.run_name,
                   config=args, 
                   dir=args.log_dir
                )

    accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.train_bsz_per_gpu
    accelerator.state.deepspeed_plugin.deepspeed_config["train_batch_size"] = (args.train_bsz_per_gpu * dist.get_world_size() * accelerator.gradient_accumulation_steps)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    tac_isize = args.tactile_intermediate_size if args.tactile_intermediate_size > 0 else None

    # Embedded VQ-VAE (on-the-fly F6 tokenizer).  Enabled explicitly
    # (--use_tactile_vqvae 1 [--vqvae_ckpt ...]) or AUTOMATICALLY when resuming a
    # checkpoint merged with an embedded VQ-VAE (training_args.json carries
    # use_tactile_vqvae=1 + vqvae_config; weights live in that model.pt).
    vqvae_blob = None
    vqvae_config = None
    if args.vqvae_ckpt and os.path.exists(args.vqvae_ckpt):
        vqvae_blob = torch.load(args.vqvae_ckpt, map_location="cpu", weights_only=False)
        vqvae_config = vqvae_blob["config"]

    resume_ta = {}
    if args.resume_checkpoint and os.path.isdir(args.resume_checkpoint):
        _ta_path = os.path.join(args.resume_checkpoint, "training_args.json")
        if os.path.exists(_ta_path):
            with open(_ta_path) as f:
                resume_ta = json.load(f)
    if resume_ta.get("use_tactile_vqvae") and not bool(getattr(args, "use_tactile_vqvae", 0)):
        args.use_tactile_vqvae = 1
        accelerator.print("[vqvae] embedded VQ-VAE auto-detected from resume checkpoint "
                          "(codes encoded on the fly; weights from model.pt).")
    if bool(getattr(args, "use_tactile_vqvae", 0)) and vqvae_config is None:
        vqvae_config = resume_ta.get("vqvae_config")

    if bool(getattr(args, "use_tactile_vqvae", 0)):
        if vqvae_config is None:
            raise FileNotFoundError(
                "--use_tactile_vqvae 1 needs either --vqvae_ckpt or a resume checkpoint "
                "whose training_args.json carries vqvae_config.")
        args.vqvae_codebook_size = int(vqvae_config["codebook_size"])
        args.vqvae_window = int(vqvae_config["window"])
        args.vqvae_config_dict = vqvae_config
        args.use_tactile_code = 1

    if args.data_format == "taptic":
        # Resume auto-detection must not silently enable the incompatible F6 VQ path.
        from qwen_vla.taptic_dataset import validate_vqvae_config
        validate_config(args)
        if vqvae_config is not None:
            validate_vqvae_config(vqvae_config)

    tacf6_dim = 6
    if args.data_format == "taptic":
        from qwen_vla.taptic_dataset import tactile_feature_dim
        tacf6_dim = tactile_feature_dim(args)

    # Creazione del modello VLA che verrà chiamato
    # Sta soltanto costruendo la struttura dei moduli e caricando i pesi di partenza.
    model = Qwen3VLVLAModel.from_pretrained_qwen3vl(
        pretrained_path=args.model_path,
        action_dim=args.action_dim,
        action_chunk=args.action_chunk,
        tacf6_dim=tacf6_dim,
        use_tactile_deform=bool(args.use_tactile_deform),
        use_robot_state=bool(args.use_robot_state),
        torch_dtype=torch.bfloat16,
        tactile_intermediate_size=tac_isize,
        n_flare_tokens_per_frame=args.n_flare_tokens_per_frame if args.use_flare else 0,
        n_flare_steps=args.n_flare_steps if args.use_flare else 0,
        flare_layer_index=args.flare_layer_index,
        use_tactile_code=bool(getattr(args, "use_tactile_code", 0)),
        vqvae_codebook_size=getattr(args, "vqvae_codebook_size", 64),
        use_tactile_vqvae=bool(getattr(args, "use_tactile_vqvae", 0)),
        vqvae_config=vqvae_config,
        use_risk_tokens=bool(getattr(args, "use_risk_tokens", 0)),
        risk_n_frequencies=getattr(args, "risk_n_frequencies", 8),
        risk_future_head=float(getattr(args, "risk_future_head_weight", 0.)) > 0,
        risk_future_probe=bool(getattr(args, "risk_future_probe", 0)),
    )
    if args.use_tactile_deform:
        model.load_deform_encoder_weights(args.deform_encoder_ckpt)
    # Inizializzazione dei pesi    
    model.initialize_vla_weights()
    if model.use_risk_tokens:
        accelerator.print(f"[risk] risk tokens ON: initialized {len(model.initialize_risk_weights())} "
                          f"new risk-embedder parameters (Action r_S + Tactile r_F)")
    if vqvae_blob is not None:
        model.load_tactile_vqvae(vqvae_blob)
    if args.use_flare:
        accelerator.print(
            f"Flare alignment: {args.n_flare_steps} steps × {args.n_flare_tokens_per_frame} tok/frame "
            f"= {model.n_flare_tokens} total tokens, layer_index={args.flare_layer_index}"
        )

    is_stage1 = (args.training_stage == 1)
    has_any_tactile = bool(args.use_tactile_vec or args.use_tactile_deform)
    freeze_tactile = not has_any_tactile

    if not args.resume_checkpoint:
        named_params = dict(model.named_parameters())
        for name, param in model.named_parameters():
            if "_action" in name:
                base = name.replace("_action", "")
                if base in named_params:
                    param.data.copy_(named_params[base].data)
        accelerator.print("Action expert initialized from latent expert.")

    # Se è stato specificato un checkpoint da cui fare resume
    # ripristina parametri del precedente pretraining/midtraining
    if args.resume_checkpoint:
        ckpt_path = args.resume_checkpoint
        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path, "model.pt")
        resume_sd = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in resume_sd:
            resume_sd = resume_sd["state_dict"]
        # Explicit architecture contract: shape-mismatched keys are still skipped
        # (e.g. tactile MLP from pretrain) but listed; risk weights are never
        # silently dropped and newly introduced risk parameters are logged.
        load_report = load_with_contract(model, resume_sd, architecture_from_training_args(resume_ta),
                                         log=accelerator.print)
        args.resume_load_report = {k: v for k, v in load_report.items()
                                   if k in ('new_risk_parameters_initialized', 'shape_mismatch_skipped',
                                            'checkpoint_architecture')}
        accelerator.print(f"Resumed: missing={len(load_report['missing'])}, "
                          f"unexpected={len(load_report['unexpected'])}")

    resumed_tactile = bool(args.resume_checkpoint) and args.resume_source == "midtrain"
    if not freeze_tactile and not resumed_tactile:
        # Tactile expert is a velocity predictor under cascaded flow matching,
        # so the head must start non-trivial (no final-layer zero-init here).
        reinit_tactile_expert(model)
        accelerator.print("Tactile expert re-initialized (resume_source=pretrain or no resume).")
    elif resumed_tactile:
        accelerator.print("Tactile expert weights kept from resumed midtrain checkpoint.")

    for name, param in model.named_parameters(): #funzione che restituisce coppie nome, parametro
        if (name.startswith("visual") or name.startswith("deform_encoder")
                or name.startswith("tactile_vqvae")):
            # congelano visual encoder, deform encoder e VQ-VAE
            param.requires_grad = False   # embedded VQ-VAE stays frozen
        elif "_tactile" in name or "final_layer_tactile" in name:
            param.requires_grad = (not freeze_tactile)
        else:
            param.requires_grad = True

    if getattr(model, "tactile_vqvae", None) is not None:
        model.tactile_vqvae.float().eval()
        for _b in ("tacf6_vqvae_min", "tacf6_vqvae_max"):
            setattr(model, _b, getattr(model, _b).float())

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Model: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")

    #Gli autori creano due gruppi di parametri:
    #- Parametri ai quali applicare weight decay.
    #- Parametri ai quali non applicarlo, come bias e determinati pesi di normalizzazione.
    #L'optimizer aggiornerà soltanto i parametri inseriti nei gruppi, che il codice seleziona tra quelli con requires_grad=True.
    #Questa parte è comune a molti progetti PyTorch, ma è indispensabile per comprendere cosa viene effettivamente addestrato
    no_decay = ["bias", "norm.weight", "q_norm.weight", "k_norm.weight"]
    param_groups = [
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate)

    # Loading del dataset che può avere formati differenti
    if getattr(args, "data_format", "json") == "lerobot":
        from qwen_vla.lerobot_dataset import TRexLeRobotDataset
        dataset = TRexLeRobotDataset(args, processor, accelerator)
    elif args.data_format == "taptic":
        from qwen_vla.taptic_dataset import TRexTapticDataset
        dataset = TRexTapticDataset(args, processor, accelerator)
    else:
        dataset = MidtrainTacFlareDataset(args, processor, accelerator)

    # ── Episode-cached dataloader knobs ──
    # persistent_workers=True keeps each worker's per-episode cache alive
    # across epochs (saves the ~30–60 s "first batch of epoch" stall after
    # epoch 0). prefetch_factor=4 buffers 4 batches per worker so the GPU
    # has work in flight while another worker stalls on a new-episode load
    # (bulk-read of raw.h5 tactile_deform + 3 mp4 sequential decodes).
    n_workers_train = getattr(args, "num_workers", 4)
    n_workers_val = max(1, n_workers_train // 2)

    # Val dataloader: NO custom sampler — let accelerate add its own
    # DistributedSampler when we prepare it. Keeping the custom sampler
    # here would also trigger the double-shard interaction.

    val_dataloader = None
    if getattr(args, "val_ratio", 0) > 0:
        val_dataset = dataset.create_val_split(
            val_ratio=args.val_ratio, seed=args.seed)
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.train_bsz_per_gpu, shuffle=False,
            drop_last=True, collate_fn=val_dataset.collate_fn,
            num_workers=n_workers_val, pin_memory=True,
            persistent_workers=(n_workers_val > 0),
            prefetch_factor=4 if n_workers_val > 0 else None,
        )

    if float(getattr(args, "risk_oversample", 1.0)) > 1:
        # High risk = r_S in the gate's tau = 0.6 regime unless a threshold is given.
        from taptic.risk import RiskStats
        thr = args.risk_oversample_threshold
        if thr is None:
            gates = RiskStats.load(args.risk_stats).gate_thresholds
            if gates is None:
                raise ValueError('risk statistics carry no gate thresholds: pass --risk_oversample_threshold')
            thr = float(gates[1])
        r_rows, v_rows = dataset.anchor_risk(log=accelerator.print)
        high = v_rows & (r_rows >= thr)
        train_sampler = RiskOversampledEpisodeSampler(
            dataset, high, int(args.risk_oversample), shuffle=True, seed=args.seed, drop_last=True)
        accelerator.print(
            f"[risk] high-risk oversampling x{int(args.risk_oversample)}: r_S >= {thr:.4f} on "
            f"{int(high.sum())}/{len(high)} rows ({high.mean():.1%}; risk-valid {v_rows.mean():.1%}) "
            f"-> {train_sampler.high_share:.1%} of each epoch; epoch length unchanged")
        args.risk_oversample_report = {'threshold': thr, 'factor': int(args.risk_oversample),
                                       'high_rows': int(high.sum()), 'rows': int(len(high)),
                                       'epoch_high_share': float(train_sampler.high_share)}
    else:
        train_sampler = EpisodeGroupedSampler(
            dataset, shuffle=True, seed=args.seed, drop_last=True)
    # 3. Richiedere i campioni e combinarli in batch tramite collate_fn
    dataloader = DataLoader(
        dataset, # 1. Leggere e costruire i singoli campioni
        batch_size=args.train_bsz_per_gpu,
        sampler=train_sampler, # 2. Scegliere gli indici e l'ordine dei campioni
        drop_last=True,
        collate_fn=dataset.collate_fn,
        num_workers=n_workers_train, pin_memory=True,
        persistent_workers=(n_workers_train > 0),
        prefetch_factor=4 if n_workers_train > 0 else None,
    )

    # Compute steps from ground truth so the LR scheduler doesn't get a
    # bogus total. EpisodeGroupedSampler already shards — so per-rank
    # samples = total // world_size, batches = that // batch_size_per_gpu.
    world_size = accelerator.num_processes
    samples_per_rank = len(dataset) // world_size
    steps_per_epoch = samples_per_rank // args.train_bsz_per_gpu
    num_training_steps = (
        steps_per_epoch * args.n_epochs
        // accelerator.gradient_accumulation_steps
    )
    accelerator.print(
        f"[Midtrain] world_size={world_size}, samples_per_rank={samples_per_rank}, "
        f"steps_per_epoch={steps_per_epoch}, total_steps={num_training_steps}")

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_rates * num_training_steps),
        num_training_steps=num_training_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    # IMPORTANT: do NOT prepare the training dataloader. EpisodeGroupedSampler
    # already shards by rank; accelerator.prepare() would wrap the
    # batch_sampler in BatchSamplerShard and shard a SECOND time, giving each
    # rank 1/world_size² of the data. We move train batches to device
    # manually inside the loop.
    if val_dataloader is not None:
        model, optimizer, val_dataloader = accelerator.prepare(
            model, optimizer, val_dataloader)
    else:
        model, optimizer = accelerator.prepare(model, optimizer)

    metric = TrainingMetrics(device=torch.cuda.current_device())
    global_step = 0
    T_per_frame = args.n_flare_tokens_per_frame
    S_steps = args.n_flare_steps
    K = T_per_frame * S_steps  # total flare tokens
    use_flare = bool(args.use_flare and K > 0)
    flare_layer_idx = args.flare_layer_index
    device = accelerator.device
    model.train()

    # Restartable training (docs/TOUCH_MODULES.md). resume_every: micro-steps between saved states, rounded up to
    # whole optimizer steps so no partial gradient accumulation is ever saved.
    start_epoch, skip_batches = 0, 0
    resume_every = int(getattr(args, "resume_state_every", 0) or 0)
    if resume_every:
        ga = accelerator.gradient_accumulation_steps
        resume_every = -(-resume_every // ga) * ga
    fingerprint = {"world_size": world_size, "steps_per_epoch": steps_per_epoch, "n_epochs": args.n_epochs,
                   "train_bsz_per_gpu": args.train_bsz_per_gpu, "seed": args.seed,
                   "gradient_accumulation_steps": accelerator.gradient_accumulation_steps,
                   "num_training_steps": num_training_steps}
    if getattr(args, "resume_from_state", ""):
        prog = load_resume_state(accelerator, args, lr_scheduler, fingerprint)
        if prog is not None:
            global_step, start_epoch, skip_batches = (int(prog["global_step"]), int(prog["epoch"]),
                                                      int(prog["batches_done_in_epoch"]))
            if skip_batches >= steps_per_epoch:
                start_epoch, skip_batches = start_epoch + 1, 0
            args.resumed_from = {"global_step": global_step, "epoch": start_epoch, "skip_batches": skip_batches,
                                 "saved_at": prog.get("saved_at")}

    for epoch in range(start_epoch, args.n_epochs):
        # Required for proper shuffling each epoch when the dataloader is
        # not prepared by accelerate.
        train_sampler.set_epoch(epoch)
        n_skip = skip_batches if epoch == start_epoch else 0
        train_sampler.skip_samples = n_skip * args.train_bsz_per_gpu
        batches_done = n_skip

        from tqdm import tqdm
        it = (tqdm(dataloader, total=steps_per_epoch, initial=n_skip, desc=f"Epoch {epoch}")
              if accelerator.is_main_process else dataloader)

        for batch in it:
            # Manually move tensors to device since the train dataloader is
            # not prepared by accelerate (see comment above).
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            raw_model = accelerator.unwrap_model(model)
            handover = select_handover(args, device)

            # Vision encoder, slow/fast split, M-RoPE, FLARE queries, (risk_S),
            # action loss, no-grad handover cache and tactile loss: shared with
            # run_validation (see cascaded_losses for the step-by-step comments).
            res = cascaded_losses(
                model, raw_model, batch, args, handover, is_stage1=is_stage1, use_flare=use_flare,
                K=K, training=True,
                merge_default=getattr(processor.image_processor, "merge_size", 2))
            loss_act, loss_tac = res['loss_act'], res['loss_tac']
            outputs, hidden = res['outputs'], res['hidden']
            L_slow, B = res['L_slow'], res['B']
            r_target_norm = res['r_target_norm']

            # FLARE è un obiettivo ausiliario di previsione di rappresentazioni visive future.
            # Il modello cerca di predire feature visive future
            # che vengono calcolate da un vision encoder congelato.
            loss_flare = 0.0
            if use_flare and batch["flare_pixel_values"] is not None:
                # Extract flare hidden states from intermediate layer or last layer
                if flare_layer_idx == -1:
                    flare_source = hidden  # last layer (already normed)
                else:
                    # outputs.hidden_states is a tuple: (embed, layer0, layer1, ..., layerN)
                    # flare_layer_idx can be negative (e.g. -7 for ~3/4 depth)
                    all_hs = outputs.hidden_states
                    # Skip index 0 (embedding), so layer indices are 1..N
                    n_layers = len(all_hs) - 1  # exclude embedding

                    # Il codice permette di scegliere da quale layer del Transformer
                    # prendere le rappresentazioni.
                    # Se flare_layer_idx=-1, utilizza l'output finale.
                    # Se scegliamo, per esempio, un layer intermedio, recupera gli hidden states corrispondenti
                    # da outputs.hidden_states.
                    # Questo ci permette di applicare la supervisione visiva a diverse profondità della rete.
                    if flare_layer_idx < 0:
                        layer_i = n_layers + flare_layer_idx  # e.g. 28 + (-7) = 21
                    else:
                        layer_i = flare_layer_idx
                    layer_i = max(0, min(layer_i, n_layers - 1))
                    flare_source = all_hs[layer_i + 1]  # +1 to skip embedding

                flare_hidden = flare_source[:, L_slow: L_slow + K, :]  # [B, K, H]
                flare_pred = raw_model.flare_proj(flare_hidden)  # [B, K, H]

                # COSTRUZIONE DEL TARGET FLARE
                # Il dataset ha già preparato le immagini future e le ha inserite nel batch
                f_pv = batch["flare_pixel_values"].to(device=flare_pred.device, dtype=flare_pred.dtype)
                f_thw = batch["flare_grid_thw"].to(device=flare_pred.device)

                with torch.no_grad():
                    # Questa volta il vision encoder viene eseguito sui frame futuri
                    vit_out = raw_model.visual(f_pv, grid_thw=f_thw)
                    features = vit_out[0] if isinstance(vit_out, (tuple, list)) else vit_out

                    # Adaptive pool each frame to T_per_frame tokens
                    merge = getattr(raw_model.visual, "spatial_merge_size", 2)
                    frame_feats = []
                    offset = 0
                    for g in f_thw:
                        n_tok = int(g[0] * (g[1] // merge) * (g[2] // merge))
                        frame_tokens = features[offset: offset + n_tok]  # [n_tok, H]

                        # Adaptive average pool: [n_tok, H] → [T_per_frame, H]
                        # Perché serve il pooling?
                        # Un frame, dopo il vision encoder, può avere molti token visivi.
                        # Ma il modello utilizza soltanto quattro query FLARE per frame.
                        # Il codice deve quindi ridurre la rappresentazione del frame a quattro vettori.
                        # [N_visual, H] -> [4, H]
                        # con 8 frame futuri --> [32,H]
                        pooled = F.adaptive_avg_pool1d(
                            frame_tokens.unsqueeze(0).permute(0, 2, 1),  # [1, H, n_tok]
                            T_per_frame,
                        ).permute(0, 2, 1).squeeze(0)  # [T_per_frame, H]
                        frame_feats.append(pooled)
                        offset += n_tok
                    # frame_feats: list of B*S_steps tensors, each [T_per_frame, H]
                    # Reshape to [B, S_steps * T_per_frame, H] = [B, 32, H]
                    flare_targets = torch.stack(frame_feats).view(B, K, -1)

                # Cosine similarity tra predizione e target FLARE
                # Cosine perché così la loss si concentra sull'allineamento delle
                # rappresentazioni, non sulla corrispondenza esatta di ogni componente numerica.
                pred_norm = F.normalize(flare_pred, dim=-1)
                tgt_norm = F.normalize(flare_targets.detach(), dim=-1)
                loss_flare = (1.0 - (pred_norm * tgt_norm).sum(dim=-1)).mean()

            loss = loss_act
            if torch.is_tensor(loss_tac):
                loss = loss + args.cascaded_loss_weight * loss_tac
            if torch.is_tensor(loss_flare):
                loss = loss + args.flare_loss_weight * loss_flare
            if res.get('loss_risk_head') is not None:
                loss = loss + args.risk_future_head_weight * res['loss_risk_head']


            # BACKWARD: calcola e accumula gradienti.
            metric.update(loss, loss_act, res.get('loss_tac_plain', loss_tac), loss_flare)
            # The future-risk probe sees detached features: its loss only trains the probe, and it is
            # kept out of the logged total so totals stay comparable across variants.
            accelerator.backward(loss + res['loss_risk_probe'] if res.get('loss_risk_probe') is not None
                                 else loss)

            # La condizione verifica se abbiamo accumulato il numero richiesto di microbatch.
            # Se gradient_accumulation_steps=4, il codice entra nel blocco agli step 4, 8, 12 e così via.
            if (global_step + 1) % accelerator.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    # Limita la norma dei gradienti quando supera la soglia impostata.
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # Aggiornamento dei pesi
                optimizer.step()
                # Learning rate scheduler
                lr_scheduler.step()
                # Azzeramento dei gradienti
                optimizer.zero_grad()

                m = metric.get_metric()
                if accelerator.is_main_process:
                    lr_now = lr_scheduler.get_last_lr()[0]
                    if hasattr(it, "set_postfix"):
                        it.set_postfix(epoch=epoch, step=global_step,
                                       loss=f"{m['total_loss']:.6f}",
                                       act=f"{m['action_loss']:.6f}",
                                       tac=f"{m['tactile_loss']:.6f}",
                                       fut=f"{m['flare_loss']:.6f}",
                                       lr=f"{lr_now:.2e}")
                    log_dict = {
                        "total_loss":   m["total_loss"],
                        "action_loss":  m["action_loss"],
                        "tactile_loss": m["tactile_loss"],
                        "flare_loss":  m["flare_loss"],
                        "lr": lr_now,
                        "flow/tau_split": handover.tau,
                        **({"tactile/valid_frac": float(batch["tactile_loss_mask"].float().mean())}
                           if batch.get("tactile_loss_mask") is not None else {}),
                        "flow/action_steps": handover.action_steps,
                        "flow/tactile_steps": handover.tactile_steps,
                    }
                    if r_target_norm is not None:
                        log_dict["refine/r_target_norm"] = float(r_target_norm)
                    log_dict.update(batch_diagnostics(batch))
                    log_dict["tactile/dropout_fired"] = float(res['tactile_dropped'])
                    log_dict["tactile/self_handover_frac"] = float(res.get('self_handover_frac') or 0.)
                    if res.get('force_contact_frac') is not None:
                        log_dict["tactile/force_contact_frac"] = res['force_contact_frac']
                    if res.get('touch_event_frac') is not None:
                        log_dict["tactile/touch_event_frac"] = res['touch_event_frac']
                    if torch.is_tensor(loss_tac):
                        log_dict["tactile/loss_weighted"] = float(loss_tac.detach())
                    for k in ('head', 'probe'):
                        if res.get(f'loss_risk_{k}') is not None:
                            log_dict[f"risk_future/{k}_loss"] = float(res[f'loss_risk_{k}'].detach())
                    wandb.log(log_dict, step=global_step)

            # ── Validation ──
            if (val_dataloader is not None
                    and getattr(args, "val_freq", 0) > 0
                    and (global_step + 1) % args.val_freq == 0):
                val_m = run_validation(
                    model, val_dataloader, accelerator, args,
                    is_stage1, use_flare, K, T_per_frame, flare_layer_idx)
                if accelerator.is_main_process:
                    accelerator.print(
                        f"  [Val step={global_step}] "
                        f"act={val_m['val/action_loss']:.6f} "
                        f"tac={val_m['val/tactile_loss']:.6f}"
                        + "".join(f" risk_{k}_skill={val_m[f'val/risk_{k}_skill']:.4f}"
                                  for k in ('head', 'probe') if f'val/risk_{k}_skill' in val_m))
                    wandb.log(val_m, step=global_step)

            global_step += 1
            batches_done += 1
            if resume_every and global_step % resume_every == 0:
                save_resume_state(accelerator, args, lr_scheduler, {
                    "epoch": epoch, "global_step": global_step, "batches_done_in_epoch": batches_done,
                    "fingerprint": fingerprint, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            if args.max_train_steps and global_step >= args.max_train_steps:
                break

        if args.max_train_steps and global_step >= args.max_train_steps:
            # Probe mode (memory/throughput check): stop without writing checkpoints.
            if torch.cuda.is_available():
                accelerator.print(f"[probe] {global_step} steps done; peak CUDA memory "
                                  f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB on this rank")
            break

        if (epoch + 1) % args.save_freq == 0 or epoch == args.n_epochs - 1:
            accelerator.wait_for_everyone()
            save_checkpoint(model, processor, accelerator, args,
                            epoch, global_step, dataset.stats_data)
        if resume_every and epoch < args.n_epochs - 1 and global_step % resume_every != 0:
            save_resume_state(accelerator, args, lr_scheduler, {
                "epoch": epoch, "global_step": global_step, "batches_done_in_epoch": batches_done,
                "fingerprint": fingerprint, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")})

    # Run complete: the restart state (~30 GB) is no longer needed (the final checkpoint has the weights).
    if (resume_every and not (args.max_train_steps and global_step >= args.max_train_steps)
            and accelerator.is_main_process):
        shutil.rmtree(resume_state_dir(args), ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", type=str, default="qwen3vl_midtrain_tac_flare")
    parser.add_argument("--run_name", type=str, default="run_1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="",
                        help="Directory containing batch_*/pretrain_manifest.json "
                             "(produced by gen_pretrain_mecka_parallel.py). "
                             "Required for --data_format json.")
    parser.add_argument("--data_path", type=str, default="",
                        help="(Unused; kept for backwards compatibility.)")
    parser.add_argument("--data_format", type=str, default="json", choices=["json", "lerobot", "taptic"])
    parser.add_argument("--lerobot_root", type=str, default="",
                        help="LeRobot dataset dir (required when --data_format lerobot).")
    parser.add_argument("--taptic_stats", type=str, default="", help="Train-only statistics from python -m taptic.inspect_dataset")
    parser.add_argument("--taptic_tactile_input", type=str, default="taxels", choices=["taxels", "sdk_force"],
                        help="taxels: 5x360 raw taxels, no VQ. sdk_force: 5x3 SDK force + native ~76 Hz "
                             "history for the TAPTIC VQ-VAE; legacy 10 Hz episodes masked from tactile loss.")
    parser.add_argument("--lerobot_repo_id", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--max_ckpts", type=int, default=10)

    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--max_train_steps", type=int, default=0,
                        help="Probe: stop after this many micro-steps, no checkpoint (0 = off).")
    parser.add_argument("--save_freq", type=int, default=50)
    parser.add_argument("--train_bsz_per_gpu", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.0)
    parser.add_argument("--warmup_rates", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--action_dim", type=int, default=62)
    parser.add_argument("--action_chunk", type=int, default=16)
    parser.add_argument("--use_robot_state", type=int, default=0)
    parser.add_argument("--use_tactile_vec", type=int, default=0)
    parser.add_argument("--use_tactile_deform", type=int, default=1)
    parser.add_argument("--deform_encoder_ckpt", type=str, default="")
    parser.add_argument("--tactile_intermediate_size", type=int, default=0)
    parser.add_argument("--training_stage", type=int, default=2, choices=[1, 2])
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--resume_source", type=str, default="pretrain",
                        choices=["pretrain", "midtrain"],
                        help="'pretrain': resumed ckpt did not train tactile (re-init); "
                             "'midtrain': resumed ckpt already trained tactile (keep).")
    parser.add_argument("--tactile_loss_weight", type=float, default=1.0)
    parser.add_argument("--image_size", type=int, nargs=2, default=None, metavar=("W", "H"))

    parser.add_argument("--tactile_delay_offsets", type=int, nargs="+",
                        default=[0, 4, 8, 12],
                        help="Delay offsets (frames) sampled uniformly when reading "
                             "tactile observations for the tactile expert.")
    parser.add_argument("--tactile_delay_ms", type=int, nargs="+", default=[0],
                        help="TAPTIC physical fast-tick offsets F-S (ms), sampled uniformly per "
                             "sample: tactile force/VQ history/r_F at F, Action context/r_S at S, "
                             "elapsed target slots masked. [0] = baseline alignment.")

    # Contact risk (OURS) — see taptic/risk.py and docs/taptic_ours/REPORT.md.
    parser.add_argument("--variant", default="", choices=["", "vanilla", "multisplit", "ours"],
                        help="Optional guard: refuse flags inconsistent with the named variant.")
    parser.add_argument("--risk_stats", type=str, default="",
                        help="Train-only risk statistics (python -m taptic.risk fit). Enables r_S/r_F "
                             "features in the batch (logging, gate, risk tokens).")
    parser.add_argument("--use_risk_tokens", type=int, default=0,
                        help="1: learned risk token in the Action (r_S) and Tactile (r_F) streams.")
    parser.add_argument("--risk_n_frequencies", type=int, default=8)
    parser.add_argument("--risk_coefficients", type=float, nargs=3, default=[0.25, 0.50, 0.25],
                        help="(a, b, c) in r = clip(a*M + b*R + c*M*V, 0, 1); fixed, not learned.")
    parser.add_argument("--risk_rate_window_ms", type=int, default=100)
    parser.add_argument("--risk_velocity_window_ms", type=int, default=100)
    parser.add_argument("--risk_joint_source", default="sidecar", choices=["sidecar", "state10hz"])
    parser.add_argument("--risk_rate_mode", default="vector", choices=["vector", "positive"],
                        help="R100: vector force variation (increases, decreases, direction changes; "
                             "default) or positive |F| increases only (previous definition, ablation).")
    parser.add_argument("--risk_future_head_weight", type=float, default=0.0,
                        help=">0: linear head on the Tactile expert's chunk tokens predicting the causal risk "
                             "r(t_k) at each future slot time (aux MSE with this weight).")
    parser.add_argument("--risk_future_probe", type=int, default=0,
                        help="1: linear probe on DETACHED Action-expert chunk tokens, same target "
                             "(measures future risk visible to the slow stream; never changes the model).")
    parser.add_argument("--risk_dropout", type=float, default=0.0,
                        help="Training only: mark a valid r_S / r_F missing with this probability (each "
                             "independently), so 'missing risk' is not a proxy for the legacy tasks.")
    parser.add_argument("--risk_oversample", type=float, default=1.0,
                        help=">1 (integer): each epoch, high-risk rows appear this many times and as many "
                             "other rows are dropped (epoch length unchanged).")
    parser.add_argument("--risk_oversample_threshold", type=float, default=None,
                        help="High risk = r_S >= this (default: the risk statistics' upper gate "
                             "threshold, i.e. the tau = 0.6 regime).")

    # Cascaded flow matching (see train_qwen3vl_flare.py for details).
    parser.add_argument("--cascaded_split_mode", choices=["fixed", "multi"], default="fixed")
    parser.add_argument("--cascaded_splits", type=float, nargs="+", default=[0.2, 0.4, 0.6])
    parser.add_argument("--flow_time_precision", choices=["fp32", "legacy_bf16"], default="fp32")
    parser.add_argument("--tactile_stale_ms", type=float, default=50.)
    parser.add_argument("--tactile_legacy_stale_ms", type=float, default=150.)
    parser.add_argument("--tactile_gap_ms", type=float, default=50.)
    parser.add_argument("--cascaded_total_steps", type=int, default=10)
    parser.add_argument("--cascaded_split_step",  type=int, default=6)
    parser.add_argument("--cascaded_tactile_dropout", type=float, default=0.1)
    parser.add_argument("--cascaded_loss_weight", type=float, default=1.0)
    # trex-touch: let the Tactile expert use touch (all defaults = baseline, bitwise).
    parser.add_argument("--tactile_history_span_ms", type=float, default=0.,
                        help=">0 (sdk_force): each per-finger tactile vector token carries "
                             "--tactile_history_points force samples spread evenly over this span (oldest "
                             "first, last = current sample). 0 = current sample only (baseline).")
    parser.add_argument("--tactile_history_points", type=int, default=8)
    parser.add_argument("--tactile_self_handover", type=float, default=0.,
                        help="Training only: fraction of samples whose Tactile-expert pair starts at the "
                             "Action expert's own state x_split (straight path x_split -> demo) instead of "
                             "the ground-truth noisy point; inference always continues from x_split. 0 = baseline.")
    parser.add_argument("--tactile_force_contact_weight", type=float, default=1.,
                        help="Training only: weight of the hand dims of the tactile loss on samples of "
                             "--tactile_force_tasks with fingertip |F| > --force_contact_threshold at the "
                             "fast tick. 1 = baseline.")
    parser.add_argument("--tactile_force_tasks", nargs="*",
                        default=["dosing_pipette", "insert_credit_card", "insert_game_object_cylinder",
                                 "unscrew_the_bottle", "grasp_plastic_glass"])
    parser.add_argument("--force_contact_threshold", type=float, default=2.0,
                        help="Raw SDK calc_force units (same threshold as the offline touch probes).")
    parser.add_argument("--tactile_event_weight", type=float, default=1.,
                        help="Training only: weight of the tactile loss (all action dims) on samples with a touch "
                             "event (taptic/touch_events.py) in the window, for --tactile_event_tasks with valid "
                             "touch. 1 = baseline.")
    parser.add_argument("--tactile_event_types", nargs="*", default=["grasp_onset", "resistance"],
                        help="Any of grasp_onset resistance release label.")
    parser.add_argument("--tactile_event_window_ms", nargs=2, type=float, default=[200., 1000.],
                        metavar=("PRE", "POST"), help="Sample t is an event sample if an event lies in "
                                                      "[t - POST, t + PRE].")
    parser.add_argument("--tactile_event_tasks", nargs="*",
                        default=["dosing_pipette", "insert_credit_card", "insert_game_object_cylinder",
                                 "unscrew_the_bottle", "grasp_plastic_glass"])
    parser.add_argument("--tactile_event_labels", nargs="*", default=["insert", "squeeze", "unscrew", "push"],
                        help="Subtask names whose start is a 'label' event.")
    parser.add_argument("--tactile_event_jump_quantile", type=float, default=0.97,
                        help="Per-task 'resistance' threshold: this quantile of the strongest-finger force rise "
                             "over 0.2 s during contact, training episodes only.")
    parser.add_argument("--tactile_event_normalize", type=int, default=1,
                        help="1: rescale the per-sample event weights to mean 1 in each batch.")
    parser.add_argument("--taptic_exclude_sources", nargs="*", default=[],
                        help="Dataset variant: drop every episode whose source_repo is listed (e.g. "
                             "TAPTIC/unscrew_the_bottle in the derived dataset with the re-collected unscrew). "
                             "The --taptic_stats file must be fitted on the same selection.")
    parser.add_argument("--resume_state_every", type=int, default=0,
                        help="Restartable training: save the full training state (model, optimizer, RNG, LR scheduler, step, position in the epoch) every N micro-steps (rounded up to whole optimizer steps) and at each epoch end, to <output_dir>/resume_state. 0 = off (default).")
    parser.add_argument("--resume_from_state", type=str, default="",
                        help="'auto': continue from <output_dir>/resume_state if it exists (fresh start otherwise); a directory: continue from that state; empty (default): fresh run.")
    parser.add_argument("--resume_state_dir", type=str, default="",
                        help="Where the restart state lives (default <output_dir>/resume_state).")
    parser.add_argument("--tactile_async_touch", type=int, default=0,
                        help="1: allow --tactile_delay_ms != [0] for --variant vanilla (asynchronous touch: touch newer than the image, T-Rex fast ticks; docs/TOUCH_MODULES.md). Explicit so the baseline cannot drift.")
    parser.add_argument("--taptic_vqvae_source_identity", type=str, default="",
                        help="Use a touch encoder (VQ-VAE) trained on ANOTHER dataset: >= 12 leading hex chars of "
                             "that dataset's identity (e.g. the feedback-off run with the owner's VQ-VAE). The two "
                             "datasets must share no recordings. Empty (default): encoder must come from this dataset.")

    # VQ-VAE tactile code tokens (fast-path only).  When 0 (default) the model
    # graph and dataloader are identical to the pre-feature version — flip
    # --use_tactile_code 1 and run extract_codes.py first to drop a
    # tactile_codes.h5 next to each pretrain.hdf5.
    parser.add_argument("--use_tactile_code", type=int, default=0,
                        help="1: read codes_per_frame from per-episode "
                             "tactile_codes.h5 and add VQ-VAE code tokens to "
                             "the tactile expert's observation context.")
    parser.add_argument("--use_tactile_vqvae", type=int, default=0,
                        help="1: embed the VQ-VAE (from --vqvae_ckpt) in the "
                             "model and tokenize a raw F6 history window "
                             "on-the-fly — no pre-extracted tactile_codes.h5 "
                             "needed.  Implies use_tactile_code.")
    parser.add_argument("--vqvae_codebook_size", type=int, default=64,
                        help="Codebook size of the VQ-VAE that produced the codes.")
    parser.add_argument("--vqvae_codes_h5_name", type=str,
                        default="tactile_codes.h5",
                        help="Filename of the per-episode codes HDF5 produced "
                             "by tactile_vqvae.extract_codes.")
    parser.add_argument("--tactile_code_per_finger", type=int, default=0,
                        help="1 if the VQ-VAE was trained with granularity="
                             "'finger' (so codes_per_frame has shape [N, 2, 5] "
                             "instead of [N, 2]).  Used only to size the "
                             "fallback zero codes.")
    parser.add_argument("--n_fingers", type=int, default=5,
                        help="Number of fingers per hand (used for the finger-"
                             "mode fallback length).")
    parser.add_argument("--vqvae_ckpt", type=str, default="",
                        help="Path to a TactileVQVAE checkpoint (latest.pt).  "
                             "When provided AND the data manifests don't "
                             "include tactile_f6 q01/q99 statistics, the "
                             "trainer loads tactile_f6 min/max from the "
                             "checkpoint's baked-in stats.  Avoids the silent "
                             "[-1, 1] saturation fallback.")
    parser.add_argument("--tactile_f6_stats_ckpt", type=str, default="",
                        help="Alias for --vqvae_ckpt purely for naming "
                             "clarity when you want to reuse the stats "
                             "without enabling tactile_codes.")

    # Flare
    parser.add_argument("--use_flare", type=int, default=1, help="Enable flare visual prediction for latent expert.")
    parser.add_argument("--n_flare_tokens_per_frame", type=int, default=4, help="Number of tokens per future frame.")
    parser.add_argument("--n_flare_steps", type=int, default=8, help="Number of future steps to predict.")
    parser.add_argument("--flare_loss_weight", type=float, default=0.5, help="Weight for flare prediction cosine loss.")
    parser.add_argument("--flare_frame_stride", type=int, default=2, help="Temporal stride for flare frame targets.")
    parser.add_argument("--flare_layer_index", type=int, default=-1, help="Layer to extract flare hidden states from (-1=last, e.g. -7 for ~3/4 depth).")
    parser.add_argument("--frame_stride", type=int, default=2)

    # Validation
    parser.add_argument("--val_ratio", type=float, default=0.0,
                        help="Fraction of samples for validation (0=disable)")
    parser.add_argument("--val_freq", type=int, default=0,
                        help="Run validation every N steps (0=disable)")
    parser.add_argument("--max_val_batches", type=int, default=50,
                        help="Max batches per validation run")

    args = parser.parse_args()
    validate_split_config(args)
    validate_variant(args)

    if args.data_format in ("lerobot", "taptic"):
        if not args.lerobot_root:
            parser.error("--data_format lerobot requires --lerobot_root")
    elif not args.data_root:
        parser.error("--data_format json requires --data_root")

    if args.data_format == "taptic":
        from qwen_vla.taptic_dataset import validate_config
        validate_config(args)
        if not args.taptic_stats:
            parser.error("--data_format taptic requires --taptic_stats")

    args.log_dir = os.path.join(args.log_dir, args.experiment_name)
    args.output_dir = os.path.join(args.output_dir, args.experiment_name, args.run_name)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    train(args)


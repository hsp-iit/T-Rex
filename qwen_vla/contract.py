"""Architecture / checkpoint compatibility contract for the TAPTIC T-Rex variants.

Variants (see docs/taptic_ours/REPORT.md):
  vanilla     fixed tau=0.4, no risk tokens                 (the launched baseline)
  multisplit  uniform tau in {0.2,0.4,0.6}, no risk tokens
  external    multisplit checkpoint + inference-time risk gate (no risk tokens)
  ours        risk tokens in both experts + multisplit + inference-time risk gate

`training_args.json["architecture"]` records the model layout. Loading enforces:
  * risk parameters in a checkpoint but not in the model   -> error (never dropped)
  * risk-token model from a risk-free checkpoint            -> the new parameters keep
    their fresh init and are LISTED in the log/report
  * any shape mismatch on a risk parameter                  -> error
  * other shape mismatches / unexpected keys                -> listed (pre-existing
    behaviour, e.g. a pretrain tactile MLP of a different width)
Checkpoints written before this contract have no "architecture" block; they are
treated as risk-free (they cannot contain risk parameters).
"""
from typing import Optional

RISK_PREFIXES = ('risk_embedder_action.', 'risk_embedder_tactile.')


def is_risk_key(k: str) -> bool:
    return k.startswith(RISK_PREFIXES)


def architecture_from_training_args(ta: Optional[dict]) -> dict:
    if not ta:
        return {'risk_tokens': False, 'source': 'none'}
    arch = ta.get('architecture')
    if arch is None:
        return {'risk_tokens': bool(ta.get('use_risk_tokens', 0)), 'source': 'pre-contract training_args'}
    return dict(arch, source='training_args.architecture')


def check_architecture(model_arch: dict, ckpt_arch: dict):
    """Reject layouts that cannot be loaded faithfully (before touching weights)."""
    if ckpt_arch.get('risk_tokens') and not model_arch.get('risk_tokens'):
        raise ValueError('checkpoint has risk tokens but the model was built without them '
                         '(--use_risk_tokens 1 is required to load it; refusing to drop trained weights)')
    if (ckpt_arch.get('risk_tokens') and model_arch.get('risk_tokens')
            and ckpt_arch.get('risk_n_frequencies') not in (None, model_arch.get('risk_n_frequencies'))):
        raise ValueError('risk embedder frequency count differs from the checkpoint')
    for k in ('action_dim', 'action_chunk'):
        if k in ckpt_arch and k in model_arch and ckpt_arch[k] != model_arch[k]:
            raise ValueError(f'checkpoint {k}={ckpt_arch[k]} != model {k}={model_arch[k]}')


def load_with_contract(model, state_dict: dict, ckpt_arch: Optional[dict] = None, log=print) -> dict:
    """strict=False load with explicit accounting; returns a JSON-able report."""
    model_arch = model.architecture_contract()
    ckpt_arch = ckpt_arch or {'risk_tokens': any(is_risk_key(k) for k in state_dict), 'source': 'inferred'}
    check_architecture(model_arch, ckpt_arch)
    ckpt_risk = sorted(k for k in state_dict if is_risk_key(k))
    if ckpt_risk and not model_arch['risk_tokens']:
        raise ValueError(f'checkpoint contains risk parameters {ckpt_risk[:3]} but the model has none')
    if ckpt_arch.get('risk_tokens') and model_arch['risk_tokens'] and not ckpt_risk:
        raise ValueError('checkpoint declares risk tokens but carries no risk parameters')
    model_sd = model.state_dict()
    filtered, skipped = {}, []
    for k, v in state_dict.items():
        if k in model_sd and model_sd[k].shape != v.shape:
            if is_risk_key(k):
                raise ValueError(f'risk parameter {k}: checkpoint {tuple(v.shape)} != model {tuple(model_sd[k].shape)}')
            skipped.append(k)
        else:
            filtered[k] = v
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    new_risk = sorted(k for k in missing if is_risk_key(k))
    report = {'model_architecture': model_arch, 'checkpoint_architecture': ckpt_arch,
              'loaded': len(filtered) - len(unexpected), 'shape_mismatch_skipped': skipped,
              'missing': sorted(missing), 'unexpected': sorted(unexpected),
              'new_risk_parameters_initialized': new_risk}
    if skipped:
        log(f'[contract] skipped {len(skipped)} shape-mismatched keys: {skipped[:8]}')
    if new_risk:
        log(f'[contract] {len(new_risk)} NEW risk-token parameters are not in the checkpoint and keep '
            f'their fresh Xavier/zero-bias init: {new_risk}')
    other_missing = [k for k in missing if not is_risk_key(k)]
    if other_missing:
        log(f'[contract] {len(other_missing)} other missing keys (fresh init): {other_missing[:8]}')
    if unexpected:
        log(f'[contract] {len(unexpected)} unexpected checkpoint keys ignored: {sorted(unexpected)[:8]}')
    return report


# ── training variants (guard against mislabelled launches) ──────────────────

def validate_variant(args):
    """If --variant is set, the flags must define exactly that variant."""
    v = getattr(args, 'variant', '') or ''
    if not v:
        return
    mode = getattr(args, 'cascaded_split_mode', 'fixed')
    splits = set(float(x) for x in getattr(args, 'cascaded_splits', (.2, .4, .6)))
    risk_tokens = bool(getattr(args, 'use_risk_tokens', 0))
    delays = tuple(getattr(args, 'tactile_delay_ms', (0,)) or (0,))
    errors = []
    if v == 'vanilla':
        if mode != 'fixed' or getattr(args, 'cascaded_split_step', 6) != 6:
            errors.append('vanilla = fixed tau 0.4 (cascaded_split_mode fixed, split_step 6)')
        if risk_tokens:
            errors.append('vanilla has no risk tokens')
        if delays != (0,):
            errors.append('vanilla uses tactile_delay_ms [0] (the launched baseline)')
    elif v in ('multisplit', 'ours'):
        if mode != 'multi' or splits != {.2, .4, .6}:
            errors.append(f'{v} = uniform multi-split over {{0.2, 0.4, 0.6}}')
        if v == 'multisplit' and risk_tokens:
            errors.append('multisplit has no risk tokens (an external gate is inference-only)')
        if v == 'ours' and not (risk_tokens and getattr(args, 'risk_stats', '')):
            errors.append('ours needs --use_risk_tokens 1 and --risk_stats')
    else:
        errors.append(f'unknown variant {v}')
    if errors:
        raise ValueError('variant check failed: ' + '; '.join(errors))

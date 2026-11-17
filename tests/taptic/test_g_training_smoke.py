"""Test G — a few real midtraining steps through scripts/midtrain.py.

Resumes from a synthetic checkpoint with the OFFICIAL pretrain layout
(62-D actions, F6 tactile embedder, robot-state embedder, full-size tactile
MLP) built from the tiny backbone, so the shape-mismatch / re-init / report
path is exercised exactly as with miniFranka/T-Rex_pretrain_mecka22k_epoch1.
Asserts (via --debug_checks): finite losses and grads, optimizer updates every
parameter that received gradient, frozen VQ-VAE unchanged; then checks the
saved checkpoint and the loading report.
"""
import json
import os
import subprocess
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _fake_pretrain_ckpt(tiny_qwen, out):
    from qwen_vla.modeling_vla import Qwen3VLVLAModel
    m = Qwen3VLVLAModel.from_pretrained_qwen3vl(
        tiny_qwen, action_dim=62, action_chunk=16, tacf6_dim=6, use_robot_state=True,
        n_flare_tokens_per_frame=4, n_flare_steps=8)
    m.initialize_vla_weights()
    os.makedirs(out, exist_ok=True)
    torch.save(m.state_dict(), os.path.join(out, "model.pt"))
    json.dump({"action_dim": 62, "use_robot_state": 1, "training_stage": 1}, open(os.path.join(out, "training_args.json"), "w"))
    return out


def test_midtrain_smoke(tmp_path, tiny_qwen, tdir, vq_ckpt):
    ck = _fake_pretrain_ckpt(tiny_qwen, str(tmp_path / "pretrain62"))
    out = tmp_path / "out"
    cmd = [sys.executable, "midtrain.py", "--model_path", tiny_qwen, "--data_format", "taptic",
           "--taptic_dirs", tdir, "--n_epochs", "1", "--save_freq", "1",
           "--action_dim", "21", "--action_chunk", "16", "--tacf6_dim", "3", "--n_tactile_fingers", "5",
           "--train_bsz_per_gpu", "2", "--learning_rate", "1e-4", "--weight_decay", "0.01",
           "--output_dir", str(out), "--log_dir", str(out), "--experiment_name", "g", "--run_name", "r",
           "--use_robot_state", "0", "--use_tactile_vec", "1", "--use_tactile_deform", "0",
           "--use_tactile_vqvae", "1", "--vqvae_ckpt", vq_ckpt, "--tactile_intermediate_size", "64",
           "--training_stage", "2", "--tactile_delay_offsets", "0", "4", "8", "12",
           "--cascaded_total_steps", "10", "--cascaded_split_step", "6", "--cascaded_tactile_dropout", "0.1",
           "--use_flare", "1", "--n_flare_tokens_per_frame", "4", "--n_flare_steps", "8",
           "--flare_loss_weight", "0.5", "--flare_frame_stride", "4", "--image_size", "384", "288",
           "--num_workers", "1", "--val_ratio", "0.1", "--val_freq", "2", "--max_val_batches", "1",
           "--resume_checkpoint", ck, "--resume_source", "pretrain",
           "--max_train_steps", "3", "--debug_checks", "1"]
    env = dict(os.environ, WANDB_MODE="disabled", PYTHONPATH=REPO)
    p = subprocess.run(cmd, cwd=os.path.join(REPO, "scripts"), env=env, capture_output=True, text=True, timeout=1500)
    log = p.stdout + p.stderr
    assert p.returncode == 0, log[-4000:]
    assert log.count("[debug step") >= 3 and "[Val step=" in log
    run = out / "g" / "r"
    ck_dirs = [d for d in os.listdir(run) if d.startswith("checkpoint-")]
    assert ck_dirs, os.listdir(run)
    c = run / ck_dirs[0]
    for f in ("model.pt", "stats_data.json", "training_args.json", "processor"):
        assert (c / f).exists(), f
    ta = json.load(open(c / "training_args.json"))
    assert (ta["action_dim"], ta["action_chunk"], ta["tacf6_dim"], ta["n_tactile_fingers"]) == (21, 16, 3, 5)
    assert (ta["cascaded_total_steps"], ta["cascaded_split_step"]) == (10, 6)
    sd = torch.load(c / "model.pt", map_location="cpu")
    assert sd["x_embedder.mlp.fc1.weight"].shape[1] == 21 and sd["final_layer.mlp.fc2.weight"].shape[0] == 21
    assert sd["tacf6_embedder.mlp.fc1.weight"].shape[1] == 3 and sd["tacf6_vqvae_min"].shape == (15,)
    assert all(torch.isfinite(v.float()).all() for v in sd.values() if v.is_floating_point())
    rep = json.load(open(run / "checkpoint_load_report.json"))
    mm = {m["name"] for m in rep["shape_mismatch"]}
    assert {"x_embedder.mlp.fc1.weight", "final_layer.mlp.fc2.weight", "final_layer.mlp.fc2.bias",
            "final_layer_tactile.mlp.fc2.weight", "tacf6_embedder.mlp.fc1.weight"} <= mm
    assert any(k.startswith("state_embedder") for k in rep["unexpected_in_ckpt"])
    assert "tactile_code_embedder.weight" in rep["missing_in_ckpt"]
    assert rep["status"]["model.layers.0.mlp_action.up_proj.weight"] == "loaded"
    assert rep["status"]["model.layers.0.self_attn.q_proj_tactile.weight"] == "reinit_tactile_expert"

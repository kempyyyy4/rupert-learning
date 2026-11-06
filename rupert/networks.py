"""Rupert's brain shape, shared by training, testing and filming so they always match."""
import json
from pathlib import Path

from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

# Bigger than Brax's default (4 x 32), closer to what robot-running projects use.
DEFAULT_SIZES = {"policy_hidden_layer_sizes": [256, 256, 128], "value_hidden_layer_sizes": [256, 256, 256]}
# Runs from before 2026-11-07 (e.g. colab_run3) used Brax's defaults.
LEGACY_SIZES = {"policy_hidden_layer_sizes": [32, 32, 32, 32], "value_hidden_layer_sizes": [256, 256, 256, 256, 256]}


def sizes_for_run(run_dir):
    cfg = Path(run_dir) / "config.json"
    if cfg.exists():
        return json.loads(cfg.read_text()).get("network", LEGACY_SIZES)
    return DEFAULT_SIZES


def factory(sizes=None):
    sizes = sizes or DEFAULT_SIZES

    def make(obs_size, action_size, preprocess_observations_fn=running_statistics.normalize, **kw):
        return ppo_networks.make_ppo_networks(
            obs_size, action_size, preprocess_observations_fn=preprocess_observations_fn,
            policy_hidden_layer_sizes=tuple(sizes["policy_hidden_layer_sizes"]),
            value_hidden_layer_sizes=tuple(sizes["value_hidden_layer_sizes"]), **kw)
    return make


def make_policy_fn(env, run_dir):
    nets = factory(sizes_for_run(run_dir))(env.observation_size, env.action_size)
    return ppo_networks.make_inference_fn(nets)

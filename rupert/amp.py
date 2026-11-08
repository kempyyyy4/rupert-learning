"""The style judge: a second small AI that has watched real runners (CMU motion capture,
fitted onto Rupert's body by rupert/retarget.py) and learns to tell them apart from Rupert.

Rupert scores style points when the judge thinks his movement looks like a real runner's.
The judge keeps learning too, so as Rupert gets better at looking human, it gets pickier.
This is Adversarial Motion Priors (AMP): Peng et al. (2021), "AMP: Adversarial Motion Priors
for Stylized Physics-Based Character Control", ACM Transactions on Graphics 40(4).

What the judge sees: two moments, 1/50 s apart. For each moment:
  pelvis height, chest tilt (with his heading removed), all 21 joint angles, and where his
  hands, feet and head are relative to his pelvis. It does NOT see how fast he's going
  along the track, so it judges form, not speed.
"""
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jp
import mujoco
import numpy as np

REFERENCE = Path(__file__).parent.parent / "mocap" / "run_reference.npz"
KEY_BODIES = ("left_hand", "right_hand", "left_foot", "right_foot", "head")


class Ids:
    """Body ids the features need (from a MuJoCo model)."""
    def __init__(self, m):
        self.pelvis = m.body("pelvis").id
        self.torso = m.body("torso").id
        self.key = np.array([m.body(b).id for b in KEY_BODIES])


def amp_state(qpos, xpos, xmat, ids):
    """One moment of movement -> feature vector. Works on jax or numpy arrays.
    xmat: (nbody, 3, 3)."""
    fwd = xmat[ids.torso][:, 0]
    yaw = jp.arctan2(fwd[1], fwd[0])
    c, s = jp.cos(yaw), jp.sin(yaw)
    heading_inv = jp.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])   # world -> heading frame
    torso = heading_inv @ xmat[ids.torso]
    rel = (xpos[ids.key] - xpos[ids.pelvis]) @ heading_inv.T
    return jp.concatenate([
        xpos[ids.pelvis, 2:3],
        torso[:, 0], torso[:, 2],     # chest forward and up directions
        qpos[7:],                     # joint angles
        rel.ravel(),
    ])


def state_size(m):
    return 1 + 6 + (m.nq - 7) + 3 * len(KEY_BODIES)


def reference_transitions(m):
    """All consecutive pairs of moments from the real-runner clips: (N, 2 * state_size)."""
    z = np.load(REFERENCE)
    qpos, starts = z["qpos"].astype(np.float64), z["starts"]
    d = mujoco.MjData(m)
    ids = Ids(m)
    feats = []
    for q in qpos:
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        feats.append(np.asarray(amp_state(q, d.xpos.copy(), d.xmat.reshape(-1, 3, 3).copy(), ids)))
    feats = np.array(feats, dtype=np.float32)
    pairs = []
    for a, b in zip(starts[:-1], starts[1:]):
        pairs.append(np.concatenate([feats[a:b - 1], feats[a + 1:b]], axis=1))
    return np.concatenate(pairs)


class Discriminator(nn.Module):
    hidden: tuple = (512, 256)

    @nn.compact
    def __call__(self, x):
        for h in self.hidden:
            x = nn.relu(nn.Dense(h)(x))
        return nn.Dense(1)(x)[..., 0]


class Judge:
    """Everything the trainer needs: network, reference data, normalisation, losses."""

    def __init__(self, m, style_weight=1.5, task_weight=1.0, learning_rate=5e-5,
                 grad_penalty=5.0, batch=4096, updates_per_step=2):
        self.ref = jp.asarray(reference_transitions(m))
        self.mean = self.ref.mean(0)
        self.std = self.ref.std(0) + 1e-2
        self.net = Discriminator()
        self.style_weight, self.task_weight = style_weight, task_weight
        self.learning_rate, self.grad_penalty = learning_rate, grad_penalty
        self.batch, self.updates_per_step = batch, updates_per_step

    def init(self, key):
        return self.net.init(key, jp.zeros(self.ref.shape[1]))

    def score(self, params, x):
        return self.net.apply(params, (x - self.mean) / self.std)

    def style_reward(self, params, x):
        """1 when the judge is sure he's a real runner, 0 when sure he's not (AMP's formula)."""
        d = self.score(params, x)
        return jp.clip(1.0 - 0.25 * jp.square(d - 1.0), 0.0, 1.0)

    def loss(self, params, real, fake):
        """Least-squares GAN loss: real -> +1, Rupert -> -1, plus a gradient penalty on real
        samples that keeps the judge smooth (stops its scores swinging wildly)."""
        real_n = (real - self.mean) / self.std
        d_real = self.net.apply(params, real_n)
        d_fake = self.score(params, fake)
        grad = jax.vmap(jax.grad(lambda x: self.net.apply(params, x)))(real_n)
        gp = jp.mean(jp.sum(jp.square(grad), axis=-1))
        loss = jp.mean(jp.square(d_real - 1.0)) + jp.mean(jp.square(d_fake + 1.0)) + self.grad_penalty * gp
        return loss, {"judge/loss": loss, "judge/real": jp.mean(d_real), "judge/rupert": jp.mean(d_fake),
                      "judge/grad_penalty": gp}

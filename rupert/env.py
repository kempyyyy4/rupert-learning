"""Rupert's training world: a human-sized body on a straight track, scored for running forward.

Track: start line at x = 0, running towards +x, lane centred on y = 0.

Scoring (all per control step, 50 times a second). Weights are in REWARD_WEIGHTS.
  + forward_speed : forward speed along the track in m/s, capped at SPEED_CAP_MS,
                    multiplied by a posture factor: full credit only when running TALL
                    (head near standing height). Added after colab_run3 learned a hunched shuffle.
  + alive         : small bonus for every moment he is still up
  + upright       : chest AND pelvis upright (1 = perfectly upright)
  + heading       : facing down the track
  + stride        : when a foot lands, metres it travelled since it left the ground
                    (only alternating feet, only after a real swing of MIN_SWING_S+, and only
                    if it lands in front of the other foot), scaled by how EVEN the last two steps
                    are (shorter / longer). EXP-008 limped with one long and one short step.
                    Rewards long strides directly. Air-time-based v2 was cheated with
                    tiny stiff-legged bounces (run rupert_20261106_2121).
  - energy        : mechanical power used by the muscles, per kg (wasted effort = weird gaits)
  - jerk          : sudden changes in muscle commands (jitter)
  - lane          : drifting sideways out of the lane, and sideways speed
  - arms          : a hand held far in front of / behind the chest (v2 ran with a 'zombie arm')
  + arm_swing     : arms MOVING opposite to the legs (left hand swinging forward while the right
                    foot swings forward). EXP-007 clamped one arm to his chest ('T-rex scuttle');
                    EXP-008 faked a position-based version with a frozen pose and a gallop.
  + knees         : swinging leg's knee bent (up to ~90 deg) while the OTHER foot is planted, minus a penalty for locking a
                    knee past straight. v3 (run rupert_20261106_2220) ran on near-straight legs.
The attempt ENDS if he falls: head below FALL_HEAD_HEIGHT_M, or a knee or hand near the ground
(no knee-running or crawling: run rupert_20261107_1159 learned to run on its knees).

Muscles (body v2): each joint has a peak torque per direction (rupert/body.py, an ordinary
young man), and like real muscle it
  - gets weaker the faster the joint is already moving in that direction
    (linear force-velocity limit, MAX_JOINT_SPEED), and
  - takes time to switch on and off (first-order activation, MUSCLE_ON_S / MUSCLE_OFF_S).
EXP-007 (body v1: strong, instant muscles) ran 100 m in 10.0 s at 6+ steps a second.
Old runs are replayed with the body they learned in: see load_for_run().
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import numpy as np
import jax.numpy as jp
import mujoco
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.dm_control_suite import common

from rupert import amp as rupert_amp
from rupert import body as rupert_body

XML_PATH = Path(__file__).parent / "assets" / "rupert.xml"

SPEED_CAP_MS = 12.5       # Bolt's peak was ~12.4 m/s; no extra credit beyond this
FALL_HEAD_HEIGHT_M = 1.3  # standing head height is ~1.75 m; kneeling upright is ~1.3
LOW_LIMB_M = 0.22         # a knee joint or hand this close to the ground counts as a fall
TALL_HEAD_HEIGHT_M = (1.40, 1.62)  # v2 used (1.35, 1.65) and ran stiff-legged; real running bobs a little  # posture factor goes from 0.3 at the low end to 1.0 at the high end
LANE_HALF_WIDTH_M = 0.61  # standard lane is 1.22 m wide

REWARD_WEIGHTS = dict(
    forward_speed=0.25,
    alive=0.2,     # kept small: big "just stay up" bonuses teach standing still (the stock setup's trap)
    upright=0.1,
    heading=0.1,
    stride=2.0,
    energy=-0.004,
    jerk=-0.1,
    lane=-0.3,
    arms=-1.0,
    knees=0.3,
    arm_swing=0.5,
)
# Scoring v8 (style judge, EXP-010+): only the basic rules. All the special fixes for his cheats
# (posture, strides, knees, arm swing, hand reach) are switched off: the style judge, which has
# watched real runners, scores how human he looks instead (rupert/amp.py, added in amp_ppo.py).
REWARD_WEIGHTS_JUDGE = dict(
    forward_speed=0.25,
    alive=0.2,
    upright=0.0,
    heading=0.1,
    stride=0.0,
    energy=-0.004,
    jerk=-0.1,
    lane=-0.3,
    arms=0.0,
    knees=0.0,
    arm_swing=0.0,
)

# EXP-005 ran with near-straight legs (left knee bent only 12 deg, both pushed past straight).
# Real runners fold the swinging knee to ~90 deg or more.
KNEE_SWING_TARGET_RAD = 1.6   # ~90 deg: full credit for this much bend while the foot is in the air
KNEE_ORDER = ("right_knee", "left_knee")   # same order as FOOT_GEOMS
HAND_REACH_M = 0.35   # hands further than this in front of / behind the chest are penalised
# arm_swing (scoring v7) = -(left - right hand forward SPEED) x (left - right foot forward SPEED)
# / this, capped at +-1. Uses speeds, not positions: EXP-008 held a frozen 'opposite' pose and
# galloped to collect the position-based v6 reward without swinging.
ARM_SWING_SCALE = 40.0   # calibrated below on replays; a real swing averages well above 0, frozen = 0
LEAD_MARGIN_M = 0.10     # a step only counts if the landing foot lands this far ahead of the other

# Muscle switch-on / switch-off time constants (s): Zajac (1989), Winters (1990), as used by
# OpenSim's first-order activation model. Body v1 had instant muscles.
MUSCLE_ON_S = 0.010
MUSCLE_OFF_S = 0.040

# Body v3 hip sideways range (deg): negative = outwards (abduction 33), positive = inwards (adduction 14).
# Passive normative values 33.0 +- 4.8 / 13.8 +- 6.3 (AAOS-based table, Sports Health 2020, PMC8167346).
HIP_X_RANGE_DEG = (-33.0, 14.0)

# rad/s at which a joint's torque drops to zero (concentric). TO VERIFY against sources.
MAX_JOINT_SPEED = {"hip_y": 25.0, "knee": 35.0, "ankle_y": 35.0, "default": 20.0}

FOOT_GEOMS = (("right_right_foot", "left_right_foot"), ("left_left_foot", "right_left_foot"))
MIN_SWING_S = 0.15     # ignore foot 'chatter' on the ground
MAX_STRIDE_CREDIT_M = 3.0   # a sprinting foot travels ~4-5 m per swing; 3 m is plenty for now


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,
        episode_length=1000,   # 20 s per attempt
        action_repeat=1,
        impl="warp",
        naconmax=200_000,
        njmax=250,
        init_noise=0.05,
        style_judge=True,     # scoring v8: basic rules + style judge (False = scoring v7 rules)
        ref_start_prob=0.0,   # chance an attempt starts mid-stride in a real runner's pose (training only)
        body_version=3,   # 1 = EXP-001..007 (strong, instant muscles); 2 = ordinary young man (EXP-008/009);
                          # 3 = v2 + human hip side-to-side range (needed to copy real runners)
    )


def _joint_key(joint_name):
    for side in ("left_", "right_"):
        joint_name = joint_name.replace(side, "")
    return joint_name


class RupertRun(mjx_env.MjxEnv):
    def __init__(
        self,
        config: Optional[config_dict.ConfigDict] = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        # Fresh settings each time: a shared default gets changed in place by overrides.
        super().__init__(config if config is not None else default_config(), config_overrides)
        self._xml_path = XML_PATH.as_posix()
        self._model_assets = common.get_assets()
        self._mj_model = mujoco.MjModel.from_xml_string(XML_PATH.read_text(), self._model_assets)
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

        m = self._mj_model
        self._head_id = m.body("head").id
        self._torso_id = m.body("torso").id
        self._pelvis_id = m.body("pelvis").id
        self._hand_ids = jp.array([m.body("left_hand").id, m.body("right_hand").id])
        self._extremity_ids = jp.array([m.body(s + l).id for s in ("left_", "right_") for l in ("hand", "foot")])
        self._foot_geom_ids = jp.array([[m.geom(g).id for g in pair] for pair in FOOT_GEOMS])
        self._foot_radius = float(m.geom_size[m.geom(FOOT_GEOMS[0][0]).id][0])
        self._foot_half_len = jp.array(m.geom_size[:, 1])
        self._body_mass = float(m.body_subtreemass[self._torso_id])

        joint_ids = m.actuator_trnid[:, 0]
        self._act_dof = jp.array(m.jnt_dofadr[joint_ids])
        self._version = int(self._config.body_version)
        peaks = [rupert_body.peak_torques(_joint_key(m.joint(j).name), self._version) for j in joint_ids]
        m.actuator_gear[:, 0] = [max(p) for p in peaks]   # v1 runs get their old strengths back
        if self._version >= 3:   # hip sideways: out 33 deg, in 14 deg (AAOS-based norms; stock was 25 / 5)
            for side in ("right", "left"):
                m.jnt_range[m.joint(f"{side}_hip_x").id] = np.radians(HIP_X_RANGE_DEG)
        self._mjx_model = mjx.put_model(m, impl=self._config.impl)
        self._gear = jp.array(m.actuator_gear[:, 0])
        self._pos_scale = jp.array([p[0] / max(p) for p in peaks])   # weaker direction scaled down
        self._neg_scale = jp.array([p[1] / max(p) for p in peaks])
        on, off = (MUSCLE_ON_S, MUSCLE_OFF_S) if self._version >= 2 else (0.0, 0.0)
        self._on_rate = min(self.sim_dt / on, 1.0) if on else 1.0
        self._off_rate = min(self.sim_dt / off, 1.0) if off else 1.0
        self._foot_ids = jp.array([m.body("left_foot").id, m.body("right_foot").id])   # same order as _hand_ids
        self._act_max_speed = jp.array([
            MAX_JOINT_SPEED.get(_joint_key(m.joint(j).name), MAX_JOINT_SPEED["default"]) for j in joint_ids
        ])
        self._qpos0 = jp.array(m.qpos0)
        self._knee_qpos = jp.array([m.jnt_qposadr[m.joint(j).id] for j in KNEE_ORDER])
        self._shin_ids = jp.array([m.body("right_shin").id, m.body("left_shin").id])   # body origin = knee joint
        self._judge_on = bool(self._config.style_judge)
        self._weights = REWARD_WEIGHTS_JUDGE if self._judge_on else REWARD_WEIGHTS
        self._amp_ids = rupert_amp.Ids(m)
        self._ref_prob = float(self._config.ref_start_prob)
        if self._ref_prob > 0:
            self._ref_qpos, self._ref_qvel = self._reference_starts(m)

    def _reference_starts(self, m):
        """Poses and speeds from the real-runner clips, for starting some attempts mid-stride."""
        z = np.load(rupert_amp.REFERENCE)
        q, starts, fps = z["qpos"].astype(np.float64), z["starts"], float(z["fps"])
        qs, vs = [], []
        for a, b in zip(starts[:-1], starts[1:]):
            for i in range(a, b - 1):
                v = np.zeros(m.nv)
                mujoco.mj_differentiatePos(m, v, 1.0 / fps, q[i], q[i + 1])
                start = q[i].copy()
                start[0] = 0.0   # on the start line
                qs.append(start); vs.append(v)
        return jp.array(np.array(qs)), jp.array(np.array(vs))

    # ------------------------------------------------------------------ reset
    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, k1, k2 = jax.random.split(rng, 3)
        n = self._config.init_noise
        qpos = self._qpos0.at[7:].add(jax.random.uniform(k1, (self.mj_model.nq - 7,), minval=-n, maxval=n))
        qvel = jax.random.uniform(k2, (self.mj_model.nv,), minval=-n, maxval=n)
        if self._ref_prob > 0:
            rng, k3, k4 = jax.random.split(rng, 3)
            i = jax.random.randint(k3, (), 0, self._ref_qpos.shape[0])
            use_ref = jax.random.uniform(k4) < self._ref_prob
            qpos = jp.where(use_ref, self._ref_qpos[i], qpos)
            qvel = jp.where(use_ref, self._ref_qvel[i], qvel)
        data = mjx_env.make_data(
            self.mj_model, qpos=qpos, qvel=qvel, impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax, njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)
        info = {
            "rng": rng,
            "last_action": jp.zeros(self.action_size),
            "activation": jp.zeros(self.action_size),
            "air_time": jp.zeros(2),
            "last_foot": jp.array(-1.0),
            "liftoff_x": jp.zeros(2),
            "limb_fwd": self._limb_fwd(data),
            "last_step": jp.array(-1.0),   # unknown until the first proper step
        }
        s0 = self._amp_state(data)
        info["amp_obs"] = jp.concatenate([s0, s0])
        metrics = {k: jp.zeros(()) for k in (
            *(f"reward/{w}" for w in REWARD_WEIGHTS), "reward/task",
            "distance_m", "time_upright_s")}
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    # ------------------------------------------------------------------- step
    def _muscle_ctrl(self, data, act):
        """Muscle activation -> motor control: per-direction strength, and less torque
        available the faster the joint is already moving the same way (force-velocity)."""
        w = data.qvel[self._act_dof]
        same_dir = jp.sign(act) == jp.sign(w)
        factor = jp.where(same_dir, jp.clip(1.0 - jp.abs(w) / self._act_max_speed, 0.0, 1.0), 1.0)
        strength = jp.where(act > 0, self._pos_scale, self._neg_scale)
        return act * factor * strength

    def _activate(self, act, action):
        """Muscles switch on fast and off slowly; they can't jump straight to a new command."""
        rate = jp.where(jp.abs(action) > jp.abs(act), self._on_rate, self._off_rate)
        return act + (action - act) * rate

    def _amp_state(self, data):
        return rupert_amp.amp_state(data.qpos, data.xpos, data.xmat, self._amp_ids)

    def _fresh_info(self, state):
        """The trainer's auto-restart only resets the body, not our bookkeeping. On the first step
        of a new attempt (sim clock at 0), clear what carried over from the last attempt."""
        fresh = state.data.time < 1e-9
        info = dict(state.info)
        foot_x = state.data.geom_xpos[self._foot_geom_ids[:, 0], 0]
        clean = {
            "last_action": jp.zeros(self.action_size),
            "activation": jp.zeros(self.action_size),
            "air_time": jp.zeros(2),
            "last_foot": jp.array(-1.0),
            "liftoff_x": foot_x,
            "limb_fwd": self._limb_fwd(state.data),
            "last_step": jp.array(-1.0),
        }
        for k, v in clean.items():
            info[k] = jp.where(fresh, v, info[k])
        return state.replace(info=info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        state = self._fresh_info(state)
        amp_before = self._amp_state(state.data)   # from the body itself, so restarts can't mix attempts

        def substep(carry, _):
            data, act = carry
            act = self._activate(act, action)
            ctrl = self._muscle_ctrl(data, act)
            data = mjx.step(self.mjx_model, data.replace(ctrl=ctrl))
            return (data, act), None

        data, act = jax.lax.scan(substep, (state.data, state.info["activation"]), (), self.n_substeps)[0]
        info = dict(state.info)
        info["activation"] = act
        info["amp_obs"] = jp.concatenate([amp_before, self._amp_state(data)])
        reward, parts, info = self._get_reward(data, action, info)

        low = jp.concatenate([data.xpos[self._shin_ids, 2], data.xpos[self._hand_ids, 2]])
        fell = (self._head_height(data) < FALL_HEAD_HEIGHT_M) | jp.any(low < LOW_LIMB_M)
        nan = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = (fell | nan).astype(float)

        metrics = dict(state.metrics)
        for k, v in parts.items():
            metrics[f"reward/{k}"] = v
        vx = self._com_vel(data)[0]
        metrics["distance_m"] = vx * self.dt            # summed over an attempt = distance
        metrics["time_upright_s"] = jp.array(self.dt)   # summed = time before falling
        info["last_action"] = action

        obs = self._get_obs(data, info)
        reward = jp.where(nan, 0.0, reward)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    # ----------------------------------------------------------------- reward
    def _get_reward(self, data, action, info):
        vel = self._com_vel(data)
        torso = data.xmat[self._torso_id]
        parts = {}
        lo, hi = TALL_HEAD_HEIGHT_M
        tall = jp.clip((self._head_height(data) - lo) / (hi - lo), 0.0, 1.0)
        posture = 0.3 + 0.7 * tall if not self._judge_on else 1.0   # judge mode: no 'run tall' rule
        parts["forward_speed"] = jp.clip(vel[0], -2.0, SPEED_CAP_MS) * posture
        parts["alive"] = jp.array(1.0)
        parts["upright"] = 0.5 * (torso[2, 2] + data.xmat[self._pelvis_id][2, 2])
        parts["heading"] = torso[0, 0]   # torso's forward axis pointing down the track (+x)

        # Strides: foot touchdown after time in the air, alternating feet.
        contact = self._feet_on_ground(data)
        air = info["air_time"]
        touchdown = contact & (air > 0)
        foot_idx = jp.arange(2.0)
        alternating = foot_idx != info["last_foot"]
        foot_x = data.geom_xpos[self._foot_geom_ids[:, 0], 0]
        travelled = jp.clip(foot_x - info["liftoff_x"], 0.0, MAX_STRIDE_CREDIT_M)
        # Even steps: credit is scaled by (shorter step / longer step) of the last two steps, where a
        # step = how far the landing foot lands in front of the other foot. EXP-008 limped along with
        # left steps ~0.3 m and right steps ~0.75 m (factor 0.4); an even run scores ~1.
        step_len = foot_x - foot_x[::-1]
        last = info["last_step"]
        even = jp.where(last > 0, jp.minimum(step_len, last) / jp.maximum(step_len, last), 1.0)
        valid = touchdown & alternating & (air >= MIN_SWING_S) & (step_len > LEAD_MARGIN_M)
        parts["stride"] = jp.sum(jp.where(valid, travelled * jp.clip(even, 0.0, 1.0), 0.0))
        info["last_step"] = jp.where(valid.any(), jp.sum(jp.where(valid, step_len, 0.0)), last)
        liftoff = (~contact) & (air == 0)
        info["liftoff_x"] = jp.where(liftoff, foot_x, info["liftoff_x"])
        info["last_foot"] = jp.where(touchdown.any(), jp.argmax(touchdown).astype(float), info["last_foot"])
        info["air_time"] = jp.where(contact, 0.0, air + self.dt)

        torque = self._muscle_ctrl(data, info["activation"]) * self._gear
        joint_speed = data.qvel[self._act_dof]
        parts["energy"] = jp.sum(jp.abs(torque * joint_speed)) / self._body_mass
        parts["jerk"] = jp.mean(jp.square(action - info["last_action"]))
        y = data.subtree_com[self._torso_id][1]
        parts["lane"] = jp.abs(vel[1]) + jp.clip(jp.abs(y) - LANE_HALF_WIDTH_M, 0.0, None) * 5.0
        limbs = self._limb_fwd(data)               # hand L, hand R, foot L, foot R
        parts["arms"] = jp.sum(jp.clip(jp.abs(limbs[:2]) - HAND_REACH_M, 0.0, None))
        # Arm swing from MOVEMENT: frozen arms score 0 whatever pose they're held in.
        v = (limbs - info["limb_fwd"]) / self.dt
        swing_product = (v[0] - v[1]) * (v[2] - v[3])
        parts["arm_swing"] = jp.clip(-swing_product / ARM_SWING_SCALE, -1.0, 1.0)
        info["limb_fwd"] = limbs
        knee = data.qpos[self._knee_qpos]   # 0 = straight, negative = bent
        bend = jp.clip(-knee, 0.0, KNEE_SWING_TARGET_RAD) / KNEE_SWING_TARGET_RAD
        swing = (~contact) & contact[::-1]          # this foot in the air WHILE the other one is planted
        parts["knees"] = jp.sum(jp.where(swing, bend, 0.0)) - 3.0 * jp.sum(jp.clip(knee, 0.0, None))

        reward = sum(self._weights[k] * v for k, v in parts.items())
        parts["task"] = reward   # logged; the style judge's points are added by the trainer
        return reward, parts, info

    # ------------------------------------------------------------ observations
    def _get_obs(self, data, info):
        torso = data.xmat[self._torso_id]
        torso_pos = data.xpos[self._torso_id]
        extremities = (data.xpos[self._extremity_ids] - torso_pos) @ torso
        return jp.concatenate([
            data.qpos[7:],                         # joint angles
            self._head_height(data).reshape(1),
            data.subtree_com[self._torso_id][1:2], # sideways position in lane
            extremities.ravel(),
            torso.ravel(),                         # full body orientation (heading + tilt)
            self._com_vel(data),
            data.qvel,
            self._feet_on_ground(data).astype(float),
            info["air_time"],
            info["last_action"],
            *([info["activation"]] if self._version >= 2 else []),   # v1 brains never saw this
        ])

    # ---------------------------------------------------------------- helpers
    def _limb_fwd(self, data):
        """How far each hand and foot is in front of the chest (chest's own forward axis):
        [left hand, right hand, left foot, right foot]."""
        torso = data.xmat[self._torso_id]
        ids = jp.concatenate([self._hand_ids, self._foot_ids])
        return ((data.xpos[ids] - data.xpos[self._torso_id]) @ torso)[:, 0]

    def _feet_on_ground(self, data):
        # Lowest point of each foot capsule, so running on the toes (heel up) still counts as contact.
        ids = self._foot_geom_ids                                    # (2 feet, 2 capsules)
        centre_z = data.geom_xpos[ids][..., 2]
        axis_z = data.geom_xmat[ids][..., 2, 2]                      # capsule's long axis, vertical part
        lowest = centre_z - jp.abs(self._foot_half_len[ids] * axis_z) - self._foot_radius
        return jp.min(lowest, axis=1) < 0.01

    def _head_height(self, data):
        return data.xpos[self._head_id, 2]

    def _com_vel(self, data):
        return mjx_env.get_sensor_data(self.mj_model, data, "torso_subtreelinvel")

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self.mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model


def load_for_run(run_dir, task: str = "Rupert100m", config_overrides=None):
    """The environment a saved run was trained in (same body version), for replaying it."""
    cfg = Path(run_dir) / "config.json"
    env_cfg = (json.loads(cfg.read_text()).get("env") or {}) if cfg.exists() else {}
    if task == "Rupert100m":
        config_overrides = {"body_version": env_cfg.get("body_version", 1), **(config_overrides or {})}
    return load(task, config_overrides)


def load(task: str = "Rupert100m", config_overrides=None):
    """One place to get any environment by name (ours or Playground's)."""
    if task == "Rupert100m":
        return RupertRun(config_overrides=config_overrides)
    from mujoco_playground import registry
    return registry.load(task, config_overrides=config_overrides)

"""Checks that MuJoCo and MuJoCo Playground work on this PC and prints a plain report."""
import importlib
import platform
import sys


def try_step(name, fn):
    try:
        detail = fn()
        print(f"  [OK]   {name}" + (f" - {detail}" if detail else ""))
        return True
    except Exception as e:  # report everything in plain words, never a traceback
        print(f"  [FAIL] {name} - {type(e).__name__}: {str(e)[:200]}")
        return False


def main():
    print("Rupert Sim install check")
    print(f"  Python {sys.version.split()[0]} on {platform.system()} {platform.release()}")
    results = []

    def mujoco_step():
        import mujoco
        m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='1'/></worldbody></mujoco>")
        mujoco.mj_step(m, mujoco.MjData(m))
        return f"MuJoCo {mujoco.__version__} runs a physics step"

    def jax_step():
        import jax
        return f"JAX {jax.__version__}, devices: {[str(d) for d in jax.devices()]}"

    def mjx_step():
        import jax
        from mujoco_playground import registry
        env = registry.load("HumanoidRun")
        state = jax.jit(env.reset)(jax.random.PRNGKey(0))
        state = jax.jit(env.step)(state, jax.numpy.zeros(env.action_size))
        return f"humanoid env runs on MJX: {env.action_size} motors, reward {float(state.reward):.3f}"

    def playground_step():
        from mujoco_playground import registry
        names = list(registry.dm_control_suite.ALL_ENVS)
        return f"{len(names)} DM Control tasks incl. {[n for n in names if 'umanoid' in n]}"

    results.append(try_step("MuJoCo", mujoco_step))
    results.append(try_step("JAX", jax_step))
    results.append(try_step("MJX", mjx_step))
    results.append(try_step("MuJoCo Playground", playground_step))

    def rupert_step():
        import jax
        from rupert.env import load
        env = load("Rupert100m")
        state = jax.jit(env.reset)(jax.random.PRNGKey(0))
        state = jax.jit(env.step)(state, jax.numpy.zeros(env.action_size))
        mass = env.mj_model.body_subtreemass[1]
        return f"Rupert's body loads: {mass:.0f} kg, {env.action_size} muscles, head at {float(env._head_height(state.data)):.2f} m"

    results.append(try_step("Rupert", rupert_step))
    print("ALL GOOD" if all(results) else "SOME THINGS FAILED (see [FAIL] lines above)")


if __name__ == "__main__":
    main()

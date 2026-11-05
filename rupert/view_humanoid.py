"""Opens the standard MuJoCo humanoid in an interactive window (drag to rotate, scroll to zoom).

It has no brain yet, so it just collapses. That is the 'before' picture.
"""
import time

import mujoco
import mujoco.viewer

from mujoco_playground._src import mjx_env


def main():
    xml_path = mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "humanoid.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    print(f"Loaded {xml_path.name}: {model.nbody} bodies, {model.nu} motors. Close the window to stop.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))


if __name__ == "__main__":
    main()

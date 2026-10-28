"""Regression checks for false 'unreachable' results; no viewer required."""
import mujoco
import numpy as np
import pytest
from env import RowEnv
from engine import StackEngine


@pytest.fixture(scope="module")
def engine():
    env = RowEnv(render=False)
    yield StackEngine(env)
    env.close()


@pytest.mark.parametrize("height",[.109,.169,.199])
def test_upper_tower_positions_are_reachable_without_moving_real_arm(engine,height):
    original = engine.env.data.qpos.copy()
    target = np.array([.09,.17,height])
    joints,error,tilt,ok = engine._solve(target,0.,90)
    assert ok and error<.003 and tilt<90
    # Independently forward the returned joints through the full model. This
    # catches stale Jacobians/transforms or a solver reporting a false success.
    data = mujoco.MjData(engine.m)
    data.qpos[:] = original
    for name,value in joints.items():data.qpos[engine.qadr[name]] = value
    data.qpos[engine.qadr["wrist_roll"]] = 0.
    mujoco.mj_forward(engine.m,data)
    assert np.linalg.norm(data.site_xpos[engine.site]-target)<.003
    for name,value in joints.items():assert engine.rng[name][0]<=value<=engine.rng[name][1]
    np.testing.assert_array_equal(engine.env.data.qpos,original)


def test_unreachable_point_and_tilt_limit_are_still_rejected(engine):
    assert not engine._solve(np.array([2.,2.,2.]),0.,90)[3]
    assert not engine._solve(np.array([.09,.17,.199]),0.,5)[3]


def test_search_pumps_viewer_without_advancing_physics(engine):
    updates=[]
    engine.env.on_plan=lambda:updates.append(1)
    before=engine.env.data.time
    try:engine._solve(np.array([.09,.17,.169]),0.,90)
    finally:engine.env.on_plan=None
    assert updates and engine.env.data.time==before

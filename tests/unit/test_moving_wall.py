# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Particle trajectories at the fixed x_max moving-wall boundary."""

import numpy as np
import pytest
from conftest import Config, rtol
from helpers import N_AXES, make_sim

import pywarpx
from pywarpx import picmi

constants = picmi.constants

pytestmark = pytest.mark.skipif(
    pywarpx.libwarpx.geometry_dim not in ("2d", "3d"),
    reason="moving_wall is supported at Cartesian x_max in 2D and 3D",
)


def _make_wall_sim(beta):
    n_axes = N_AXES[pywarpx.libwarpx.geometry_dim]
    sim = make_sim(
        lower_bound=[-1.0] * n_axes,
        upper_bound=[1.0] * n_axes,
        dt=1.0 / (64.0 * constants.c),
    )
    sim.max_steps = 3
    grid = sim.solver.grid
    grid.lower_boundary_conditions[0] = "dirichlet"
    grid.upper_boundary_conditions[0] = "dirichlet"
    grid.lower_boundary_conditions_particles[0] = "absorbing"
    grid.upper_boundary_conditions_particles[0] = "moving_wall"
    grid.moving_wall_beta = beta
    sim.add_species(
        picmi.Species(name="neutral", mass=constants.m_e, charge=0.0), layout=None
    )
    sim.initialize_inputs()
    assert pywarpx.boundary.moving_wall_beta == beta
    sim.initialize_warpx()
    return sim


def _particle_state(pc):
    """Copy the six phase-space coordinates, including GPU-host transfers."""
    names = ["x", "ux", "uy", "uz", "z"]
    if pywarpx.libwarpx.geometry_dim == "3d":
        names.append("y")
    pieces = []
    for pti in pc.iterator(level=0):
        if pti.num_particles:
            real = pti.soa().to_numpy(copy=True).real
            pieces.append(np.column_stack([real[name] for name in names]))
    state = np.concatenate(pieces) if pieces else np.empty((0, len(names)))
    if Config.have_mpi:
        from mpi4py import MPI

        state = np.concatenate(MPI.COMM_WORLD.allgather(state))
    return dict(zip(names, state.T))


def _add_particle(pc, x, u):
    pc.add_particles(
        x=x,
        y=0.0,
        z=0.0,
        ux=u[0],
        uy=u[1],
        uz=u[2],
        w=1.0,
        unique_particles=False,
    )


def _lorentz_reflection(u, beta):
    """Boost to the wall frame, reverse p_x there, and boost back."""
    q = np.asarray(u) / constants.c
    energy = np.sqrt(1.0 + np.dot(q, q))
    wall_gamma = 1.0 / np.sqrt(1.0 - beta**2)
    wall_energy = wall_gamma * (energy - beta * q[0])
    wall_qx = wall_gamma * (q[0] - beta * energy)
    reflected_q = q.copy()
    reflected_q[0] = wall_gamma * (-wall_qx + beta * wall_energy)
    reflected_energy = wall_gamma * (wall_energy - beta * wall_qx)
    return constants.c * reflected_q, constants.c * reflected_q / reflected_energy


@pytest.mark.parametrize("beta", [0.0, 0.2, -0.25])
def test_moving_wall_relativistic_reflection(beta):
    """Validate the normal impulse and collision time with transverse momentum."""
    sim = _make_wall_sim(beta)
    pc = sim.particles.get("neutral")
    dt = sim.extension.warpx.getdt(0)
    u = constants.c * np.array([1.4, 0.3, -0.2])
    v = u / np.sqrt(1.0 + np.dot(u, u) / constants.c**2)
    xmax = 1.0
    x0 = xmax - 0.25 * v[0] * dt
    _add_particle(pc, x0, u)

    reflected_u, reflected_v = _lorentz_reflection(u, beta)
    collision_time = (xmax - x0) / (v[0] - beta * constants.c)
    expected_x = (
        xmax
        + beta * constants.c * collision_time
        + reflected_v[0] * (dt - collision_time)
    )
    assert 0.0 < collision_time < dt
    assert expected_x < xmax
    assert reflected_u[0] < 0.0

    sim.step(1)
    assert pc.size == 1
    state = _particle_state(pc)
    np.testing.assert_allclose(state["x"], expected_x, rtol=rtol(), atol=0.0)
    for name, expected in zip(("ux", "uy", "uz"), reflected_u):
        np.testing.assert_allclose(state[name], expected, rtol=rtol(), atol=0.0)
    np.testing.assert_allclose(state["z"], v[2] * dt, rtol=rtol(), atol=0.0)
    if "y" in state:
        np.testing.assert_allclose(state["y"], v[1] * dt, rtol=rtol(), atol=0.0)

    # The virtual wall resets at x_max each step. This reflected particle is
    # inside the domain and must subsequently free-stream without another kick.
    sim.step(1)
    assert pc.size == 1
    state = _particle_state(pc)
    np.testing.assert_allclose(
        state["x"], expected_x + reflected_v[0] * dt, rtol=rtol(), atol=0.0
    )
    for name, expected in zip(("ux", "uy", "uz"), reflected_u):
        np.testing.assert_allclose(state[name], expected, rtol=rtol(), atol=0.0)


@pytest.mark.parametrize(
    "beta,speed,fraction,reason",
    [
        (0.75, 0.45, 0.25, "wall_overtakes"),
        (0.5, 0.65, 0.1, "positive_reflected_momentum"),
        (0.3, 0.65, 0.5, "reflected_position_outside"),
    ],
    ids=["wall-overtakes", "positive-momentum", "position-outside"],
)
def test_moving_wall_removes_rejected_particles(beta, speed, fraction, reason):
    sim = _make_wall_sim(beta)
    pc = sim.particles.get("neutral")
    dt = sim.extension.warpx.getdt(0)
    v = speed * constants.c
    u = np.array([v / np.sqrt(1.0 - speed**2), 0.0, 0.0])
    x0 = 1.0 - fraction * v * dt
    pushed_x = x0 + v * dt
    assert pushed_x > 1.0
    if reason == "wall_overtakes":
        assert pushed_x < 1.0 + beta * constants.c * dt
    else:
        assert pushed_x > 1.0 + beta * constants.c * dt
        reflected_u, reflected_v = _lorentz_reflection(u, beta)
        collision_time = (1.0 - x0) / (v - beta * constants.c)
        assert 0.0 < collision_time < dt
        if reason == "positive_reflected_momentum":
            assert reflected_u[0] > 0.0
        else:
            assert reflected_u[0] < 0.0
            assert (
                1.0
                + beta * constants.c * collision_time
                + reflected_v[0] * (dt - collision_time)
                > 1.0
            )
    _add_particle(pc, x0, u)
    sim.step(1)
    assert pc.size == 0


def test_moving_wall_does_not_change_interior_particles():
    sim = _make_wall_sim(-0.4)
    pc = sim.particles.get("neutral")
    dt = sim.extension.warpx.getdt(0)
    u = constants.c * np.array([0.4, -0.2, 0.1])
    v = u / np.sqrt(1.0 + np.dot(u, u) / constants.c**2)
    # The advancing virtual wall passes this particle, but its pushed position
    # stays inside the fixed computational domain. The EPOCH boundary rule
    # only considers particles that have escaped through x_max.
    x0 = 1.0 - 0.6 * constants.c * dt
    expected_x = x0 + v[0] * dt
    assert 1.0 - 0.4 * constants.c * dt < expected_x < 1.0
    _add_particle(pc, x0, u)
    sim.step(1)
    assert pc.size == 1
    state = _particle_state(pc)
    np.testing.assert_allclose(state["x"], expected_x, rtol=rtol(), atol=0.0)
    for name, expected in zip(("ux", "uy", "uz"), u):
        np.testing.assert_allclose(state[name], expected, rtol=rtol(), atol=0.0)

/* Copyright 2021 David Grote, Remi Lehe, Revathi Jambunathan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLEBOUNDARIES_K_H_
#define WARPX_PARTICLEBOUNDARIES_K_H_

#include "ParticleBoundaries.H"
#include "Initialization/SampleGaussianFluxDistribution.H"
#include "Utils/WarpXConst.H"

#include <AMReX_AmrCore.H>

#include <cmath>
#include <limits>

namespace ApplyParticleBoundaries {
    using namespace amrex::literals;

    /** Reflect an escaped massive particle from a virtual wall that starts at
     * x_max each step and moves at beta*c during that step. The computational
     * boundary remains fixed: particles that cannot return inside are lost.
     * WarpX stores u = p/m, so energy/(m*c) = sqrt(c*c + u*u).
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void
    apply_moving_wall (amrex::ParticleReal& x, amrex::Real const xmax,
                       amrex::ParticleReal& ux, amrex::ParticleReal const uy,
                       amrex::ParticleReal const uz, amrex::Real const dt,
                       amrex::Real const beta, amrex::Real const gamma_sq,
                       bool& particle_lost)
    {
        amrex::Real const wall_v = beta * PhysConst::c;
        if (dt <= 0._rt || x <= xmax + wall_v * dt) {
            particle_lost = true;
            return;
        }

        amrex::Real const energy_c = std::sqrt(
            PhysConst::c * PhysConst::c + ux*ux + uy*uy + uz*uz);
        amrex::Real const v_initial = PhysConst::c * ux / energy_c;
        // A particle must catch up to the wall to reflect. This also protects
        // the denominator when the particle and wall have identical speeds.
        if (v_initial <= wall_v) {
            particle_lost = true;
            return;
        }

        amrex::Real t_refl = (xmax - x + v_initial * dt) / (v_initial - wall_v);
        amrex::Real const time_tol = 16._rt *
            std::numeric_limits<amrex::ParticleReal>::epsilon() * dt;
        if (t_refl < -time_tol || t_refl > dt + time_tol) {
            // For example, a particle injected outside the domain did not
            // necessarily cross the wall during this position push.
            particle_lost = true;
            return;
        }
        if (t_refl < 0._rt) { t_refl = 0._rt; }
        if (t_refl > dt) { t_refl = dt; }

        ux = -gamma_sq * ((1._rt + beta*beta) * ux - 2._rt * beta * energy_c);
        amrex::Real const v_reflected = PhysConst::c * ux / std::sqrt(
            PhysConst::c * PhysConst::c + ux*ux + uy*uy + uz*uz);
        x = xmax + wall_v * t_refl + v_reflected * (dt - t_refl);
        particle_lost = (ux > 0._prt || x > xmax);
    }

    /* \brief Applies the boundary condition on a specific axis
     *        This is called by apply_boundaries.
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void
    apply_boundary (amrex::ParticleReal& x, amrex::Real xmin, amrex::Real xmax,
                    bool& change_sign_ux, bool& rethermalize_x, bool& particle_lost,
                    ParticleBoundaryType xmin_bc, ParticleBoundaryType xmax_bc,
                    amrex::Real refl_probability_xmin, amrex::Real refl_probability_xmax,
                    amrex::RandomEngine const& engine )
    {
        if (x < xmin) {
            if (xmin_bc == ParticleBoundaryType::Open) {
                particle_lost = true;
            }
            else if (xmin_bc == ParticleBoundaryType::Absorbing) {
                if (refl_probability_xmin == 0 || amrex::Random(engine) > refl_probability_xmin) {
                    particle_lost = true;
                }
                else
                {
                    x = 2*xmin - x;
                    change_sign_ux = true;
                }
            }
            else if (xmin_bc == ParticleBoundaryType::Reflecting) {
                x = 2*xmin - x;
                change_sign_ux = true;
            }
            else if (xmin_bc == ParticleBoundaryType::Thermal) {
                x = 2*xmin - x;
                rethermalize_x = true;
            }
        }
        else if (x > xmax) {
            if (xmax_bc == ParticleBoundaryType::Open) {
                particle_lost = true;
            }
            else if (xmax_bc == ParticleBoundaryType::Absorbing) {
                if (refl_probability_xmax == 0 || amrex::Random(engine) > refl_probability_xmax) {
                    particle_lost = true;
                }
                else
                {
                    x = 2*xmax - x;
                    change_sign_ux = true;
                }
            }
            else if (xmax_bc == ParticleBoundaryType::Reflecting) {
                x = 2*xmax - x;
                change_sign_ux = true;
            }
            else if (xmax_bc == ParticleBoundaryType::Thermal) {
                x = 2*xmax - x;
                rethermalize_x = true;
            }
        }
    }

    /* \brief Thermalize particles that have been identified to cross the boundary.
     *        The normal component samples from a half-Maxwellian,
     *        and the two tangential components will sample from full Maxwellian disbutions
     *        with thermal velocity uth
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void thermalize_boundary_particle (amrex::ParticleReal& u_norm, amrex::ParticleReal& u_tang1,
                                       amrex::ParticleReal& u_tang2, amrex::Real uth,
                                       amrex::RandomEngine const& engine)
    {
        u_tang1 = (uth > 0._rt) ? PhysConst::c * amrex::RandomNormal(0._rt, uth, engine) : 0._rt;
        u_tang2 = (uth > 0._rt) ? PhysConst::c * amrex::RandomNormal(0._rt, uth, engine) : 0._rt;
        u_norm = (uth > 0._rt) ? std::copysign(1._prt, -u_norm) * PhysConst::c * generateGaussianFluxDist(0._rt, uth, engine) : 0._rt;
    }


    /* \brief Applies absorbing, reflecting or thermal boundary condition to the input particles, along all axis.
     *        For reflecting boundaries, the position of the particle is changed appropriately and
     *        the sign of the velocity is changed (depending on the reflect_all_velocities flag).
     *        For absorbing, a flag is set whether the particle has been lost (it is up to the calling
     *        code to take appropriate action to remove any lost particles). Absorbing boundaries can
     *        be given a reflection coefficient for stochastic reflection of particles, this
     *        coefficient is zero by default.
     *        For thermal boundaries, the particle is first reflected and the position of the particle
     *        is changed appropriately.
     *        Note that periodic boundaries are handled in AMReX code.
     *
     * \param x, y, z: particle position. For RZ and RCYLINDER, these are the mapped
     *                  coordinates (r, theta, z); for RSPHERE, they are (r, theta, phi).
     * \param gridmin, gridmax: locations of the lower and upper boundaries in mapped space
     * \param ux, uy, uz: particle momentum. For RZ and RCYLINDER, these are the mapped
     *                    components (u_r, u_theta, u_z); for RSPHERE, they are
     *                    (u_r, u_theta, u_phi).
     * \param particle_lost: output, flags whether the particle was lost
     * \param dt: duration of the position push (for moving-wall reflection)
     * \param boundaries: object with boundary condition settings
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void
    apply_boundaries ([[maybe_unused]] amrex::ParticleReal& x,
                      [[maybe_unused]] amrex::ParticleReal& y,
                      [[maybe_unused]] amrex::ParticleReal& z,
                      amrex::XDim3 gridmin, amrex::XDim3 gridmax,
                      amrex::ParticleReal& ux, amrex::ParticleReal& uy, amrex::ParticleReal& uz,
                      [[maybe_unused]] amrex::Real const dt,
                      bool& particle_lost,
                      ParticleBoundaries::ParticleBoundariesData const& boundaries,
                      amrex::RandomEngine const& engine)
    {
        bool change_sign_ux = false;
        bool change_sign_uy = false;
        bool change_sign_uz = false;

#ifndef WARPX_DIM_1D_Z
        if (boundaries.xmax_bc == ParticleBoundaryType::Moving_Wall && x > gridmax.x) {
            apply_moving_wall(x, gridmax.x, ux, uy, uz, dt,
                              boundaries.moving_wall_beta, boundaries.moving_wall_gamma_sq,
                              particle_lost);
            if (particle_lost) { return; }
        }
        bool rethermalize_x = false; // stores if particle crosses x boundary and needs to be thermalized
        apply_boundary(x, gridmin.x, gridmax.x, change_sign_ux, rethermalize_x, particle_lost,
                       boundaries.xmin_bc, boundaries.xmax_bc,
                       boundaries.reflection_model_xlo(-ux), boundaries.reflection_model_xhi(ux),
                       engine);
        if (rethermalize_x) {
            thermalize_boundary_particle(ux, uy, uz, boundaries.m_uth, engine);
        }
#endif
#ifdef WARPX_DIM_3D
        bool rethermalize_y = false; // stores if particle crosses y boundary and needs to be thermalized
        apply_boundary(y, gridmin.y, gridmax.y, change_sign_uy, rethermalize_y, particle_lost,
                       boundaries.ymin_bc, boundaries.ymax_bc,
                       boundaries.reflection_model_ylo(-uy), boundaries.reflection_model_yhi(uy),
                       engine);
        if (rethermalize_y) {
            thermalize_boundary_particle(uy, uz, ux, boundaries.m_uth, engine);
        }
#endif
#if defined(WARPX_ZINDEX)
        bool rethermalize_z = false; // stores if particle crosses z boundary and needs to be thermalized
        apply_boundary(z, gridmin.z, gridmax.z, change_sign_uz, rethermalize_z, particle_lost,
                       boundaries.zmin_bc, boundaries.zmax_bc,
                       boundaries.reflection_model_zlo(-uz), boundaries.reflection_model_zhi(uz),
                       engine);
        if (rethermalize_z) {
            thermalize_boundary_particle(uz, ux, uy, boundaries.m_uth, engine);
        }
#endif

        if (boundaries.reflect_all_velocities && (change_sign_ux | change_sign_uy | change_sign_uz)) {
            change_sign_ux = true;
            change_sign_uy = true;
            change_sign_uz = true;
        }

        if (change_sign_ux) { ux = -ux; }
        if (change_sign_uy) { uy = -uy; }
        if (change_sign_uz) { uz = -uz; }
    }

}
#endif //WARPX_PARTICLEBOUNDARIES_K_H_

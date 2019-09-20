#pragma once
#include "parameters.h"
#include "thermodynamic_functions.h"
#include "advection_interpolation.h"
#include "entropies.h"

// CLIMA microphysics parameters
#define C_drag 0.55
#define rho_cloud_liq 1e3
#define MP_n_0 16 * 1e6
#define tau_cond_evap 10
#define q_liq_threshold 5e-4
#define tau_acnv 1e3
#define E_col 0.8
#define nu_air 1.6e-5
#define K_therm 2.4e-2
#define D_vapor 2.26e-5
#define a_vent 1.5
#define b_vent 0.53
// additional parameters for pycles implementation
#define max_iter 10
#define microph_eps 1e-3

// see CLIMA Microphysics docs for info

double CLIMA_terminal_velocity_single_drop_coeff(double rho){

    return sqrt(8./3 / C_drag * (rho_cloud_liq / rho - 1.));
}

double CLIMA_terminal_velocity(double q_rai, double rho){

    double v_c = terminal_velocity_single_drop_coeff(rho);
    double gamma_9_2 = 11.631728396567448;

    double term_vel = 0.;

    if(q_rai > 0){
        double lambda_param = pow(8. * pi * rho_cloud_liq * MP_n_0 / rho / q_rai, 0.25);
        term_vel = gamma_9_2 * v_c / 6. * sqrt(g / lambda_param);
    }

    return term_vel;
}

void CLIMA_conv_q_liq_to_q_rai_acnv(double q_liq, double* qr_tendency_aut){

    *qr_tendency_aut = fmax(0., q_liq - q_liq_threshold) / tau_acnv;
    return;
}

void CLIMA_conv_q_liq_to_q_rai_accr(double q_liq, double q_rai, double rho,
                                    double* qr_tendency_acc){

    double v_c = terminal_velocity_single_drop_coeff(rho);
    double gamma_7_2 = 3.3233509704478426;

    double accr_coeff = gamma_7_2 * pow(8., -7./8) * pow(pi, 1./8) * v_c * E_col *
                        pow(rho / rho_cloud_liq, 7./8);

    *qr_tendency_acc = accr_coeff * pow(MP_n_0, 1./8) * sqrt(g) * q_liq * pow(q_rai, 7./8);
    return;
}

void CLIMA_conv_q_rai_to_q_vap(double q_rai, double q_tot, double q_liq,
                               double T, double p, double rho,
                               struct LookupStruct *LT,
                               double (*lam_fp)(double),
                               double (*L_fp)(double, double),
                               double* qr_tendency_evp){

    double gamma_11_4 = 1.6083594219855457;
    double v_c = terminal_velocity_single_drop_coeff(rho);
    double N_Sc = nu_air / D_vapor;

    double av_param = sqrt(2. * pi) * a_vent * sqrt(rho / rho_cloud_liq);
    double bv_param = pow(2., 7./16) * gamma_11_4 * pow(pi, 5./16) * b_vent *
                      pow(N_Sc, 1./3) * sqrt(v_c) * pow(rho / rho_cloud_liq, 11./16);

    double lam = lam_fp(T);
    double L = L_fp(T, lam);
    double pv_s = lookup(LT, T);

    double qv_sat = qv_star_c(p, q_tot, pv_s);
    double q_v = q_tot - q_liq;
    double S = q_v/qv_sat - 1;

    double G_param = 1. / (L / K_therm / T * (L / Rv / T - 1.) +
                           Rv * T / D_vapor / pv_s
                          );

    double F_param = av_param * sqrt(q_rai) +
                     bv_param * pow(g, 0.25) / pow(MP_n_0, 3./16) /
                       sqrt(nu_air) * pow(q_rai, 11./16);

    *qr_tendency_evp = S * F_param * G_param * sqrt(MP_n_0) / rho;
    return;
}

void CLIMA_sedimentation_velocity_rain(const struct DimStruct *dims,
                                       double* restrict density,
                                       double* restrict qr,
                                       double* restrict qr_velocity){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = 0;
    const ssize_t jmin = 0;
    const ssize_t kmin = 0;
    const ssize_t imax = dims->nlg[0];
    const ssize_t jmax = dims->nlg[1];
    const ssize_t kmax = dims->nlg[2];

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin-1; k<kmax+1; k++){
                const ssize_t ijk = ishift + jshift + k;

                double qr_tmp = fmax(qr[ijk], 0.0);

                qr_velocity[ijk] = -fmin(CLIMA_terminal_velocity(qr_tmp, density[k]),
                                         10.0
                                    );
            }
        }
    }

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax-1 ; k++){
                const ssize_t ijk = ishift + jshift + k;

                qr_velocity[ijk] = interp_2(qr_velocity[ijk], qr_velocity[ijk+1]) ;

            }
        }
    }
    return;
}

void CLIMA_microphysics_sources(const struct DimStruct *dims, struct LookupStruct *LT,
                                double (*lam_fp)(double), double (*L_fp)(double, double),
                                double* restrict density, double* restrict p0,
                                double* restrict temperature, double* restrict qt,
                                double* restrict ql, double* restrict qr, double dt,
                                double* restrict qr_tendency_micro,
                                double* restrict qr_tendency){

    double qr_tendency_tmp, ql_tendency_tmp;
    double qr_tendency_aut, qr_tendency_acc,  qr_tendency_evp;
    double sat_ratio;

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;

                double qr_tmp = fmax(qr[ijk],0.0);
                double qt_tmp = fmax(qt[ijk], 0.0);
                double ql_tmp = fmax(ql[ijk], 0.0);
                double qv_tmp = fmax(qt_tmp - ql_tmp, 0.0);

                double time_added = 0.0, dt_, rate;
                ssize_t iter_count = 0;
                do{
                    iter_count += 1;

                    qr_tendency_aut = 0.0;
                    qr_tendency_acc = 0.0;
                    qr_tendency_evp = 0.0;

                    //compute the source terms
                    CLIMA_conv_q_liq_to_q_rai_acnv(ql_tmp, &qr_tendency_aut);
                    CLIMA_conv_q_liq_to_q_rai_accr(
                        ql_tmp, qr_tmp, density[k], &qr_tendency_acc
                    );
                    CLIMA_conv_q_rai_to_q_vap(
                        qr_tmp, qt_tmp, ql_tmp, temperature[ijk], p0[k],
                        density[k], LT, lam_fp, L_fp, &qr_tendency_evp
                    );

                    //find the maximum substep time:
                    dt_ = dt - time_added;
                    //... check the source term magnitudes ...
                    qr_tendency_tmp = qr_tendency_aut + qr_tendency_acc + qr_tendency_evp;
                    ql_tendency_tmp = -qr_tendency_aut - qr_tendency_acc;
                    //... adjust the rates if necessary (factor of 1.05 is ad-hoc)
                    rate = 1.05 * ql_tendency_tmp * dt_ / (-fmax(ql_tmp, microph_eps));
                    rate = fmax(1.05 * qr_tendency_tmp * dt_ / (-fmax(qr_tmp, microph_eps)), rate);
                    if(rate > 1.0 && iter_count < max_iter){
                        //Limit the timestep, but don't allow it to become vanishingly small
                        //Don't adjust if we have reached the maximum iteration number
                        dt_ = fmax(dt_/rate, 1.0e-3);
                    }

                    //integrate forward in time
                    ql_tmp += ql_tendency_tmp * dt_;
                    qr_tmp += qr_tendency_tmp * dt_;
                    qv_tmp -= qr_tendency_evp * dt_;

                    qr_tmp = fmax(qr_tmp,0.0);
                    ql_tmp = fmax(ql_tmp,0.0);
                    qt_tmp = ql_tmp + qv_tmp;

                    time_added += dt_ ;

                }while(time_added < dt);
                qr_tendency_micro[ijk] = (qr_tmp - qr[ijk])/dt;
                qr_tendency[ijk] += qr_tendency_micro[ijk];
            }
        }
    }
    return;
}

void CLIMA_qt_source_formation(const struct DimStruct *dims,
                               double* restrict qr_tendency,
                               double* restrict qt_tendency){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;

                qt_tendency[ijk] += -qr_tendency[ijk];
            }
        }
    }
    return;
}

void CLIMA_entropy_source_formation(const struct DimStruct *dims, struct LookupStruct *LT,
                                    double (*lam_fp)(double), double (*L_fp)(double, double),
                                    double* restrict p0,
                                    double* restrict T, double* restrict Twet,
                                    double* restrict qt, double* restrict qv,
                                    double* restrict qr_tendency,
                                    double* restrict entropy_tendency){

    // Source terms of entropy related to microphysics.
    // See Pressel et al. 2015, Eq. 49-54
    // (copied from SB implementation)

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;

    //entropy tendencies from formation or evaporation of precipitation
    //we use fact that P = d(qr)/dt > 0, E =  d(qr)/dt < 0
    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;

                const double lam_T = lam_fp(T[ijk]);
                const double L_fp_T = L_fp(T[ijk],lam_T);
                const double lam_Tw = lam_fp(Twet[ijk]);
                const double L_fp_Tw = L_fp(Twet[ijk],lam_Tw);
                const double pv_star_T = lookup(LT, T[ijk]);
                const double pv_star_Tw = lookup(LT,Twet[ijk]);
                const double pv = pv_c(p0[k], qt[ijk], qv[ijk]);
                const double pd = p0[k] - pv;
                const double sd_T = sd_c(pd, T[ijk]);
                const double sv_star_T = sv_c(pv_star_T,T[ijk] );
                const double sv_star_Tw = sv_c(pv_star_Tw, Twet[ijk]);
                const double S_P = sd_T - sv_star_T + L_fp_T/T[ijk];
                const double S_E = sv_star_Tw - L_fp_Tw/Twet[ijk] - sd_T;
                const double S_D = -Rv * log(pv/pv_star_T) + cpv * log(T[ijk]/Twet[ijk]);

                entropy_tendency[ijk] += S_P         * 0.5 * (qr_tendency[ijk] + fabs(qr_tendency[ijk])) -
                                         (S_E + S_D) * 0.5 * (qr_tendency[ijk] - fabs(qr_tendency[ijk]));
            }
        }
    }
    return;
}

void CLIMA_entropy_source_heating(const struct DimStruct *dims,
                               double* restrict T, double* restrict Twet,
                               double* restrict qr,
                               double* restrict w_qr, double* restrict w,
                               double* restrict entropy_tendency){

    // (copied from SB implementation)

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;
    const double dzi = 1.0/dims->dx[2];

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;

                entropy_tendency[ijk] += qr[ijk] * (fabs(w_qr[ijk]) - w[ijk]) *
                                         cl * (Twet[ijk+1] - Twet[ijk]) * dzi / T[ijk];
            }
        }
    }
    return;
}

void CLIMA_entropy_source_drag(const struct DimStruct *dims,
                               double* restrict T,
                               double* restrict qr, double* restrict w_qr,
                               double* restrict entropy_tendency){

    // (copied from SB implementation)

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;

                entropy_tendency[ijk] += g * qr[ijk] * fabs(w_qr[ijk]) / T[ijk];
            }
        }
    }
    return;
}

// diagnostics for output
void CLIMA_autoconversion_rain_wrapper(const struct DimStruct *dims,
                                       double* restrict ql,
                                       double* restrict qr_tendency){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;

                //compute the source terms
                double ql_tmp = fmax(ql[ijk], 0.0);

                CLIMA_conv_q_liq_to_q_rai_acnv(ql_tmp, &qr_tendency[ijk]);
            }
        }
    }
    return;
}
void CLIMA_accretion_rain_wrapper(const struct DimStruct *dims,
                                  double* restrict density,
                                  double* restrict ql,
                                  double* restrict qr,
                                  double* restrict qr_tendency){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;

                const double ql_tmp = fmax(ql[ijk], 0.0);
                const double qr_tmp = fmax(qr[ijk], 0.0);

                CLIMA_conv_q_liq_to_q_rai_accr(ql_tmp, qr_tmp, density[k],
                                               &qr_tendency[ijk]);
            }
        }
    }
    return;
}
void CLIMA_evaporation_rain_wrapper(const struct DimStruct *dims,
                                    struct LookupStruct *LT,
                                    double (*lam_fp)(double),
                                    double (*L_fp)(double, double),
                                    double* restrict density,
                                    double* restrict p0,
                                    double* restrict temperature,
                                    double* restrict qt,
                                    double* restrict ql,
                                    double* restrict qr,
                                    double* restrict qr_tendency){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;

                const double qr_tmp = fmax(qr[ijk], 0.0);
                const double qt_tmp = fmax(qt[ijk], 0.0);
                const double ql_tmp = fmax(ql[ijk], 0.0);
                const double qv = qt[ijk] - ql[ijk];

                CLIMA_conv_q_rai_to_q_vap(qr_tmp, qt_tmp, ql_tmp, temperature[ijk],
                                          p0[k], density[k], LT, lam_fp, L_fp,
                                          &qr_tendency[ijk]);
            }
        }
    }
    return;
}

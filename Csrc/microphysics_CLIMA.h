#pragma once
#include "parameters.h"
#include "thermodynamic_functions.h"
#include "advection_interpolation.h"
#include "entropies.h"

// see CLIMA CloudMicrophysics.jl documentation
// CLIMA microphysics parameters - size distributions and relations
#define rho_cloud_liq 1e3
#define rho_cloud_ice 916.7

#define r0_ice 1e-5
#define r0_rai 1e-3
#define r0_sno 1e-3

#define n0_ice 2e7
#define me_ice 3.0
#define m0_ice 4.0/3 * pi * rho_cloud_ice * pow(r0_ice, me_ice)
#define Chi_m_ice 1.0
#define Delta_m_ice 0.0

// TODO - run simulations with default edmf parameters
// tau_acnv_rai = 2.5e3
#define q_liq_threshold 5e-4
#define q_ice_threshold 1e-6
#define tau_acnv_rai 1e3
#define tau_acnv_sno 1e2

#define n0_rai 8e6 * 2
#define me_rai 3.0
#define ae_rai 2.0
#define ve_rai 0.5
#define m0_rai 4.0/3 * pi * rho_cloud_liq * pow(r0_rai, me_rai)
#define a0_rai pi * pow(r0_rai, ae_rai)
#define Chi_m_rai 1.0
#define Delta_m_rai 0.0
#define Chi_a_rai 1.0
#define Delta_a_rai 0.0
#define Chi_v_rai 1.0
#define Delta_v_rai 0.0

#define mu_sno 4.36 * 1e9
#define nu_sno 0.63
#define me_sno 2.0
#define ae_sno 2.0
#define ve_sno 0.25
#define m0_sno 1e-1 * pow(r0_sno, me_sno)
#define a0_sno 0.3 * pi * pow(r0_sno, ae_sno)
#define v0_sno pow(2.0, 9.0/4) * pow(r0_sno, ve_sno)
#define Chi_m_sno 1.0
#define Delta_m_sno 0.0
#define Chi_a_sno 1.0
#define Delta_a_sno 0.0
#define Chi_v_sno 1.0
#define Delta_v_sno 0.0

// CLIMA microphysics parameters - processes
#define C_drag 0.55
#define K_therm 2.4e-2
#define D_vapor 2.26e-5
#define nu_air 1.6e-5
#define N_sc 1.6/2.26

#define a_vent_rai 1.5
#define b_vent_rai 0.53
#define a_vent_sno 0.65
#define b_vent_sno 0.44

#define E_liq_rai 0.8
#define E_liq_sno 0.1
#define E_ice_rai 1.0
#define E_ice_sno 0.1
#define E_rai_sno 1.0

#define T_freeze 273.15

#define cv_l 4181.0
#define LH_f0 2.8344e6 - 2.5008e6
#define cp_l 4181.0
#define cp_i 2100.0
#define T_0 273.16

// additional parameters for pycles implementation
#define max_iter 10
#define microph_eps 1e-3

double CLIMA_latent_heat_fusion(double T){
  return LH_f0 + (cp_l - cp_i) * (T - T_0);
}

double CLIMA_v0_rai(double rho){
  return sqrt(8./3./ C_drag * (rho_cloud_liq / rho - 1.) * g * r0_rai);
}

double CLIMA_n0_sno(double q_sno, double rho){
  return mu_sno * pow(rho * fmax(0., q_sno), nu_sno);
}

double CLIMA_lambda(double q, double rho, double n0, double m0, double me,
              double r0, double Chi_m, double Delta_m){
  if(q > 0.){
    return pow(
      Chi_m * m0 * n0 * tgamma(me + Delta_m + 1.) / rho / q / pow(r0, me + Delta_m),
      1. / (me + Delta_m + 1.)
    );
  }
  else{
    return 0.;
  }
}

double CLIMA_terminal_velocity_rai(double rho, double q_rai){
  if(q_rai > 0.){
    double lambda_rai = CLIMA_lambda(q_rai, rho, n0_rai, m0_rai, me_rai, r0_rai, Chi_m_rai, Delta_m_rai);

    return Chi_v_rai * CLIMA_v0_rai(rho) * pow(lambda_rai * r0_rai, - ve_rai - Delta_v_rai) *
           tgamma(me_rai + ve_rai + Delta_m_rai + Delta_v_rai + 1.) /
           tgamma(me_rai + Delta_m_rai + 1.);
  }
  else{
    return 0.;
  }
}

double CLIMA_terminal_velocity_sno(double rho, double q_sno){
  if(q_sno > 0.){
    double lambda_sno = CLIMA_lambda(q_sno, rho, CLIMA_n0_sno(q_sno, rho), m0_sno, me_sno, r0_sno, Chi_m_sno, Delta_m_sno);

    return Chi_v_sno * v0_sno * pow(lambda_sno * r0_sno, - ve_sno - Delta_v_sno) *
           tgamma(me_sno + ve_sno + Delta_m_sno + Delta_v_sno + 1.) /
           tgamma(me_sno + Delta_m_sno + 1.);
  }
  else{
    return 0.;
  }
}

void CLIMA_conv_q_liq_to_q_rai(double _q_liq, double* qr_tendency_aut){
  double q_liq = fmax(0., _q_liq);
  *qr_tendency_aut = fmax(0., q_liq - q_liq_threshold) / tau_acnv_rai;
  return;
}

void CLIMA_conv_q_ice_to_q_sno_no_supersat(double _q_ice, double* qs_tendency_aut){
  double q_ice = fmax(0., _q_ice);
  *qs_tendency_aut = fmax(0., q_ice - q_ice_threshold) / tau_acnv_sno;
  return;
}

void CLIMA_accretion(double _q_liq, double _q_ice, double _q_rai, double _q_sno,
                     double rho, double T,
                     double* ql_tendency_acc,
                     double* qi_tendency_acc,
                     double* qr_tendency_acc,
                     double* qs_tendency_acc){

    double q_liq = fmax(0., _q_liq);
    double q_ice = fmax(0., _q_ice);
    double q_rai = fmax(0., _q_rai);
    double q_sno = fmax(0., _q_sno);

    double lambda_ice = CLIMA_lambda(q_ice, rho, n0_ice,                   m0_ice, me_ice, r0_ice, Chi_m_ice, Delta_m_ice);
    double lambda_rai = CLIMA_lambda(q_rai, rho, n0_rai,                   m0_rai, me_rai, r0_rai, Chi_m_rai, Delta_m_rai);
    double lambda_sno = CLIMA_lambda(q_sno, rho, CLIMA_n0_sno(q_sno, rho), m0_sno, me_sno, r0_sno, Chi_m_sno, Delta_m_sno);

    double v_rai = CLIMA_terminal_velocity_rai(rho, q_rai);
    double v_sno = CLIMA_terminal_velocity_sno(rho, q_sno);

    double tmp = 0.;

    *qr_tendency_acc = 0.;
    *ql_tendency_acc = 0.;
    *qs_tendency_acc = 0.;
    *qi_tendency_acc = 0.;

    // accretion qr ql
    if(q_rai > 0. && q_liq > 0.){
        tmp =
          q_liq * E_liq_rai * n0_rai * a0_rai * CLIMA_v0_rai(rho) * Chi_a_rai * Chi_v_rai / lambda_rai *
          tgamma(ae_rai + ve_rai + Delta_a_rai + Delta_v_rai + 1.) /
          pow(lambda_rai * r0_rai, ae_rai + ve_rai + Delta_a_rai + Delta_v_rai);

        *qr_tendency_acc += tmp;
        *ql_tendency_acc -= tmp;
    }

    // accretion qs qi
    if(q_sno > 0. && q_ice > 0.){
        tmp =
          q_ice * E_ice_sno * CLIMA_n0_sno(q_sno, rho) * a0_sno * v0_sno * Chi_a_sno * Chi_v_sno / lambda_sno *
          tgamma(ae_sno + ve_sno + Delta_a_sno + Delta_v_sno + 1.) /
          pow(lambda_sno * r0_sno, ae_sno + ve_sno + Delta_a_sno + Delta_v_sno);

        *qs_tendency_acc += tmp;
        *qi_tendency_acc -= tmp;
    }

    // accretion qr qi
    if(q_rai > 0. && q_ice > 0.){
        double acc_q_ice_q_rai_ice_sink =
          q_ice * E_ice_rai * n0_rai * a0_rai * CLIMA_v0_rai(rho) * Chi_a_rai * Chi_v_rai / lambda_rai *
          tgamma(ae_rai + ve_rai + Delta_a_rai + Delta_v_rai + 1.) /
          pow(lambda_rai * r0_rai, ae_rai + ve_rai + Delta_a_rai + Delta_v_rai);
        double acc_q_ice_q_rai_rain_sink = E_ice_rai / rho * n0_rai * n0_ice * m0_rai * a0_rai * CLIMA_v0_rai(rho) *
          Chi_m_rai * Chi_a_rai * Chi_v_rai / lambda_ice / lambda_rai *
          tgamma(me_rai + ae_rai + ve_rai + Delta_m_rai + Delta_a_rai + Delta_v_rai +1.) /
          pow(r0_rai * lambda_rai, me_rai + ae_rai + ve_rai + Delta_m_rai + Delta_a_rai + Delta_v_rai);

        *qr_tendency_acc -= acc_q_ice_q_rai_rain_sink;
        *qs_tendency_acc += acc_q_ice_q_rai_rain_sink + acc_q_ice_q_rai_ice_sink;
        *qi_tendency_acc -= acc_q_ice_q_rai_ice_sink;
    }

    // accretion qs ql
    if(q_sno > 0. && q_liq > 0.){
        tmp = -q_liq * E_liq_sno * CLIMA_n0_sno(q_sno, rho) * a0_sno * v0_sno * Chi_a_sno * Chi_v_sno / lambda_sno *
          tgamma(ae_sno + ve_sno + Delta_a_sno + Delta_v_sno + 1.) /
          pow(lambda_sno * r0_sno, ae_sno + ve_sno + Delta_a_sno + Delta_v_sno);

        if(T>T_freeze){
          double L_f = CLIMA_latent_heat_fusion(T);
          double alpha = cv_l / L_f * (T - T_freeze);
          *qs_tendency_acc += tmp * alpha;
          *qr_tendency_acc -= tmp * (1. + alpha);
          *ql_tendency_acc += tmp;
        }
        else{
          *qs_tendency_acc -= tmp;
          *ql_tendency_acc += tmp;
        }
    }

    // accretion qr qs
    if(q_sno > 0. && q_rai > 0.){
        if(T>T_freeze){
           tmp = pi / rho * n0_rai * CLIMA_n0_sno(q_sno, rho) * m0_sno * Chi_m_sno * E_rai_sno *
            fabs(v_rai - v_sno) / pow(r0_sno, me_sno + Delta_m_sno) * (
              2. * tgamma(me_sno + Delta_m_sno + 1.) / pow(lambda_rai, 3) / pow(lambda_sno, me_sno + Delta_m_sno + 1) +
              2. * tgamma(me_sno + Delta_m_sno + 2.) / pow(lambda_rai, 2) / pow(lambda_sno, me_sno + Delta_m_sno + 2) +
                   tgamma(me_sno + Delta_m_sno + 3.) / lambda_rai /         pow(lambda_sno, me_sno + Delta_m_sno + 3));
           *qr_tendency_acc += tmp;
           *qs_tendency_acc -= tmp;
        }
        else{
          tmp = pi / rho * CLIMA_n0_sno(q_sno, rho) * n0_rai * m0_rai * Chi_m_rai * E_rai_sno *
            fabs(v_rai - v_sno) / pow(r0_rai, me_rai + Delta_m_rai) * (
              2. * tgamma(me_rai + Delta_m_rai + 1.) / pow(lambda_sno, 3) / pow(lambda_rai, me_rai + Delta_m_rai + 1) +
              2. * tgamma(me_rai + Delta_m_rai + 2.) / pow(lambda_sno, 2) / pow(lambda_rai, me_rai + Delta_m_rai + 2) +
                   tgamma(me_rai + Delta_m_rai + 3.) / lambda_sno /         pow(lambda_rai, me_rai + Delta_m_rai + 3));
          *qs_tendency_acc += tmp;
          *qr_tendency_acc -= tmp;
        }
    }
    return;
}

void CLIMA_rain_evaporation(double q_tot, double q_liq, double q_ice, double q_rai, double rho, double T, double p0,
                            struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
                            double* qr_tendency_evp){

  double lam = lam_fp(T);
  double L = L_fp(T, lam);
  double pv_s = lookup(LT, T);

  double qv_sat = qv_star_c(p0, q_tot, pv_s);
  double q_v = fmax(0., q_tot - q_liq - q_ice);
  double S = q_v/qv_sat - 1;

  if(q_rai > 0. && S < 0.){

    double G = 1. / (L / K_therm / T * (L / Rv / T - 1.) + Rv * T / D_vapor / pv_s);

    double lambda_rai = CLIMA_lambda(q_rai, rho, n0_rai, m0_rai, me_rai, r0_rai, Chi_m_rai, Delta_m_rai);

    *qr_tendency_evp = fmin(0.,
      4. * pi * n0_rai / rho * S * G / pow(lambda_rai, 2) *
      (a_vent_rai + b_vent_rai * pow(nu_air / D_vapor, 1./3.) /
        pow(r0_rai * lambda_rai, (ve_rai + Delta_v_rai) / 2.) *
        pow(2. * CLIMA_v0_rai(rho) * Chi_v_rai / nu_air / lambda_rai, 1./2.) *
        tgamma((ve_rai + Delta_v_rai + 5.) / 2.)
      ));
  }
  else{
    *qr_tendency_evp = 0.;
  }
  return;
}

void CLIMA_snow_deposition_sublimation(double q_tot, double q_liq, double q_ice, double q_sno, double rho, double T, double p0,
                            struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
                            double* qs_tendency_dep_sub){
  double lam = lam_fp(T);
  double L = L_fp(T, lam);
  double pv_s = lookup(LT, T);

  double qv_sat = qv_star_c(p0, q_tot, pv_s);
  double q_v = fmax(0.0, q_tot - q_liq - q_ice);
  double S = q_v/qv_sat - 1;

  if(q_sno > 0.){

    double G = 1. / (L / K_therm / T * (L / Rv / T - 1.) + Rv * T / D_vapor / pv_s);

    double lambda_sno = CLIMA_lambda(q_sno, rho, CLIMA_n0_sno(q_sno, rho), m0_sno, me_sno, r0_sno, Chi_m_sno, Delta_m_sno);

    *qs_tendency_dep_sub =
      4. * pi * CLIMA_n0_sno(q_sno, rho) / rho * S * G / pow(lambda_sno, 2) *
      (a_vent_sno + b_vent_sno * pow(nu_air / D_vapor, 1./3.) /
        pow(r0_sno * lambda_sno, (ve_sno + Delta_v_sno) / 2.) *
        pow(2. * v0_sno * Chi_v_sno / nu_air / lambda_sno, 1./2.) *
        tgamma((ve_sno + Delta_v_sno + 5.) / 2.)
      );
  }
  else{
    *qs_tendency_dep_sub = 0.;
  }
  return;
}

void CLIMA_snow_melt(double q_sno, double rho, double T, double* qs_tendency_melt){

  if(q_sno > 0. && T > T_freeze){

    double L_f = CLIMA_latent_heat_fusion(T);

    double lambda_sno = CLIMA_lambda(q_sno, rho, CLIMA_n0_sno(q_sno, rho), m0_sno, me_sno, r0_sno, Chi_m_sno, Delta_m_sno);

    *qs_tendency_melt =
      4. * pi * CLIMA_n0_sno(q_sno, rho) / rho * K_therm / L_f * (T - T_freeze) / pow(lambda_sno, 2) *
      (a_vent_sno + b_vent_sno * pow(nu_air / D_vapor, 1./3.) /
        pow(r0_sno * lambda_sno, (ve_sno + Delta_v_sno) / 2.) *
        pow(2. * v0_sno * Chi_v_sno / nu_air / lambda_sno, 1./2.) *
        tgamma((ve_sno + Delta_v_sno + 5.) / 2.)
      );
  }
  else{
    *qs_tendency_melt = 0.;
  }
  return;
}

void CLIMA_sedimentation_velocity(const struct DimStruct *dims,
                                  double* restrict density,
                                  double* restrict qr,
                                  double* restrict qs,
                                  double* restrict qr_velocity,
                                  double* restrict qs_velocity){

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
                double qs_tmp = fmax(qs[ijk], 0.0);

                qr_velocity[ijk] = -fmin(CLIMA_terminal_velocity_rai(density[k], qr_tmp), 10.0);
                qs_velocity[ijk] = -fmin(CLIMA_terminal_velocity_sno(density[k], qs_tmp), 10.0);
            }
        }
    }

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax-1 ; k++){
                const ssize_t ijk = ishift + jshift + k;

                qr_velocity[ijk] = interp_2(qr_velocity[ijk], qr_velocity[ijk+1]);
                qs_velocity[ijk] = interp_2(qs_velocity[ijk], qs_velocity[ijk+1]);
            }
        }
    }
    return;
}

void CLIMA_microphysics_sources(const struct DimStruct *dims, struct LookupStruct *LT,
                                double (*lam_fp)(double), double (*L_fp)(double, double),
                                double* restrict density, double* restrict p0,
                                double* restrict temperature, double* restrict qt,
                                double* restrict ql, double* restrict qi,
                                double* restrict qr, double* restrict qs,
                                double dt,
                                double* restrict precip_formation_rate,
                                double* restrict evaporation_sublimation_rate,
                                double* restrict melt_rate,
                                double* restrict qr_tendency_micro,
                                double* restrict qs_tendency_micro,
                                double* restrict qr_tendency,
                                double* restrict qs_tendency){

    double qr_tendency_tmp = 0.;
    double ql_tendency_tmp = 0.;
    double qs_tendency_tmp = 0.;
    double qi_tendency_tmp = 0.;
    double precip_formation_rate_tmp = 0;
    double evaporation_sublimation_rate_tmp = 0;
    double melt_rate_tmp = 0;

    double qr_tendency_aut = 0.;
    double qs_tendency_aut = 0.;
    double qr_tendency_acc = 0.;
    double qs_tendency_acc = 0.;
    double ql_tendency_acc = 0.;
    double qi_tendency_acc = 0.;
    double qr_tendency_evp = 0.;
    double qs_tendency_dep_sub = 0.;
    double qs_tendency_melt = 0.;

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
                double qs_tmp = fmax(qs[ijk],0.0);

                double qt_tmp = fmax(qt[ijk], 0.0);
                double ql_tmp = fmax(ql[ijk], 0.0);
                double qi_tmp = fmax(qi[ijk], 0.0);

                double qv_tmp = fmax(qt_tmp - ql_tmp - qi_tmp, 0.0);

                precip_formation_rate_tmp = 0.;
                evaporation_sublimation_rate_tmp = 0.;
                melt_rate_tmp = 0.;

                double time_added = 0.0, dt_, rate;
                ssize_t iter_count = 0;
                do{
                    iter_count += 1;

                    qr_tendency_aut = 0.;
                    qs_tendency_aut = 0.;
                    qr_tendency_acc = 0.;
                    qs_tendency_acc = 0.;
                    ql_tendency_acc = 0.;
                    qi_tendency_acc = 0.;
                    qr_tendency_evp = 0.;
                    qs_tendency_dep_sub = 0.;
                    qs_tendency_melt = 0.;

                    // compute the source terms:
                    // autoconversion
                    CLIMA_conv_q_liq_to_q_rai(ql_tmp, &qr_tendency_aut);
                    CLIMA_conv_q_ice_to_q_sno_no_supersat(qi_tmp, &qs_tendency_aut);

                    // accretion
                    CLIMA_accretion(ql_tmp, qi_tmp, qr_tmp, qs_tmp, density[k], temperature[ijk],
                                    &ql_tendency_acc, &qi_tendency_acc, &qr_tendency_acc, &qs_tendency_acc);

                    // evaporation, deposition/sublimation, melting
                    CLIMA_rain_evaporation(qt_tmp, ql_tmp, qi_tmp, qr_tmp, density[k], temperature[ijk], p0[k],
                            LT, lam_fp, L_fp, &qr_tendency_evp);
                    CLIMA_snow_deposition_sublimation(qt_tmp, ql_tmp, qi_tmp, qs_tmp, density[k], temperature[ijk], p0[k],
                            LT, lam_fp, L_fp, &qs_tendency_dep_sub);
                    CLIMA_snow_melt(qs_tmp, density[k], temperature[ijk], &qs_tendency_melt);

                    // find the maximum substep time:
                    dt_ = dt - time_added;
                    // ... check the source term magnitudes ...
                    ql_tendency_tmp = -qr_tendency_aut + ql_tendency_acc;
                    qi_tendency_tmp = -qs_tendency_aut + qs_tendency_acc;
                    qr_tendency_tmp =  qr_tendency_aut + qr_tendency_acc + qr_tendency_evp + qs_tendency_melt;
                    qs_tendency_tmp =  qs_tendency_aut + qs_tendency_acc + qs_tendency_dep_sub - qs_tendency_melt;

                    //... adjust the rates if necessary (rate factor is ad-hoc)
                    double rate_constant = 1.25;
                    rate = rate_constant * ql_tendency_tmp * dt_ / (-fmax(ql_tmp, microph_eps));
                    rate = fmax(rate_constant * qr_tendency_tmp * dt_ / (-fmax(qr_tmp, microph_eps)), rate);
                    rate = fmax(rate_constant * qi_tendency_tmp * dt_ / (-fmax(qi_tmp, microph_eps)), rate);
                    rate = fmax(rate_constant * qs_tendency_tmp * dt_ / (-fmax(qs_tmp, microph_eps)), rate);
                    if(rate > 1.0 && iter_count < max_iter){
                        //Limit the timestep, but don't allow it to become vanishingly small
                        //Don't adjust if we have reached the maximum iteration number
                        dt_ = fmax(dt_/rate, 1.0e-3);
                    }

                    precip_formation_rate_tmp += (qr_tendency_aut + qr_tendency_acc + qs_tendency_aut + qs_tendency_acc + fmax(0.0, qs_tendency_dep_sub)) * dt_;
                    evaporation_sublimation_rate_tmp += (qr_tendency_evp + fmin(0.0, qs_tendency_dep_sub)) * dt_;
                    melt_rate_tmp += qs_tendency_melt * dt_;

                    //integrate forward in time
                    ql_tmp += ql_tendency_tmp * dt_;
                    qi_tmp += qi_tendency_tmp * dt_;
                    qr_tmp += qr_tendency_tmp * dt_;
                    qs_tmp += qs_tendency_tmp * dt_;

                    qv_tmp += (-qr_tendency_evp -qs_tendency_dep_sub) * dt_;

                    ql_tmp = fmax(ql_tmp, 0.0);
                    qi_tmp = fmax(qi_tmp, 0.0);
                    qr_tmp = fmax(qr_tmp, 0.0);
                    qs_tmp = fmax(qs_tmp, 0.0);
                    qt_tmp = qv_tmp + ql_tmp + qi_tmp;

                    time_added += dt_ ;

                }while(time_added < dt);

                qr_tendency_micro[ijk] = (qr_tmp - qr[ijk])/dt;
                qs_tendency_micro[ijk] = (qs_tmp - qs[ijk])/dt;

                qr_tendency[ijk] += qr_tendency_micro[ijk];
                qs_tendency[ijk] += qs_tendency_micro[ijk];

                precip_formation_rate[ijk] = precip_formation_rate_tmp/dt;
                evaporation_sublimation_rate[ijk]  =  evaporation_sublimation_rate_tmp/dt;
                melt_rate[ijk] = melt_rate_tmp/dt;
            }
        }
    }
    return;
}

void CLIMA_qt_source_formation(const struct DimStruct *dims,
                               double* restrict qr_tendency_micro,
                               double* restrict qs_tendency_micro,
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
                qt_tendency[ijk] += -(qr_tendency_micro[ijk] + qs_tendency_micro[ijk]);
            }
        }
    }
    return;
}

//TODO - double check the signs with EDMF implementation
//TODO - check specific and latent heats ci, cv, cp_l, L_f etc
void CLIMA_entropy_source_formation(const struct DimStruct *dims, struct LookupStruct *LT,
                                    double (*lam_fp)(double), double (*L_fp)(double, double),
                                    double* restrict p0,
                                    double* restrict T, double* restrict Twet,
                                    double* restrict qt, double* restrict qv,
                                    double* restrict precip_formation_rate,
                                    double* restrict evaporation_sublimation_rate,
                                    double* restrict entropy_tendency){

    // Source terms of entropy related to microphysics.
    // See Pressel et al. 2015, Eq. 49-54
    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;

    //entropy tendencies from formation or evaporation of precipitation
    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;

                const double lam_T  = lam_fp(T[ijk]);
                const double lam_Tw = lam_fp(Twet[ijk]);

                const double L_fp_T  = L_fp(T[ijk], lam_T);
                const double L_fp_Tw = L_fp(Twet[ijk], lam_Tw);

                const double pv_star_T  = lookup(LT, T[ijk]);
                const double pv_star_Tw = lookup(LT, Twet[ijk]);

                const double pv = pv_c(p0[k], qt[ijk], qv[ijk]);
                const double pd = p0[k] - pv;

                const double sd_T = sd_c(pd, T[ijk]);

                const double sv_star_T  = sv_c(pv_star_T,  T[ijk] );
                const double sv_star_Tw = sv_c(pv_star_Tw, Twet[ijk]);

                const double S_P = sd_T - sv_star_T + L_fp_T/T[ijk];
                const double S_E = sv_star_Tw - L_fp_Tw/Twet[ijk] - sd_T;
                const double S_D = -Rv * log(pv/pv_star_T) + cpv * log(T[ijk]/Twet[ijk]);

                entropy_tendency[ijk] += S_P * precip_formation_rate[ijk] - (S_E + S_D) * evaporation_sublimation_rate[ijk];

            }
        }
    }
    return;
}

void CLIMA_entropy_source_heating(const struct DimStruct *dims,
                               double* restrict T, double* restrict Twet,
                               double* restrict qr, double* restrict w_qr,
                               double* restrict qs, double* restrict w_qs,
                               double* restrict w,
                               double* restrict melt_rate,
                               double* restrict entropy_tendency){

    const ssize_t istride = dims->nlg[1] * dims->nlg[2];
    const ssize_t jstride = dims->nlg[2];
    const ssize_t imin = dims->gw;
    const ssize_t jmin = dims->gw;
    const ssize_t kmin = dims->gw;
    const ssize_t imax = dims->nlg[0]-dims->gw;
    const ssize_t jmax = dims->nlg[1]-dims->gw;
    const ssize_t kmax = dims->nlg[2]-dims->gw;
    const double dzi = 1.0/dims->dx[2];

    const double lhf = 3.34e5; //TODO

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;

                entropy_tendency[ijk] += qr[ijk] * (fabs(w_qr[ijk]) - w[ijk]) * cl * (Twet[ijk+1] - Twet[ijk]) * dzi / T[ijk];
                entropy_tendency[ijk] += qs[ijk] * (fabs(w_qs[ijk]) - w[ijk]) * ci * (Twet[ijk+1] - Twet[ijk]) * dzi / T[ijk];
                entropy_tendency[ijk] += fabs(melt_rate[ijk]) * lhf / T[ijk]; //TODO - what is the sign here
            }
        }
    }
    return;
}

void CLIMA_entropy_source_drag(const struct DimStruct *dims,
                               double* restrict T,
                               double* restrict qr, double* restrict w_qr,
                               double* restrict qs, double* restrict w_qs,
                               double* restrict entropy_tendency){

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
                entropy_tendency[ijk] += g * qs[ijk] * fabs(w_qs[ijk]) / T[ijk];
            }
        }
    }
    return;
}

// diagnostics for output
void CLIMA_autoconversion_wrapper(const struct DimStruct *dims,
                                       double* restrict ql,
                                       double* restrict qi,
                                       double* restrict qr_tendency_aut,
                                       double* restrict qs_tendency_aut){

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
                double qi_tmp = fmax(qi[ijk], 0.0);

                CLIMA_conv_q_liq_to_q_rai(ql_tmp, &qr_tendency_aut[ijk]);
                CLIMA_conv_q_ice_to_q_sno_no_supersat(qi_tmp, &qs_tendency_aut[ijk]);
            }
        }
    }
    return;
}

void CLIMA_accretion_wrapper(const struct DimStruct *dims,
                                  double* restrict density,
                                  double* restrict temperature,
                                  double* restrict ql,
                                  double* restrict qi,
                                  double* restrict qr,
                                  double* restrict qs,
                                  double* restrict ql_tendency_acc,
                                  double* restrict qi_tendency_acc,
                                  double* restrict qr_tendency_acc,
                                  double* restrict qs_tendency_acc){

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
                const double qi_tmp = fmax(qi[ijk], 0.0);
                const double qr_tmp = fmax(qr[ijk], 0.0);
                const double qs_tmp = fmax(qs[ijk], 0.0);

                CLIMA_accretion(
                    ql_tmp, qi_tmp, qr_tmp, qs_tmp, density[k], temperature[ijk],
                    &ql_tendency_acc[ijk], &qi_tendency_acc[ijk], &qr_tendency_acc[ijk], &qs_tendency_acc[ijk]
               );
            }
        }
    }
    return;
}

void CLIMA_evaporation_deposition_sublimation_wrapper(const struct DimStruct *dims,
                                    struct LookupStruct *LT,
                                    double (*lam_fp)(double),
                                    double (*L_fp)(double, double),
                                    double* restrict density,
                                    double* restrict p0,
                                    double* restrict temperature,
                                    double* restrict qt,
                                    double* restrict ql,
                                    double* restrict qi,
                                    double* restrict qr,
                                    double* restrict qs,
                                    double* restrict qr_tendency_evap,
                                    double* restrict qs_tendency_dep_sub){

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
                const double qs_tmp = fmax(qs[ijk], 0.0);

                const double qt_tmp = fmax(qt[ijk], 0.0);
                const double ql_tmp = fmax(ql[ijk], 0.0);
                const double qi_tmp = fmax(qi[ijk], 0.0);

                CLIMA_rain_evaporation(qt_tmp, ql_tmp, qi_tmp, qr_tmp, density[k], temperature[ijk], p0[k],
                    LT, lam_fp, L_fp, &qr_tendency_evap[ijk]);

                CLIMA_snow_deposition_sublimation(qt_tmp, ql_tmp, qi_tmp, qs_tmp, density[k], temperature[ijk], p0[k],
                    LT, lam_fp, L_fp, &qs_tendency_dep_sub[ijk]);
            }
        }
    }
    return;
}

void CLIMA_snow_melt_wrapper(const struct DimStruct *dims,
                                    struct LookupStruct *LT,
                                    double (*lam_fp)(double),
                                    double (*L_fp)(double, double),
                                    double* restrict density,
                                    double* restrict temperature,
                                    double* restrict qs,
                                    double* restrict qs_tendency_melt){

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

                const double qs_tmp = fmax(qs[ijk], 0.0);

                CLIMA_snow_melt(qs_tmp, density[k], temperature[ijk], &qs_tendency_melt[ijk]);
            }
        }
    }
    return;
}

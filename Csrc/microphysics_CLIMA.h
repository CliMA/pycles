#pragma once
#include "parameters.h"
#include "thermodynamic_functions.h"
#include "advection_interpolation.h"
#include "entropies.h"

double CLIMA_latent_heat_fusion(double T){
  const double LH_f0 = CLIMA_LH_s0 - CLIMA_LH_v0;
  return LH_f0 + (cl - ci) * (T - Tt);
}

double CLIMA_v0_rai(double rho){
  return sqrt(8./3./ CLIMA_C_drag * (CLIMA_rho_cloud_liq / rho - 1.) * g * CLIMA_r0_rai);
}

double CLIMA_n0_sno(double q_sno, double rho){
  return CLIMA_mu_sno * pow(rho * fmax(0., q_sno), CLIMA_nu_sno);
}

double CLIMA_lambda(double q, double rho, double n0, double m0, double me,
                    double r0, double Chi_m, double Delta_m){
  if(q > CLIMA_microph_eps){
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
  if(q_rai > CLIMA_microph_eps){
    double lambda_rai = CLIMA_lambda(q_rai, rho, CLIMA_n0_rai, CLIMA_m0_rai,
                                     CLIMA_me_rai, CLIMA_r0_rai, CLIMA_Chi_m_rai, CLIMA_Delta_m_rai);

    return CLIMA_Chi_v_rai * CLIMA_v0_rai(rho) * pow(lambda_rai * CLIMA_r0_rai, - CLIMA_ve_rai - CLIMA_Delta_v_rai) *
           tgamma(CLIMA_me_rai + CLIMA_ve_rai + CLIMA_Delta_m_rai + CLIMA_Delta_v_rai + 1.) /
           tgamma(CLIMA_me_rai + CLIMA_Delta_m_rai + 1.);
  }
  else{
    return 0.;
  }
}

double CLIMA_terminal_velocity_sno(double rho, double q_sno){
  if(q_sno > CLIMA_microph_eps){
    double lambda_sno = CLIMA_lambda(q_sno, rho, CLIMA_n0_sno(q_sno, rho), CLIMA_m0_sno,
                                     CLIMA_me_sno, CLIMA_r0_sno, CLIMA_Chi_m_sno, CLIMA_Delta_m_sno);

    return CLIMA_Chi_v_sno * CLIMA_v0_sno * pow(lambda_sno * CLIMA_r0_sno, - CLIMA_ve_sno - CLIMA_Delta_v_sno) *
           tgamma(CLIMA_me_sno + CLIMA_ve_sno + CLIMA_Delta_m_sno + CLIMA_Delta_v_sno + 1.) /
           tgamma(CLIMA_me_sno + CLIMA_Delta_m_sno + 1.);
  }
  else{
    return 0.;
  }
}

void CLIMA_conv_q_liq_to_q_rai(double _q_liq, double* qr_tendency_aut){
  double q_liq = fmax(0., _q_liq);
  *qr_tendency_aut = fmax(0., q_liq - CLIMA_q_liq_threshold) / CLIMA_tau_acnv_rai;
  return;
}

void CLIMA_conv_q_ice_to_q_sno_no_supersat(double _q_ice, double* qs_tendency_aut){
  double q_ice = fmax(0., _q_ice);
  *qs_tendency_aut = fmax(0., q_ice - CLIMA_q_ice_threshold) / CLIMA_tau_acnv_sno;
  return;
}

void CLIMA_conv_q_ice_to_q_sno(double _q_tot, double _q_liq, double _q_ice, double rho, double T, double p0,
                               struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
                               double* qs_tendency_aut){
  double q_tot = fmax(0., _q_tot);
  double q_liq = fmax(0., _q_liq);
  double q_ice = fmax(0., _q_ice);
  double q_v = fmax(0.0, q_tot - q_liq - q_ice);

  double lam = lam_fp(T);
  double L = L_fp(T, lam);
  double pv_s = lookup(LT, T);

  double qv_sat = qv_star_c(p0, q_tot, pv_s);
  double S = q_v/qv_sat - 1;

  *qs_tendency_aut = 0.;

  if(q_ice > CLIMA_microph_eps && S > 0.){
    double G = 1. / (L / CLIMA_K_therm / T * (L / Rv / T - 1.) + Rv * T / CLIMA_D_vapor / pv_s);
    double lambda_ice = CLIMA_lambda(q_ice, rho, CLIMA_n0_ice, CLIMA_m0_ice, CLIMA_me_ice,
                                     CLIMA_r0_ice, CLIMA_Chi_m_ice, CLIMA_Delta_m_ice);

    *qs_tendency_aut = 4. * pi * S * G * CLIMA_n0_ice / rho * exp(-lambda_ice * CLIMA_r_ice_snow) *
                      (pow(CLIMA_r_ice_snow, 2) / (CLIMA_me_ice + CLIMA_Delta_m_ice) +
                      (CLIMA_r_ice_snow * lambda_ice + 1.) / pow(lambda_ice, 2));
  }
  return;
}

void CLIMA_accretion(double _q_liq, double _q_ice, double _q_rai, double _q_sno,
                     double rho, double T, double* ql_tendency_acc, double* qi_tendency_acc,
                     double* qr_tendency_acc, double* qs_tendency_acc){
  double q_liq = fmax(0., _q_liq);
  double q_ice = fmax(0., _q_ice);
  double q_rai = fmax(0., _q_rai);
  double q_sno = fmax(0., _q_sno);

  double lambda_ice = CLIMA_lambda(q_ice, rho, CLIMA_n0_ice, CLIMA_m0_ice,
                                   CLIMA_me_ice, CLIMA_r0_ice, CLIMA_Chi_m_ice, CLIMA_Delta_m_ice);
  double lambda_rai = CLIMA_lambda(q_rai, rho, CLIMA_n0_rai, CLIMA_m0_rai,
                                   CLIMA_me_rai, CLIMA_r0_rai, CLIMA_Chi_m_rai, CLIMA_Delta_m_rai);
  double lambda_sno = CLIMA_lambda(q_sno, rho, CLIMA_n0_sno(q_sno, rho), CLIMA_m0_sno,
                                   CLIMA_me_sno, CLIMA_r0_sno, CLIMA_Chi_m_sno, CLIMA_Delta_m_sno);

  double v_rai = CLIMA_terminal_velocity_rai(rho, q_rai);
  double v_sno = CLIMA_terminal_velocity_sno(rho, q_sno);

  double tmp = 0.;

  *qr_tendency_acc = 0.;
  *ql_tendency_acc = 0.;
  *qs_tendency_acc = 0.;
  *qi_tendency_acc = 0.;

  // accretion qr ql
  if(q_rai > CLIMA_microph_eps && q_liq > CLIMA_microph_eps){
    tmp =
        q_liq * CLIMA_E_liq_rai * CLIMA_n0_rai * CLIMA_a0_rai * CLIMA_v0_rai(rho) * CLIMA_Chi_a_rai * CLIMA_Chi_v_rai / lambda_rai *
        tgamma(CLIMA_ae_rai + CLIMA_ve_rai + CLIMA_Delta_a_rai + CLIMA_Delta_v_rai + 1.) /
        pow(lambda_rai * CLIMA_r0_rai, CLIMA_ae_rai + CLIMA_ve_rai + CLIMA_Delta_a_rai + CLIMA_Delta_v_rai);

    *qr_tendency_acc += tmp;
    *ql_tendency_acc -= tmp;
  }

  // accretion qs qi
  if(q_sno > CLIMA_microph_eps && q_ice > CLIMA_microph_eps){
    tmp =
        q_ice * CLIMA_E_ice_sno * CLIMA_n0_sno(q_sno, rho) * CLIMA_a0_sno * CLIMA_v0_sno * CLIMA_Chi_a_sno * CLIMA_Chi_v_sno / lambda_sno *
        tgamma(CLIMA_ae_sno + CLIMA_ve_sno + CLIMA_Delta_a_sno + CLIMA_Delta_v_sno + 1.) /
        pow(lambda_sno * CLIMA_r0_sno, CLIMA_ae_sno + CLIMA_ve_sno + CLIMA_Delta_a_sno + CLIMA_Delta_v_sno);

    *qs_tendency_acc += tmp;
    *qi_tendency_acc -= tmp;
  }

  // accretion qr qi
  if(q_rai > CLIMA_microph_eps && q_ice > CLIMA_microph_eps){
    double acc_q_ice_q_rai_ice_sink =
      q_ice * CLIMA_E_ice_rai * CLIMA_n0_rai * CLIMA_a0_rai * CLIMA_v0_rai(rho) * CLIMA_Chi_a_rai * CLIMA_Chi_v_rai / lambda_rai *
      tgamma(CLIMA_ae_rai + CLIMA_ve_rai + CLIMA_Delta_a_rai + CLIMA_Delta_v_rai + 1.) /
      pow(lambda_rai * CLIMA_r0_rai, CLIMA_ae_rai + CLIMA_ve_rai + CLIMA_Delta_a_rai + CLIMA_Delta_v_rai);
    double acc_q_ice_q_rai_rain_sink = CLIMA_E_ice_rai / rho * CLIMA_n0_rai * CLIMA_n0_ice * CLIMA_m0_rai * CLIMA_a0_rai * CLIMA_v0_rai(rho) *
      CLIMA_Chi_m_rai * CLIMA_Chi_a_rai * CLIMA_Chi_v_rai / lambda_ice / lambda_rai *
      tgamma(CLIMA_me_rai + CLIMA_ae_rai + CLIMA_ve_rai + CLIMA_Delta_m_rai + CLIMA_Delta_a_rai + CLIMA_Delta_v_rai +1.) /
      pow(CLIMA_r0_rai * lambda_rai, CLIMA_me_rai + CLIMA_ae_rai + CLIMA_ve_rai + CLIMA_Delta_m_rai + CLIMA_Delta_a_rai + CLIMA_Delta_v_rai);

    *qr_tendency_acc -= acc_q_ice_q_rai_rain_sink;
    *qs_tendency_acc += acc_q_ice_q_rai_rain_sink + acc_q_ice_q_rai_ice_sink;
    *qi_tendency_acc -= acc_q_ice_q_rai_ice_sink;
  }

  // accretion qs ql
  if(q_sno >  CLIMA_microph_eps && q_liq > CLIMA_microph_eps){
    tmp = -q_liq * CLIMA_E_liq_sno * CLIMA_n0_sno(q_sno, rho) * CLIMA_a0_sno * CLIMA_v0_sno * CLIMA_Chi_a_sno * CLIMA_Chi_v_sno / lambda_sno *
          tgamma(CLIMA_ae_sno + CLIMA_ve_sno + CLIMA_Delta_a_sno + CLIMA_Delta_v_sno + 1.) /
          pow(lambda_sno * CLIMA_r0_sno, CLIMA_ae_sno + CLIMA_ve_sno + CLIMA_Delta_a_sno + CLIMA_Delta_v_sno);

    if(T>Tf){
      double L_f = CLIMA_latent_heat_fusion(T);
      double alpha = cl / L_f * (T - Tf);
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
  if(q_sno >  CLIMA_microph_eps && q_rai > CLIMA_microph_eps){
    if(T>Tf){
      tmp = pi / rho * CLIMA_n0_rai * CLIMA_n0_sno(q_sno, rho) * CLIMA_m0_sno * CLIMA_Chi_m_sno * CLIMA_E_rai_sno *
        fabs(v_rai - v_sno) / pow(CLIMA_r0_sno, CLIMA_me_sno + CLIMA_Delta_m_sno) * (
          2. * tgamma(CLIMA_me_sno + CLIMA_Delta_m_sno + 1.) / pow(lambda_rai, 3) / pow(lambda_sno, CLIMA_me_sno + CLIMA_Delta_m_sno + 1) +
          2. * tgamma(CLIMA_me_sno + CLIMA_Delta_m_sno + 2.) / pow(lambda_rai, 2) / pow(lambda_sno, CLIMA_me_sno + CLIMA_Delta_m_sno + 2) +
               tgamma(CLIMA_me_sno + CLIMA_Delta_m_sno + 3.) / lambda_rai /         pow(lambda_sno, CLIMA_me_sno + CLIMA_Delta_m_sno + 3));
       *qr_tendency_acc += tmp;
       *qs_tendency_acc -= tmp;
    }
    else{
      tmp = pi / rho * CLIMA_n0_sno(q_sno, rho) * CLIMA_n0_rai * CLIMA_m0_rai * CLIMA_Chi_m_rai * CLIMA_E_rai_sno *
        fabs(v_rai - v_sno) / pow(CLIMA_r0_rai, CLIMA_me_rai + CLIMA_Delta_m_rai) * (
          2. * tgamma(CLIMA_me_rai + CLIMA_Delta_m_rai + 1.) / pow(lambda_sno, 3) / pow(lambda_rai, CLIMA_me_rai + CLIMA_Delta_m_rai + 1) +
          2. * tgamma(CLIMA_me_rai + CLIMA_Delta_m_rai + 2.) / pow(lambda_sno, 2) / pow(lambda_rai, CLIMA_me_rai + CLIMA_Delta_m_rai + 2) +
               tgamma(CLIMA_me_rai + CLIMA_Delta_m_rai + 3.) / lambda_sno /         pow(lambda_rai, CLIMA_me_rai + CLIMA_Delta_m_rai + 3));
      *qs_tendency_acc += tmp;
      *qr_tendency_acc -= tmp;
    }
  }
  return;
}

void CLIMA_rain_evaporation(double _q_tot, double _q_liq, double _q_ice, double _q_rai, double rho, double T, double p0,
                            struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
                            double* qr_tendency_evp){
  double q_tot = fmax(0., _q_tot);
  double q_liq = fmax(0., _q_liq);
  double q_ice = fmax(0., _q_ice);
  double q_rai = fmax(0., _q_rai);

  double lam = lam_fp(T);
  double L = L_fp(T, lam);
  double pv_s = lookup(LT, T);

  double qv_sat = qv_star_c(p0, q_tot, pv_s);
  double q_v = fmax(0., q_tot - q_liq - q_ice);
  double S = q_v/qv_sat - 1;

  if(q_rai > CLIMA_microph_eps && S < 0.){

    double G = 1. / (L / CLIMA_K_therm / T * (L / Rv / T - 1.) + Rv * T / CLIMA_D_vapor / pv_s);

    double lambda_rai = CLIMA_lambda(q_rai, rho, CLIMA_n0_rai, CLIMA_m0_rai, CLIMA_me_rai, CLIMA_r0_rai, CLIMA_Chi_m_rai, CLIMA_Delta_m_rai);

    *qr_tendency_evp = fmin(0.,
      4. * pi * CLIMA_n0_rai / rho * S * G / pow(lambda_rai, 2) *
      (CLIMA_a_vent_rai + CLIMA_b_vent_rai * pow(CLIMA_nu_air / CLIMA_D_vapor, 1./3.) /
        pow(CLIMA_r0_rai * lambda_rai, (CLIMA_ve_rai + CLIMA_Delta_v_rai) / 2.) *
        pow(2. * CLIMA_v0_rai(rho) * CLIMA_Chi_v_rai / CLIMA_nu_air / lambda_rai, 1./2.) *
        tgamma((CLIMA_ve_rai + CLIMA_Delta_v_rai + 5.) / 2.)
      ));
  }
  else{
    *qr_tendency_evp = 0.;
  }
  return;
}

void CLIMA_snow_deposition_sublimation(double _q_tot, double _q_liq, double _q_ice, double _q_sno, double rho, double T, double p0,
                            struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
                            double* qs_tendency_dep_sub){
  double q_tot = fmax(0., _q_tot);
  double q_liq = fmax(0., _q_liq);
  double q_ice = fmax(0., _q_ice);
  double q_sno = fmax(0., _q_sno);

  double lam = lam_fp(T);
  double L = L_fp(T, lam);
  double pv_s = lookup(LT, T);

  double qv_sat = qv_star_c(p0, q_tot, pv_s);
  double q_v = fmax(0.0, q_tot - q_liq - q_ice);
  double S = q_v/qv_sat - 1;

  if(q_sno > CLIMA_microph_eps){

    double G = 1. / (L / CLIMA_K_therm / T * (L / Rv / T - 1.) + Rv * T / CLIMA_D_vapor / pv_s);

    double lambda_sno = CLIMA_lambda(q_sno, rho, CLIMA_n0_sno(q_sno, rho), CLIMA_m0_sno, CLIMA_me_sno, CLIMA_r0_sno, CLIMA_Chi_m_sno, CLIMA_Delta_m_sno);

    *qs_tendency_dep_sub =
      4. * pi * CLIMA_n0_sno(q_sno, rho) / rho * S * G / pow(lambda_sno, 2) *
      (CLIMA_a_vent_sno + CLIMA_b_vent_sno * pow(CLIMA_nu_air / CLIMA_D_vapor, 1./3.) /
        pow(CLIMA_r0_sno * lambda_sno, (CLIMA_ve_sno + CLIMA_Delta_v_sno) / 2.) *
        pow(2. * CLIMA_v0_sno * CLIMA_Chi_v_sno / CLIMA_nu_air / lambda_sno, 1./2.) *
        tgamma((CLIMA_ve_sno + CLIMA_Delta_v_sno + 5.) / 2.)
      );
  }
  else{
    *qs_tendency_dep_sub = 0.;
  }
  return;
}

void CLIMA_snow_melt(double _q_sno, double rho, double T, double* qs_tendency_melt){

  double q_sno = fmax(0., _q_sno);

  if(q_sno > CLIMA_microph_eps && T > Tf){

    double L_f = CLIMA_latent_heat_fusion(T);

    double lambda_sno = CLIMA_lambda(q_sno, rho, CLIMA_n0_sno(q_sno, rho), CLIMA_m0_sno, CLIMA_me_sno, CLIMA_r0_sno, CLIMA_Chi_m_sno, CLIMA_Delta_m_sno);

    *qs_tendency_melt =
      -4. * pi * CLIMA_n0_sno(q_sno, rho) / rho * CLIMA_K_therm / L_f * (T - Tf) / pow(lambda_sno, 2) *
      (CLIMA_a_vent_sno + CLIMA_b_vent_sno * pow(CLIMA_nu_air / CLIMA_D_vapor, 1./3.) /
        pow(CLIMA_r0_sno * lambda_sno, (CLIMA_ve_sno + CLIMA_Delta_v_sno) / 2.) *
        pow(2. * CLIMA_v0_sno * CLIMA_Chi_v_sno / CLIMA_nu_air / lambda_sno, 1./2.) *
        tgamma((CLIMA_ve_sno + CLIMA_Delta_v_sno + 5.) / 2.)
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
                    //CLIMA_conv_q_ice_to_q_sno(qt_tmp, ql_tmp, qi_tmp, density[k], temperature[ijk], p0[k],
                    //                          LT, lam_fp, L_fp, &qs_tendency_aut);

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
                    qi_tendency_tmp = -qs_tendency_aut + qi_tendency_acc;
                    qr_tendency_tmp =  qr_tendency_aut + qr_tendency_acc + qr_tendency_evp - qs_tendency_melt;
                    qs_tendency_tmp =  qs_tendency_aut + qs_tendency_acc + qs_tendency_dep_sub + qs_tendency_melt;

                    //... adjust the rates if necessary (rate factor is ad-hoc)
                    double rate_constant = 1.25;
                    rate = rate_constant * ql_tendency_tmp * dt_ / (-fmax(ql_tmp, CLIMA_microph_rate_eps));
                    rate = fmax(rate_constant * qr_tendency_tmp * dt_ / (-fmax(qr_tmp, CLIMA_microph_rate_eps)), rate);
                    rate = fmax(rate_constant * qi_tendency_tmp * dt_ / (-fmax(qi_tmp, CLIMA_microph_rate_eps)), rate);
                    rate = fmax(rate_constant * qs_tendency_tmp * dt_ / (-fmax(qs_tmp, CLIMA_microph_rate_eps)), rate);
                    if(rate > 1.0 && iter_count < CLIMA_max_iter){
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

    for(ssize_t i=imin; i<imax; i++){
        const ssize_t ishift = i * istride;
        for(ssize_t j=jmin; j<jmax; j++){
            const ssize_t jshift = j * jstride;
            for(ssize_t k=kmin; k<kmax; k++){
                const ssize_t ijk = ishift + jshift + k;

                entropy_tendency[ijk] += qr[ijk] * (fabs(w_qr[ijk]) - w[ijk]) * cl * (Twet[ijk+1] - Twet[ijk]) * dzi / T[ijk];
                entropy_tendency[ijk] += qs[ijk] * (fabs(w_qs[ijk]) - w[ijk]) * ci * (Twet[ijk+1] - Twet[ijk]) * dzi / T[ijk];
                entropy_tendency[ijk] += melt_rate[ijk] * CLIMA_latent_heat_fusion(T[ijk]) / T[ijk];
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
                //CLIMA_conv_q_ice_to_q_sno(qt_tmp, ql_tmp, qi_tmp, density[k], temperature[ijk], p0[k],
                //                          LT, lam_fp, L_fp, &qs_tendency_aut[ijk]);
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

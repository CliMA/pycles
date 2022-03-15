import numpy as np
from collections import OrderedDict


def main():

    #########################
    # Users should modify here
    #########################

    parameters = OrderedDict()

    parameters['pi'] = np.pi
    parameters['g'] = 9.80665
    parameters['Rd'] = 287.1
    parameters['Rv'] = 461.5
    parameters['eps_v'] = parameters['Rd'] / parameters['Rv']
    parameters['eps_vi'] = 1.0 / parameters['eps_v']
    parameters['cpd'] = 1004.0
    parameters['cpv'] = 1859.0
    parameters['cl'] = 4181.0
    parameters['ci'] = 2100.0
    parameters['kappa'] = parameters['Rd'] / parameters['cpd']
    parameters['Tf'] = 273.15
    parameters['Tt'] = 273.16
    parameters['T_tilde'] = 298.15
    parameters['p_tilde'] = 10.0**5
    parameters['pv_star_t'] = 611.7
    parameters['sd_tilde'] = 6864.8
    parameters['sv_tilde'] = 10513.6
    parameters['omega'] = 7.2921151467064e-5  # Earth's rotational rate (http://hpiers.obspm.fr/eop-pc/models/constants.html)
    parameters['ql_threshold'] = 1.0e-8

    # Clima 1-moment microphysics
    parameters['CLIMA_rho_cloud_liq'] = 1e3
    parameters['CLIMA_rho_cloud_ice'] = 916.7
    parameters['CLIMA_T_icenuc'] = 263.15
    parameters['CLIMA_pow_icenuc'] = 1.0
    parameters['CLIMA_LH_v0'] = 2.5008e6
    parameters['CLIMA_LH_s0'] = 2.8344e6

    parameters['CLIMA_r0_ice'] = 1e-5
    parameters['CLIMA_r0_rai'] = 1e-3
    parameters['CLIMA_r0_sno'] = 1e-3

    parameters['CLIMA_n0_ice'] = 2e7
    parameters['CLIMA_me_ice'] = 3.0
    parameters['CLIMA_m0_ice'] = 4.0/3 * np.pi * parameters['CLIMA_rho_cloud_ice'] * parameters['CLIMA_r0_ice']**parameters['CLIMA_me_ice']
    parameters['CLIMA_Chi_m_ice'] = 1.0
    parameters['CLIMA_Delta_m_ice'] = 0.0

    parameters['CLIMA_r_ice_snow'] = 62.5 * 1e-6

    parameters['CLIMA_q_liq_threshold'] = 5e-4
    parameters['CLIMA_q_ice_threshold'] = 1e-6
    parameters['CLIMA_tau_acnv_rai'] = 2.5e3
    parameters['CLIMA_tau_acnv_sno'] = 1e2

    parameters['CLIMA_n0_rai'] = 8e6 * 2
    parameters['CLIMA_me_rai'] = 3.0
    parameters['CLIMA_ae_rai'] = 2.0
    parameters['CLIMA_ve_rai'] = 0.5
    parameters['CLIMA_m0_rai'] = 4.0/3 * np.pi * parameters['CLIMA_rho_cloud_liq'] * parameters['CLIMA_r0_rai']**parameters['CLIMA_me_rai']
    parameters['CLIMA_a0_rai'] = np.pi * parameters['CLIMA_r0_rai']**parameters['CLIMA_ae_rai']
    parameters['CLIMA_Chi_m_rai'] = 1.0
    parameters['CLIMA_Delta_m_rai'] = 0.0
    parameters['CLIMA_Chi_a_rai'] = 1.0
    parameters['CLIMA_Delta_a_rai'] = 0.0
    parameters['CLIMA_Chi_v_rai'] = 1.0
    parameters['CLIMA_Delta_v_rai'] = 0.0

    parameters['CLIMA_mu_sno'] = 4.36 * 1e9
    parameters['CLIMA_nu_sno'] = 0.63
    parameters['CLIMA_me_sno'] = 2.0
    parameters['CLIMA_ae_sno'] = 2.0
    parameters['CLIMA_ve_sno'] = 0.25
    parameters['CLIMA_m0_sno'] = 1e-1 * parameters['CLIMA_r0_sno']**parameters['CLIMA_me_sno']
    parameters['CLIMA_a0_sno'] = 0.3 * np.pi * parameters['CLIMA_r0_sno']**parameters['CLIMA_ae_sno']
    parameters['CLIMA_v0_sno'] = 2.0**(9.0/4) * parameters['CLIMA_r0_sno']**parameters['CLIMA_ve_sno']
    parameters['CLIMA_Chi_m_sno'] = 1.0
    parameters['CLIMA_Delta_m_sno'] = 0.0
    parameters['CLIMA_Chi_a_sno'] = 1.0
    parameters['CLIMA_Delta_a_sno'] = 0.0
    parameters['CLIMA_Chi_v_sno'] = 1.0
    parameters['CLIMA_Delta_v_sno'] = 0.0

    # CLIMA microphysics parameters - processes
    parameters['CLIMA_C_drag'] = 0.55
    parameters['CLIMA_K_therm'] = 2.4e-2
    parameters['CLIMA_D_vapor'] = 2.26e-5
    parameters['CLIMA_nu_air'] = 1.6e-5
    parameters['CLIMA_N_sc'] = 1.6/2.26

    parameters['CLIMA_a_vent_rai'] = 1.5
    parameters['CLIMA_b_vent_rai'] = 0.53
    parameters['CLIMA_a_vent_sno'] = 0.65
    parameters['CLIMA_b_vent_sno'] = 0.44

    parameters['CLIMA_E_liq_rai'] = 0.8
    parameters['CLIMA_E_liq_sno'] = 0.1
    parameters['CLIMA_E_ice_rai'] = 1.0
    parameters['CLIMA_E_ice_sno'] = 0.1
    parameters['CLIMA_E_rai_sno'] = 1.0

    #additional parameters for pycles implementation
    parameters['CLIMA_max_iter'] = 10
    parameters['CLIMA_microph_rate_eps'] = 1e-3
    parameters['CLIMA_microph_eps'] = 1e-16

    # Surface Monin-Obukhov related parameters
    parameters['vkb'] = 0.35     # Von Karman constant from Businger 1971 used by Byun surface formulation
    parameters['Pr0'] = 0.74
    parameters['beta_m'] = 4.7
    parameters['beta_h'] = parameters['beta_m']/parameters['Pr0']
    parameters['gamma_m'] = 15.0
    parameters['gamma_h'] = 9.0


    # Surface Monin-Obukhov related parameters
    parameters['vkb'] = 0.35     # Von Karman constant from Businger 1971 used by Byun surface formulation
    parameters['Pr0'] = 0.74
    parameters['beta_m'] = 4.7
    parameters['beta_h'] = parameters['beta_m']/parameters['Pr0']
    parameters['gamma_m'] = 15.0
    parameters['gamma_h'] = 9.0

    # if GABLS use these values:
    # parameters['vkb'] = 0.4
    # parameters['Pr0'] = 1.0
    # parameters['beta_m'] = 4.8
    # parameters['beta_h'] = 7.8


    #############################
    # Users shouldn't modify below
    #############################

    # Some warning to put in the generated code
    message1 = 'Generated code! Absolutely DO NOT modify this file, ' \
               'parameters should be modified in generate_parameters.py \n'
    message2 = 'End generated code'

    # First write the pxi file
    f = './parameters.pxi'
    fh = open(f, 'w')
    fh.write('#' + message1)
    fh.write('\n')
    for param in parameters:
        fh.write(
            'cdef double ' + param + ' = ' + str(parameters[param]) + '\n')
    fh.write('#' + 'End Generated Code')
    fh.close()

    # Now write the C include file
    f = './Csrc/parameters.h'
    fh = open(f, 'w')
    fh.write('//' + message1)
    for param in parameters:
        fh.write('#define ' + param + ' ' + str(parameters[param]) + '\n')
    fh.write('//' + message2)
    fh.close()

    print('Generated ./parameters.pxi and '
          './Csrc/parameters.h with the following values:')
    for param in parameters:
        print('\t' + param + ' = ' + str(parameters[param]))

    return


if __name__ == "__main__":
    main()

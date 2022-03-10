import numpy as np
from collections import OrderedDict
from scipy.special import gamma


def main():

    #########################
    # Users should modify here
    #########################

    # see CLIMA CloudMicrophysics.jl documentation
    parameters = OrderedDict()

    # porperties of water
    parameters['rho_cloud_liq'] = 1e3
    parameters['rho_cloud_ice'] = 916.7

    parameters['T_icenuc'] = 263.15
    parameters['pow_icenuc'] = 1.0

    # latent heat of vaporization and sublimation at reference temperature
    parameters['LH_v0'] = 2.5008e6
    parameters['LH_s0'] = 2.8344e6

    # CLIMA microphysics parameters - size distributions and relations
    parameters['r0_ice'] = 1e-5
    parameters['r0_rai'] = 1e-3
    parameters['r0_sno'] = 1e-3

    parameters['n0_ice'] = 2e7
    parameters['me_ice'] = 3.0
    parameters['m0_ice'] = 4.0/3 * np.pi * parameters['rho_cloud_ice'] * parameters['r0_ice']**parameters['me_ice']
    parameters['Chi_m_ice'] = 1.0
    parameters['Delta_m_ice'] = 0.0

    parameters['r_ice_snow'] = 62.5 * 1e-6

    parameters['q_liq_threshold'] = 5e-4
    parameters['q_ice_threshold'] = 1e-6
    parameters['tau_acnv_rai'] = 2.5e3
    parameters['tau_acnv_sno'] = 1e2

    parameters['n0_rai'] = 8e6 * 2
    parameters['me_rai'] = 3.0
    parameters['ae_rai'] = 2.0
    parameters['ve_rai'] = 0.5
    parameters['m0_rai'] = 4.0/3 * np.pi * parameters['rho_cloud_liq'] * parameters['r0_rai']**parameters['me_rai']
    parameters['a0_rai'] = np.pi * parameters['r0_rai']**parameters['ae_rai']
    parameters['Chi_m_rai'] = 1.0
    parameters['Delta_m_rai'] = 0.0
    parameters['Chi_a_rai'] = 1.0
    parameters['Delta_a_rai'] = 0.0
    parameters['Chi_v_rai'] = 1.0
    parameters['Delta_v_rai'] = 0.0

    parameters['mu_sno'] = 4.36 * 1e9
    parameters['nu_sno'] = 0.63
    parameters['me_sno'] = 2.0
    parameters['ae_sno'] = 2.0
    parameters['ve_sno'] = 0.25
    parameters['m0_sno'] = 1e-1 * parameters['r0_sno']**parameters['me_sno']
    parameters['a0_sno'] = 0.3 * np.pi * parameters['r0_sno']**parameters['ae_sno']
    parameters['v0_sno'] = 2.0**(9.0/4) * parameters['r0_sno']**parameters['ve_sno']
    parameters['Chi_m_sno'] = 1.0
    parameters['Delta_m_sno'] = 0.0
    parameters['Chi_a_sno'] = 1.0
    parameters['Delta_a_sno'] = 0.0
    parameters['Chi_v_sno'] = 1.0
    parameters['Delta_v_sno'] = 0.0

    # CLIMA microphysics parameters - processes
    parameters['C_drag'] = 0.55
    parameters['K_therm'] = 2.4e-2
    parameters['D_vapor'] = 2.26e-5
    parameters['nu_air'] = 1.6e-5
    parameters['N_sc'] = 1.6/2.26

    parameters['a_vent_rai'] = 1.5
    parameters['b_vent_rai'] = 0.53
    parameters['a_vent_sno'] = 0.65
    parameters['b_vent_sno'] = 0.44

    parameters['E_liq_rai'] = 0.8
    parameters['E_liq_sno'] = 0.1
    parameters['E_ice_rai'] = 1.0
    parameters['E_ice_sno'] = 0.1
    parameters['E_rai_sno'] = 1.0

    #additional parameters for pycles implementation
    parameters['max_iter'] = 10
    parameters['microph_rate_eps'] = 1e-3
    parameters['microph_eps'] = 1e-16

    # Users shouldn't modify below
    #############################

    # Some warning to put in the generated code
    message1 = 'Generated code! Absolutely DO NOT modify this file, ' \
               'microphysical parameters should be modified in generate_parameters_clima.py \n'
    message2 = 'End generated code'

    # First write the pxi file
    f = './parameters_clima.pxi'
    fh = open(f, 'w')
    fh.write('#' + message1)
    fh.write('\n')
    for param in parameters:
        fh.write(
            'cdef double ' + param + ' = ' + str(parameters[param]) + '\n')
    fh.write('#' + 'End Generated Code')
    fh.close()

    # Now write the C include file
    f = './Csrc/parameters_clima.h'
    fh = open(f, 'w')
    fh.write('//' + message1)
    for param in parameters:
        fh.write('#define ' + param + ' ' + str(parameters[param]) + '\n')
    fh.write('//' + message2)
    fh.close()

    print('Generated ./parameters_clima.pxi and '
          './Csrc/parameters_clima.h with the following values:')
    for param in parameters:
        print('\t' + param + ' = ' + str(parameters[param]))
    return

if __name__ == "__main__":
    main()

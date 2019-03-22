#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True


cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
from NetCDFIO cimport NetCDFIO_Stats
cimport ParallelMPI
cimport TimeStepping
cimport Surface
from Forcing cimport AdjustedMoistAdiabat
from Thermodynamics cimport LatentHeat

import numpy as np
cimport numpy as np
import netCDF4 as nc
from scipy.interpolate import pchip_interpolate
from libc.math cimport pow, cbrt, exp, fmin, fmax
from thermodynamic_functions cimport cpm_c
include 'parameters.pxi'
from profiles import profile_data
import math

def RadiationFactory(namelist, LatentHeat LH, ParallelMPI.ParallelMPI Pa):
    # if namelist specifies RRTM is to be used, this will override any case-specific radiation schemes
    try:
        use_rrtm = namelist['radiation']['use_RRTM']
    except:
        use_rrtm = False
    if use_rrtm:
        return RadiationRRTM(namelist,LH, Pa)
    else:
        casename = namelist['meta']['casename']
        if casename == 'DYCOMS_RF01':
            return RadiationDyCOMS_RF01(namelist)
        elif casename == 'DYCOMS_RF02':
            #Dycoms RF01 and RF02 use the same radiation
            return RadiationDyCOMS_RF01(namelist)
        elif casename == 'SMOKE':
            return RadiationSmoke()
        elif casename == 'CGILS':
            return RadiationRRTM(namelist,LH, Pa)
        elif casename == 'ZGILS':
            return RadiationRRTM(namelist, LH, Pa)
        elif casename == 'TRMM_LBA':
            return RadiationTRMM_LBA(namelist, LH, Pa)
        else:
            return RadiationNone()



cdef class RadiationBase:
    def __init__(self):
        return

    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        self.z_pencil = ParallelMPI.Pencil()
        self.z_pencil.initialize(Gr, Pa, 2)
        self.heating_rate = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
        self.dTdt_rad = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')

        NS.add_profile('radiative_heating_rate', Gr, Pa)
        NS.add_profile('radiative_entropy_tendency', Gr, Pa)
        NS.add_profile('radiative_temperature_tendency',Gr, Pa)
        NS.add_ts('srf_lw_flux_up', Gr, Pa)
        NS.add_ts('srf_lw_flux_down', Gr, Pa)
        NS.add_ts('srf_sw_flux_up', Gr, Pa)
        NS.add_ts('srf_sw_flux_down', Gr, Pa)


        return

    cpdef initialize_profiles(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                     NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 Surface.SurfaceBase Sur, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):
        return

    cpdef stats_io(self, Grid.Grid Gr,  ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t i,j,k
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t ishift, jshift, ijk

            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            double [:] entropy_tendency = np.zeros((Gr.dims.npg,), dtype=np.double, order='c')
            double [:] tmp

        # Now update entropy tendencies
        with nogil:
            for i in xrange(gw,imax):
                ishift = i * istride
                for j in xrange(gw,jmax):
                    jshift = j * jstride
                    for k in xrange(gw,kmax):
                        ijk = ishift + jshift + k
                        entropy_tendency[ijk] =  self.heating_rate[ijk] * RS.alpha0_half[k] / DV.values[ijk + t_shift]

        tmp = Pa.HorizontalMean(Gr, &self.heating_rate[0])
        NS.write_profile('radiative_heating_rate', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &entropy_tendency[0])
        NS.write_profile('radiative_entropy_tendency', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp = Pa.HorizontalMean(Gr, &self.dTdt_rad[0])
        NS.write_profile('radiative_temperature_tendency', tmp[Gr.dims.gw:-Gr.dims.gw], Pa)


        NS.write_ts('srf_lw_flux_up',self.srf_lw_up, Pa ) # Units are W/m^2
        NS.write_ts('srf_lw_flux_down', self.srf_lw_down, Pa)
        NS.write_ts('srf_sw_flux_up', self.srf_sw_up, Pa)
        NS.write_ts('srf_sw_flux_down', self.srf_sw_down, Pa)
        return


cdef class RadiationNone(RadiationBase):
    def __init__(self):
        return
    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return
    cpdef initialize_profiles(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                     NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 Surface.SurfaceBase Sur,TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):
        return
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return


cdef class RadiationDyCOMS_RF01(RadiationBase):
    def __init__(self, namelist):
        self.alpha_z = 1.0
        self.kap = 85.0
        try:
            self.f0 = namelist['radiation']['dycoms_f0']
        except:
            self.f0 = 70.0
        self.f1 = 22.0
        self.divergence = 3.75e-6

        return

    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        RadiationBase.initialize(self, Gr, NS, Pa)

        return

    cpdef initialize_profiles(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                     NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 Surface.SurfaceBase Sur,TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw

            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw

            Py_ssize_t pi, i, j, k, ijk, ishift, jshift
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t ql_shift = DV.get_varshift(Gr, 'ql')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t gw = Gr.dims.gw
            double [:, :] ql_pencils =  self.z_pencil.forward_double(&Gr.dims, Pa, &DV.values[ql_shift])
            double [:, :] qt_pencils =  self.z_pencil.forward_double(&Gr.dims, Pa, &PV.values[qt_shift])
            double[:, :] f_rad = np.empty((self.z_pencil.n_local_pencils, Gr.dims.n[2] + 1), dtype=np.double, order='c')
            double[:, :] f_heat = np.empty((self.z_pencil.n_local_pencils, Gr.dims.n[2]), dtype=np.double, order='c')
            double q_0
            double q_1

            double zi
            double rhoi
            double dz = Gr.dims.dx[2]
            double dzi = Gr.dims.dxi[2]
            double[:] z = Gr.zp
            double[:] rho = RS.rho0
            double[:] rho_half = RS.rho0_half
            double cbrt_z = 0

        with nogil:
            for pi in xrange(self.z_pencil.n_local_pencils):

                # Compute zi (level of 8.0 g/kg isoline of qt)
                for k in xrange(Gr.dims.n[2]):
                    if qt_pencils[pi, k] > 8e-3:
                        zi = z[gw + k]
                        rhoi = rho_half[gw + k]

                # Now compute the third term on RHS of Stevens et al 2005
                # (equation 3)
                f_rad[pi, 0] = 0.0
                for k in xrange(Gr.dims.n[2]):
                    if z[gw + k] >= zi:
                        cbrt_z = cbrt(z[gw + k] - zi)
                        f_rad[pi, k + 1] = rhoi * cpd * self.divergence * self.alpha_z * (pow(cbrt_z,4)  / 4.0
                                                                                     + zi * cbrt_z)
                    else:
                        f_rad[pi, k + 1] = 0.0

                # Compute the second term on RHS of Stevens et al. 2005
                # (equation 3)
                q_1 = 0.0
                f_rad[pi, 0] += self.f1 * exp(-q_1)
                for k in xrange(1, Gr.dims.n[2] + 1):
                    q_1 += self.kap * \
                        rho_half[gw + k - 1] * ql_pencils[pi, k - 1] * Gr.dims.dzpl_half[gw+k-1]
                    f_rad[pi, k] += self.f1 * exp(-q_1)

                # Compute the first term on RHS of Stevens et al. 2005
                # (equation 3)
                q_0 = 0.0
                f_rad[pi, Gr.dims.n[2]] += self.f0 * exp(-q_0)
                for k in xrange(Gr.dims.n[2] - 1, -1, -1):
                    q_0 += self.kap * rho_half[gw + k] * ql_pencils[pi, k] *  Gr.dims.dzpl_half[gw+k]
                    f_rad[pi, k] += self.f0 * exp(-q_0)

                for k in xrange(Gr.dims.n[2]):
                    f_heat[pi, k] = - \
                       (f_rad[pi, k + 1] - f_rad[pi, k]) * dzi * Gr.dims.imet_half[k] / rho_half[k]

        # Now transpose the flux pencils
        self.z_pencil.reverse_double(&Gr.dims, Pa, f_heat, &self.heating_rate[0])


        # Now update entropy tendencies
        with nogil:
            for i in xrange(imin, imax):
                ishift = i * istride
                for j in xrange(jmin, jmax):
                    jshift = j * jstride
                    for k in xrange(kmin, kmax):
                        ijk = ishift + jshift + k
                        PV.tendencies[
                            s_shift + ijk] +=  self.heating_rate[ijk] / DV.values[ijk + t_shift] 
                        self.dTdt_rad[ijk] = self.heating_rate[ijk] / cpm_c(PV.values[ijk + qt_shift]) 

        return

    cpdef stats_io(self, Grid.Grid Gr,  ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        RadiationBase.stats_io(self, Gr, RS, DV, NS,  Pa)


        return


cdef class RadiationSmoke(RadiationBase):
    '''
    Radiation for the smoke cloud case

    Bretherton, C. S., and coauthors, 1999:
    An intercomparison of radiatively- driven entrainment and turbulence in a smoke cloud,
    as simulated by different numerical models. Quart. J. Roy. Meteor. Soc., 125, 391-423. Full text copy.

    '''


    def __init__(self):
        self.f0 = 60.0
        self.kap = 0.02

        return

    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        RadiationBase.initialize(self, Gr, NS, Pa)
        return
    cpdef initialize_profiles(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                     NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 Surface.SurfaceBase Sur, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw

            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw

            Py_ssize_t pi, i, j, k, ijk, ishift, jshift
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t smoke_shift = PV.get_varshift(Gr, 'smoke')
            Py_ssize_t gw = Gr.dims.gw
            double [:, :] smoke_pencils =  self.z_pencil.forward_double(&Gr.dims, Pa, &PV.values[smoke_shift])
            double[:, :] f_rad = np.zeros((self.z_pencil.n_local_pencils, Gr.dims.n[2] + 1), dtype=np.double, order='c')
            double[:, :] f_heat = np.zeros((self.z_pencil.n_local_pencils, Gr.dims.n[2]), dtype=np.double, order='c')

            double q_0

            double zi
            double rhoi
            double dz = Gr.dims.dx[2]
            double dzi = Gr.dims.dxi[2]
            double[:] z = Gr.zp
            double[:] rho = RS.rho0
            double[:] rho_half = RS.rho0_half
            double cbrt_z = 0
            Py_ssize_t kk


        with nogil:
            for pi in xrange(self.z_pencil.n_local_pencils):

                q_0 = 0.0
                f_rad[pi, Gr.dims.n[2]] = self.f0 * exp(-q_0)
                for k in xrange(Gr.dims.n[2] - 1, -1, -1):
                    q_0 += self.kap * rho_half[gw + k] * smoke_pencils[pi, k] * Gr.dims.dzpl_half[gw+k]
                    f_rad[pi, k] = self.f0 * exp(-q_0)

                for k in xrange(Gr.dims.n[2]):
                    f_heat[pi, k] = - \
                       (f_rad[pi, k + 1] - f_rad[pi, k]) * dzi * Gr.dims.imet_half[k] / rho_half[k]

        # Now transpose the flux pencils
        self.z_pencil.reverse_double(&Gr.dims, Pa, f_heat, &self.heating_rate[0])


        # Now update entropy tendencies
        with nogil:
            for i in xrange(imin, imax):
                ishift = i * istride
                for j in xrange(jmin, jmax):
                    jshift = j * jstride
                    for k in xrange(kmin, kmax):
                        ijk = ishift + jshift + k
                        PV.tendencies[
                            s_shift + ijk] +=  self.heating_rate[ijk] / DV.values[ijk + t_shift] * RS.alpha0_half[k]
                        self.dTdt_rad[ijk] = self.heating_rate[ijk] / cpd * RS.alpha0_half[k]

        return

    cpdef stats_io(self, Grid.Grid Gr,  ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        RadiationBase.stats_io(self, Gr, RS, DV, NS,  Pa)

        return


# Note: the RRTM modules are compiled in the 'RRTMG' directory:
cdef extern:
    void c_rrtmg_lw_init(double *cpdair)
    void c_rrtmg_lw (
             int *ncol    ,int *nlay    ,int *icld    ,int *idrv    ,
             double *play    ,double *plev    ,double *tlay    ,double *tlev    ,double *tsfc    ,
             double *h2ovmr  ,double *o3vmr   ,double *co2vmr  ,double *ch4vmr  ,double *n2ovmr  ,double *o2vmr,
             double *cfc11vmr,double *cfc12vmr,double *cfc22vmr,double *ccl4vmr ,double *emis    ,
             int *inflglw ,int *iceflglw,int *liqflglw,double *cldfr   ,
             double *taucld  ,double *cicewp  ,double *cliqwp  ,double *reice   ,double *reliq   ,
             double *tauaer  ,
             double *uflx    ,double *dflx    ,double *hr      ,double *uflxc   ,double *dflxc,  double *hrc,
             double *duflx_dt,double *duflxc_dt )
    void c_rrtmg_sw_init(double *cpdair)
    void c_rrtmg_sw (int *ncol    ,int *nlay    ,int *icld    ,int *iaer    ,
             double *play    ,double *plev    ,double *tlay    ,double *tlev    ,double *tsfc    ,
             double *h2ovmr  ,double *o3vmr   ,double *co2vmr  ,double *ch4vmr  ,double *n2ovmr  ,double *o2vmr,
             double *asdir   ,double *asdif   ,double *aldir   ,double *aldif   ,
             double *coszen  ,double *adjes   ,int *dyofyr  ,double *scon    ,
             int *inflgsw ,int *iceflgsw,int *liqflgsw,double *cldfr   ,
             double *taucld  ,double *ssacld  ,double *asmcld  ,double *fsfcld  ,
             double *cicewp  ,double *cliqwp  ,double *reice   ,double *reliq   ,
             double *tauaer  ,double *ssaaer  ,double *asmaer  ,double *ecaer   ,
             double *swuflx  ,double *swdflx  ,double *swhr    ,double *swuflxc ,double *swdflxc ,double *swhrc)



cdef class RadiationRRTM(RadiationBase):

    def __init__(self, namelist, LatentHeat LH, ParallelMPI.ParallelMPI Pa):


        # Required for surface energy budget calculations, can also be used for stats io
        self.srf_lw_down = 0.0
        self.srf_sw_down = 0.0
        self.srf_lw_up = 0.0
        self.srf_sw_up = 0.0


        casename = namelist['meta']['casename']
        self.modified_adiabat = False
        if casename == 'SHEBA':
            self.profile_name = 'sheba'
        elif casename == 'DYCOMS_RF01':
            self.profile_name = 'cgils_ctl_s12'
        elif casename == 'CGILS':
            loc = namelist['meta']['CGILS']['location']
            is_p2 = namelist['meta']['CGILS']['P2']
            if is_p2:
                self.profile_name = 'cgils_p2_s'+str(loc)
            else:
                self.profile_name = 'cgils_ctl_s'+str(loc)
        elif casename == 'ZGILS':
            loc = namelist['meta']['ZGILS']['location']
            self.profile_name = 'cgils_ctl_s'+str(loc)
            self.modified_adiabat = True
            self.reference_profile = AdjustedMoistAdiabat(namelist, LH, Pa)
            self.Tg_adiabat = 295.0
            self.Pg_adiabat = 1000.0e2
            self.RH_adiabat = 0.3

        else:
            Pa.root_print('RadiationRRTM: Case ' + casename + ' has no known extension profile')
            Pa.kill()

        # Namelist options related to the profile extension
        try:
            self.n_buffer = namelist['radiation']['RRTM']['buffer_points']
        except:
            self.n_buffer = 0
        try:
            self.stretch_factor = namelist['radiation']['RRTM']['stretch_factor']
        except:
            self.stretch_factor = 1.0

        try:
            self.patch_pressure = namelist['radiation']['RRTM']['patch_pressure']
        except:
            self.patch_pressure = 1000.00*100.0

        # Namelist options related to gas concentrations
        try:
            self.co2_factor = namelist['radiation']['RRTM']['co2_factor']
        except:
            self.co2_factor = 1.0

        try:
            self.h2o_factor = namelist['radiation']['RRTM']['h2o_factor']
        except:
            self.h2o_factor = 1.0

        # Namelist options related to insolation
        try:
            self.dyofyr = namelist['radiation']['RRTM']['dyofyr']
        except:
            self.dyofyr = 0
        try:
            self.adjes = namelist['radiation']['RRTM']['adjes']
        except:
            Pa.root_print('Insolation adjustive factor not set so RadiationRRTM takes default value: adjes = 0.5 (12 hour of daylight).')
            self.adjes = 0.5

        try:
            self.scon = namelist['radiation']['RRTM']['solar_constant']
        except:
            Pa.root_print('Solar Constant not set so RadiationRRTM takes default value: scon = 1360.0 .')
            self.scon = 1360.0

        try:
            self.coszen =namelist['radiation']['RRTM']['coszen']
        except:
            Pa.root_print('Mean Daytime cos(SZA) not set so RadiationRRTM takes default value: coszen = 2.0/pi .')
            self.coszen = 2.0/pi

        try:
            self.adif = namelist['radiation']['RRTM']['adif']
        except:
            Pa.root_print('Surface diffusive albedo not set so RadiationRRTM takes default value: adif = 0.06 .')
            self.adif = 0.06

        try:
            self.adir = namelist['radiation']['RRTM']['adir']
        except:
            if (self.coszen > 0.0):
                self.adir = (.026/(self.coszen**1.7 + .065)+(.15*(self.coszen-0.10)*(self.coszen-0.50)*(self.coszen- 1.00)))
            else:
                self.adir = 0.0
            Pa.root_print('Surface direct albedo not set so RadiationRRTM computes value: adif = %5.4f .'%(self.adir))

        try:
            self.uniform_reliq = namelist['radiation']['RRTM']['uniform_reliq']
        except:
            Pa.root_print('uniform_reliq not set so RadiationRRTM takes default value: uniform_reliq = False.')
            self.uniform_reliq = False

        try:
            self.radiation_frequency = namelist['radiation']['RRTM']['frequency']
        except:
            Pa.root_print('radiation_frequency not set so RadiationRRTM takes default value: radiation_frequency = 0.0 (compute at every step).')
            self.radiation_frequency = 0.0



        self.next_radiation_calculate = 0.0



        return


    cpdef initialize(self, Grid.Grid Gr,  NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        RadiationBase.initialize(self, Gr, NS, Pa)
        return



    cpdef initialize_profiles(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                     NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):


        cdef:
            Py_ssize_t qv_shift = DV.get_varshift(Gr, 'qv')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            double [:,:] qv_pencils =  self.z_pencil.forward_double(&Gr.dims, Pa, &DV.values[qv_shift])
            double [:,:] t_pencils =  self.z_pencil.forward_double(&Gr.dims, Pa, &DV.values[t_shift])
            Py_ssize_t nz = Gr.dims.n[2]
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t i,k
            Py_ssize_t n_adiabat
            double [:] pressures_adiabat


        # Construct the extension of the profiles, including a blending region between the given profile and LES domain (if desired)
        if self.modified_adiabat:
            # pressures = profile_data[self.profile_name]['pressure'][:]
            pressures = np.arange(25*100, 1015*100, 10*100)
            pressures = np.array(pressures[::-1], dtype=np.double)
            n_adiabat = np.shape(pressures)[0]
            self.reference_profile.initialize(Pa, pressures, n_adiabat, self.Pg_adiabat, self.Tg_adiabat, self.RH_adiabat)
            temperatures =np.array( self.reference_profile.temperature)
            vapor_mixing_ratios = np.array(self.reference_profile.rv)

        else:
            pressures = profile_data[self.profile_name]['pressure'][:]
            temperatures = profile_data[self.profile_name]['temperature'][:]
            vapor_mixing_ratios = profile_data[self.profile_name]['vapor_mixing_ratio'][:]


        # Sanity check that patch_pressure < minimum LES domain pressure
        dp = np.abs(RS.p0_half_global[nz + gw -1] - RS.p0_half_global[nz + gw -2])
        self.patch_pressure = np.minimum(self.patch_pressure, RS.p0_half_global[nz + gw -1] - dp  )

        #n_profile = len(pressures[pressures<=self.patch_pressure]) # nprofile = # of points in the fixed profile to use
        # above syntax tends to cause problems so use a more robust way
        n_profile = 0
        for pressure in pressures:
            if pressure <= self.patch_pressure:
                n_profile += 1

        self.n_ext =  n_profile + self.n_buffer # n_ext = total # of points to add to LES domain (buffer portion + fixed profile portion)


        # Create the space for the extensions (to be tacked on to top of LES pencils)
        # we declare these as class members in case we want to modify the buffer zone during run time
        # i.e. if there is some drift to top of LES profiles

        self.p_ext = np.zeros((self.n_ext,),dtype=np.double)
        self.t_ext = np.zeros((self.n_ext,),dtype=np.double)
        self.rv_ext = np.zeros((self.n_ext,),dtype=np.double)
        cdef Py_ssize_t count = 0
        for k in xrange(len(pressures)-n_profile, len(pressures)):
            self.p_ext[self.n_buffer+count] = pressures[k]
            self.t_ext[self.n_buffer+count] = temperatures[k]
            self.rv_ext[self.n_buffer+count] = vapor_mixing_ratios[k]
            count += 1


        # Now  create the buffer zone
        if self.n_buffer > 0:
            dp = np.abs(RS.p0_half_global[nz + gw -1] - RS.p0_half_global[nz + gw -2])
            self.p_ext[0] = RS.p0_half_global[nz + gw -1] - dp
            for i in range(1,self.n_buffer):
                self.p_ext[i] = self.p_ext[i-1] - (i+1.0)**self.stretch_factor * dp


            # Sanity check the buffer zone
            if self.p_ext[self.n_buffer-1] < self.p_ext[self.n_buffer]:
                Pa.root_print('Radiation buffer zone extends too far')
                Pa.kill()

            # Pressures of "data" points for interpolation, must be INCREASING pressure
            xi = np.array([self.p_ext[self.n_buffer+1],self.p_ext[self.n_buffer],RS.p0_half_global[nz + gw -1],RS.p0_half_global[nz + gw -2] ],dtype=np.double)

            # interpolation for temperature
            ti = np.array([self.t_ext[self.n_buffer+1],self.t_ext[self.n_buffer], t_pencils[0,nz-1],t_pencils[0,nz-2] ], dtype = np.double)
            # interpolation for vapor mixing ratio
            rv_m2 = qv_pencils[0, nz-2]/ (1.0 - qv_pencils[0, nz-2])
            rv_m1 = qv_pencils[0,nz-1]/(1.0-qv_pencils[0,nz-1])
            ri = np.array([self.rv_ext[self.n_buffer+1],self.rv_ext[self.n_buffer], rv_m1, rv_m2 ], dtype = np.double)

            for i in xrange(self.n_buffer):
                self.rv_ext[i] = pchip_interpolate(xi, ri, self.p_ext[i] )
                self.t_ext[i] = pchip_interpolate(xi,ti, self.p_ext[i])


        #--- Plotting to evaluate implementation of buffer zone
        #--- Comment out when not running locally
        for i in xrange(Gr.dims.nlg[2]):
            qv_pencils[0,i] = qv_pencils[0, i]/ (1.0 - qv_pencils[0, i])
        #
        # Plotting to evaluate implementation of buffer zone
        # plt.figure(1)
        # plt.plot(self.rv_ext,self.p_ext,'or')
        # plt.plot(vapor_mixing_ratios, pressures)
        # plt.plot(qv_pencils[0,:], RS.p0_half_global[gw:-gw],'ob')
        # plt.gca().invert_yaxis()
        # plt.figure(2)
        # plt.plot(self.t_ext,self.p_ext,'-or')
        # plt.plot(temperatures,pressures)
        # plt.plot(t_pencils[0,:], RS.p0_half_global[gw:-gw],'-ob')
        # plt.gca().invert_yaxis()
        # plt.show()
        #---END Plotting to evaluate implementation of buffer zone


        self.p_full = np.zeros((self.n_ext+nz,), dtype=np.double)
        self.pi_full = np.zeros((self.n_ext+1+nz,),dtype=np.double)

        self.p_full[0:nz] = RS.p0_half_global[gw:nz+gw]
        self.p_full[nz:]=self.p_ext[:]

        self.pi_full[0:nz] = RS.p0_global[gw:nz+gw]
        for i in range(nz,self.n_ext+nz):
            self.pi_full[i] = (self.p_full[i] + self.p_full[i-1]) * 0.5
        self.pi_full[self.n_ext +  nz] = 2.0 * self.p_full[self.n_ext + nz -1 ] - self.pi_full[self.n_ext + nz -1]

        # try to get ozone
        try:
            o3_trace = profile_data[self.profile_name]['o3_vmr'][:]   # O3 VMR (from SRF to TOP)
            o3_pressure = profile_data[self.profile_name]['pressure'][:]/100.0       # Pressure (from SRF to TOP) in hPa
            # can't do simple interpolation... Need to conserve column path !!!
            use_o3in = True
        except:
            try:
                o3_trace = profile_data[self.profile_name]['o3_mmr'][:]*28.97/47.9982   # O3 MR converted to VMR
                o3_pressure = profile_data[self.profile_name]['pressure'][:]/100.0       # Pressure (from SRF to TOP) in hPa
                # can't do simple interpolation... Need to conserve column path !!!
                use_o3in = True

            except:
                Pa.root_print('O3 profile not set so default RRTM profile will be used.')
                use_o3in = False

        #Initialize rrtmg_lw and rrtmg_sw
        cdef double cpdair = np.float64(cpd)
        c_rrtmg_lw_init(&cpdair)
        c_rrtmg_sw_init(&cpdair)

        # Read in trace gas data
        lw_input_file = './RRTMG/lw/data/rrtmg_lw.nc'
        lw_gas = nc.Dataset(lw_input_file,  "r")

        lw_pressure = np.asarray(lw_gas.variables['Pressure'])
        lw_absorber = np.asarray(lw_gas.variables['AbsorberAmountMLS'])
        lw_absorber = np.where(lw_absorber>2.0, np.zeros_like(lw_absorber), lw_absorber)
        lw_ngas = lw_absorber.shape[1]
        lw_np = lw_absorber.shape[0]

        # 9 Gases: O3, CO2, CH4, N2O, O2, CFC11, CFC12, CFC22, CCL4
        # From rad_driver.f90, lines 546 to 552
        trace = np.zeros((9,lw_np),dtype=np.double,order='F')
        for i in xrange(lw_ngas):
            gas_name = ''.join(lw_gas.variables['AbsorberNames'][i,:])
            if 'O3' in gas_name:
                trace[0,:] = lw_absorber[:,i].reshape(1,lw_np)
            elif 'CO2' in gas_name:
                trace[1,:] = lw_absorber[:,i].reshape(1,lw_np)*self.co2_factor
            elif 'CH4' in gas_name:
                trace[2,:] = lw_absorber[:,i].reshape(1,lw_np)
            elif 'N2O' in gas_name:
                trace[3,:] = lw_absorber[:,i].reshape(1,lw_np)
            elif 'O2' in gas_name:
                trace[4,:] = lw_absorber[:,i].reshape(1,lw_np)
            elif 'CFC11' in gas_name:
                trace[5,:] = lw_absorber[:,i].reshape(1,lw_np)
            elif 'CFC12' in gas_name:
                trace[6,:] = lw_absorber[:,i].reshape(1,lw_np)
            elif 'CFC22' in gas_name:
                trace[7,:] = lw_absorber[:,i].reshape(1,lw_np)
            elif 'CCL4' in gas_name:
                trace[8,:] = lw_absorber[:,i].reshape(1,lw_np)

        # From rad_driver.f90, lines 585 to 620
        trpath = np.zeros((nz + self.n_ext + 1, 9),dtype=np.double,order='F')
        # plev = self.pi_full[:]/100.0
        for i in xrange(1, nz + self.n_ext + 1):
            trpath[i,:] = trpath[i-1,:]
            if (self.pi_full[i-1]/100.0 > lw_pressure[0]):
                trpath[i,:] = trpath[i,:] + (self.pi_full[i-1]/100.0 - np.max((self.pi_full[i]/100.0,lw_pressure[0])))/g*trace[:,0]
            for m in xrange(1,lw_np):
                plow = np.min((self.pi_full[i-1]/100.0,np.max((self.pi_full[i]/100.0, lw_pressure[m-1]))))
                pupp = np.min((self.pi_full[i-1]/100.0,np.max((self.pi_full[i]/100.0, lw_pressure[m]))))
                if (plow > pupp):
                    pmid = 0.5*(plow+pupp)
                    wgtlow = (pmid-lw_pressure[m])/(lw_pressure[m-1]-lw_pressure[m])
                    wgtupp = (lw_pressure[m-1]-pmid)/(lw_pressure[m-1]-lw_pressure[m])
                    trpath[i,:] = trpath[i,:] + (plow-pupp)/g*(wgtlow*trace[:,m-1]  + wgtupp*trace[:,m])
            if (self.pi_full[i]/100.0 < lw_pressure[lw_np-1]):
                trpath[i,:] = trpath[i,:] + (np.min((self.pi_full[i-1]/100.0,lw_pressure[lw_np-1]))-self.pi_full[i]/100.0)/g*trace[:,lw_np-1]

        tmpTrace = np.zeros((nz + self.n_ext,9),dtype=np.double,order='F')
        for i in xrange(9):
            for k in xrange(nz + self.n_ext):
                tmpTrace[k,i] = g*100.0/(self.pi_full[k]-self.pi_full[k+1])*(trpath[k+1,i]-trpath[k,i])

        if use_o3in == False:
            self.o3vmr  = np.array(tmpTrace[:,0],dtype=np.double, order='F')
        else:
            # o3_trace, o3_pressure
            trpath_o3 = np.zeros(nz + self.n_ext+1, dtype=np.double, order='F')
            # plev = self.pi_full/100.0
            o3_np = o3_trace.shape[0]
            for i in xrange(1, nz + self.n_ext+1):
                trpath_o3[i] = trpath_o3[i-1]
                if (self.pi_full[i-1]/100.0 > o3_pressure[0]):
                    trpath_o3[i] = trpath_o3[i] + (self.pi_full[i-1]/100.0 - np.max((self.pi_full[i]/100.0,o3_pressure[0])))/g*o3_trace[0]
                for m in xrange(1,o3_np):
                    plow = np.min((self.pi_full[i-1]/100.0,np.max((self.pi_full[i]/100.0, o3_pressure[m-1]))))
                    pupp = np.min((self.pi_full[i-1]/100.0,np.max((self.pi_full[i]/100.0, o3_pressure[m]))))
                    if (plow > pupp):
                        pmid = 0.5*(plow+pupp)
                        wgtlow = (pmid-o3_pressure[m])/(o3_pressure[m-1]-o3_pressure[m])
                        wgtupp = (o3_pressure[m-1]-pmid)/(o3_pressure[m-1]-o3_pressure[m])
                        trpath_o3[i] = trpath_o3[i] + (plow-pupp)/g*(wgtlow*o3_trace[m-1]  + wgtupp*o3_trace[m])
                if (self.pi_full[i]/100.0 < o3_pressure[o3_np-1]):
                    trpath_o3[i] = trpath_o3[i] + (np.min((self.pi_full[i-1]/100.0,o3_pressure[o3_np-1]))-self.pi_full[i]/100.0)/g*o3_trace[o3_np-1]
            tmpTrace_o3 = np.zeros( nz + self.n_ext, dtype=np.double, order='F')
            for k in xrange(nz + self.n_ext):
                tmpTrace_o3[k] = g *100.0/(self.pi_full[k]-self.pi_full[k+1])*(trpath_o3[k+1]-trpath_o3[k])
            self.o3vmr = np.array(tmpTrace_o3[:],dtype=np.double, order='F')

        self.co2vmr = np.array(tmpTrace[:,1],dtype=np.double, order='F')
        self.ch4vmr =  np.array(tmpTrace[:,2],dtype=np.double, order='F')
        self.n2ovmr =  np.array(tmpTrace[:,3],dtype=np.double, order='F')
        self.o2vmr  =  np.array(tmpTrace[:,4],dtype=np.double, order='F')
        self.cfc11vmr =  np.array(tmpTrace[:,5],dtype=np.double, order='F')
        self.cfc12vmr =  np.array(tmpTrace[:,6],dtype=np.double, order='F')
        self.cfc22vmr = np.array( tmpTrace[:,7],dtype=np.double, order='F')
        self.ccl4vmr  =  np.array(tmpTrace[:,8],dtype=np.double, order='F')


        return
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 Surface.SurfaceBase Sur, TimeStepping.TimeStepping TS,
                 ParallelMPI.ParallelMPI Pa):


        if TS.rk_step == 0:
            if self.radiation_frequency <= 0.0:
                self.update_RRTM(Gr, RS, PV, DV,Sur, Pa)
            elif TS.t >= self.next_radiation_calculate:
                self.update_RRTM(Gr, RS, PV, DV, Sur, Pa)
                self.next_radiation_calculate = (TS.t//self.radiation_frequency + 1.0) * self.radiation_frequency


        cdef:
            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw


            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw

            Py_ssize_t i, j, k, ijk, ishift, jshift
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')



        # Now update entropy tendencies
        with nogil:
            for i in xrange(imin, imax):
                ishift = i * istride
                for j in xrange(jmin, jmax):
                    jshift = j * jstride
                    for k in xrange(kmin, kmax):
                        ijk = ishift + jshift + k
                        PV.tendencies[
                            s_shift + ijk] +=  self.heating_rate[ijk] / DV.values[ijk + t_shift] * RS.alpha0_half[k]
                        self.dTdt_rad[ijk] = self.heating_rate[ijk] * RS.alpha0_half[k]/cpm_c(PV.values[ijk + qt_shift])


        return

    cdef update_RRTM(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                      DiagnosticVariables.DiagnosticVariables DV, Surface.SurfaceBase Sur, ParallelMPI.ParallelMPI Pa):
        cdef:
            Py_ssize_t nz = Gr.dims.n[2]
            Py_ssize_t nz_full = self.n_ext + nz
            Py_ssize_t n_pencils = self.z_pencil.n_local_pencils
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t qv_shift = DV.get_varshift(Gr, 'qv')
            Py_ssize_t ql_shift = DV.get_varshift(Gr, 'ql')
            Py_ssize_t qi_shift
            double [:,:] t_pencil = self.z_pencil.forward_double(&Gr.dims, Pa, &DV.values[t_shift])
            double [:,:] qv_pencil = self.z_pencil.forward_double(&Gr.dims, Pa, &DV.values[qv_shift])
            double [:,:] ql_pencil = self.z_pencil.forward_double(&Gr.dims, Pa, &DV.values[ql_shift])
            double [:,:] qi_pencil = np.zeros((n_pencils,nz),dtype=np.double, order='c')
            double [:,:] rl_full = np.zeros((n_pencils,nz_full), dtype=np.double, order='F')
            Py_ssize_t k, ip
            bint use_ice = False
            Py_ssize_t gw = Gr.dims.gw


        if 'qi' in DV.name_index:
            qi_shift = DV.get_varshift(Gr, 'qi')
            qi_pencil = self.z_pencil.forward_double(&Gr.dims, Pa, &DV.values[qi_shift])
            use_ice = True



        # Define input arrays for RRTM
        cdef:
            double [:,:] play_in = np.zeros((n_pencils,nz_full), dtype=np.double, order='F')
            double [:,:] plev_in = np.zeros((n_pencils,nz_full + 1), dtype=np.double, order='F')
            double [:,:] tlay_in = np.zeros((n_pencils,nz_full), dtype=np.double, order='F')
            double [:,:] tlev_in = np.zeros((n_pencils,nz_full + 1), dtype=np.double, order='F')
            double [:] tsfc_in = np.ones((n_pencils),dtype=np.double,order='F') * Sur.T_surface
            double [:,:] h2ovmr_in = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] o3vmr_in  = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] co2vmr_in = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] ch4vmr_in = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] n2ovmr_in = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] o2vmr_in  = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] cfc11vmr_in = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] cfc12vmr_in = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] cfc22vmr_in = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] ccl4vmr_in = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] emis_in = np.ones((n_pencils,16),dtype=np.double,order='F') * 0.95
            double [:,:] cldfr_in  = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] cicewp_in = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] cliqwp_in = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] reice_in  = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:] reliq_in  = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double [:] coszen_in = np.ones((n_pencils),dtype=np.double,order='F') *self.coszen
            double [:] asdir_in = np.ones((n_pencils),dtype=np.double,order='F') * self.adir
            double [:] asdif_in = np.ones((n_pencils),dtype=np.double,order='F') * self.adif
            double [:] aldir_in = np.ones((n_pencils),dtype=np.double,order='F') * self.adir
            double [:] aldif_in = np.ones((n_pencils),dtype=np.double,order='F') * self.adif
            double [:,:,:] taucld_lw_in  = np.zeros((16,n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:,:] tauaer_lw_in  = np.zeros((n_pencils,nz_full,16),dtype=np.double,order='F')
            double [:,:,:] taucld_sw_in  = np.zeros((14,n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:,:] ssacld_sw_in  = np.zeros((14,n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:,:] asmcld_sw_in  = np.zeros((14,n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:,:] fsfcld_sw_in  = np.zeros((14,n_pencils,nz_full),dtype=np.double,order='F')
            double [:,:,:] tauaer_sw_in  = np.zeros((n_pencils,nz_full,14),dtype=np.double,order='F')
            double [:,:,:] ssaaer_sw_in  = np.zeros((n_pencils,nz_full,14),dtype=np.double,order='F')
            double [:,:,:] asmaer_sw_in  = np.zeros((n_pencils,nz_full,14),dtype=np.double,order='F')
            double [:,:,:] ecaer_sw_in  = np.zeros((n_pencils,nz_full,6),dtype=np.double,order='F')

            # Output
            double[:,:] uflx_lw_out = np.zeros((n_pencils,nz_full +1),dtype=np.double,order='F')
            double[:,:] dflx_lw_out = np.zeros((n_pencils,nz_full +1),dtype=np.double,order='F')
            double[:,:] hr_lw_out = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double[:,:] uflxc_lw_out = np.zeros((n_pencils,nz_full +1),dtype=np.double,order='F')
            double[:,:] dflxc_lw_out = np.zeros((n_pencils,nz_full +1),dtype=np.double,order='F')
            double[:,:] hrc_lw_out = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double[:,:] duflx_dt_out = np.zeros((n_pencils,nz_full +1),dtype=np.double,order='F')
            double[:,:] duflxc_dt_out = np.zeros((n_pencils,nz_full +1),dtype=np.double,order='F')
            double[:,:] uflx_sw_out = np.zeros((n_pencils,nz_full +1),dtype=np.double,order='F')
            double[:,:] dflx_sw_out = np.zeros((n_pencils,nz_full +1),dtype=np.double,order='F')
            double[:,:] hr_sw_out = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')
            double[:,:] uflxc_sw_out = np.zeros((n_pencils,nz_full +1),dtype=np.double,order='F')
            double[:,:] dflxc_sw_out = np.zeros((n_pencils,nz_full +1),dtype=np.double,order='F')
            double[:,:] hrc_sw_out = np.zeros((n_pencils,nz_full),dtype=np.double,order='F')

            double rv_to_reff = np.exp(np.log(1.2)**2.0)*10.0*1000.0

        with nogil:
            for k in xrange(nz, nz_full):
                for ip in xrange(n_pencils):
                    tlay_in[ip, k] = self.t_ext[k-nz]
                    h2ovmr_in[ip, k] = self.rv_ext[k-nz] * Rv/Rd * self.h2o_factor
                    # Assuming for now that there is no condensate above LES domain!
            for k in xrange(nz):
                for ip in xrange(n_pencils):
                    tlay_in[ip,k] = t_pencil[ip,k]
                    h2ovmr_in[ip,k] = qv_pencil[ip,k]/ (1.0 - qv_pencil[ip,k])* Rv/Rd * self.h2o_factor
                    rl_full[ip,k] = (ql_pencil[ip,k])/ (1.0 - qv_pencil[ip,k])
                    cliqwp_in[ip,k] = ((ql_pencil[ip,k])/ (1.0 - qv_pencil[ip,k])
                                       *1.0e3*(self.pi_full[k] - self.pi_full[k+1])/g)
                    cicewp_in[ip,k] = ((qi_pencil[ip,k])/ (1.0 - qv_pencil[ip,k])
                                       *1.0e3*(self.pi_full[k] - self.pi_full[k+1])/g)
                    if ql_pencil[ip,k] + qi_pencil[ip,k] > ql_threshold:
                        cldfr_in[ip,k] = 1.0


        with nogil:
            for k in xrange(nz_full):
                for ip in xrange(n_pencils):
                    play_in[ip,k] = self.p_full[k]/100.0
                    o3vmr_in[ip, k] = self.o3vmr[k]
                    co2vmr_in[ip, k] = self.co2vmr[k]
                    ch4vmr_in[ip, k] = self.ch4vmr[k]
                    n2ovmr_in[ip, k] = self.n2ovmr[k]
                    o2vmr_in [ip, k] = self.o2vmr[k]
                    cfc11vmr_in[ip, k] = self.cfc11vmr[k]
                    cfc12vmr_in[ip, k] = self.cfc12vmr[k]
                    cfc22vmr_in[ip, k] = self.cfc22vmr[k]
                    ccl4vmr_in[ip, k] = self.ccl4vmr[k]


                    if self.uniform_reliq:
                        reliq_in[ip, k] = 14.0*cldfr_in[ip,k]
                    else:
                        reliq_in[ip, k] = ((3.0*self.p_full[k]/Rd/tlay_in[ip,k]*rl_full[ip,k]/
                                                    fmax(cldfr_in[ip,k],1.0e-6))/(4.0*pi*1.0e3*100.0))**(1.0/3.0)
                        reliq_in[ip, k] = fmin(fmax(reliq_in[ip, k]*rv_to_reff, 2.5), 60.0)

            for ip in xrange(n_pencils):
                tlev_in[ip, 0] = Sur.T_surface
                plev_in[ip,0] = self.pi_full[0]/100.0
                for k in xrange(1,nz_full):
                    tlev_in[ip, k] = 0.5*(tlay_in[ip,k-1]+tlay_in[ip,k])
                    plev_in[ip,k] = self.pi_full[k]/100.0
                tlev_in[ip, nz_full] = 2.0*tlay_in[ip,nz_full-1] - tlev_in[ip,nz_full-1]
                plev_in[ip,nz_full] = self.pi_full[nz_full]/100.0


        cdef:
            int ncol = n_pencils
            int nlay = nz_full
            int icld = 1
            int idrv = 0
            int iaer = 0
            int inflglw = 2
            int iceflglw = 3
            int liqflglw = 1
            int inflgsw = 2
            int iceflgsw = 3
            int liqflgsw = 1

        c_rrtmg_lw (
             &ncol    ,&nlay    ,&icld    ,&idrv,
             &play_in[0,0]    ,&plev_in[0,0]    ,&tlay_in[0,0]    ,&tlev_in[0,0]    ,&tsfc_in[0]    ,
             &h2ovmr_in[0,0]  ,&o3vmr_in[0,0]   ,&co2vmr_in[0,0]  ,&ch4vmr_in[0,0]  ,&n2ovmr_in[0,0]  ,&o2vmr_in[0,0],
             &cfc11vmr_in[0,0],&cfc12vmr_in[0,0],&cfc22vmr_in[0,0],&ccl4vmr_in[0,0] ,&emis_in[0,0]    ,
             &inflglw ,&iceflglw,&liqflglw,&cldfr_in[0,0]   ,
             &taucld_lw_in[0,0,0]  ,&cicewp_in[0,0]  ,&cliqwp_in[0,0]  ,&reice_in[0,0]   ,&reliq_in[0,0]   ,
             &tauaer_lw_in[0,0,0]  ,
             &uflx_lw_out[0,0]    ,&dflx_lw_out[0,0]    ,&hr_lw_out[0,0]      ,&uflxc_lw_out[0,0]   ,&dflxc_lw_out[0,0],  &hrc_lw_out[0,0],
             &duflx_dt_out[0,0],&duflxc_dt_out[0,0] )


        c_rrtmg_sw (
            &ncol, &nlay, &icld, &iaer, &play_in[0,0], &plev_in[0,0], &tlay_in[0,0], &tlev_in[0,0],&tsfc_in[0],
            &h2ovmr_in[0,0], &o3vmr_in[0,0], &co2vmr_in[0,0], &ch4vmr_in[0,0], &n2ovmr_in[0,0],&o2vmr_in[0,0],
             &asdir_in[0]   ,&asdif_in[0]   ,&aldir_in[0]   ,&aldif_in[0]   ,
             &coszen_in[0]  ,&self.adjes   ,&self.dyofyr  ,&self.scon   ,
             &inflgsw ,&iceflgsw,&liqflgsw,&cldfr_in[0,0]   ,
             &taucld_sw_in[0,0,0]  ,&ssacld_sw_in[0,0,0]  ,&asmcld_sw_in[0,0,0]  ,&fsfcld_sw_in[0,0,0]  ,
             &cicewp_in[0,0]  ,&cliqwp_in[0,0]  ,&reice_in[0,0]   ,&reliq_in[0,0]   ,
             &tauaer_sw_in[0,0,0]  ,&ssaaer_sw_in[0,0,0]  ,&asmaer_sw_in[0,0,0]  ,&ecaer_sw_in[0,0,0]   ,
             &uflx_sw_out[0,0]    ,&dflx_sw_out[0,0]    ,&hr_sw_out[0,0]      ,&uflxc_sw_out[0,0]   ,&dflxc_sw_out[0,0], &hrc_sw_out[0,0])

        cdef double [:,:] heating_rate_pencil = np.zeros((n_pencils,nz), dtype=np.double, order='c')
        cdef double srf_lw_up_local =0.0, srf_lw_down_local=0.0, srf_sw_up_local=0.0, srf_sw_down_local=0.0
        cdef double nxny_i = 1.0/(Gr.dims.n[0]*Gr.dims.n[1])
        with nogil:
           for ip in xrange(n_pencils):
               srf_lw_up_local   += uflx_lw_out[ip,0] * nxny_i
               srf_lw_down_local += dflx_lw_out[ip,0] * nxny_i
               srf_sw_up_local   +=  uflx_sw_out[ip,0] * nxny_i
               srf_sw_down_local += dflx_sw_out[ip,0] * nxny_i
               for k in xrange(nz):
                   heating_rate_pencil[ip, k] = (hr_lw_out[ip,k] + hr_sw_out[ip,k]) * RS.rho0_half_global[k+gw] * cpm_c(qv_pencil[ip,k])/86400.0

        self.srf_lw_up = Pa.domain_scalar_sum(srf_lw_up_local)
        self.srf_lw_down = Pa.domain_scalar_sum(srf_lw_down_local)
        self.srf_sw_up= Pa.domain_scalar_sum(srf_sw_up_local)
        self.srf_sw_down= Pa.domain_scalar_sum(srf_sw_down_local)


        self.z_pencil.reverse_double(&Gr.dims, Pa, heating_rate_pencil, &self.heating_rate[0])



        return
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        RadiationBase.stats_io(self, Gr, RS, DV, NS,  Pa)



        return

cdef class RadiationTRMM_LBA(RadiationBase):

    def __init__(self, namelist, LatentHeat LH, ParallelMPI.ParallelMPI Pa):

        self.rad_time     = np.linspace(10,360,36)*60
        # radiation time is 10min : 10:min :360min
        self.z_in         = np.array([42.5, 200.92, 456.28, 743, 1061.08, 1410.52, 1791.32, 2203.48, 2647,
                                      3121.88, 3628.12, 4165.72, 4734.68, 5335, 5966.68, 6629.72, 7324.12,
                                      8049.88, 8807, 9595.48, 10415.32, 11266.52, 12149.08, 13063, 14008.28,
                                      14984.92, 15992.92, 17032.28, 18103, 19205.08, 20338.52, 21503.32, 22699.48])
        # a[i,j] - here i is the number of vector bounded by [] which corresponds to time, j is the number of element in each vector that corresponds to height
        self.rad_in=np.array([[-1.386, -1.927, -2.089, -1.969, -1.805, -1.585, -1.406, -1.317, -1.188, -1.106, -1.103, -1.025,
                              -0.955, -1.045, -1.144, -1.119, -1.068, -1.092, -1.196, -1.253, -1.266, -1.306,  -0.95,  0.122,
                               0.255,  0.258,  0.322,  0.135,    0.0,    0.0,    0.0,    0.0,    0.0],
                             [ -1.23, -1.824, -2.011, -1.895, -1.729, -1.508, -1.331, -1.241, -1.109, -1.024, -1.018,  -0.94,
                              -0.867, -0.953, -1.046, -1.018, -0.972, -1.006, -1.119, -1.187, -1.209, -1.259, -0.919,  0.122,
                               0.264,  0.262,  0.326,  0.137,    0.0,    0.0,    0.0,    0.0,    0.0],
                             [-1.043, -1.692, -1.906, -1.796,  -1.63,  -1.41, -1.233, -1.142,  -1.01,  -0.92, -0.911, -0.829,
                              -0.754, -0.837, -0.923,  -0.89, -0.847, -0.895, -1.021, -1.101, -1.138, -1.201,  -0.88,  0.131,
                               0.286,  0.259,  0.332,   0.14,    0.0,    0.0,    0.0,    0.0,    0.0],
                             [-0.944, -1.613, -1.832,  -1.72, -1.555, -1.339, -1.163, -1.068, -0.935, -0.846, -0.835,  -0.75,
                              -0.673, -0.751, -0.833, -0.798,  -0.76, -0.817, -0.952, -1.042, -1.088, -1.159, -0.853,  0.138,
                               0.291,  0.265,  0.348,  0.136,    0.0,    0.0,    0.0,    0.0,    0.0],
                             [-0.833, -1.526, -1.757, -1.648, -1.485,  -1.27, -1.093, -0.998, -0.867, -0.778, -0.761, -0.672,
                              -0.594, -0.671, -0.748, -0.709, -0.676, -0.742, -0.887, -0.986, -1.041, -1.119, -0.825,  0.143,
                               0.296,  0.271,  0.351,  0.138,    0.0,    0.0,    0.0,    0.0,    0.0],
                             [-0.719, -1.425, -1.657,  -1.55, -1.392, -1.179, -1.003, -0.909, -0.778, -0.688, -0.667, -0.573,
                              -0.492, -0.566, -0.639, -0.596, -0.568, -0.647, -0.804, -0.914, -0.981,  -1.07, -0.793,  0.151,
                               0.303,  0.279,  0.355,  0.141,    0.0,    0.0,    0.0,    0.0,    0.0],
                             [-0.724, -1.374, -1.585, -1.482, -1.328, -1.116, -0.936, -0.842, -0.715, -0.624, -0.598, -0.503,
                              -0.421, -0.494, -0.561, -0.514,  -0.49,  -0.58, -0.745, -0.863, -0.938, -1.035, -0.764,  0.171,
                               0.291,  0.284,  0.358,  0.144,    0.0,    0.0,    0.0,    0.0,    0.0],
                             [-0.587,  -1.28, -1.513, -1.416, -1.264, -1.052, -0.874, -0.781, -0.655, -0.561, -0.532, -0.436,
                              -0.354, -0.424, -0.485, -0.435, -0.417, -0.517, -0.691, -0.817, -0.898,     -1,  -0.74,  0.176,
                               0.297,  0.289,   0.36,  0.146,    0.0,    0.0,    0.0,    0.0,    0.0],
                             [-0.506, -1.194, -1.426, -1.332, -1.182, -0.972, -0.795, -0.704, -0.578,  -0.48, -0.445, -0.347,
                              -0.267, -0.336, -0.391, -0.337, -0.325, -0.436,  -0.62, -0.756, -0.847,  -0.96, -0.714,   0.18,
                               0.305,  0.317,  0.348,  0.158,    0.0,    0.0,    0.0,    0.0,    0.0],
                             [-0.472,  -1.14, -1.364, -1.271, -1.123, -0.914, -0.738, -0.649, -0.522, -0.422, -0.386, -0.287,
                              -0.207, -0.273, -0.322, -0.267,  -0.26, -0.379, -0.569, -0.712, -0.811, -0.931, -0.696,  0.183,
                               0.311,   0.32,  0.351,   0.16,    0.0,    0.0,    0.0,    0.0,    0.0],
                             [-0.448, -1.091, -1.305, -1.214, -1.068, -0.858, -0.682, -0.594, -0.469, -0.368, -0.329, -0.229,
                              -0.149, -0.213, -0.257,   -0.2, -0.199, -0.327, -0.523, -0.668, -0.774, -0.903, -0.678,  0.186,
                               0.315,  0.323,  0.355,  0.162,    0.0,    0.0,    0.0,    0.0,    0.0],
                             [-0.405, -1.025, -1.228, -1.139, -0.996, -0.789, -0.615, -0.527, -0.402,   -0.3, -0.256, -0.156,
                              -0.077, -0.136, -0.173, -0.115, -0.121, -0.259, -0.463, -0.617, -0.732, -0.869, -0.656,   0.19,
                               0.322,  0.326,  0.359,  0.164,    0.0,    0.0,    0.0,    0.0,    0.0],
                             [-0.391, -0.983, -1.174, -1.085, -0.945, -0.739, -0.566, -0.478, -0.354, -0.251, -0.205, -0.105,
                              -0.027, -0.082, -0.114, -0.056, -0.069, -0.213,  -0.42, -0.579, -0.699,  -0.84, -0.642,  0.173,
                               0.327,  0.329,  0.362,  0.165,    0.0,    0.0,    0.0,    0.0,    0.0],
                             [-0.385, -0.946, -1.121, -1.032, -0.898, -0.695, -0.523, -0.434, -0.307, -0.203, -0.157, -0.057,
                               0.021, -0.031, -0.059, -0.001, -0.018, -0.168, -0.381, -0.546, -0.672, -0.819, -0.629,  0.176,
                               0.332,  0.332,  0.364,  0.166,    0.0,    0.0,    0.0,    0.0,    0.0],
                             [-0.383, -0.904, -1.063, -0.972, -0.834, -0.632, -0.464, -0.378, -0.251, -0.144, -0.096,  0.001,
                               0.079,  0.032,  0.011,  0.069,  0.044, -0.113, -0.332, -0.504, -0.637, -0.791, -0.611,  0.181,
                               0.338,  0.335,  0.367,  0.167,    0.0,    0.0,    0.0,    0.0,    0.0],
                             [-0.391, -0.873, -1.016, -0.929, -0.794, -0.591, -0.423, -0.337, -0.212, -0.104, -0.056,  0.043,
                               0.121,  0.077,  0.058,  0.117,  0.088, -0.075, -0.298, -0.475, -0.613, -0.772, -0.599,  0.183,
                               0.342,  0.337,   0.37,  0.168,    0.0,    0.0,    0.0,    0.0,    0.0],
                             [-0.359, -0.836, -0.976, -0.888, -0.755, -0.554, -0.386,   -0.3, -0.175, -0.067, -0.018,  0.081,
                                0.16,  0.119,  0.103,  0.161,  0.129, -0.039, -0.266, -0.448, -0.591, -0.755, -0.587,  0.187,
                               0.345,  0.339,  0.372,  0.169,    0.0,    0.0,    0.0,    0.0,    0.0],
                             [-0.328, -0.792, -0.928, -0.842, -0.709, -0.508, -0.341, -0.256, -0.131, -0.022,  0.029,  0.128,
                               0.208,   0.17,  0.158,  0.216,  0.179,  0.005, -0.228, -0.415, -0.564, -0.733, -0.573,   0.19,
                               0.384,  0.313,  0.375,   0.17,    0.0,    0.0,    0.0,    0.0,    0.0],
                             [-0.324, -0.767, -0.893, -0.807, -0.676, -0.476,  -0.31, -0.225, -0.101,  0.008,   0.06,  0.159,
                               0.239,  0.204,  0.195,  0.252,  0.212,  0.034, -0.203, -0.394, -0.546, -0.719, -0.564,  0.192,
                               0.386,  0.315,  0.377,  0.171,    0.0,    0.0,    0.0,    0.0,    0.0],
                             [ -0.31,  -0.74,  -0.86, -0.775, -0.647, -0.449, -0.283, -0.197, -0.073,  0.036,  0.089,  0.188,
                               0.269,  0.235,  0.229,  0.285,  0.242,  0.061, -0.179, -0.374,  -0.53, -0.706, -0.556,  0.194,
                               0.388,  0.317,  0.402,  0.158,    0.0,    0.0,    0.0,    0.0,    0.0],
                             [-0.244, -0.694, -0.818,  -0.73, -0.605, -0.415, -0.252, -0.163, -0.037,  0.072,  0.122,   0.22,
                               0.303,  0.273,  0.269,  0.324,  0.277,  0.093, -0.152,  -0.35,  -0.51, -0.691, -0.546,  0.196,
                               0.39,   0.32,  0.403,  0.159,    0.0,    0.0,    0.0,    0.0,    0.0],
                             [-0.284, -0.701, -0.803, -0.701, -0.568, -0.381, -0.225, -0.142, -0.017,  0.092,  0.143,  0.242,
                               0.325,  0.298,  0.295,   0.35,    0.3,  0.112, -0.134, -0.334, -0.497,  -0.68,  -0.54,  0.198,
                               0.392,  0.321,  0.404,   0.16,    0.0,    0.0,    0.0,    0.0,    0.0],
                             [-0.281, -0.686, -0.783,  -0.68, -0.547, -0.359, -0.202, -0.119,  0.005,  0.112,  0.163,  0.261,
                               0.345,  0.321,  0.319,  0.371,  0.319,   0.13, -0.118, -0.321, -0.486, -0.671, -0.534,  0.199,
                               0.393,  0.323,  0.405,  0.161,    0.0,    0.0,    0.0,    0.0,    0.0],
                             [-0.269, -0.667,  -0.76, -0.655, -0.522, -0.336, -0.181, -0.096,  0.029,  0.136,  0.188,  0.286,
                                0.37,  0.346,  0.345,  0.396,  0.342,   0.15, -0.102, -0.307, -0.473, -0.661, -0.528,    0.2,
                               0.393,  0.324,  0.405,  0.162,    0.0,    0.0,    0.0,    0.0,    0.0],
                             [-0.255, -0.653, -0.747, -0.643, -0.511, -0.325, -0.169, -0.082,  0.042,  0.149,  0.204,  0.304,
                               0.388,  0.363,  0.36 ,  0.409,  0.354,  0.164, -0.085, -0.289, -0.457, -0.649, -0.523,  0.193,
                               0.394,  0.326,  0.406,  0.162,    0.0,    0.0,    0.0,    0.0,    0.0],
                             [-0.265,  -0.65, -0.739, -0.634,   -0.5, -0.314, -0.159, -0.072,  0.052,  0.159,  0.215,  0.316,
                               0.398,  0.374,  0.374,  0.424,   0.37,  0.181, -0.065, -0.265, -0.429, -0.627, -0.519,   0.18,
                               0.394,  0.326,  0.406,  0.162,    0.0,    0.0,    0.0,    0.0,    0.0],
                             [-0.276, -0.647, -0.731, -0.626, -0.492, -0.307, -0.152, -0.064,  0.058,  0.166,  0.227,  0.329,
                               0.411,  0.389,   0.39,  0.441,  0.389,  0.207, -0.032, -0.228, -0.394, -0.596, -0.494,  0.194,
                               0.376,  0.326,  0.406,  0.162,    0.0,    0.0,    0.0,    0.0,    0.0],
                             [-0.271, -0.646,  -0.73, -0.625, -0.489, -0.303, -0.149, -0.061,  0.062,  0.169,  0.229,  0.332,
                               0.412,  0.388,  0.389,  0.439,  0.387,  0.206, -0.028, -0.209, -0.347, -0.524, -0.435,  0.195,
                               0.381,  0.313,  0.405,  0.162,    0.0,    0.0,    0.0,    0.0,    0.0],
                             [-0.267, -0.647, -0.734, -0.628,  -0.49, -0.304, -0.151, -0.062,  0.061,  0.168,  0.229,  0.329,
                               0.408,  0.385,  0.388,  0.438,  0.386,  0.206, -0.024, -0.194, -0.319,  -0.48,  -0.36,  0.318,
                               0.405,  0.335,  0.394,  0.162,    0.0,    0.0,    0.0,    0.0,    0.0],
                             [-0.274, -0.656, -0.745,  -0.64,   -0.5, -0.313, -0.158, -0.068,  0.054,  0.161,  0.223,  0.325,
                               0.402,  0.379,  0.384,  0.438,  0.392,  0.221,  0.001, -0.164, -0.278, -0.415, -0.264,  0.445,
                               0.402,  0.304,  0.389,  0.157,    0.0,    0.0,    0.0,    0.0,    0.0],
                             [-0.289, -0.666, -0.753, -0.648, -0.508,  -0.32, -0.164, -0.073,  0.049,  0.156,   0.22,  0.321,
                               0.397,  0.374,  0.377,   0.43,  0.387,  0.224,  0.014, -0.139, -0.236, -0.359, -0.211,  0.475,
                                 0.4,  0.308,  0.375,  0.155,    0.0,    0.0,    0.0,    0.0,    0.0],
                             [-0.302, -0.678, -0.765, -0.659, -0.517, -0.329, -0.176, -0.085,  0.038,  0.145,  0.208,   0.31,
                               0.386,  0.362,  0.366,  0.421,  0.381,  0.224,  0.022, -0.119, -0.201,   -0.3, -0.129,  0.572,
                               0.419,  0.265,  0.364,  0.154,    0.0,    0.0,    0.0,    0.0,    0.0],
                             [-0.314, -0.696, -0.786, -0.681, -0.539, -0.349, -0.196, -0.105,  0.019,  0.127,  0.189,  0.289,
                               0.364,   0.34,  0.346,  0.403,   0.37,  0.222,  0.036, -0.081, -0.133, -0.205, -0.021,  0.674,
                               0.383,  0.237,  0.359,  0.151,    0.0,    0.0,    0.0,    0.0,    0.0],
                             [-0.341, -0.719, -0.807, -0.702, -0.558, -0.367, -0.211,  -0.12,  0.003,  0.111,  0.175,  0.277,
                               0.351,  0.325,  0.331,   0.39,   0.36,  0.221,  0.048, -0.046, -0.074, -0.139,  0.038,  0.726,
                               0.429,  0.215,  0.347,  0.151,    0.0,    0.0,    0.0,    0.0,    0.0],
                             [ -0.35, -0.737, -0.829, -0.724, -0.577, -0.385, -0.229, -0.136, -0.011,  0.098,  0.163,  0.266,
                               0.338,   0.31,  0.316,  0.378,  0.354,  0.221,  0.062, -0.009, -0.012, -0.063,  0.119,  0.811,
                               0.319,  0.201,  0.343,  0.148,    0.0,    0.0,    0.0,    0.0,    0.0],
                             [-0.344,  -0.75, -0.856, -0.757, -0.607, -0.409,  -0.25, -0.156, -0.033,  0.076,  0.143,  0.246,
                               0.316,  0.287,  0.293,  0.361,  0.345,  0.225,  0.082,  0.035,  0.071,  0.046,  0.172,  0.708,
                               0.255,   0.21,  0.325,  0.146,    0.0,    0.0,    0.0,    0.0,    0.0]])

        return
    cpdef initialize_profiles(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                     NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        # build a matrix of interpulated radiative forcing
        A = np.interp(Gr.zp_half,self.z_in,self.rad_in[0,:]) # Gr.zp_half,self.rad
        for tt in range(1,36):
            A = np.vstack((A, np.interp(Gr.zp_half,self.z_in,self.rad_in[tt,:])))
        self.rad = A # store matrix in self
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 Surface.SurfaceBase Sur, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa):

        self.rad_cool = np.zeros(Gr.dims.nlg[2], dtype=np.double)
        ind1 = int(math.trunc(TS.t/600.0))                   # the index preceding the current time step
        ind2 = int(math.ceil(TS.t/600.0))                    # the index following the current time step

        if TS.t<600.0: # first 10 min use the radiative forcing of t=10min
            for kk in range(0,Gr.dims.nlg[2]):
                self.rad_cool[kk] = self.rad[0,kk]
        elif TS.t>18900.0:
            for kk in range(0,Gr.dims.nlg[2]):
                self.rad_cool[kk] = (self.rad[31,kk]-self.rad[30,kk])/(self.rad_time[31]-self.rad_time[30])*(18900.0/60.0-self.rad_time[30])+self.rad[30,kk]

        else:
            if TS.t%600.0 == 0:     # in case you step right on the data point
                for kk in range(0,Gr.dims.nlg[2]):
                    self.rad_cool[kk] = self.rad[ind1,kk]
            else: # in all other cases - interpolate
                for kk in range(0,Gr.dims.nlg[2]):
                    if Gr.zp_half[kk] < 22699.48:
                        self.rad_cool[kk]    = (self.rad[ind2,kk]-self.rad[ind1,kk])/(self.rad_time[ind2]-self.rad_time[ind1])*(TS.t/60.0-self.rad_time[ind1])+self.rad[ind1,kk] # yair check the impact of the dt typo
                    else:
                        self.rad_cool[kk] = 0.1
                #self.rad_cool[kk]    = (self.rad[ind2,kk]-self.rad[ind1,kk])/(self.rad_time[ind2]-self.rad_time[ind1])*TS.dt+self.rad[ind1,kk] # yair check the impact of the dt typ
        # get the radiative cooling to the moist entropy equation - here is it in K /day
        cdef:
            Py_ssize_t imin = Gr.dims.gw
            Py_ssize_t jmin = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw

            Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw

            Py_ssize_t pi, i, j, k, ijk, ishift, jshift
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')

         # Now update entropy tendencies
        with nogil:
            for i in xrange(imin, imax):
                ishift = i * istride
                for j in xrange(jmin, jmax):
                    jshift = j * jstride
                    for k in xrange(kmin, kmax):
                        ijk = ishift + jshift + k
                        PV.tendencies[
                            s_shift + ijk] +=  cpm_c(PV.values[ijk + qt_shift])*self.rad_cool[k]/(86400.0)/ DV.values[ijk + t_shift]
        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        RadiationBase.stats_io(self, Gr, RS, DV, NS,  Pa)



        return
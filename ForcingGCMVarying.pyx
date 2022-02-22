#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

import netCDF4 as nc
cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
cimport TimeStepping
from thermodynamic_functions cimport cpm_c, pv_c, pd_c, exner_c
from entropies cimport sv_c, sd_c, s_tendency_c
import numpy as np
import cython
from libc.math cimport fabs, sin, cos, exp
from NetCDFIO cimport NetCDFIO_Stats
cimport ParallelMPI
cimport Lookup
from Thermodynamics cimport LatentHeat, ClausiusClapeyron
import cPickle
from scipy.interpolate import pchip
from cfsites_forcing_reader import cfreader
from cfgrid_forcing_reader import cfreader_grid
#import pylab as plt
include 'parameters.pxi'

cdef class ForcingGCMVarying:
    def __init__(self, namelist, LatentHeat LH, ParallelMPI.ParallelMPI Pa):
        self.file = str(namelist['gcm']['file'])
        try:
            self.griddata = namelist['gcm']['griddata']
        except:
            self.griddata = False
        if self.griddata:
            self.lat = namelist['gcm']['lat']
            self.lon = namelist['gcm']['lon']
        else:
            self.site = namelist['gcm']['site']
        try:
            self.forcing_frequency = namelist['forcing']['frequency']
        except:
            self.forcing_frequency = 6.0	
        try:
            self.relax_scalar = namelist['forcing']['relax_scalar']
        except:
            self.relax_scalar = False
        try:
            self.tau_scalar = namelist['forcing']['tau_scalar']
        except:
            self.tau_scalar = 21600.0
        try:
            self.z_i = namelist['forcing']['z_i']
        except:
            self.z_i = 5000.0
        try:
            self.z_r = namelist['forcing']['z_r']
        except:
            self.z_r = 5000.0
        try:
            self.relax_wind = namelist['forcing']['relax_wind']
        except:
            self.relax_wind = False
        try:
            self.tau_wind = namelist['forcing']['tau_wind']
        except:
            self.tau_wind = 21600.0
        try:
            self.add_horiz_advection = namelist['forcing']['add_horiz_advection']
        except:
            self.add_horiz_advection = False
        try:
            self.add_subsidence = namelist['forcing']['add_subsidence']
        except:
            self.add_subsidence = False
        try:
            self.add_subsidence_wind = namelist['forcing']['add_subsidence_wind']
        except:
            self.add_subsidence_wind = False
        try:
            self.add_coriolis = namelist['forcing']['add_coriolis']
        except:
            self.add_coriolis = False
        try:
            self.add_ls_pgradient = namelist['forcing']['add_ls_pgradient']
        except:
            self.add_ls_pgradient = False
        self.gcm_profiles_initialized = False
        self.t_indx = 0
        return

    @cython.wraparound(True)
    cpdef initialize(self, Grid.Grid Gr,ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        self.qt_tend_nudge = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.t_tend_nudge = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.u_tend_nudge = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.v_tend_nudge = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.qt_tend_adv = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.t_tend_adv = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.s_tend_adv = np.zeros(Gr.dims.npg,dtype=np.double,order='c')
        self.qt_tend_hadv = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.t_tend_hadv = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.qt_tend_fluc = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.t_tend_fluc = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.s_tend_hadv = np.zeros(Gr.dims.npg,dtype=np.double,order='c')
        self.subsidence = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.ug = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.vg = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')

        NS.add_profile('ls_subsidence', Gr, Pa)
        NS.add_profile('dqtdt_nudge', Gr, Pa)
        NS.add_profile('dtdt_nudge', Gr, Pa)
        NS.add_profile('dudt_nudge', Gr, Pa)
        NS.add_profile('dvdt_nudge', Gr, Pa)
        NS.add_profile('dqtdt_adv', Gr, Pa)
        NS.add_profile('dtdt_adv', Gr, Pa)
        NS.add_profile('dsdt_adv', Gr, Pa)
        NS.add_profile('dqtdt_hadv', Gr, Pa)
        NS.add_profile('dtdt_hadv', Gr, Pa)
        NS.add_profile('dsdt_hadv', Gr, Pa)
        NS.add_profile('dqtdt_fluc', Gr, Pa)
        NS.add_profile('dtdt_fluc', Gr, Pa)
        NS.add_profile('dqtdt_sub', Gr, Pa)
        NS.add_profile('dtdt_sub', Gr, Pa)
        NS.add_profile('dudt_sub', Gr, Pa)
        NS.add_profile('dvdt_sub', Gr, Pa)
        NS.add_profile('dudt_cor', Gr, Pa)
        NS.add_profile('dvdt_cor', Gr, Pa)

        return

    #@cython.wraparound(True)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS,
                 ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0] - gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - gw
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t i,j,k,ishift,jshift,ijk
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t ql_shift = DV.get_varshift(Gr,'ql')
            Py_ssize_t qi_shift = DV.get_varshift(Gr,'qi')
            double pd, pv, qt, qv, p0, rho0, t
            double zmax, weight, weight_half
            double u0_new, v0_new
            double [:] qt_mean = Pa.HorizontalMean(Gr, &PV.values[qt_shift])
            double [:] t_mean = Pa.HorizontalMean(Gr, &DV.values[t_shift])
            double [:] u_mean = Pa.HorizontalMean(Gr, &PV.values[u_shift])
            double [:] v_mean = Pa.HorizontalMean(Gr, &PV.values[v_shift])

        if not self.gcm_profiles_initialized or int(TS.t // (3600.0 * self.forcing_frequency)) > self.t_indx:
            self.t_indx = int(TS.t // (3600.0 * self.forcing_frequency))
            self.gcm_profiles_initialized = True
            Pa.root_print(self.t_indx)
            Pa.root_print(self.forcing_frequency)
            Pa.root_print('Updating Time Varying Forcing')

            if self.griddata:
                rdr = cfreader_grid(self.file, self.lat, self.lon)
            else:
                rdr = cfreader(self.file, self.site)
                self.lat = rdr.get_value('lat')
            Pa.root_print(self.lat)
            self.coriolis_param = 2.0 * omega * sin(self.lat * pi / 180.0)
            self.temp = rdr.get_interp_profile_old('ta', Gr.zp_half, instant=True, t_idx=self.t_indx)
            self.sphum = rdr.get_interp_profile_old('hus', Gr.zp_half, instant=True, t_idx=self.t_indx)
            self.ucomp = rdr.get_interp_profile_old('ua', Gr.zp_half, instant=True, t_idx=self.t_indx)
            self.vcomp = rdr.get_interp_profile_old('va', Gr.zp_half, instant=True, t_idx=self.t_indx)
            temp_at_zp = rdr.get_interp_profile('ta', Gr.zp, instant=True, t_idx=self.t_indx)
            sphum_at_zp = rdr.get_interp_profile('hus', Gr.zp, instant=True, t_idx=self.t_indx)
            if self.add_ls_pgradient:
                self.ug = rdr.get_interp_profile_old('u_geos', Gr.zp_half, instant=True, t_idx=self.t_indx)
                self.vg = rdr.get_interp_profile_old('v_geos', Gr.zp_half, instant=True, t_idx=self.t_indx)
            if self.add_horiz_advection:
                self.t_tend_hadv = rdr.get_interp_profile_old('tntha', Gr.zp_half, instant=True, t_idx=self.t_indx)
                self.qt_tend_hadv = rdr.get_interp_profile_old('tnhusha', Gr.zp_half, instant=True, t_idx=self.t_indx)
            if self.add_subsidence or self.add_subsidence_wind:
                self.omega_vv = rdr.get_interp_profile_old('wap', Gr.zp_half, instant=True, t_idx=self.t_indx)
                alpha = rdr.get_interp_profile_old('alpha', Gr.zp_half, instant=True, t_idx=self.t_indx)
                self.subsidence = -np.array(self.omega_vv) * alpha / g
            Pa.root_print('Finished updating time varying forcing')

        #Apply Coriolis Forcing
        coriolis_force(&Gr.dims, &PV.values[u_shift], &PV.values[v_shift], &PV.tendencies[u_shift],
                      &PV.tendencies[v_shift], &self.ug[0], &self.vg[0], self.coriolis_param, Ref.u0, Ref.v0)

        # Apply subsidence
        if self.add_subsidence:
            apply_subsidence(&Gr.dims, &Ref.rho0[0], &Ref.rho0_half[0], &self.subsidence[0], &PV.values[s_shift], &PV.tendencies[s_shift])
            apply_subsidence(&Gr.dims, &Ref.rho0[0], &Ref.rho0_half[0], &self.subsidence[0], &PV.values[qt_shift], &PV.tendencies[qt_shift])
        if self.add_subsidence_wind:
            apply_subsidence(&Gr.dims, &Ref.rho0[0], &Ref.rho0_half[0], &self.subsidence[0], &PV.values[u_shift], &PV.tendencies[u_shift])
            apply_subsidence(&Gr.dims, &Ref.rho0[0], &Ref.rho0_half[0], &self.subsidence[0], &PV.values[v_shift], &PV.tendencies[v_shift])

        # Relaxation
        cdef double [:] xi_relax_scalar = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        cdef double [:] xi_relax_wind = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')

        if self.relax_scalar:
            with nogil:
                for k in xrange(Gr.dims.nlg[2]):
                    if Gr.zpl_half[k] >= self.z_r:
                        xi_relax_scalar[k] = 1.0 / self.tau_scalar
                    elif Gr.zpl_half[k] < self.z_i:
                        xi_relax_scalar[k] = 0.0
                    else:
                        xi_relax_scalar[k] = 0.5 / self.tau_scalar * (1.0 - cos(pi*(Gr.zpl_half[k]-self.z_i)/(self.z_r-self.z_i)))
                    # Nudging rates
                    self.qt_tend_nudge[k] = -xi_relax_scalar[k] * (qt_mean[k] - self.sphum[k])
                    self.t_tend_nudge[k]  = -xi_relax_scalar[k] * (t_mean[k] - self.temp[k])

        if self.relax_wind:
            with nogil:
                for k in xrange(Gr.dims.nlg[2]):
                    xi_relax_wind[k] = 1.0 / self.tau_wind
                    # Nudging rates
                    self.u_tend_nudge[k] = -xi_relax_wind[k] * (u_mean[k] - self.ucomp[k])
                    self.v_tend_nudge[k] = -xi_relax_wind[k] * (v_mean[k] - self.vcomp[k])

        cdef double total_t_source, total_qt_source
        with nogil:
            for i in xrange(gw,imax):
                ishift = i * istride
                for j in xrange(gw,jmax):
                    jshift = j * jstride
                    for k in xrange(gw,kmax):
                        ijk = ishift + jshift + k
                        p0 = Ref.p0_half[k]
                        rho0 = Ref.rho0_half[k]
                        qt = PV.values[qt_shift + ijk]
                        qv = qt - DV.values[ql_shift + ijk] - DV.values[qi_shift + ijk]
                        pd = pd_c(p0,qt,qv)
                        pv = pv_c(p0,qt,qv)
                        t  = DV.values[t_shift + ijk]

                        total_t_source = self.t_tend_hadv[k] + self.t_tend_nudge[k]
                        total_qt_source = self.qt_tend_hadv[k] + self.qt_tend_nudge[k]
                        PV.tendencies[s_shift + ijk] += s_tendency_c(p0, qt, qv, t, total_qt_source, total_t_source)
                        PV.tendencies[qt_shift + ijk] += total_qt_source
                        #PV.tendencies[s_shift + ijk] += (cpm_c(qt) * (self.t_tend_adv[k]+self.t_tend_nudge[k]))/t
                        #PV.tendencies[s_shift + ijk] += (sv_c(pv,t) - sd_c(pd,t)) * (self.qt_tend_adv[k]+self.qt_tend_nudge[k])
                        #PV.tendencies[qt_shift + ijk] += (self.qt_tend_adv[k]+self.qt_tend_nudge[k])
                        PV.tendencies[u_shift + ijk] += self.u_tend_nudge[k]
                        PV.tendencies[v_shift + ijk] += self.v_tend_nudge[k]
                        self.s_tend_hadv[ijk]= s_tendency_c(p0,qt, qv, t, self.qt_tend_hadv[k], self.t_tend_hadv[k])

        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0] - gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - gw
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t i,j,k,ishift,jshift,ijk
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            double [:] tmp_tendency  = np.zeros((Gr.dims.npg),dtype=np.double,order='c')
            double [:] tmp_tendency_2  = np.zeros((Gr.dims.npg),dtype=np.double,order='c')
            double [:] mean_tendency = np.empty((Gr.dims.nlg[2],),dtype=np.double,order='c')
            double [:] mean_tendency_2 = np.empty((Gr.dims.nlg[2],),dtype=np.double,order='c')

        # Output Coriolis tendencies
        if self.add_coriolis:
            coriolis_force(&Gr.dims,&PV.values[u_shift],&PV.values[v_shift],&tmp_tendency[0],
                           &tmp_tendency_2[0],&self.ug[0], &self.vg[0],self.coriolis_param, Ref.u0, Ref.v0)
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        mean_tendency_2 = Pa.HorizontalMean(Gr,&tmp_tendency_2[0])
        NS.write_profile('dudt_cor',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)
        NS.write_profile('dvdt_cor',mean_tendency_2[Gr.dims.gw:-Gr.dims.gw],Pa)

        #Output subsidence tendencies
        tmp_tendency[:] = 0.0
        if self.add_subsidence:
            apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0], &PV.values[qt_shift],
                             &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('dqtdt_sub', mean_tendency[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp_tendency[:] = 0.0
        if self.add_subsidence:
            apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0], &DV.values[t_shift],
                             &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('dtdt_sub', mean_tendency[Gr.dims.gw:-Gr.dims.gw], Pa)
        
        tmp_tendency[:] = 0.0
        if self.add_subsidence_wind:
            apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0], &PV.values[u_shift],
                             &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('dudt_sub', mean_tendency[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp_tendency[:] = 0.0
        if self.add_subsidence_wind:
            apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0], &PV.values[v_shift],
                             &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('dvdt_sub', mean_tendency[Gr.dims.gw:-Gr.dims.gw], Pa)

        NS.write_profile('ls_subsidence', self.subsidence[Gr.dims.gw:-Gr.dims.gw],Pa)
        NS.write_profile('dqtdt_nudge', self.qt_tend_nudge[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_profile('dtdt_nudge', self.t_tend_nudge[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_profile('dudt_nudge', self.u_tend_nudge[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_profile('dvdt_nudge', self.v_tend_nudge[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_profile('dqtdt_adv', self.qt_tend_adv[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_profile('dtdt_adv', self.t_tend_adv[Gr.dims.gw:-Gr.dims.gw], Pa)
        mean_tendency = Pa.HorizontalMean(Gr,&self.s_tend_adv[0])
        NS.write_profile('dsdt_adv',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)
        NS.write_profile('dqtdt_hadv', self.qt_tend_hadv[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_profile('dtdt_hadv', self.t_tend_hadv[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_profile('dqtdt_fluc', self.qt_tend_fluc[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_profile('dtdt_fluc', self.t_tend_fluc[Gr.dims.gw:-Gr.dims.gw], Pa)
        mean_tendency = Pa.HorizontalMean(Gr,&self.s_tend_hadv[0])
        NS.write_profile('dsdt_hadv',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)

        return

from scipy.interpolate import pchip, interp1d
def interp_pchip(z_out, z_in, v_in, pchip_type=True):
    if pchip_type:
        p = pchip(z_in, v_in, extrapolate=True)
        #p = interp1d(z_in, v_in, kind='linear', fill_value='extrapolate')
        return p(z_out)
    else:
        return np.interp(z_out, z_in, v_in)

cdef coriolis_force(Grid.DimStruct *dims, double *u, double *v, double *ut, double *vt, double *ug, double *vg, double coriolis_param, double u0, double v0 ):

    cdef:
        Py_ssize_t imin = dims.gw
        Py_ssize_t jmin = dims.gw
        Py_ssize_t kmin = dims.gw
        Py_ssize_t imax = dims.nlg[0] -dims.gw
        Py_ssize_t jmax = dims.nlg[1] -dims.gw
        Py_ssize_t kmax = dims.nlg[2] -dims.gw
        Py_ssize_t istride = dims.nlg[1] * dims.nlg[2]
        Py_ssize_t jstride = dims.nlg[2]
        Py_ssize_t ishift, jshift, ijk, i,j,k
        double u_at_v, v_at_u

    with nogil:
        for i in xrange(imin,imax):
            ishift = i*istride
            for j in xrange(jmin,jmax):
                jshift = j*jstride
                for k in xrange(kmin,kmax):
                    ijk = ishift + jshift + k
                    u_at_v = 0.25*(u[ijk] + u[ijk-istride] + u[ijk-istride+jstride] + u[ijk +jstride]) + u0
                    v_at_u = 0.25*(v[ijk] + v[ijk+istride] + v[ijk+istride-jstride] + v[ijk-jstride]) + v0
                    ut[ijk] = ut[ijk] - coriolis_param * (vg[k] - v_at_u)
                    vt[ijk] = vt[ijk] + coriolis_param * (ug[k] - u_at_v)
    return



cdef apply_subsidence_temperature(Grid.DimStruct *dims, double *rho0, double *rho0_half, double *subsidence, double *qt, double* values,  double *tendencies):

    cdef:
        Py_ssize_t imin = dims.gw
        Py_ssize_t jmin = dims.gw
        Py_ssize_t kmin = dims.gw
        Py_ssize_t imax = dims.nlg[0] -dims.gw
        Py_ssize_t jmax = dims.nlg[1] -dims.gw
        Py_ssize_t kmax = dims.nlg[2] -dims.gw -1
        Py_ssize_t istride = dims.nlg[1] * dims.nlg[2]
        Py_ssize_t jstride = dims.nlg[2]
        Py_ssize_t ishift, jshift, ijk, i,j,k
        double dxi = dims.dxi[2]
        double tend
    with nogil:
        for i in xrange(imin,imax):
            ishift = i*istride
            for j in xrange(jmin,jmax):
                jshift = j*jstride
                for k in xrange(kmin,kmax):
                    ijk = ishift + jshift + k
                    if(subsidence[k] < 0):
                        tend = cpm_c(qt[ijk])/values[ijk] *(values[ijk+1] - values[ijk]) * dxi * subsidence[k] * dims.imetl[k]
                    else:
                        tend = cpm_c(qt[ijk])/values[ijk] *(values[ijk] - values[ijk-1]) * dxi * subsidence[k] * dims.imetl[k-1]
                    #+ g / values[ijk] * subsidence[k]
                    tendencies[ijk] -= tend
                for k in xrange(kmax, dims.nlg[2]):
                    ijk = ishift + jshift + k
                    tendencies[ijk] -= tend
    return

cdef apply_subsidence(Grid.DimStruct *dims, double *rho0, double *rho0_half, double *subsidence, double* values,  double *tendencies):

    cdef:
        Py_ssize_t imin = dims.gw
        Py_ssize_t jmin = dims.gw
        Py_ssize_t kmin = dims.gw
        Py_ssize_t imax = dims.nlg[0] -dims.gw
        Py_ssize_t jmax = dims.nlg[1] -dims.gw
        Py_ssize_t kmax = dims.nlg[2] -dims.gw -1
        Py_ssize_t istride = dims.nlg[1] * dims.nlg[2]
        size_t jstride = dims.nlg[2]
        Py_ssize_t ishift, jshift, ijk, i,j,k
        double dxi = dims.dxi[2]
        double tend
    with nogil:
        for i in xrange(imin,imax):
            ishift = i*istride
            for j in xrange(jmin,jmax):
                jshift = j*jstride
                for k in xrange(kmin,kmax):
                    ijk = ishift + jshift + k
                    if(subsidence[k] < 0):
                        tend = (values[ijk+1] - values[ijk]) * dxi * subsidence[k] * dims.imetl[k]
                    else:
                        tend = (values[ijk] - values[ijk-1]) * dxi * subsidence[k] * dims.imetl[k-1]
                    tendencies[ijk] -= tend
                for k in xrange(kmax, dims.nlg[2]):
                    ijk = ishift + jshift + k
                    tendencies[ijk] -= tend

    return


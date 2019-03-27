#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
cimport ParallelMPI
cimport TimeStepping
cimport Radiation
from Thermodynamics cimport LatentHeat,ClausiusClapeyron
from SurfaceBudget cimport SurfaceBudget
from NetCDFIO cimport NetCDFIO_Stats
import cython
from thermodynamic_functions import exner, cpm
from thermodynamic_functions cimport cpm_c, pv_c, pd_c, exner_c, qv_star_c
from entropies cimport sv_c, sd_c
from libc.math cimport sqrt, log, fabs,atan, exp, fmax
cimport numpy as np
import numpy as np
include "parameters.pxi"
import cython


cdef extern from "advection_interpolation.h":
    double interp_2(double phi, double phip1) nogil
cdef extern from "thermodynamic_functions.h":
    inline double pd_c(double p0, double qt, double qv) nogil
    inline double pv_c(double p0, double qt, double qv) nogil
    inline double exner_c(const double p0) nogil
    inline double theta_rho_c(double p0, double T,double qt, double qv) nogil
    inline double cpm_c(double qt) nogil
cdef extern from "surface.h":
    double compute_ustar(double windspeed, double buoyancy_flux, double z0, double z1) nogil
    inline double entropyflux_from_thetaflux_qtflux(double thetaflux, double qtflux, double p0_b, double T_b, double qt_b, double qv_b) nogil
    void compute_windspeed(Grid.DimStruct *dims, double* u, double*  v, double*  speed, double u0, double v0, double gustiness ) nogil
    void exchange_coefficients_byun(double Ri, double zb, double z0, double* cm, double* ch, double* lmo) nogil
cdef extern from "entropies.h":
    inline double sd_c(double pd, double T) nogil
    inline double sv_c(double pv, double T) nogil

def SurfaceFactory(namelist, LatentHeat LH, ParallelMPI.ParallelMPI Par):

        casename = namelist['meta']['casename']
        if casename == 'SullivanPatton':
           return SurfaceSullivanPatton(LH)
        elif casename == 'Bomex':
            return SurfaceBomex(LH)
        elif casename == 'Soares':
            return SurfaceSoares(LH)
        elif casename == 'Soares_moist':
            return SurfaceSoares_moist(LH)
        elif casename == 'Gabls':
            return SurfaceGabls(namelist,LH)
        elif casename == 'DYCOMS_RF01':
            return SurfaceDYCOMS_RF01(namelist, LH)
        elif casename == 'DYCOMS_RF02':
            return SurfaceDYCOMS_RF02(namelist, LH)
        elif casename == 'Rico':
            return SurfaceRico(LH)
        elif casename == 'CGILS':
            return SurfaceCGILS(namelist, LH, Par)
        elif casename == 'ZGILS':
            return SurfaceZGILS(namelist, LH, Par)
        elif casename == 'TRMM_LBA':
            return SurfaceTRMM_LBA(namelist, LH, Par)
        elif casename == 'ARM_SGP':
            return SurfaceARM_SGP(namelist, LH, Par)
        elif casename == 'SMCS':
            return SurfaceSCMS2(namelist, LH, Par)
        elif casename == 'GATE_III':
            return SurfaceGATE_III(namelist, LH, Par)
        else:
            return SurfaceNone()



cdef class SurfaceBase:
    def __init__(self):
        return

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        self.u_flux = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
        self.v_flux = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
        self.qt_flux = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
        self.s_flux = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')

        self.obukhov_length = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
        self.friction_velocity = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
        self.shf = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
        self.lhf = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
        self.b_flux = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')

        # If not overridden in the specific case, set T_surface = Tg
        self.T_surface = Ref.Tg


        NS.add_ts('uw_surface_mean', Gr, Pa)
        NS.add_ts('vw_surface_mean', Gr, Pa)
        NS.add_ts('s_flux_surface_mean', Gr, Pa)
        NS.add_ts('shf_surface_mean', Gr, Pa)
        NS.add_ts('lhf_surface_mean', Gr, Pa)
        NS.add_ts('obukhov_length_mean', Gr, Pa)
        NS.add_ts('friction_velocity_mean', Gr, Pa)
        NS.add_ts('buoyancy_flux_surface_mean', Gr, Pa)

        return
    cpdef init_from_restart(self, Restart):
        self.T_surface = Restart.restart_data['surf']['T_surf']
        return
    cpdef restart(self, Restart):
        Restart.restart_data['surf'] = {}
        Restart.restart_data['surf']['T_surf'] = self.T_surface
        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV,  ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):

        cdef :
            Py_ssize_t i, j, ijk, ij
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0]-gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - gw
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t istride_2d = Gr.dims.nlg[1]
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t ql_shift, qt_shift
            double [:] t_mean =  Pa.HorizontalMean(Gr, &DV.values[t_shift])
            double cp_, lam, lv, pv, pd, sv, sd
            double dzi = 1.0/Gr.dims.zp_0
            double tendency_factor = Ref.alpha0_half[gw]/Ref.alpha0[gw-1]*dzi


        if self.dry_case:
            with nogil:
                for i in xrange(gw, imax):
                    for j in xrange(gw, jmax):
                        ijk = i * istride + j * jstride + gw
                        ij = i * istride_2d + j
                        self.shf[ij] = self.s_flux[ij] * Ref.rho0_half[gw] * DV.values[t_shift+ijk]
                        self.b_flux[ij] = self.shf[ij] * g * Ref.alpha0_half[gw]/cpd/t_mean[gw]
                        self.obukhov_length[ij] = -self.friction_velocity[ij] *self.friction_velocity[ij] *self.friction_velocity[ij] /self.b_flux[ij]/vkb

                        PV.tendencies[u_shift  + ijk] +=  self.u_flux[ij] * tendency_factor
                        PV.tendencies[v_shift  + ijk] +=  self.v_flux[ij] * tendency_factor
                        PV.tendencies[s_shift  + ijk] +=  self.s_flux[ij] * tendency_factor

        else:
            ql_shift = DV.get_varshift(Gr,'ql')
            qt_shift = PV.get_varshift(Gr, 'qt')
            with nogil:
                for i in xrange(gw, imax):
                    for j in xrange(gw, jmax):
                        ijk = i * istride + j * jstride + gw
                        ij = i * istride_2d + j
                        lam = self.Lambda_fp(DV.values[t_shift+ijk])
                        lv = self.L_fp(DV.values[t_shift+ijk],lam)
                        self.lhf[ij] = self.qt_flux[ij] * Ref.rho0_half[gw] * lv
                        pv = pv_c(Ref.p0_half[gw], PV.values[ijk + qt_shift], PV.values[ijk + qt_shift] - DV.values[ijk + ql_shift])
                        pd = pd_c(Ref.p0_half[gw], PV.values[ijk + qt_shift], PV.values[ijk + qt_shift] - DV.values[ijk + ql_shift])
                        sv = sv_c(pv,DV.values[t_shift+ijk])
                        sd = sd_c(pd,DV.values[t_shift+ijk])
                        self.shf[ij] = (self.s_flux[ij] * Ref.rho0_half[gw] - self.lhf[ij]/lv * (sv-sd)) * DV.values[t_shift+ijk]
                        cp_ = cpm_c(PV.values[qt_shift+ijk])
                        self.b_flux[ij] = g * Ref.alpha0_half[gw]/cp_/t_mean[gw] * \
                                          (self.shf[ij] + (eps_vi-1.0)*cp_*t_mean[gw]*self.lhf[ij]/lv)
                        self.obukhov_length[ij] = -self.friction_velocity[ij] *self.friction_velocity[ij] *self.friction_velocity[ij] /self.b_flux[ij]/vkb


                        PV.tendencies[u_shift  + ijk] +=  self.u_flux[ij] * tendency_factor
                        PV.tendencies[v_shift  + ijk] +=  self.v_flux[ij] * tendency_factor
                        PV.tendencies[s_shift  + ijk] +=  self.s_flux[ij] * tendency_factor
                        PV.tendencies[qt_shift + ijk] +=  self.qt_flux[ij] * tendency_factor

        return

    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        cdef double tmp

        tmp = Pa.HorizontalMeanSurface(Gr, &self.u_flux[0])
        NS.write_ts('uw_surface_mean', tmp, Pa)
        tmp = Pa.HorizontalMeanSurface(Gr, &self.v_flux[0])
        NS.write_ts('vw_surface_mean', tmp, Pa)
        tmp = Pa.HorizontalMeanSurface(Gr, &self.s_flux[0])
        NS.write_ts('s_flux_surface_mean', tmp, Pa)
        tmp = Pa.HorizontalMeanSurface(Gr,&self.b_flux[0])
        NS.write_ts('buoyancy_flux_surface_mean', tmp, Pa)
        tmp = Pa.HorizontalMeanSurface(Gr,&self.shf[0])
        NS.write_ts('shf_surface_mean', tmp, Pa)
        tmp = Pa.HorizontalMeanSurface(Gr,&self.lhf[0])
        NS.write_ts('lhf_surface_mean', tmp, Pa)

        tmp = Pa.HorizontalMeanSurface(Gr,&self.friction_velocity[0])
        NS.write_ts('friction_velocity_mean', tmp, Pa)
        tmp = Pa.HorizontalMeanSurface(Gr,&self.obukhov_length[0])
        NS.write_ts('obukhov_length_mean', tmp, Pa)
        return


cdef class SurfaceNone(SurfaceBase):
    def __init__(self):
        pass

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):
        return
    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        return


cdef class SurfaceSullivanPatton(SurfaceBase):
    def __init__(self, LatentHeat LH):
        self.theta_flux = 0.24 # K m/s
        self.z0 = 0.1 #m (Roughness length)
        self.gustiness = 0.001 #m/s, minimum surface windspeed for determination of u*

        self.L_fp = LH.L_fp
        self.Lambda_fp = LH.Lambda_fp

        self.dry_case = True
        return

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        SurfaceBase.initialize(self,Gr,Ref,NS,Pa)

        return



    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):

        # Since this case is completely dry, the computation of entropy flux from sensible heat flux is very simple

        if Pa.sub_z_rank != 0:
            return

        cdef:
            Py_ssize_t i, j, ijk, ij
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0]
            Py_ssize_t jmax = Gr.dims.nlg[1]
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t istride_2d = Gr.dims.nlg[1]
            Py_ssize_t temp_shift = DV.get_varshift(Gr, 'temperature')
            double T0 = Ref.p0_half[Gr.dims.gw] * Ref.alpha0_half[Gr.dims.gw]/Rd
            
        self.buoyancy_flux = self.theta_flux * exner(Ref.p0_half[Gr.dims.gw]) * g /T0

        #Get the scalar flux (dry entropy only)
        with nogil:
            for i in xrange(imax):
                for j in xrange(jmax):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.s_flux[ij] = cpd * self.theta_flux*exner_c(Ref.p0_half[gw])/DV.values[temp_shift+ijk]

        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            double [:] windspeed = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1],dtype=np.double,order='c')

        compute_windspeed(&Gr.dims, &PV.values[u_shift], &PV.values[v_shift], &windspeed[0],Ref.u0, Ref.v0,self.gustiness)


        # Get the shear stresses
        with nogil:
            for i in xrange(1,imax):
                for j in xrange(1,jmax):
                    ij = i * istride_2d + j
                    self.friction_velocity[ij] = compute_ustar(windspeed[ij],self.buoyancy_flux,self.z0, Gr.dims.dx[2]/2.0)
            for i in xrange(1,imax-1):
                for j in xrange(1,jmax-1):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.u_flux[ij] = -interp_2(self.friction_velocity[ij], self.friction_velocity[ij+istride_2d])**2/interp_2(windspeed[ij], windspeed[ij+istride_2d]) \
                                      * (PV.values[u_shift + ijk] + Ref.u0)
                    self.v_flux[ij] = -interp_2(self.friction_velocity[ij], self.friction_velocity[ij+1])**2/interp_2(windspeed[ij], windspeed[ij+1]) \
                                      * (PV.values[v_shift + ijk] + Ref.v0)


        SurfaceBase.update(self, Gr, Ref, PV, DV, Pa, TS)
        return

    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.stats_io(self, Gr, NS, Pa)
        return



cdef class SurfaceBomex(SurfaceBase):
    def __init__(self,  LatentHeat LH):
        self.L_fp = LH.L_fp
        self.Lambda_fp = LH.Lambda_fp
        self.dry_case = False
        return

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.initialize(self,Gr,Ref,NS,Pa)
        self.qt_flux = np.add(self.qt_flux,5.2e-5) # m/s

        self.theta_flux = 8.0e-3 # K m/s
        self.ustar_ = 0.28 #m/s
        self.theta_surface = 299.1 #K
        self.qt_surface = 22.45e-3 # kg/kg
        self.buoyancy_flux = g * ((self.theta_flux + (eps_vi-1.0)*(self.theta_surface*self.qt_flux[0]
                                                                   + self.qt_surface *self.theta_flux))
                              /(self.theta_surface*(1.0 + (eps_vi-1)*self.qt_surface)))

        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV,ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):

        if Pa.sub_z_rank != 0:
            return


        cdef :
            Py_ssize_t i
            Py_ssize_t j
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t ijk, ij
            Py_ssize_t imax = Gr.dims.nlg[0]
            Py_ssize_t jmax = Gr.dims.nlg[1]
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t istride_2d = Gr.dims.nlg[1]
            Py_ssize_t temp_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t qv_shift = DV.get_varshift(Gr,'qv')


        # Get the scalar flux
        with nogil:
            for i in xrange(imax):
                for j in xrange(jmax):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.friction_velocity[ij] = self.ustar_
                    self.s_flux[ij] = entropyflux_from_thetaflux_qtflux(self.theta_flux, self.qt_flux[ij], Ref.p0_half[gw],
                                                                        DV.values[temp_shift+ijk], PV.values[qt_shift+ijk], DV.values[qv_shift+ijk])

        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            double [:] windspeed = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')

        compute_windspeed(&Gr.dims, &PV.values[u_shift], &PV.values[v_shift], &windspeed[0], Ref.u0, Ref.v0, self.gustiness)

        # Get the shear stresses
        with nogil:
            for i in xrange(1,imax-1):
                for j in xrange(1,jmax-1):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.u_flux[ij] = -self.ustar_**2/interp_2(windspeed[ij], windspeed[ij+istride_2d]) * (PV.values[u_shift + ijk] + Ref.u0)
                    self.v_flux[ij] = -self.ustar_**2/interp_2(windspeed[ij], windspeed[ij+1]) * (PV.values[v_shift + ijk] + Ref.v0)

        SurfaceBase.update(self, Gr, Ref, PV, DV, Pa, TS)

        return


    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.stats_io(self, Gr, NS, Pa)


        return


cdef class SurfaceSoares(SurfaceBase):
    def __init__(self, LatentHeat LH):
        self.z0 = 0.001 #m (Roughness length)
        self.gustiness = 0.001 #m/s, minimum surface windspeed for determination of u*

        self.L_fp = LH.L_fp
        self.Lambda_fp = LH.Lambda_fp
        self.dry_case = True
        return

    @cython.boundscheck(False)  #Turn off numpy array index bounds checking
    @cython.wraparound(False)   #Turn off numpy array wrap around indexing
    @cython.cdivision(True)
    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.initialize(self,Gr,Ref,NS,Pa)
        self.theta_surface = 300.0 # K
        self.theta_flux = 0.06 # K m/s
        #T0 = Ref.p0_half[gw] * Ref.alpha0_half[gw]/Rd
        # yair - I chenged self.buoyancy_flux = self.theta_flux * exner(Ref.p0_half[Gr.dims.gw]) * g /T0 using the theta flux and theta surface as Ref had no values
        self.buoyancy_flux = self.theta_flux * g /self.theta_surface

        return

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
# # update adopted and modified from Sullivan + Bomex
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):
        # Since this case is completely dry, the computation of entropy flux from sensible heat flux is very simple

        if Pa.sub_z_rank != 0:
            return

        cdef:
            Py_ssize_t i, j, ijk, ij
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0]
            Py_ssize_t jmax = Gr.dims.nlg[1]
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t istride_2d = Gr.dims.nlg[1]
            Py_ssize_t temp_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            # Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            # Py_ssize_t qv_shift = DV.get_varshift(Gr,'qv')

        # Scalar fluxes (adopted from Bomex)
        with nogil:
            for i in xrange(imax):
                for j in xrange(jmax):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    # Sullivan
                    self.s_flux[ij] = cpd * self.theta_flux*exner_c(Ref.p0_half[gw])/DV.values[temp_shift+ijk]
                    # Bomex (entropy flux includes qt flux)
                    # self.s_flux[ij] = entropyflux_from_thetaflux_qtflux(self.theta_flux, self.qt_flux[ij], Ref.p0_half[gw], DV.values[temp_shift+ijk], PV.values[qt_shift+ijk], DV.values[qv_shift+ijk])

        # Windspeed (adopted from Sullivan, equivalent to Bomex)
        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            double [:] windspeed = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1],dtype=np.double,order='c')
        compute_windspeed(&Gr.dims, &PV.values[u_shift], &PV.values[v_shift], &windspeed[0],Ref.u0, Ref.v0,self.gustiness)

       # Surface Values: friction velocity, obukhov lenght (adopted from Sullivan, since same Surface parameters prescribed)
        with nogil:
            for i in xrange(1,imax):
                for j in xrange(1,jmax):
                    ij = i * istride_2d + j
                    self.friction_velocity[ij] = compute_ustar(windspeed[ij],self.buoyancy_flux,self.z0, Gr.dims.dx[2]/2.0)

        # Get the shear stresses (adopted from Sullivan, since same Surface parameters prescribed)
            for i in xrange(1,imax-1):
                for j in xrange(1,jmax-1):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.u_flux[ij] = -interp_2(self.friction_velocity[ij], self.friction_velocity[ij+istride_2d])**2/interp_2(windspeed[ij], windspeed[ij+istride_2d]) * (PV.values[u_shift + ijk] + Ref.u0)
                    self.v_flux[ij] = -interp_2(self.friction_velocity[ij], self.friction_velocity[ij+1])**2/interp_2(windspeed[ij], windspeed[ij+1]) * (PV.values[v_shift + ijk] + Ref.v0)


        SurfaceBase.update(self, Gr, Ref, PV, DV, Pa, TS)
        return


    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.stats_io(self, Gr, NS, Pa)
        return



cdef class SurfaceSoares_moist(SurfaceBase):
    def __init__(self, LatentHeat LH):
        self.z0 = 0.001 #m (Roughness length)
        self.gustiness = 0.001 #m/s, minimum surface windspeed for determination of u*

        self.L_fp = LH.L_fp
        self.Lambda_fp = LH.Lambda_fp
        self.dry_case = False
        return

    @cython.boundscheck(False)  #Turn off numpy array index bounds checking
    @cython.wraparound(False)   #Turn off numpy array wrap around indexing
    @cython.cdivision(True)
    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):     # Sullivan
        SurfaceBase.initialize(self,Gr,Ref,NS,Pa)

        # ### Bomex
        # self.qt_flux = np.add(self.qt_flux,5.2e-5) # m/s
        # self.theta_flux = 8.0e-3 # K m/s
        # # self.ustar_ = 0.28 #m/s
        # self.theta_surface = 299.1 #K
        # self.qt_surface = 22.45e-3 # kg/kg

        ### Soares_moist
        # self.qt_flux = 5.2e-5 # m/s (Soares: 2.5e-5) (Bomex: 5.2e-5)
        self.qt_flux = np.add(self.qt_flux,2.5e-5)
        # self.qt_flux = np.add(self.qt_flux,0.0)
        self.theta_flux = 8.0e-3 # K m/s (Bomex)
        self.theta_surface = 300.0 # K
        self.qt_surface = 5.0e-3 # kg/kg
        #
        # # Bomex:
        self.buoyancy_flux = g * ((self.theta_flux + (eps_vi-1.0)*(self.theta_surface*self.qt_flux[0]
                                                                   + self.qt_surface *self.theta_flux))
                              /(self.theta_surface*(1.0 + (eps_vi-1)*self.qt_surface)))
        # # Sullivan:
        # # T0 = Ref.p0_half[Gr.dims.gw] * Ref.alpha0_half[Gr.dims.gw]/Rd
        # # self.buoyancy_flux = self.theta_flux * exner(Ref.p0_half[Gr.dims.gw]) * g /T0

        return

    @cython.boundscheck(False)  #Turn off numpy array index bounds checking
    @cython.wraparound(False)   #Turn off numpy array wrap around indexing
    @cython.cdivision(True)
# # update adopted and modified from Sullivan + Bomex
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):
        # Since this case is completely dry, the computation of entropy flux from sensible heat flux is very simple

        if Pa.sub_z_rank != 0:
            return

        cdef:
            Py_ssize_t i, j, ij, ijk
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0]
            Py_ssize_t jmax = Gr.dims.nlg[1]
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t istride_2d = Gr.dims.nlg[1]
            Py_ssize_t temp_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t qv_shift = DV.get_varshift(Gr,'qv')

        # Scalar fluxes (adopted from Bomex)
        with nogil:
            for i in xrange(imax):
                for j in xrange(jmax):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    # Sullivan
                    # self.s_flux[ij] = cpd * self.theta_flux*exner_c(Ref.p0_half[gw])/DV.values[temp_shift+ijk]
                    # Bomex (entropy flux includes qt flux)
                    self.s_flux[ij] = entropyflux_from_thetaflux_qtflux(self.theta_flux, self.qt_flux[ij], Ref.p0_half[gw],
                                                                        DV.values[temp_shift+ijk], PV.values[qt_shift+ijk], DV.values[qv_shift+ijk])

        # Windspeed (adopted from Sullivan, equivalent to Bomex)
        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            double [:] windspeed = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1],dtype=np.double,order='c')
        compute_windspeed(&Gr.dims, &PV.values[u_shift], &PV.values[v_shift], &windspeed[0],Ref.u0, Ref.v0,self.gustiness)

       # Surface Values: friction velocity, obukhov lenght (adopted from Sullivan, since same Surface parameters prescribed)
       #  cdef :
            # Py_ssize_t lmo_shift = DV.get_varshift_2d(Gr, 'obukhov_length')
            # Py_ssize_t ustar_shift = DV.get_varshift_2d(Gr, 'friction_velocity')
        with nogil:
            for i in xrange(1,imax):
                for j in xrange(1,jmax):
                    ij = i * istride_2d + j
                    self.friction_velocity[ij] = compute_ustar(windspeed[ij],self.buoyancy_flux,self.z0, Gr.dims.dx[2]/2.0)
                    # self.obukhov_length[ij] = -self.friction_velocity[ij]*self.friction_velocity[ij]*self.friction_velocity[ij]/self.buoyancy_flux/vkb

        # Get the shear stresses (adopted from Sullivan, since same Surface parameters prescribed)
            for i in xrange(1,imax-1):
                for j in xrange(1,jmax-1):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.u_flux[ij] = -interp_2(self.friction_velocity[ij], self.friction_velocity[ij+istride_2d])**2/interp_2(windspeed[ij], windspeed[ij+istride_2d]) * (PV.values[u_shift + ijk] + Ref.u0)
                    self.v_flux[ij] = -interp_2(self.friction_velocity[ij], self.friction_velocity[ij+1])**2/interp_2(windspeed[ij], windspeed[ij+1]) * (PV.values[v_shift + ijk] + Ref.v0)
                    # PV.tendencies[u_shift + ijk] += self.u_flux[ij] * tendency_factor
                    # PV.tendencies[v_shift + ijk] += self.v_flux[ij] * tendency_factor

        SurfaceBase.update(self, Gr, Ref, PV, DV, Pa, TS)
        return

    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.stats_io(self, Gr, NS, Pa)
        return

cdef class SurfaceGabls(SurfaceBase):
    def __init__(self, namelist,  LatentHeat LH):
        self.gustiness = 0.001
        self.z0 = 0.1
        # Rate of change of surface temperature, in K/hour
        # GABLS1 IC (Beare et al) value is 0.25 (given as default)
        try:
            self.cooling_rate = namelist['surface']['cooling_rate']
        except:
            self.cooling_rate = 0.25

        self.L_fp = LH.L_fp
        self.Lambda_fp = LH.Lambda_fp

        self.dry_case = True


        return

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        SurfaceBase.initialize(self,Gr,Ref,NS,Pa)


        return


    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV,ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):

        if Pa.sub_z_rank != 0:
            return

        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t th_shift = DV.get_varshift(Gr, 'theta')
            double [:] windspeed = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')

        compute_windspeed(&Gr.dims, &PV.values[u_shift], &PV.values[v_shift], &windspeed[0], Ref.u0, Ref.v0, self.gustiness)

        cdef:
            Py_ssize_t i,j, ijk, ij
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0]
            Py_ssize_t jmax = Gr.dims.nlg[1]
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t istride_2d = Gr.dims.nlg[1]

            double theta_rho_b, Nb2, Ri
            double zb = Gr.dims.zp_half_0
            double [:] cm= np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
            double ch=0.0


        self.T_surface = 265.0 - self.cooling_rate * TS.t/3600.0 # sst = theta_surface also


        cdef double theta_rho_g = theta_rho_c(Ref.Pg, self.T_surface, 0.0, 0.0)
        cdef double s_star = sd_c(Ref.Pg,self.T_surface)


        with nogil:
            for i in xrange(gw-1, imax-gw+1):
                for j in xrange(gw-1,jmax-gw+1):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    theta_rho_b = DV.values[th_shift + ijk]
                    Nb2 = g/theta_rho_g*(theta_rho_b-theta_rho_g)/zb
                    Ri = Nb2 * zb* zb/(windspeed[ij] * windspeed[ij])
                    exchange_coefficients_byun(Ri,zb,self.z0, &cm[ij], &ch, &self.obukhov_length[ij])
                    self.s_flux[ij] = -ch * windspeed[ij] * (PV.values[s_shift+ijk] - s_star)
                    self.friction_velocity[ij] = sqrt(cm[ij]) * windspeed[ij]
            for i in xrange(gw, imax-gw):
                for j in xrange(gw, jmax-gw):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.u_flux[ij] = -interp_2(cm[ij], cm[ij+istride_2d])*interp_2(windspeed[ij], windspeed[ij+istride_2d]) * (PV.values[u_shift + ijk] + Ref.u0)
                    self.v_flux[ij] = -interp_2(cm[ij], cm[ij+1])*interp_2(windspeed[ij], windspeed[ij+1]) * (PV.values[v_shift + ijk] + Ref.v0)

        SurfaceBase.update(self, Gr, Ref, PV, DV, Pa, TS)

        return


    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.stats_io(self, Gr, NS, Pa)

        return


cdef class SurfaceDYCOMS_RF01(SurfaceBase):
    def __init__(self,namelist, LatentHeat LH):
        self.ft = 15.0
        self.fq = 115.0
        self.gustiness = 0.0
        self.cm = 0.0011
        self.L_fp = LH.L_fp
        self.Lambda_fp = LH.Lambda_fp
        sst = 292.5 # K
        psurface = 1017.8e2 # Pa
        theta_surface = sst/exner(psurface)
        qt_surface = 13.84e-3 # qs(sst) using Teten's formula
        density_surface = 1.22 #kg/m^3
        theta_flux = self.ft/(density_surface*cpm(qt_surface)*exner(psurface))
        qt_flux_ = self.fq/self.L_fp(sst,self.Lambda_fp(sst))
        self.buoyancy_flux = g * ((theta_flux + (eps_vi-1.0)*(theta_surface*qt_flux_ + qt_surface * theta_flux))
                              /(theta_surface*(1.0 + (eps_vi-1)*qt_surface)))

        self.dry_case = False

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.initialize(self,Gr,Ref,NS,Pa)
        self.windspeed = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
        self.T_surface = 292.5

        return


    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV,ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):

        if Pa.sub_z_rank != 0:
            return

        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t ql_shift = DV.get_varshift(Gr, 'ql')



        compute_windspeed(&Gr.dims, &PV.values[u_shift], &PV.values[v_shift], &self.windspeed[0],Ref.u0, Ref.v0,self.gustiness)

        cdef:
            Py_ssize_t i,j, ijk, ij
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0]
            Py_ssize_t jmax = Gr.dims.nlg[1]
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t istride_2d = Gr.dims.nlg[1]

            double lam, lv, pv, pd, sv, sd

            double [:] windspeed = self.windspeed


        with nogil:
            for i in xrange(gw-1, imax-gw+1):
                for j in xrange(gw-1, jmax-gw+1):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.friction_velocity[ij] = sqrt(self.cm) * self.windspeed[ij]
                    lam = self.Lambda_fp(DV.values[t_shift+ijk])
                    lv = self.L_fp(DV.values[t_shift+ijk],lam)
                    pv = pv_c(Ref.p0_half[gw], PV.values[ijk + qt_shift], PV.values[ijk + qt_shift] - DV.values[ijk + ql_shift])
                    pd = pd_c(Ref.p0_half[gw], PV.values[ijk + qt_shift], PV.values[ijk + qt_shift] - DV.values[ijk + ql_shift])
                    sv = sv_c(pv,DV.values[t_shift+ijk])
                    sd = sd_c(pd,DV.values[t_shift+ijk])
                    self.qt_flux[ij] = self.fq / lv / 1.22
                    self.s_flux[ij] = Ref.alpha0_half[gw] * (self.ft/DV.values[t_shift+ijk] + self.fq*(sv - sd)/lv)
            for i in xrange(gw, imax-gw):
                for j in xrange(gw, jmax-gw):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.u_flux[ij] = -self.cm * interp_2(windspeed[ij], windspeed[ij+istride_2d]) * (PV.values[u_shift + ijk] + Ref.u0)
                    self.v_flux[ij] = -self.cm * interp_2(windspeed[ij], windspeed[ij+1]) * (PV.values[v_shift + ijk] + Ref.v0)

        SurfaceBase.update(self, Gr, Ref, PV, DV, Pa,TS)
        return

    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.stats_io(self, Gr, NS, Pa)

        return


cdef class SurfaceDYCOMS_RF02(SurfaceBase):
    def __init__(self,namelist, LatentHeat LH):
        self.ft = 16.0
        self.fq = 93.0
        self.gustiness = 0.0
        self.ustar = 0.25
        self.L_fp = LH.L_fp
        self.Lambda_fp = LH.Lambda_fp
        sst = 292.5 # K
        psurface = 1017.8e2 # Pa
        theta_surface = sst/exner(psurface)
        qt_surface = 13.84e-3 # qs(sst) using Teten's formula
        density_surface = 1.22 #kg/m^3
        theta_flux = self.ft/(density_surface*cpm(qt_surface)*exner(psurface))
        qt_flux_ = self.fq/self.L_fp(sst,self.Lambda_fp(sst))
        self.buoyancy_flux = g * ((theta_flux + (eps_vi-1.0)*(theta_surface*qt_flux_ + qt_surface * theta_flux))
                              /(theta_surface*(1.0 + (eps_vi-1)*qt_surface)))

        self.dry_case = False




    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.initialize(self,Gr,Ref,NS,Pa)
        self.windspeed = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
        self.T_surface = 292.5 # assuming same sst as DYCOMS RF01


        return


    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV,ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):

        if Pa.sub_z_rank != 0:
            return

        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t ql_shift = DV.get_varshift(Gr, 'ql')



        compute_windspeed(&Gr.dims, &PV.values[u_shift], &PV.values[v_shift], &self.windspeed[0],Ref.u0, Ref.v0,self.gustiness)

        cdef:
            Py_ssize_t i,j, ijk, ij
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0]
            Py_ssize_t jmax = Gr.dims.nlg[1]
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t istride_2d = Gr.dims.nlg[1]

            double tendency_factor = Ref.alpha0_half[gw]/Ref.alpha0[gw-1]/Gr.dims.dx[2]
            double lam
            double lv
            double pv
            double pd
            double sv
            double sd

            double [:] windspeed = self.windspeed

        with nogil:
            for i in xrange(gw-1, imax-gw+1):
                for j in xrange(gw-1, jmax-gw+1):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.friction_velocity[ij] = self.ustar
                    lam = self.Lambda_fp(DV.values[t_shift+ijk])
                    lv = self.L_fp(DV.values[t_shift+ijk],lam)
                    pv = pv_c(Ref.p0_half[gw], PV.values[ijk + qt_shift], PV.values[ijk + qt_shift] - DV.values[ijk + ql_shift])
                    pd = pd_c(Ref.p0_half[gw], PV.values[ijk + qt_shift], PV.values[ijk + qt_shift] - DV.values[ijk + ql_shift])
                    sv = sv_c(pv,DV.values[t_shift+ijk])
                    sd = sd_c(pd,DV.values[t_shift+ijk])
                    self.qt_flux[ij] = self.fq / lv / 1.21
                    self.s_flux[ij] = Ref.alpha0_half[gw] * (self.ft/DV.values[t_shift+ijk] + self.fq*(sv - sd)/lv)
            for i in xrange(gw, imax-gw):
                for j in xrange(gw, jmax-gw):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.u_flux[ij] = -self.ustar*self.ustar / interp_2(windspeed[ij], windspeed[ij+istride_2d]) * (PV.values[u_shift + ijk] + Ref.u0)
                    self.v_flux[ij] = -self.ustar*self.ustar / interp_2(windspeed[ij], windspeed[ij+1]) * (PV.values[v_shift + ijk] + Ref.v0)

        SurfaceBase.update(self, Gr, Ref, PV, DV, Pa, TS)


        return

    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.stats_io(self, Gr, NS, Pa)

        return




cdef class SurfaceRico(SurfaceBase):
    def __init__(self, LatentHeat LH):
        self.cm =0.001229
        self.ch = 0.001094
        self.cq = 0.001133
        self.z0 = 0.00015
        self.gustiness = 0.0
        self.L_fp = LH.L_fp
        self.Lambda_fp = LH.Lambda_fp
        self.dry_case = False
        return


    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.initialize(self,Gr,Ref,NS,Pa)

        self.cm = self.cm*(log(20.0/self.z0)/log(Gr.zpl_half[Gr.dims.gw]/self.z0))**2
        self.ch = self.ch*(log(20.0/self.z0)/log(Gr.zpl_half[Gr.dims.gw]/self.z0))**2
        self.cq = self.cq*(log(20.0/self.z0)/log(Gr.zpl_half[Gr.dims.gw]/self.z0))**2

        return

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV,ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):

        cdef double pv_star = pv_c(Ref.Pg, Ref.qtg, Ref.qtg)
        cdef double  pd_star = Ref.Pg - pv_star
        self.s_star = (1.0-Ref.qtg) * sd_c(pd_star, Ref.Tg) + Ref.qtg * sv_c(pv_star,Ref.Tg)


        if Pa.sub_z_rank != 0:
            return

        cdef:
            Py_ssize_t i,j, ijk, ij
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0]
            Py_ssize_t jmax = Gr.dims.nlg[1]
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t istride_2d = Gr.dims.nlg[1]
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')

            double [:] windspeed = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
            double ustar_
            double buoyancy_flux, theta_flux
            double theta_surface = Ref.Tg * exner_c(Ref.Pg)

            double cm_sqrt = sqrt(self.cm)

        compute_windspeed(&Gr.dims, &PV.values[u_shift], &PV.values[v_shift], &windspeed[0],Ref.u0, Ref.v0,self.gustiness)

        with nogil:
            for i in xrange(gw, imax-gw):
                for j in xrange(gw,jmax-gw):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    theta_flux = -self.ch * windspeed[ij] * (DV.values[t_shift + ijk]*exner_c(Ref.p0_half[gw]) - theta_surface)

                    self.s_flux[ij]  = -self.ch * windspeed[ij] * (PV.values[s_shift + ijk] - self.s_star)
                    self.qt_flux[ij] = -self.cq * windspeed[ij] * (PV.values[qt_shift + ijk] - Ref.qtg)
                    buoyancy_flux = g * ((theta_flux + (eps_vi-1.0)*(theta_surface*self.qt_flux[ij] + Ref.qtg * theta_flux))/(theta_surface*(1.0 + (eps_vi-1)*Ref.qtg)))
                    self.u_flux[ij]  = -self.cm * interp_2(windspeed[ij], windspeed[ij + istride_2d]) * (PV.values[u_shift + ijk] + Ref.u0)
                    self.v_flux[ij] = -self.cm * interp_2(windspeed[ij], windspeed[ij + 1])* (PV.values[v_shift + ijk] + Ref.v0)
                    ustar_ = cm_sqrt * windspeed[ij]
                    self.friction_velocity[ij] = ustar_

        SurfaceBase.update(self, Gr, Ref, PV, DV, Pa, TS)

        return


    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.stats_io(self, Gr, NS, Pa)
        return






cdef class SurfaceCGILS(SurfaceBase):
    def __init__(self, namelist, LatentHeat LH, ParallelMPI.ParallelMPI Pa):
        try:
            self.loc = namelist['meta']['CGILS']['location']
            if self.loc !=12 and self.loc != 11 and self.loc != 6:
                Pa.root_print('Invalid CGILS location (must be 6, 11, or 12)')
                Pa.kill()
        except:
            Pa.root_print('Must provide a CGILS location (6/11/12) in namelist')
            Pa.kill()
        try:
            self.is_p2 = namelist['meta']['CGILS']['P2']
        except:
            Pa.root_print('Must specify if CGILS run is perturbed')
            Pa.kill()

        self.gustiness = 0.001
        self.z0 = 1.0e-4
        self.L_fp = LH.L_fp
        self.Lambda_fp = LH.Lambda_fp
        self.CC = ClausiusClapeyron()
        self.CC.initialize(namelist, LH, Pa)
        self.dry_case = False

        return

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.initialize(self,Gr,Ref,NS,Pa)

        # Find the scalar transfer coefficient consistent with the vertical grid spacing
        cdef double z1 = Gr.dims.zp_half_0
        cdef double cq = 1.2e-3
        cdef double u10m=0.0, ct_ic=0.0, z1_ic=0.0
        if self.loc == 12:
            ct_ic = 0.0104
            z1_ic = 2.5
        elif self.loc == 11:
            ct_ic = 0.0081
            z1_ic = 12.5
        elif self.loc == 6:
            ct_ic = 0.0081
            z1_ic = 20.0

        u10m = ct_ic/cq * np.log(z1_ic/self.z0)**2/np.log(10.0/self.z0)**2

        self.ct = cq * u10m * (np.log(10.0/self.z0)/np.log(z1/self.z0))**2


        return


    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV,ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):

        if Pa.sub_z_rank != 0:
            return

        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            double [:] windspeed = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')

        compute_windspeed(&Gr.dims, &PV.values[u_shift], &PV.values[v_shift], &windspeed[0], Ref.u0, Ref.v0, self.gustiness)

        cdef:
            Py_ssize_t i,j, ijk, ij
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0]
            Py_ssize_t jmax = Gr.dims.nlg[1]
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t istride_2d = Gr.dims.nlg[1]
            double zb = Gr.dims.zp_half_0
            double [:] cm = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
            double pv_star = self.CC.LT.fast_lookup(self.T_surface)
            double qv_star = eps_v * pv_star/(Ref.Pg + (eps_v-1.0)*pv_star)
            double [:] t_mean = Pa.HorizontalMean(Gr, &DV.values[t_shift])
            double buoyancy_flux, th_flux
            double exner_b = exner_c(Ref.p0_half[gw])
            double theta_0 = self.T_surface/exner_c(Ref.Pg)



        with nogil:
            for i in xrange(gw-1, imax-gw+1):
                for j in xrange(gw-1,jmax-gw+1):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.qt_flux[ij] = self.ct * (0.98 * qv_star - PV.values[qt_shift + ijk])
                    th_flux = self.ct * (theta_0 - DV.values[t_shift + ijk]/exner_b )
                    buoyancy_flux = g * th_flux * exner_b/t_mean[gw] + g * (eps_vi-1.0)*self.qt_flux[ij]

                    self.friction_velocity[ij] = compute_ustar(windspeed[ij],buoyancy_flux,self.z0, zb)
                    self.s_flux[ij] = entropyflux_from_thetaflux_qtflux(th_flux, self.qt_flux[ij],
                                                                        Ref.p0_half[gw], DV.values[t_shift + ijk],
                                                                        PV.values[qt_shift + ijk], PV.values[qt_shift + ijk])
                    cm[ij] = (self.friction_velocity[ij]/windspeed[ij]) *  (self.friction_velocity[ij]/windspeed[ij])


            for i in xrange(gw, imax-gw):
                for j in xrange(gw, jmax-gw):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.u_flux[ij] = -interp_2(cm[ij], cm[ij+istride_2d])*interp_2(windspeed[ij], windspeed[ij+istride_2d]) * (PV.values[u_shift + ijk] + Ref.u0)
                    self.v_flux[ij] = -interp_2(cm[ij], cm[ij+1])*interp_2(windspeed[ij], windspeed[ij+1]) * (PV.values[v_shift + ijk] + Ref.v0)

        SurfaceBase.update(self, Gr, Ref, PV, DV, Pa, TS)

        return


    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.stats_io(self, Gr, NS, Pa)

        return



cdef class SurfaceZGILS(SurfaceBase):
    def __init__(self, namelist, LatentHeat LH, ParallelMPI.ParallelMPI Pa):


        self.gustiness = 0.001
        self.z0 = 1.0e-3
        self.L_fp = LH.L_fp
        self.Lambda_fp = LH.Lambda_fp
        self.CC = ClausiusClapeyron()
        self.CC.initialize(namelist, LH, Pa)

        self.dry_case = False
        try:
            self.loc = namelist['meta']['ZGILS']['location']
            if self.loc !=12 and self.loc != 11 and self.loc != 6:
                Pa.root_print('SURFACE: Invalid ZGILS location (must be 6, 11, or 12) '+ str(self.loc))
                Pa.kill()
        except:
            Pa.root_print('SURFACE: Must provide a ZGILS location (6/11/12) in namelist')
            Pa.kill()

        # Get the multiplying factor for current levels of CO2
        # Then convert to a number of CO2 doublings, which is how forcings are rescaled
        try:
            co2_factor =  namelist['radiation']['RRTM']['co2_factor']
        except:
            co2_factor = 1.0
        n_double_co2 = int(np.log2(co2_factor))

        try:
            constant_sst = namelist['surface_budget']['constant_sst']
        except:
            constant_sst = False

        # Set the initial sst value to the Fixed-SST case value (Tan et al 2016a, Table 1)
        if self.loc == 12:
            self.T_surface  = 289.8
        elif self.loc == 11:
            self.T_surface = 292.2
        elif self.loc == 6:
            self.T_surface = 298.9

        # adjust surface temperature for fixed-SST climate change experiments
        if constant_sst:
            self.T_surface = self.T_surface + 3.0 * n_double_co2

        return

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.initialize(self,Gr,Ref,NS,Pa)


        return


    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV,ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):

        if Pa.sub_z_rank != 0:
            return

        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t th_shift = DV.get_varshift(Gr, 'theta_rho')
            double [:] windspeed = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')

        compute_windspeed(&Gr.dims, &PV.values[u_shift], &PV.values[v_shift], &windspeed[0], Ref.u0, Ref.v0, self.gustiness)

        cdef:
            Py_ssize_t i,j, ijk, ij
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0]
            Py_ssize_t jmax = Gr.dims.nlg[1]
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t istride_2d = Gr.dims.nlg[1]


            double ustar, t_flux, b_flux
            double theta_rho_b, Nb2, Ri
            double zb = Gr.dims.zp_half_0
            double [:] cm = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
            double ch=0.0

            double pv_star = self.CC.LT.fast_lookup(self.T_surface)
            double qv_star = eps_v * pv_star/(Ref.Pg + (eps_v-1.0)*pv_star)



            # Find the surface entropy
            double pd_star = Ref.Pg - pv_star

            double theta_rho_g = theta_rho_c(Ref.Pg, self.T_surface, qv_star, qv_star)
            double s_star = sd_c(pd_star,self.T_surface) * (1.0 - qv_star) + sv_c(pv_star, self.T_surface) * qv_star

            double [:] t_mean = Pa.HorizontalMean(Gr, &DV.values[t_shift])

        with nogil:
            for i in xrange(gw-1, imax-gw+1):
                for j in xrange(gw-1,jmax-gw+1):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    theta_rho_b = DV.values[th_shift + ijk]
                    Nb2 = g/theta_rho_g*(theta_rho_b-theta_rho_g)/zb
                    Ri = Nb2 * zb * zb/(windspeed[ij] * windspeed[ij])
                    exchange_coefficients_byun(Ri, zb, self.z0, &cm[ij], &ch, &self.obukhov_length[ij])
                    self.s_flux[ij] = -ch *windspeed[ij] * (PV.values[s_shift + ijk] - s_star)
                    self.qt_flux[ij] = -ch *windspeed[ij] *  (PV.values[qt_shift + ijk] - qv_star)
                    ustar = sqrt(cm[ij]) * windspeed[ij]
                    self.friction_velocity[ij] = ustar

            for i in xrange(gw, imax-gw):
                for j in xrange(gw, jmax-gw):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.u_flux[ij] = -interp_2(cm[ij], cm[ij+istride_2d])*interp_2(windspeed[ij], windspeed[ij+istride_2d]) * (PV.values[u_shift + ijk] + Ref.u0)
                    self.v_flux[ij] = -interp_2(cm[ij], cm[ij+1])*interp_2(windspeed[ij], windspeed[ij+1]) * (PV.values[v_shift + ijk] + Ref.v0)

        SurfaceBase.update(self, Gr, Ref, PV, DV, Pa, TS)

        return


    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.stats_io(self, Gr, NS, Pa)

        return


cdef class SurfaceTRMM_LBA(SurfaceBase):
    def __init__(self, namelist, LatentHeat LH, ParallelMPI.ParallelMPI Pa):

        self.L_fp = LH.L_fp # is this related ?
        self.Lambda_fp = LH.Lambda_fp # is this related ?

        self.dry_case = False

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.initialize(self,Gr,Ref,NS,Pa)
        self.windspeed = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
        self.T_surface = 292.5

        return


    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV,ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):

        if Pa.sub_z_rank != 0:
            return

        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t ql_shift = DV.get_varshift(Gr, 'ql')
        #cdef double[:]
        F_pert = np.random.random_sample(Gr.dims.npg)
        cdef double th_flux
        cdef double Tmax = 5.25
        if TS.rk_step == 0:
            if TS.t<Tmax*3600.0:
                self.ft = 270.0 * np.power(np.maximum(0, np.cos(np.pi/2*((5.25*3600.0 - TS.t)/5.25/3600.0))),1.5) #*(1.0 + 0.1*F_pert) # F_S equation from TRMM paper with 10% random pert
                self.fq = 554.0 * np.power(np.maximum(0, np.cos(np.pi/2*((5.25*3600.0 - TS.t)/5.25/3600.0))),1.3) #*(1.0 + 0.1*F_pert) # F_L equation from TRMM paper with 10% random pert
                #self.ft = np.cos(np.pi/2.0*((5.25*3600.0 - TS.t)/5.25/3600.0)) # F_S equation from TRMM paper with 10% random pert
                #self.fq = np.cos(np.pi/2.0*((5.25*3600.0 - TS.t)/5.25/3600.0)) # F_L equation from TRMM paper with 10% random pert
            else:
                self.ft = 270.0 * np.power(np.maximum(0, np.cos(np.pi/2*((5.25*3600.0 - Tmax*3600.0)/5.25/3600.0))),1.5) #*(1.0 + 0.1*F_pert) # F_S equation from TRMM paper with 10% random pert
                self.fq = 554.0 * np.power(np.maximum(0, np.cos(np.pi/2*((5.25*3600.0 - Tmax*3600.0)/5.25/3600.0))),1.3) #*(1.0 + 0.1*F_pert) # F_L equation from TRMM paper with 10% random pert

        cdef double EX = exner_c(Ref.Pg)
            #th_flux = self.ft#/Ref.rho0[Gr.dims.gw-1]/cpd/EX

        cdef:
            Py_ssize_t i,j, ijk, ij
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0]
            Py_ssize_t jmax = Gr.dims.nlg[1]
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t istride_2d = Gr.dims.nlg[1]

            double lam, lv, pv, pd, sv, sd



        with nogil:
            for i in xrange(gw-1, imax-gw+1):
                for j in xrange(gw-1, jmax-gw+1):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.friction_velocity[ij] = 0
                    lam = self.Lambda_fp(DV.values[t_shift+ijk])
                    lv = self.L_fp(DV.values[t_shift+ijk],lam)
                    pv = pv_c(Ref.p0_half[gw], PV.values[ijk + qt_shift], PV.values[ijk + qt_shift] - DV.values[ijk + ql_shift])
                    pd = pd_c(Ref.p0_half[gw], PV.values[ijk + qt_shift], PV.values[ijk + qt_shift] - DV.values[ijk + ql_shift])
                    sv = sv_c(pv,DV.values[t_shift+ijk])
                    sd = sd_c(pd,DV.values[t_shift+ijk])
                    th_flux = self.ft/Ref.rho0[Gr.dims.gw-1]/cpd/EX
                    self.qt_flux[ij] = self.fq/lv/Ref.rho0[Gr.dims.gw-1]
                    self.s_flux[ij] = entropyflux_from_thetaflux_qtflux(th_flux, self.qt_flux[ij],
                                                                     Ref.p0_half[gw], DV.values[t_shift + ijk],
                                                                     PV.values[qt_shift + ijk], PV.values[qt_shift + ijk])

        SurfaceBase.update(self, Gr, Ref, PV, DV, Pa,TS)
        return

    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.stats_io(self, Gr, NS, Pa)

        return

cdef class SurfaceARM_SGP(SurfaceBase):
    def __init__(self, namelist, LatentHeat LH, ParallelMPI.ParallelMPI Pa):


        self.L_fp = LH.L_fp # is this related ?
        self.Lambda_fp = LH.Lambda_fp # is this related ?

        self.dry_case = False

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.initialize(self,Gr,Ref,NS,Pa)
        self.windspeed = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
        self.T_surface = 299.0
        self.ft  = -30.0 # W/m^2
        self.fq  = 5.0 # W/m^2

        return


    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV,ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):

        if Pa.sub_z_rank != 0:
            return
        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t ql_shift = DV.get_varshift(Gr, 'ql')
        #cdef double[:]
        #F_pert = np.random.random_sample(Gr.dims.npg)
        cdef:
            double th_flux
            double [:] SH = np.zeros(5580,dtype=np.double,order='c') # simulation time / dt
            double [:] LH = np.zeros(5580,dtype=np.double,order='c')
        # update fluxes start from the second
        t_in = np.array([0.0, 4.0, 6.5, 7.5, 10.0, 12.5, 14.5]) * 3600 #LES time is in sec
        SH = np.array([-30.0, 90.0, 140.0, 140.0, 100.0, -10, -10]) # W/m^2
        LH = np.array([5.0, 250.0, 450.0, 500.0, 420.0, 180.0, 0.0]) # W/m^2
        if TS.rk_step == 0:

            self.ft = np.interp(TS.t,t_in,SH)
            self.fq = np.interp(TS.t,t_in,LH)

        cdef double EX = exner_c(Ref.Pg)
            #th_flux = self.ft#/Ref.rho0[Gr.dims.gw-1]/cpd/EX

        cdef:
            Py_ssize_t i,j, ijk, ij
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0]
            Py_ssize_t jmax = Gr.dims.nlg[1]
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t istride_2d = Gr.dims.nlg[1]

            double lam, lv, pv, pd, sv, sd



        with nogil:
            for i in xrange(gw-1, imax-gw+1):
                for j in xrange(gw-1, jmax-gw+1):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.friction_velocity[ij] = 0
                    lam = self.Lambda_fp(DV.values[t_shift+ijk])
                    lv = self.L_fp(DV.values[t_shift+ijk],lam)
                    pv = pv_c(Ref.p0_half[gw], PV.values[ijk + qt_shift], PV.values[ijk + qt_shift] - DV.values[ijk + ql_shift])
                    pd = pd_c(Ref.p0_half[gw], PV.values[ijk + qt_shift], PV.values[ijk + qt_shift] - DV.values[ijk + ql_shift])
                    sv = sv_c(pv,DV.values[t_shift+ijk])
                    sd = sd_c(pd,DV.values[t_shift+ijk])
                    th_flux = self.ft/Ref.rho0[Gr.dims.gw-1]/cpd/EX
                    self.qt_flux[ij] = self.fq/lv/Ref.rho0[Gr.dims.gw-1]
                    self.s_flux[ij] = entropyflux_from_thetaflux_qtflux(th_flux, self.qt_flux[ij],
                                                                     Ref.p0_half[gw], DV.values[t_shift + ijk],
                                                                     PV.values[qt_shift + ijk], PV.values[qt_shift + ijk])
        SurfaceBase.update(self, Gr, Ref, PV, DV, Pa,TS)
        return

    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.stats_io(self, Gr, NS, Pa)

        return

cdef class SurfaceSCMS2(SurfaceBase):
    def __init__(self, namelist, LatentHeat LH, ParallelMPI.ParallelMPI Pa):


        self.L_fp = LH.L_fp # is this related ?
        self.Lambda_fp = LH.Lambda_fp # is this related ?

        self.dry_case = False

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.initialize(self,Gr,Ref,NS,Pa)
        self.windspeed = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
        self.T_surface = 299.0
        self.ft  = 0.0 # W/m^2
        self.fq  = 0.0 # W/m^2

        return


    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV,ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):

        if Pa.sub_z_rank != 0:
            return
        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t ql_shift = DV.get_varshift(Gr, 'ql')
        #cdef double[:]
        #F_pert = np.random.random_sample(Gr.dims.npg)
        cdef:
            double th_flux
            double [:] SH = np.zeros(5580.0,dtype=np.double,order='c') # simulation time / dt
            double [:] LH = np.zeros(5580.0,dtype=np.double,order='c')
        # update fluxes start from the second
        if TS.rk_step == 0:
            self.ft = 100.0*np.sin(TS.t*np.pi/(12.0*3600.0))
            self.fq = 300.0*np.sin(TS.t*np.pi/(12.0*3600.0))

        cdef double EX = exner_c(Ref.Pg)
            #th_flux = self.ft#/Ref.rho0[Gr.dims.gw-1]/cpd/EX

        cdef:
            Py_ssize_t i,j, ijk, ij
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0]
            Py_ssize_t jmax = Gr.dims.nlg[1]
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t istride_2d = Gr.dims.nlg[1]

            double lam, lv, pv, pd, sv, sd



        with nogil:
            for i in xrange(gw-1, imax-gw+1):
                for j in xrange(gw-1, jmax-gw+1):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.friction_velocity[ij] = 0
                    lam = self.Lambda_fp(DV.values[t_shift+ijk])
                    lv = self.L_fp(DV.values[t_shift+ijk],lam)
                    pv = pv_c(Ref.p0_half[gw], PV.values[ijk + qt_shift], PV.values[ijk + qt_shift] - DV.values[ijk + ql_shift])
                    pd = pd_c(Ref.p0_half[gw], PV.values[ijk + qt_shift], PV.values[ijk + qt_shift] - DV.values[ijk + ql_shift])
                    sv = sv_c(pv,DV.values[t_shift+ijk])
                    sd = sd_c(pd,DV.values[t_shift+ijk])
                    th_flux = self.ft/Ref.rho0[Gr.dims.gw-1]/cpd/EX
                    self.qt_flux[ij] = self.fq/lv/Ref.rho0[Gr.dims.gw-1]
                    self.s_flux[ij] = entropyflux_from_thetaflux_qtflux(th_flux, self.qt_flux[ij],
                                                                     Ref.p0_half[gw], DV.values[t_shift + ijk],
                                                                     PV.values[qt_shift + ijk], PV.values[qt_shift + ijk])
        SurfaceBase.update(self, Gr, Ref, PV, DV, Pa,TS)
        return

    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.stats_io(self, Gr, NS, Pa)

        return

cdef class SurfaceSCMS(SurfaceBase):
    def __init__(self, namelist, LatentHeat LH, ParallelMPI.ParallelMPI Pa):

        self.L_fp = LH.L_fp
        self.Lambda_fp = LH.Lambda_fp
        self.z0 = 0.035 #m (Roughness length)
        self.gustiness = 0.001 #m/s, minimum surface windspeed for determination of u*
        self.dry_case = False

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.initialize(self,Gr,Ref,NS,Pa)
        self.windspeed = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
        self.buoyancy_flux = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
        self.T_surface = 299.0

        return


    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV,ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):

        if Pa.sub_z_rank != 0:
            return

        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t ql_shift = DV.get_varshift(Gr, 'ql')
        #cdef double[:]
        #F_pert = np.random.random_sample(Gr.dims.npg)
        cdef double th_flux, SH, LH
        # update fluxes start from the second
        if TS.rk_step == 0:
            self.ft = 100.0*np.sin(TS.t*np.pi/(12.0*3600.0))
            self.fq = 300.0*np.sin(TS.t*np.pi/(12.0*3600.0))

        cdef double EX = exner_c(Ref.Pg)

        cdef:
            Py_ssize_t i,j, ijk, ij
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0]
            Py_ssize_t jmax = Gr.dims.nlg[1]
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t istride_2d = Gr.dims.nlg[1]
            double [:] windspeed = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1],dtype=np.double,order='c')
            double lam, lv, pv, pd, sv, sd, theta_flux, qt_flux
            double T0 = Ref.p0_half[Gr.dims.gw] * Ref.alpha0_half[Gr.dims.gw]/Rd

        theta_flux = self.ft/(cpm(Ref.qtg)*1.22*exner_c(Ref.p0_half[gw])) # yair calculate this exactly from a function
        qt_flux_ = self.fq/self.L_fp(T0,self.Lambda_fp(T0))
        self.buoyancy_flux = g * ((theta_flux + (eps_vi-1.0)*(T0*exner_c(Ref.p0_half[gw]) * qt_flux + Ref.qtg * theta_flux))
                                  /(T0*exner_c(Ref.p0_half[gw])*(1.0 + (eps_vi-1)*Ref.qtg)))

        with nogil:
            for i in xrange(gw-1, imax-gw+1):
                for j in xrange(gw-1, jmax-gw+1):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.friction_velocity[ij] = 0
                    lam = self.Lambda_fp(DV.values[t_shift+ijk])
                    lv = self.L_fp(DV.values[t_shift+ijk],lam)
                    pv = pv_c(Ref.p0_half[gw], PV.values[ijk + qt_shift], PV.values[ijk + qt_shift] - DV.values[ijk + ql_shift])
                    pd = pd_c(Ref.p0_half[gw], PV.values[ijk + qt_shift], PV.values[ijk + qt_shift] - DV.values[ijk + ql_shift])
                    sv = sv_c(pv,DV.values[t_shift+ijk])
                    sd = sd_c(pd,DV.values[t_shift+ijk])
                    th_flux = self.ft/Ref.rho0[Gr.dims.gw-1]/cpd/EX
                    self.qt_flux[ij] = self.fq/lv/Ref.rho0[Gr.dims.gw-1]
                    self.s_flux[ij] = entropyflux_from_thetaflux_qtflux(th_flux, self.qt_flux[ij],
                                                                     Ref.p0_half[gw], DV.values[t_shift + ijk],
                                                                     PV.values[qt_shift + ijk], PV.values[qt_shift + ijk])


        compute_windspeed(&Gr.dims, &PV.values[u_shift], &PV.values[v_shift], &windspeed[0],Ref.u0, Ref.v0,self.gustiness)

        # Get the shear stresses
        with nogil:
            for i in xrange(1,imax):
                for j in xrange(1,jmax):
                    ij = i * istride_2d + j
                    self.friction_velocity[ij] = compute_ustar(windspeed[ij],self.buoyancy_flux,self.z0, Gr.dims.dx[2]/2.0)
            for i in xrange(1,imax-1):
                for j in xrange(1,jmax-1):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.u_flux[ij] = -interp_2(self.friction_velocity[ij], self.friction_velocity[ij+istride_2d])**2/interp_2(windspeed[ij], windspeed[ij+istride_2d]) \
                                      * (PV.values[u_shift + ijk] + Ref.u0)
                    self.v_flux[ij] = -interp_2(self.friction_velocity[ij], self.friction_velocity[ij+1])**2/interp_2(windspeed[ij], windspeed[ij+1]) \
                                      * (PV.values[v_shift + ijk] + Ref.v0)

        SurfaceBase.update(self, Gr, Ref, PV, DV, Pa,TS)
        return

    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.stats_io(self, Gr, NS, Pa)

        return

cdef class SurfaceGATE_III(SurfaceBase):
    def __init__(self, namelist, LatentHeat LH, ParallelMPI.ParallelMPI Pa):

        # surface fluxes are computed , surface temperature is constant
        self.z0 = 1.0e-3 # yair check what is the correct value for that
        self.L_fp = LH.L_fp # is this related ?
        self.Lambda_fp = LH.Lambda_fp # is this related ?
        self.dry_case = False
        self.CC = ClausiusClapeyron()
        self.CC.initialize(namelist, LH, Pa)

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.initialize(self,Gr,Ref,NS,Pa)
        #self.windspeed = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
        self.gustiness = 0.001
        self.T_surface = 299.88
        self.qt_surface = 16.5/1000.0
        self.p_surface = 1012.0*100
        return


    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV,ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):

        if Pa.sub_z_rank != 0:
            return

        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t ql_shift = DV.get_varshift(Gr, 'ql')
            Py_ssize_t th_shift = DV.get_varshift(Gr, 'theta_rho')
            double [:] windspeed = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
            double zb = Gr.dims.zp_half_0
            double [:] cm= np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
            double ch=0.0
        #cdef double[:]
        #F_pert = np.random.random_sample(Gr.dims.npg)
        cdef double th_flux, SH, LH
        compute_windspeed(&Gr.dims, &PV.values[u_shift], &PV.values[v_shift], &windspeed[0],Ref.u0, Ref.v0,self.gustiness)

        cdef:
            Py_ssize_t i,j, ijk, ij
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0]
            Py_ssize_t jmax = Gr.dims.nlg[1]
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t istride_2d = Gr.dims.nlg[1]

            double pv_star = self.CC.LT.fast_lookup(self.T_surface)
            double qv_star = eps_v * pv_star/(Ref.Pg + (eps_v-1.0)*pv_star)
            # Find the surface entropy
            double pd_star = Ref.Pg - pv_star
            double theta_rho_g = theta_rho_c(Ref.Pg, self.T_surface, qv_star, qv_star)
            double s_star = sd_c(pd_star,self.T_surface) * (1.0 - qv_star) + sv_c(pv_star, self.T_surface) * qv_star

            double theta_rho_b, Nb2, Ri, ustar


       # I am following here the calculation in Surface Gabls, but I need to think about the calculation of the surface qt flux as well as s
        with nogil:
            for i in xrange(gw-1, imax-gw+1):
                for j in xrange(gw-1,jmax-gw+1):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    theta_rho_b = DV.values[th_shift + ijk]
                    Nb2 = g/theta_rho_g*(theta_rho_b-theta_rho_g)/zb
                    Ri = Nb2 * zb * zb/(windspeed[ij] * windspeed[ij])
                    exchange_coefficients_byun(Ri, zb, self.z0, &cm[ij], &ch, &self.obukhov_length[ij])
                    self.s_flux[ij] = -ch *windspeed[ij] * (PV.values[s_shift + ijk] - s_star)
                    self.qt_flux[ij] = -ch *windspeed[ij] *  (PV.values[qt_shift + ijk] - qv_star)
                    ustar = sqrt(cm[ij]) * windspeed[ij]
                    self.friction_velocity[ij] = ustar


            for i in xrange(gw, imax-gw):
                for j in xrange(gw, jmax-gw):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.u_flux[ij] = -interp_2(cm[ij], cm[ij+istride_2d])*interp_2(windspeed[ij], windspeed[ij+istride_2d]) * (PV.values[u_shift + ijk] + Ref.u0)
                    self.v_flux[ij] = -interp_2(cm[ij], cm[ij+1])*interp_2(windspeed[ij], windspeed[ij+1]) * (PV.values[v_shift + ijk] + Ref.v0)

        SurfaceBase.update(self, Gr, Ref, PV, DV, Pa,TS)
        return

cdef class SurfaceGATE_III_2(SurfaceBase):
    def __init__(self, namelist, LatentHeat LH, ParallelMPI.ParallelMPI Pa):


        self.L_fp = LH.L_fp # is this related ?
        self.Lambda_fp = LH.Lambda_fp # is this related ?

        self.dry_case = False
        self.CC = ClausiusClapeyron()
        self.CC.initialize(namelist, LH, Pa)

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.initialize(self,Gr,Ref,NS,Pa)
        self.gustiness = 0.001
        self.z0 = 1.0e-3
        self.windspeed = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
        self.p_surface = 1012.0*100
        self.ft  = 2.933 # W/m^2
        self.fq  = 47.304 # W/m^2
        self.gustiness = 0.001
        self.T_surface = 299.88

        return


    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV,ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS):

        if Pa.sub_z_rank != 0:
            return
        cdef:
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t ql_shift = DV.get_varshift(Gr, 'ql')


        #cdef double[:]
        #F_pert = np.random.random_sample(Gr.dims.npg)
        cdef:

            double [:] SH = np.zeros(5580.0,dtype=np.double,order='c') # simulation time / dt
            double [:] LH = np.zeros(5580.0,dtype=np.double,order='c')
        # update fluxes start from the second
        SHF_time = [0, 3773.065, 11513.734, 21072.119 ,22120.268, 25553.215, 24417.061, 27760.012, 34521.359, 36935.926, 42593.543, 44989.590, 52826.871, 58628.070,63144.234, 65225.969, 66544.758, 72381.023, 73517.188, 78149.797, 81568.867, 86400.0]#LES time is in sec
        SHF      = [2.933, 3.198, 3.692, 4.348, 4.972, 7.830, 7.213, 8.078, 7.476, 7.848, 8.123, 8.750, 8.769, 8.609, 8.623, 8.663, 9.018, 9.359, 9.613, 9.094, 9.139, 9.951 ] # W/m^2
        LHF_time = [0, 1054.103, 2939.973, 5206.326, 19933.324, 22267.168,25899.947, 27494.670, 29747.123, 34256.668, 36969.680, 42208.430 , 45773.051, 48876.465, 53519.016, 55598.770, 58390.520, 62381.949, 63882.035, 67553.219, 70486.555, 73776.578, 78832.031, 81598.641, 86400.0]
        LHF      = [47.304, 45.878, 47.095, 49.497, 45.012, 46.001, 55.322, 56.429, 53.930, 51.266, 53.446, 53.865, 57.003, 55.447, 54.898, 54.311, 55.490, 54.992, 54.522, 57.013, 57.013, 58.035, 57.184, 56.919, 61.078] # W/m^2
        if TS.rk_step == 0:

            self.ft = np.interp(TS.t,SHF_time,SHF)
            self.fq = np.interp(TS.t,LHF_time,LHF)

        compute_windspeed(&Gr.dims, &PV.values[u_shift], &PV.values[v_shift], &self.windspeed[0],Ref.u0, Ref.v0,self.gustiness)

        cdef:
            double EX = exner_c(Ref.Pg)
            double th_flux

        cdef:
            Py_ssize_t i,j, ijk, ij
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0]
            Py_ssize_t jmax = Gr.dims.nlg[1]
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t istride_2d = Gr.dims.nlg[1]
            Py_ssize_t th_shift = DV.get_varshift(Gr, 'theta_rho')

            double lam, lv, pv, pd, sv, sd
            double theta_rho_b, Nb2, Ri
            double ustar #, t_flux, b_flux
            double zb = Gr.dims.zp_half_0
            double [:] windspeed = self.windspeed
            double pv_star = self.CC.LT.fast_lookup(self.T_surface)
            double qv_star = eps_v * pv_star/(Ref.Pg + (eps_v-1.0)*pv_star)
            double [:] cm = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
            double theta_rho_g = theta_rho_c(Ref.Pg, self.T_surface, qv_star, qv_star)
            double ch=0.0


        with nogil:
            for i in xrange(gw-1, imax-gw+1):
                for j in xrange(gw-1, jmax-gw+1):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    theta_rho_b = DV.values[th_shift + ijk]
                    Nb2 = g/theta_rho_g*(theta_rho_b-theta_rho_g)/zb
                    Ri = Nb2 * zb * zb/(windspeed[ij] * windspeed[ij])
                    exchange_coefficients_byun(Ri, zb, self.z0, &cm[ij], &ch, &self.obukhov_length[ij])
                    lam = self.Lambda_fp(DV.values[t_shift+ijk])
                    lv = self.L_fp(DV.values[t_shift+ijk],lam)
                    pv = pv_c(Ref.p0_half[gw], PV.values[ijk + qt_shift], PV.values[ijk + qt_shift] - DV.values[ijk + ql_shift])
                    pd = pd_c(Ref.p0_half[gw], PV.values[ijk + qt_shift], PV.values[ijk + qt_shift] - DV.values[ijk + ql_shift])
                    sv = sv_c(pv,DV.values[t_shift+ijk])
                    sd = sd_c(pd,DV.values[t_shift+ijk])
                    th_flux = self.ft/Ref.rho0[Gr.dims.gw-1]/cpd/EX
                    self.qt_flux[ij] = self.fq/lv/Ref.rho0[Gr.dims.gw-1]
                    self.s_flux[ij] = entropyflux_from_thetaflux_qtflux(th_flux, self.qt_flux[ij],
                                                                     Ref.p0_half[gw], DV.values[t_shift + ijk],
                                                                     PV.values[qt_shift + ijk], PV.values[qt_shift + ijk])
                    ustar = sqrt(cm[ij]) * windspeed[ij]
                    self.friction_velocity[ij] = ustar

            for i in xrange(gw, imax-gw):
                for j in xrange(gw, jmax-gw):
                    ijk = i * istride + j * jstride + gw
                    ij = i * istride_2d + j
                    self.u_flux[ij] = -interp_2(cm[ij], cm[ij+istride_2d])*interp_2(windspeed[ij], windspeed[ij+istride_2d]) * (PV.values[u_shift + ijk] + Ref.u0)
                    self.v_flux[ij] = -interp_2(cm[ij], cm[ij+1])*interp_2(windspeed[ij], windspeed[ij+1]) * (PV.values[v_shift + ijk] + Ref.v0)


        SurfaceBase.update(self, Gr, Ref, PV, DV, Pa,TS)
        return

    cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):
        SurfaceBase.stats_io(self, Gr, NS, Pa)

        return


# Anderson, R. J., 1993: A Study of Wind Stress and Heat Flux over the Open
# Ocean by the Inertial-Dissipation Method. J. Phys. Oceanogr., 23, 2153--“2161.
# See also: ARPS documentation
cdef inline double compute_z0(double z1, double windspeed) nogil:
    cdef double z0 =z1*exp(-kappa/sqrt((0.4 + 0.079*windspeed)*1e-3))
    return z0


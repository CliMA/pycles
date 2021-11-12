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

from fms_forcing_reader import reader
from cfsites_forcing_reader import cfreader
from cfgrid_forcing_reader import cfreader_grid

#import pylab as plt
include 'parameters.pxi'

cdef extern from 'advection_interpolation.h':
    double interp_weno3(double phim1, double phi, double phip) nogil

cdef class ForcingGCMMean:
    def __init__(self, namelist, LatentHeat LH, ParallelMPI.ParallelMPI Pa):
        self.file = str(namelist['gcm']['file'])
        self.lat = namelist['gcm']['lat']
        self.lon = namelist['gcm']['lon']
        try:
            self.relax_scalar = namelist['forcing']['relax_scalar']
        except:
            self.relax_scalar = False
        try:
            self.tau_scalar = namelist['forcing']['tau_scalar']
        except:
            self.tau_scalar = 21600.0
        self.gcm_profiles_initialized = False
        self.t_indx = 0
        return

    @cython.wraparound(True)
    cpdef initialize(self, Grid.Grid Gr,ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        self.coriolis_param = 2.0 * omega * sin(self.lat * pi / 180.0 )
        self.qt_tend_nudge = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.t_tend_nudge = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')

        NS.add_profile('ls_subsidence', Gr, Pa)
        NS.add_profile('ls_subsidence_tr', Gr, Pa)
        NS.add_profile('ls_dtdt_hadv', Gr, Pa)
        NS.add_profile('ls_dtdt_fino', Gr, Pa)
        NS.add_profile('ls_dtdt_resid', Gr, Pa)
        NS.add_profile('ls_dtdt_fluc', Gr, Pa)
        NS.add_profile('dtdt_nudge', Gr, Pa)
        NS.add_profile('ls_dsdt_hadv', Gr, Pa)
        NS.add_profile('ls_dqtdt_hadv', Gr, Pa)
        NS.add_profile('ls_dqtdt_resid', Gr, Pa)
        NS.add_profile('ls_dqtdt_fluc', Gr, Pa)
        NS.add_profile('dqtdt_nudge', Gr, Pa)
        NS.add_profile('ls_subs_dtdt', Gr, Pa)
        NS.add_profile('ls_subs_dsdt', Gr, Pa)
        NS.add_profile('ls_fino_dsdt', Gr, Pa)
        NS.add_profile('ls_subs_dqtdt', Gr, Pa)


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
            Py_ssize_t s_shift
            Py_ssize_t thli_shift
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t ql_shift = DV.get_varshift(Gr,'ql')
            Py_ssize_t qi_shift = DV.get_varshift(Gr, 'qi') 
            double pd, pv, qt, qv, p0, rho0, t
            double zmax, weight, weight_half
            double u0_new, v0_new
            double [:] qt_mean = Pa.HorizontalMean(Gr, &PV.values[qt_shift])
            double [:] t_mean = Pa.HorizontalMean(Gr, &DV.values[t_shift])

        if not self.gcm_profiles_initialized:
            self.t_indx = int(TS.t // (3600.0 * 6.0))
            self.gcm_profiles_initialized = True
            Pa.root_print('Updating Time Varying Forcing')

            rdr = reader(self.file, self.lat, self.lon)

            #fh = open(self.file, 'r')
            #input_data_tv = cPickle.load(fh)
            #fh.close()

            #zfull = np.mean(input_data_tv['zfull'][:,::-1], axis=0)
            #alpha = np.mean(input_data_tv['alpha'][:,::-1], axis=0)
            #ug = np.mean(input_data_tv['u_geos'][:,::-1], axis=0)
            #vg = np.mean(input_data_tv['v_geos'][:,::-1], axis=0)
            #temp_dt_hadv = np.mean(input_data_tv['temp_hadv'][:,::-1], axis=0)
            #temp_dt_vadv = np.mean(input_data_tv['temp_vadv'][:,::-1], axis=0)
            #temp_dt_fino = np.mean(input_data_tv['temp_fino'][:,::-1], axis=0)
            #temp_dt_resid = np.mean(input_data_tv['temp_real1'][:,::-1], axis=0) - np.mean(input_data_tv['temp_total'][:,::-1],axis=0)
            #shum_dt_hadv = np.mean(input_data_tv['dt_qg_hadv'][:,::-1], axis=0)
            #shum_dt_vadv = np.mean(input_data_tv['dt_qg_vadv'][:,::-1], axis=0)
            #shum_dt_resid = np.mean(input_data_tv['dt_qg_real1'][:,::-1], axis=0) - np.mean(input_data_tv['dt_qg_total'][:,::-1],axis=0)

            #v_dt_tot = np.mean(input_data_tv['dt_vg_real1'][:,::-1], axis=0)
            #v_dt_cof = np.mean(input_data_tv['dt_vg_cori'][:,::-1], axis=0)
            #u_dt_tot = np.mean(input_data_tv['dt_ug_real1'][:,::-1], axis=0)
            #u_dt_cof = np.mean(input_data_tv['dt_ug_cori'][:,::-1], axis=0)
            #omega = np.mean(input_data_tv['omega'][:,::-1], axis=0)
            #temp = np.mean(input_data_tv['temp'][:,::-1], axis=0)
            #shum = np.mean(input_data_tv['shum'][:,::-1], axis=0)



            self.ug = rdr.get_interp_profile_old('u_geos', Gr.zp_half)#interp_pchip(Gr.zp_half, zfull, ug)
            self.vg = rdr.get_interp_profile_old('v_geos', Gr.zp_half)
            self.omega_vv = rdr.get_interp_profile_old('omega', Gr.zp_half)
            alpha = rdr.get_interp_profile_old('alpha', Gr.zp_half)
            self.subsidence =  -np.array(self.omega_vv) * alpha / g

            self.temp = rdr.get_interp_profile_old('temp', Gr.zp_half)
            self.shum = rdr.get_interp_profile_old('sphum', Gr.zp_half)

            self.temp_dt_hadv  = rdr.get_interp_profile_old('dt_tg_hadv', Gr.zp_half)
            self.temp_dt_fino = rdr.get_interp_profile_old('dt_tg_fino', Gr.zp_half)
            self.temp_dt_resid = rdr.get_interp_profile_old('dt_tg_real1', Gr.zp_half) - rdr.get_interp_profile_old('dt_tg_total', Gr.zp_half)
            self.shum_dt_hadv  = rdr.get_interp_profile_old('dt_qg_hadv', Gr.zp_half)
            #092119[ZS]: Change dt_qg_real1 to dt_qg_total since do_spec_tracer_filter=False in GCM
            self.shum_dt_resid  = rdr.get_interp_profile_old('dt_qg_total', Gr.zp_half) - rdr.get_interp_profile_old('dt_qg_total', Gr.zp_half)
            #self.temp_dt_hadv = interp_pchip(Gr.zp_half, zfull, temp_dt_hadv)
            #self.temp_dt_fino = interp_pchip(Gr.zp_half, zfull, temp_dt_fino)
            #self.temp_dt_resid = interp_pchip(Gr.zp_half, zfull, temp_dt_resid)
            #self.shum_dt_hadv = interp_pchip(Gr.zp_half, zfull, shum_dt_hadv)
            #self.shum_dt_resid = interp_pchip(Gr.zp_half, zfull, shum_dt_resid)
            '''
            data_dump = {}
            data_dump['zp_half'] = np.array(Gr.zp_half)
            data_dump['zfull'] = zfull
            data_dump['subsidence'] = np.array(-omega * alpha / g)
            data_dump['temp_dt_hadv'] = np.array(temp_dt_hadv)
            data_dump['temp_dt_fino'] = np.array(temp_dt_fino)
            data_dump['temp_dt_vadv'] = np.array(temp_dt_vadv)
            data_dump['temp_dt_resid'] = np.array(temp_dt_resid)
            data_dump['temp'] = temp

            import cPickle as pkl
            f = open('forcing_serial.pkl', 'wb')

            pkl.dump(data_dump, f)
            f.close()

            '''

            temp_dt_vadv_interp = rdr.get_interp_profile('dt_tg_vadv', Gr.zp_half)#interp_pchip(Gr.zp_half, zfull, temp_dt_vadv)
            temp_at_zp  = rdr.get_interp_profile('temp', Gr.zp) #interp_pchip(Gr.zp, zfull, temp)
            temp_vadv_pp = np.zeros(np.shape(self.temp_dt_hadv))
            temp_vadv_ppp = np.zeros(np.shape(self.temp_dt_hadv))

            shum_dt_vadv_interp = rdr.get_interp_profile('dt_qg_vadv', Gr.zp_half)#interp_pchip(Gr.zp_half, zfull, shum_dt_vadv)
            shum_at_zp = rdr.get_interp_profile('sphum', Gr.zp)#interp_pchip(Gr.zp, zfull, shum)
            shum_vadv_pp = np.zeros(np.shape(self.temp_dt_hadv))


            for k in xrange(temp_at_zp.shape[0]-1):
                temp_vadv_pp[k] = temp_dt_vadv_interp[k] + self.temp_dt_fino[k] + ( (temp_at_zp[k+1] - temp_at_zp[k]) * Gr.dims.dxi[2] * Gr.dims.imetl_half[k] + g/cpd)* self.subsidence[k]
                shum_vadv_pp[k] = shum_dt_vadv_interp[k] + ( (shum_at_zp[k+1] - shum_at_zp[k]) * Gr.dims.dxi[2] * Gr.dims.imetl_half[k])* self.subsidence[k]


            for k in xrange(2, temp_at_zp.shape[0]-1):
                tp1 = interp_weno3(temp_at_zp[k-1], temp_at_zp[k], temp_at_zp[k+1])
                tm1 = interp_weno3(temp_at_zp[k-2], temp_at_zp[k-1], temp_at_zp[k])
                temp_vadv_ppp[k] = temp_dt_vadv_interp[k] + self.temp_dt_fino[k] + ( (tp1 - tm1) * Gr.dims.dxi[2] * Gr.dims.imetl_half[k] + g/cpd)* self.subsidence[k]


            #Set some boundary conditions for smoothing
            temp_vadv_pp[:Gr.dims.gw] = temp_vadv_pp[Gr.dims.gw]
            shum_vadv_pp[:Gr.dims.gw] = shum_vadv_pp[Gr.dims.gw]


            #import pylab as plt
            ##plt.plot(rdr.get_interp_profile('omega', Gr.zp_half))
            #plt.plot(temp_vadv_pp*86400)
            #plt.show()

            # import pylab as plt
            from scipy.signal import savgol_filter
            #Smoothing flucation source terms is helpful because taking the vertical derivative of interpolated GCM
            #fields is noisy
            self.temp_dt_fluc = temp_vadv_pp #savgol_filter(temp_vadv_pp, 5, 3 )
            self.shum_dt_fluc = shum_vadv_pp  #savgol_filter(shum_vadv_pp, 5, 3 )


            #import pylab as plt
            #plt.plot(np.array(self.temp_dt_fluc) * 3600.0 * 24.0)
            #plt.show()
            #
            #import sys; sys.exit()


            #temp_dt_vadv = interp_pchip(Gr.zp_half, zfull, temp_dt_vadv)


            self.rho_gcm =  1.0 / rdr.get_interp_profile('alpha', Gr.zp)
            self.rho_half_gcm = 1.0 / rdr.get_interp_profile('alpha', Gr.zp_half)
            #self.v_dt_tot = interp_pchip(Gr.zp_half, zfull, v_dt_tot - v_dt_cof) * 0.0
            #self.u_dt_tot = interp_pchip(Gr.zp_half, zfull, u_dt_tot - u_dt_cof) * 0.0

            Pa.root_print('Finished updating time varying forcing')

            #temp = interp_pchip(Gr.zp, zfull, temp)


            #Now preform Galelian transformation
            #umean = Pa.HorizontalMean(Gr, &PV.values[u_shift])
            #vmean = Pa.HorizontalMean(Gr, &PV.values[v_shift])

            #u0_new = (np.max(umean) - np.min(umean))/2.0
            #v0_new = (np.max(vmean) - np.min(vmean))/2.0

            #with nogil:
            #    for i in xrange(0,Gr.dims.nlg[0]):
            #        ishift = i * istride
            #        for j in xrange(0,Gr.dims.nlg[1]):
            #            jshift = j * jstride
            #            for k in xrange(0,Gr.dims.nlg[2]):
            #                ijk = ishift + jshift + k
            #                PV.values[u_shift + ijk] -= (u0_new - Ref.u0)
            #                PV.values[v_shift + ijk] -= (v0_new - Ref.v0)

            #Ref.u0 = u0_new
            #Ref.v0 = v0_new

            #print "\t Ref.u0 = ", Ref.u0
            #print "\t Ref.v0 = ", Ref.v0


        #Apply Coriolis Forcing
        coriolis_force(&Gr.dims,&PV.values[u_shift],&PV.values[v_shift],&PV.tendencies[u_shift],
                       &PV.tendencies[v_shift],&self.ug[0], &self.vg[0],self.coriolis_param, Ref.u0, Ref.v0)
        
        cdef double [:] dtdt_pdv = np.zeros(Gr.dims.npg, dtype=np.double)
        # Apply Subsidence
        if 's' in PV.name_index:
            s_shift = PV.get_varshift(Gr, 's')
            apply_subsidence_temperature(&Gr.dims, &self.rho_gcm[0], &self.rho_half_gcm[0], &self.subsidence[0], &PV.values[qt_shift], &DV.values[t_shift], &PV.tendencies[s_shift])
            compute_pdv_work(&Gr.dims, &self.omega_vv[0], &Ref.p0_half[0], &DV.values[t_shift], &dtdt_pdv[0])
            #apply_subsidence_den(&Gr.dims, &Ref.rho0[0], &Ref.rho0_half[0], &self.subsidence[0], &PV.values[s_shift], &PV.tendencies[s_shift])
        else:
            thli_shift = PV.get_varshift(Gr, 'thli')
            apply_subsidence_temperature_thli(&Gr.dims, &self.rho_gcm[0], &Ref.p0_half[0], &self.rho_half_gcm[0], &self.subsidence[0], &PV.values[qt_shift], &DV.values[t_shift], &PV.tendencies[thli_shift])

        #print np.amax(dtdt_pdv)


        cdef double [:] qt_tend_tmp = np.zeros(Gr.dims.npg, dtype=np.double)
        #apply_subsidence(&Gr.dims, &self.rho_gcm[0], &self.rho_half_gcm[0], &self.subsidence[0], &PV.values[qt_shift], &qt_tend_tmp[0])
        apply_subsidence(&Gr.dims, &Ref.rho0[0], &Ref.rho0_half[0], &self.subsidence[0], &PV.values[qt_shift], &qt_tend_tmp[0])

        # Relaxation
        cdef double [:] xi_relax = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        if self.relax_scalar:
            with nogil:
                for k in xrange(Gr.dims.nlg[2]):
                    xi_relax[k] = 1.0 / self.tau_scalar
                    # Nudging rates
                    self.qt_tend_nudge[k] = -xi_relax[k] * (qt_mean[k] - self.shum[k])
                    self.t_tend_nudge[k]  = -xi_relax[k] * (t_mean[k] - self.temp[k])

        if 's' in PV.name_index:
            s_shift = PV.get_varshift(Gr, 's')
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

                            PV.tendencies[s_shift + ijk] += (cpm_c(qt) * (self.temp_dt_resid[k] + self.temp_dt_hadv[k] + self.temp_dt_fino[k]*0.0 + self.temp_dt_fluc[k] + dtdt_pdv[ijk] + self.t_tend_nudge[k]))/t
                            PV.tendencies[s_shift + ijk] += (sv_c(pv,t) - sd_c(pd,t)) * ( self.shum_dt_resid[k] + self.shum_dt_hadv[k]  + qt_tend_tmp[ijk]  + self.shum_dt_fluc[k] + self.qt_tend_nudge[k])
                            PV.tendencies[qt_shift + ijk] += (self.shum_dt_resid[k] + self.shum_dt_hadv[k] + qt_tend_tmp[ijk] + self.shum_dt_fluc[k] + self.qt_tend_nudge[k])
                            #PV.tendencies[u_shift + ijk] += self.u_dt_tot[k]
                            #PV.tendencies[v_shift + ijk] += self.v_dt_tot[k]
        else:
            thli_shift = PV.get_varshift(Gr, 'thli')
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
                            qv = qt - DV.values[ql_shift + ijk]
                            pd = pd_c(p0,qt,qv)
                            pv = pv_c(p0,qt,qv)
                            t  = DV.values[t_shift + ijk]

                            PV.tendencies[thli_shift + ijk] += (self.temp_dt_resid[k] + self.temp_dt_hadv[k] + self.temp_dt_fino[k])/exner_c(Ref.p0_half[k])
                            PV.tendencies[qt_shift + ijk] += (self.shum_dt_resid[k] + self.shum_dt_hadv[k] + qt_tend_tmp[ijk])
                            #PV.tendencies[u_shift + ijk] += self.u_dt_tot[k]
                            #PV.tendencies[v_shift + ijk] += self.v_dt_tot[k]


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

            #Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t ql_shift = DV.get_varshift(Gr,'ql')
            double pd, pv, qt, qv, p0, rho0, t
            double zmax, weight, weight_half

            double [:] qtmean = Pa.HorizontalMean(Gr, &PV.values[qt_shift])
            double [:] tmean = Pa.HorizontalMean(Gr, &DV.values[t_shift])

            double [:] tmp_tendency  = np.zeros((Gr.dims.npg),dtype=np.double,order='c')
            double [:] mean_tendency = np.empty((Gr.dims.nlg[2],),dtype=np.double,order='c')
            double [:] ls_dstd_hadv = np.empty((Gr.dims.nlg[2],),dtype=np.double,order='c')



        apply_subsidence_temperature(&Gr.dims,&self.rho_gcm[0],&self.rho_half_gcm[0],&self.subsidence[0],&PV.values[qt_shift], &DV.values[t_shift],
                         &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('ls_subs_dsdt', mean_tendency[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp_tendency[:] = 0.0
        apply_subsidence(&Gr.dims,&self.rho_gcm[0],&self.rho_half_gcm[0],&self.subsidence[0], &DV.values[t_shift],
                         &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('ls_subs_dtdt', mean_tendency[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp_tendency[:] = 0.0
        apply_subsidence(&Gr.dims,&self.rho_gcm[0],&self.rho_half_gcm[0],&self.subsidence[0], &PV.values[qt_shift],
                         &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('ls_subs_dqtdt', mean_tendency[Gr.dims.gw:-Gr.dims.gw], Pa)


        with nogil:
            for k in xrange(kmin, kmax):
                mean_tendency[k]  = mean_tendency[k] * tmean[k] / cpm_c(qtmean[k]) / Ref.rho0_half[k]


        with nogil:

            for i in xrange(Gr.dims.npg):
                tmp_tendency[i] = 0.0

            for i in xrange(gw,imax):
                ishift = i * istride
                for j in xrange(gw,jmax):
                    jshift = j * jstride
                    for k in xrange(gw,kmax):
                        ijk = ishift + jshift + k
                        p0 = Ref.p0_half[k]
                        rho0 = Ref.rho0_half[k]
                        qt = PV.values[qt_shift + ijk]
                        qv = qt - DV.values[ql_shift + ijk]
                        pd = pd_c(p0,qt,qv)
                        pv = pv_c(p0,qt,qv)
                        t  = DV.values[t_shift + ijk]
                        tmp_tendency[ijk] += (cpm_c(qt) * (self.temp_dt_hadv[k]) )/t
                        tmp_tendency[ijk] += (sv_c(pv,t) - sd_c(pd,t)) * (  self.shum_dt_hadv[k] )

        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])

        NS.write_profile('ls_subsidence', self.subsidence[Gr.dims.gw:-Gr.dims.gw],Pa)
        NS.write_profile('ls_dtdt_fino',self.temp_dt_fino[Gr.dims.gw:-Gr.dims.gw],Pa)
        NS.write_profile('ls_dtdt_fluc',self.temp_dt_fluc[Gr.dims.gw:-Gr.dims.gw],Pa)
        NS.write_profile('ls_dsdt_hadv', mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)
        NS.write_profile('ls_dtdt_hadv', self.temp_dt_hadv[Gr.dims.gw:-Gr.dims.gw],Pa)
        NS.write_profile('ls_dqtdt_hadv', self.shum_dt_hadv[Gr.dims.gw:-Gr.dims.gw],Pa)
        NS.write_profile('ls_dqtdt_fluc', self.shum_dt_fluc[Gr.dims.gw:-Gr.dims.gw], Pa)

        NS.write_profile('ls_dqtdt_resid', self.shum_dt_resid[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_profile('ls_dtdt_resid', self.temp_dt_resid[Gr.dims.gw:-Gr.dims.gw], Pa)

        NS.write_profile('dqtdt_nudge', self.qt_tend_nudge[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_profile('dtdt_nudge', self.t_tend_nudge[Gr.dims.gw:-Gr.dims.gw], Pa)

        return

cdef class ForcingGCMNew:
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
            self.instant_forcing = namelist['gcm']['instant_forcing']
        except:
            self.instant_forcing = False
        try:
            self.gcm_tidx = namelist['gcm']['gcm_tidx']
        except:
            self.gcm_tidx = 0
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
            self.add_advection = namelist['forcing']['add_advection']
        except:
            self.add_advection = False
        try:
            self.add_horiz_adv_vert_fluc = namelist['forcing']['add_horiz_adv_vert_fluc']
        except:
            self.add_horiz_adv_vert_fluc = False
        try:
            self.add_horiz_advection = namelist['forcing']['add_horiz_advection']
        except:
            self.add_horiz_advection = False
        try:
            self.add_vert_fluctuation = namelist['forcing']['add_vert_fluctuation']
        except:
            self.add_vert_fluctuation = False
        try:
            self.add_subsidence = namelist['forcing']['add_subsidence']
        except:
            self.add_subsidence = False
        try:
            self.add_subsidence_wind = namelist['forcing']['add_subsidence_wind']
        except:
            self.add_subsidence_wind = False
        if self.add_advection and (self.add_horiz_adv_vert_fluc or self.add_horiz_advection or self.add_vert_fluctuation or self.add_subsidence):
            Pa.root_print('ForcingGCMNew: Cannot specify both total advective tendency and different terms of advective tendency')
            Pa.kill()
        if self.add_horiz_adv_vert_fluc and (self.add_horiz_advection or self.add_vert_fluctuation):
            Pa.root_print('ForcingGCMNew: Cannot specify horizontal advection or vertical fluctuation when add_horiz_adv_vert_fluc is set to True')
            Pa.kill()
        try:
            self.coherent_variance = namelist['forcing']['coherent_variance']
        except:
            self.coherent_variance = True
        try:
            self.instant_variance = namelist['forcing']['instant_variance']
        except:
            self.instant_variance = False
        try:
            self.hadv_variance_factor = namelist['forcing']['hadv_variance_factor']
        except:
            self.hadv_variance_factor = 0.0
        try:
            self.sub_variance_factor = namelist['forcing']['sub_variance_factor']
        except:
            self.sub_variance_factor = 0.0
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
        self.qt_tend_hadv_tr = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.t_tend_hadv = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.t_tend_hadv_tr = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.qt_tend_fluc = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.t_tend_fluc = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.s_tend_hadv = np.zeros(Gr.dims.npg,dtype=np.double,order='c')
        self.s_tend_hadv_tr = np.zeros(Gr.dims.npg,dtype=np.double,order='c')
        self.subsidence = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.subsidence_tr = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.ug = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')
        self.vg = np.zeros(Gr.dims.nlg[2],dtype=np.double,order='c')

        NS.add_profile('ls_subsidence', Gr, Pa)
        NS.add_profile('ls_subsidence_tr', Gr, Pa)
        NS.add_profile('dqtdt_nudge', Gr, Pa)
        NS.add_profile('dtdt_nudge', Gr, Pa)
        NS.add_profile('dudt_nudge', Gr, Pa)
        NS.add_profile('dvdt_nudge', Gr, Pa)
        NS.add_profile('dqtdt_adv', Gr, Pa)
        NS.add_profile('dtdt_adv', Gr, Pa)
        NS.add_profile('dsdt_adv', Gr, Pa)
        NS.add_profile('dqtdt_hadv', Gr, Pa)
        NS.add_profile('dqtdt_hadv_tr', Gr, Pa)
        NS.add_profile('dtdt_hadv', Gr, Pa)
        NS.add_profile('dtdt_hadv_tr', Gr, Pa)
        NS.add_profile('dsdt_hadv', Gr, Pa)
        NS.add_profile('dsdt_hadv_tr', Gr, Pa)
        NS.add_profile('dqtdt_fluc', Gr, Pa)
        NS.add_profile('dtdt_fluc', Gr, Pa)
        NS.add_profile('dqtdt_sub', Gr, Pa)
        NS.add_profile('dqtdt_sub_tr', Gr, Pa)
        NS.add_profile('dtdt_sub', Gr, Pa)
        NS.add_profile('dtdt_sub_tr', Gr, Pa)
        NS.add_profile('dudt_sub', Gr, Pa)
        NS.add_profile('dudt_sub_tr', Gr, Pa)
        NS.add_profile('dvdt_sub', Gr, Pa)
        NS.add_profile('dvdt_sub_tr', Gr, Pa)
        NS.add_profile('dudt_cor', Gr, Pa)
        NS.add_profile('dvdt_cor', Gr, Pa)

        np.random.seed(Pa.rank)

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
            Py_ssize_t qi_shift = DV.get_varshift(Gr, 'qi') 
            double pd, pv, qt, qv, p0, rho0, t
            double zmax, weight, weight_half
            double u0_new, v0_new
            double sub_factor, hadv_factor
            double [:] qt_mean = Pa.HorizontalMean(Gr, &PV.values[qt_shift])
            double [:] t_mean = Pa.HorizontalMean(Gr, &DV.values[t_shift])
            double [:] u_mean = Pa.HorizontalMean(Gr, &PV.values[u_shift])
            double [:] v_mean = Pa.HorizontalMean(Gr, &PV.values[v_shift])

        if not self.gcm_profiles_initialized:
            self.gcm_profiles_initialized = True
            Pa.root_print('Updating forcing')

            if self.griddata:
                rdr = cfreader_grid(self.file, self.lat, self.lon)
            else:
                rdr = cfreader(self.file, self.site)
                self.lat = rdr.get_value('lat')
            Pa.root_print(self.lat)
            self.coriolis_param = 2.0 * omega * sin(self.lat * pi / 180.0)
            self.temp = rdr.get_interp_profile_old('ta', Gr.zp_half, instant=self.instant_forcing, t_idx=self.gcm_tidx)
            self.sphum = rdr.get_interp_profile_old('hus', Gr.zp_half, instant=self.instant_forcing, t_idx=self.gcm_tidx)
            self.ucomp = rdr.get_interp_profile_old('ua', Gr.zp_half, instant=self.instant_forcing, t_idx=self.gcm_tidx)
            self.vcomp = rdr.get_interp_profile_old('va', Gr.zp_half, instant=self.instant_forcing, t_idx=self.gcm_tidx)
            temp_at_zp = rdr.get_interp_profile('ta', Gr.zp, instant=self.instant_forcing, t_idx=self.gcm_tidx)
            sphum_at_zp = rdr.get_interp_profile('hus', Gr.zp, instant=self.instant_forcing, t_idx=self.gcm_tidx)
            if self.add_ls_pgradient:
                self.ug = rdr.get_interp_profile_old('u_geos', Gr.zp_half, instant=self.instant_forcing, t_idx=self.gcm_tidx)
                self.vg = rdr.get_interp_profile_old('v_geos', Gr.zp_half, instant=self.instant_forcing, t_idx=self.gcm_tidx)
            if self.add_advection:
                self.t_tend_adv = rdr.get_interp_profile_old('tnta', Gr.zp_half, instant=self.instant_forcing, t_idx=self.gcm_tidx)
                self.qt_tend_adv = rdr.get_interp_profile_old('tnhusa',Gr.zp_half, instant=self.instant_forcing, t_idx=self.gcm_tidx)
            if self.add_horiz_adv_vert_fluc:
                tnta = rdr.get_interp_profile_old('tnta', Gr.zp_half, instant=self.instant_forcing, t_idx=self.gcm_tidx)
                tnhusa = rdr.get_interp_profile_old('tnhusa',Gr.zp_half, instant=self.instant_forcing, t_idx=self.gcm_tidx)
                tntwork =  rdr.get_interp_profile_old('tntwork', Gr.zp_half, instant=self.instant_forcing, t_idx=self.gcm_tidx)
                omega_vv = rdr.get_interp_profile_old('wap', Gr.zp_half, instant=self.instant_forcing, t_idx=self.gcm_tidx)
                alpha = rdr.get_interp_profile_old('alpha', Gr.zp_half, instant=self.instant_forcing, t_idx=self.gcm_tidx)
                subsidence = -np.array(omega_vv) * alpha / g
                temp_hadv_fluc = np.zeros(np.shape(tnta))
                sphum_hadv_fluc = np.zeros(np.shape(tnta))
                for k in xrange(temp_at_zp.shape[0]-1):
                    temp_hadv_fluc[k] = tnta[k] - tntwork[k] + ((temp_at_zp[k+1]-temp_at_zp[k]) * Gr.dims.dxi[2] * Gr.dims.imetl_half[k]) * subsidence[k]
                    sphum_hadv_fluc[k] = tnhusa[k] + ((sphum_at_zp[k+1]-sphum_at_zp[k]) * Gr.dims.dxi[2] * Gr.dims.imetl_half[k]) * subsidence[k]
                self.t_tend_hadv = temp_hadv_fluc #hadv includes vertical fluctuation
                self.qt_tend_hadv = sphum_hadv_fluc
            if self.add_horiz_advection:
                self.t_tend_hadv = rdr.get_interp_profile_old('tntha', Gr.zp_half, instant=self.instant_forcing, t_idx=self.gcm_tidx)
                self.qt_tend_hadv = rdr.get_interp_profile_old('tnhusha', Gr.zp_half, instant=self.instant_forcing, t_idx=self.gcm_tidx)
            if self.add_vert_fluctuation:
                tntva = rdr.get_interp_profile_old('tntva', Gr.zp_half, instant=self.instant_forcing, t_idx=self.gcm_tidx)
                tnhusva = rdr.get_interp_profile_old('tnhusva',Gr.zp_half, instant=self.instant_forcing, t_idx=self.gcm_tidx)
                omega_vv = rdr.get_interp_profile_old('wap', Gr.zp_half, instant=self.instant_forcing, t_idx=self.gcm_tidx)
                alpha = rdr.get_interp_profile_old('alpha', Gr.zp_half, instant=self.instant_forcing, t_idx=self.gcm_tidx)
                subsidence = -np.array(omega_vv) * alpha / g
                temp_fluc = np.zeros(np.shape(tntva))
                sphum_fluc = np.zeros(np.shape(tntva))
                for k in xrange(temp_at_zp.shape[0]-1):
                    temp_fluc[k] = tntva[k] + ((temp_at_zp[k+1]-temp_at_zp[k]) * Gr.dims.dxi[2] * Gr.dims.imetl_half[k]) * subsidence[k]
                    sphum_fluc[k] = tnhusva[k] + ((sphum_at_zp[k+1]-sphum_at_zp[k]) * Gr.dims.dxi[2] * Gr.dims.imetl_half[k]) * subsidence[k]
                self.t_tend_fluc = temp_fluc
                self.qt_tend_fluc = sphum_fluc
            if self.add_subsidence or self.add_subsidence_wind:
                self.omega_vv = rdr.get_interp_profile_old('wap', Gr.zp_half, instant=self.instant_forcing, t_idx=self.gcm_tidx)
                alpha = rdr.get_interp_profile_old('alpha', Gr.zp_half, instant=self.instant_forcing, t_idx=self.gcm_tidx)
                self.subsidence = -np.array(self.omega_vv) * alpha / g
            Pa.root_print('Finished updating forcing')
     
        if int(TS.t // (3600.0 * 6.0)) > self.t_indx and (not self.instant_variance):
            self.t_indx = int(TS.t // (3600.0 * 6.0))
            Pa.root_print('Add stochastic forcing') 
            if self.coherent_variance:
                sub_factor = np.random.normal(0.0, self.sub_variance_factor)
                hadv_factor = np.random.normal(0.0, self.hadv_variance_factor)       
                for k in xrange(Gr.dims.nlg[2]):
                    self.subsidence_tr[k] = sub_factor*self.subsidence[k]
                    self.qt_tend_hadv_tr[k] = hadv_factor*self.qt_tend_hadv[k]
                    self.t_tend_hadv_tr[k] = hadv_factor*self.t_tend_hadv[k]
            else:
                for k in xrange(Gr.dims.nlg[2]):
                    self.subsidence_tr[k] = np.random.normal(0.0, self.sub_variance_factor*np.abs(self.subsidence[k]))
                    self.qt_tend_hadv_tr[k] = np.random.normal(0.0, self.hadv_variance_factor*np.abs(self.qt_tend_hadv[k]))
                    self.t_tend_hadv_tr[k] = np.random.normal(0.0, self.hadv_variance_factor*np.abs(self.t_tend_hadv[k]))
     
        # stochastic forcing
        if self.instant_variance:
            if self.coherent_variance:
                sub_factor = np.random.normal(0.0, self.sub_variance_factor)
                hadv_factor = np.random.normal(0.0, self.hadv_variance_factor)       
                for k in xrange(Gr.dims.nlg[2]):
                    self.subsidence_tr[k] = sub_factor*self.subsidence[k]
                    self.qt_tend_hadv_tr[k] = hadv_factor*self.qt_tend_hadv[k]
                    self.t_tend_hadv_tr[k] = hadv_factor*self.t_tend_hadv[k]
            else:
                for k in xrange(Gr.dims.nlg[2]):
                    self.subsidence_tr[k] = np.random.normal(0.0, self.sub_variance_factor*np.abs(self.subsidence[k]))
                    self.qt_tend_hadv_tr[k] = np.random.normal(0.0, self.hadv_variance_factor*np.abs(self.qt_tend_hadv[k]))
                    self.t_tend_hadv_tr[k] = np.random.normal(0.0, self.hadv_variance_factor*np.abs(self.t_tend_hadv[k]))

        # Apply Coriolis forcing
        if self.add_coriolis:
            coriolis_force(&Gr.dims, &PV.values[u_shift], &PV.values[v_shift], &PV.tendencies[u_shift],
                           &PV.tendencies[v_shift], &self.ug[0], &self.vg[0], self.coriolis_param, Ref.u0, Ref.v0)
       
        # Apply subsidence
        if self.add_subsidence:
            apply_subsidence(&Gr.dims, &Ref.rho0[0], &Ref.rho0_half[0], &self.subsidence[0], &PV.values[s_shift], &PV.tendencies[s_shift])
            apply_subsidence(&Gr.dims, &Ref.rho0[0], &Ref.rho0_half[0], &self.subsidence_tr[0], &PV.values[s_shift], &PV.tendencies[s_shift])
            apply_subsidence(&Gr.dims, &Ref.rho0[0], &Ref.rho0_half[0], &self.subsidence[0], &PV.values[qt_shift], &PV.tendencies[qt_shift])
            apply_subsidence(&Gr.dims, &Ref.rho0[0], &Ref.rho0_half[0], &self.subsidence_tr[0], &PV.values[qt_shift], &PV.tendencies[qt_shift])
        if self.add_subsidence_wind:
            apply_subsidence(&Gr.dims, &Ref.rho0[0], &Ref.rho0_half[0], &self.subsidence[0], &PV.values[u_shift], &PV.tendencies[u_shift])
            apply_subsidence(&Gr.dims, &Ref.rho0[0], &Ref.rho0_half[0], &self.subsidence_tr[0], &PV.values[u_shift], &PV.tendencies[u_shift])
            apply_subsidence(&Gr.dims, &Ref.rho0[0], &Ref.rho0_half[0], &self.subsidence[0], &PV.values[v_shift], &PV.tendencies[v_shift])
            apply_subsidence(&Gr.dims, &Ref.rho0[0], &Ref.rho0_half[0], &self.subsidence_tr[0], &PV.values[v_shift], &PV.tendencies[v_shift])
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
                        
                        total_t_source = self.t_tend_hadv[k] + self.t_tend_hadv_tr[k] + self.t_tend_fluc[k] + self.t_tend_adv[k] + self.t_tend_nudge[k]
                        total_qt_source = self.qt_tend_hadv[k] + self.qt_tend_hadv_tr[k] + self.qt_tend_fluc[k] + self.qt_tend_adv[k] + self.qt_tend_nudge[k] 
                        PV.tendencies[s_shift + ijk] += s_tendency_c(p0, qt, qv, t, total_qt_source, total_t_source)
                        PV.tendencies[qt_shift + ijk] += total_qt_source
                        #PV.tendencies[s_shift + ijk] += (cpm_c(qt) * (self.t_tend_adv[k]+self.t_tend_nudge[k]))/t
                        #PV.tendencies[s_shift + ijk] += (sv_c(pv,t) - sd_c(pd,t)) * (self.qt_tend_adv[k]+self.qt_tend_nudge[k])
                        #PV.tendencies[qt_shift + ijk] += (self.qt_tend_adv[k]+self.qt_tend_nudge[k])
                        PV.tendencies[u_shift + ijk] += self.u_tend_nudge[k]
                        PV.tendencies[v_shift + ijk] += self.v_tend_nudge[k]
                        self.s_tend_adv[ijk]= s_tendency_c(p0,qt, qv, t, self.qt_tend_adv[k], self.t_tend_adv[k])
                        self.s_tend_hadv[ijk]= s_tendency_c(p0,qt, qv, t, self.qt_tend_hadv[k], self.t_tend_hadv[k])
                        self.s_tend_hadv_tr[ijk]= s_tendency_c(p0,qt, qv, t, self.qt_tend_hadv_tr[k], self.t_tend_hadv_tr[k])

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
            apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence_tr[0], &PV.values[qt_shift],
                             &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('dqtdt_sub_tr', mean_tendency[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp_tendency[:] = 0.0
        if self.add_subsidence:
            apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0], &DV.values[t_shift],
                             &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('dtdt_sub', mean_tendency[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp_tendency[:] = 0.0
        if self.add_subsidence:
            apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence_tr[0], &DV.values[t_shift],
                             &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('dtdt_sub_tr', mean_tendency[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp_tendency[:] = 0.0

        if self.add_subsidence_wind:
            apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0], &PV.values[u_shift],
                             &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('dudt_sub', mean_tendency[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp_tendency[:] = 0.0
        if self.add_subsidence_wind:
            apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence_tr[0], &PV.values[u_shift],
                             &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('dudt_sub_tr', mean_tendency[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp_tendency[:] = 0.0
        if self.add_subsidence_wind:
            apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence[0], &PV.values[v_shift],
                             &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('dvdt_sub', mean_tendency[Gr.dims.gw:-Gr.dims.gw], Pa)
        tmp_tendency[:] = 0.0
        if self.add_subsidence_wind:
            apply_subsidence(&Gr.dims,&Ref.rho0[0],&Ref.rho0_half[0],&self.subsidence_tr[0], &PV.values[v_shift],
                             &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('dvdt_sub_tr', mean_tendency[Gr.dims.gw:-Gr.dims.gw], Pa)
        
        NS.write_profile('ls_subsidence', self.subsidence[Gr.dims.gw:-Gr.dims.gw],Pa)
        NS.write_profile('ls_subsidence_tr', self.subsidence_tr[Gr.dims.gw:-Gr.dims.gw],Pa)
        NS.write_profile('dqtdt_nudge', self.qt_tend_nudge[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_profile('dtdt_nudge', self.t_tend_nudge[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_profile('dudt_nudge', self.u_tend_nudge[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_profile('dvdt_nudge', self.v_tend_nudge[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_profile('dqtdt_adv', self.qt_tend_adv[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_profile('dtdt_adv', self.t_tend_adv[Gr.dims.gw:-Gr.dims.gw], Pa)
        mean_tendency = Pa.HorizontalMean(Gr,&self.s_tend_adv[0])
        NS.write_profile('dsdt_adv',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)
        NS.write_profile('dqtdt_hadv', self.qt_tend_hadv[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_profile('dqtdt_hadv_tr', self.qt_tend_hadv_tr[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_profile('dtdt_hadv', self.t_tend_hadv[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_profile('dtdt_hadv_tr', self.t_tend_hadv_tr[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_profile('dqtdt_fluc', self.qt_tend_fluc[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_profile('dtdt_fluc', self.t_tend_fluc[Gr.dims.gw:-Gr.dims.gw], Pa)
        mean_tendency = Pa.HorizontalMean(Gr,&self.s_tend_hadv[0])
        NS.write_profile('dsdt_hadv',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)
        mean_tendency = Pa.HorizontalMean(Gr,&self.s_tend_hadv_tr[0])
        NS.write_profile('dsdt_hadv_tr',mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)

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

@cython.wraparound(False)
@cython.boundscheck(False)
cdef apply_subsidence_den(Grid.DimStruct *dims, double *rho0, double *rho0_half, double *subsidence, double* values,  double *tendencies):

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
                        tend = (values[ijk+1]*rho0_half[k+1] - values[ijk]*rho0_half[k]) * dxi * subsidence[k] * dims.imetl[k] / rho0_half[k]
                    else:
                        tend = (values[ijk]*rho0_half[k] - values[ijk-1]*rho0_half[k-1]) * dxi * subsidence[k] * dims.imetl[k-1]/rho0_half[k]
                    tendencies[ijk] -= tend
                for k in xrange(kmax, dims.nlg[2]):
                    ijk = ishift + jshift + k
                    tendencies[ijk] -= tend

    return

cdef compute_pdv_work(Grid.DimStruct *dims, double *omega_vv, double *p0_half, double *T, double* tendency):
    cdef:
        Py_ssize_t imin = dims.gw
        Py_ssize_t jmin = dims.gw
        Py_ssize_t kmin = dims.gw
        Py_ssize_t imax = dims.nlg[0] -dims.gw
        Py_ssize_t jmax = dims.nlg[1] -dims.gw
        Py_ssize_t kmax = dims.nlg[2] -dims.gw
        Py_ssize_t istride = dims.nlg[1] * dims.nlg[2]
        size_t jstride = dims.nlg[2]
        Py_ssize_t ishift, jshift, ijk, i,j,k

    with nogil:
        for i in xrange(imin,imax):
            ishift = i*istride
            for j in xrange(jmin,jmax):
                jshift = j*jstride
                for k in xrange(kmin,kmax):
                    ijk = ishift + jshift + k
                    tendency[ijk] = omega_vv[k] * (Rd * T[ijk])/(p0_half[k] * cpd)
                    #with gil:
                    #    print tendency[ijk], omega_vv[ijk], T[ijk], p0_half[k]
                    #    if k == kmax-1:
                    #        import sys; sys.exit()

    return


cdef apply_subsidence_temperature_thli(Grid.DimStruct *dims, double *rho0, double *p0_half, double *rho0_half, double *subsidence, double *qt, double* values,  double *tendencies):

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
                        tend = (values[ijk+1] - values[ijk]) * dxi * subsidence[k] * dims.imetl[k]
                    else:
                        tend = (values[ijk] - values[ijk-1]) * dxi * subsidence[k] * dims.imetl[k-1]
                    #+ g / values[ijk] * subsidence[k]
                    tendencies[ijk] -= tend / exner_c(p0_half[k])
                for k in xrange(kmax, dims.nlg[2]):
                    ijk = ishift + jshift + k
                    tendencies[ijk] -= tend / exner_c(p0_half[k])

    return

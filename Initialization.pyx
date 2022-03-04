#!python
#cython: boundscheck=False
#cython: wraparound=True
#cython: initializedcheck=False
#cython: cdivision=True

import sys
import netCDF4 as nc
import numpy as np
cimport numpy as np
from scipy.interpolate import PchipInterpolator,pchip_interpolate
cimport ParallelMPI
from NetCDFIO cimport NetCDFIO_Stats
cimport Grid
cimport PrognosticVariables
cimport DiagnosticVariables
from thermodynamic_functions cimport exner_c, entropy_from_thetas_c, thetas_t_c, qv_star_c, thetas_c, thetali_c
cimport ReferenceState
cimport Restart
from Forcing cimport AdjustedMoistAdiabat
from Thermodynamics cimport LatentHeat
from libc.math cimport sqrt, fmin, cos, exp, fabs
include 'parameters.pxi'
import pickle as pickle
from scipy import interpolate
from cfsites_forcing_reader import cfreader
from cfgrid_forcing_reader import cfreader_grid

def InitializationFactory(namelist):

        casename = namelist['meta']['casename']
        if casename == 'SullivanPatton':
            return InitSullivanPatton
        elif casename == 'StableBubble':
            return InitStableBubble
        elif casename == 'SaturatedBubble':
            return InitSaturatedBubble
        elif casename == 'Bomex' or casename == 'lifecycle_Tan2018':
            return InitBomex
        elif casename == 'Soares':
            return InitSoares
        elif casename == 'Soares_moist':
            return InitSoares_moist
        elif casename == 'Gabls':
            return InitGabls
        elif casename == 'DYCOMS_RF01':
            return InitDYCOMS_RF01
        elif casename == 'DYCOMS_RF02':
            return InitDYCOMS_RF02
        elif casename == 'SMOKE':
            return InitSmoke
        elif casename == 'Rico':
            return InitRico
        elif casename == 'CGILS':
            return  InitCGILS
        elif casename == 'ZGILS':
            return  InitZGILS
        elif casename == 'TRMM_LBA':
            return  InitTRMM_LBA
        elif casename == 'ARM_SGP':
            return  InitARM_SGP
        elif casename == 'GATE_III':
            return  InitGATE_III
        elif casename == 'WANGARA':
            return  InitWANGARA
        elif casename == 'GCMVarying':
            return InitGCMVarying
        elif casename == 'GCMNew':
            return InitGCMNew
        else:
            pass

def InitStableBubble(namelist, Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, LatentHeat LH):

    #Generate reference profiles
    RS.Pg = 1.0e5
    RS.Tg = 300.0
    RS.qtg = 0.0
    #Set velocities for Galilean transformation
    RS.u0 = 0.0
    RS.v0 = 0.0

    RS.initialize(Gr, Th, NS, Pa)

    #Get the variable number for each of the velocity components
    cdef:
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t i,j,k
        Py_ssize_t ishift, jshift
        Py_ssize_t ijk
        double t
        double dist

    t_min = 9999.9
    for i in xrange(Gr.dims.nlg[0]):
        ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
        for j in xrange(Gr.dims.nlg[1]):
            jshift = j * Gr.dims.nlg[2]
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                PV.values[u_varshift + ijk] = 0.0
                PV.values[v_varshift + ijk] = 0.0
                PV.values[w_varshift + ijk] = 0.0
                dist  = np.sqrt(((Gr.x_half[i + Gr.dims.indx_lo[0]]/1000.0 - 25.6)/4.0)**2.0 + ((Gr.zp_half[k + Gr.dims.indx_lo[2]]/1000.0 - 3.0)/2.0)**2.0)
                dist = fmin(dist,1.0)
                t = (300.0 )*exner_c(RS.p0_half[k]) - 15.0*( cos(np.pi * dist) + 1.0) /2.0
                PV.values[s_varshift + ijk] = Th.entropy(RS.p0_half[k],t,0.0,0.0,0.0)


    return

def InitSaturatedBubble(namelist,Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, LatentHeat LH ):

    #Generate reference profiles
    RS.Pg = 1.0e5
    RS.qtg = 0.02
    #RS.Tg = 300.0

    thetas_sfc = 320.0
    qt_sfc = 0.0196 #RS.qtg
    RS.qtg = qt_sfc

    #Set velocities for Galilean transformation
    RS.u0 = 0.0
    RS.v0 = 0.0

    def theta_to_T(p0_,thetas_,qt_):


         T1 = Tt
         T2 = Tt + 1.

         pv1 = Th.get_pv_star(T1)
         pv2 = Th.get_pv_star(T2)

         qs1 = qv_star_c(p0_, RS.qtg,pv1)

         ql1 = np.max([0.0,qt_ - qs1])
         L1 = Th.get_lh(T1)
         f1 = thetas_ - thetas_t_c(p0_,T1,qt_,qt_-ql1,ql1,L1)

         delta = np.abs(T1 - T2)
         while delta >= 1e-12:


            L2 = Th.get_lh(T2)
            pv2 = Th.get_pv_star(T2)
            qs2 = qv_star_c(p0_, RS.qtg, pv2)
            ql2 = np.max([0.0,qt_ - qs2])
            f2 = thetas_ - thetas_t_c(p0_,T2,qt_,qt_-ql2,ql2,L2)

            Tnew = T2 - f2 * (T2 - T1)/(f2 - f1)
            T1 = T2
            T2 = Tnew
            f1 = f2

            delta = np.abs(T1 - T2)
         return T2, ql2

    RS.Tg, ql = theta_to_T(RS.Pg,thetas_sfc,qt_sfc)
    RS.initialize(Gr, Th, NS, Pa)

    #Get the variable number for each of the velocity components
    cdef:
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t qt_varshift = PV.get_varshift(Gr,'qt')
        Py_ssize_t i,j,k
        Py_ssize_t ishift, jshift
        Py_ssize_t ijk
        double t
        double dist
        double thetas

    for i in xrange(Gr.dims.nlg[0]):
        ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
        for j in xrange(Gr.dims.nlg[1]):
            jshift = j * Gr.dims.nlg[2]
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                dist = np.sqrt(((Gr.x_half[i + Gr.dims.indx_lo[0]]/1000.0 - 10.0)/2.0)**2.0 + ((Gr.zp_half[k + Gr.dims.indx_lo[2]]/1000.0 - 2.0)/2.0)**2.0)
                dist = np.minimum(1.0,dist)
                thetas = RS.Tg
                thetas += 2.0 * np.cos(np.pi * dist / 2.0)**2.0
                PV.values[s_varshift + ijk] = entropy_from_thetas_c(thetas,RS.qtg)
                PV.values[u_varshift + ijk] = 0.0 - RS.u0
                PV.values[v_varshift + ijk] = 0.0 - RS.v0
                PV.values[w_varshift + ijk] = 0.0
                PV.values[qt_varshift + ijk] = RS.qtg

    return

def InitSullivanPatton(namelist,Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, LatentHeat LH ):

    #Generate the reference profiles
    RS.Pg = 1.0e5  #Pressure at ground
    RS.Tg = 300.0  #Temperature at ground
    RS.qtg = 0.0   #Total water mixing ratio at surface
    RS.u0 = 1.0  # velocities removed in Galilean transformation
    RS.v0 = 0.0

    RS.initialize(Gr, Th, NS, Pa)
    try:
        random_seed_factor = namelist['initialization']['random_seed_factor']
    except:
        random_seed_factor = 1

    try:
        random_seed_factor = namelist['initialization']['random_seed_factor']
    except:
        random_seed_factor = 1
    #Get the variable number for each of the velocity components
    np.random.seed(Pa.rank * random_seed_factor)
    cdef:
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t i,j,k
        Py_ssize_t ishift, jshift, e_varshift
        Py_ssize_t ijk
        double [:] theta = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')
        double t

        #Generate initial perturbations (here we are generating more than we need)
        cdef double [:] theta_pert = np.random.random_sample(Gr.dims.npg)
        cdef double theta_pert_

    for k in xrange(Gr.dims.nlg[2]):
        if Gr.zp_half[k] <=  974.0:
            theta[k] = 300.0
        elif Gr.zp_half[k] <= 1074.0:
            theta[k] = 300.0 + (Gr.zp_half[k] - 974.0) * 0.08
        else:
            theta[k] = 308.0 + (Gr.zp_half[k] - 1074.0) * 0.003

    cdef double [:] p0 = RS.p0_half

    #Now loop and set the initial condition
    for i in xrange(Gr.dims.nlg[0]):
        ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
        for j in xrange(Gr.dims.nlg[1]):
            jshift = j * Gr.dims.nlg[2]
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                PV.values[u_varshift + ijk] = 1.0 - RS.u0
                PV.values[v_varshift + ijk] = 0.0 - RS.v0
                PV.values[w_varshift + ijk] = 0.0

                #Now set the entropy prognostic variable including a potential temperature perturbation
                if Gr.zp_half[k] < 200.0:
                    theta_pert_ = (theta_pert[ijk] - 0.5)* 0.1
                else:
                    theta_pert_ = 0.0
                t = (theta[k] + theta_pert_)*exner_c(RS.p0_half[k])

                PV.values[s_varshift + ijk] = Th.entropy(RS.p0_half[k],t,0.0,0.0,0.0)
    if 'e' in PV.name_index:
        e_varshift = PV.get_varshift(Gr, 'e')
        for i in xrange(Gr.dims.nlg[0]):
            ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
            for j in xrange(Gr.dims.nlg[1]):
                jshift = j * Gr.dims.nlg[2]
                for k in xrange(Gr.dims.nlg[2]):
                    ijk = ishift + jshift + k
                    PV.values[e_varshift + ijk] = 0.0
    return

def InitBomex(namelist,Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, LatentHeat LH ):

    #First generate the reference profiles
    RS.Pg = 1.015e5  #Pressure at ground
    RS.Tg = 300.4  #Temperature at ground
    RS.qtg = 0.02245   #Total water mixing ratio at surface

    RS.initialize(Gr, Th, NS, Pa)

    try:
        random_seed_factor = namelist['initialization']['random_seed_factor']
    except:
        random_seed_factor = 1

    np.random.seed(Pa.rank * random_seed_factor)

    #Get the variable number for each of the velocity components

    cdef:
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t qt_varshift = PV.get_varshift(Gr,'qt')
        Py_ssize_t i,j,k
        Py_ssize_t ishift, jshift
        Py_ssize_t ijk, e_varshift
        double temp
        double qt_
        double [:] thetal = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')
        double [:] qt = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')
        double [:] u = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')
        Py_ssize_t count

        theta_pert = (np.random.random_sample(Gr.dims.npg )-0.5)*0.1
        qt_pert = (np.random.random_sample(Gr.dims.npg )-0.5)*0.025/1000.0

    for k in xrange(Gr.dims.nlg[2]):

        #Set Thetal profile
        if Gr.zp_half[k] <= 520.:
            thetal[k] = 298.7
        if Gr.zp_half[k] > 520.0 and Gr.zp_half[k] <= 1480.0:
            thetal[k] = 298.7 + (Gr.zp_half[k] - 520)  * (302.4 - 298.7)/(1480.0 - 520.0)
        if Gr.zp_half[k] > 1480.0 and Gr.zp_half[k] <= 2000:
            thetal[k] = 302.4 + (Gr.zp_half[k] - 1480.0) * (308.2 - 302.4)/(2000.0 - 1480.0)
        if Gr.zp_half[k] > 2000.0:
            thetal[k] = 308.2 + (Gr.zp_half[k] - 2000.0) * (311.85 - 308.2)/(3000.0 - 2000.0)

        #Set qt profile
        if Gr.zp_half[k] <= 520:
            qt[k] = 17.0 + (Gr.zp_half[k]) * (16.3-17.0)/520.0
        if Gr.zp_half[k] > 520.0 and Gr.zp_half[k] <= 1480.0:
            qt[k] = 16.3 + (Gr.zp_half[k] - 520.0)*(10.7 - 16.3)/(1480.0 - 520.0)
        if Gr.zp_half[k] > 1480.0 and Gr.zp_half[k] <= 2000.0:
            qt[k] = 10.7 + (Gr.zp_half[k] - 1480.0) * (4.2 - 10.7)/(2000.0 - 1480.0)
        if Gr.zp_half[k] > 2000.0:
            qt[k] = 4.2 + (Gr.zp_half[k] - 2000.0) * (3.0 - 4.2)/(3000.0  - 2000.0)

        #Change units to kg/kg
        qt[k]/= 1000.0

        #Set u profile
        if Gr.zp_half[k] <= 700.0:
            u[k] = -8.75
        if Gr.zp_half[k] > 700.0:
            u[k] = -8.75 + (Gr.zp_half[k] - 700.0) * (-4.61 - -8.75)/(3000.0 - 700.0)

    #Set velocities for Galilean transformation
    RS.v0 = 0.0
    RS.u0 = 0.5 * (np.amax(u)+np.amin(u))

    #Now loop and set the initial condition
    #First set the velocities
    count = 0
    for i in xrange(Gr.dims.nlg[0]):
        ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
        for j in xrange(Gr.dims.nlg[1]):
            jshift = j * Gr.dims.nlg[2]
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                PV.values[u_varshift + ijk] = u[k] - RS.u0
                PV.values[v_varshift + ijk] = 0.0 - RS.v0
                PV.values[w_varshift + ijk] = 0.0
                if Gr.zp_half[k] <= 1600.0:
                    temp = (thetal[k] + (theta_pert[count])) * exner_c(RS.p0_half[k])
                    qt_ = qt[k]+qt_pert[count]
                else:
                    temp = (thetal[k]) * exner_c(RS.p0_half[k])
                    qt_ = qt[k]
                PV.values[s_varshift + ijk] = Th.entropy(RS.p0_half[k],temp,qt_,0.0,0.0)
                PV.values[qt_varshift + ijk] = qt_
                count += 1

    if 'e' in PV.name_index:
        e_varshift = PV.get_varshift(Gr, 'e')
        for i in xrange(Gr.dims.nlg[0]):
            ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
            for j in xrange(Gr.dims.nlg[1]):
                jshift = j * Gr.dims.nlg[2]
                for k in xrange(Gr.dims.nlg[2]):
                    ijk = ishift + jshift + k
                    PV.values[e_varshift + ijk] = 1.0-Gr.zp_half[k]/3000.0


    return

def InitSoares(namelist,Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, LatentHeat LH ):

    #First generate the reference profiles
    RS.Pg = 1000.0 * 100.0  #Pressure at ground
    RS.Tg = 300.0  #Temperature at ground
    RS.qtg = 0.0 #This was set to 4.5e-3 earlier, but Soares 2004 sets 5e-3   #Total water mixing ratio at surface

    RS.initialize(Gr, Th, NS, Pa)

    try:
        random_seed_factor = namelist['initialization']['random_seed_factor']
    except:
        random_seed_factor = 1

    np.random.seed(Pa.rank * random_seed_factor)

    #Get the variable number for each of the velocity components

    cdef:
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t i,j,k
        Py_ssize_t ishift, jshift
        Py_ssize_t ijk, e_varshift
        double temp
        double [:] thetal = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')
        double [:] qt = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')
        double [:] u = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')
        Py_ssize_t count

        theta_pert = (np.random.random_sample(Gr.dims.npg )-0.5)*0.1 # Yair check what is the correct perturbation

    for k in xrange(Gr.dims.nlg[2]):

        #Set Thetal and qt profile
        if Gr.zp_half[k] <= 1350.0:
            thetal[k] = 300.0
        else:
            thetal[k] = 300.0 + 3.0 * (Gr.zp_half[k]-1350.0)/1000.0
        #Set u profile
        u[k] = 0.01


    #Set velocities for Galilean transformation
    RS.v0 = 0.0
    RS.u0 = 0.5 * (np.amax(u)+np.amin(u))

    #Now loop and set the initial condition
    #First set the velocities
    count = 0
    for i in xrange(Gr.dims.nlg[0]):
        ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
        for j in xrange(Gr.dims.nlg[1]):
            jshift = j * Gr.dims.nlg[2]
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                PV.values[u_varshift + ijk] = u[k] - RS.u0
                PV.values[v_varshift + ijk] = 0.0 - RS.v0
                PV.values[w_varshift + ijk] = 0.0
                if Gr.zp_half[k] <= 1600.0:
                    temp = (thetal[k] + (theta_pert[count])) * exner_c(RS.p0_half[k])
                else:
                    temp = (thetal[k]) * exner_c(RS.p0_half[k])
                PV.values[s_varshift + ijk] = Th.entropy(RS.p0_half[k],temp,0.0,0.0,0.0)
                count += 1

    if 'e' in PV.name_index:
        e_varshift = PV.get_varshift(Gr, 'e')
        for i in xrange(Gr.dims.nlg[0]):
            ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
            for j in xrange(Gr.dims.nlg[1]):
                jshift = j * Gr.dims.nlg[2]
                for k in xrange(Gr.dims.nlg[2]):
                    ijk = ishift + jshift + k
                    PV.values[e_varshift + ijk] = 0.1*1.46*1.46*(1.0-Gr.zp_half[k]/1600.0)

    return

# This case is based on (Soares et al, 2004): An EDMF parameterization for dry and shallow cumulus convection
def InitSoares_moist(namelist, Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, LatentHeat La):
    # Generate the reference profiles
    # RS.Pg = 1.015e5  #Pressure at ground (Bomex)
    RS.Pg = 1.0e5     #Pressure at ground (Soares)
    RS.Tg = 300.0     #Temperature at ground (Soares)
    RS.qtg = 5.0e-3     #Total water mixing ratio at surface: qt = 5 g/kg (Soares)
    RS.u0 = 0.01   # velocities removed in Galilean transformation (Soares: u = 0.01 m/s, IOP: 0.0 m/s)
    RS.v0 = 0.0   # (Soares: v = 0.0 m/s)

    RS.initialize(Gr, Th, NS, Pa)

    #Get the variable number for each of the velocity components
    np.random.seed(Pa.rank)
    cdef:
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t qt_varshift = PV.get_varshift(Gr,'qt')
        Py_ssize_t i,j,k
        Py_ssize_t ishift, jshift, e_varshift
        Py_ssize_t ijk
        double temp
        double qt_
        double [:] theta = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')
        # double [:] thetal = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')
        double [:] qt = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')
        # double [:] u = np.zeros((Gr.dims.nlg[2]),dtype=np.double,order='c')
        Py_ssize_t count

        #Generate initial perturbations (here we are generating more than we need)      ??? where amplitude of perturbations given?
        theta_pert = (np.random.random_sample(Gr.dims.npg )-0.5)*0.1
        qt_pert = (np.random.random_sample(Gr.dims.npg )-0.5)*0.025/1000.0

    for k in xrange(Gr.dims.nlg[2]):
        # Initial theta profile (Soares)
        if Gr.zpl_half[k] <= 1350.0:
            theta[k] = 300.0
        else:
            theta[k] = 300.0 + 2.0/1000.0 * (Gr.zpl_half[k] - 1350.0)

        # Initial qt profile (Soares)
        if Gr.zpl_half[k] <= 1350:
            qt[k] = 5.0 - (Gr.zpl_half[k]) * 3.7e-4
        if Gr.zpl_half[k] > 1350:
            qt[k] = 5.0 - 1350.0 * 3.7e-4 - (Gr.zpl_half[k] - 1350.0) * 9.4e-4

        #Change units to kg/kg
        qt[k]/= 1000.0

    #Now loop and set the initial condition
    #First set the velocities
    count = 0
    for i in xrange(Gr.dims.nlg[0]):
        ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
        for j in xrange(Gr.dims.nlg[1]):
            jshift = j * Gr.dims.nlg[2]
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                PV.values[u_varshift + ijk] = 0.0 - RS.u0
                PV.values[v_varshift + ijk] = 0.0 - RS.v0
                PV.values[w_varshift + ijk] = 0.0
                # Set the entropy prognostic variable including a potential temperature perturbation
                # fluctuation height = 200m; fluctuation amplitude = 0.1 K
                if Gr.zpl_half[k] < 200.0:
                    temp = (theta[k] + (theta_pert[count])) * exner_c(RS.p0_half[k])
                    qt_ = qt[k]+qt_pert[count]
                else:
                    temp = (theta[k]) * exner_c(RS.p0_half[k])
                    qt_ = qt[k]

                PV.values[s_varshift + ijk] = Th.entropy(RS.p0_half[k],temp,qt_,0.0,0.0)
                PV.values[qt_varshift + ijk] = qt_
                count += 1

    if 'e' in PV.name_index:
        e_varshift = PV.get_varshift(Gr, 'e')
        for i in xrange(Gr.dims.nlg[0]):
            ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
            for j in xrange(Gr.dims.nlg[1]):
                jshift = j * Gr.dims.nlg[2]
                for k in xrange(Gr.dims.nlg[2]):
                    ijk = ishift + jshift + k
                    PV.values[e_varshift + ijk] = 0.1*1.46*1.46*(1.0-Gr.zp_half[k]/1600.0)  

    # __ Initialize phi __
    try:
        use_tracers = namelist['tracers']['use_tracers']
    except:
        use_tracers = False

    cdef:
        Py_ssize_t kmin = 0
        Py_ssize_t kmax = 10
        Py_ssize_t var_shift

    if use_tracers == 'passive':
        Pa.root_print('initializing passive tracer phi')
        var_shift = PV.get_varshift(Gr, 'phi')
        with nogil:
            for i in xrange(Gr.dims.nlg[0]):
                ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
                for j in xrange(Gr.dims.nlg[1]):
                    jshift = j * Gr.dims.nlg[2]
                    for k in xrange(Gr.dims.nlg[2]):
                        ijk = ishift + jshift + k
                        if k > kmin and k < kmax:
                    # for k in xrange(kmin, kmax):
                            PV.values[var_shift + ijk] = 1.0
                        else:
                            PV.values[var_shift + ijk] = 0.0
    # __


   # __

    imax = Gr.dims.nlg[0]
    jmax = Gr.dims.nlg[1]
    kmax = Gr.dims.nlg[2]
    istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
    jstride = Gr.dims.nlg[2]
    ijk_max = imax*istride + jmax*jstride + kmax
    if np.isnan(PV.values[s_varshift:qt_varshift]).any():   # nans
        print('nan in s')
    else:
        print('No nan in s')
    if np.isnan(PV.values[qt_varshift:qt_varshift+ijk_max]).any():
        print('nan in qt')
    else:
        print('No nan in qt')
    if np.nanmin(PV.values[qt_varshift:qt_varshift+ijk_max]) < 0:
        print('Init: qt < 0')
    # __

    if 'e' in PV.name_index:
        e_varshift = PV.get_varshift(Gr, 'e')
        for i in xrange(Gr.dims.nlg[0]):
            ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
            for j in xrange(Gr.dims.nlg[1]):
                jshift = j * Gr.dims.nlg[2]
                for k in xrange(Gr.dims.nlg[2]):
                    ijk = ishift + jshift + k
                    PV.values[e_varshift + ijk] = 0.0

    Pa.root_print('finished Initialization Soares_moist')

    return



def InitGabls(namelist,Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, LatentHeat LH ):

    #Generate the reference profiles
    RS.Pg = 1.0e5  #Pressure at ground
    RS.Tg = 265.0  #Temperature at ground
    RS.qtg = 0.0   #Total water mixing ratio at surface
    RS.u0 = 8.0  # velocities removed in Galilean transformation
    RS.v0 = 0.0

    RS.initialize(Gr, Th, NS, Pa)
    try:
        random_seed_factor = namelist['initialization']['random_seed_factor']
    except:
        random_seed_factor = 1

    #Get the variable number for each of the velocity components
    np.random.seed(Pa.rank * random_seed_factor)
    cdef:
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t i,j,k
        Py_ssize_t ishift, jshift, e_varshift
        Py_ssize_t ijk
        double [:] theta = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')
        double t

        #Generate initial perturbations (here we are generating more than we need)

        cdef double [:] theta_pert = np.random.random_sample(Gr.dims.npg)
        cdef double theta_pert_

    for k in xrange(Gr.dims.nlg[2]):
        if Gr.zp_half[k] <=  100.0:
            theta[k] = 265.0

        else:
            theta[k] = 265.0 + (Gr.zp_half[k] - 100.0) * 0.01

    cdef double [:] p0 = RS.p0_half

    #Now loop and set the initial condition
    #First set the velocities
    for i in xrange(Gr.dims.nlg[0]):
        ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
        for j in xrange(Gr.dims.nlg[1]):
            jshift = j * Gr.dims.nlg[2]
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                PV.values[u_varshift + ijk] = 8.0 - RS.u0
                PV.values[v_varshift + ijk] = 0.0 - RS.v0
                PV.values[w_varshift + ijk] = 0.0

                #Now set the entropy prognostic variable including a potential temperature perturbation
                if Gr.zp_half[k] < 50.0:
                    theta_pert_ = (theta_pert[ijk] - 0.5)* 0.1
                else:
                    theta_pert_ = 0.0
                t = (theta[k] + theta_pert_)*exner_c(RS.p0_half[k])

                PV.values[s_varshift + ijk] = Th.entropy(RS.p0_half[k],t,0.0,0.0,0.0)


    if 'e' in PV.name_index:
        e_varshift = PV.get_varshift(Gr, 'e')
        for i in xrange(Gr.dims.nlg[0]):
            ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
            for j in xrange(Gr.dims.nlg[1]):
                jshift = j * Gr.dims.nlg[2]
                for k in xrange(Gr.dims.nlg[2]):
                    ijk = ishift + jshift + k
                    if Gr.zp_half[k] <= 250.0:
                        PV.values[e_varshift + ijk] = 0.4*(1.0-Gr.zp_half[k]/250.0)**3.0
                    else:
                        PV.values[e_varshift + ijk] = 0.0


    return

def InitDYCOMS_RF01(namelist,Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa , LatentHeat LH):

    '''
    Initialize the DYCOMS_RF01 case described in
    Bjorn Stevens, Chin-Hoh Moeng, Andrew S. Ackerman, Christopher S. Bretherton, Andreas Chlond, Stephan de Roode,
    James Edwards, Jean-Christophe Golaz, Hongli Jiang, Marat Khairoutdinov, Michael P. Kirkpatrick, David C. Lewellen,
    Adrian Lock, Frank Müller, David E. Stevens, Eoin Whelan, and Ping Zhu, 2005: Evaluation of Large-Eddy Simulations
    via Observations of Nocturnal Marine Stratocumulus. Mon. Wea. Rev., 133, 1443–1462.
    doi: https://doi.org/10.1175/MWR2930.1
    :param Gr: Grid cdef extension class
    :param PV: PrognosticVariables cdef extension class
    :param RS: ReferenceState cdef extension class
    :param Th: Thermodynamics class
    :return: None
    '''

    # Generate Reference Profiles
    RS.Pg = 1017.8 * 100.0
    RS.qtg = 9.0/1000.0
    RS.u0 = 7.0
    RS.v0 = -5.5

    # Use an exner function with values for Rd, and cp given in Stevens 2004 to compute temperature given $\theta_l$
    RS.Tg = 289.0 * (RS.Pg/p_tilde)**(287.0/1015.0)

    RS.initialize(Gr ,Th, NS, Pa)

    #Set up $\tehta_l$ and $\qt$ profiles
    cdef:
        Py_ssize_t i
        Py_ssize_t j
        Py_ssize_t k
        Py_ssize_t ijk, ishift, jshift
        Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
        Py_ssize_t jstride = Gr.dims.nlg[2]
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t qt_varshift = PV.get_varshift(Gr,'qt')
        double [:] thetal = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        double [:] qt = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        Py_ssize_t e_varshift

    for k in xrange(Gr.dims.nlg[2]):
        if Gr.zp_half[k] <=840.0:
            thetal[k] = 289.0
            qt[k] = 9.0/1000.0
        if Gr.zp_half[k] > 840.0:
            thetal[k] = 297.5 + (Gr.zp_half[k] - 840.0)**(1.0/3.0)
            qt[k] = 1.5/1000.0

    def compute_thetal(p_,T_,ql_):
        theta_ = T_ / (p_/p_tilde)**(287.0/1015.0)
        return theta_ * exp(-2.47e6 * ql_ / (1015.0 * T_))

    def sat_adjst(p_,thetal_,qt_):
        '''
        Use saturation adjustment scheme to compute temperature and ql given thetal and qt.
        :param p: pressure [Pa]
        :param thetal: liquid water potential temperature  [K]
        :param qt:  total water specific humidity
        :return: T, ql
        '''

        #Compute temperature
        t_1 = thetal_ * (p_/p_tilde)**(287.0/1015.0)
        #Compute saturation vapor pressure
        pv_star_1 = Th.get_pv_star(t_1)
        #Compute saturation mixing ratio
        qs_1 = qv_star_c(p_,qt_,pv_star_1)

        if qt_ <= qs_1:
            #If not saturated return temperature and ql = 0.0
            return t_1, 0.0
        else:
            ql_1 = qt_ - qs_1
            f_1 = thetal_ - compute_thetal(p_,t_1,ql_1)
            t_2 = t_1 + 2.47e6*ql_1/1015.0
            pv_star_2 = Th.get_pv_star(t_2)
            qs_2 = qv_star_c(p_,qt_,pv_star_2)
            ql_2 = qt_ - qs_2

            while fabs(t_2 - t_1) >= 1e-9:
                pv_star_2 = Th.get_pv_star(t_2)
                qs_2 = qv_star_c(p_,qt_,pv_star_2)
                ql_2 = qt_ - qs_2
                f_2 = thetal_ - compute_thetal(p_, t_2, ql_2)
                t_n = t_2 - f_2 * (t_2 - t_1)/(f_2 - f_1)
                t_1 = t_2
                t_2 = t_n
                f_1 = f_2

            return t_2, ql_2

    try:
        random_seed_factor = namelist['initialization']['random_seed_factor']
    except:
        random_seed_factor = 1

    #Generate initial perturbations (here we are generating more than we need)
    np.random.seed(Pa.rank * random_seed_factor)
    cdef double [:] theta_pert = np.random.random_sample(Gr.dims.npg)
    cdef double theta_pert_

    for i in xrange(Gr.dims.nlg[0]):
        ishift = istride * i
        for j in xrange(Gr.dims.nlg[1]):
            jshift = jstride * j
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                PV.values[ijk + u_varshift] = 0.0
                PV.values[ijk + v_varshift] = 0.0
                PV.values[ijk + w_varshift] = 0.0
                PV.values[ijk + qt_varshift]  = qt[k]

                #Now set the entropy prognostic variable including a potential temperature perturbation
                if Gr.zp_half[k] < 200.0:
                    theta_pert_ = (theta_pert[ijk] - 0.5)* 0.1
                else:
                    theta_pert_ = 0.0
                T,ql = sat_adjst(RS.p0_half[k],thetal[k] + theta_pert_,qt[k])
                PV.values[ijk + s_varshift] = Th.entropy(RS.p0_half[k], T, qt[k], ql, 0.0)


    if 'e' in PV.name_index:
        e_varshift = PV.get_varshift(Gr, 'e')
        for i in xrange(Gr.dims.nlg[0]):
            ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
            for j in xrange(Gr.dims.nlg[1]):
                jshift = j * Gr.dims.nlg[2]
                for k in xrange(Gr.dims.nlg[2]):
                    ijk = ishift + jshift + k
                    if Gr.zp_half[k] < 200.0:
                        PV.values[e_varshift + ijk] = 0.0

    return



def InitDYCOMS_RF02(namelist,Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, LatentHeat LH ):


    # Generate Reference Profiles
    RS.Pg = 1017.8 * 100.0
    RS.qtg = 9.0/1000.0
    RS.u0 = 5.0
    RS.v0 = -5.5
    cdef double cp_ref = 1004.0
    cdef double L_ref = 2.5e6

    # Use an exner function with values for Rd, and cp given in Stevens 2004 to compute temperature given $\theta_l$
    RS.Tg = 288.3 * (RS.Pg/p_tilde)**(287.0/cp_ref)

    RS.initialize(Gr ,Th, NS, Pa)

    #Set up $\tehta_l$ and $\qt$ profiles
    cdef:
        Py_ssize_t i
        Py_ssize_t j
        Py_ssize_t k
        Py_ssize_t ijk, ishift, jshift
        Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
        Py_ssize_t jstride = Gr.dims.nlg[2]
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t qt_varshift = PV.get_varshift(Gr,'qt')
        double [:] thetal = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        double [:] qt = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        double [:] u = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        double [:] v = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')

    for k in xrange(Gr.dims.nlg[2]):
        if Gr.zp_half[k] <=795.0:
            thetal[k] = 288.3
            qt[k] = 9.45/1000.0
        if Gr.zp_half[k] > 795.0:
            thetal[k] = 295.0 + (Gr.zp_half[k] - 795.0)**(1.0/3.0)
            qt[k] = (5.0 - 3.0 * (1.0 - np.exp(-(Gr.zp_half[k] - 795.0)/500.0)))/1000.0
        v[k] = -9.0 + 5.6 * Gr.zp_half[k]/1000.0 - RS.v0
        u[k] = 3.0 + 4.3*Gr.zp_half[k]/1000.0 - RS.u0

    def compute_thetal(p_,T_,ql_):
        theta_ = T_ / (p_/p_tilde)**(287.0/cp_ref)
        return theta_ * exp(-L_ref * ql_ / (cp_ref * T_))

    def sat_adjst(p_,thetal_,qt_):
        '''
        Use saturation adjustment scheme to compute temperature and ql given thetal and qt.
        :param p: pressure [Pa]
        :param thetal: liquid water potential temperature  [K]
        :param qt:  total water specific humidity
        :return: T, ql
        '''

        #Compute temperature
        t_1 = thetal_ * (p_/p_tilde)**(287.0/cp_ref)
        #Compute saturation vapor pressure
        pv_star_1 = Th.get_pv_star(t_1)
        #Compute saturation mixing ratio
        qs_1 = qv_star_c(p_,qt_,pv_star_1)

        if qt_ <= qs_1:
            #If not saturated return temperature and ql = 0.0
            return t_1, 0.0
        else:
            ql_1 = qt_ - qs_1
            f_1 = thetal_ - compute_thetal(p_,t_1,ql_1)
            t_2 = t_1 + L_ref*ql_1/cp_ref
            pv_star_2 = Th.get_pv_star(t_2)
            qs_2 = qv_star_c(p_,qt_,pv_star_2)
            ql_2 = qt_ - qs_2

            while fabs(t_2 - t_1) >= 1e-9:
                pv_star_2 = Th.get_pv_star(t_2)
                qs_2 = qv_star_c(p_,qt_,pv_star_2)
                ql_2 = qt_ - qs_2
                f_2 = thetal_ - compute_thetal(p_, t_2, ql_2)
                t_n = t_2 - f_2 * (t_2 - t_1)/(f_2 - f_1)
                t_1 = t_2
                t_2 = t_n
                f_1 = f_2

            return t_2, ql_2

    try:
        random_seed_factor = namelist['initialization']['random_seed_factor']
    except:
        random_seed_factor = 1

    #Generate initial perturbations (here we are generating more than we need)
    np.random.seed(Pa.rank * random_seed_factor)
    cdef double [:] theta_pert = np.random.random_sample(Gr.dims.npg)
    cdef double theta_pert_

    for i in xrange(Gr.dims.nlg[0]):
        ishift = istride * i
        for j in xrange(Gr.dims.nlg[1]):
            jshift = jstride * j
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                PV.values[ijk + u_varshift] = u[k]
                PV.values[ijk + v_varshift] = v[k]
                PV.values[ijk + w_varshift] = 0.0
                PV.values[ijk + qt_varshift]  = qt[k]

                #Now set the entropy prognostic variable including a potential temperature perturbation
                if Gr.zp_half[k] < 795.0:
                    theta_pert_ = (theta_pert[ijk] - 0.5)* 0.1
                else:
                    theta_pert_ = 0.0
                T,ql = sat_adjst(RS.p0_half[k],thetal[k] + theta_pert_,qt[k])
                PV.values[ijk + s_varshift] = Th.entropy(RS.p0_half[k], T, qt[k], ql, 0.0)

    return

def InitSmoke(namelist,Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, LatentHeat LH ):
    '''
    Initialization for the smoke cloud case
    Bretherton, C. S., and coauthors, 1999:
    An intercomparison of radiatively- driven entrainment and turbulence in a smoke cloud,
    as simulated by different numerical models. Quart. J. Roy. Meteor. Soc., 125, 391-423. Full text copy.
    :param Gr:
    :param PV:
    :param RS:
    :param Th:
    :param NS:
    :param Pa:
    :return:
    '''


    RS.Pg = 1000.0 * 100.0
    RS.qtg = 0.0
    RS.u0 = 0.0
    RS.v0 = 0.0
    RS.Tg = 288.0

    RS.initialize(Gr ,Th, NS, Pa)
    try:
        random_seed_factor = namelist['initialization']['random_seed_factor']
    except:
        random_seed_factor = 1

    #Get the variable number for each of the velocity components
    np.random.seed(Pa.rank * random_seed_factor)
    cdef:
        Py_ssize_t u_varshift = PV.get_varshift(Gr, 'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr, 'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr, 'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr, 's')
        Py_ssize_t smoke_varshift = PV.get_varshift(Gr, 'smoke')
        Py_ssize_t i,j,k
        Py_ssize_t ishift, jshift, e_varshift
        Py_ssize_t ijk
        double [:] theta = np.empty((Gr.dims.nlg[2]), dtype=np.double, order='c')
        double [:] smoke = np.empty((Gr.dims.nlg[2]), dtype=np.double, order='c')
        double t

        #Generate initial perturbations (here we are generating more than we need)
        cdef double [:] theta_pert = np.random.random_sample(Gr.dims.npg)
        cdef double theta_pert_

    for k in xrange(Gr.dims.nlg[2]):
        if Gr.zp_half[k] <=  687.5:
            theta[k] = 288.0
            smoke[k] = 1.0
        elif Gr.zp_half[k] >= 687.5 and Gr.zp_half[k] <= 712.5:
            theta[k] = 288.0 + (Gr.zp_half[k] - 687.5) * 0.28
            smoke[k] = 1.0 - 0.04 * (Gr.zp_half[k] - 687.5)
            print(k, Gr.zp_half[k], smoke[k])
        else:
            theta[k] = 295.0 + (Gr.zp_half[k] - 712.5) * 1e-4
            smoke[k] = 0.0

    cdef double [:] p0 = RS.p0_half

    #Now loop and set the initial condition
    for i in xrange(Gr.dims.nlg[0]):
        ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
        for j in xrange(Gr.dims.nlg[1]):
            jshift = j * Gr.dims.nlg[2]
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                PV.values[u_varshift + ijk] = 0.0 - RS.u0
                PV.values[v_varshift + ijk] = 0.0 - RS.v0
                PV.values[w_varshift + ijk] = 0.0

                #Now set the entropy prognostic variable including a potential temperature perturbation
                if Gr.zp_half[k] < 700.0:
                    theta_pert_ = (theta_pert[ijk] - 0.5)* 0.1
                else:
                    theta_pert_ = 0.0
                t = (theta[k] + theta_pert_)*exner_c(RS.p0_half[k])

                PV.values[s_varshift + ijk] = Th.entropy(RS.p0_half[k],t,0.0,0.0,0.0)
                PV.values[smoke_varshift + ijk] = smoke[k]

    if 'e' in PV.name_index:
        e_varshift = PV.get_varshift(Gr, 'e')
        for i in xrange(Gr.dims.nlg[0]):
            ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
            for j in xrange(Gr.dims.nlg[1]):
                jshift = j * Gr.dims.nlg[2]
                for k in xrange(Gr.dims.nlg[2]):
                    ijk = ishift + jshift + k
                    if Gr.zp_half[k] < 700.0:
                        PV.values[e_varshift + ijk] = 0.1
                    else:
                        PV.values[e_varshift + ijk] = 0.0

    return


def InitRico(namelist,Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa, LatentHeat LH ):

    #First generate the reference profiles
    RS.Pg = 1.0154e5  #Pressure at ground
    RS.Tg = 299.8  #Temperature at ground
    pvg = Th.get_pv_star(RS.Tg)
    RS.qtg = eps_v * pvg/(RS.Pg - pvg)   #Total water mixing ratio at surface = qsat

    RS.initialize(Gr, Th, NS, Pa)
    try:
        random_seed_factor = namelist['initialization']['random_seed_factor']
    except:
        random_seed_factor = 1

    #Get the variable number for each of the velocity components
    np.random.seed(Pa.rank * random_seed_factor)
    cdef:
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t qt_varshift = PV.get_varshift(Gr,'qt')
        Py_ssize_t i,j,k
        Py_ssize_t ishift, jshift
        Py_ssize_t ijk, e_varshift
        double temp
        double qt_
        double [:] theta = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')
        double [:] qt = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')
        double [:] u = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')
        double [:] v = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')
        Py_ssize_t count

        theta_pert = (np.random.random_sample(Gr.dims.npg )-0.5)*0.1
        qt_pert = (np.random.random_sample(Gr.dims.npg )-0.5) * 2.5e-5

    for k in xrange(Gr.dims.nlg[2]):

        #Set Thetal profile
        if Gr.zp_half[k] <= 740.0:
            theta[k] = 297.9
        else:
            theta[k] = 297.9 + (317.0-297.9)/(4000.0-740.0)*(Gr.zp_half[k] - 740.0)


        #Set qt profile
        if Gr.zp_half[k] <= 740.0:
            qt[k] =  16.0 + (13.8 - 16.0)/740.0 * Gr.zp_half[k]
        elif Gr.zp_half[k] > 740.0 and Gr.zp_half[k] <= 3260.0:
            qt[k] = 13.8 + (2.4 - 13.8)/(3260.0-740.0) * (Gr.zp_half[k] - 740.0)
        else:
            qt[k] = 2.4 + (1.8-2.4)/(4000.0-3260.0)*(Gr.zp_half[k] - 3260.0)


        #Change units to kg/kg
        qt[k]/= 1000.0

        #Set u profile
        u[k] = -9.9 + 2.0e-3 * Gr.zp_half[k]
        #set v profile
        v[k] = -3.8
    #Set velocities for Galilean transformation
    RS.v0 = -3.8
    RS.u0 = 0.5 * (np.amax(u)+np.amin(u))



    #Now loop and set the initial condition
    #First set the velocities
    count = 0
    for i in xrange(Gr.dims.nlg[0]):
        ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
        for j in xrange(Gr.dims.nlg[1]):
            jshift = j * Gr.dims.nlg[2]
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                PV.values[u_varshift + ijk] = u[k] - RS.u0
                PV.values[v_varshift + ijk] = v[k] - RS.v0
                PV.values[w_varshift + ijk] = 0.0
                if Gr.zp_half[k] <= 740.0:
                    temp = (theta[k] + (theta_pert[count])) * exner_c(RS.p0_half[k])
                    qt_ = qt[k]+qt_pert[count]
                else:
                    temp = (theta[k]) * exner_c(RS.p0_half[k])
                    qt_ = qt[k]
                PV.values[s_varshift + ijk] = Th.entropy(RS.p0_half[k],temp,qt_,0.0,0.0)
                PV.values[qt_varshift + ijk] = qt_
                count += 1

    if 'e' in PV.name_index:
        e_varshift = PV.get_varshift(Gr, 'e')
        for i in xrange(Gr.dims.nlg[0]):
            ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
            for j in xrange(Gr.dims.nlg[1]):
                jshift = j * Gr.dims.nlg[2]
                for k in xrange(Gr.dims.nlg[2]):
                    ijk = ishift + jshift + k
                    if Gr.zp_half[k] <= 740.0:
                        PV.values[e_varshift + ijk] = 0.1


    return



def InitCGILS(namelist, Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa , LatentHeat LH):
    #
    try:
        loc = namelist['meta']['CGILS']['location']
        if loc !=12 and loc != 11 and loc != 6:
            Pa.root_print('Invalid CGILS location (must be 6, 11, or 12)')
            Pa.kill()
    except:
        Pa.root_print('Must provide a CGILS location (6/11/12) in namelist')
        Pa.kill()
    try:
        is_p2 = namelist['meta']['CGILS']['P2']
    except:
        Pa.root_print('Must specify if CGILS run is perturbed')
        Pa.kill()


    if is_p2:
        file = './CGILSdata/p2k_s'+str(loc)+'.nc'
    else:
        file = './CGILSdata/ctl_s'+str(loc)+'.nc'

    data = nc.Dataset(file, 'r')
    # Get the profile information we need from the data file
    pressure_data = data.variables['lev'][::-1]
    temperature_data = data.variables['T'][0,::-1,0,0]
    q_data = data.variables['q'][0,::-1,0,0]
    u_data = data.variables['u'][0,::-1,0,0]
    v_data = data.variables['v'][0,::-1,0,0]

    for index in np.arange(len(q_data)):
        q_data[index] = q_data[index]/ (1.0 + q_data[index])




    # Get the surface information we need from the data file
    RS.Tg= data.variables['Tg'][0,0,0]
    RS.Pg= data.variables['Ps'][0,0,0]
    rh_srf = data.variables['rh_srf'][0,0,0]

    data.close()

    # Find the surface moisture and initialize the basic state
    pv_ = Th.get_pv_star(RS.Tg)*rh_srf
    RS.qtg =  eps_v * pv_ / (RS.Pg + (eps_v-1.0)*pv_)


    RS.initialize(Gr ,Th, NS, Pa)




    cdef:
        Py_ssize_t i, j, k, ijk, ishift, jshift
        Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
        Py_ssize_t jstride = Gr.dims.nlg[2]
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t qt_varshift = PV.get_varshift(Gr,'qt')
        double [:] thetal = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        double [:] qt = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        double [:] u = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        double [:] v = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        double p_inversion = 940.0 * 100.0 # for S11, S12: pressure at the inversion
        double p_interp_les = 880 * 100.0 # for S11, S12:  pressure at which we start interpolating to the forcing profile
        double p_interp_data = 860 * 100.0 # for S11, S12: pressure at which we blend full to forcing profile

    #Set up profiles. First create thetal from the forcing data, to be used for interpolation
    thetal_data = np.zeros(np.shape(temperature_data))
    for k in xrange(len(thetal_data)):
        thetal_data[k] = temperature_data[k]/exner_c(pressure_data[k])


    # First we handle the S12 and S11 cases
    # This portion of the profiles is fitted to Figures #,# (S11) and #,# (S12) from Blossey et al
    if loc == 12:
        # Use a mixed layer profile
        if not is_p2:
            # CTL profiles
            for k in xrange(Gr.dims.nlg[2]):
                if RS.p0_half[k] > p_inversion:
                    thetal[k] = 288.35
                    qt[k] = 9.374/1000.0
                else:
                    thetal[k] = (3.50361862e+02 +  -5.11283538e-02 * RS.p0_half[k]/100.0)
                    qt[k] = 3.46/1000.0
        else:
            # P2K
            for k in xrange(Gr.dims.nlg[2]):
                if RS.p0_half[k] > p_inversion:
                    thetal[k] = 290.35
                    qt[k] = 11.64/1000.0
                else:
                    thetal[k] = (3.55021347e+02 +  -5.37703211e-02 * RS.p0_half[k]/100.0)
                    qt[k] = 4.28/1000.0
    elif loc == 11:
        if not is_p2:
            for k in xrange(Gr.dims.nlg[2]):
                if RS.p0_half[k] > 935.0*100.0:
                    thetal[k] = 289.6
                    qt[k] = 10.25/1000.0
                else:
                    thetal[k] = (3.47949119e+02 +  -5.02475698e-02 * RS.p0_half[k]/100.0)
                    qt[k] = 3.77/1000.0
        else:
            # P2 parameters
            for k in xrange(Gr.dims.nlg[2]):
                if RS.p0_half[k] > 935.0*100.0:
                    thetal[k] = 291.6
                    qt[k] =11.64/1000.0
                else:
                    thetal[k] = (3.56173912e+02 +  -5.70945946e-02 * RS.p0_half[k]/100.0)
                    qt[k] = 4.28/1000.0


    # Set up for interpolation to forcing profiles
    if loc == 11 or loc == 12:
        pressure_interp = np.empty(0)
        thetal_interp = np.empty(0)
        qt_interp = np.empty(0)
        for k in xrange(Gr.dims.gw,Gr.dims.nlg[2]-Gr.dims.gw):
            if RS.p0_half[k] > p_interp_les:
                pressure_interp = np.append(pressure_interp,RS.p0_half[k])
                thetal_interp = np.append(thetal_interp,thetal[k])
                # qt_interp = np.append(qt_interp, qt[k]/(1.0+qt[k]))
                qt_interp = np.append(qt_interp, qt[k])

        pressure_interp = np.append(pressure_interp, pressure_data[pressure_data<p_interp_data] )
        thetal_interp = np.append(thetal_interp, thetal_data[pressure_data<p_interp_data] )
        qt_interp = np.append(qt_interp, q_data[pressure_data<p_interp_data] )

        # Reverse the arrays so pressure is increasing
        pressure_interp = pressure_interp[::-1]
        thetal_interp = thetal_interp[::-1]
        qt_interp = qt_interp[::-1]

    else:
        # for S6 case, interpolate ALL values
        p_interp_les = RS.Pg
        pressure_interp = pressure_data[::-1]
        thetal_interp = thetal_data[::-1]
        qt_interp = q_data[::-1]

    # PCHIP interpolation helps to make the S11 and S12 thermodynamic profiles nice, but the scipy pchip interpolator
    # does not handle extrapolation kindly. Thus we tack on our own linear extrapolation to deal with the S6 case
    # We also use linear extrapolation to handle the velocity profiles, which it seems are fine to interpolate linearly

    thetal_right = thetal_interp[-1] + (thetal_interp[-2] - thetal_interp[-1])/(pressure_interp[-2]-pressure_interp[-1]) \
                                                 * ( RS.Pg-pressure_interp[-1])
    thetal_interp = np.append(thetal_interp, thetal_right)
    qt_right = qt_interp[-1] + (qt_interp[-2] - qt_interp[-1])/(pressure_interp[-2]-pressure_interp[-1]) \
                                                 * ( RS.Pg-pressure_interp[-1])
    qt_interp = np.append(qt_interp, qt_right)
    pressure_interp = np.append(pressure_interp, RS.Pg)



    # Now do the interpolation
    for k in xrange(Gr.dims.nlg[2]):
            if RS.p0_half[k] <= p_interp_les:

                # thetal_right = thetal_interp[-1] + (thetal_interp[-2] - thetal_interp[-1])/(pressure_interp[-2]-pressure_interp[-1]) \
                #                                  * ( RS.p0_half[k]-pressure_interp[-1])
                # qt_right = qt_interp[-1] + (qt_interp[-2] - qt_interp[-1])/(pressure_interp[-2]-pressure_interp[-1]) \
                #                                  * ( RS.p0_half[k]-pressure_interp[-1])

                # thetal[k] = np.interp(RS.p0_half[k], pressure_interp, thetal_interp, right = thetal_right)
                # qt[k] = np.interp(RS.p0_half[k],pressure_interp,qt_interp, right=qt_right)
                thetal[k] = pchip_interpolate(pressure_interp, thetal_interp, RS.p0_half[k])
                qt[k] = pchip_interpolate(pressure_interp, qt_interp, RS.p0_half[k])
            # Interpolate entire velocity profiles
            u_right = u_data[0] + (u_data[1] - u_data[0])/(pressure_data[1]-pressure_data[0]) * ( RS.p0_half[k]-pressure_data[0])
            v_right = v_data[0] + (v_data[1] - v_data[0])/(pressure_data[1]-pressure_data[0]) * ( RS.p0_half[k]-pressure_data[0])

            u[k] = np.interp(RS.p0_half[k],pressure_data[::-1], u_data[::-1], right=u_right)
            v[k] = np.interp(RS.p0_half[k],pressure_data[::-1], v_data[::-1],right=v_right)
    #Set velocities for Galilean transformation
    RS.u0 = 0.5 * (np.amax(u)+np.amin(u))
    RS.v0 = 0.5 * (np.amax(v)+np.amin(v))


    # We will need these functions to perform saturation adjustment
    def compute_thetal(p_,T_,ql_):
        theta_ = T_ / (p_/p_tilde)**kappa
        return theta_ * exp(-2.501e6 * ql_ / (cpd* T_))

    def sat_adjst(p_,thetal_,qt_):
        '''
        Use saturation adjustment scheme to compute temperature and ql given thetal and qt.
        :param p: pressure [Pa]
        :param thetal: liquid water potential temperature  [K]
        :param qt:  total water specific humidity
        :return: T, ql
        '''

        #Compute temperature
        t_1 = thetal_ * (p_/p_tilde)**kappa
        #Compute saturation vapor pressure
        pv_star_1 = Th.get_pv_star(t_1)
        #Compute saturation mixing ratio
        qs_1 = qv_star_c(p_,qt_,pv_star_1)

        if qt_ <= qs_1:
            #If not saturated return temperature and ql = 0.0
            return t_1, 0.0
        else:
            ql_1 = qt_ - qs_1
            f_1 = thetal_ - compute_thetal(p_,t_1,ql_1)
            t_2 = t_1 + 2.501e6*ql_1/cpd
            pv_star_2 = Th.get_pv_star(t_2)
            qs_2 = qv_star_c(p_,qt_,pv_star_2)
            ql_2 = qt_ - qs_2

            while fabs(t_2 - t_1) >= 1e-9:
                pv_star_2 = Th.get_pv_star(t_2)
                qs_2 = qv_star_c(p_,qt_,pv_star_2)
                ql_2 = qt_ - qs_2
                f_2 = thetal_ - compute_thetal(p_, t_2, ql_2)
                t_n = t_2 - f_2 * (t_2 - t_1)/(f_2 - f_1)
                t_1 = t_2
                t_2 = t_n
                f_1 = f_2

            return t_2, ql_2

    try:
        random_seed_factor = namelist['initialization']['random_seed_factor']
    except:
        random_seed_factor = 1

    #Generate initial perturbations (here we are generating more than we need)
    np.random.seed(Pa.rank * random_seed_factor)
    cdef double [:] theta_pert = np.random.random_sample(Gr.dims.npg)
    cdef double theta_pert_

    # Here we fill in the 3D arrays
    # We perform saturation adjustment on the S6 data, although this should not actually be necessary (but doesn't hurt)
    for i in xrange(Gr.dims.nlg[0]):
        ishift = istride * i
        for j in xrange(Gr.dims.nlg[1]):
            jshift = jstride * j
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                PV.values[ijk + u_varshift] = u[k] - RS.u0
                PV.values[ijk + v_varshift] = v[k] - RS.v0
                PV.values[ijk + w_varshift] = 0.0
                PV.values[ijk + qt_varshift]  = qt[k]

                #Now set the entropy prognostic variable including a potential temperature perturbation
                if Gr.zp_half[k] < 200.0:
                    theta_pert_ = (theta_pert[ijk] - 0.5)* 0.1
                else:
                    theta_pert_ = 0.0
                T,ql = sat_adjst(RS.p0_half[k],thetal[k] + theta_pert_,qt[k])
                PV.values[ijk + s_varshift] = Th.entropy(RS.p0_half[k], T, qt[k], ql, 0.0)



    return






def InitZGILS(namelist, Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa , LatentHeat LH):

    reference_profiles = AdjustedMoistAdiabat(namelist, LH, Pa)

    RS.Tg= 289.472
    RS.Pg= 1018.0e2
    RS.qtg = 0.008449

    RS.initialize(Gr ,Th, NS, Pa)


    cdef double Pg_parcel = 1000.0e2
    cdef double Tg_parcel = 295.0
    cdef double RH_ref = 0.3
    reference_profiles.initialize(Pa, RS.p0_half[:], Gr.dims.nlg[2],Pg_parcel, Tg_parcel, RH_ref)



    cdef:
        Py_ssize_t i, j, k, ijk, ishift, jshift
        Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
        Py_ssize_t jstride = Gr.dims.nlg[2]
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t qt_varshift = PV.get_varshift(Gr,'qt')
        double [:] thetal = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        double [:] qt = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        double [:] u = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        double [:] v = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')

    for k in xrange(Gr.dims.nlg[2]):
        if RS.p0_half[k]  > 920.0e2:
            thetal[k] = RS.Tg /exner_c(RS.Pg)
            qt[k] = RS.qtg
        u[k] = min(-10.0 + (-7.0-(-10.0))/(750.0e2-1000.0e2)*(RS.p0_half[k]-1000.0e2),-4.0)


      #Set velocities for Galilean transformation
    RS.u0 = 0.5 * (np.amax(u)+np.amin(u))
    RS.v0 = 0.5 * (np.amax(v)+np.amin(v))


    # We will need these functions to perform saturation adjustment
    def compute_thetal(p_,T_,ql_):
        theta_ = T_ / (p_/p_tilde)**kappa
        return theta_ * exp(-2.501e6 * ql_ / (cpd* T_))

    def sat_adjst(p_,thetal_,qt_):


        #Compute temperature
        t_1 = thetal_ * (p_/p_tilde)**kappa
        #Compute saturation vapor pressure
        pv_star_1 = Th.get_pv_star(t_1)
        #Compute saturation mixing ratio
        qs_1 = qv_star_c(p_,qt_,pv_star_1)

        if qt_ <= qs_1:
            #If not saturated return temperature and ql = 0.0
            return t_1, 0.0
        else:
            ql_1 = qt_ - qs_1
            f_1 = thetal_ - compute_thetal(p_,t_1,ql_1)
            t_2 = t_1 + 2.501e6*ql_1/cpd
            pv_star_2 = Th.get_pv_star(t_2)
            qs_2 = qv_star_c(p_,qt_,pv_star_2)
            ql_2 = qt_ - qs_2

            while fabs(t_2 - t_1) >= 1e-9:
                pv_star_2 = Th.get_pv_star(t_2)
                qs_2 = qv_star_c(p_,qt_,pv_star_2)
                ql_2 = qt_ - qs_2
                f_2 = thetal_ - compute_thetal(p_, t_2, ql_2)
                t_n = t_2 - f_2 * (t_2 - t_1)/(f_2 - f_1)
                t_1 = t_2
                t_2 = t_n
                f_1 = f_2

            return t_2, ql_2

    try:
        random_seed_factor = namelist['initialization']['random_seed_factor']
    except:
        random_seed_factor = 1
    #Generate initial perturbations (here we are generating more than we need)
    np.random.seed(Pa.rank * random_seed_factor)
    cdef double [:] theta_pert = np.random.random_sample(Gr.dims.npg)
    cdef double theta_pert_

    # Here we fill in the 3D arrays
    # We perform saturation adjustment on the S6 data, although this should not actually be necessary (but doesn't hurt)
    for i in xrange(Gr.dims.nlg[0]):
        ishift = istride * i
        for j in xrange(Gr.dims.nlg[1]):
            jshift = jstride * j
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                PV.values[ijk + u_varshift] = u[k] - RS.u0
                PV.values[ijk + v_varshift] = v[k] - RS.v0
                PV.values[ijk + w_varshift] = 0.0
                if RS.p0_half[k] > 920.0e2:
                    PV.values[ijk + qt_varshift]  = qt[k]
                    theta_pert_ = (theta_pert[ijk] - 0.5)* 0.1
                    T,ql = sat_adjst(RS.p0_half[k],thetal[k] + theta_pert_,qt[k])
                    PV.values[ijk + s_varshift] = Th.entropy(RS.p0_half[k], T, qt[k], ql, 0.0)
                else:
                    PV.values[ijk + qt_varshift]  = reference_profiles.qt[k]
                    PV.values[ijk + s_varshift] = reference_profiles.s[k]


    return


def InitTRMM_LBA(namelist, Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa , LatentHeat LH):

    reference_profiles = AdjustedMoistAdiabat(namelist, LH, Pa)

    RS.Tg  = 296.85   # surface values for reference state (RS) which outputs p0 rho0 alpha0
    RS.Pg  = 991.3*100
    pvg = Th.get_pv_star(RS.Tg)
    RS.qtg = eps_v * pvg/(RS.Pg - pvg)
    # TRMM_LBA inputs

    z_in = np.array([0.130,  0.464,  0.573,  1.100,  1.653,  2.216,  2.760,
                     3.297,  3.824,  4.327,  4.787,  5.242,  5.686,  6.131,
                     6.578,  6.996,  7.431,  7.881,  8.300,  8.718,  9.149,
                     9.611, 10.084, 10.573, 11.008, 11.460, 11.966, 12.472,
                    12.971, 13.478, 13.971, 14.443, 14.956, 15.458, 16.019,
                    16.491, 16.961, 17.442, 17.934, 18.397, 18.851, 19.331,
                    19.809, 20.321, 20.813, 21.329, 30.000]) * 1000 - 130.0 #LES z is in meters

    p_in = np.array([991.3, 954.2, 942.0, 886.9, 831.5, 778.9, 729.8,
                     684.0, 641.7, 603.2, 570.1, 538.6, 509.1, 480.4,
                     454.0, 429.6, 405.7, 382.5, 361.1, 340.9, 321.2,
                     301.2, 281.8, 263.1, 246.1, 230.1, 213.2, 197.0,
                     182.3, 167.9, 154.9, 143.0, 131.1, 119.7, 108.9,
                     100.1,  92.1,  84.6,  77.5,  71.4,  65.9,  60.7,
                      55.9,  51.3,  47.2,  43.3,  10.3]) * 100 # LES pres is in pasc

    T_in = np.array([23.70,  23.30,  22.57,  19.90,  16.91,  14.09,  11.13,
                      8.29,   5.38,   2.29,  -0.66,  -3.02,  -5.28,  -7.42,
                    -10.34, -12.69, -15.70, -19.21, -21.81, -24.73, -27.76,
                    -30.93, -34.62, -38.58, -42.30, -46.07, -50.03, -54.67,
                    -59.16, -63.60, -67.68, -70.77, -74.41, -77.51, -80.64,
                    -80.69, -80.00, -81.38, -81.17, -78.32, -74.77, -74.52,
                    -72.62, -70.87, -69.19, -66.90, -66.90]) + 273.15 # LES T is in deg K

    RH_in = np.array([98.00,  86.00,  88.56,  87.44,  86.67,  83.67,  79.56,
                      84.78,  84.78,  89.33,  94.33,  92.00,  85.22,  77.33,
                      80.11,  66.11,  72.11,  72.67,  52.22,  54.67,  51.00,
                      43.78,  40.56,  43.11,  54.78,  46.11,  42.33,  43.22,
                      45.33,  39.78,  33.78,  28.78,  24.67,  20.67,  17.67,
                      17.11,  16.22,  14.22,  13.00,  13.00,  12.22,   9.56,
                       7.78,   5.89,   4.33,   3.00,   3.00])

    u_in = np.array([0.00,   0.81,   1.17,   3.44,   3.53,   3.88,   4.09,
                     3.97,   1.22,   0.16,  -1.22,  -1.72,  -2.77,  -2.65,
                    -0.64,  -0.07,  -1.90,  -2.70,  -2.99,  -3.66,  -5.05,
                    -6.64,  -4.74,  -5.30,  -6.07,  -4.26,  -7.52,  -8.88,
                    -9.00,  -7.77,  -5.37,  -3.88,  -1.15,  -2.36,  -9.20,
                    -8.01,  -5.68,  -8.83, -14.51, -15.55, -15.36, -17.67,
                   -17.82, -18.94, -15.92, -15.32, -15.32])

    v_in = np.array([-0.40,  -3.51,  -3.88,  -4.77,  -5.28,  -5.85,  -5.60,
                     -2.67,  -1.47,   0.57,   0.89,  -0.08,   1.11,   2.15,
                      3.12,   3.22,   3.34,   1.91,   1.15,   1.01,  -0.57,
                     -0.67,   0.31,   2.97,   2.32,   2.66,   4.79,   3.40,
                      3.14,   3.93,   7.57,   2.58,   2.50,   6.44,   6.84,
                      0.19,  -2.20,  -3.60,   0.56,   6.68,   9.41,   7.03,
                      5.32,   1.14,  -0.65,   5.27,   5.27])


    RS.initialize(Gr ,Th, NS, Pa)


    cdef:
        Py_ssize_t i, j, k, ijk, ishift, jshift
        Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
        Py_ssize_t jstride = Gr.dims.nlg[2]
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t qt_varshift = PV.get_varshift(Gr,'qt')
        double [:] T = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c') # change to temp interp to zp_hlaf (LES press is in pasc)
        double [:] qt = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        double [:] u = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        double [:] v = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')

    T = np.interp(Gr.zp_half,z_in,T_in)
    p = np.interp(Gr.zp_half,z_in,p_in)
    RH = np.interp(Gr.zp_half,z_in,RH_in)
    u = np.interp(Gr.zp_half,z_in,u_in)
    v = np.interp(Gr.zp_half,z_in,v_in)

      #Set velocities for Galilean transformation
    RS.u0 = 0.5 * (np.amax(u)+np.amin(u))
    RS.v0 = 0.5 * (np.amax(v)+np.amin(v))

    #Generate initial perturbations (here we are generating more than we need)
    #random fluctuations
    #I need to perturbed the temperature and only later calculate the entropy
    try:
        random_seed_factor = namelist['initialization']['random_seed_factor']
    except:
        random_seed_factor = 1
    np.random.seed(Pa.rank * random_seed_factor)
    cdef double [:] T_pert = np.random.random_sample(Gr.dims.npg)
    cdef double T_pert_
    cdef double pv_star
    cdef double qv_star

    epsi = 287.1/461.5
    # Here we fill in the 3D arrays
    # We perform saturation adjustment on the S6 data, although this should not actually be necessary (but doesn't hurt)
    for i in xrange(Gr.dims.nlg[0]):
        ishift = istride * i
        for j in xrange(Gr.dims.nlg[1]):
            jshift = jstride * j
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                pv_star = Th.get_pv_star(T[k])
                qv_star = pv_star*epsi/(p[k]- pv_star + epsi*pv_star*RH[k]/100.0) # eq. 37 in pressel et al and the def of RH
                qt[k] = qv_star*RH[k]/100.0
                PV.values[ijk + u_varshift] = u[k] - RS.u0
                PV.values[ijk + v_varshift] = v[k] - RS.v0
                PV.values[ijk + w_varshift] = 0.0
                PV.values[ijk + qt_varshift]  = qt[k]

                if Gr.zp_half[k] < 1000.0:
                    T_pert_ = (T_pert[ijk] - 0.5)* 0.1
                    PV.values[ijk + s_varshift] = Th.entropy(RS.p0_half[k], T[k] + T_pert_, qt[k], 0.0, 0.0)
                else:
                    PV.values[ijk + s_varshift] = Th.entropy(RS.p0_half[k], T[k] , qt[k], 0.0, 0.0)

    return

def InitARM_SGP(namelist, Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa , LatentHeat LH):

    reference_profiles = AdjustedMoistAdiabat(namelist, LH, Pa)

    RS.Tg  = 299.0   # surface values for reference state (RS) which outputs p0 rho0 alpha0
    RS.Pg  = 970.0*100
    RS.qtg = 15.2/1000

    # ARM_SGP inputs

    z_in = np.array([0.0, 50.0,350.0, 650.0, 700.0, 1300.0, 2500.0, 5500.0 ]) #LES z is in meters
    Theta_in = np.array([299.0, 301.5, 302.5, 303.53, 303.7, 307.13, 314.0, 343.2]) # K
    qt_in = np.array([15.2,15.17,14.98,14.8,14.7,13.5,3.0,3.0])/1000 # qt should be in kg/kg
    u = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')+10
    v = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')

    RS.initialize(Gr ,Th, NS, Pa)


    cdef:
        Py_ssize_t i, j, k, ijk, ishift, jshift
        Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
        Py_ssize_t jstride = Gr.dims.nlg[2]
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t qt_varshift = PV.get_varshift(Gr,'qt')
        double [:] Theta = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c') # change to temp interp to zp_hlaf (LES press is in pasc)
        double [:] qt = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        #double [:] u = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        #double [:] v = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')

    Theta = np.interp(Gr.zp_half,z_in,Theta_in)
    qt = np.interp(Gr.zp_half,z_in,qt_in)


      #Set velocities for Galilean transformation
    RS.u0 = 0.5 * (np.amax(u)+np.amin(u))
    RS.v0 = 0.5 * (np.amax(v)+np.amin(v))

    #Generate initial perturbations (here we are generating more than we need)
    #random fluctuations
    #I need to perturbed the temperature and only later calculate the entropy
    try:
        random_seed_factor = namelist['initialization']['random_seed_factor']
    except:
        random_seed_factor = 1

    np.random.seed(Pa.rank * random_seed_factor)
    cdef double [:] T_pert = np.random.random_sample(Gr.dims.npg)
    cdef double T_pert_
    cdef double pv_star
    cdef double qv_star

    epsi = 287.1/461.5
    # Here we fill in the 3D arrays
    # We perform saturation adjustment on the S6 data, although this should not actually be necessary (but doesn't hurt)
    for i in xrange(Gr.dims.nlg[0]):
        ishift = istride * i
        for j in xrange(Gr.dims.nlg[1]):
            jshift = jstride * j
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                PV.values[ijk + u_varshift] = u[k] - RS.u0
                PV.values[ijk + v_varshift] = v[k] - RS.v0
                PV.values[ijk + w_varshift] = 0.0
                PV.values[ijk + qt_varshift]  = qt[k]
                T  = Theta[k]*exner_c(RS.p0_half[k])
                if Gr.zp_half[k] < 200.0: # perturbation temp on the lower 200 m and decrease linearly from 0 to 200m
                    T_pert_ = T_pert[ijk]*(1 - Gr.zp_half[k]/200)* 0.1
                    PV.values[ijk + s_varshift] = Th.entropy(RS.p0_half[k], T + T_pert_, qt[k], 0.0, 0.0)
                else:
                    PV.values[ijk + s_varshift] = Th.entropy(RS.p0_half[k], T , qt[k], 0.0, 0.0)
    return


def InitGATE_III(namelist, Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa , LatentHeat LH):

    reference_profiles = AdjustedMoistAdiabat(namelist, LH, Pa)

    RS.Tg  = 299.88   # surface values for reference state (RS) which outputs p0 rho0 alpha0
    RS.Pg  = 1012.0*100.0
    RS.qtg = 16.5/1000.0

    # GATE_III inputs
    T_in = np.array([299.184, 294.836, 294.261, 288.773, 276.698, 265.004, 253.930, 243.662, 227.674, 214.266, 207.757, 201.973, 198.278, 197.414, 198.110, 198.110])
    z_T_in = np.array([0.0, 0.492, 0.700, 1.698, 3.928, 6.039, 7.795, 9.137, 11.055, 12.645, 13.521, 14.486, 15.448, 16.436, 17.293, 22.0])*1000.0 # for km
    z_in  = np.array([ 0.0,   0.5,  1.0,  1.5,  2.0,   2.5,    3.0,   3.5,   4.0,   4.5,   5.0,  5.5,  6.0,  6.5, 7.0, 7.5, 8.0,  8.5,   9.0,   9.5,  10.0,   10.5,   11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 27.0]) * 1000.0 #LES z is in meters
    qt_in = np.array([16.5,  16.5, 13.5, 12.0, 10.0,   8.7,    7.1,   6.1,   5.2,   4.5,   3.6,  3.0,  2.3, 1.75, 1.3, 0.9, 0.5, 0.25, 0.125, 0.065, 0.003, 0.0015, 0.0007,  0.0003,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001, 0.0001])/1000 # qt should be in kg/kg
    U_in  = np.array([  -1, -1.75, -2.5, -3.6, -6.0, -8.75, -11.75, -13.0, -13.1, -12.1, -11.0, -8.5, -5.0, -2.6, 0.0, 0.5, 0.4,  0.3,   0.0,  -1.0,  -2.5,   -3.5,   -4.5, -4.8, -5.0, -3.5, -2.0, -1.0, -1.0, -1.0, -1.5, -2.0, -2.5, -2.6, -2.7, -3.0, -3.0, -3.0])# [m/s]
    v = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')

    RS.initialize(Gr ,Th, NS, Pa)

    cdef:
        Py_ssize_t i, j, k, ijk, ishift, jshift
        Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
        Py_ssize_t jstride = Gr.dims.nlg[2]
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t qt_varshift = PV.get_varshift(Gr,'qt')
        double [:] T = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c') # change to temp interp to zp_hlaf (LES press is in pasc)
        double [:] TK = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        double [:] qt = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        double [:] u = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        #double [:] v = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')

    T = np.interp(Gr.zp_half,z_T_in,T_in)
    T[Gr.dims.gw-1] = T[Gr.dims.gw]
    T[Gr.dims.gw-2] = T[Gr.dims.gw+1]
    T[Gr.dims.gw-3] = T[Gr.dims.gw+2]
    #T[1] = T[Gr.gw+1]

    qt = np.interp(Gr.zp_half,z_in,qt_in)
    u  = np.interp(Gr.zp_half,z_in,U_in)
    #Set velocities for Galilean transformation
    RS.u0 = 0.5 * (np.amax(u)+np.amin(u))
    RS.v0 = 0.5 * (np.amax(v)+np.amin(v))

    #Generate initial perturbations (here we are generating more than we need)
    #random fluctuations
    #I need to perturbed the temperature and only later calculate the entropy
    try:
        random_seed_factor = namelist['initialization']['random_seed_factor']
    except:
        random_seed_factor = 1

    np.random.seed(Pa.rank * random_seed_factor)
    cdef double [:] T_pert = np.random.random_sample(Gr.dims.npg)
    cdef double T_pert_
    cdef double pv_star
    cdef double qv_star

    #epsi = 287.1/461.5
    # Here we fill in the 3D arrays
    # We perform saturation adjustment on the S6 data, although this should not actually be necessary (but doesn't hurt)
    for i in xrange(Gr.dims.nlg[0]):
        ishift = istride * i
        for j in xrange(Gr.dims.nlg[1]):
            jshift = jstride * j
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                PV.values[ijk + u_varshift] = u[k] - RS.u0
                PV.values[ijk + v_varshift] = v[k] - RS.v0
                PV.values[ijk + w_varshift] = 0.0
                PV.values[ijk + qt_varshift]  = qt[k]

                if Gr.zp_half[k] < 200.0: # perturbation temp on the lower 200 m and decrease linearly from 0 to 200m
                    T_pert_ = T_pert[ijk]*(1 - Gr.zp_half[k]/200)* 0.1
                    PV.values[ijk + s_varshift] = Th.entropy(RS.p0_half[k], T[k]+ T_pert_, qt[k], 0.0, 0.0)#
                else:
                    PV.values[ijk + s_varshift] = Th.entropy(RS.p0_half[k], T[k] , qt[k], 0.0, 0.0)

    return

def InitWANGARA(namelist, Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa , LatentHeat LH):

    reference_profiles = AdjustedMoistAdiabat(namelist, LH, Pa)

    RS.Tg  = 297.0   # surface values for reference state (RS) which outputs p0 rho0 alpha0
    RS.Pg  = 1013*100
    RS.qtg = 17.5/1000

    # Wangara inputs
    z_v_in   = np.array([54.653, 106.148,175.241, 314.417,418.539, 521.036, 600.848, 765.701, 897.974, 1002.897, 1120.990, 1149.140,1220.283,1308.151,1402.827,1501.767,1617.903,1710.719,1905.112,1948.174,1988.340]) # [m]
    v_in     = np.array([-0.419, -0.627,-0.514, -0.497, -0.435, -0.284, -0.237, -0.501,-0.900,-1.366, -0.824, -0.579, -0.211, 0.086, 0.117, 0.278, 1.212, 1.696, 1.099, 0.571, 0.098]) # [m/s]

    z_u_in   = np.fliplr(np.array([1986.677, 1897.052, 1805.682, 1744.848, 1680.877, 1604.032, 1499.080, 1405.682, 1305.517, 1200.566, 1097.335, 1015.656, 957.698, 897.948,847.819, 798.184, 744.848, 715.869, 658.359, 609.644, 552.888, 452.487, 345.437, 303.490, 198.562,153.643, 105.730, 49.682])) # [m]
    u_in     = np.fliplr(np.array([0.060, 0.459, 0.057, -0.390, -0.903, -1.460, -1.193 , -1.235, -1.435, -2.096, -1.968, -2.260, -2.538, -2.290, -2.317, -2.435, -2.466, -2.436, -2.438, -2.670, -2.813, -2.523, -3.230, -3.373, -2.814, -2.800, -2.900, -2.933])) # [m/s]

    z_ug_in  = np.array([3.984, 26.382,104.279,172.557,401.839,714.228,885.041,1095.556,1404.998,1479.547,1865.095,1975.504,1994.082]) # [m]
    ug_in    = np.array([-5.273,-5.261,-4.977   ,-4.735,-3.966,-3.063,-2.623,-2.165,-1.605,-1.487,-0.994, -0.885, -0.796]) # [m/s]

    z_vg_in  = np.fliplr(np.array([1981.726, 1857.332, 1636.376,1384.139,1088.321, 804.141, 461.978, 5.052])) # [m]
    vg_in    = np.fliplr(np.array([-0.240, -0.287, -0.347, -0.436, -0.460, -0.482, -0.505, -0.442])) # [m/s]

    z_qt_in  = np.fliplr(np.array([1994.880,1905.640, 1705.566, 1597.886, 1504.521, 1406.677, 1293.189, 1105.156, 1041.776, 850.591, 655.424, 597.803, 548.478, 246.554, 154.636, 102.182, 49.444])) # [m]
    qt_in    = np.fliplr(np.array([0.607, 0.697, 0.694, 0.792, 0.982, 0.813, 1.214, 1.772, 1.917, 2.303, 3.088, 3.203, 3.204, 3.808, 3.798, 3.518, 3.695]))/1000 # to [kg/kg]

    z_Theta_in = np.array([50.083,103.545,160.541,198.014, 267.149, 343.602, 391.366, 446.674, 492.321, 552.065, 595.978, 699.433, 799.532, 896.748, 951.336, 1003.018, 1099.874, 1208.958, 1306.040, 1409.877, 1500.023, 1603.815, 1716.480, 1812.615, 1897.514, 1993.447])
    Theta_in   = np.array([275.962,276.918,279.268,280.081,280.836,281.293,281.436,281.435, 281.305, 281.324, 281.235, 281.331, 282.050, 282.489, 282.670, 283.183, 284.259, 285.112, 285.584, 287.001, 288.148, 288.728, 289.522, 290.515, 290.887, 290.9]) # K


    RS.initialize(Gr ,Th, NS, Pa)


    cdef:
        Py_ssize_t i, j, k, ijk, ishift, jshift
        Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
        Py_ssize_t jstride = Gr.dims.nlg[2]
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
        Py_ssize_t qt_varshift = PV.get_varshift(Gr,'qt')
        double [:] Theta = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c') # change to temp interp to zp_hlaf (LES press is in pasc)
        double [:] qt = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        double [:] u = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        double [:] v = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        double [:] ug = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')
        double [:] vg = np.zeros((Gr.dims.nlg[2],),dtype=np.double,order='c')


    u = np.interp(Gr.zp_half,z_u_in,u_in)
    v = np.interp(Gr.zp_half,z_v_in,v_in)
    ug = np.interp(Gr.zp_half,z_ug_in,ug_in)
    vg = np.interp(Gr.zp_half,z_vg_in,vg_in)
    qt = np.interp(Gr.zp_half,z_qt_in,qt_in)
    Theta = np.interp(Gr.zp_half,z_Theta_in,Theta_in)


      #Set velocities for Galilean transformation
    RS.u0 = 0.5 * (np.amax(u)+np.amin(u))
    RS.v0 = 0.5 * (np.amax(v)+np.amin(v))

    try:
        random_seed_factor = namelist['initialization']['random_seed_factor']
    except:
        random_seed_factor = 1
    # it is not clear if there is any random perturbation in the Neggers paper
    np.random.seed(Pa.rank * random_seed_factor)
    cdef double [:] T_pert = np.random.random_sample(Gr.dims.npg)
    cdef double T_pert_
    cdef double pv_star
    cdef double qv_star

    epsi = 287.1/461.5
    # Here we fill in the 3D arrays
    # We perform saturation adjustment on the S6 data, although this should not actually be necessary (but doesn't hurt)
    for i in xrange(Gr.dims.nlg[0]):
        ishift = istride * i
        for j in xrange(Gr.dims.nlg[1]):
            jshift = jstride * j
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                PV.values[ijk + u_varshift] = u[k] - RS.u0
                PV.values[ijk + v_varshift] = v[k] - RS.v0
                PV.values[ijk + w_varshift] = 0.0
                PV.values[ijk + qt_varshift]  = qt[k]
                T  = Theta[k]*exner_c(RS.p0_half[k])
                if Gr.zp_half[k] < 1000.0:
                    T_pert_ = (T_pert[ijk] - 0.5)* 0.1
                    PV.values[ijk + s_varshift] = Th.entropy(RS.p0_half[k], T + T_pert_, qt[k], 0.0, 0.0)
                else:
                    PV.values[ijk + s_varshift] = Th.entropy(RS.p0_half[k], T , qt[k], 0.0, 0.0)
    return

def InitGCMNew(namelist, Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa , LatentHeat LH):

    #Generate the reference profiles
    data_path = namelist['gcm']['file']
    try:
        griddata = namelist['gcm']['griddata']
    except:
        griddata = False
    try:
        instant_forcing = namelist['gcm']['instant_forcing']
    except:
        instant_forcing = False
    try:
        gcm_tidx = namelist['gcm']['gcm_tidx']
    except:
        gcm_tidx = 0
    if griddata:
        lat = namelist['gcm']['lat']
        lon = namelist['gcm']['lon']
    else:
        site = namelist['gcm']['site']
    #fh = open(data_path, 'r')
    #input_data_tv = pickle.load(fh)
    #fh.close()

    if griddata:
        rdr = cfreader_grid(data_path, lat, lon)
    else:
        rdr = cfreader(data_path, site)

    RS.Pg = rdr.get_timeseries_mean('ps', instant=instant_forcing, t_idx=gcm_tidx)
    RS.Tg = rdr.get_timeseries_mean('ts', instant=instant_forcing, t_idx=gcm_tidx)
    RS.qtg = rdr.get_profile_mean('hus', instant=instant_forcing, t_idx=gcm_tidx)[0]

    RS.u0 = 0.0
    RS.v0 = 0.0

    RS.initialize(Gr, Th, NS, Pa)

    try:
        random_seed_factor = namelist['initialization']['random_seed_factor']
    except:
        random_seed_factor = 1

    np.random.seed(Pa.rank * random_seed_factor)

    cdef:
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift
        Py_ssize_t thli_varshift
        Py_ssize_t qt_varshift = PV.get_varshift(Gr, 'qt')
        Py_ssize_t i,j,k
        Py_ssize_t ishift, jshift, e_varshift
        Py_ssize_t ijk



    cdef double [:] t = rdr.get_interp_profile('ta', Gr.zp_half, instant=instant_forcing, t_idx=gcm_tidx)#interp_pchip(Gr.zp_half, z_in, np.log(t_in))
    cdef double [:] qt = rdr.get_interp_profile('hus', Gr.zp_half, instant=instant_forcing, t_idx=gcm_tidx)
    cdef double [:] u = rdr.get_interp_profile('ua', Gr.zp_half, instant=instant_forcing, t_idx=gcm_tidx)
    cdef double [:] v = rdr.get_interp_profile('va', Gr.zp_half, instant=instant_forcing, t_idx=gcm_tidx)



    #Generate initial perturbations (here we are generating more than we need)
    cdef double [:] theta_pert = np.random.random_sample(Gr.dims.npg)
    cdef int jk
    #Now set the initial condition
    if 's' in PV.name_index:
        s_varshift = PV.get_varshift(Gr, 's')
        for i in xrange(Gr.dims.nlg[0]):
            ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
            for j in xrange(Gr.dims.nlg[1]):
                jshift = j * Gr.dims.nlg[2]
                for k in xrange(Gr.dims.nlg[2]):
                    ijk = ishift + jshift + k
                    jk = ishift + jshift + k 
                    PV.values[u_varshift + ijk] = u[k]
                    PV.values[v_varshift + ijk] = v[k]
                    PV.values[w_varshift + ijk] = 0.0
                    PV.values[s_varshift + ijk] = Th.entropy(RS.p0_half[k],t[k],qt[k],0.0,0.0)
                    PV.values[qt_varshift + ijk] = qt[k]
                    if Gr.zpl_half[k] < 500.0:
                        PV.values[s_varshift + ijk] = PV.values[s_varshift + ijk]  + (theta_pert[jk] - 0.5)*0.3
    else:
        thli_varshift = PV.get_varshift(Gr, 'thli')
        for i in xrange(Gr.dims.nlg[0]):
            ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
            for j in xrange(Gr.dims.nlg[1]):
                jshift = j * Gr.dims.nlg[2]
                for k in xrange(Gr.dims.nlg[2]):
                    ijk = ishift + jshift + k
                    PV.values[u_varshift + ijk] = u[k]
                    PV.values[v_varshift + ijk] = v[k]
                    PV.values[w_varshift + ijk] = 0.0
                    PV.values[thli_varshift + ijk] = thetali_c(RS.p0_half[k], t[k], 0.0, 0.0, 0.0, Th.get_lh(t[k]))
                    PV.values[qt_varshift + ijk] = qt[k]
                    if Gr.zpl_half[k] < 500.0:
                        PV.values[thli_varshift + ijk] = PV.values[thli_varshift + ijk]  + (theta_pert[ijk] - 0.5)*0.3

    return

def InitGCMVarying(namelist, Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa , LatentHeat LH):

    #Generate the reference profiles
    data_path = namelist['gcm']['file']
    try:
        griddata = namelist['gcm']['griddata']
    except:
        griddata = False
    if griddata:
        lat = namelist['gcm']['lat']
        lon = namelist['gcm']['lon']
    else:
        site = namelist['gcm']['site']

    if griddata:
        rdr = cfreader_grid(data_path, lat, lon)
    else:
        rdr = cfreader(data_path, site)

    RS.Pg = rdr.get_timeseries_mean('ps', instant=False, t_idx=0)
    RS.Tg = rdr.get_timeseries_mean('ts', instant=False, t_idx=0)
    RS.qtg = rdr.get_profile_mean('hus', instant=False, t_idx=0)[0]

    RS.u0 = 0.0
    RS.v0 = 0.0

    RS.initialize(Gr, Th, NS, Pa)

    try:
        random_seed_factor = namelist['initialization']['random_seed_factor']
    except:
        random_seed_factor = 1

    np.random.seed(Pa.rank * random_seed_factor)

    cdef:
        Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
        Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
        Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
        Py_ssize_t s_varshift
        Py_ssize_t thli_varshift
        Py_ssize_t qt_varshift = PV.get_varshift(Gr, 'qt')
        Py_ssize_t i,j,k
        Py_ssize_t ishift, jshift, e_varshift
        Py_ssize_t ijk

    cdef double [:] t = rdr.get_interp_profile('ta', Gr.zp_half, instant=True, t_idx=0)#interp_pchip(Gr.zp_half, z_in, np.log(t_in))
    cdef double [:] qt = rdr.get_interp_profile('hus', Gr.zp_half, instant=True, t_idx=0)
    cdef double [:] u = rdr.get_interp_profile('ua', Gr.zp_half, instant=True, t_idx=0)
    cdef double [:] v = rdr.get_interp_profile('va', Gr.zp_half, instant=True, t_idx=0)

    #Generate initial perturbations (here we are generating more than we need)
    cdef double [:] theta_pert = np.random.random_sample(Gr.dims.npg)
    cdef int jk
    #Now set the initial condition
    s_varshift = PV.get_varshift(Gr, 's')
    for i in xrange(Gr.dims.nlg[0]):
        ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
        for j in xrange(Gr.dims.nlg[1]):
            jshift = j * Gr.dims.nlg[2]
            for k in xrange(Gr.dims.nlg[2]):
                ijk = ishift + jshift + k
                jk = ishift + jshift + k
                PV.values[u_varshift + ijk] = u[k]
                PV.values[v_varshift + ijk] = v[k]
                PV.values[w_varshift + ijk] = 0.0
                PV.values[s_varshift + ijk] = Th.entropy(RS.p0_half[k],t[k],qt[k],0.0,0.0)
                PV.values[qt_varshift + ijk] = qt[k]
                if Gr.zpl_half[k] < 500.0:
                    PV.values[s_varshift + ijk] = PV.values[s_varshift + ijk]  + (theta_pert[jk] - 0.5)*0.3

    return


def FillAlteredFields(namelist, Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
                       ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa , LatentHeat LH, Restart.Restart Restart):

    cdef:
        Py_ssize_t i, j, k, ijk, ishift, jshift, var_shift
        Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
        Py_ssize_t jstride = Gr.dims.nlg[2]

    #First get the files to initialzie from
    #xl = (Gr.dims.indx_lo[0] + 1.0 + np.arange(Gr.dims.nl[0], dtype=np.double)) * Gr.dims.dx[0]
    #yl = (Gr.dims.indx_lo[1] + 1.0 + np.arange(Gr.dims.nl[1], dtype=np.double)) * Gr.dims.dx[1]

    #xl_half = (Gr.dims.indx_lo[0] + 1.0 + np.arange(Gr.dims.nl[0], dtype=np.double)) * Gr.dims.dx[0] - Gr.dims.dx[0] * 0.5
    #yl_half = (Gr.dims.indx_lo[1] + 1.0 + np.arange(Gr.dims.nl[1], dtype=np.double)) * Gr.dims.dx[1] - Gr.dims.dx[1] * 0.5


    data_tmp = np.zeros((Gr.dims.nl[0], Gr.dims.nl[1], Gr.dims.nlg[2]), dtype=np.double)
    cdef double [:,:,:] data_tmp_pt = data_tmp

    np.random.seed(Pa.rank)
    cdef double [:] w_pert = (np.random.random_sample(Gr.dims.npg) - 0.5)
    cdef int count = 0


    print(np.array(w_pert))
    #import time; time.sleep(Pa.rank)
    for v in PV.index_name:
        count = 0
        print(v)
        var_shift = PV.get_varshift(Gr,v)

        #Read dataset
        rt_grp = nc.Dataset(Restart.fields_path + '/' + v + '_scratch.nc', 'r')

        x = rt_grp.variables['xh'][:]
        y = rt_grp.variables['yh'][:]

        xl = (Gr.dims.indx_lo[0] + 1.0 + np.arange(Gr.dims.nl[0], dtype=np.double)) * Gr.dims.dx[0] - Gr.dims.dx[0] * 0.5
        yl = (Gr.dims.indx_lo[1] + 1.0 + np.arange(Gr.dims.nl[1], dtype=np.double)) * Gr.dims.dx[1] - Gr.dims.dx[1] * 0.5


        #print "Gr", np.min(np.array(Gr.xl_half)[Gr.dims.gw:-Gr.dims.gw]), np.max(Gr.xl_half[Gr.dims.gw:-Gr.dims.gw]), np.min(np.array(Gr.yl_half)[Gr.dims.gw:-Gr.dims.gw]), np.max(Gr.yl_half[Gr.dims.gw:-Gr.dims.gw])
        #print 'x', np.array(xl).min(), np.array(xl).max() , 'y', np.array(yl).min(), np.array(yl).max()
        #import sys; sys.exit()

        if v == 'u':
            x = rt_grp.variables['x'][:]
            xl = (Gr.dims.indx_lo[0] + 1.0 + np.arange(Gr.dims.nl[0], dtype=np.double)) * Gr.dims.dx[0]
        elif v == 'v':
            #yl = np.copy(np.array(Gr.yl)[Gr.dims.gw:-Gr.dims.gw])
            yl = (Gr.dims.indx_lo[1] + 1.0 + np.arange(Gr.dims.nl[1], dtype=np.double)) * Gr.dims.dx[1]
            y = rt_grp.variables['y'][:]

        #print rt_grp.variables['data'].shape
        #print np.shape(data_tmp), np.shape(x), np.shape(y), Gr.dims.nl[0], Gr.dims.nl[1]
        for k in range(Gr.dims.gw,Gr.dims.nlg[2]-Gr.dims.gw):
            data_k = k - Gr.dims.gw
            f_interp = interpolate.interp2d(x, y, rt_grp.variables['data'][:,:,data_k], kind='linear', bounds_error=True)
            data_tmp[:,:,k] = f_interp(yl, xl)

        if v == 'w':
            print('Adding perturbations')
            with nogil:
                for i in xrange(Gr.dims.nl[0]):
                    for j in xrange(Gr.dims.nl[1]):
                        for k in xrange(Gr.dims.nlg[2]):
                            if k <= 10:
                                data_tmp_pt[i,j,k] = data_tmp_pt[i,j,k]  +  w_pert[count]
                                count += 1


        #Now fill in prgnostic variables array
        with nogil:
            for i in xrange(Gr.dims.nl[0]):
                ishift =  (i + Gr.dims.gw) * Gr.dims.nlg[1] * Gr.dims.nlg[2]
                for j in xrange(Gr.dims.nl[1]):
                    jshift = (j + Gr.dims.gw) * Gr.dims.nlg[2]
                    for k in xrange(Gr.dims.nlg[2]):
                        ijk = ishift + jshift + k
                        PV.values[var_shift + ijk] = data_tmp_pt[i,j,k]

        rt_grp.close()

    return

def AuxillaryVariables(nml, PrognosticVariables.PrognosticVariables PV,
                       DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):

    casename = nml['meta']['casename']
    if casename == 'SMOKE':
        PV.add_variable('smoke', 'kg/kg', 'smoke', 'radiatively active smoke', "sym", "scalar", Pa)
        return
    return

from scipy.interpolate import pchip, interp1d
def interp_pchip(z_out, z_in, v_in, pchip_type=True):
    if pchip_type:
        p = pchip(z_in, v_in, extrapolate=True)
        #p = interp1d(z_in, v_in, kind='linear', fill_value='extrapolate')
        return p(z_out)
    else:
        return np.interp(z_out, z_in, v_in)


def integral_interp(p_half_gcm, variable_gcm):

    int_var = np.zeros(p_half_gcm.shape[0], dtype=np.double)
    for k in range(variable_gcm.shape[0]-1, -1, -1):
        dsigma = -(p_half_gcm[k] - p_half_gcm[k+1])/9.81
        int_var[k] = int_var[k+1] + variable_gcm[k] * dsigma


    #import pylab as plt
    #plt.plot(int_var)
    #plt.show()

    return


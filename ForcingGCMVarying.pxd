cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
from NetCDFIO cimport NetCDFIO_Stats
cimport ParallelMPI
from Thermodynamics cimport LatentHeat, ClausiusClapeyron
cimport Thermodynamics
cimport TimeStepping

cdef class ForcingGCMVarying:
    cdef:
        bint griddata
        bint gcm_profiles_initialized
        int t_indx
        double lat
        double lon
        double coriolis_param
        double [:] temp
        double [:] sphum
        double [:] ucomp
        double [:] vcomp
        double [:] ug
        double [:] vg

        double forcing_frequency
        bint relax_scalar
        bint relax_wind
        double tau_scalar
        double z_i
        double z_r
        double tau_wind
        int gcm_tidx
        bint add_horiz_advection
        bint add_subsidence
        bint add_subsidence_wind
        bint add_coriolis
        bint add_ls_pgradient
        double [:] qt_tend_nudge
        double [:] t_tend_nudge
        double [:] u_tend_nudge
        double [:] v_tend_nudge
        double [:] qt_tend_adv
        double [:] t_tend_adv
        double [:] s_tend_adv
        double [:] qt_tend_hadv
        double [:] t_tend_hadv
        double [:] s_tend_hadv
        double [:] qt_tend_fluc
        double [:] t_tend_fluc
        double [:] omega_vv
        double [:] subsidence
        str file
        int site
    cpdef initialize(self, Grid.Grid Gr,ReferenceState.ReferenceState RS, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,  TimeStepping.TimeStepping TS,
                 ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

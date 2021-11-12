cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
from NetCDFIO cimport NetCDFIO_Stats
cimport ParallelMPI
from Thermodynamics cimport LatentHeat, ClausiusClapeyron
cimport Thermodynamics
cimport TimeStepping

cdef class ForcingGCMNew:
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

        bint relax_scalar
        bint relax_wind
        double tau_scalar
        double z_i
        double z_r
        double tau_wind
        int gcm_tidx
        bint instant_forcing
        bint add_advection
        bint add_horiz_adv_vert_fluc
        bint add_horiz_advection
        bint add_vert_fluctuation
        bint add_subsidence
        bint add_subsidence_wind
        bint instant_variance
        bint coherent_variance
        double hadv_variance_factor
        double sub_variance_factor
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
        double [:] qt_tend_hadv_tr
        double [:] t_tend_hadv
        double [:] t_tend_hadv_tr
        double [:] s_tend_hadv
        double [:] s_tend_hadv_tr
        double [:] qt_tend_fluc
        double [:] t_tend_fluc
        double [:] omega_vv
        double [:] subsidence
        double [:] subsidence_tr
        str file
        int site
    cpdef initialize(self, Grid.Grid Gr,ReferenceState.ReferenceState RS, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,  TimeStepping.TimeStepping TS,
                 ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

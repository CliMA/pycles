cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
from NetCDFIO cimport NetCDFIO_Stats
cimport ParallelMPI
from Thermodynamics cimport LatentHeat, ClausiusClapeyron
cimport Thermodynamics
cimport TimeStepping

cdef class ForcingGCMMean:
    cdef:
        bint gcm_profiles_initialized
        int t_indx
        double [:] ug
        double [:] vg
        double [:] subsidence
        double [:] temp
        double [:] shum
        double [:] temp_dt_hadv
        double [:] temp_dt_fino
        double [:] temp_dt_resid
        double [:] temp_dt_fluc
        double [:] shum_dt_vadv
        double [:] shum_dt_hadv
        double [:] shum_dt_resid
        double [:] shum_dt_fluc
        double [:] omega_vv


        double [:] u_dt_hadv
        double [:] u_dt_vadv
        double [:] u_dt_cof
        double [:] u_dt_pres
        double [:] u_dt_tot

        double [:] v_dt_hadv
        double [:] v_dt_vadv
        double [:] v_dt_cof
        double [:] v_dt_pres
        double [:] v_dt_tot

        double [:] p_gcm
        double [:] rho_gcm
        double [:] rho_half_gcm
        double coriolis_param
        bint relax_scalar
        double tau_scalar
        double [:] qt_tend_nudge
        double [:] t_tend_nudge
        str file
        double lat
        double lon
    cpdef initialize(self, Grid.Grid Gr,ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,  TimeStepping.TimeStepping TS,
                 ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

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
    cpdef initialize(self, Grid.Grid Gr,ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,  TimeStepping.TimeStepping TS,
                 ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

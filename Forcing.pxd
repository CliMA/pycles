cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
from NetCDFIO cimport NetCDFIO_Stats
cimport ParallelMPI
from Thermodynamics cimport LatentHeat, ClausiusClapeyron
cimport Thermodynamics
cimport TimeStepping

cdef class Forcing:
    cdef:
        object scheme
    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class ForcingNone:
    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class ForcingBomex:
    cdef:
        double [:] ug
        double [:] vg
        double [:] dtdt
        double [:] dqtdt
        double [:] subsidence
        double coriolis_param
    cpdef initialize(self, Grid.Grid Gr,ReferenceState.ReferenceState RS, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class ForcingSoares:
    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class ForcingSullivanPatton:
    cdef:
        double [:] ug
        double [:] vg
        double coriolis_param
    cpdef initialize(self, Grid.Grid Gr,ReferenceState.ReferenceState RS, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class ForcingGabls:
    cdef:
        double [:] ug
        double [:] vg
        double coriolis_param
    cpdef initialize(self, Grid.Grid Gr,ReferenceState.ReferenceState RS, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class ForcingDyCOMS_RF01:
    cdef:
        double [:] ug
        double [:] vg
        double divergence
        double [:] subsidence
        double coriolis_param
        bint rf02_flag
    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)


cdef class ForcingRico:
    cdef:
        double [:] ug
        double [:] vg
        double [:] dtdt
        double [:] dqtdt
        double [:] subsidence
        double coriolis_param
        Py_ssize_t momentum_subsidence
    cpdef initialize(self, Grid.Grid Gr,ReferenceState.ReferenceState RS, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)


cdef class ForcingCGILS:
    cdef:
        Py_ssize_t loc
        bint is_p2
        bint is_ctl_omega
        double [:] dtdt
        double [:] dqtdt
        double [:] subsidence
        double z_relax
        double z_relax_plus
        double tau_inverse
        double tau_vel_inverse
        double qt_floor
        Py_ssize_t floor_index
        double[:] gamma_zhalf
        double[:] gamma_z
        double [:] nudge_qt
        double [:] nudge_temperature
        double [:] nudge_u
        double [:] nudge_v
        double [:] source_qt_floor
        double [:] source_qt_nudge
        double [:] source_t_nudge
        double [:] source_u_nudge
        double [:] source_v_nudge
        double [:] source_s_nudge
        double [:] s_ls_adv

    cpdef initialize(self, Grid.Grid Gr,ReferenceState.ReferenceState RS, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)




cdef class ForcingZGILS:
    cdef:
        Py_ssize_t loc
        double [:] dtdt
        double [:] dqtdt
        double [:] subsidence
        double [:] ug
        double [:] vg
        double [:] source_rh_nudge
        double [:] source_qt_nudge
        double [:] source_t_nudge
        double [:] source_s_nudge
        double [:] s_ls_adv
        double coriolis_param
        double divergence
        double t_adv_max
        double qt_adv_max
        double tau_relax_inverse
        double alpha_h
        double h_BL
        ClausiusClapeyron CC
        AdjustedMoistAdiabat forcing_ref


    cpdef initialize(self, Grid.Grid Gr,ReferenceState.ReferenceState RS, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class ForcingTRMM_LBA:
    cdef:

        double [:] nudge_u
        double [:] nudge_v
        double [:] source_u_nudge
        double [:] source_v_nudge
        double tau_inverse

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class ForcingARM_SGP:
    cdef:

        double [:] ug
        double [:] vg
        double tau_inverse
        double coriolis_param
        double [:] dtdt
        double [:] dqtdt

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class ForcingGATE_III:
    cdef:

        double tau_inverse
        double [:] nudge_u
        double [:] nudge_v
        double [:] source_u_nudge
        double [:] source_v_nudge
        double [:] dtdt
        double [:] dqtdt

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class AdjustedMoistAdiabat:
    cdef:
        double [:] s
        double [:] qt
        double [:] temperature
        double [:] rv
        double (*L_fp)(double T, double Lambda) nogil
        double (*Lambda_fp)(double T) nogil
        Thermodynamics.ClausiusClapeyron CC
    cpdef get_pv_star(self, t)
    cpdef entropy(self,double p0, double T,double qt, double ql, double qi)
    cpdef eos(self, double p0, double s, double qt)
    cpdef initialize(self,  ParallelMPI.ParallelMPI Pa, double [:] pressure_array, Py_ssize_t n_levels,
                     double Pg, double Tg, double RH)
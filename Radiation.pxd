cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
from NetCDFIO cimport NetCDFIO_Stats
cimport ParallelMPI
cimport TimeStepping
cimport Surface
from Forcing cimport AdjustedMoistAdiabat

cdef class RadiationBase:
    cdef:
        double [:] heating_rate
        double [:] dTdt_rad
        ParallelMPI.Pencil z_pencil
        double srf_lw_down
        double srf_lw_up
        double srf_sw_down
        double srf_sw_up



    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef initialize_profiles(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                     NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, Surface.SurfaceBase Sur,
                 TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class RadiationNone(RadiationBase):
    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef initialize_profiles(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                     NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 Surface.SurfaceBase Sur, TimeStepping.TimeStepping TS,ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class RadiationDyCOMS_RF01(RadiationBase):
    cdef:
        double alpha_z
        double kap
        double f0
        double f1
        double divergence

    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef initialize_profiles(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                     NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 Surface.SurfaceBase Sur,TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class RadiationSmoke(RadiationBase):
    cdef:
        double f0
        double kap


    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef initialize_profiles(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                     NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, Surface.SurfaceBase Sur,
                 TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)


cdef class RadiationRRTM(RadiationBase):
    cdef:
        str profile_name
        bint modified_adiabat
        AdjustedMoistAdiabat reference_profile
        double Tg_adiabat
        double Pg_adiabat
        double RH_adiabat
        Py_ssize_t n_buffer
        Py_ssize_t n_ext
        double stretch_factor
        double patch_pressure
        double [:] p_ext
        double [:] t_ext
        double [:] rv_ext
        double [:] p_full
        double [:] pi_full


        double co2_factor
        double h2o_factor
        int dyofyr
        double scon
        double adjes
        double solar_constant
        double coszen
        double adif
        double adir
        double radiation_frequency
        double next_radiation_calculate

        double [:] o3vmr
        double [:] co2vmr
        double [:] ch4vmr
        double [:] n2ovmr
        double [:] o2vmr
        double [:] cfc11vmr
        double [:] cfc12vmr
        double [:] cfc22vmr
        double [:] ccl4vmr
        bint uniform_reliq


    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef initialize_profiles(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                     NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 Surface.SurfaceBase Sur, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cdef update_RRTM(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                     Surface.SurfaceBase Sur,ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr,  ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class RadiationTRMM_LBA(RadiationBase):
    cdef:
        double [:,:] rad
        double [:] rad_time
        double [:] rad_temp
        double [:] rad_cool
        double [:,:] rad_in
        double [:] z_in
     #   double [:] heating_rate
    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef initialize_profiles(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                     NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, Surface.SurfaceBase Sur,
                 TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)


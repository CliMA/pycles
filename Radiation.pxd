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

cdef class RadiationPrescribed(RadiationBase):
    cdef:
        bint gcm_profiles_initialized
        double [:] t_tend_rad
        double [:] s_tend_rad
        str file
        int site
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
        str file
        bint modified_adiabat
        bint griddata
        bint read_file
        bint time_varying
        bint radiation_initialized
        int t_indx
        int site
        double forcing_frequency
        double lat
        double lon
        AdjustedMoistAdiabat reference_profile
        double Tg_adiabat
        double Pg_adiabat
        double RH_adiabat
        Py_ssize_t n_buffer
        Py_ssize_t n_ext
        double stretch_factor
        double patch_pressure
        double toa_lw_down
        double toa_lw_up
        double toa_sw_down
        double toa_sw_up
        double srf_lw_down_clr
        double srf_lw_up_clr
        double srf_sw_down_clr
        double srf_sw_up_clr
        double toa_lw_down_clr
        double toa_lw_up_clr
        double toa_sw_down_clr
        double toa_sw_up_clr
        double [:] lw_flux_up
        double [:] lw_flux_up_clr
        double [:] lw_flux_down
        double [:] lw_flux_down_clr
        double [:] sw_flux_up
        double [:] sw_flux_up_clr
        double [:] sw_flux_down
        double [:] sw_flux_down_clr
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
        double toa_sw
        double coszen
        bint time_varying_coszen
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
        bint clear_sky


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

cdef class RadiationGCMGrey(RadiationBase):

    cdef:
        double insolation
        double solar_constant
        double lw_tau0_eqtr
        double lw_tau0_pole
        double atm_abs
        double sw_diff
        double lw_linear_frac
        double albedo_value
        double lw_tau_exponent
        double sw_tau_exponent
        double del_sol
        double lat_rad
        str file

        double lw_tau0
        double sw_tau0

        double odp
        double [:] p_gcm
        double [:] alpha_gcm
        double [:] t_gcm
        double [:] z_gcm
        double [:] p_ext
        double [:] t_ext
        double [:] z_ext
        double [:] alpha_ext

        Py_ssize_t n_ext_profile

        double [:] lw_tau
        double [:] sw_tau

        double [:] lw_dtrans
        double [:] lw_down
        double [:] sw_down
        double [:] lw_up
        double [:] sw_up
        double [:] net_flux
        double [:] h_profile
        double [:] dsdt_profile

        double t_ref

        double p0_les_min
        double dp
        double lat
        double lon

    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef initialize_profiles(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                     NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 Surface.SurfaceBase Sur, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)


cdef class RadiationGCMGreyVarying(RadiationBase):

    cdef:
        bint gcm_profiles_initialized
        int t_indx
        double insolation
        double solar_constant
        double lw_tau0_eqtr
        double lw_tau0_pole
        double atm_abs
        double sw_diff
        double lw_linear_frac
        double albedo_value
        double lw_tau_exponent
        double sw_tau_exponent
        double del_sol
        double lat
        str file

        double lw_tau0
        double sw_tau0

        double odp
        double [:] p_gcm
        double [:] alpha_gcm
        double [:] t_gcm
        double [:] z_gcm
        double [:] p_ext
        double [:] t_ext
        double [:] z_ext
        double [:] alpha_ext

        Py_ssize_t n_ext_profile

        double [:] lw_tau
        double [:] sw_tau

        double [:] lw_dtrans
        double [:] lw_down
        double [:] sw_down
        double [:] lw_up
        double [:] sw_up
        double [:] net_flux
        double [:] h_profile
        double [:] dsdt_profile

        double t_ref

        double p0_les_min
        double dp

    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef initialize_profiles(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                     NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 Surface.SurfaceBase Sur, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)


cdef class RadiationGCMGreyMean(RadiationBase):

    cdef:
        bint gcm_profiles_initialized
        int t_indx
        double insolation
        double solar_constant
        double lw_tau0_eqtr
        double lw_tau0_pole
        double atm_abs
        double sw_diff
        double lw_linear_frac
        double albedo_value
        double lw_tau_exponent
        double sw_tau_exponent
        double del_sol
        double lat_rad
        str file

        double lw_tau0
        double sw_tau0

        double odp
        double [:] p_gcm
        double [:] alpha_gcm
        double [:] t_gcm
        double [:] z_gcm
        double [:] p_ext
        double [:] t_ext
        double [:] z_ext
        double [:] alpha_ext

        Py_ssize_t n_ext_profile

        double [:] lw_tau
        double [:] sw_tau

        double [:] lw_dtrans
        double [:] lw_down
        double [:] sw_down
        double [:] lw_up
        double [:] sw_up
        double [:] net_flux
        double [:] h_profile
        double [:] dsdt_profile

        double t_ref

        double p0_les_min
        double dp
        double lat
        double lon

    cpdef initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef initialize_profiles(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                     NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 Surface.SurfaceBase Sur, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

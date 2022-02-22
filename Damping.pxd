cimport ParallelMPI as ParallelMPI
cimport PrognosticVariables as PrognosticVariables
cimport Grid as Grid
cimport ReferenceState
cimport DiagnosticVariables
cimport TimeStepping

cdef class Damping:
    cdef:
        object scheme
    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS)
cdef class Dummy:
    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS)



cdef class RayleighGCMMeanNudge:
    cdef:
        double z_d  # Depth of damping layer
        double gamma_r  # Inverse damping timescale
        double tau_max  # Maximum damping timescale
        double z_r  # Depth of relaxation layer
        double tau_wind  # Wind relaxation timescale
        double[:] gamma_zhalf
        double[:] gamma_z
        double[:] xi_z
        double[:] ucomp
        double[:] vcomp
        double[:] tend_flat
        double[:] tend_flat_half
        double tend_flat_z_d
        double [:] dt_tg_total
        double [:] dt_qg_total
        bint gcm_profiles_initialized
        bint truncate
        bint damp_w
        bint damp_scalar
        bint relax_wind
        int t_indx
        str file
        double lat
        double lon

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS)

cdef class RayleighGCMNew:
    cdef:
        double z_d  # Depth of damping layer
        double gamma_r  # Inverse damping timescale
        double tau_max  # Maximum damping timescale
        double[:] gamma_zhalf
        double[:] gamma_z
        double[:] xi_z
        double[:] ucomp
        double[:] vcomp
        double [:] dt_tg_total
        double [:] dt_qg_total
        bint griddata
        bint gcm_profiles_initialized
        bint truncate
        bint damp_w
        bint damp_scalar
        int t_indx
        str file
        int site
        double lat
        double lon

    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS)


cdef class RayleighGCMMean:
    cdef:
        double z_d  # Depth of damping layer
        double gamma_r  # Inverse damping timescale
        double[:] gamma_zhalf
        double[:] gamma_z
        double[:] tend_flat
        double[:] tend_flat_half
        double tend_flat_z_d
        double [:] dt_tg_total
        double [:] dt_qg_total
        bint gcm_profiles_initialized
        int t_indx
        str file


    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS)


cdef class RayleighGCMVarying:
    cdef:
        double z_d  # Depth of damping layer
        double gamma_r  # Inverse damping timescale
        double[:] gamma_zhalf
        double[:] gamma_z
        double[:] tend_flat
        double[:] tend_flat_half
        double tend_flat_z_d
        double [:] dt_tg_total
        double [:] dt_qg_total
        bint gcm_profiles_initialized
        int t_indx
        str file


    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS)

cdef class Rayleigh:
    cdef:
        double z_d  # Depth of damping layer
        double gamma_r  # Inverse damping timescale
        double[:] gamma_zhalf
        double[:] gamma_z
        double[:] tend_flat
        double[:] tend_flat_half
        double tend_flat_z_d
        double [:] dt_tg_total
        double [:] dt_qg_total
        bint gcm_profiles_initialized
        int t_indx



    cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState RS)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                 DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa, TimeStepping.TimeStepping TS)


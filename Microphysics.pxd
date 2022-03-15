cimport Grid
cimport PrognosticVariables
cimport DiagnosticVariables
cimport ReferenceState
cimport ParallelMPI
cimport TimeStepping
from NetCDFIO cimport NetCDFIO_Stats
from Thermodynamics cimport LatentHeat, ClausiusClapeyron
from libc.math cimport pow, fmax, fmin, tanh
include 'parameters.pxi'
include 'parameters_micro.pxi'

cdef:
    double lambda_constant(double T) nogil
    double lambda_T_clima(double T) nogil
    double lambda_T(double T) nogil
    double latent_heat_constant(double T, double Lambda) nogil
    double latent_heat_variable_with_T(double T, double Lambda) nogil
    double latent_heat_variable_with_lambda(double T, double Lambda) nogil

cdef class No_Microphysics_Dry:
    # Make the thermodynamics_type member available from Python-Space
    cdef public:
        str thermodynamics_type

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, Th, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, Th, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class No_Microphysics_SA:
    # Make the thermodynamics_type member available from Python-Space
    cdef public:
        str thermodynamics_type
    cdef:
        double (*L_fp)(double T, double Lambda) nogil
        double (*Lambda_fp)(double T) nogil
        double ccn
        Py_ssize_t order
        bint cloud_sedimentation
        bint stokes_sedimentation
    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, Th, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, Th, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class Microphysics_SB_Liquid:
    # Make the thermodynamics_type member available from Python-Space
    cdef public:
        str thermodynamics_type

    cdef:
        double (*L_fp)(double T, double Lambda) nogil
        double (*Lambda_fp)(double T) nogil
        double (*compute_rain_shape_parameter)(double density, double qr, double Dm) nogil
        double (*compute_droplet_nu)(double density, double ql) nogil
        ClausiusClapeyron CC
        double ccn
        Py_ssize_t order
        bint cloud_sedimentation
        bint stokes_sedimentation

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, Th, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, Th, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class Microphysics_T_Liquid:
    cdef public:
        str thermodynamics_type

    cdef:
        double (*L_fp)(double T, double Lambda) nogil
        double (*Lambda_fp)(double T) nogil
        ClausiusClapeyron CC

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, Th, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, Th, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)

cdef class Microphysics_CLIMA_1M:

    cdef public:
        str thermodynamics_type

    cdef:
        double (*L_fp)(double T, double Lambda) nogil
        double (*Lambda_fp)(double T) nogil
        ClausiusClapeyron CC
        Py_ssize_t order

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV,\
                     DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS,\
                     ParallelMPI.ParallelMPI Pa)

    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, Th,\
                 PrognosticVariables.PrognosticVariables PV,\
                 DiagnosticVariables.DiagnosticVariables DV,\
                 TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, Th,\
                   PrognosticVariables.PrognosticVariables PV,\
                   DiagnosticVariables.DiagnosticVariables DV,\
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)


cdef inline double lambda_constant(double T) nogil:
    return 1.0

cdef inline double lambda_T_clima(double T) nogil:
    cdef:
        double Lambda = 0.0

    if T > CLIMA_T_icenuc and T <= Tf:
        Lambda = pow((T - CLIMA_T_icenuc)/(Tf - CLIMA_T_icenuc), CLIMA_pow_icenuc)
    elif T > Tf:
        Lambda = 1.0
    else:
        Lambda = 0.0

    return Lambda

cdef inline double lambda_T(double T) nogil:
    cdef:
        double Twarm = 273.15
        double Tcold = 263.15
        double Lambda = 0.0

    #POW_N can be modified in generate_parameters_a1m.py
    if T > Tcold and T <= Twarm:
        Lambda = pow((T - Tcold)/(Twarm - Tcold), POW_N)
    elif T > Twarm:
        Lambda = 1.0
    else:
        Lambda = 0.0

    return Lambda

cdef inline double latent_heat_constant(double T, double Lambda) nogil:
    return 2.501e6

cdef inline double latent_heat_variable_with_T(double T, double Lambda) nogil:
    cdef:
        double TC = T - 273.15
    return (2500.8 - 2.36 * TC + 0.0016 * TC *
            TC - 0.00006 * TC * TC * TC) * 1000.0

cdef inline double latent_heat_variable_with_lambda(double T, double Lambda) nogil:
    cdef:
        double Lv = CLIMA_LH_v0
        double Ls = CLIMA_LH_s0

    return (Lv * Lambda) + (Ls * (1.0 - Lambda))

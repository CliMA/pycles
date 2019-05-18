cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
cimport ParallelMPI
cimport TimeStepping
from libc.math cimport pow, fmax, fmin, tanh, exp
include 'parameters_micro.pxi'
from Thermodynamics cimport ClausiusClapeyron
from NetCDFIO cimport NetCDFIO_Fields, NetCDFIO_Stats

cdef:
    double lambda_constant_Arctic(double T) nogil
    double latent_heat_constant_Arctic(double T, double Lambda) nogil
    double lambda_Arctic(double T) nogil
    double latent_heat_Arctic(double T, double Lambda) nogil
    double latent_heat_variable_Arctic(double T, double Lambda) nogil

cdef inline double lambda_constant_Arctic(double T) nogil:
    return 1.0

cdef inline double latent_heat_constant_Arctic(double T, double Lambda) nogil:
    return 2.501e6

cdef inline double lambda_Arctic(double T) nogil:
    cdef:
        double Twarm = 273.15
        double Tcold = 263.15
        double Lambda = 0.0

    #POW_N can be modified in generate_parameters_a1m.py

    #Lambda = 0.5 + 0.5 * tanh((T - 266.65000072654806)/2.6409572185027406)


    if T > Tcold and T <= Twarm:
        Lambda = pow((T - Tcold)/(Twarm - Tcold), POW_N)
    elif T > Twarm:
        Lambda = 1.0
    else:
        Lambda = 0.0

    return Lambda

cdef inline double latent_heat_Arctic(double T, double Lambda) nogil:
    cdef:
        double Lv = 2.501e6
        double Ls = 2.8334e6

    return (Lv * Lambda) + (Ls * (1.0 - Lambda))

cdef inline double latent_heat_variable_Arctic(double T, double Lambda) nogil:
    cdef:
        double TC = T - 273.15
    return (2500.8 - 2.36 * TC + 0.0016 * TC *
            TC - 0.00006 * TC * TC * TC) * 1000.0

cdef inline double lambda_Hu2010(double T) nogil:
    #Equation 1 and 2 from Hu et al. (2010)
    cdef:
        double TC = T - 273.15 #Convert to Celcius
        double p = 5.3608 + 0.4025 * TC + 0.08387 * TC * TC + 0.007182 * TC * TC * TC + \
            2.39e-4 * TC * TC * TC * TC + 2.87e-6 * TC * TC * TC * TC * TC
    return 1.0 / (1.0 + exp(-p))

cdef inline double lambda_logistic(double T) nogil:
    #Cubic logistic function fitted to the Hu et al. (2009) expression
    #(Temperature is in Kelvin)
    cdef:
        double k = 0.32928
        double Tlf = 241.92
        double Twarm = 273.15
        double Tcold = 235.0
        double Lambda = 0.0

    if Tcold <= T <= Twarm:
        Lambda = 1.0/pow((1.0 + exp(-k*(T - Tlf))), 3)
    elif T > Twarm:
        Lambda = 1.0
    else:
        Lambda = 0.0
    return Lambda


cdef class Microphysics_Arctic_1M:


    cdef public:
        str thermodynamics_type

    cdef:
        double ccn
        double n0_ice_input
        Py_ssize_t order

        double (*L_fp)(double T, double Lambda) nogil
        double (*Lambda_fp)(double T) nogil

        ClausiusClapeyron CC

        double [:] evap_rate
        double [:] precip_rate
        double [:] melt_rate

    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                     NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, Th,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                 TimeStepping.TimeStepping TS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, Th, PrognosticVariables.PrognosticVariables PV,
                   DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef ice_stats(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
                    DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
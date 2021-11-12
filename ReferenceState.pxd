cimport Grid
cimport Restart
from NetCDFIO cimport NetCDFIO_Stats
cimport ParallelMPI
cdef class ReferenceState:
    cdef:
        public double [:] p0
        public double [:] p0_half
        public double [:] alpha0
        public double [:] alpha0_half
        public double [:] rho0
        public double [:] rho0_half
        public double [:] thetali0
        public double [:] thetali0_half

        public double [:] p0_global
        public double [:] p0_half_global
        public double [:] alpha0_global
        public double [:] alpha0_half_global
        public double [:] rho0_global
        public double [:] rho0_half_global
        public double [:] thetali0_global
        public double [:] thetali0_half_global

        double sg

    cdef public:
        #These public values should be set in the case initialization routine
        double Tg  #Temperature at ground level
        double Pg  #Pressure at ground level
        double qtg #Surface total water mixing ratio
        double u0 #u velocity removed in Galilean transformation
        double v0 #v velocity removed in Galilean transformation

    cpdef restart(self, Grid.Grid Gr, Restart.Restart Re)
    cpdef init_from_restart(self, Grid.Grid Gr, Restart.Restart Re, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)


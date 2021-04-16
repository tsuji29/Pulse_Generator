import numpy as np
import scipy
import copy
from scipy import interpolate 
import numpy.linalg as LA

try:
    from ._func import *
except ImportError:
    from _func import *

##########  copy from labber drivers ##############

class Pulse:
    """Represents physical pulses played by an AWG.

    Parameters
    ----------
    complex_value : bool
        If True, pulse has both I and Q, otherwise it's real valued.
        Phase, frequency and drag only applies for complex_value waveforms.

    Attributes
    ----------
    amplitude : float
        Pulse amplitude.
    width : float
        Pulse width.
    plateau : float
        Pulse plateau.
    frequency : float
        SSB frequency.
    phase : float
        Pulse phase.
    use_drag : bool
        If True, applies DRAG correction.
    drag_coefficient : float
        Drag coefficient.
    drag_detuning : float
        Applies a frequnecy detuning for DRAG pulses.
    start_at_zero : bool
        If True, forces the pulse to start in 0.

    """

    def __init__(self,complex_value = False):

        # set variables
        self.amplitude = 0.5
        self.width = 10E-9
        self.plateau = 0.0
        self.frequency = 0.0
        self.phase = 0.0
        self.use_drag = False
        self.drag_coefficient = 0.0
        self.drag_detuning = 0.0
        self.start_at_zero = False
        self.complex_value = complex_value
        self.pulse_trunc = False
        self.trunc_start = 0
        self.trunc_end = 0

        # For IQ mixer corrections
        self.iq_ratio = 1.0
        self.iq_skew = 0.0

    def show_params(self):
        print(self.__dict__)

    def total_duration(self):
        """Get the total duration for the pulse.

        Returns
        -------
        float
            Total duration in seconds.

        """
        raise NotImplementedError()

    def calculate_envelope(self, t0, t):
        """Calculate pulse envelope.

        Parameters
        ----------
        t0 : float
            Pulse position, referenced to center of pulse.

        t : numpy array
            Array with time values for which to calculate the pulse envelope.

        Returns
        -------
        waveform : numpy array
            Array containing pulse envelope.

        """
        raise NotImplementedError()

    def calculate_waveform(self, t0, t):
        """Calculate pulse waveform including phase shifts and SSB-mixing.

        Parameters
        ----------
        t0 : float
            Pulse position, referenced to center of pulse.

        t : numpy array
            Array with time values for which to calculate the pulse waveform.

        Returns
        -------
        waveform : numpy array
            Array containing pulse waveform.

        """
        y = self.calculate_envelope(t0, t)
        # Make sure the waveform is zero outside the pulse
        y[t < (t0 - self.total_duration() / 2)] = 0
        y[t > (t0 + self.total_duration() / 2)] = 0

        if self.pulse_trunc == True:
            y[t < self.trunc_start] = 0
            y[t >= self.trunc_end] = 0

        if self.use_drag:
            beta = self.drag_coefficient / (t[1] - t[0])
            y = y + 1j * beta * np.gradient(y)
            y = y * np.exp(1j * 2 * np.pi * self.drag_detuning *
                           (t - t0 + 0*self.total_duration() / 2))

        # Apply phase and SSB
        phase = self.phase
        # single-sideband mixing, get frequency
        omega = 2 * np.pi * self.frequency
        # apply SSBM transform
        data_i = self.iq_ratio * (y.real * np.cos(omega * t - phase) +
                                    - y.imag * np.cos(omega * t - phase +
                                                    np.pi / 2))
        data_q = (y.real * np.sin(omega * t - phase + self.iq_skew) +
                    -y.imag * np.sin(omega * t - phase + self.iq_skew +
                                    np.pi / 2))
        


        if self.complex_value:
            return data_i + 1j * data_q
        else:
            return data_i


class Gaussian(Pulse):
    def __init__(self, complex_value = False):
        super().__init__(complex_value)
        self.truncation_range = 5

    def total_duration(self):
        return self.width + self.plateau

    def calculate_envelope(self, t0, t):
        # width == 2 * truncation_range * std
        if self.truncation_range == 0:
            std = np.inf
        else:
            std = self.width / 2 / self.truncation_range
        values = np.zeros_like(t)
        if self.plateau == 0:
            # pure gaussian, no plateau
            if std > 0:
                values = np.exp(-(t - t0)**2 / (2 * std**2))
        else:
            # add plateau
            values = np.array(
                ((t >= (t0 - self.plateau / 2)) & (t <
                                                   (t0 + self.plateau / 2))),
                dtype=float)
            if std > 0:
                # before plateau
                values += ((t < (t0 - self.plateau / 2)) * np.exp(
                    -(t - (t0 - self.plateau / 2))**2 / (2 * std**2)))
                # after plateau
                values += ((t >= (t0 + self.plateau / 2)) * np.exp(
                    -(t - (t0 + self.plateau / 2))**2 / (2 * std**2)))

        mask = (t>=(t0-self.total_duration()/2)) & (t<=(t0+self.total_duration()/2))
        values[~mask] = 0
        if self.start_at_zero:
            values[mask] = values[mask] - values[mask].min()
            # renormalize max value to 1
            values = values / values.max()
        values = values * self.amplitude

        return values


class Ramp(Pulse):
    def total_duration(self):
        return self.width + self.plateau

    def calculate_envelope(self, t0, t):
        # rising and falling slopes
        vRise = ((2*t - (2*t0 - self.plateau - self.width)) / self.width)
        vRise[vRise < 0.0] = 0.0
        vRise[vRise > 1.0] = 1.0
        vFall = (((2*t0 + self.plateau + self.width) - 2*t) / self.width)
        vFall[vFall < 0.0] = 0.0
        vFall[vFall > 1.0] = 1.0
        values = vRise * vFall

        values = values * self.amplitude

        return values


class Square(Pulse):
    def total_duration(self):
        return self.width + self.plateau

    def calculate_envelope(self, t0, t):
        # reduce risk of rounding errors by putting checks between samples
        # if len(t) > 1:
        #     t0 += (t[1] - t[0]) / 2.0

        values = ((t >= (t0 - (self.width + self.plateau) / 2)) &
                  (t < (t0 + (self.width + self.plateau) / 2)))

        values = values * self.amplitude

        return values


class ReadoutSquare(Pulse):
    def __init__(self, complex_value = False):
        super().__init__(complex_value)
        self.plateau = []
        self.rel_amplitude = []

    def total_duration(self):
        return np.sum(self.plateau)

    def calculate_envelope(self, t0, t):
        # reduce risk of rounding errors by putting checks between samples
        # if len(t) > 1:
        #     t0 += (t[1] - t[0]) / 2.0

        values = np.zeros_like(t)

        t_start = t0 - self.total_duration()/2
        for i, l in enumerate(self.plateau):
            values[(t>=t_start)&(t<t_start+l)] = self.rel_amplitude[i]
            t_start += l

        values = values * self.amplitude

        return values


class Cosine(Pulse):
    def __init__(self, complex_value = False):
        super().__init__(complex_value)
        self.half_cosine = False

    def total_duration(self):
        return self.width + self.plateau

    def calculate_envelope(self, t0, t):
        tau = self.width
        values = np.zeros_like(t) 
        x1 = ( abs(t - t0) <=  self.plateau/2 + self.width/2)   
        x2 = ( abs(t - t0)  <= self.plateau/2 )
        if self.half_cosine:
            values[x1]= self.amplitude * np.sin(np.pi * (self.plateau/2 + self.width/2 - abs(t[x1] - t0)) / tau)
            values[x2] = self.amplitude
        else:
            values[x1 ]= self.amplitude / 2 * (1 - np.cos(2 * np.pi * (self.plateau/2 + self.width/2 - abs(t[x1] - t0)) / tau))
            values[x2] = self.amplitude
        return values


class Fourier(Pulse):
    def __init__(self, complex_value = False):
        super().__init__(complex_value)
        self.sine=False
        self.F_Terms = 2
        self.Lcoeff = np.array([1,0.1])

    def total_duration(self):
        return self.width + self.plateau

    def calculate_envelope(self, t0, t):
        tau = self.width
        values = np.zeros_like(t) 
        x1 = ( abs(t - t0) <=  self.plateau/2 + self.width/2)   
        x2 = ( abs(t - t0) < self.plateau/2 )
        if self.sine:
            for i in range(self.F_Terms):
                values[x1] += self.Lcoeff[i] * np.sin(np.pi * (2*i+1) * (self.plateau/2 + self.width/2 - (t[x1] - t0)) / tau)
            values[x1] = values[x1] * self.amplitude
            values[x2] = self.amplitude
        else:
            for i in range(self.F_Terms):
                values[x1] += self.Lcoeff[i] * 0.5 * (1 - np.cos(2 * np.pi * (i+1) * (self.plateau/2 + self.width/2 -(t[x1] - t0)) / tau))
            values[x1] = values[x1] * self.amplitude
            values[x2] = self.amplitude
        return values

class CosH(Pulse):

    def __init__(self, complex_value = False):
        super().__init__(complex_value)
        self.beta=10

    def total_duration(self):
        return self.width + self.plateau

    def calculate_envelope(self,t0,t):
        values = np.zeros_like(t) 
        x1 = ( abs(t - t0) <=  self.plateau/2 + self.width/2)   
        x2 = ( abs(t - t0) < self.plateau/2 )
        values[x1] = ( np.cosh(0.5*self.beta) - np.cosh( (abs(t[x1]-t0)-self.plateau/2) /self.width*self.beta) )/( np.cosh(0.5*self.beta)-1 )
        values[x2] = 1

        return values * self.amplitude


class CosH_Full(Pulse):

    def __init__(self, complex_value = False):
        super().__init__(complex_value)
        self.beta=10
        self.relative_len=0.5   ## relative length of first raising edge

    def total_duration(self):
        return self.width + self.plateau

    def calculate_envelope(self,t0,t):
        values = np.zeros_like(t) 
        x1 = ( abs(t - t0) <=  self.plateau/2 + self.width/2)   
        x2 = ( abs(t - t0) < self.plateau/2 )
        values[x1] = self.get_unit_pulse( (self.plateau/2 + self.width/2 - abs(t[x1]-t0))/self.width * (self.relative_len*2 ) )
        values[x2] = self.get_unit_pulse( 0.5 * (self.relative_len*2 ) )
        return values * self.amplitude

    def get_unit_pulse(self,x):
        ## range(x) : x >= 0 , 
        ## x=0.5 ,  return 1
        return 1 + np.sign(x-0.5)*(1 - (np.cosh(0.5*self.beta)-np.cosh((x-0.5)*self.beta))/(np.cosh(0.5*self.beta)-1))



class Slepian(Pulse):
    def __init__(self, complex_value=False ,*args, **kwargs):
        super().__init__(complex_value)
        # For CZ pulses
        self.F_Terms = 1
        self.Coupling = 20E6
        self.Offset = 300E6
        self.Lcoeff = np.array([0.3])
        self.dfdV = 500E6
        self.qubit = None
        self.negative_amplitude = False

        self.theta_f = None
        self.t_tau = None

    def total_duration(self):
        return self.width+self.plateau

    def calculate_envelope(self, t0, t):
        if self.t_tau is None:
            self.calculate_cz_waveform()

        # Plateau is added as an extra extension of theta_f.
        theta_t = np.ones(len(t)) * self.theta_i
        for i in range(len(t)):
            if 0 < (t[i] - t0 + self.plateau / 2) < self.plateau:
                theta_t[i] = self.theta_f
            elif (0 < (t[i] - t0 + self.width / 2 + self.plateau / 2) <
                  (self.width + self.plateau) / 2):
                theta_t[i] = np.interp(
                    t[i] - t0 + self.width / 2 + self.plateau / 2, self.t_tau,
                    self.theta_tau)

            elif (0 < (t[i] - t0 + self.width / 2 + self.plateau / 2) <
                  (self.width + self.plateau)):
                theta_t[i] = np.interp(
                    t[i] - t0 + self.width / 2 - self.plateau / 2, self.t_tau,
                    self.theta_tau)
        # Clip theta_t to remove numerical outliers:
        theta_t = np.clip(theta_t, self.theta_i, None)

        # clip theta_f to remove numerical outliers
        theta_t = np.clip(theta_t, self.theta_i, None)
        df = 2*self.Coupling * (1 / np.tan(theta_t) - 1 / np.tan(self.theta_i))

        if self.qubit is None:
            # Use linear dependence if no qubit was given
            # log.info('---> df (linear): ' +str(df))
            values = df / self.dfdV
            # values = theta_t
        else:
            values = self.qubit.df_to_dV(df)
        if self.negative_amplitude is True:
            values = -values

        return values

    def calculate_cz_waveform(self):
        """Calculate waveform for c-phase and store in object"""
        # notation and calculations are based on
        # "Fast adiabatic qubit gates using only sigma_z control"
        # PRA 90, 022307 (2014)
        # Initial and final angles on the |11>-|02> bloch sphere
        self.theta_i = np.arctan(2*self.Coupling / self.Offset)

        if not self.theta_f:
            if self.amplitude>0:
                self.theta_f = np.arctan(2*self.Coupling / self.amplitude)
            elif self.amplitude==0:
                self.theta_f= np.pi/2
            else:
                self.theta_f = np.pi - np.arctan( - 2*self.Coupling / self.amplitude)

        # log.log(msg="calc", level=30)

        # Renormalize fourier coefficients to initial and final angles
        # Consistent with both Martinis & Geller and DiCarlo 1903.02492
        Lcoeff = self.Lcoeff
        Lcoeff[0] = (((self.theta_f - self.theta_i) / 2)
                     - np.sum(self.Lcoeff[range(2, self.F_Terms, 2)]))

        # defining helper variabels
        n = np.arange(1, self.F_Terms + 1, 1)
        n_points = 1000  # Number of points in the numerical integration

        # Calculate pulse width in tau variable - See paper for details
        tau = np.linspace(0, 1, n_points)
        self.theta_tau = np.zeros(n_points)
        # This corresponds to the sum in Eq. (15) in Martinis & Geller
        for i in range(n_points):
            self.theta_tau[i] = (
                np.sum(Lcoeff * (1 - np.cos(2 * np.pi * n * tau[i]))) +
                self.theta_i)
        # Now calculate t_tau according to Eq. (20)
        t_tau = np.trapz(np.sin(self.theta_tau), x=tau)
        # log.info('t tau: ' + str(t_tau))
        # t_tau = np.sum(np.sin(self.theta_tau))*(tau[1] - tau[0])
        # Find the width in units of tau:
        Width_tau = self.width / t_tau

        # Calculating time as functions of tau
        # we normalize to width_tau (calculated above)
        tau = np.linspace(0, Width_tau, n_points)
        self.t_tau = np.zeros(n_points)
        self.t_tau2 = np.zeros(n_points)
        for i in range(n_points):
            if i > 0:
                self.t_tau[i] = np.trapz(
                    np.sin(self.theta_tau[0:i+1]), x=tau[0:i+1])
                # self.t_tau[i] = np.sum(np.sin(self.theta_tau[0:i+1]))*(tau[1]-tau[0])



class Slepian_Triple(Pulse):
    def __init__(self, complex_value=False ,*args, **kwargs):
        super().__init__(complex_value)
        self.F_Terms = 2
        self.Lcoeff = np.array([5,1])
        self.Q1_freq = 6.0e9
        self.CPLR_idle_freq = 8e9
        self.Q2_freq = 5.5e9
        
        self.constant_coupling = False 
        ## if not constant_coupling, use r1c r2c
        self.g1c = 100e6 ## coupling strength
        self.g2c = 100e6
        self.r1c = 0.016
        self.r2c = 0.016

        self.dfdV = 500e6
        self.negative_amplitude = False
        self.anhar_CPLR = -400e6
        
    def total_duration(self):
        return self.width+self.plateau
    
    def calculate_envelope(self,t0,t):
        self.get_interp_eigen_spline()
        self.calculate_f_tau()
        self.calculate_t_tau()
        # print(self.f_tau_arr)
        # print(self.t_tau_arr)
        ft_spline = interpolate.splrep(self.t_tau_arr,self.f_tau_arr,k=3)
        
        values = np.zeros_like(t) 
        x1 = ( abs(t - t0) <=  self.plateau/2 + self.width/2)   
        x2 = ( abs(t - t0) < self.plateau/2 )
        values[x1] = self.CPLR_idle_freq -interpolate.splev( self.width/2 + abs(t[x1]-t0)-self.plateau/2,ft_spline )
        values[x2] =self.CPLR_idle_freq-  interpolate.splev( self.width/2,ft_spline )
        
        if self.negative_amplitude:
            values = values*-1
        return values/self.dfdV

    def get_eigen(self,fc):
        if not self.constant_coupling:
            g1c = self.r1c*np.sqrt(self.Q1_freq*fc)
            g2c = self.r2c*np.sqrt(self.Q2_freq*fc)
        else:
            g1c = self.g1c
            g2c = self.g2c
        self.H = np.array( [[self.Q1_freq+self.Q2_freq,g1c,0],
                            [g1c,self.Q2_freq+fc,np.sqrt(2)*g2c],
                            [0,np.sqrt(2)*g2c,2*fc+self.anhar_CPLR]])

        eigen_eners,eigen_states = eigensolve_sort(self.H)
        ener_alpha = eigen_eners[0]
        ener_beta  = eigen_eners[1]
        eigstate_alpha = eigen_states[:,0]
        eigstate_beta  = eigen_states[:,1]
        return ener_alpha,ener_beta,eigstate_alpha,eigstate_beta

    def get_derivative_state(self,state_trace,df):
        return (state_trace[1:]-state_trace[0:-1])/df

    def smooth_state_trace(self,state_list,inver_direc = False):
        last_state = state_list[0] 
        new_state_list = [last_state]
        for i in range(1,len(state_list)):
            if LA.norm(state_list[i] - last_state) >= LA.norm(state_list[i] + last_state):
                last_state = -1* state_list[i]
            else:
                last_state = state_list[i]
            new_state_list.append(last_state)
        return np.array(new_state_list)

    def get_interp_eigen_spline(self):
        self.fc_arr = np.linspace(self.Q2_freq-1000e6,self.CPLR_idle_freq+100e6,1001)
        ener_alpha_arr = np.array([])
        ener_beta_arr = np.array([])
        eigstate_alpha_list = []
        eigstate_beta_list = []

        for fc in self.fc_arr:
            ener_alpha,ener_beta,eigstate_alpha,eigstate_beta = self.get_eigen(fc)
            ener_alpha_arr = np.append(ener_alpha_arr,ener_alpha)
            ener_beta_arr = np.append(ener_beta_arr,ener_beta)
            eigstate_alpha_list.append( eigstate_alpha)
            eigstate_beta_list.append( eigstate_beta)

        alpha_deriv = self.get_derivative_state(self.smooth_state_trace(eigstate_alpha_list,False), self.fc_arr[1]-self.fc_arr[0] )
        eigstate_beta_smooth = self.smooth_state_trace(eigstate_beta_list,False)
        
        beta_alpha_deriv = np.array([])
        for i in range(len(alpha_deriv)):
            beta_alpha_deriv = np.append( beta_alpha_deriv, np.dot(eigstate_beta_smooth[i].T,alpha_deriv[i]) )
        
        self.gap_spline = interpolate.splrep(self.fc_arr[:-1],ener_beta_arr[:-1]-ener_alpha_arr[:-1],k=3)
        self.beta_alpha_deriv_spline = interpolate.splrep(self.fc_arr[:-1],beta_alpha_deriv,k=3)
    
    def calculate_f_tau(self):
        n = np.arange(1, self.F_Terms + 1, 1)
        n_points = 4001  # Number of points in the numerical integration
        self.tau_arr = np.linspace(0, 1, n_points)
        self.d_tau = self.tau_arr[1]-self.tau_arr[0]
        
        f_tau0=self.CPLR_idle_freq
        f_tau_arr = np.array([f_tau0])
        for i in range( int((n_points-1)/2) ):
            df_dtau = np.sum(self.Lcoeff*( np.sin(2*np.pi*n*self.tau_arr[i])))/interpolate.splev(f_tau0,self.beta_alpha_deriv_spline)
            f_tau0 += df_dtau * self.d_tau
            f_tau_arr =np.append( f_tau_arr, f_tau0 )
        self.f_tau_arr = np.append(f_tau_arr,f_tau_arr[-2::-1])
    
    def calculate_t_tau(self):
        T_gate = np.array([])
        t0=0
        for ftau in self.f_tau_arr:
            t0+=1/interpolate.splev( ftau, self.gap_spline )*self.d_tau 
            T_gate = np.append(T_gate,t0 )

        self.t_tau_arr = T_gate/max(T_gate)*self.width


class Adiabatic(Pulse):
    
    def __init__(self, complex_value=False ,*args, **kwargs):
        super().__init__(complex_value)

        self.min_C=1e9     # minimum value is calculating the adiabaticity factor
        self.max_C=10e9    # maximum value is calculating the adiabaticity factor
        self.down_tuning = True

        self.F_Terms = 2
        self.Lcoeff = np.array([1,0.1])
        self.dfdV = 500e6
        self.negative_amplitude = False
        self.up_limit=None    # set uplimit of pulse value, prevent outliers
        self.down_limit=None  # set down of pulse value

        self.constant_coupling = False 
        self.qubit = None

        self.Q1_freq = 6.0e9
        self.CPLR_idle_freq = 8e9
        self.Q2_freq = 5.4e9
        ## if not constant_coupling, use r1c r2c
        self.g1c = 100e6 
        self.g2c = 100e6
        self.g12 = 12e6
        self.r1c = 0.016
        self.r2c = 0.016
        self.r12 = 0.001
        self.anhar_Q1 = -250e6
        self.anhar_Q2 = -250e6
        self.anhar_CPLR = -400e6
        
        self.gap_threshold = 10e6  # ignore small gaps between eigentraces
        self.pulsepoints = 601  # Number of points in integrating f(t)
        self.freqpoints = 301   # Number of points in calculating the adiabaticity factor

    def total_duration(self):
        return self.width+self.plateau
    
    def calculate_envelope(self,t0,t):
        self.get_adia_factor_spline()
        self.calculate_f_t_sinosoidal()
        ft_spline = scipy.interpolate.splrep(self.t_arr,self.f_t_arr,k=3)
        
        dfreq = np.zeros_like(t) 
        x1 = ( abs(t - t0) <=  self.plateau/2 + self.width/2)   
        x2 = ( abs(t - t0) < self.plateau/2 )
        dfreq[x1] = scipy.interpolate.splev( (self.width/2+abs(t[x1]-t0)-self.plateau/2)/self.width,ft_spline ) - self.CPLR_idle_freq
        dfreq[x2] = scipy.interpolate.splev( 0.5 ,ft_spline ) - self.CPLR_idle_freq

        if self.qubit is None:
            # Use linear dependence if no qubit was given
            # log.info('---> df (linear): ' +str(df))
            values = -1*dfreq / self.dfdV
            # values = theta_t
        else:
            values = self.qubit.df_to_dV(dfreq)

        if self.negative_amplitude:
            values = values*-1

        if self.up_limit:
            values[values>self.up_limit]=self.up_limit
        if self.down_limit:
            values[values<self.down_limit]=self.down_limit

        return values

    def calculate_f_t_sinosoidal(self):
        n = np.arange(1, self.F_Terms + 1, 1)
        n_points = self.pulsepoints  # Number of points in the numerical integration
        self.t_arr = np.linspace(0, 1, n_points)
        self.dt = (self.t_arr[1]-self.t_arr[0])*self.width
        
        f_t0=self.CPLR_idle_freq
        f_t_arr = np.array([f_t0])
        for i in range( int((n_points-1)/2) ):
            df_dt = -1*np.sum( self.Lcoeff*( np.sin(2*np.pi*n*self.t_arr[i])) ) / scipy.interpolate.splev(f_t0,self.adia_spline)  
            f_t0 += df_dt * self.dt
            f_t_arr =np.append( f_t_arr, f_t0 )
        self.f_t_arr = np.append(f_t_arr,f_t_arr[-2::-1])

    def get_adia_factor_spline(self):
        if self.down_tuning:
            self.fc_arr = np.linspace(self.min_C,self.CPLR_idle_freq+1e6,self.freqpoints)[::-1]
        else:
            self.fc_arr = np.linspace(self.CPLR_idle_freq-1e6,self.max_C,self.freqpoints)
        df = self.fc_arr[1]-self.fc_arr[0]
        
        position_idx = self.get_maximum_overlap_index(self.get_Hamiltonian(self.fc_arr[0]))
        self.Ener_All=[]
        self.Estate_All=[]
        for fc in self.fc_arr:
            eigen_eners,eigen_states = self.get_eigen(fc)
            self.Ener_All.append(eigen_eners)
            self.Estate_All.append(eigen_states)
        self.Ener_All = np.asarray(self.Ener_All)
        self.Estate_All = np.asarray(self.Estate_All)
        if self.gap_threshold:
            self.rearrangement_eigen_traces_by_ignore_small_gap()

        # 001,010,100,011,101,110,002,020,200
        Ener9trace = [[],[],[],[],[],[],[],[],[]]
        Estate9trace = [[],[],[],[],[],[],[],[],[]]
        for trace_idx in range(len(self.Ener_All)):
            for ii,idx in enumerate([1,3,9,4,10,12,2,6,18]):
                Ener9trace[ii].append( self.Ener_All[trace_idx][position_idx][idx]  ) 
                Estate9trace[ii].append( self.Estate_All[trace_idx][position_idx][idx]  )                
                
        self.Adia_Factor_Total = 0
        self.Adia_Factor_Total += np.abs( self.get_adia_factor( Estate9trace[0],Estate9trace[1],Ener9trace[0],Ener9trace[1],df) )
        self.Adia_Factor_Total += np.abs( self.get_adia_factor( Estate9trace[0],Estate9trace[2],Ener9trace[0],Ener9trace[2],df) )
        self.Adia_Factor_Total += np.abs( self.get_adia_factor( Estate9trace[1],Estate9trace[2],Ener9trace[1],Ener9trace[2],df) )
        for jj in [4]:
            for kk in range(3,9):
                if kk !=jj:
                    self.Adia_Factor_Total += np.abs(self.get_adia_factor( Estate9trace[jj],Estate9trace[kk],Ener9trace[jj],Ener9trace[kk],df))
        # if freq_ascend == False:
        if self.down_tuning:
            self.adia_spline = scipy.interpolate.splrep(self.fc_arr[::-1],self.Adia_Factor_Total[::-1],k=3)
        else:
            self.adia_spline = scipy.interpolate.splrep(self.fc_arr,self.Adia_Factor_Total,k=3)

    def get_Hamiltonian(self,fc):
        if not self.constant_coupling:
            g1c = self.r1c*np.sqrt(self.Q1_freq*fc)
            g2c = self.r2c*np.sqrt(self.Q2_freq*fc)
            g12 = self.r12*np.sqrt(self.Q2_freq*self.Q1_freq)
        else:
            g1c = self.g1c
            g2c = self.g2c
            g12 = self.g12
        fq1 = self.Q1_freq
        fq2 = self.Q2_freq
        anhar1 = self.anhar_Q1
        anharc = self.anhar_CPLR
        anhar2 = self.anhar_Q2
            
        Hq1 = fq1*mat_mul_all(create(3),destroy(3))+anhar1/2*mat_mul_all(create(3),create(3),destroy(3),destroy(3))
        Hq1_full = np.kron(np.kron(Hq1,np.eye(3)),np.eye(3))
        Hc = fc*mat_mul_all(create(3),destroy(3))+anharc/2*mat_mul_all(create(3),create(3),destroy(3),destroy(3))
        Hc_full = np.kron(np.kron(np.eye(3),Hc),np.eye(3))
        Hq2 = fq2*mat_mul_all(create(3),destroy(3))+anhar2/2*mat_mul_all(create(3),create(3),destroy(3),destroy(3))
        Hq2_full = np.kron(np.kron(np.eye(3),np.eye(3)),Hq2)
        H_g1c = g1c*np.kron(np.kron(create(3)+destroy(3),create(3)+destroy(3) ),np.eye(3))
        H_g2c = g2c*np.kron(np.kron(np.eye(3),create(3)+destroy(3) ),create(3)+destroy(3))
        H_g12 = g12*np.kron(np.kron(create(3)+destroy(3),np.eye(3)),create(3)+destroy(3) )
        return Hq1_full+Hc_full+Hq2_full+H_g1c+H_g2c+H_g12
    
    def get_adia_factor(self,alpha,beta,E_alpha,E_beta,df):
        alpha_deriv = self.get_derivative_state( self.smooth_state_trace(alpha),df )
        beta_smooth = self.smooth_state_trace(beta)
        return np.array([ np.dot(beta_smooth[i].T.conj(),alpha_deriv[i])/(E_alpha[i]-E_beta[i]) for i in range(len(alpha_deriv))])

    def get_eigen(self,fc,position_index=False):  
        self.H = self.get_Hamiltonian(fc)
        eigen_eners,eigen_states = eigensolve_sort(self.H)
        if position_index:
            return eigen_eners[position_index],eigen_states.T[position_index]
        else:
            return eigen_eners,eigen_states.T

    def get_maximum_overlap_index(self,H):
        ## be careful using this function, it may fail in degenerate case !!!!
        eigenvalues = eigensolve_close(H)[0]
        position_index = np.argsort(eigenvalues)
        return np.argsort(position_index)
    
    def get_derivative_state(self,state_trace,df):
        deriv_list = [ (state_trace[i+1]-state_trace[i-1])/2/df for i in range(1,len(state_trace)-1)] 
        deriv_list.insert(0, (state_trace[1]-state_trace[0])/df )
        deriv_list.append( (state_trace[-1]-state_trace[-2])/df )
        return deriv_list

    def smooth_state_trace(self,state_list):
        last_state = state_list[0] 
        new_state_list = [last_state]
        for i in range(1,len(state_list)):
            if np.linalg.norm(state_list[i] - last_state) >= np.linalg.norm(state_list[i] + last_state):
                last_state = -1* state_list[i]
            else:
                last_state = state_list[i]
            new_state_list.append(last_state)
        return np.array(new_state_list)

    def rearrangement_eigen_traces_by_ignore_small_gap(self):
        for i in range(len(self.Ener_All[0])-5):
            for k in range(1,4):
                self.swap_two_eigen_trace(self.Ener_All[:,i],self.Ener_All[:,i+k],self.Estate_All[:,i],self.Estate_All[:,i+k],self.gap_threshold )

    def swap_two_eigen_trace(self,eigen_ener1,eigen_ener2,eigen_state1,eigen_state2,gap):
        ener_diff = eigen_ener2 - eigen_ener1
        anticross_idx = np.where( ener_diff < gap )[0]
        if len(anticross_idx) == 0 or isinstance(ener_diff,float):
            pass
        else:
            extreme_points  = self.get_extreme_points(ener_diff,anticross_idx)
            for point in extreme_points:
                eigen_ener1_temp = copy.deepcopy(eigen_ener1)
                eigen_state1_temp = copy.deepcopy(eigen_state1)
                eigen_ener1[point:] = eigen_ener2[point:]
                eigen_ener2[point:] = eigen_ener1_temp[point:]
                eigen_state1[point:] = eigen_state2[point:]
                eigen_state2[point:] = eigen_state1_temp[point:]

    def get_extreme_points(self,ener_diff,anticross_idx):
        start_idxs = [anticross_idx[0]]
        end_idxs = []
        for idx_count,idx in enumerate(anticross_idx):
            if idx+1 in anticross_idx:
                continue
            else:
                end_idxs.append(idx)
                if idx_count != len(anticross_idx)-1:
                    start_idxs.append(anticross_idx[idx_count+1])
        extreme_points = []
        for i in range(len(start_idxs)):
            if start_idxs[i] == end_idxs[i]:
                extreme_points.append(start_idxs[i])
            else:
                extreme_points.append( np.argmin(ener_diff[start_idxs[i]:end_idxs[i]])+start_idxs[i] )    
        return extreme_points
        

class Spline(Pulse):
    def __init__(self,complex_value=False, *args, **kwargs):
        super().__init__(complex_value)
        self.k = 3  # cubic interpolate
        self.assigned_Point_arr=np.array([0.1,0.4,0.8])
        self.assigned_Value_arr=np.array([0.8,0.9,0.95])
        self.negative_amplitude = False
        self.use_deriv = False

    def total_duration(self):
        return self.width+self.plateau

    def calculate_envelope(self, t0, t):
        values = np.zeros_like(t)
        accum_values = np.zeros_like(values)
        sym_sign = -1 if self.use_deriv else 1 
        self.get_Bspline(sym_sign)

        norm_factor_for_deriv = (self.width/4) / (t[1] - t[0])
        last_accum_value = 0

        for i in range(len(t)):
            if 0 < abs( t[i] - t0 )< self.plateau/2:
                values[i] = 0 if self.use_deriv else 1
                accum_values[i] = last_accum_value
                last_accum_value += values[i] / norm_factor_for_deriv
            elif self.plateau/2 <= abs( t[i] - t0  ) <= self.plateau/2 + self.width/2 :
                # values[i] = sym_sign *self.get_interp_value(self.width/2 + self.plateau/2 - abs(t[i] - t0))
                values[i] = self.get_interp_value( self.width/2 - self.plateau/2 * np.sign(t[i]-t0) + (t[i] - t0) )
                accum_values[i] = last_accum_value
                last_accum_value += values[i] / norm_factor_for_deriv

        values = accum_values * self.amplitude if self.use_deriv else values * self.amplitude
        if self.negative_amplitude is True:
            values = -values
        return values

    def get_interp_value(self,t):
        return scipy.interpolate.splev(t,self.Bspline)

    def get_Bspline(self,sym_sign):
        self.assigned_Time_arr = np.sort(self.assigned_Point_arr) * self.width / 2
        time_arr=np.append( self.assigned_Time_arr , self.width - self.assigned_Time_arr[::-1] )
        value_arr=np.append( self.assigned_Value_arr , sym_sign * self.assigned_Value_arr[::-1] )
        self.Bspline = scipy.interpolate.splrep(time_arr,value_arr,k=self.k)
        

class NetZero(Pulse):
    def __init__(self, pulse, *args, **kwargs):
        super().__init__(False)
        self.__dict__ = copy.copy(pulse.__dict__)
        self.pulse = copy.copy(pulse)
        self.pulse.width /= 2
        self.pulse.plateau /= 2
        self.net_zero_delay = 0

    def total_duration(self):
        return 2*self.pulse.total_duration() + self.net_zero_delay

    def calculate_envelope(self, t0, t):
        t_offset = (self.pulse.total_duration() + self.net_zero_delay) / 2
        return (self.pulse.calculate_envelope(t0-t_offset, t) -
                self.pulse.calculate_envelope(t0+t_offset, t))


if __name__ == '__main__':
    pass

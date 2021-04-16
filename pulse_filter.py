from scipy import fft
import matplotlib.pyplot as plt
import numpy as np

def gauss_low_pass(f,*args):
    f_c=args[0]
    try:
        return np.exp( -1 * f**2 / f_c**2 * 0.346724)
    except:
        raise Exception(f'{args}, {f_c}')

def common_ansatz(f,*args):
    s= 1j*f
    H = 1 + 0j
    for i in range( int(len(args)/2)):
        H += args[2*i]*s / ( 1/args[2*i+1] + s ) 
    return H

def rlc_low_pass(f,*args):
    fc = args[0]
    zeta = args[1]  # zeta = 1/(2Q)
    s = 1j*f
    return fc**2/(s**2+zeta*s+fc**2)

def linear_low_pass(f,*args):
    s=1j*f
    f_c = args[0]
    return 1/(1+s/f_c)

# def reflection(f,*args):
#     omega = 2*np.pi * f
#     s = 1j*omega
#     r=args[0]
#     T=args[1]
#     H_ri= 1-r
#     for k in range(1,15):
#         H_ri += r**k * np.exp(-s * 2 *k * T )
#     return H_ri

def reflection(f,*args):
    omega = 2*np.pi * f
    s = 1j*omega
    r=args[0]
    T=args[1]
    if isinstance(r,float) or isinstance(r,int):
        return 1 + r*np.exp(-2*s*T)/(1-r*np.exp(-2*s*T)) - r 
    else:
        H_r = 1
        for i in range(r):
            H_r += r[i]*np.exp(-2*s*T[i])/(1-r[i]*np.exp(-2*s*T[i])) - r[i] 
            return H_r

def Z_N(f,*args):
    s=1j*f
    M=0
    N=0
    for i,p in enumerate(args[0]):
        M += p*s**(i)
    for i,p in enumerate(args[1]):
        N += p*s**(i)
    return M / N

def ZeroPole(f,*args):
    s = 1j*f
    factor = args[0]
    zeros = args[1]
    poles = args[2]
    s_all = factor
    for i in range(len(zeros)):
        s_all *= s - zeros[i]

    for i in range(len(poles)):
        s_all /= s - poles[i]
    return s_all

MODE_GEN = 'Distorted signal'
MODE_CAL = 'Predistort signal'
FILTER_FUNC = {'RLC Low Pass':rlc_low_pass,'Gauss Low Pass':gauss_low_pass,'Low Pass Linear':linear_low_pass,'Reflection':reflection,'Z_N':Z_N,'Zero Pole':ZeroPole,'Common Ansatz':common_ansatz}

#%%
class Filter():

    def __init__(self,filter_type,*args,sampling_rate=1e9,mode = MODE_GEN ):
        self.filter_type=filter_type
        self.parameters = args
        self.mode = mode
        self.sampling_rate = sampling_rate

    def get_distorted_waveform(self,pulse_value):
        original_waveform = pulse_value
        if len(original_waveform)==0:
            return np.array([])

        original_ff = fft.fft(original_waveform)
        pulse_len = len(original_waveform)
        distorted_ff = []
        for i in range(pulse_len):
            f = (i/pulse_len)*self.sampling_rate  if i < (pulse_len/2) else (pulse_len-i)/pulse_len*self.sampling_rate

            h_i = FILTER_FUNC[self.filter_type](f,*self.parameters )

            if i >= pulse_len / 2:
                h_i = h_i.conjugate()

            distorted_ff.append( original_ff[i] * h_i if self.mode == MODE_GEN else original_ff[i] / h_i )

        return np.real(fft.ifft(distorted_ff))

    def get_abs_freq_response(self,freq_arr,unit='V'):
        H_arr = FILTER_FUNC[self.filter_type](freq_arr,*self.parameters)
        if unit=='V':
            return abs(H_arr)**2
        elif unit=='dB':
            return 20*np.log10(abs(H_arr))
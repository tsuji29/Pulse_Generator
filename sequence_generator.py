
import numpy as np
import matplotlib.pyplot as plt
import copy

try:
    from .pulse_func import *
    from .pulse_filter import *
except ImportError:
    from pulse_func import *
    from pulse_filter import *


supported_pulse_type = ('Gauss','Square','Cosine','Ramp','Slepian','Spline')

class Sequence():

    def __init__(self, total_len = 100e-9,sample_rate = 1e9,complex_trace = False):
        self.sequence_len = total_len
        self.sample_rate = sample_rate
        self.tlist = np.linspace(0,total_len, int(np.round( total_len * sample_rate)) + 1 )   
        self.complex = complex_trace
        self.p_idx = 0
        # print('supported pulse type:',supported_pulse_type)
        self.filter = []

    def add_pulse(self,pulse_type,amplitude = 0.5,t0 = 20e-9,width = 20E-9,plateau = 0.0,frequency = 0.0,netzero=False, 
            pulse_trunc =False, trunc_start =0,trunc_end = 100e-9,beta = 10 ,relative_len=0.95,
            **kwargs):
        
        if pulse_type in ('gauss','Gauss','gaussian','Gaussian'):
            pulse = Gaussian(self.complex) 
        elif pulse_type in ('square','Square'):
            pulse = Square(self.complex)
        elif pulse_type in ('cos','cosine','Cos','Cosine','sine','Sine','sin','Sin'):
            pulse = Cosine(self.complex)
        elif pulse_type in ('ramp','Ramp'):
            pulse = Ramp(self.complex)
        elif pulse_type in ('CosH','CosineH'):
            pulse = CosH(self.complex)
        elif pulse_type in ('CosH_Full','CosineH_Full'):
            pulse = CosH_Full(self.complex)
        elif pulse_type in ('Slepian','slepian'):
            pulse = Slepian()
        elif pulse_type in ('Slepian_Triple','slepian_triple'):
            pulse = Slepian_Triple()
        elif pulse_type in ('Spline','spline'):
            pulse = Spline()
        elif pulse_type in ('Adiabatic','adiabatic'):
            pulse = Adiabatic()
        elif pulse_type in ('Fourier','fourier'):
            pulse = Fourier()
        else:
            raise Exception(f'{pulse_type} is not a defined pulse type!')
        
        pulse.name = pulse_type
        pulse.amplitude = amplitude
        pulse.t0 = t0
        pulse.width = width
        pulse.plateau = plateau
        pulse.frequency = frequency
        pulse.pulse_trunc = pulse_trunc
        pulse.trunc_start = trunc_start
        pulse.trunc_end = trunc_end
        pulse.beta = beta
        pulse.relative_len = relative_len
        for key,value in kwargs.items():
            if hasattr(pulse,key):
                setattr(pulse,key,value)
            else:
                raise Exception(f'{key} is not an attribute of pulse type {pulse_type}!' )

        if netzero:
            setattr(self,f'pulse_{self.p_idx}',NetZero(pulse))
        else:
            setattr(self,f'pulse_{self.p_idx}',pulse)

        self.p_idx += 1

    def add_filter(self,filter_name,*args,mode = MODE_GEN ):
        self.filter.append( Filter(filter_name,*args,sampling_rate=self.sample_rate,mode=mode) )

    def show_sequence(self):
        ## this function show all pulses
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if not self.complex:
            ax.plot(self.tlist,self.get_sequence(),label='Pulse')
        else:
            ax.plot(self.tlist,np.real(self.get_sequence()),label='I')
            ax.plot(self.tlist,np.imag(self.get_sequence()),label='Q')

        plt.legend()


    def get_sequence(self):
        ## this function calculate all pulses
        if not self.complex:
            sequence_value = np.zeros_like(self.tlist)
        else:
            sequence_value = np.zeros_like(self.tlist) * 1j

        for i in range(self.p_idx):
            if hasattr(self,f'pulse_{i}'):
                t0 = getattr(self,f'pulse_{i}').t0
                sequence_value += getattr(self,f'pulse_{i}').calculate_waveform(t0,self.tlist) 


        for filter in self.filter:
            sequence_value = filter.get_distorted_waveform(sequence_value)
            
        return sequence_value
        
    def clear_pulse(self,tips_on = True):
        key_dict = copy.deepcopy(self.__dict__)
        for key in key_dict.keys():
            if key.startswith('pulse_'):
                self.__delattr__(key)
        self.p_idx = 0
        if tips_on:
            print('all pulses have been cleared')
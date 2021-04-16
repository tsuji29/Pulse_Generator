# Pulse_Generator

This package is a supplement for Circuit Simulator. It is used to produce pulse sequences.

1. the main entrance is sequence_generator.py, where the sampling rate , pulse length ... are difined. 

2. 'pulse_func.py' defines some pulse functions. most functions are exactly the same as the ones used in 'Labber_Drivers\Drivers\MultiQubit_PulseGenerator_Custom\pulses', which is managed by Jiahao Yuan now.

3. 'pulse_filter.py' defines some virtual filters.  

4. '_func.py' defines some functions for pulses in pulse_func.

5. 'qubits.py' is are exactly the one in 'Labber_Drivers\Drivers\MultiQubit_PulseGenerator_Custom'






An example:

    import PulseGenerator as PG
    import matplotlib.pyplot as plt

    srate=10e9
    total_len = 100e-9
    Seq=PG.Sequence(total_len=total_len,sample_rate=srate,complex_trace=False)
    Seq.clear_pulse(tips_on=False)
    Seq.add_pulse('Cosine',t0=total_len/2-30e-9,width=10e-9,plateau=10e-9,amplitude=0.3255,frequency=0,half_cosine=True)
    Seq.add_pulse('Square',t0=total_len/2+30e-9,width=10e-9,plateau=10e-9,amplitude=0.1,frequency=0,half_cosine=True)
    Seq.add_filter('Gauss Low Pass',300e6)
    flux_pulse=Seq.get_sequence()
    plt.plot(flux_pulse)
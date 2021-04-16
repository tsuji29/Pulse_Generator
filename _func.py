
from scipy.linalg import eig
import numpy as np

def eigensolve_close(H):
    '''
    get eigensolution of hamiltonian 'H'.
    '''
    vals, vecs = eig(H)    
    for i in range(len(vecs[:,1])):
        idx=np.append(range(i),(-abs(vecs[i,i:])).argsort()+i) if i>0 else (-abs(vecs[i,i:])).argsort()   
        vecs=vecs[:,idx]
        vals=vals[idx]
    return np.real(vals), vecs

def eigensolve_sort(H,ascending = True):
    '''
      get eigensolution of hamiltonian 'H', default ascending order is True.
      The return eigenenergies are in ascending order is ascending is True, else they will be is descending order.
    '''
    vals, vecs = eig(H)    
    if ascending:
        idx = vals.argsort()
    else:
        idx = vals.argsort()[::-1] 
    vals = vals[idx]
    vecs = vecs[:,idx]
    return np.real(vals), vecs

def create(n):
    A = np.zeros([n,n])*1j
    for i in range(n-1):
        A[i+1,i] = np.sqrt(i+1)*(1+0j)
    return A

def destroy(n):
    return create(n).conj().transpose()

def mat_mul_all(*args):
    if len(args)<2:
        return args
    else:
        A = args[0]
        for i in range(1,len(args)):
            A = np.matmul(A,args[i])
    return A


def get_transmon_freq(f01_max,f01_min,Ec,Voltage_Period,Voltage_operating_point,flux):
    if Voltage_Period >= 0:
        Ej_max = (f01_max + Ec)**2/(8*Ec)
        d = (f01_min + Ec)**2/(8*Ec*Ej_max)
        phi = (Voltage_operating_point + flux)/ Voltage_Period
        # self.config.log('d',d)
        # self.config.log('Ej_max',phi)
        Ej_flux = Ej_max * np.sqrt( np.cos(np.pi * phi)**2 + np.sin(np.pi*phi)**2 * d**2)

        freq= np.sqrt(8*Ej_flux * Ec) -Ec
    else:
        freq = (f01_min-f01_max) * flux + f01_max
    anhar = -1 * Ec
    return freq,anhar
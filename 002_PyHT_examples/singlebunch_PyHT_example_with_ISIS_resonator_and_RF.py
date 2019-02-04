from __future__ import division


import sys, os
# PyHEADTAIL installation path
BIN = os.path.expanduser("../../Codes/Singlebunch/PyHEADTAIL")
sys.path.append(BIN)

from mpi4py import MPI
import matplotlib.pyplot as plt
import sys
import numpy as np
import time
from scipy.constants import c, e, m_p
import h5py

# Loads the required PyHEADTAIL modules
from PyHEADTAIL.monitors.monitors import BunchMonitor, SliceMonitor
from PyHEADTAIL.machines.synchrotron import Synchrotron
from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.impedances.wakes import WakeField, CircularResonator

from PyHEADTAIL.particles.generators import RFBucketMatcher
from PyHEADTAIL.trackers.rf_bucket import RFBucket
from PyHEADTAIL.particles.rfbucket_matching import WaterbagDistribution
from scipy.signal import hilbert

# Output path for the simulation results
outputpath = 'Data'

m_p_MeV = 938.272e6


def run(job_id,accQ_y):
    it = job_id
    
  
    # SIMULATION PARAMETERS
    # =====================
    
    # Simulation parameters 
    n_turns = 10000
    n_macroparticles = 100000 # per bunch 
    
    # MACHINE PARAMETERS
    # ==================
    
    intensity = 1e13 # protons
#    intensity = 2*4e12 # protons
    E0 = 71e6 # Kinetic energy [eV]
    p0 = np.sqrt((m_p_MeV+E0)**2 - m_p_MeV**2) * e /c
    
    print('Beam kinetic energy: ' + str(E0*1e-6) + ' MeV')
    print('Beam momentum: ' + str(p0*1e-6*c/e) + ' MeV/c')
    
    
    accQ_x = 4.31 # Horizontal tune
#    accQ_y = 3.80 # Vertical tune is an input argument
    
    
    chroma=-1.4 # Chromaticity

    alpha = 5.034**-2 # momentum compaction factor
    
    circumference = 160. # [meters]
    
    # Approximated average beta functions (lumped wake normalizations)
    beta_x = circumference / (2.*np.pi*accQ_x) 
    beta_y = circumference / (2.*np.pi*accQ_y)
    
    # Harmonic number for RF
    h_RF = [2] # a list of harmonic number for RF
    V_RF = [5e3*2] # a list of RF voltages
    p_increment = 0.
    dphi_RF = [np.pi]  # a list of RF phases
#    dphi_RF = 0.
    
    # non-linear longitudinal mode includes RF, otherwise linear needs synhotron tune Q_s
    longitudinal_mode = 'non-linear'
#    Q_s=0.02 # Longitudinal tune
    
    optics_mode = 'smooth'
    n_segments = 1    
    s = None
    alpha_x = None
    alpha_y = None
    D_x = 0
    D_y = 0     
    charge = e
    mass = m_p
    name = None
    app_x = 0
    app_y = 0
    app_xy = 0
    
    # Creates PyHEADTAIL object for the synchotron
    machine = Synchrotron(optics_mode=optics_mode, circumference=circumference,
            n_segments=n_segments, s=s, name=name,
            alpha_x=alpha_x, beta_x=beta_x, D_x=D_x,
            alpha_y=alpha_y, beta_y=beta_y, D_y=D_y,
            accQ_x=accQ_x, accQ_y=accQ_y, Qp_x=chroma, Qp_y=chroma,
            app_x=app_x, app_y=app_y, app_xy=app_xy,
            alpha_mom_compaction=alpha, longitudinal_mode=longitudinal_mode,
            h_RF=np.atleast_1d(h_RF), V_RF=np.atleast_1d(V_RF),
            dphi_RF=np.atleast_1d(dphi_RF), p0=p0, p_increment=p_increment,
            charge=charge, mass=mass)
    
    print('')
    print('machine.beta: ')
    print(machine.beta)
    print('')
    
    print('')
    print('machine.gamma: ')
    print(machine.gamma)
    print('')
    
    
    epsn_x = 300e-6
    epsn_y = 300e-6
    sigma_z = 15 # bunch length in meters to be matched to the bucket 

    # Creates transverse macroparticle distribution 
    allbunches = machine.generate_6D_Gaussian_bunch(n_macroparticles, intensity, epsn_x, epsn_y, sigma_z)
    
    # Creates longitudinal macroparticle distribution 
    rfb = RFBucket(circumference, machine.gamma, m_p, e, [alpha], 0., h_RF, V_RF, dphi_RF)
    rfb_matcher = RFBucketMatcher(rfb, WaterbagDistribution, sigma_z=sigma_z)
    
    rfb_matcher.integrationmethod = 'cumtrapz'
    
    z, dp, _, _ = rfb_matcher.generate(n_macroparticles)
    np.copyto(allbunches.z, z)
    np.copyto(allbunches.dp, dp)
    
    # Slicer object, which used for wakefields and slice monitors
    slicer = UniformBinSlicer(75, z_cuts=(-2.*sigma_z, 2.*sigma_z))

    # WAKE FIELDS
    # ===========
    
    # Length of the wake function in turns, wake 
    n_turns_wake = 150
    
    # Parameters for a resonator
    # frequency is in the units of (mode-Q_frac), where
    #       mode: integer number of coupled bunch mode (1 matches to the observations)
    #       Q_frac: resonance fractional tune
    
    f_r = (1-0.83)*1./(circumference/(c*machine.beta))
    Q = 15
    R = 1.0e6
    
    # Renator wake object, which is added to the one turn map
    wakes = CircularResonator(R, f_r, Q, n_turns_wake=n_turns_wake)
    wake_field = WakeField(slicer, wakes)
    machine.one_turn_map.append(wake_field)


    # CREATE MONITORS
    # ===============
    simulation_parameters_dict = {'gamma'           : machine.gamma,\
                                  'intensity'       : intensity,\
                                  'Qx'              : accQ_x,\
                                  'Qy'              : accQ_y,\
#                                  'Qs'              : Q_s,\
                                  'beta_x'          : beta_x,\
                                  'beta_y'          : beta_y,\
    #                               'beta_z'          : bucket.beta_z,\
                                  'epsn_x'          : epsn_x,\
                                  'epsn_y'          : epsn_y,\
                                  'sigma_z'         : sigma_z,\
                                 }
    # Bunch monitor strores bunch average positions for all the bunches
    bunchmonitor = BunchMonitor(outputpath + '/bunchmonitor_{:04d}'.format(it), n_turns,
                                simulation_parameters_dict,
                                write_buffer_every=32, buffer_size=32)
    
    # Slice monitors saves slice-by-slice data for each bunch
    slicemonitor = SliceMonitor(
        outputpath + '/slicemonitor_{:01d}_{:04d}'.format(0,it),
        60, slicer,
        simulation_parameters_dict, write_buffer_every=60, buffer_size=60)
        
    # Counter for a number of turns stored to slice monitors
    s_cnt = 0

    # TRACKING LOOP
    # =============
    monitor_active = False
    print('\n--> Begin tracking...\n')

    for i in range(n_turns):
        t0 = time.clock()

        # Tracks beam through the one turn map simulation map
        machine.track(allbunches)
        
        # Stores bunch mean coordinate values
        bunchmonitor.dump(allbunches)

        
    
        # If the total oscillation amplitude of bunches exceeds the threshold
        # or the simulation is running on the last turns, triggers the slice
        # monitors for headtail motion data
        if (allbunches.mean_x() > 1e-1 or allbunches.mean_y() > 1e-1 or i > (n_turns-64)):
                monitor_active = True
        
        # saves slice monitor data if monitors are activated and less than 
        # 64 turns have been stored
        if monitor_active and s_cnt<64:
            slicemonitor.dump(allbunches)
            s_cnt += 1  
        elif s_cnt == 64:
            break

        # If this script is runnin on the first processor, prints the current
        # bunch coordinates and emittances
        if (i%100 == 0):            
            print('{:4d} \t {:+3e} \t {:+3e} \t {:+3e} \t {:3e} \t {:3e} \t {:3f} \t {:3f} \t {:3f} \t {:3s}'.format(i, allbunches.mean_x(), allbunches.mean_y(), allbunches.mean_z(), allbunches.epsn_x(), allbunches.epsn_y(), allbunches.epsn_z(), allbunches.sigma_z(), allbunches.sigma_dp(), str(time.clock() - t0)))

def analyze(job_id):
    # This function analyses the stored data
    it = job_id
    
    # Reads data from the files
    bunchmonitor_file =  outputpath+'/bunchmonitor_{:04d}'.format(it)
    slicemonitor_file = outputpath + '/slicemonitor_0_{:04d}'.format(it)

    h5f_bunch = h5py.File(bunchmonitor_file+'.h5','r')
    h5f_slice = h5py.File(slicemonitor_file+'.h5','r')
    
    n_macroparticles = np.array(h5f_slice['Slices']['n_macroparticles_per_slice'])
    slice_data_z = np.array(h5f_slice['Slices']['mean_z'])
    slice_data_y = np.array(h5f_slice['Slices']['mean_y'])
    
    n_traces = 10 # Number of turn headtail data plotted
    
    # reads bunch-by-bunch data and removes empty turns
    epsn_data = np.array(h5f_bunch['Bunch']['epsn_y'])
    valid_data_map = (epsn_data>0)
    epsn_data = epsn_data[valid_data_map]
    
    mean_y = np.array(h5f_bunch['Bunch']['mean_y'])
    mean_y = mean_y[valid_data_map]

    # calculates an envelope of turn-by-turn oscillations with Hilbert transform
    analytic_signal = hilbert(mean_y)
    amplitude_envelope = np.abs(analytic_signal)

    # calculates singal frequency from the Hilbert transform. Tune can be extracted from here
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase)/(2.0*np.pi) * 1.)

    turns = np.linspace(1,len(mean_y),len(mean_y))
    
    # Turn range for growth rate fitting
    fit_from = int(0.5*len(mean_y))
    fit_to = int(0.9*len(mean_y))
    
    coeffs_y = np.polyfit(turns[fit_from:fit_to], np.log(amplitude_envelope[fit_from:fit_to]),1)
    fit_y = np.exp(coeffs_y[0]*turns+coeffs_y[1])
    
    # Turn by turn data and fits
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8,8))  
    ax1.plot(turns,mean_y)
    ax1.plot(turns,fit_y,'r-', label="Growth time {:.2f} turns".format(1./coeffs_y[0]))
    ax1.plot(turns[fit_from:fit_to],-amplitude_envelope[fit_from:fit_to],'r--', label='Fit data')
    ax1.legend()
    
    # plots headtail-tail motion    
    for i in range(n_traces):
        ax2.plot(slice_data_z[:,-i-3],slice_data_y[:,-i-3]*n_macroparticles[:,-i-3],'-')
        
    ax3.plot(slice_data_z[:,-5],n_macroparticles[:,-5],'r.')

    ax1.set_xlabel('Turn')
    ax1.set_ylabel('BPM signal')
    
    ax2.set_xlabel('z [m]')
    ax2.set_ylabel('BPM signal')
    
    ax3.set_xlabel('z [m]')
    ax3.set_ylabel('Charge')
    
    plt.show()
    
if __name__=="__main__":
    tunes = np.linspace(3.83-0.05,3.83+0.05,41)
    
    tunes = [3.83]
    
    # scans tune range 
    for i, Q in enumerate(tunes):
            run(i,Q)
            analyze(i)
#    analyze(sys.argv[1:])

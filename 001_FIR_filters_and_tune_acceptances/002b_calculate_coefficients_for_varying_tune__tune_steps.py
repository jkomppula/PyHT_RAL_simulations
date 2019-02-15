import numpy as np
import matplotlib.pyplot as plt

def calculate_coefficients(Q, delay, additional_phase):
    # Generates a 3-tap FIR fiter coefficients for arbitrary 
    # phase advance and delay
    
    Q = -Q
    ppt = 2.* np.pi
    c12 = np.cos(Q *1.*ppt)
    s12 = np.sin(Q *1.*ppt)
    c13 = np.cos(Q *2.*ppt)
    s13 = np.sin(Q *2.*ppt)
    c14 = np.cos((Q *(2+delay)-additional_phase)*ppt)
    s14 = np.sin((Q *(2+delay)-additional_phase)*ppt)
    
    divider = -1.*(-c12*s13+c13*s12-s12+s13)

    cx1 = c14*(1-(c12*s13-c13*s12)/divider)+s14*(-c12+c13)/divider
    cx2 = (c14*(-(-s13))+s14*(-c13+1))/divider
    cx3 = (c14*(-(s12))+s14*(c12-1))/divider
    
    return np.array([cx3, cx2, cx1])


def find_object_locations(optics_file, pickup_object, kicker_object,
                          phase_x_col, beta_x_col,
                          phase_y_col, beta_y_col):
    
    with open(optics_file) as f:
        content = f.readlines()
        content = [x.strip() for x in content]

    list_of_objects = [(pickup_object),(kicker_object)]
    object_locations = []
    for j, element in enumerate(list_of_objects):
        element_name = element
        line_found = False
        for i, l in enumerate(content):
            s = l.split()
            
            if s[0] == ('"' + element_name + '"'):
                line_found = True
                object_locations.append((element,
                                 float(s[phase_x_col]),
                                 float(s[beta_x_col]),
                                 float(s[phase_y_col]),
                                 float(s[beta_y_col])))
                                
        if line_found == False:
            raise ValueError('Element ' + element_name + ' not found from the optics file ' + optics_file)
    
    print('| Element name | Phase x | Beta X  | Phase y | Beta y  |')
    print('========================================================')
    for d in object_locations:
        print('| {:12.5} | {:7.3f} | {:6.2f}  | {:7.3f} | {:6.2f}  |'.format(d[0], d[1], d[2], d[3], d[4]))
    
    phase_difference_x = object_locations[1][1]-object_locations[0][1]
    phase_difference_y = object_locations[1][3]-object_locations[0][3]
    
    print('')
    print('')
    print('Phase advance difference between pickup ' + str(pickup_element) + ' and kicker ' + str(kicker_element) + ':')
    print('| Plane | Tune units | Radian  | Degrees |')
    print('|========================================|')
    print('| {:5.5} | {:10.3f} | {:6.3f}  | {:7.2f} |'.format('X', phase_difference_x, phase_difference_x*2.*np.pi, phase_difference_x*360.))
    print('| {:5.5} | {:10.3f} | {:6.3f}  | {:7.2f} |'.format('Y', phase_difference_y, phase_difference_y*2.*np.pi, phase_difference_y*360.))
    
    
    return phase_difference_x, phase_difference_y


###########################################
## READS PHASE ADVANCE FROM A TWISS FILE ##
###########################################

twiss_file = 'ISIS_twiss.dat'
pickup_element = 'R4VM1'
kicker_element = 'R6VM1'

  
phase_x_column = 5
beta_x_column = 3
phase_y_column = 8
beta_y_column = 6



phase_difference_x, phase_difference_y = find_object_locations(twiss_file, pickup_element, kicker_element,
                          phase_x_column, beta_x_column,
                          phase_y_column, beta_y_column)


########################################
## CALCULATES FIR FILTER COEFFICIENTS ##
########################################

# Tune 
#Q = 3.75 

# Minimum signal processing delay in turns
delay = 3

# Phase advance between the pickup and the kicker. Additional 0.25 must
# be added because the pickup reads beam displacement but changes beam
# angle.
additional_phase = phase_difference_y + 0.25

tunes = np.loadtxt('Tune_data_MD.csv')


#times_for_filter = np.arange(1.5, 5,0.05)
times_for_filter_temp = np.linspace(0, 10,100000)
tunes_for_filter_temp = np.interp(times_for_filter_temp, tunes[:,0], tunes[:,1])

max_tune_step = 0.02

times_for_filter = []
tunes_for_filter = []

times_for_filter.append(times_for_filter_temp[0])
tunes_for_filter.append(tunes_for_filter_temp[0])

for i, Q in enumerate(tunes_for_filter_temp):
#    print('np.abs(Q - tunes_for_filter_temp[-1]) > max_tune_step')
#    print(np.abs(Q - tunes_for_filter_temp[-1]))
    if np.abs(Q - tunes_for_filter[-1]) > max_tune_step:
#        print('i: ' + str(i))
#        print('Q: ' + str(Q))
#        print('tunes_for_filter_temp[-1]: ' + str(tunes_for_filter[-1]))
        times_for_filter.append(times_for_filter_temp[i])
        tunes_for_filter.append(tunes_for_filter_temp[i])
        


coefficients = np.zeros((len(tunes_for_filter),3))

for i, Q in enumerate(tunes_for_filter):
    if i < (len(tunes_for_filter)-1):
        a, b, c = calculate_coefficients((Q+tunes_for_filter[i+1])/2., delay, additional_phase)
    else:
        a, b, c = calculate_coefficients(Q, delay, additional_phase)
    coefficients[i,0] = a
    coefficients[i,1] = b
    coefficients[i,2] = c
    
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4,4), sharex=True)
ax1.plot(tunes[:,0], tunes[:,1])
ax1.plot(times_for_filter, tunes_for_filter, 'ro')
ax1.set_ylabel('Vertical tune')
ax1.set_xlim(-0.1,7.3)

ax2.plot(times_for_filter, coefficients[:,0],'C0.-', label='$b_0$')
ax2.plot(times_for_filter, coefficients[:,1],'C1.-', label='$b_1$')
ax2.plot(times_for_filter, coefficients[:,2],'C2.-', label='$b_2$')
ax2.set_xlabel('Time [ms]')
ax2.set_ylabel('FIR coefficients')
ax2.legend()
print('')
print('FIR filter coefficients:')
print(calculate_coefficients(Q, delay, additional_phase))
plt.tight_layout()
fig.savefig('varying_filter_coefficients.png', format='png', dpi=300)

output = np.zeros((len(times_for_filter),5))
np.copyto(output[:,0],times_for_filter)
np.copyto(output[:,1],tunes_for_filter)
np.copyto(output[:,2],coefficients[:,0])
np.copyto(output[:,3],coefficients[:,1])
np.copyto(output[:,4],coefficients[:,2])
np.savetxt('filter_time_evoluation.txt', output)


print('| Time [ms] |  Tune  |   b_0   |   b_1   |   b_2   |')
print('====================================================')
for i in range(len(times_for_filter)):
    print('| {:9.3} | {:6.3f} | {:7.3f} | {:7.3f} | {:7.3f}  |'.format(output[i,0], output[i,1], output[i,2], output[i,3], output[i,4]))


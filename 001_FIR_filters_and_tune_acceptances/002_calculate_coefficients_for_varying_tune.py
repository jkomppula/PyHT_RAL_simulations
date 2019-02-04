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
Q = 3.75 

# Minimum signal processing delay in turns
delay = 3

# Phase advance between the pickup and the kicker. Additional 0.25 must
# be added because the pickup reads beam displacement but changes beam
# angle.
additional_phase = phase_difference_y + 0.25

tunes = np.loadtxt('Tune_data.csv')


times_for_filter = np.linspace(np.min(tunes[:,0]), np.max(tunes[:,0]),100)
tunes_for_filter = np.interp(times_for_filter, tunes[:,0], tunes[:,1])

coefficients = np.zeros((len(tunes_for_filter),3))

for i, Q in enumerate(tunes_for_filter):
    a, b, c = calculate_coefficients(Q, delay, additional_phase)
    coefficients[i,0] = a
    coefficients[i,1] = b
    coefficients[i,2] = c
    
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7,5), sharex=True)
ax1.plot(tunes[:,0], tunes[:,1])
ax1.set_ylabel('Tune')


ax2.plot(times_for_filter, coefficients[:,0], label='$b_0$')
ax2.plot(times_for_filter, coefficients[:,1], label='$b_1$')
ax2.plot(times_for_filter, coefficients[:,2], label='$b_2$')
ax2.set_xlabel('Time [ms]')
ax2.set_ylabel('Coefficients value')
ax2.legend()
print('')
print('FIR filter coefficients:')
print(calculate_coefficients(Q, delay, additional_phase))


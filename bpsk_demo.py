import matplotlib
matplotlib.use('Agg')  # Non-interactive backends such as Agg do not require GUI support

import numpy as np
import matplotlib.pyplot as plt

### BPSK DEMO ###

## Binary Phase Shift Keying modulates a data stream of 1s and 0s onto a Carrier ##
## This demo will construct the wave so it is well understood                    ## 

### END OF HEADER ###


#### GLOBAL VARIABLES ####
## Frequency: The frequency of the main carrier ##
## This needs to be at least a multiple of 6, 10, 14, etc so the code doesn't break ##
frequency = 5 # the frequency of the carrier
## Symbol Rate: The nyquist limit is 2 times the frequency ##
    ## You will see worse performance if this value is changed ##
    
## this "symbol_rate" is actually 2x the symbol_frequency, pretty confusing lol
symbol_rate = 10
duration = 1
sampling_rate = 300
def create_carrier_data(duration, sampling_rate, frequency):
    
    ## Creating the x values to be used in the rest of the plots based on the global variables ##
    x = np.linspace(0, duration, int(sampling_rate*duration))

    ## Phase Shift: Where the carrier starts in relation to the data ##
    ## This effectively changes the sync marker, you can see some interesting results when this is configured poorly ##
    phase_shift = np.pi/2

    ## Carrier Amplitude: Creatation of the Carrier Wave ##
    carrier_amplitude = np.cos(2*np.pi*frequency*x+phase_shift)

    return carrier_amplitude, x

carrier_amplitude, x = create_carrier_data(duration, sampling_rate, frequency)

#### END OF GLOBAL VARIABLES ####

##### SQUARE BINARY DATA #####

# Binary data is strings of 1s and 0s. We are using a NRZ format so for us it is strings of 1s and -1s
# Real world binary data is: infinite and random.
# What this means for us is that binary data is actually the mixing of squarewaves with increasing periods
# The smallest period is our symbol rate. i.e. the shortest we have to wait before changing to a new signal

## Lets start with a square wave function
def repeat_for_length(longer_array,shorter_array):
    return np.append(shorter_array,shorter_array[0:(len(longer_array)-len(shorter_array))])

def create_rectangular_wave(half_period,symbol_rate,wave_length):
    binary_segments = np.tile(np.repeat([1,-1],half_period),int(symbol_rate / (2*half_period)))
    data_amplitude = repeat_for_length(wave_length, np.repeat(binary_segments, sampling_rate//symbol_rate))
    return data_amplitude

def create_random_binary_wave(symbol_rate,wave_length):
    binary_segments = np.random.choice([-1,1], size=symbol_rate)
    data_amplitude = repeat_for_length(wave_length, np.repeat(binary_segments, sampling_rate//symbol_rate))
    return data_amplitude

data_amplitude = create_random_binary_wave(symbol_rate, carrier_amplitude)
# however the demo looks better with random data lol

##### END OF SQUARE BINARY DATA #####

###### BPSK DATA ######
# Modulation in the time domain requires multiplying our data signal to our binary signal
def create_bpsk_time_domain(carrier_data, symbol_rate, harmonics=1, random=False):
    if random:
        binary_data = create_random_binary_wave(symbol_rate, carrier_amplitude)
    else:
        binary_data = [1]*len(carrier_data)
        for i in range(harmonics):
            binary_data *= create_rectangular_wave(i+1, symbol_rate, carrier_amplitude)
    bpsk_amplitude = carrier_data * binary_data 
    return bpsk_amplitude



bpsk_amplitude = carrier_amplitude * data_amplitude

###### END OF BPSK DATA ######

####### PLOT THE TIME DOMAIN #######
fig, ax = plt.subplots(3, 1, figsize=(10, 15))

# First subplot: Sine Wave
ax[0].plot(x, carrier_amplitude, label='Sine Wave', color='blue')
ax[0].set_title('Carrier Sine Wave')
ax[0].set(xlabel='Time (s)')
ax[0].grid(True)

# Second subplot: Data Wave
ax[1].plot(x, data_amplitude, label='Data Wave', color='red')
ax[1].set_title('Square Binary Wave')
ax[1].set(xlabel='Time (s)')
ax[1].grid(True)

# Third subplot: BPSK Wave
ax[2].plot(x, bpsk_amplitude, label='BPSK Wave', color='green')
ax[2].set_title('BPSK Modulated Wave')
ax[2].set(xlabel='Time (s)')
ax[2].grid(True)

plt.tight_layout()
plt.savefig('plot1_simple_time_domain.png')

####### END OF PLOT THE TIME DOMAIN #######

### create much larger data so that the ffts work fine lol ###
frequency = 2500
symbol_rate = 50
duration = 1
sampling_rate = 3000000
xlim = symbol_rate * 4
carrier_amplitude, x = create_carrier_data(duration, sampling_rate, frequency)

######## Convert to the Frequency Domain ########
# take the fourier transform of our time domain signals 

# filter our frequencies so we are close to the action

def create_fft(time_signal, frequency):
    fft_frequencies = np.fft.fftfreq(len(time_signal), d=(x[1] - x[0]))
    fft_signal = abs(np.fft.fft(time_signal))
    return fft_frequencies, fft_signal

ff, fft_carrier = create_fft(carrier_amplitude, frequency)
ff, fft_data = create_fft(create_rectangular_wave(1, symbol_rate, carrier_amplitude), symbol_rate)
ff, fft_bpsk = create_fft(create_bpsk_time_domain(carrier_amplitude, symbol_rate), frequency)

######## END OF Convert to the Frequency Domain ########

######### PLOT THE FREQUENCY DOMAIN #########
fig, ax = plt.subplots(3, 1, figsize=(10, 15))




ax[0].plot(ff, fft_carrier, label='FFT of Sine Wave', color='purple')
ax[0].set_title('Fourier Transform of Sine Wave')
ax[0].set(xlabel='Frequency (Hz)')
ax[0].grid(True)
ax[0].set_xlim([-frequency*8, frequency*8])

ax[1].plot(ff, fft_data, label='FFT of Data Wave', color='orange')
ax[1].set_title('Fourier Transform of Square Wave')
ax[1].set(xlabel='Frequency (Hz)')
ax[1].grid(True)
ax[1].set_xlim([-xlim, xlim])

ax[2].plot(ff, fft_bpsk, label='FFT of BPSK Wave', color='red')
ax[2].set_title('Fourier Transform of BPSK Wave')
ax[2].set(xlabel='Frequency (Hz)')
ax[2].grid(True)
ax[2].set_xlim([frequency-xlim, frequency+xlim])

plt.tight_layout()
plt.savefig('plot2_simple_freq_domain.png')

######### END OF PLOT THE FREQUENCY DOMAIN #########
######### PLOT THE FREQUENCY DOMAIN #########
fig, ax = plt.subplots(3, 1, figsize=(10, 15))
xlim = xlim
# First subplot: FFT of Sine Wave
ax[0].plot(ff, fft_carrier, label='FFT of Sine Wave', color='purple')
ax[0].set_title('Fourier Transform of Sine Wave')
ax[0].set(xlabel='Frequency (Hz)')
ax[0].grid(True)
ax[0].set_xlim([0, frequency*8])

# Second subplot: FFT of Data Wave
ax[1].plot(ff, fft_data, label='FFT of Data Wave', color='orange')
ax[1].set_title('Fourier Transform of Square Wave')
ax[1].set(xlabel='Frequency (Hz)')
ax[1].grid(True)
ax[1].set_xlim([0, xlim])

# Third subplot: FFT of BPSK Wave
ax[2].plot(ff, fft_bpsk, label='FFT of BPSK Wave', color='red')
ax[2].set_title('Fourier Transform of BPSK Wave')
ax[2].set(xlabel='Frequency (Hz)')
ax[2].grid(True)
ax[2].set_xlim([frequency-xlim, frequency+xlim])

plt.tight_layout()
plt.savefig('plot3_simple_freq_domain_pos.png')

######### END OF PLOT THE FREQUENCY DOMAIN #########


########## MAKE THE BINARY DATA MORE RANDOM ##########
# Remember how BPSK data is actually random, this means we need to add rectangular waves that are odd harmonics of the symbol

ff, fft_data_random = create_fft(create_random_binary_wave(symbol_rate, carrier_amplitude), frequency)
ff, fft_bpsk_random = create_fft(create_bpsk_time_domain(carrier_amplitude, symbol_rate, random=True), frequency)
ff, fft_data = create_fft(create_rectangular_wave(1, symbol_rate, carrier_amplitude), frequency)
ff, fft_data_2 = create_fft(create_rectangular_wave(2, symbol_rate, carrier_amplitude), frequency)
ff, fft_data_3 = create_fft(create_rectangular_wave(3, symbol_rate, carrier_amplitude), frequency)
ff, fft_data_4 = create_fft(create_rectangular_wave(4, symbol_rate, carrier_amplitude), frequency)
ff, fft_data_5 = create_fft(create_rectangular_wave(5, symbol_rate, carrier_amplitude), frequency)
ff, fft_data_6 = create_fft(create_rectangular_wave(6, symbol_rate, carrier_amplitude), frequency)
ff, fft_bpsk_12 = create_fft(create_bpsk_time_domain(carrier_amplitude, symbol_rate, 2), frequency)
ff, fft_bpsk_123 = create_fft(create_bpsk_time_domain(carrier_amplitude, symbol_rate, 3), frequency)
ff, fft_bpsk_1234 = create_fft(create_bpsk_time_domain(carrier_amplitude, symbol_rate, 4), frequency)
ff, fft_bpsk_12345 = create_fft(create_bpsk_time_domain(carrier_amplitude, symbol_rate, 5), frequency)
ff, fft_bpsk_123456 = create_fft(create_bpsk_time_domain(carrier_amplitude, symbol_rate, 6), frequency)

######### END OF PLOT THE FREQUENCY DOMAIN #########
######### PLOT THE FREQUENCY DOMAIN #########
fig, ax = plt.subplots(3, 2, figsize=(10, 15))
xlim = xlim
# Second subplot: FFT of Data Wave
ax[0,0].plot(ff, fft_data, label='FFT of Data Wave', color='orange')
ax[0,0].plot(ff, fft_data_2, label='FFT of Data Wave', color='blue')
ax[0,0].set_title('Fourier Transform of Square (1,2)')
ax[0,0].set(xlabel='Frequency (Hz)')
ax[0,0].grid(True)
ax[0,0].set_xlim([0, xlim])

ax[0,1].plot(ff, fft_bpsk_12, label='FFT of BPSK_12 Wave', color='orange')
ax[0,1].set_title('Fourier Transform of BPSK Wave (1,2)')
ax[0,1].set(xlabel='Frequency (Hz)')
ax[0,1].grid(True)
ax[0,1].set_xlim([frequency-xlim, frequency+xlim])

ax[1,0].plot(ff, fft_data, label='FFT of Data Wave', color='orange')
ax[1,0].plot(ff, fft_data_2, label='FFT of Data Wave', color='blue')
ax[1,0].plot(ff, fft_data_3, label='FFT of Data Wave', color='red')
ax[1,0].set_title('Fourier Transform of Square (1,2,3)')
ax[1,0].set(xlabel='Frequency (Hz)')
ax[1,0].grid(True)
ax[1,0].set_xlim([0, xlim])

ax[1,1].plot(ff, fft_bpsk_123, label='FFT of BPSK_12 Wave', color='orange')
ax[1,1].set_title('Fourier Transform of BPSK Wave (1,2,3)')
ax[1,1].set(xlabel='Frequency (Hz)')
ax[1,1].grid(True)
ax[1,1].set_xlim([frequency-xlim, frequency+xlim])

ax[2,0].plot(ff, fft_data, label='FFT of Data Wave', color='orange')
ax[2,0].plot(ff, fft_data_2, label='FFT of Data Wave', color='blue')
ax[2,0].plot(ff, fft_data_3, label='FFT of Data Wave', color='red')
ax[2,0].plot(ff, fft_data_4, label='FFT of Data Wave', color='black')
ax[2,0].set_title('Fourier Transform of Square (1,2,3,4)')
ax[2,0].set(xlabel='Frequency (Hz)')
ax[2,0].grid(True)
ax[2,0].set_xlim([0, xlim])

ax[2,1].plot(ff, fft_bpsk_1234, label='FFT of BPSK_12 Wave', color='orange')
ax[2,1].set_title('Fourier Transform of BPSK Wave (1,2,3,4)')
ax[2,1].set(xlabel='Frequency (Hz)')
ax[2,1].grid(True)
ax[2,1].set_xlim([frequency-xlim, frequency+xlim])

plt.tight_layout()
plt.savefig('plot4_freq_domain_extra.png')

######### END OF PLOT THE FREQUENCY DOMAIN #########

## plot the random data fft as well as the fft of the "generated random data"
fig, ax = plt.subplots(3, 2, figsize=(10, 15))
xlim = xlim
# Second subplot: FFT of Data Wave
ax[0,0].plot(ff, fft_data, label='FFT of Data Wave', color='orange')
ax[0,0].plot(ff, fft_data_2, label='FFT of Data Wave', color='blue')
ax[0,0].plot(ff, fft_data_3, label='FFT of Data Wave', color='red')
ax[0,0].plot(ff, fft_data_4, label='FFT of Data Wave', color='black')
ax[0,0].plot(ff, fft_data_5, label='FFT of Data Wave', color='cyan')
ax[0,0].plot(ff, fft_data_6, label='FFT of Data Wave', color='purple')
ax[0,0].set_title('Fourier Transform of Square (1,2,3,4,5,6)')
ax[0,0].set(xlabel='Frequency (Hz)')
ax[0,0].grid(True)
ax[0,0].set_xlim([0, xlim])

ax[0,1].plot(ff, fft_bpsk_123456, label='FFT of BPSK_12 Wave', color='orange')
ax[0,1].set_title('Fourier Transform of BPSK Wave (1,2,3,4,5,6)')
ax[0,1].set(xlabel='Frequency (Hz)')
ax[0,1].grid(True)
ax[0,1].set_xlim([frequency-xlim, frequency+xlim])

ax[1,0].plot(ff, fft_data_random, label='FFT of Data Wave', color='orange')
ax[1,0].set_title('Fourier Transform of Random Data')
ax[1,0].set(xlabel='Frequency (Hz)')
ax[1,0].grid(True)
ax[1,0].set_xlim([0, xlim])

ax[1,1].plot(ff, fft_bpsk_random, label='FFT of BPSK_12 Wave', color='orange')
ax[1,1].set_title('Fourier Transform of Random BPSK')
ax[1,1].set(xlabel='Frequency (Hz)')
ax[1,1].grid(True)
ax[1,1].set_xlim([frequency-xlim, frequency+xlim])

plt.tight_layout()
plt.savefig('plot5_freq_domain_extra.png')

# the apparent dip that forms around the quirk about plotting but it smoothes out
# I am still not sold on the fact that the rectangalur plots are multiplied...
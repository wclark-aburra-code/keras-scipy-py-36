import wave, numpy
from scipy import signal
import utility
import wavefile
from scipy.io import wavfile 
from scipy import fftpack
import math

def is_power2(num):
    return ((num & (num - 1)) == 0) and num != 0

def nextPowerOf2(num,i=1):
    while i < num:
        i *= 2
    return i

class Filter:
    def __init__(self, byte_array, resolution): #output length must be pwr of 2 (RAISE)
        if not isinstance(byte_array, (numpy.ndarray, list)):
            raise TypeError("input must be an array")
        self.resolution = resolution
        self.ir_length = len(byte_array)
        self.output_length = nextPowerOf2(self.ir_length)
        self.window_size = self.output_length - self.ir_length + 1
        self.byte_array = padded(byte_array, self.output_length)
        self.complex_phasors = fftpack.fft(self.byte_array)
    def step_response(self):
        step_impulse = numpy.ones(len(self.byte_array))
        return signal.fftconvolve(step_impulse, self.byte_array)
    def normalization_factor(self):
        return self.step_response().max()
    def convolve(self, dry_signal):
        return signal.fftconvolve(dry_signal, self.byte_array)

def padded(byte_array, output_length):
    return numpy.append(byte_array,numpy.zeros(output_length-len(byte_array))) # check - append, not concat

def hanning_multiplier(i, block_size):
    return 0.5 * (1. - numpy.cos(2.0*numpy.pi*i/(block_size-1)))

def overlap_add(filter_object, dry_signal):
    num_sections = math.floor(len(dry_signal)/filter_object.window_size)
    if num_sections * filter_object.window_size != len(dry_signal) :
        dry_signal = padded(dry_signal,(num_sections+1)*filter_object.window_size)
        num_sections += 1
    overlap_length = filter_object.ir_length - 1
    sections_list = numpy.split(dry_signal, num_sections)
    overlap_kernel = numpy.zeros(overlap_length)
    output = numpy.array([])
    for section in sections_list:
        section_length = len(section)
        padded_section = numpy.zeros(filter_object.output_length)
        for n, sample in enumerate(section):
            padded_section[n] = hanning_multiplier(n,section_length)*section[n]
        section_phasors = fftpack.fft(padded_section)
        convolved_section_phasors = numpy.multiply(section_phasors, filter_object.complex_phasors) # dbl check complex multiplication
        convolved_section_samples = fftpack.ifft(convolved_section_phasors)
        overlapped_sum = numpy.copy(convolved_section_samples[0:section_length]) # must copy, we refer to this later
        for n, _ in enumerate(overlapped_sum):
            overlapped_sum[n] += overlap_kernel[n]
        output = numpy.append(output, overlapped_sum)
        overlap_kernel = convolved_section_samples[section_length:overlap_length]
    return numpy.real(output)

class OverlapAddingMachine:        
    def __init__(self, filter_object, dry_signal):
        self.filter_object = filter_object
        self.num_sections = math.floor(len(dry_signal)/filter_object.window_size)
        self.dry_signal = dry_signal
        if self.num_sections * filter_object.window_size != len(dry_signal) :
            self.dry_signal = padded(self.dry_signal,(self.num_sections+1)*self.filter_object.window_size)
            self.num_sections += 1
        self.overlap_length = filter_object.ir_length - 1
        self.sections_list = numpy.split(self.dry_signal, self.num_sections)
        self.overlap_kernel = numpy.zeros(self.overlap_length)
        self.output = numpy.array([])
        self.section_num = 0
        self.section_length = len(self.sections_list[0])    
    def __next__(self):        
        if (self.section_num+1 < self.num_sections):
            self.section_num += 1
            section = self.sections_list[self.section_num]
            padded_section = numpy.zeros(self.filter_object.output_length)
            for n, sample in enumerate(section):
                padded_section[n] = hanning_multiplier(n,self.section_length)*section[n]
            section_phasors = fftpack.fft(padded_section)
            convolved_section_phasors = numpy.multiply(section_phasors, self.filter_object.complex_phasors)
            convolved_section_samples = fftpack.ifft(convolved_section_phasors)
            overlapped_sum = numpy.copy(convolved_section_samples[0:self.section_length])
            for n, _ in enumerate(overlapped_sum):
                overlapped_sum[n] += self.overlap_kernel[n]
            self.overlap_kernel = convolved_section_samples[self.section_length:self.overlap_length]
            return numpy.real(overlapped_sum)
        else:
            raise StopIteration

def unit_impulse():
    return numpy.array([1.])

class EffectChain:
    def __init__(self, filter_list):
        self.filter_list = filter_list
    def convolve(self, dry_signal):
        temp_signal = dry_signal
        for filter_object in self.filter_list:
            temp_signal = filter_object.convolve(temp_signal)
        return temp_signal
    def as_convolved_filter(self):
        return self.convolve(unit_impulse())

# sine generator for tests
def sin_signal(freq, length):
    x = numpy.linspace(0,length-1) # or arange
    return numpy.sin(2*(numpy.pi)*freq*x)

# naive IR's for tests
multitap_delay = numpy.array([1., 0., 0.6, 0., 0.4, 0., 0.2])
slapback_delay = numpy.array([1., 0.8])
basic_reverb = numpy.array([0., .1, -.1, .2, -.1, .2, -.1])
turn_up = numpy.array([2.])
lowpass_filter = numpy.array([1., 0.2, 0.2, 0.2, 0.2, 0.2]) # moving average
linear_phase_highpass_filter = numpy.array([-0.2, -0.2, 1., -0.2, -0.2])
mute = numpy.array([0.])
differential = numpy.array([1., -1.]) # like a high pass filter

def input_side_convolve(sig1, sig2):
    output = numpy.zeros(len(sig1) + len(sig2) - 1)
    for num1, sample1 in enumerate(sig1):
        for num2, sample2 in enumerate(sig2):
            output[num1 + num2] += sample1 * sample2
    return output

class WaveSample:        
    def __init__(self, filename):
        wf = wavefile.load(filename=filename)
        self.resolution = wf[0]
        self.signal_vector = wf[1]
        if self.signal_vector.shape[0] == 2:
            self.left = self.signal_vector[0]
            self.right = self.signal_vector[1]            
        elif self.signal_vector.shape[0] == 1:
            self.mono = self.signal_vector[0]

# local manual tests
def test_local_files():
    input_signal = wavefile.load(filename="./audio_dataset/test/hi_hat/ALCHH36.WAV")
    impulse_response = wavefile.load(filename="./impulse_responses/spaceEchoIR.wav")  # already floating point bytearray
    second_IR = wavefile.load(filename="./impulse_responses/echo2IR.wav")
    (left, right) = input_signal[1]
    output_signal = signal.fftconvolve(input_signal[1][0], impulse_response[1][0]) 
    output_signal2 = signal.fftconvolve(output_signal, second_IR[1][0])
    wavfile.write("./audio_dataset/convolved_hihat.wav", 44100, utility.float2pcm(output_signal2))
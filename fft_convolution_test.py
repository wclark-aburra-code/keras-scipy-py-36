
from fft_convolution import Filter, unit_impulse
import numpy
import unittest

multitap_delay = numpy.array([1., 0., 0.6, 0., 0.4, 0., 0.2])
slapback_delay = numpy.array([1., 0.8])
basic_reverb = numpy.array([0., .1, -.1, .2, -.1, .2, -.1])
turn_up = numpy.array([2.])
lowpass_filter = numpy.array([1., 0.2, 0.2, 0.2, 0.2, 0.2]) # moving average
linear_phase_highpass_filter = numpy.array([-0.2, -0.2, 1., -0.2, -0.2])
mute = numpy.array([0.])
differential = numpy.array([1., -1.]) # like a high pass filter

test_signal = numpy.array([1.,0.,-1.,0])

class FilterTest(unittest.TestCase):
    def testConvolveTurnUp(self):
        f0 = Filter(turn_up,10)
        i0 = unit_impulse()
        total_length = len(i0) + len(turn_up) - 1 
        wet_signal = f0.convolve(i0)
        clipped_signal = wet_signal[0:total_length]
        self.assertTrue(numpy.allclose(clipped_signal, numpy.array([2.])))
    def testConvolveDelay(self):
        f0 = Filter(slapback_delay,10)
        i0 = [1., 0., -1., -.1]
        total_length = len(slapback_delay) + len(i0) - 1 
        wet_signal = f0.convolve(i0)
        clipped_signal = wet_signal[0:total_length]
        self.assertTrue(numpy.allclose(clipped_signal, numpy.array([ 1.  ,  0.8 , -1.  , -0.9 , -0.08])))
    def testStepResponse(self):
        bigMovingAverage = numpy.array([1.,1.,1.,1.])
        filterMovingAvg = Filter(bigMovingAverage,10)
        self.assertTrue(numpy.allclose(filterMovingAvg.step_response(), numpy.array([1., 2., 3., 4., 3., 2., 1.])))
        self.assertEqual(round(filterMovingAvg.normalization_factor()), 4.0)
    def testError(self):
        with self.assertRaises(TypeError):
            Filter(0) # diff sig?

if __name__ == '__main__':
    unittest.main()
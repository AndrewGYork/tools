import ctypes as C
import numpy as np
"""
TODO: One way or another, make it harder to forget the daq.close()
method, which can cause crazy voltages to persist. _del_? _enter_ and
_exit_? Try to do it better than we are.

Requires nicaiu.dll to be in the same directory, or located in the
os.environ['PATH'] search path.

If you get an error, google for NIDAQmx.h to decypher it.
"""
api = C.cdll.LoadLibrary("nicaiu")

class Analog_Out:
    def __init__(
        self,
        num_channels='all',
        rate=1e4,
        verbose=True,
        daq_type='6733',
        board_name='Dev1', # Also popular: 'cDAQ1Mod1'
        clock_name=None,
        ):
        """Play analog voltages via a National Instruments analog-out DAQ board.

        So far, I've only tested this for the PCI 6733 and the NI 9263.
        """
        allowed_cards = ['6733', '6733_digital',
                         '6738', '6738_digital',
                         '6739', '6739_digital',
                         '9263', '9401', '6001']
        assert daq_type in allowed_cards, "We don't support this card (%s) yet" % daq_type
        self.daq_type = daq_type
        if self.daq_type == '6733':
            self.max_channels = 8
            self.max_rate = 1e6
            self.channel_type = 'analog'
            self.has_clock = True
        elif self.daq_type =='6733_digital':
            # WARNING: Init is weird for 6733 digital lines, because
            # they don't have their own clock. You probably want to use
            # the analog clock of the 6733 as the clock for the digital
            # lines. Since the init method relies on this clock playing,
            # in order to zero the output voltages, this can get
            # confusing. Here's a recipe to follow:
            #
            # do = Analog_Out(num_channels=8, rate=2e4, daq_type='6733_digital',
            #     board_name='Dev1', clock_name='/Dev1/ao/SampleClock')
            #     # This leaves the digital lines stalled, waiting for a clock.
            # ao = Analog_Out(num_channels=8, rate=2e4, daq_type='6733',
            #     board_name='Dev1', clock_name=None)
            #     # This has the side effect of finishing the digital init.
            #
            # Then, later, when you want to play digital voltages, you
            # have to use non-blocking calls and play analog voltages too:
            #
            # do.play_voltages(block=False) # Stalls
            # ao.play_voltages(block=True)  # Plays both types, synchronized
            self.max_channels = 8
            self.max_rate = 1e6
            self.channel_type = 'digital'
            self.has_clock = False
        elif self.daq_type == '6738':
            self.max_channels = 32
            self.max_rate = 1e6 #TODO is this the correct max rate?
            self.channel_type = 'analog'
            self.has_clock = True
        elif self.daq_type == '6738_digital':
            # WARNING: See note about 6733_digital lines. Also note that there
            # are 8 digital lines on `port1`, but these are not "buffered". We
            # have not yet included any functionality for these lines.
            self.max_channels = 2
            self.max_rate = 1e6 #TODO is this the correct max rate?
            self.channel_type = 'digital'
            self.has_clock = False
        elif self.daq_type == '6739':
            self.max_channels = 64 #TODO is this correct?
            self.max_rate = 1e6 #TODO is this the correct max rate?
            self.channel_type = 'analog'
            self.has_clock = True
        elif self.daq_type == '6739_digital':
            self.max_channels = 2 #TODO is this correct?
            self.max_rate = 1e6 #TODO is this the correct max rate?
            self.channel_type = 'digital'
            self.has_clock = False
        elif self.daq_type == '9263':
            self.max_channels = 4
            self.max_rate = 1e5
            self.channel_type = 'analog'
            self.has_clock = True
        elif self.daq_type == '9401':
            self.max_channels = 8
            self.max_rate = 8e7
            self.channel_type = 'digital'
            self.has_clock = True
        elif self.daq_type == '6001':
            ## NOTE: This only controls the two AO channels on the
            ## USB-6001. There are a number of other features of this
            ## device (AI, DI/O, Counters) that are not exposed by this
            ## class. It will take more effort and likely a large
            ## reorganization of this code to expose these
            ## functionalities.
            self.max_channels = 2
            self.max_rate = 5e3
            self.channel_type = 'analog'
            self.has_clock = True
        if num_channels == 'all':
            num_channels = self.max_channels
        assert 1 <= num_channels <= self.max_channels
        self.num_channels = num_channels
        if clock_name is not None:
            assert isinstance(clock_name, str)
            clock_name = bytes(clock_name, 'ascii')
        self.verbose = verbose

        if self.verbose: print("Opening %s-out board..."%self.channel_type)
        self.task_handle = C.c_void_p(0)
        api.create_task(bytes(), self.task_handle)
        # If I were a real man, I would automatically detect the proper
        # board name somehow
    # (http://digital.ni.com/public.nsf/allkb/86256F0E001DA9FF492572A5006FD7D3)
        # instead of demanding user input via the 'init' argument. If
        # this next api call crashes for you, check the name of your
        # analog-out card using NI Measurement and Automation Explorer
        # (NI MAX):
        device_name = bytes(
            board_name +
            {'digital':'/port0/line', 'analog':'/ao'}[self.channel_type] +
            '0:%i'%(self.num_channels - 1),
            'ascii')
        if self.channel_type == 'analog':
            api.create_ao_voltage_channel(
                self.task_handle,
                device_name,
                b"",
                -10, #Minimum voltage
                +10.0, #Maximum voltage
                10348, #DAQmx_Val_Volts; don't question it!
                None) #NULL
        elif self.channel_type == 'digital':
            api.create_do_channel(
                self.task_handle,
                device_name,
                b"",
                1) #DAQmx_Val_ChanForAllLines; don't question it!
        if self.verbose: print(" Board open.")
        self.board_name = board_name
        dtype = {'digital': np.uint8, 'analog': np.float64}[self.channel_type]
        self.voltages = np.zeros((2, self.num_channels), dtype=dtype)
        # Play initial voltages with the internal clock
        if self.has_clock:
            self.clock_name = None
        else:
            self.clock_name = clock_name
        self.set_rate(rate)
        self._write_voltages(self.voltages)
        if self.has_clock:
            self.play_voltages(force_final_zeros=False, block=True)
        else:
            self.play_voltages(force_final_zeros=False, block=False)
        if clock_name is not None and self.has_clock: # Switch to external clock
            self.clock_name = clock_name
            self.set_rate(rate)
        return None

    def set_rate(self, rate):
        self._ensure_task_is_stopped()
        assert 0 < rate <= self.max_rate
        self.rate = float(rate)
        api.clock_timing(
            self.task_handle,
            self.clock_name, #NULL, to specify onboard clock for timing
            self.rate,
            10280, #DAQmx_Val_Rising (doesn't matter)
            10178, #DAQmx_Val_FiniteSamps (run once)
            self.voltages.shape[0])
        return None

    def play_voltages(
        self,
        voltages=None,
        force_final_zeros=True,
        block=True,
        ):
        """
        If voltage is None, play the previously set voltage.
        If 'force_final_zeros', the last entry of each channel of
        'voltages' is set to zero.
        If 'block', this function will not return until the voltages are
        finished playing. Not performant, but easier to reason about.

        NB: by default, play_voltages() blocks until the voltages finish
        playing. This makes it harder to accidentally code yourself into
        ugly race conditions, but it obviously makes it hard to do
        anything else while the board is playing voltages. Since
        we're just issuing a DLL call, it's easy for play_voltages() to
        return as soon as the voltage task has started playing. This is
        probably what you want! But easier to write bugs with.
        Regardless, if a previous voltage task is still playing, we have
        to wait for it to finish before we can start the next one.
        """
        self._ensure_task_is_stopped()
        if voltages is not None:
            self._write_voltages(voltages, force_final_zeros)
        if self.verbose: print("Playing voltages...")
        api.start_task(self.task_handle)
        self._task_running = True
        if block:
            self._ensure_task_is_stopped()
        return None

    def close(self):
        self._ensure_task_is_stopped()
        if self.verbose: print("Closing %s board..."%self.daq_type)
        api.clear_task(self.task_handle)
        if self.verbose: print(" %s board is closed."%self.daq_type)
        return None

    def s2p(self, seconds):
        '''Convert a duration in seconds to a number of AO "pixels."

        Frequently I know how many seconds I want to play a voltage for,
        and I do simple math to convert this to how many "pixels" of
        voltage I should use on the analog out card to get this many
        seconds. Frequently I get this math wrong. That's why I wrote
        this function.
        '''
        num_pixels = int(round(self.rate * seconds))
        return num_pixels

    def p2s(self, num_pixels):
        '''Convert a  number of AO "pixels to a duration in seconds."

        Frequently I know how many "pixels" of voltage I'm playing on
        the analog out card, and I do simple math to convert this to how
        many seconds I'm playing that voltage for. Frequently I get this
        math wrong. That's why I wrote this function.
        '''
        seconds = num_pixels / self.rate
        return seconds

    def s2s(self, seconds):
        '''Calculate nearest duration the AO card can exactly deliver.

        This function rounds a time (in seconds) to the nearest time
        that the AO card can exactly deliver via an integer number of
        "pixels". For example, maybe you'd like to play a 10 millisecond
        pulse of voltage, but the AO rate is set to 300; how many
        "pixels" should we expect the AO card play for?
        '''
        seconds = self.p2s(self.s2p(seconds))
        return seconds

    def _ensure_task_is_stopped(self):
        if not hasattr(self, '_task_running'):
            self._task_running = False
        if self._task_running:
            if self.verbose: print("Waiting for board to finish playing...")
            api.finish_task(self.task_handle, -1)
            if self.verbose: print(" NI%s is finished playing."%self.daq_type)
            api.stop_task(self.task_handle)
            self._task_running = False
        return None

    def _write_voltages(self, voltages, force_final_zeros=True):
        assert len(voltages.shape) == 2
        assert voltages.dtype == self.voltages.dtype
        assert voltages.shape[0] >= 2
        assert voltages.shape[1] == self.num_channels
        if force_final_zeros:
            if self.verbose:
                print("***Coercing voltages to end in zero!***")
            voltages[-1, :] = 0
        old_voltages_shape = self.voltages.shape
        self.voltages = voltages
        if self.voltages.shape[0] != old_voltages_shape[0]:
            self.set_rate(self.rate)
        if not hasattr(self, 'num_points_written'):
            self.num_points_written = C.c_int32(0)
        write = {'analog': api.write_voltages,
                 'digital': api.write_digits}[self.channel_type]
        self._ensure_task_is_stopped()
        write(
            self.task_handle,
            self.voltages.shape[0], #Samples per channel
            0, #Set autostart to False
            10.0, #Timeout for writing, in seconds. We could be smarter...
            1, #DAQmx_Val_GroupByScanNumber (interleaved)
            self.voltages,
            self.num_points_written,
            None)
        if self.verbose:
            print(self.num_points_written.value,
                  "points written to each %s channel."%self.daq_type)
        return None


PCI_6733 = Analog_Out # Backwards compatible

# DLL api management
#
# Mostly just sets a bunch of argtypes and renames the DLL functions to
# a pythonier style.
api.get_error_info = api.DAQmxGetExtendedErrorInfo
api.get_error_info.argtypes = [C.c_char_p, C.c_uint32]

def check_error(error_code):
    if error_code != 0:
        num_bytes = api.get_error_info(None, 0)
        print("Error message from NI DAQ: (", num_bytes, "bytes )")
        error_buffer = (C.c_char * num_bytes)()
        api.get_error_info(error_buffer, num_bytes)
        print(error_buffer.value.decode('ascii'))
        raise UserWarning(
            "NI DAQ error code: %i; see above for details."%(error_code))
    return error_code

api.create_task = api.DAQmxCreateTask
api.create_task.argtypes = [C.c_char_p, C.POINTER(C.c_void_p)]
api.create_task.restype = check_error

api.create_ao_voltage_channel = api.DAQmxCreateAOVoltageChan
api.create_ao_voltage_channel.argtypes = [
    C.c_void_p,
    C.c_char_p,
    C.c_char_p,
    C.c_double,
    C.c_double,
    C.c_int32,
    C.c_char_p]
api.create_ao_voltage_channel.restype = check_error

api.create_do_channel = api.DAQmxCreateDOChan
api.create_do_channel.argtypes = [
    C.c_void_p,
    C.c_char_p,
    C.c_char_p,
    C.c_int32]
api.create_do_channel.restype = check_error

api.clock_timing = api.DAQmxCfgSampClkTiming
api.clock_timing.argtypes = [
    C.c_void_p,
    C.c_char_p,
    C.c_double,
    C.c_int32,
    C.c_int32,
    C.c_uint64]
api.clock_timing.restype = check_error

api.write_voltages = api.DAQmxWriteAnalogF64
api.write_voltages.argtypes = [
    C.c_void_p,
    C.c_int32,
    C.c_uint32, #NI calls this a 'bool32' haha awesome
    C.c_double,
    C.c_uint32,
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2), #Numpy is awesome.
    C.POINTER(C.c_int32),
    C.POINTER(C.c_uint32)]
api.write_voltages.restype = check_error

api.write_digits = api.DAQmxWriteDigitalLines
api.write_digits.argtypes = [
    C.c_void_p,
    C.c_int32,
    C.c_uint32, #NI calls this a 'bool32' haha awesome
    C.c_double,
    C.c_uint32,
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2), #Numpy is awesome.
    C.POINTER(C.c_int32),
    C.POINTER(C.c_uint32)]
api.write_digits.restype = check_error

api.start_task = api.DAQmxStartTask
api.start_task.argtypes = [C.c_void_p]
api.start_task.restype = check_error

api.finish_task = api.DAQmxWaitUntilTaskDone
api.finish_task.argtypes = [C.c_void_p, C.c_double]
api.finish_task.restype = check_error

api.stop_task = api.DAQmxStopTask
api.stop_task.argtypes = [C.c_void_p]
api.stop_task.restype = check_error

api.clear_task = api.DAQmxClearTask
api.clear_task.argtypes = [C.c_void_p]
api.clear_task.restype = check_error

if __name__ == '__main__':
    ## Test basic functionality of the Analog_Out object
    # daq = Analog_Out(
    #     rate=1e3,
    #     num_channels=2,
    #     verbose=True,
    #     daq_type='6001', # Change this if you want to test another card
    #     board_name='Dev1')
    # try:
    #     daq.play_voltages()
    #     v = np.ones((1000, daq.num_channels), dtype=np.float64)
    #     v[:, :] = np.sin(np.linspace(0, np.pi, v.shape[0]
    #                                  )).reshape(v.shape[0], 1)
    #     daq.play_voltages(v)
    #     daq.verbose=False
    #     for i in range(10):
    #         daq.play_voltages()
    # finally:
    #     daq.verbose = True
    #     daq.close()

## This block tests an AO/DO play of 9401/9263 cards in a cDAQ-9174 chassis.

    ## 6733 test block
    # rate = 2e4
    # do_type = '6733_digital'
    # do_name = 'Dev1'
    # do_nchannels = 8
    # do_clock = '/Dev1/ao/SampleClock'
    # do = Analog_Out(
    #     num_channels=do_nchannels,
    #     rate=rate,
    #     daq_type=do_type,
    #     board_name=do_name,
    #     clock_name=do_clock,
    #     verbose=False)
    # ao_type = '6733'
    # ao_name = 'Dev1'
    # ao_nchannels = 8
    # ao = Analog_Out(
    #     num_channels=ao_nchannels,
    #     rate=rate,
    #     daq_type=ao_type,
    #     board_name=ao_name,
    #     verbose=False)

##    ## 6738 test block
##    rate = 2e4
##    do_type = '6738_digital'
##    do_name = 'Dev1'
##    do_nchannels = 2
##    do_clock = '/Dev1/ao/SampleClock'
##    do = Analog_Out(
##        num_channels=do_nchannels,
##        rate=rate,
##        daq_type=do_type,
##        board_name=do_name,
##        clock_name=do_clock,
##        verbose=False)
##    ao_type = '6738'
##    ao_name = 'Dev1'
##    ao_nchannels = 8
##    ao = Analog_Out(
##        num_channels=ao_nchannels,
##        rate=rate,
##        daq_type=ao_type,
##        board_name=ao_name,
##        verbose=False)
##    digits = np.zeros((do.s2p(1), do_nchannels), np.dtype(np.uint8))
##    volts = np.zeros((ao.s2p(1), ao_nchannels), np.dtype(np.float64))
##    digits[do.s2p(.25):do.s2p(.75), :] = 1
##    volts[ao.s2p(.25):ao.s2p(.75), :] = 10
##    do.play_voltages(digits, block=False)
##    ao.play_voltages(volts, block=True)
##    do.close()
##    ao.close()

    ## PXI 6739 test block
    ao_nchannels = 30
    ao = Analog_Out(
        num_channels=ao_nchannels,
        rate=1e5,
        daq_type='6739',
        board_name='PXI1Slot2',
        verbose=True)
    volts = np.zeros((ao.s2p(1), ao_nchannels), 'float64')
    volts[ao.s2p(.25):ao.s2p(.75), :] = 10
    ao.play_voltages(volts, block=True)
    ao.close()

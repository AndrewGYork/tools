import ctypes as C
import numpy as np
"""
Requires nicaiu.dll to be in the same directory, or located in the
os.environ['PATH'] search path.

If you get an error, google for NIDAQmx.h to decypher it.
"""
api = C.cdll.LoadLibrary("nicaiu")

class PCI_6733:
    def __init__(
        self,
        num_channels=8,
        rate=1e5,
        verbose=True,
        ):
        """
        There's a half-decent chance this would work for other NI DAQ
        cards, but I've only tested it for the PCI 6733.
        """
        assert 1 <= num_channels <= 8
        self.num_channels = num_channels
        self.verbose = verbose

        if self.verbose: print("Opening DAQ card...")
        self.task_handle = C.c_uint32(0)
        check(api.create_task(bytes(), self.task_handle))
        """
        If I were a real man, I would automatically detect the proper
        board name somehow. Instead I just hard-code the one ours uses.
        If this next api call crashes for you, check the name of your
        analog-out card using NI Measurement and Automation Explorer (NI
        MAX):
        """
        board_name = 'Dev1'
        device_name = bytes(board_name + "/ao0:%i"%(self.num_channels - 1),
                            'ascii')
        check(api.create_ao_voltage_channel(
            self.task_handle,
            device_name,
            b"",
            -10, #Minimum voltage
            +10.0, #Maximum voltage
            10348, #DAQmx_Val_Volts; don't question it!
            None)) #NULL
        if self.verbose: print(" DAQ card open.")

        self.voltages = np.zeros((2, num_channels), dtype=np.float64)
        self.set_rate(rate)
        self._write_voltages()
        return None

    def set_rate(self, rate):
        self._ensure_task_is_stopped()
        assert 0 < rate <= 1e6 #PCI 6733 can't go faster than 1 MS/s
        self.rate = float(rate)
        check(api.clock_timing(
            self.task_handle,
            None, #NULL, to specify onboard clock for timing
            self.rate,
            10280, #DAQmx_Val_Rising (doesn't matter)
            10178, #DAQmx_Val_FiniteSamps (run once)
            self.voltages.shape[0]))
        if self.verbose:
            print("DAQ card scan rate set to", self.rate, "points per second")
        return None

    def play_voltages(self, voltages=None, force_final_zeros=True):
        """
        If voltage is None, play the previously set voltage.
        """
        self._ensure_task_is_stopped()
        if voltages is not None:
            assert len(voltages.shape) == 2
            assert voltages.dtype == np.float64
            assert voltages.shape[0] >= 2
            assert voltages.shape[1] == self.num_channels
            if force_final_zeros:
                if self.verbose:
                    print("***Coercing DAQ voltages to end in zero!***")
                voltages[-1, :] = 0
            old_voltages_shape = self.voltages.shape
            self.voltages = voltages
            if voltages.shape[0] != old_voltages_shape[0]:
                self.set_rate(self.rate)
            self._write_voltages()
        if self.verbose: print("Playing DAQ voltages...")
        check(api.start_task(self.task_handle))
        self._task_running = True
        return None

    def close(self):
        self._ensure_task_is_stopped()
        if self.verbose: print("Closing DAQ...")
        check(api.clear_task(self.task_handle))
        if self.verbose: print(" DAQ is closed.")
        return None

    def _ensure_task_is_stopped(self):
        if not hasattr(self, '_task_running'):
            self._task_running = False
        if self._task_running:
            if self.verbose: print("Waiting for DAQ to finish playing...")
            check(api.finish_task(self.task_handle, -1))
            if self.verbose: print(" DAQ is finished playing.")
            check(api.stop_task(self.task_handle))
            self._task_running = False
        return None
    
    def _write_voltages(self):
        if not hasattr(self, 'num_points_written'):
            self.num_points_written = C.c_int32(0)
        check(api.write_voltages(
            self.task_handle,
            self.voltages.shape[0], #Samples per channel
            0, #Set autostart to False
            10.0, #Timeout for writing, in seconds. We could be smarter...
            1, #DAQmx_Val_GroupByScanNumber (interleaved)
            self.voltages,
            self.num_points_written,
            None))
        if self.verbose:
            print(self.num_points_written.value,
                  "points written to each DAQ channel.")
        return None

"""
DLL api management, just sets a bunch of argtypes and renames the DLL
functions to a pythonier style.
"""

api.get_error_info = api.DAQmxGetExtendedErrorInfo
api.get_error_info.argtypes = [C.c_char_p, C.c_uint32]

api.create_task = api.DAQmxCreateTask
api.create_task.argtypes = [C.c_char_p, C.POINTER(C.c_uint32)]

api.create_ao_voltage_channel = api.DAQmxCreateAOVoltageChan
api.create_ao_voltage_channel.argtypes = [
    C.c_uint32,
    C.c_char_p,
    C.c_char_p,
    C.c_double,
    C.c_double,
    C.c_int32,
    C.c_char_p]

api.clock_timing = api.DAQmxCfgSampClkTiming
api.clock_timing.argtypes = [
    C.c_uint32,
    C.c_char_p,
    C.c_double,
    C.c_int32,
    C.c_int32,
    C.c_uint64]

api.write_voltages = api.DAQmxWriteAnalogF64
api.write_voltages.argtypes = [
    C.c_uint32,
    C.c_int32,
    C.c_uint32, #NI calls this a 'bool32' haha awesome
    C.c_double,
    C.c_uint32,
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2), #Numpy is awesome.
    C.POINTER(C.c_int32),
    C.POINTER(C.c_uint32)]

api.start_task = api.DAQmxStartTask
api.start_task.argtypes = [C.c_uint32]

api.finish_task = api.DAQmxWaitUntilTaskDone
api.finish_task.argtypes = [C.c_uint32, C.c_double]

api.stop_task = api.DAQmxStopTask
api.stop_task.argtypes = [C.c_uint32]

api.clear_task = api.DAQmxClearTask
api.clear_task.argtypes = [C.c_uint32]

def check(error_code):
    if error_code != 0:
        num_bytes = api.get_error_info(None, 0)
        print("Error message from NI DAQ: (", num_bytes, "bytes )")
        error_buffer = (C.c_char * num_bytes)()
        api.get_error_info(error_buffer, num_bytes)
        print(error_buffer.value.decode('ascii'))
        raise UserWarning(
            "NI DAQ error code: %i; see above for details."%(error_code))

if __name__ == '__main__':
    """
    Test basic functionality of the DAQ object
    """
    daq = PCI_6733(rate=7e5, num_channels=2, verbose=True)
    daq.play_voltages()
    v = np.ones((1000, daq.num_channels))
    v[8:, :] = 0.5
    daq.play_voltages(v)
    daq.verbose=False
    for i in range(100000):
        daq.play_voltages()
    daq.close()

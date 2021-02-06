import time
import serial

class MDT694B_piezo_controller:
    def __init__(self, which_port, verbose=True):
        self.verbose = verbose
        if self.verbose: print("Opening piezo...", end='')
        self.port = serial.Serial(port=which_port, baudrate=115200)
        self.port.write(b'restore\n') # reset controller to default settings
        response = self.port.read(51)
        assert response == (
            b'restore\n*All settings restored to default values.\r*')
        assert self.port.inWaiting() == 0
        if self.verbose: print(" done.")
        self.port.write(b'vlimit?\n') # check voltage limit set point
        response = self.port.read(16)
        assert self.port.inWaiting() == 0
        limit_settings = {b'vlimit?\n*[  75]\r': 75,
                          b'vlimit?\n*[ 100]\r': 100,
                          b'vlimit?\n*[ 150]\r': 150,}
        assert response in limit_settings
        self.voltage_limit_setting = limit_settings[response]
        if self.verbose:
            print("Piezo voltage limit setting:",
                  self.voltage_limit_setting)
        self._pending_cmd = None
        self._settle_time = None
        return None

    def set_voltage(self, voltage, settle_time=None, block=True):
        if self._pending_cmd is not None:
            self._finish_set_voltage()
        voltage = float(voltage)
        assert 0 <= voltage <= self.voltage_limit_setting
        if self.verbose:
            print("Setting piezo voltage to %0.2f volts..."%voltage)
        cmd = ('xvoltage=%0.2f\n'%voltage).encode('ascii')
        self.port.write(cmd)
        self._pending_cmd = cmd
        if settle_time is not None: self._settle_time = float(settle_time)
        if block:
            self._finish_set_voltage()
        return None

    def _finish_set_voltage(self):
        if self._pending_cmd is None:
            return
        response = self.port.read(len(self._pending_cmd) + 1)
        assert response == self._pending_cmd + b'*'
        assert self.port.inWaiting() == 0
        if self._settle_time is not None:
            time.sleep(self._settle_time) # for piezo and mechanics to settle
        self._pending_cmd = None
        if self.verbose: print("Done setting piezo voltage.")
        return None

    def close(self):
        if self.verbose: print("Closing piezo...", end='')
        self.port.close()
        if self.verbose: print(" done.")
        return None

if __name__ == '__main__':
    # init:
    start = time.perf_counter()
    piezo = MDT694B_piezo_controller('COM7', verbose=True)
    print('Initialze time (s):', time.perf_counter() - start)

    piezo.verbose = False
    
    # regular call:
    start = time.perf_counter()
    piezo.set_voltage(0)
    print('Set voltage time (s):', time.perf_counter() - start)

    # non blocking call:
    start = time.perf_counter()
    piezo.set_voltage(0, block=False)
    print('Set voltage non-blocking (s):', time.perf_counter() - start)

    # non blocking call + finish:
    start = time.perf_counter()
    piezo.set_voltage(0, block=False)
    piezo._finish_set_voltage()
    print('Set voltage non-blocking + finish (s):', time.perf_counter() - start)

    # settle time:
    piezo.verbose = True
    for i in range(5):
        piezo.set_voltage(0, settle_time=0.1)
        piezo.set_voltage(5, settle_time=0.5)

    piezo.close()

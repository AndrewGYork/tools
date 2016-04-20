import time
import serial

class MDT694B_piezo_controller:
    def __init__(
        self,
        port_number=2, # COM3 is port 2. Windows/Python mismatch?
        verbose=True
        ):
        self.verbose = verbose
        if self.verbose: print("Opening piezo...", end='')
        self.port = serial.Serial(port=port_number, baudrate=115200)
        # Restore the piezo controller to default settings
        self.port.write(b'restore\n')
        response = self.port.read(51)
        assert response == (
            b'restore\n*All settings restored to default values.\r*')
        assert self.port.inWaiting() == 0
        if self.verbose: print(" done.")
        # Check what the voltage limit of the piezo controller is set to
        self.port.write(b'vlimit?\n')
        response = self.port.read(16)
        assert self.port.inWaiting() == 0
        limit_settings = {b'vlimit?\n*[  75]\r': 75,
                          b'vlimit?\n*[ 100]\r': 100,
                          b'vlimit?\n*[ 150]\r': 150,}
        assert response in limit_settings
        self.voltage_limit_setting = limit_settings[response]
        if self.verbose:
            print(" Piezo voltage limit setting:",
                  self.voltage_limit_setting)
        return None

    def set_voltage(self, v):
        v = float(v)
        assert 0 <= v <= self.voltage_limit_setting
        if self.verbose:
            print("Setting piezo voltage to %0.2f volts..."%v, end='')
        command = ('xvoltage=%0.2f\n'%v).encode('ascii')
        self.port.write(command)
        response = self.port.read(len(command) + 1)
        assert response == command + b'*'
        assert self.port.inWaiting() == 0
        if self.verbose: print(" done.")
        return None

    def close(self):
        if self.verbose: print("Closing piezo...", end='')
        self.port.close()
        if self.verbose: print(" done.")
        return None

if __name__ == '__main__':
    # Simple test code to show that the piezo controller is working
    piezo = MDT694B_piezo_controller()
    for i in range(10):
        piezo.set_voltage(1)
        time.sleep(0.2)
        piezo.set_voltage(2)
        time.sleep(0.2)
    piezo.close()

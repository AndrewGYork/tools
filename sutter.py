import serial

class Lambda_10_3:
    # TODO: test for failure
    def __init__(self, which_port, verbose=False):
        self.verbose = verbose
        if self.verbose: print('Initializing Sutter Lambda_10_3 with' +
                          ' "Wheel A" = 10 position 25mm...', end='')
        try:
            self.port = serial.Serial(port=which_port, timeout=5)
        except serial.serialutil.SerialException:
            raise IOError('Unable to connect to Lambda_10_3 on %s'%which_port)
        self.port.write(b'\xFD') # get controller type and configuration
        response = self.port.readline()
        if response != b'\xfd10-3WA-25WB-NCWC-NCSA-VSSB-VS\r':
            print('Controller response =', response)
            raise IOError("We don't yet support this Lambda_10_3 configuration")
        if self.verbose: print('done.')
        self._pending_cmd = None
        self.move(0)

    def move(self, position, wheel=0, speed=7, block=True):
        if self._pending_cmd is not None:
            self._finish_moving()
        assert position in range(10)
        assert speed in range(8)
        assert wheel == 0
        if self.verbose: print('Moving filter wheel %d'%wheel +
                          ' to position %d'%position +
                          ' with speed %s'%speed +
                          '... ', end='')
        cmd = bytes([(wheel << 7) + (speed << 4) + position])
        self.port.write(cmd)
        self._pending_cmd = cmd
        if block:
            self._finish_moving()

    def _finish_moving(self):
        if self._pending_cmd is None:
            return
        response = self.port.read(2)
        if response != self._pending_cmd + b'\r':
            print('Controller response =', response)
            raise IOError('Unexpected response from Lambda_10_3')
        assert self.port.in_waiting == 0
        self.position = self._pending_cmd[0] & 0b00001111
        self._pending_cmd = None
        if self.verbose: print('done.')

    def close(self):
        if self.verbose: print('Closing Sutter Lambda_10_3 COM port...', end='')
        self.move(0)
        self.port.close()
        if self.verbose: print('done.')
        return None

if __name__ == '__main__':
    import random
    fw = Lambda_10_3(which_port='COM9', verbose=True)
    for i in range(100):
        position = i%10
        fw.move(position, speed=4)
    for i in range(100):
        position = random.randint(0, 9)
        fw.move(position, speed=6)
    fw.verbose = False
    for i in range(100):
        position = i%10
        fw.move(position, speed=4)
    for i in range(100):
        position = random.randint(0, 9)
        fw.move(position, speed=6)
    fw.close()

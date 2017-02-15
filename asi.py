import serial
import time

class FW_1000:
    def __init__(self, which_port, which_wheels=(0,), verbose=True):
        try:
            self.port = serial.Serial(
                port=which_port, baudrate=9600, timeout=10)
        except serial.serialutil.SerialException:
            raise IOError('\n\nUnable to communicate with ASI filter wheel.\n'
                          'Is it powered on?\n'
                          'Is it attached to serial port %s?\n'%which_port + 
                          'Does other software have custody of this port?')
        self.which_wheels, self.active_wheel = which_wheels, which_wheels[0]
        self.positions_per_wheel = [8] * len(self.which_wheels) # 6 someday?!
        if verbose: print('Initializing ASI filter wheel... ', end='')
        self.verbose = False # Clam up 'til we're done with init
        self._reboot()
        for which_wheel in reversed(which_wheels): # Ends on which_wheels[0]
            for i in range(100):
                if self.set_active_wheel(which_wheel) != 'ERR': break
            else:
                raise IOError('Unable to set active ASI filter wheel')
            self.set_sequences([min(i, self.positions_per_wheel[which_wheel]-1)
                                for i in range(8)])
        if verbose: print('done.')
        self.verbose = verbose
        return None
        
    def send(self, cmd):
        if self.verbose: print(' Sent to wheel: %-6s' %cmd, end='') 
        self.port.write(bytes(cmd, 'ascii') + b'\r')
        response = self.port.readline().strip().decode('ascii')
        while response in self.expected_complaints: # What a lousy protocol!
            self.expected_complaints.remove(response)
            if self.verbose and len(response) > 0:
                print('\n  Expected complaint from wheel:', response)
            response = self.port.readline().strip().decode('ascii')
        if not response.startswith(cmd):
            raise IOError(
                'Unexpected response from ASI filter wheel: ' + response)
        response = response[len(cmd):].strip()
        if self.verbose: print(' (Response: ', response, ')', sep='')
        current_state = self.port.read(self.port.in_waiting)
        while not current_state.endswith(b'>'):
            current_state += self.port.read(self.port.in_waiting)
        assert self.active_wheel == int(current_state[-2:-1])
        return response

    def move(self, fw_pos):
        assert fw_pos in range(self.positions_per_wheel[self.active_wheel])
        if self.verbose: print('Moving filter wheel to position', fw_pos)
        self.send('MP %d' % fw_pos)    
        self.finish_moving()
        return None
        
    def finish_moving(self):
        if self.verbose: print(' Finishing motion...', end='')
        while True:
            self.port.write(b'?')
            busy_state = self.port.read(1)
            while busy_state == b'\n': # Ugh, an 'expected complaint'
                complaint = self.port.readline().strip().decode('ascii')
                assert complaint in self.expected_complaints
                self.expected_complaints.remove('')
                self.expected_complaints.remove(complaint)
                if self.verbose:
                    print('\n  Expected complaint from wheel:', complaint)
                busy_state = self.port.read(1)
            if busy_state == b'0':
                break
            if self.verbose: print('.', end='')
        if self.verbose: print(' done.')
        return busy_state

    def set_active_wheel(self, which_wheel):
        assert which_wheel in self.which_wheels
        self.active_wheel = which_wheel
        return self.send('FW %d' % which_wheel)

    def set_sequences(self, sequence):
        assert len(sequence) == 8
        for s in sequence:
            assert s in range(-1, self.positions_per_wheel[self.active_wheel])
        if not hasattr(self, 'sequences'): # Just for initialization
            self.sequences = [[-2] * 8 for i in self.which_wheels]
        for i in range(8):
            if sequence[i] != self.sequences[self.active_wheel][i]:
                self.send('P%d %d'%(i, sequence[i]))
        self.sequences[self.active_wheel] = sequence
        return None
                
    def reset_protocol(self, protocol_pos=0):
        assert 0 <= protocol_pos <= 7
        if self.verbose:
            print('Jumping to entry %d of filter wheel sequence' % protocol_pos)
        self.send('G%d' % protocol_pos)
        return None

    def _reboot(self):
        # Terminate any half-spoken commands, get to a known state ASAP
        self.port.write(b'\r')
        self.port.readline()
        for i in range(1000):
            if self.port.read(self.port.in_waiting).endswith(b'>'): break
        else:
            raise IOError('\n\nUnable to communicate with ASI filter wheel.\n'
                          'Is it powered on?\n'
                          'Is it attached to serial port %s?'%which_port)
        self.expected_complaints = []
        for w in range(2):
            if w not in self.which_wheels:
                self.expected_complaints.append('')
                self.expected_complaints.append('MOTOR %d NOT RESPONDING'%w)
        self.send('F70') # Reset hardware
        self.send('VB 0') # Set hardware verbosity
        return None
        
    def close(self):
        self.port.close()
        return None

if __name__ == '__main__':
    import time
    ## Init FW
    fw = FW_1000(which_port='COM7', which_wheels=(0,))
    for i in range(50):
        fw.move(i%8)
    print(fw.sequences)
    fw.reset_protocol(5)
    fw.finish_moving()
    fw.reset_protocol()
    fw.finish_moving()
    fw.close()


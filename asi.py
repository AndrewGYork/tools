import serial

class FW_1000:
    def __init__(self, which_port, which_wheels=(0,), verbose=True):
        self.which_wheels = which_wheels
        self.verbose = verbose
        if verbose: print('Starting ASI filter wheel initialization...')
        try:
            self.port = serial.Serial(
                port=which_port, baudrate=9600, timeout=10)
        except serial.serialutil.SerialException:
            raise IOError('\n\nUnable to communicate with ASI filter wheel.\n'
                          'Is it powered on?\n'
                          'Is it attached to serial port %s?\n'%which_port + 
                          'Does other software have custody of this port?')
        # It's actually reasonably tricky to get the ASI filter wheel
        # controller into a known state. This is my best effort so far:
        self.port.write(b'F70\r') # Reboot
        for i in range(10): # Wait for a sane response
            if self.port.readline().endswith(b'RESET\r\n'): break
        else:
            raise IOError('Unexpected ASI filter wheel reboot response.')
        self.active_wheel = int(self.port.read(1)) # Partially clear the prompt
        assert self.active_wheel in self.which_wheels
        assert self.port.read(1) == b'>' # Clear the rest of the prompt
        self.port.write(b'VB 6\r') # Set hardware verbosity to 'no prompt'
        assert self.port.readline() == b'VB 6 6\n'
        assert self.port.read(1) == b'\r'
        self.finished_rebooting = False # Might have to wait for complaints!
        return None

    def _finish_init(self):
        if self.finished_rebooting: return None
        if self.verbose:
            print(' Finishing ASI filter wheel initialization... ', end='')
        # The ASI filter wheel controller takes a *while* to notice if
        # one of its wheels is missing. I don't want the init method to
        # forcibly block this whole time. Call this method when you're
        # ready to wait for the wheel to finish its mourning process. If
        # you don't, self.send() will automatically call this method the
        # first time it's called.
        for w in range(2):
            if w not in self.which_wheels: # We expect a complaint
                assert self.port.readline() == b'\n'
                assert self.port.readline() == b'MOTOR %d NOT RESPONDING\r\n'%w
        self.finished_rebooting = True
        # Now that the wheel is done complaining, we can execute a sane
        # serial port protocol without fear of interruption. This means
        # it's time to set default sequence protocols:
        verbose, self.verbose = self.verbose, False # Clam up, temporarily
        for w in self.which_wheels:
            for i in range(100):
                if self.set_active_wheel(w) != 'ERR': break
            else:
                raise IOError('Unable to set active ASI filter wheel')
            self.set_sequences(range(8)) # 8 positions per wheel
        self.verbose = verbose
        if self.verbose: print('done.')
        return None
        
    def send(self, cmd):
        if not self.finished_rebooting: self._finish_init()
        if self.verbose: print(' Sent to wheel: %-6s' %cmd, end='') 
        self.port.write(bytes(cmd, 'ascii') + b'\r')
        r = self.port.readline().strip().decode('ascii')
        assert self.port.read(1) == b'\r'
        assert self.port.in_waiting == 0
        if not r.startswith(cmd):
            raise IOError('Unexpected response from ASI filter wheel: ' + r)
        r = r[len(cmd):].strip()
        if self.verbose: print(' (Response: ', r, ')', sep='')
        return r

    def move(self, pos):
        assert pos in range(8) # 8 positions per wheel
        if self.verbose: print('Moving filter wheel to position', pos)
        self.send('MP %d' % pos)
        self._finish_moving()
        return None
        
    def _finish_moving(self):
        if self.verbose: print(' Finishing motion...', end='')
        while True:
            self.port.write(b'?')
            if self.port.read(1) == b'0': break
            if self.verbose: print('.', end='')
        if self.verbose: print(' done.')
        return None

    def set_speed(self, s):
        """1 is slowest, 9 is fastest. Use s=0 for 'default'"""
        assert s in range(10)
        return self.send('SV %d'%s)

    def set_active_wheel(self, which_wheel):
        assert which_wheel in self.which_wheels
        self.active_wheel = which_wheel
        return self.send('FW %d' % which_wheel)

    def set_sequences(self, sequence):
        assert len(sequence) == 8 # 8 positions per sequence
        for s in sequence:
            assert s in range(-1, 8) # 8 positions per wheel
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
        return self.send('G%d' % protocol_pos)
        
    def close(self):
        self.port.close()
        return None

if __name__ == '__main__':
    for i in range(1000):
        ## Init FW
        fw = FW_1000(which_port='COM7', which_wheels=(0,), verbose=True)
        fw.set_speed(0)
        for i in range(1000):
            fw.move(i%8)
        fw.close()

##    print(fw.sequences)
##    fw.reset_protocol(5)
##    fw._finish_moving()
##    fw.reset_protocol()
##    fw._finish_moving()
    


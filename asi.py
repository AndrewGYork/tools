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

    def move(self, pos):
        assert pos in range(8) # 8 positions per wheel
        if self.verbose: print('Moving filter wheel to position', pos)
        self.send('MP %d' % pos)
        self._finish_moving()
        return None

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

    def set_speed(self, s):
        """1 is slowest, 9 is fastest. Use s=0 for 'default'"""
        assert s in range(10)
        return self.send('SV %d'%s)

    def set_active_wheel(self, which_wheel):
        assert which_wheel in self.which_wheels
        self.active_wheel = which_wheel
        return self.send('FW %d' % which_wheel)

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

    def close(self):
        self.port.close()
        return None
    
    def _finish_moving(self):
        if self.verbose: print(' Finishing motion...', end='')
        while True:
            self.port.write(b'?')
            if self.port.read(1) == b'0': break
            if self.verbose: print('.', end='')
        if self.verbose: print(' done.')
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

class MS2000(object):
    def __init__(self, which_port, verbose=True):
        self.verbose = verbose
        self._orphaned_responses = []
        if self.verbose: print('Initializing MS2000...')
        try:
            self.port = serial.Serial(port=which_port, timeout=1)
        except serial.serialutil.SerialException:
            print('Failed')
            print('Unable to open MS2000 serial port.')
            print('Is it turned on?')
            print('Is it plugged in?')
            print('Is it on the serial port that you expect?')
            raise
        self.send('V') # Say hello, expect a version number in response
        res = self.receive().strip(':A \r\n')
        supported_versions = ('Version: USB-9.2k',)
        assert res in supported_versions, 'Is this an MS2000? ' + \
               "If so, we don't support this firmware version yet."
        self._set_ttl_in_mode('disabled')
        self._set_ttl_out_mode('low')
        assert self._get_ttl_in_mode() == 'disabled'
        assert self._get_ttl_out_mode() == 'low'
        if self.verbose: print('MS2000 done initializing.')
        return None

    def send(self, cmd):
        assert type(cmd) is str, '`command` should be a string'
        if self.verbose: print(" --> Writing to MS2000: '%s\\r'" % cmd)
        self.port.write(bytes(cmd, 'ascii'))
        self.port.write(b'\r')
        return None

    def receive(self, kill_orphans=True):
        ## TODO: look for `:N` and lookup/handle error codes?
        if kill_orphans:
            self._kill_orphans() # empty the port of old, expected responses
        res = self.port.readline().decode('ascii')
        assert res != '', 'No response from MS2000'
        assert res[-1] == '\n', 'MS2000 partial response: %s'%(repr(res))
        if self.verbose: print(' <--  Read from MS2000: %s' % (repr(res)))
        if kill_orphans: # Normal usage
            assert self.port.in_waiting == 0, \
                   ' -|- Unexpected bytes from MS2000: %s' % self.port.read(
                       self.port.in_waiting)
        return res

    def close(self):
        self.port.close()
        return None

    def _kill_orphans(self):
        while len(self._orphaned_responses) > 0:
            expected_response = self._orphaned_responses.pop(0)
            res = self.receive(kill_orphans=False)
            if type(expected_response) == str:
                assert expected_response.rstrip() == res.rstrip(), \
                       'MS2000 expected ' + \
                       'response:"%s", ' % (repr(expected_response)) +\
                       'received:"%s"' % (repr(res))
        return None

    _ttl_out_codes2modes = {'0': 'low', '1': 'high', '9': 'pwm'}
    _ttl_out_modes2codes = {v: k for k, v in _ttl_out_codes2modes.items()}
    _ttl_in_codes2modes = {'0': 'disabled', '10': 'toggle_ttl_out'}
    _ttl_in_modes2codes = {v: k for k, v in _ttl_in_codes2modes.items()}

    def _set_ttl_out_mode(self, mode):
        assert mode in self._ttl_out_modes2codes, \
               'Invalid ttl_out mode(%s)' % mode
        if self.verbose:
            print('MS2000 setting ttl_out to %s-%s' % (
                mode, self._ttl_out_modes2codes[mode]))
        self.send('TTL Y=%s' % self._ttl_out_modes2codes[mode])
        res = self.receive()
        assert res.rstrip() == ':A', 'Unexpected response - %s' % res
        self._ttl_out_mode = mode
        return None

    def _get_ttl_out_mode(self):
        self.send('TTL Y?') # Get ttl_out state
        ttl_out_code = self.receive().rstrip().split('=')[1]
        self._ttl_out_mode = self._ttl_out_codes2modes.get(ttl_out_code)
        if self._ttl_out_mode is None:
            raise NotImplementedError(
                'TTL_OUT state set to %s.' % ttl_out_code +
                'We do not currently support this mode')
        return self._ttl_out_mode

    def _set_ttl_in_mode(self, mode):
        assert mode in self._ttl_in_modes2codes, 'Invalid ttl_in mode (%s)'%mode
        if self.verbose:
            print('MS2000 setting ttl_in to %s-%s' % (
                mode, self._ttl_in_modes2codes[mode]))
        self.send('TTL X=%s' % self._ttl_in_modes2codes[mode])
        res = self.receive()
        assert res.rstrip() == ':A', 'Unexpected response - %s' % res
        self._ttl_in_mode = mode
        return None

    def _get_ttl_in_mode(self):
        self.send('TTL X?') # Get ttl_in state
        ttl_in_code = self.receive().rstrip().split('=')[1]
        self._ttl_in_mode = self._ttl_in_codes2modes.get(ttl_in_code)
        if self._ttl_in_mode is None:
            raise NotImplementedError(
                'TTL_IN state set to %s.' % ttl_in_code +
                'We do not currently support this mode')
        return self._ttl_in_mode


class TransmittedLight(object):
    def __init__(self, ms2000_obj=None, which_port=None, verbose=False):
        self.verbose = verbose
        if ms2000_obj is None:
            self._close_port_on_exit = True
            assert which_port is not None, 'You must either provide a'\
                   'MS2000 object or specify a port name.'
            self.ms2000 = MS2000(which_port=which_port,
                                 verbose=verbose)
        else:
            assert which_port is None, 'MS2000 already has a port!'
            self._close_port_on_exit = False
            self.ms2000 = ms2000_obj
        if self.ms2000._ttl_in_mode == 'toggle_ttl_out': # We own the ttl_in
            self.ms2000._set_ttl_in_mode('disabled')
        self.state = 'off'
        self.ms2000._set_ttl_out_mode('low')
        self.set_pwm_intensity(99) # Set PWM to max; nb we're still `off`
        return None

    def set_state(self, state):
        if state == self.state: return self.state
        assert state in ('off', 'on', 'pwm', 'external')
        if self.verbose: print('Setting TransmittedLight to `%s`' % state)
        ## Clean up if we are leaving external state
        if self.state == 'external':
            self.ms2000._set_ttl_in_mode('disabled')
        ## Set state
        if state == 'off':
            self.ms2000._set_ttl_out_mode('low')
        elif state == 'on':
            self.ms2000._set_ttl_out_mode('high')
        elif state == 'pwm': # TODO: Handle case where PWM mode is unsupported
            self.ms2000._set_ttl_out_mode('pwm')
        elif state == 'external':
            assert self.ms2000._ttl_in_mode == 'disabled'
            self.ms2000._set_ttl_out_mode('low')
            self.ms2000._set_ttl_in_mode('toggle_ttl_out')
        self.state = state
        return None

    def set_pwm_intensity(self, intensity):
        if self.verbose:
            print('TransmittedLight setting LED pwm '
                  'intensity to %d' % int(intensity))
            if self.state != 'pwm':
                print('(TransmittedLight intensity ignored in current state)')
        assert 10 <= intensity <= 99, 'Please provide a '\
               '`pwm_intensity` between 10 and 99'
        self.ms2000.send('LED X=%d' % int(intensity))
        res = self.ms2000.receive()
        assert res.rstrip() == ':A', 'Unexpected response - %s' % res
        return None

    def close(self):
        self.set_state('off')
        if self._close_port_on_exit:
            self.ms.close()
        return None

    def _get_pwm_intensity(self):
        if self.verbose:
            print('TransmittedLight checking pwm intensity')
        self.ms2000.send('LED X?') # Produces a non-standard answer: 'X=val :A'
        self.pwm_intensity = int(
            self.ms2000.receive().strip(':A \r\n').split('=')[1])
        return self.pwm_intensity

class XYZStage(object):
    def __init__(
            self,
            axes=('x', 'y', 'z'),
            ms2000_obj=None,
            which_port=None,
            verbose=False,
            ):
        self.verbose = verbose
        self.axes = {a.lower() for a in axes} # nb this is a set
        assert self.axes <= {'x', 'y', 'z'}, 'Invalid axes name'
        self._moving = {a: False for a in self.axes}
        if ms2000_obj is None:
            self._close_port_on_exit = True
            assert which_port is not None, 'You must either provide a'\
                   'MS2000 object or specify a port name.'
            self.ms2000 = MS2000(which_port=which_port,
                                 verbose=verbose)
        else:
            assert which_port is None, 'MS2000 already has a port!'
            self._close_port_on_exit = False
            self.ms2000 = ms2000_obj
        ## Get limits from hand-calibrated ms2000
        self.ms2000.send('SL '+'? '.join(axes)+'?')
        self.min = {k:v*1000 for k,v in
                    _parse_axes(self.ms2000.receive()).items()}
        self.ms2000.send('SU '+'? '.join(axes)+'?')
        self.max = {k:v*1000 for k,v in
                    _parse_axes(self.ms2000.receive()).items()}
        self.get_position()
        ## 7.5 mm/s max speed for 6.5mm standard pitch lead screws
        self._v_max = {'x': 7.5, 'y': 7.5, 'z': 7.5}
        self.get_velocity()
        self.set_velocity(**{'v'+a:5.8 for a in self.axes})
        self.get_acceleration()
        self._get_settling_time()
        self._get_precision()
        return None

    def move(self, x_um=None, y_um=None, z_um=None, blocking=True):
        assert x_um is not None or y_um is not None or z_um is not None
        if self.verbose: print('Starting XYZStage motion...')
        cmd_string = ['M ']
        for target_um, a in zip((x_um, y_um, z_um), ('x', 'y', 'z')):
            if target_um is None:
                continue
            assert a in self.axes, '%s axis not configured.' % (a.upper())
            if um2asi(self.pos_um[a]) == um2asi(target_um):
                continue
            assert self.min[a] <= target_um <= self.max[a], \
                'Requested %s position (%.1f um) '% (a.upper(), target_um) + \
                'is out of range (%.1f um, %.1f um)'% (self.min[a], self.max[a])
            self.pos_um[a] = asi2um(um2asi(target_um))
            cmd_string.append('%s=%d ' % (a.upper(), um2asi(self.pos_um[a])))
            if self._moving[a]: self.finish_moving()
            self._moving[a] = True
        if len(cmd_string) == 1: # None of the axes need to move
            return None
        self.ms2000.send(''.join(cmd_string))
        if blocking:
            assert self.ms2000.receive().rstrip() == ':A'
            self.finish_moving()
        else:
            self.ms2000._orphaned_responses.append(':A')
        return None

    def finish_moving(self):
        if not any(self._moving.values()):
            return None
        if self.verbose: print("Finishing XYZStage motion...")
        while True:
            self.ms2000.send('/')
            res = self.ms2000.receive()
            if res == 'N\r\n':
                break
        self._moving = {a: False for a in self.axes}
        if self.verbose: print('XYZStage motion complete.')
        return None

    def get_position(self):
        self.ms2000.send('W '+' '.join(self.axes))
        res = [asi2um(int(a)) for a in
               self.ms2000.receive().strip(':A \r\n').split()]
        ## Alphabet sort order matches ASI sort order (X, Y, Z)
        self.pos_um = dict(zip(sorted(list(self.axes)), res))
        return self.pos_um

    def set_velocity(self, vx=None, vy=None, vz=None):
        '''Sets stage velocity in mm/s. 7.5mm/s is max for standard
        6.5mm pitch lead screws.'''
        assert vx is not None or vy is not None or vz is not None
        if self.verbose: print("Setting XYZStage velocity...")
        cmd_string = ['S ']
        for v, a in zip((vx, vy, vz), ('x', 'y', 'z')):
            if v is None:
                continue
            assert a in self.axes, '%s axis not configured.' % (a.upper())
            v = float(v)
            assert 0 < v <= self._v_max[a]
            self.v[a] = v
            cmd_string.append('%s=%0.6f '%(a.upper(), v))
        self.ms2000.send(''.join(cmd_string))
        res = self.ms2000.receive()
        assert res.rstrip() == ':A', 'Unexpected response "%s"'%(repr(res))
        return None

    def get_velocity(self):
        self.ms2000.send('S '+'? '.join(self.axes)+'?')
        self.v = _parse_axes(self.ms2000.receive())
        return self.v

    def set_acceleration(self, ax=None, ay=None, az=None):
        '''Time in ms to reach full velocity.'''
        assert ax is not None or ay is not None or az is not None
        if self.verbose: print("Setting XYZStage acceleration...")
        cmd_string = ['AC ']
        for a_ms, a in zip((ax, ay, az), ('x', 'y', 'z')):
            if a_ms is None:
                continue
            assert a in self.axes, '%s axis not configured' % (a.upper())
            a_ms = round(float(a_ms))
            assert 2 <= a_ms <= 1000, 'Acceleration of of allowed range.'
            self.a[a] = a_ms
            cmd_string.append('%s=%d '% (a.upper(), a_ms))
        self.ms2000.send(''.join(cmd_string))
        res = self.ms2000.receive()
        assert res.rstrip() == ':A', 'Unexpected response "%s"'%(repr(res))
        return None

    def get_acceleration(self):
        self.ms2000.send('AC '+'? '.join(self.axes)+'?')
        self.a = _parse_axes(self.ms2000.receive())
        return self.a

    def close(self):
        if self._close_port_on_exit:
            self.ms.close()
        return None

    def _set_settling_time(self, tx=None, ty=None, tz=None):
        '''Settling time in ms'''
        assert tx is not None or ty is not None or tz is not None
        if self.verbose: print('Setting XYZstage settling time...')
        cmd_string = ['WT ']
        for t_ms, a in zip((tx, ty, tz), ('x', 'y', 'z')):
            if t_ms is None:
                continue
            assert a in self.axes, '%s axis not configured' % (a.upper())
            t_ms = round(float(t_ms))
            cmd_string.append('%s=%d '% (a.upper(), t_ms))
        self.ms2000.send(''.join(cmd_string))
        res = self.ms2000.receive()
        assert res.rstrip() == ':A', 'Unexpected response "%s"'%(repr(res))
        self._get_settling_time()
        return None 
        
    def _get_settling_time(self):
        self.ms2000.send('WT '+'? '.join(self.axes)+'?')
        self.settling_time = _parse_axes(self.ms2000.receive())
        return self.settling_time

    def _set_precision(self, dx=None, dy=None, dz=None):
        '''Settling window in mm'''
        assert dx is not None or dy is not None or dz is not None
        cmd_string = ['PC ']
        for d_mm, a in zip((dx, dy, dz), ('x', 'y', 'z')):
            if d_mm is None:
                continue
            assert a in self.axes, '%s axis not configured' % (a.upper())
            d_mm = float(d_mm)
            cmd_string.append('%s=%.7f '% (a.upper(), d_mm))
        self.ms2000.send(''.join(cmd_string))
        res = self.ms2000.receive()
        assert res.rstrip() == ':A', 'Unexpected response "%s"'%(repr(res))
        self._get_precision()
        return None
    
    def _get_precision(self):
        self.ms2000.send('PC '+'? '.join(self.axes)+'?')
        self.precision = _parse_axes(self.ms2000.receive())
        return self.precision

def _parse_axes(res):
    '''Example input:  `:A X=58.90 Y=37.29 Z=500.03 \r\n`
       Example output: {'y': 37.29, 'x': 58.90, 'z': 500.03}'''
    parameters = {}
    for param in res.strip(' :A\r\n').split():
        axes, value = param.split('=')
        parameters[axes.lower()] = float(value)
    return parameters

def um2asi(microns):
    return round(microns*10)

def asi2um(asi_stage_units):
    return float(asi_stage_units)/10

if __name__ == '__main__':
    import time
    
    test_tl = True
    test_stage = True
    test_fw = False
    
    ## Stage test moves
    if test_stage:
        ms = MS2000(which_port='COM15', verbose=True)
        xyz = XYZStage(ms2000_obj=ms, axes=('X', 'z', 'y'),verbose=True)
        xyz.move(x_um = xyz.pos_um['x']+100,
                 y_um = xyz.pos_um['y']+100, blocking=True)
        xyz.move(x_um = xyz.pos_um['x']-50, blocking=False)
        xyz.move(y_um = xyz.pos_um['y']-50, blocking=True)
        xyz.move(z_um = xyz.pos_um['z']+100, blocking=True)
        xyz.move(z_um = xyz.pos_um['z']-100,
                 x_um = xyz.pos_um['x']-50,
                 y_um = xyz.pos_um['y']-50,
                 blocking=True)
        xyz.finish_moving()
        xyz_x_precision_orig = xyz.precision['x']
        xyz._set_precision(dx=.000025)
        xyz._set_precision(dx=xyz_x_precision_orig)
        xyz_x_settling_time = xyz.settling_time['x']
        xyz._set_settling_time(tx=10)
        xyz._set_settling_time(tx=xyz_x_settling_time)
        xyz.set_acceleration(ax=2)
        xyz.get_acceleration()
        xyz.set_acceleration(ax=100)
        xyz.get_acceleration()
        xyz.close()
        ms.close()
    
    ## TL tests
    if test_tl:
        ms = MS2000(which_port='COM15', verbose=True)
        tl = TransmittedLight(ms2000_obj=ms, verbose=True)
        tl.set_state('on')
        tl.set_state('external')
        tl.set_state('off')
        for pwr in [10, 40, 20, 99]:
            tl.set_state('off')
            tl.set_pwm_intensity(pwr)
            tl.set_state('pwm')
            time.sleep(.1)
        tl.close()
        ms.close()
        
    if test_fw:
        for i in range(1000):
            fw = FW_1000(which_port='COM7', which_wheels=(0,), verbose=True)
            fw.set_speed(0) 
            for i in range(1000):
                fw.move(i%8)
            fw.close()

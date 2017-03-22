import time
import serial
try:
    import numpy as np
except ImportError:
    np = None # You won't be able to use retrieve_data_log()

class C867_XY_Stage:
    def __init__(self, which_port, verbose=True):
        try:
            self.port = serial.Serial(
                port=which_port, baudrate=115200, timeout=1)
        except:
            print("Failed to open serial port", which_port, "for PI stage.",
                  "Is it on, plugged in, and on the serial port you think?")
            raise
        self.verbose = verbose
        self._moving = False
        self._joystick_enabled = True
        # The joystick has a 'startup macro', to make sure it behaves as
        # desired after power switches on. Make sure nobody messed with
        # our startup macro:
        self._set_startup_macro()
        # Get our initial conditions:
        self.get_position()
        self.get_velocity()
        self.get_acceleration()
        # Get position and velocity limits so we can validate user input:
        if self.verbose: print("Getting stage limits...")
        self.x_min, self.y_min = [
            float(a.split('=')[1]) for a in self.send('TMN? 1 2')]
        self.x_max, self.y_max = [
            float(a.split('=')[1]) for a in self.send('TMX? 1 2')]
        self.vx_max, self.vy_max = [
            float(a.split('=')[1]) for a in self.send('SPA? 1 0xA 2 0xA')]
        self.ax_max, self.ay_max = [
            float(a.split('=')[1]) for a in self.send('SPA? 1 0x4A 2 0x4A')]
        self.dx_max, self.dy_max = [
            float(a.split('=')[1]) for a in self.send('SPA? 1 0x4B 2 0x4B')]
        if self.verbose:
            print(" Stage x-limits:", self.x_min, self.x_max)
            print(" Stage y-limits:", self.y_min, self.y_max)
            print(" Stage v-limits:", self.vx_max, self.vy_max)
            print(" Stage a-limits:", self.ax_max, self.ay_max)
            print(" Stage d-limits:", self.dx_max, self.dy_max, '\n')
        return None

    def send(self, cmd, res=True):
        if self.verbose: print(" Sending command to stage:", cmd)
        # Allow cmd to be bytes or string
        if type(cmd) is str: cmd = bytes(cmd, encoding='ascii')
        assert type(cmd) is bytes
        # Communicate:
        self.port.write(cmd + b'\n')
        responses = []
        while res:
            response = self.port.readline()
            assert response.endswith(b'\n') # We timed out
            if self.verbose: print("  Response from stage:", response)
            responses.append(response.rstrip().decode('ascii'))
            # Non-final responses have a trailing space:
            if len(response) == 1: break
            if response[-2] != 32: break
        # Cleanup:
        assert self.port.in_waiting == 0
        self._check_errors()
        return responses

    def move(self, x=None, y=None, blocking=True):
        assert x is not None or y is not None
        self.finish_moving()
        if self.verbose: print("Starting stage motion...")
        if self._joystick_enabled:
            self.send('JON 3 0', res=False)
        cmd_string = ['MOV ']
        if x is not None:
            self.x = float(x)
            assert self.x_min <= self.x <= self.x_max
            cmd_string.append('1 %0.9f '%self.x)
        if y is not None:
            self.y = float(y)
            assert self.y_min <= self.y <= self.y_max
            cmd_string.append('2 %0.9f '%self.y)
        self.send(''.join(cmd_string), res=False)
        self._moving = True
        if blocking:
            self.finish_moving()
        return None

    def finish_moving(self):
        if not self._moving:
            return None
        if self.verbose: print("Finishing stage motion...")
        while True:
            self.port.write(b'\x05')
            response = self.port.read(2)
            if response == b'0\n':
                break
        self._moving = False
        if self._joystick_enabled:
            self.send('JON 3 1', res=False)
        if self.verbose: print('Stage motion complete.\n')
        self._check_errors()
        return None

    def get_position(self):
        if self.verbose: print("Getting stage position...")
        self.x, self.y = [float(a.split('=')[1]) for a in self.send('MOV? 1 2')]
        if self.verbose: print(" Stage position:", self.x, self.y)
        return self.x, self.y
    
    def set_velocity(self, vx=None, vy=None):
        assert vx is not None or vy is not None
        if self.verbose: print("Setting stage velocity...")
        cmd_string = ['VEL ']
        if vx is not None:
            vx = float(vx)
            assert 0 < vx <= self.vx_max
            self.vx = vx
            cmd_string.append('1 %0.9f '%vx)
        if vy is not None:
            vy = float(vy)
            assert 0 < vy <= self.vy_max
            self.vy = vy
            cmd_string.append('2 %0.9f '%vy)
        self.send(''.join(cmd_string), res=False)
        return None

    def get_velocity(self):
        if self.verbose: print("Getting stage velocity...")
        self.vx, self.vy = [float(a.split('=')[1]) for a in self.send('VEL? 1 2')]
        if self.verbose: print(" Stage velocity:", self.vx, self.vy)
        return self.vx, self.vy

    def set_acceleration(self, ax=None, ay=None, dx=None, dy=None):
        assert ax is not None or ay is not None or dx is not None or dy is not None
        if self.verbose: print("Setting stage acceleration...")
        cmd_string = ['ACC ']
        if ax is not None:
            ax = float(ax)
            assert 0 < ax <= self.ax_max
            self.ax = ax
            cmd_string.append('1 %0.9f '%ax)
        if ay is not None:
            ay = float(ay)
            assert 0 < ay <= self.ay_max
            self.ay = ay
            cmd_string.append('2 %0.9f '%ay)
        self.send(''.join(cmd_string), res=False)
        cmd_string = ['DEC ']
        if dx is not None:
            dx = float(dx)
            assert 0 < dx <= self.dx_max
            self.dx = dx
            cmd_string.append('1 %0.9f '%dx)
        if dy is not None:
            dy = float(dy)
            assert 0 < dy <= self.dy_max
            self.dy = dy
            cmd_string.append('2 %0.9f '%dy)
        self.send(''.join(cmd_string), res=False)
        return None

    def get_acceleration(self):
        if self.verbose: print("Getting stage acceleration...")
        self.ax, self.ay = [float(a.split('=')[1]) for a in self.send('ACC? 1 2')]
        self.dx, self.dy = [float(a.split('=')[1]) for a in self.send('DEC? 1 2')]
        if self.verbose: print(" Stage acceleration:", self.ax, self.ay)
        if self.verbose: print(" Stage deceleration:", self.dx, self.dy)
        return self.ax, self.ay, self.dx, self.dy

    def enable_joystick(self, enabled=True):
        if self.verbose: print("Joystick:", enabled)
        if enabled == self._joystick_enabled:
            return None
        self.send(('JON 3 0', 'JON 3 1')[enabled], res=False)
        self._joystick_enabled = enabled
        return None

    def _set_settling_time(self, tx=None, ty=None):
        assert tx is not None or ty is not None
        if self.verbose: print("Setting stage settling time...")
        cmd_string = ['SPA ']
        if tx is not None:
            tx = float(tx)
            assert 0 < tx <= 1 # You wanna wait longer? You crazy
            cmd_string.append('1 0x3F %0.9f '%tx)
        if ty is not None:
            ty = float(ty)
            assert 0 < ty <= 1
            cmd_string.append('2 0x3F %0.9f '%ty)
        self.send(''.join(cmd_string), res=False)
        tx, ty = [float(a.split('=')[1]) for a in self.send('SPA? 1 0x3F 2 0x3F')]
        return tx, ty

    def _set_precision(self, dx=None, dy=None):
        assert dx is not None or dy is not None
        assert not self._moving
        if self.verbose: print("Setting stage precision...")
        # Our 'precision' parameters are bounded by other 'precision' parameters:
        dx_max, dy_max = [float(a.split('=')[1])
                          for a in self.send('SPA? 1 0x416 2 0x416')]
        cmd_string_1 = ['SPA ']
        cmd_string_2 = ['SPA ']
        if dx is not None:
            dx = int(dx)
            assert 1 < dx <= dx_max
            cmd_string_1.append('1 0x407 %d '%dx)
            cmd_string_2.append('1 0x406 %d '%(dx - 1))
        if dy is not None:
            dy = int(dy)
            assert 1 < dy <= dy_max
            cmd_string_1.append('2 0x407 %d '%dy)
            cmd_string_2.append('2 0x406 %d '%(dy - 1))
        # You have to turn off the servo and joystick to change precision:
        if self.verbose: print(' ', end='', sep='')
        self.enable_joystick(False)
        self.send('SVO 1 0 2 0', res=False)
        self.send(''.join(cmd_string_2), res=False)
        self.send(''.join(cmd_string_1), res=False)
        # Turn the servo back on, re-reference the stage, and turn the
        # joystick back on:
        self.send('SVO 1 1 2 1', res=False)
        self.send('FRF', res=False)
        while True: # Finish the reference move
            self.port.write(b'\x05')
            response = self.port.read(2)
            if response == b'0\n':
                break
        if self.verbose: print(' ', end='', sep='')
        self.enable_joystick(True)
        dx, dy = [int(a.split('=')[1])
                  for a in self.send('SPA? 1 0x406 2 0x406')]
        return dx, dy
    
    def _set_startup_macro(self):
        if self.verbose: print("Checking stage STARTUP macro...")
        # Check if the STARTUP macro is set to run on startup:
        if self.send('MAC DEF?')[0] == 'STARTUP':
            # Check if the STARTUP macro is defined
            if 'STARTUP' in self.send('MAC?'):
                # Check if the STARTUP macro is what we expect:
                old_verbose, self.verbose = self.verbose, False #Temp silence
                startup_macro = self.send('MAC? STARTUP')
                self.verbose = old_verbose
                if startup_macro == [
                    'JON 1 0',
                    'SVO 1 1 2 1',
                    'FRF',
                    'WAC ONT? 1 = 1',
                    'WAC ONT? 2 = 1',
                    'JDT 3 1 2',
                    'JDT 3 2 2',
                    'JAX 3 1 1',
                    'JAX 3 2 2',
                    'JON 3 1',
                    'VEL 1 50 2 50']:
                    if self.verbose: print(' Found expected stage STARTUP macro')
                    return None
        if self.verbose: print('Resetting STARTUP macro...')
        # Check if there's a running macro:
        if self.send('RMC?')[0] != '':
            # ...which could be doing all kinds of crazy things; kill it
            # by unsetting the startup macro and rebooting
            self.send('MAC DEF', res=False)
            self._reboot(finish_macro=False)
        # Define our new startup macro:
        self.send(
            'MAC BEG STARTUP\n'
            'JON 1 0\n'
            'SVO 1 1 2 1\n'
            'FRF\n'
            'WAC ONT? 1 = 1\n'
            'WAC ONT? 2 = 1\n'
            'JDT 3 1 2\n'
            'JDT 3 2 2\n'
            'JAX 3 1 1\n'
            'JAX 3 2 2\n'
            'JON 3 1\n'
            'VEL 1 50 2 50\n'
            'MAC END',
            res=False)
        # Set it to run at startup, and reboot again.
        self.send('MAC DEF STARTUP', res=False)
        self._reboot()
        # Wait for our startup macro to finish:
        while self.send('RMC?')[0] == 'STARTUP': time.sleep(0.4)
        return None

    def _reboot(self, finish_macro=True):
        if self.verbose: print('Rebooting stage', end='')
        self.port.write(b'RBT\n')
        time.sleep(0.2) #Give it time to reboot
        self._check_errors()
        if finish_macro:
            self.verbose, old_verbose = False, self.verbose
            while self.send('RMC?')[0] != '':
                print('.', sep='', end='')
                time.sleep(0.3)
            self.verbose = old_verbose
        if self.verbose: print('done')
        return None

    def _check_errors(self):
        self.port.write(b'ERR?\n')
        self.err = self.port.readline()
        if not self.err == b'0\n':
            raise RuntimeError("XY stage error code: "+self.err.decode("ascii"))
        return None

    def close(self):
        self.port.close()

class E753_Z_Piezo:
    def __init__(self, which_port, verbose=True):
        try:
            self.port = serial.Serial(
                port=which_port, baudrate=115200, timeout=1)
        except:
            print("Failed to open serial port", which_port, "for PI Z-piezo.",
                  "Is it on, plugged in, and at the serial port you think?")
            raise
        self.verbose = False # Init quietly, get loud later.
        if verbose: print('Initializing Z-piezo...', end='')
        self.pos_min = float(self.send('TMN?')[0].split('=')[1])
        self.pos_max = float(self.send('TMX?')[0].split('=')[1])
        self.get_target_position()
        self.get_real_position()
        self.set_analog_control_state(False)
        self.analog_offset = float( # We want this to be 50
            self.send('SPA? 2 0x02000200')[0].split('=')[1])
        self.analog_gain = float( # We want this to be 0.5
            self.send('SPA? 2 0x02000300')[0].split('=')[1])
        if (abs(self.analog_offset - 50.0) > 1e-5 or
            abs(self.analog_gain - 0.5) > 1e-5):
            self._set_offset_and_gain()
        self._set_closed_loop_control_parameters(
            p_term=.09, i_term=.000166, notch1_freq=480.0, notch2_freq=520.0,
            notch1_rej=.050, notch2_rej=.050, slew_rate=2.5e6)
        self.set_closed_loop()
        self.verbose = verbose
        if self.verbose:
            print(".done!")
            print(" Z-piezo limits: (", self.pos_min, ', ',
                  self.pos_max, ') microns', sep='')
            print(" Z-piezo target:", self.target_pos, 'microns')
            print(" Z-piezo actual:", self.real_pos, 'microns')
            print(" Z-piezo analog control:", self.analog_control)
        return None
    
    def send(self, cmd, res=True):
        if self.verbose: print(" Sending command to Z-piezo:", cmd)
        # Allow cmd to be bytes or string
        if type(cmd) is str: cmd = bytes(cmd, encoding='ascii')
        assert type(cmd) is bytes
        # Communicate:
        self.port.write(cmd + b'\n')
        responses = []
        while res: # Do we expect a response?
            response = self.port.readline()
            if not response.endswith(b'\n'):
                raise TimeoutError(
                    "No response from PI Z-piezo. Did you expect a response?"
                    "Is the device plugged in? Is it on?"
                    " Is it at the serial port you expect?")
            if self.verbose: print(" Response from Z-piezo:", response)
            responses.append(response.rstrip().decode('ascii'))
            # Non-final responses have a trailing space:
            if len(response) == 1: break #...but length-1 responses don't
            if response[-2] != 32: break
        # Cleanup:
        assert self.port.in_waiting == 0
        self._check_errors()
        return responses

    def _check_errors(self):
        self.port.write(b'ERR?\n')
        self.err = self.port.readline()
        if not self.err == b'0\n':
            raise PIError("Z-piezo error code: ", int(self.err))
        return None

    def get_real_position(self):
        if self.verbose: print("Getting Z-piezo real position...")
        self.real_pos = float(self.send('POS?')[0].split('=')[1])
        if self.verbose: print(" Real Z-piezo position:", self.real_pos)
        return self.real_pos

    def get_target_position(self):
        if self.verbose: print("Getting Z-piezo target position...")
        self.target_pos = float(self.send('MOV?')[0].split('=')[1])
        if self.verbose: print(" Z-piezo target position:", self.target_pos)
        return self.target_pos
    
    def move(self, target):
        """Move the piezo to an absolute position of 'target' microns."""
        assert self.pos_min <= target <= self.pos_max
        self.target_pos = float(target)
        if self.verbose: print('Moving Z-piezo to: %0.3f' % self.target_pos)
        if self.closed_loop:
            self.send('MOV 1 %0.9f'%self.target_pos, res=False)
        else:
            self.send('SVA 1 %0.9f'%self.target_pos, res=False)
        return None

    def _finish_moving(self):
        ## This probably doesn't need to be used because the piezo is quick
        assert self.closed_loop
        if self.verbose: print("Finishing Z-piezo motion...")
        while True:
            self.port.write(b'\x05')
            if self.port.read(2) == b'0\n': break
        if self.verbose: print(' Z-piezo motion complete.')
        self._check_errors()
        return None

    def _set_offset_and_gain(self, offset=50, gain=0.5):
        # We need special permission to write these parameters
        self.send('CCL 1 advanced', res=False)
        self.send('SPA 2 0x02000200 %0.9f'%float(offset), res=False)
        self.send('SPA 2 0x02000300 %0.9f'%float(gain), res=False)
        if self.verbose:
            print('Setting Z-piezo analog offset to %s'%offset)
            print('Setting Z-piezo analog gain to %s'%gain)
        # Return permissions to default
        self.send('CCL 0', res=False)
        self.analog_offset, self.analog_gain = offset, gain
        return None

    def set_closed_loop(self, closed_loop=True):
        if self.verbose: print('Setting closed loop to:', closed_loop)
        if closed_loop:
            self.send('SVO 1 1', res=False)
        else:
            self.send('SVO 1 0', res=False)
        self.closed_loop = (self.send('SVO? 1')[0].split('=')[1] == '1')
        assert self.closed_loop == closed_loop
        if (abs(self.analog_offset - 50.0) > 1e-5 or
            abs(self.analog_gain - 0.5) > 1e-5):
            self._set_offset_and_gain()
        return None
           
    def set_analog_control_state(self, analog_control=True):
        if hasattr(self, 'analog_control'):
            if analog_control == self.analog_control: return None
        if self.verbose: print('Setting Z-piezo analog input:', analog_control)
        self.analog_control = analog_control
        # Requires special permission for writing these parameters
        self.send('CCL 1 advanced', res=False)
        if self.analog_control and not self.closed_loop:
            # Change the analog offset so that the current input voltage
            # will 'target' the current position
            target = float(self.send('SVA?')[0].split('=')[1])
            voltage = float(self.send('TSP? 2')[0].split('=')[1])
            self.analog_offset = self.analog_offset + (target - voltage)
            self.send('SPA 2 0x02000200 %0.9f'%self.analog_offset, res=False)
        if self.analog_control:
            self.send('SPA 1 0x06000500 2', res=False)
        else:
            self.send('WGO 1 0', res=False) # Make sure the wave generator's off
            self.send('SPA 1 0x06000500 0', res=False)
        # Return permissions to default
        self.send('CCL 0', res=False)
        return None

    def record_analog_movement(self, record_types=(2,), t_resolution=1):
        """Prepares tecord the piezo's response to an analog voltage.

        Note that you must trigger the piezo controller recording via a
        TTL high signal on pin 5 of the controller's IO socket, in
        addition to providing an analog control signal to the analog-in
        socket.

        args:
            record_types -- List of ints that correspond to the appropriate
                            codes of record types. Use 'HDR?' command to 
                            retrieve a list of possible codes.
            
            t_resolution -- int between 1 and 10000. Corresponds to the 
                            frequency at which a measurement is recorded. Units
                            are processor cycles (40 microseconds/cycle).

        returns:
            None
        """
        assert 0 < len(record_types) <= 8 # We only have 8 datatables
        assert ''.join([str(i) for i in record_types]).isdigit()
        self.record_types = record_types
        if self.verbose: print('Preparing Z piezo to record movement...')
        self.set_analog_control_state(False) # Disable for setup
        self.send('RTR %d'%t_resolution, res=False) # Set time resolution 
        # Set the number of tables
        self.send('CCL 1 advanced', res=False) # Mother, may I?
        self.send('SPA 1 0x16000300 %d'%len(self.record_types), res=False)
        self.send('CCL 0', res=False) # Return command level to  0
        # Set the record type for each table
        for which_table, record in enumerate(self.record_types):
            if self.verbose:
                print(' Setting z-piezo to record value type',
                      '%d on table %d' % (record, which_table+1))
            self.send('DRC %d 1 %d' % (which_table+1, record), res=False)
        # Set up wave generator, just to trigger data recording.
        # This won't affect piezo movement because we'll be in analog mode.
        self.send('WSL 1 1', res=False) # attach a random wave from wave table
        self.send('WGO 1 2', res=False) # set up wave to run on TTL to I/O port
        self.set_analog_control_state(True) # return to analog control
        if self.verbose: print('Done. Z-piezo is ready to record movement')
        return None
    
    def retrieve_data_log(self, rows, record_types=None, starting_row=1):
        """By default this reads the whole record, which can be sloooow.

        You can use the arguments to only read portions of the record.
        """
        verbose, self.verbose = self.verbose, False # Clam up, this is spammy
        if verbose: 
            print('Retrieving data log from Z-piezo...', end='')
            if rows > 1000: print(' (be patient)', end='')
        if record_types is None:
            tables = range(1, len(self.record_types) + 1)
        else:
            tables = [self.record_types.index(r) + 1 for r in record_types]
        assert 0 < rows * len(self.record_types) < 2**16
        cmd_string = 'DRR? %d %d %s'%(
            starting_row, rows, ' '.join([str(i) for i in tables]))
        data_log = self.send(cmd_string)
        # Parsing of data_table
        data = []
        for i in data_log:
            if not i.startswith("#"):
                data.append([float(m) for m in i.split('\t')])
            else:
                if 'SAMPLE_TIME' in i: ## grab the time interval
                    time_step = float(i.split()[3])
        position_um = np.asarray(data) # Microns (for most but not all outputs!)
        time_s = time_step * np.arange(position_um.shape[0]) # Seconds
        self.verbose = verbose
        if self.verbose: print(' done!')
        return position_um, time_s

    def _set_closed_loop_control_parameters(
        self,
        p_term=None,
        i_term=None,
        notch1_freq=None,
        notch2_freq=None,
        notch1_rej=None,
        notch2_rej=None,
        slew_rate=None
        ):
        """Sets parameters affecting closed-loop control.

           Be careful with these settings, as some combinations can
           cause piezo to oscillate uncontrollably. Don't mess with this
           unless you know what you're doing."""
        ##TODO check inputs
        self.send('CCL 1 advanced', res=False) # Simon says
        if p_term != None:
            self.send('SPA 1 0x07000300 %f' % p_term, res=False)
        if i_term != None:
            self.send('SPA 1 0x07000301 %f' % i_term, res=False)
        if notch1_freq != None:
            self.send('SPA 1 0x08000100 %f' % notch1_freq, res=False)
        if notch2_freq != None:
            self.send('SPA 1 0x08000101 %f' % notch2_freq, res=False)
        if notch1_rej != None:
            self.send('SPA 1 0x08000200 %f' % notch1_rej, res=False)
        if notch2_rej != None:
            self.send('SPA 1 0x08000201 %f' % notch2_rej, res=False) 
        if slew_rate != None:
            self.send('SPA 1 0x07000200 %f' % slew_rate, res=False)
        self.send('CCL 0', res=False)
        return None
                              
    def stop(self):
        try:
            self.send('STP', res=False)
        except PIError as e:
            if e.error_code != 10: raise
        return None
        
    def close(self):
        if self.verbose: print('Z-piezo is shutting down!')
        # Leave the piezo in closed-loop, non-analog control
        self.set_analog_control_state(False)
        self.set_closed_loop()
        self.move(50)
        if self.closed_loop: self._finish_moving()
        self.stop()
        self.port.close()
        return None
        
class PIError(Exception):
    def __init__(self, value, error_code):
        self.value = value
        self.error_code = error_code
    def __str__(self):
        return str(self.value) + str(self.error_code)
    
if __name__ == '__main__':

    ##
    ## RemoteRefocus test code
    ##
    z_piezo = E753_Z_Piezo(which_port = 'COM6', verbose=True)
    ## A few move tests
    z_piezo.move(10)
    z_piezo._finish_moving()
    z_piezo.get_real_position()
    z_piezo.move(50)
    z_piezo._finish_moving()
    z_piezo.close()
    ## 
#     z_piezo.record_analog_movement(record_types=[1, 2],
#                                    t_resolution=10)
#     z_piezo.retrieve_data_log(rows = 300, 
#                               tables = [1, 2])
    
    

    ##
    ## Stage test code
    ##
##    stage = C867_XY_Stage(which_port='COM5', verbose=True)
##    # Clean-ish slate for testing:
##    stage._reboot()
##    # Check how fast we can execute round-trip motions:
##    num_motions = 20
##    motion_size_x = 1
##    motion_size_y = 1
##    print("Testing speed...")
##    
##    # Test conditions for speed test 1:
##    stage.enable_joystick(False)
##    stage._set_settling_time(0.100, 0.100)
##    stage._set_precision(10, 10)
##    stage.set_velocity(120, 120)
##    stage.verbose = False
##    stage.move(0, 0)    
##
##    start = time.perf_counter()
##    for i in range(num_motions):
##        stage.move(0, 0)
##        stage.move(motion_size_x, motion_size_y)
##    end = time.perf_counter()
##    print(end - start, 'seconds')
##    # These conditions should give high-ish speed:
##    stage.enable_joystick(False)
##    stage._set_settling_time(0.001, 0.001)
##    stage._set_precision(10, 10)
##    stage.set_velocity(120, 120)
##    stage.verbose = False
##    stage.move(0, 0)
##    # Check how fast we can execute round-trip motions:
##    print("Testing speed...")
##    start = time.perf_counter()
##    for i in range(num_motions):
##        stage.move(0, 0)
##        stage.move(motion_size_x, motion_size_y)
##    end = time.perf_counter()
##    print(end - start, 'seconds')
##    
##    stage.close()



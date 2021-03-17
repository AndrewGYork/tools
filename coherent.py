import serial

class OBIS:
    def __init__(self,
                 which_port,
                 names_to_channels=None, # Dict: name to SCPI channel
                 operating_mode='CW-power', # String: 'CW-power' or 'AO-power'
                 verbose=True,
                 very_verbose=False):
        self.verbose = verbose
        self.very_verbose = very_verbose
        if self.verbose: print('Initializing OBIS laser:')
        try:
            self.port = serial.Serial(port=which_port, timeout=1)
        except serial.serialutil.SerialException:
            raise IOError('No connection to OBIS on port %s'%which_port)
        # Find devices
        self.max_channels = 6
        self.names_to_channels = {}
        for ch in range(self.max_channels): 
            try:
                self.get_device_identity(ch)
                wavelength = self._send('SYSTem%i:INFormation:WAVelength?'%ch)
                self.names_to_channels[wavelength.split('.')[0]] = ch
            except OSError as e:
                if e.args[0] not in (
                    'Controller error: Device unavailable',
                    'Controller error: Unrecognized command/query'):
                    raise
        # Use nicknames for channels (if passed on init)
        if names_to_channels != None:
            for name, channel in names_to_channels.items():
                assert channel in self.device_identities
            self.names_to_channels = names_to_channels
        self.lasers = tuple(self.names_to_channels.keys())
        self.warning_string = ('\nValid channel numbers: '+
                               str(list(self.names_to_channels)) +
                               '\nValid channel names: '+
                               str(list(self.names_to_channels.values())))
        # Configure lasers
        self.verbose = False # (set True for debugging)
        self.power_setpoint_percent_min = {}
        for laser in self.lasers:
            self._set_CDRH_delay_status(False, laser) # Mandatory for enable
            self._get_device_type(laser) # Determines operating mode options
            self.set_operating_mode(operating_mode, laser) # Also disables laser
            pwr_min = self._get_power_min_watts(laser) # Required attribute
            pwr_max = self._get_power_rating_watts(laser) # Required attribute
            self.power_setpoint_percent_min[laser] = round(
                (100 * pwr_min / pwr_max), 1) + 0.1 # + 0.1 to avoid round down
            # add other attributes as needed
        self.verbose = verbose

    def n2c(self, name_or_channel):
        if name_or_channel in range(self.max_channels):
            return name_or_channel
        elif name_or_channel in self.names_to_channels:
            return self.names_to_channels[name_or_channel]
        elif name_or_channel is None:
            if len(self.names_to_channels) == 1:
                return list(self.names_to_channels.values())[0]
            else:
                raise ValueError('Invalid name_or_channel: %s'%name_or_channel +
                                 self.warning_string)
        else:
            raise ValueError('Invalid name_or_channel: %s'%name_or_channel +
                             self.warning_string)

    def _send(self, cmd, reply=True):
        assert isinstance(cmd, str)
        cmd = bytes(cmd + '\r', 'ascii')
        if self.very_verbose: print('Sending command:', cmd)
        self.port.write(cmd)
        response = None
        if reply:
            response = self.port.readline()
            self._check_error(response)
        handshake = self.port.readline()
        if handshake != b'OK\r\n':
            raise OSError('Unexpected handshake:', self._check_error(handshake))
        assert self.port.in_waiting == 0
        if self.very_verbose:
            print('Response:', response)
            print('Handshake:', handshake)
        if not reply:
            return None
        return response.decode('ascii').strip('\r\n')

    def _check_error(self, response):
        if response[:3] == b'ERR':
            ERR = response.decode('ascii').strip('\r\n')
            error_codes = {
                'ERR-400':'Query unavailable',
                'ERR-350':'Queue overflow',
                'ERR-321':'Out of memory',
                'ERR-310':'System error',
                'ERR-257':'File to open not named',
                'ERR-256':'File does not exist',
                'ERR-241':'Device unavailable',
                'ERR-221':'Settings conflict',
                'ERR-220':'Invalid parameter',
                'ERR-203':'Command protected',
                'ERR-200':'Execution error',
                'ERR-109':'Parameter missing',
                'ERR-102':'Syntax error',
                'ERR-100':'Unrecognized command/query',
                'ERR-000':'No error',
                'ERR-500':'CCB fault',
                'ERR-510':'I2C bus fault',
                'ERR-520':'Controller time out',
                'ERR-900':'CCB message timed out',
                }
            raise OSError('Controller error: ' + error_codes[ERR])
        return None

    def _get_CDRH_delay_status(self, name=None):
        channel = self.n2c(name)
        CDRH_delay_status = self._send('SYSTem%i:CDRH?'%channel)
        CDRH_delay_status = {'ON': True, 'OFF': False}[CDRH_delay_status]
        if not hasattr(self, 'CDRH_delay_status'):
            self.CDRH_delay_status = {}
        self.CDRH_delay_status[name] = CDRH_delay_status
        if self.verbose:
            print('%s CDRH 5 second delay status:'%name,
                  CDRH_delay_status)
        return CDRH_delay_status

    def _set_CDRH_delay_status(self, enable, name=None):
        channel = self.n2c(name)
        cmd = {True: 'ON', False: 'OFF'}[enable]
        if self.very_verbose:
            print('Setting %s CDRH 5 second delay status to'%name, enable)
        self._send('SYSTem%i:CDRH '%channel + cmd, reply=False)
        assert self._get_CDRH_delay_status(name) == enable
        return None

    def _get_power_min_watts(self, name=None):
        channel = self.n2c(name)
        power_min_watts = float(self._send('SOURce%i:POWer:LIMit:LOW?'%channel))
        if not hasattr(self, 'power_min_watts'):
            self.power_min_watts = {}
        self.power_min_watts[name] = power_min_watts
        if self.verbose:
            print('%s power minimum (watts):'%name, power_min_watts)
        return power_min_watts

    def _get_power_rating_watts(self, name=None):
        channel = self.n2c(name)
        power_rating_watts = float(
            self._send('SYSTem%i:INFormation:POWer?'%channel))
        if not hasattr(self, 'power_rating_watts'):
            self.power_rating_watts = {}
        self.power_rating_watts[name] = power_rating_watts
        if self.verbose:
            print('%s power rating (watts):'%name, power_rating_watts)
        return power_rating_watts

    def _get_device_type(self, name=None):
        channel = self.n2c(name)
        device_type = self._send('SYSTem%i:INFormation:TYPe?'%channel)
        if not hasattr(self, 'device_types'):
            self.device_types = {}
        self.device_types[name] = device_type
        if self.verbose:
            print('%s device type:'%name, device_type)
        return device_type
    
    def get_device_identity(self, channel=None): # channel for unknown ID (init)
        channel = self.n2c(channel)
        device_identity = self._send('*IDN%i?'%channel)
        if not hasattr(self, 'device_identities'):
            self.device_identities = {}
        self.device_identities[channel] = device_identity
        if self.verbose:
            print('Ch%s device identity:'%channel, device_identity)
        return device_identity

    def get_wavelength(self, name=None):
        channel = self.n2c(name)
        wavelength = self._send('SYSTem%i:INFormation:WAVelength?'%channel)
        if not hasattr(self, 'wavelengths'):
            self.wavelengths = {}
        self.wavelengths[name] = wavelength
        if self.verbose:
            print('%s wavelength (nm):'%name, wavelength)
        return wavelength

    def get_operating_mode(self, name=None):
        """
        Note: ***these modes depend on the exact OBIS model***
        CWP = continuous wave, constant power
        CWC = continuous wave, constant current
        DIGITAL = CW with external digital modulation
        ANALOG = CW with external analog modulation
        MIXED = CW with external digital + analog modulation
        DIGSO = External digital modulation with power feedback
        MIXSO = External mixed modulation with power feedback
        """
        channel = self.n2c(name)
        operating_mode = self._send('SOURce%i:AM:SOURce?'%channel)
        if operating_mode == 'CWP':
            operating_mode = 'CW-power'
        elif operating_mode in ('MIXED','MIXSO'):
            operating_mode = 'AO-power'
        else:
            raise Exception('Unsupported operating mode %s'%operating_mode)
        if not hasattr(self, 'operating_mode'):
            self.operating_mode = {}
        self.operating_mode[name] = operating_mode
        if self.verbose:
            print('%s operating mode:'%name, operating_mode)
        return operating_mode

    def set_operating_mode(self, mode, name=None):
        channel = self.n2c(name)
        self.set_enabled_status(False, name) # Required
        if self.very_verbose: print('Setting %s mode to %s'%(name, mode))
        if mode == 'CW-power': # power feedback with closed light-loop
            self._send('SOURce%i:AM:INTernal CWP'%channel, reply=False)
        elif mode == 'AO-power': # power feedback with closed light-loop
            if self.device_types[name] == 'DDL':
                self._send('SOURce%i:AM:EXTernal MIXSO'%channel, reply=False)
            elif self.device_types[name] == 'OPSL':
                self._send('SOURce%i:AM:EXTernal MIXed'%channel, reply=False)
        else:
            raise Exception('Unsupported operating mode %s'%mode)
        assert self.get_operating_mode(name) == mode
        return None

    def get_power_setpoint_percent(self, name=None):
        channel = self.n2c(name)
        power_setpoint_watts = float(self._send(
            'SOURce%i:POWer:LEVel:IMMediate:AMPLitude?'%channel))
        if not hasattr(self, 'power_setpoint_watts'):
            self.power_setpoint_watts = {}
        self.power_setpoint_watts[name] = power_setpoint_watts
        power_setpoint_percent = round( # max .dp
            (100 * power_setpoint_watts / self.power_rating_watts[name]), 1)
        if not hasattr(self, 'power_setpoint_percent'):
            self.power_setpoint_percent = {}
        self.power_setpoint_percent[name] = power_setpoint_percent
        if self.verbose:
            print('%s power setpoint: %0.1f%% (%f watts)'%(
                name, power_setpoint_percent, power_setpoint_watts))
        return power_setpoint_percent

    def set_power_setpoint_percent(self, power_setpoint_percent, name=None):
        channel = self.n2c(name)
        if power_setpoint_percent == 'min':
            power_setpoint_percent = self.power_setpoint_percent_min[name]
        else:
            assert isinstance(power_setpoint_percent, (int, float))
            assert 0 <= power_setpoint_percent <= 100
            power_setpoint_percent = round(power_setpoint_percent, 1) # max .dp
            if power_setpoint_percent < (
                self.power_setpoint_percent_min[name]):
                raise Exception('Power setpoint percent %f < minimum (%f)'%(
                    power_setpoint_percent,
                    self.power_setpoint_percent_min[name]))
        power_setpoint_watts = (
            self.power_rating_watts[name] * power_setpoint_percent / 100)
        if self.very_verbose:
            print('Setting %s power setpoint to %0.1f%% (%f watts)'%(
                name, power_setpoint_percent, power_setpoint_watts))
        self._send('SOURce%i:POWer:LEVel:IMMediate:AMPLitude %f'%(
            channel, power_setpoint_watts), reply=False)
        assert self.get_power_setpoint_percent(name) == (
            power_setpoint_percent)
        return None

    def get_enabled_status(self, name=None):
        channel = self.n2c(name)
        enabled_status = self._send('SOURce%i:AM:STATe?'%channel)
        enabled_status = {'ON': True, 'OFF': False}[enabled_status]
        if not hasattr(self, 'enabled_status'):
            self.enabled_status = {}
        self.enabled_status[name] = enabled_status
        if self.verbose:
            print('%s enabled status:'%name, enabled_status)
        return enabled_status
    
    ### Turns laser ON!
    def set_enabled_status(self, enable, name=None):
        channel = self.n2c(name)
        assert self.CDRH_delay_status[name] == False # No 5 second delay!
        cmd = {True: 'ON', False: 'OFF'}[enable]
        if self.very_verbose:
            print('Setting %s enabled status to'%name, enable)
        self._send(('SOURce%i:AM:STATe '%channel) + cmd, reply=False)
        assert self.get_enabled_status(name) == enable
        return None

    def get_instantanous_power_watts(self, name=None, wait=True):
        if wait:
            from time import sleep
            sleep(1) # Allow laser to settle before measurement
        channel = self.n2c(name)
        instantanous_power_watts = float(
            self._send('SOURce%i:POWer:LEVel?'%channel))
        if self.verbose:
            print('%s instantanous power: %f watts'%(
                name, instantanous_power_watts))
        return instantanous_power_watts

    def close(self):
        for laser in self.lasers:
            self.set_enabled_status(False, laser)
        if self.verbose: print('Closing OBIS COM port...', end='')
        self.port.close()
        if self.verbose: print('done.\n')
        return None

if __name__ == '__main__':
    n2c = {'UV-':1, 'Blu':2, 'Grn':3, 'Red':4} # optional nick names
    laser_box = OBIS(which_port='COM4',
                     names_to_channels=n2c,
                     operating_mode='AO-power', # optional init to AO mode
                     verbose=True,
                     very_verbose=False)
    
    # Call single lasers and access attributes:
    laser_box.get_wavelength('Red')
    laser_box.wavelengths['Red']
    
    # Loop over all lasers and test all methods:
    for laser in laser_box.lasers:
        # Atypical methods for main block:
        laser_box._set_CDRH_delay_status(False, laser) # setters call getters
        laser_box._get_power_min_watts(laser)
        laser_box._get_power_rating_watts(laser)
        laser_box._get_device_type(laser)
        laser_box.get_device_identity(laser)
        laser_box.get_wavelength(laser)
        
        # Typical methods/usage:
        laser_box.set_operating_mode('CW-power', laser)
        laser_box.set_power_setpoint_percent('min', laser)
        laser_box.set_enabled_status(True, laser)
        laser_box.set_power_setpoint_percent(5, laser)
        laser_box.get_instantanous_power_watts(laser, wait=False)
        laser_box.get_instantanous_power_watts(laser)
        laser_box.set_enabled_status(False, laser)

    laser_box.close()

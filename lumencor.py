import serial

class SpectraX:
    """Controls Lumencor SpectraX through a serial port."""
    
    def __init__(self, which_port, verbose=True):
        """Constructor for SpectraX class.
        
        Arguments:
            which_port -- Specify which serial port the SpectraX is connected to
                          e.g. 'COM6' (on Windows).
        Keyword arguments:
            verbose -- True | False. Specify if print statements will be
                       executed. This is useful for debug purposes. 
        """
        self.verbose = verbose
        self.port = serial.Serial(port=which_port, baudrate=9600, timeout=.25)
        ## Set up initial states
        self.led_intensities = {'red':0, 'green':0, 'cyan':0, 'uv':0, 'blue':0,
                                 'teal':0}
        self.led_states = {'red':False, 'green':False, 'cyan':False, 'uv':False,
                            'blue':False, 'teal':False, 'yellow_filter':False}
        if self.verbose: print("SpectraX initializing...")
        ## Write initial 'must be issued' commands to the port
        self._send(b'\x57\x02\xff\x50')
        self._send(b'\x57\x03\xAB\x50')
        ## Set initial led intensities to 0 and ensure SpextraX is responding
        self.set_intensity(blocking=False, **self.led_intensities)
        self._force_response()
        if self.verbose: print(" SpectraX is ready!") 
    
    def _send(self, cmd, expected_response=0):
        """Send bytes to the SpectraX serial port.
        
        Arguments:
            cmd -- bytes to write to the serial port
        Keyword Arguments:
            expected_response -- number of bytes expected for a response
                                 defaults to 0 (no response) 
        Returns:
            None if expected_reponse == 0
            bytes if expected_response > 0
        """
        assert type(cmd) is bytes
        assert expected_response >= 0
        assert int(expected_response) == expected_response
        if self.verbose:
            print(' Sending command to SpectraX:',
                  ' '.join('%02x'%i for i in cmd))
        self.port.write(cmd)
        if expected_response > 0:
            response = self.port.read(expected_response)
            if self.verbose:
                print(' Reading response from SpectraX:',
                      ' '.join('%02x'%i for i in response))
            return response
        else:
            return None
            
    def get_temperature(self):
        """Read the temperature of the SpectraX unit. 
        
        Arguments:
            None
        Returns:
            The current temperature of the SpectraX unit in degrees C
        """
        if self.verbose: print("Checking SpectraX temperature...")
        response = self._send(b'\x53\x91\x02\x50', expected_response=2)
        temperature = 0.125 * (int.from_bytes(response, byteorder='big') >> 5)
        if self.verbose: print("SpectraX is %s degrees" % temperature)
        return temperature

    def _force_response(self):
        """Force a response from the SpectraX unit.
        
        The serial port protocol for interacting with the SpectraX unit
        does not provide a response upon receipt of command. This
        function uses the 'check temperature' serial command and will
        block until a response is received.
        
        If no response is received, this will result in an AssertionError.
        
        The logic of calling this function is that if the unit is able to
        repond to this command, it likely has executed all previous commands.
        
        Arguments:
            None
        Returns:
            None
        """
        if self.verbose: print("Checking SpectraX for responsiveness...")
        response = self._send(b'\x53\x91\x02\x50', expected_response=2)
        assert len(response) == 2
        if self.verbose: print (" ... ok to proceed")
        
    def _intensity_setter(self, led, intensity, blocking = True):
        """Set intensity of specific led.
    
        Arguments:
            led -- 'blue' | 'teal' | 'uv' | 'cyan' | 'green' | 'red'. Specify 
                   which led is to be set. Providing a value not listed above
                   will result in a KeyError. 
            intensity -- int between 0 and 255. Specify the intensity of this
                         led. This value has not been tested for linearity. 
                         Providing a value outside of allowable range will 
                         result in AssertionError.
            
        Keyword Arguments:
            blocking -- True | False. Set if this command will call
                        _force_response. If true, this command should block
                        until a response is received. 
        """
        assert 0 <= intensity <= 255
        color = led.lower()
        if self.verbose:
            print("Setting SpectraX %s intensity to %i / 255" % (color, 
                  intensity))
        self._send(({
            'blue': b'\x53\x1a\x03\x01',
            'teal': b'\x53\x1a\x03\x02',
            'uv':   b'\x53\x18\x03\x01',
            'cyan': b'\x53\x18\x03\x02',
            'green':b'\x53\x18\x03\x04',
            'red':  b'\x53\x18\x03\x08'}[color]+
                    (((4095-intensity) << 12)+80).to_bytes(3,byteorder='big')))
        if blocking:
            self._force_response()        
        
    def set_intensity(self, red=None, green=None, cyan=None, uv=None, blue=None,
                      teal=None, blocking=True):
        """Set intensity of leds in SpectraX.
        
        This will sequencially set the intensity for each led where the value
        is not None. There is a way in the serial port protocol to set the same
        intensity to multiple leds with a single command but that is ignored
        here. 
        
        Keyword Arguments:
            red -- int [0..255].
            green -- int [0..255].
            cyan -- int [0..255].
            uv -- int [0..255].
            blue -- int [0..255].
            teal -- int [0..255].
            blocking -- True | False. Check for the unit's ability to respond
                        after setting all intensities. 
        """
        intensities = locals()
        intensities.pop('self')
        intensities.pop('blocking')
        for color , intensity in intensities.items():
            if intensity is not None:
                self._intensity_setter(color, intensity, blocking=blocking)

    def _state_cmd_generator(self):
        """Formats bytes to write to serial port to specifiy led states.
        
        This function was written to clarify the syntax of set_led_state
        function.
        
        Returns:
            bytes -- in a format  to be written to serial port for setting led 
            enabled/disabled states.   
        """
        states_byte = (127
                       -self.led_states['red']*1
                       -self.led_states['green']*2
                       -self.led_states['cyan']*4
                       -self.led_states['uv']*8
                       -self.led_states['yellow_filter']*16
                       -self.led_states['blue']*32
                       -self.led_states['teal']*64)          
        return b'\x4f'+states_byte.to_bytes(1,byteorder='big')+b'\x50'

    def set_led_state(self, red=None, green=None, cyan=None, uv=None,
                      blue=None, teal=None, yellow_filter=None, blocking=True):
        """Enables/disables specific leds of the SpectraX.

        Enabling 'yellow_filter' will shift the spectra of the light from the
        green led.
        
        Keyword Arguments:
            red -- True | False. True for "ON". False for "OFF".
            green -- True | False. True for "ON". False for "OFF".
            cyan -- True | False. True for "ON". False for "OFF".
            uv -- True | False. True for "ON". False for "OFF".
            blue -- True | False. True for "ON". False for "OFF".
            teal -- True | False. True for "ON". False for "OFF".
            yellow_filter -- True | False. True for "IN". False for "OUT".
        """
        states_to_set = locals()
        states_to_set.pop('self')
        states_to_set.pop('blocking')
        ## Default to previous known states if no new state is provided
        for led in states_to_set:
            if states_to_set[led] is not None:
                self.led_states[led] = states_to_set[led]
        if self.verbose:
            print('Setting SpectraX LED states:')
            for led in self.led_states:
                print(' %13s: %s' % (led, self.led_states[led]))
        self._send(self._state_cmd_generator())
        if blocking:
            self._force_response()

    def close(self):
        self.set_led_state(red=False, green=False, cyan=False, uv=False,
                           blue=False, teal=False)
        self.set_intensity(red=0, green=0, cyan=0, uv=0, blue=0, teal=0)
        self.port.close()
            
if __name__ == '__main__':
    import time
    spectrax = SpectraX('COM8', verbose=False)
    print("Done initializing...")
    spectrax.set_intensity(red=255,
                           blocking=True)
    
    spectrax.set_led_state(red=True, blocking=False)
    time.sleep(1)
    spectrax.set_led_state(red=False, blocking=True)

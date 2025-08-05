"""
A device adapter for ThorLabs APT Motion Control devices.
Documentation for the communication protocol can be found here:
https://www.thorlabs.com/Software/Motion%20Control/APT_Communications_Protocol.pdf
"""

import serial # So we can send serial commands over a COM port.

class MFF10x:
    """
    A simple filter flipper from ThorLabs.
    https://www.thorlabs.com/thorproduct.cfm?partnumber=MFF102/M
    """
    def __init__(self, which_port, verbose=True):
        """
        Try to open a COM port with the filter flipper, make sure
        it's hardware/firmware that we support, and then get its current
        position.
        """
        self.verbose = verbose # Prints statements about initialization/moving.
        self.which_port = which_port
        if self.verbose:
            print('Opening ThorLabs MFF10x filter flipper on %s...'%(
                  self.which_port),
                  end='')
        try:
            self.port = serial.Serial(port=which_port,
                                      baudrate=115200,
                                      timeout=5)
        except serial.serialutil.SerialException:
            raise IOError('Unable to connect to MFF10x on %s'%which_port)
        self.refresh_attributes()
        self._finish_moving()
        if self.verbose:
            print(' done.')
        return None
            
    def refresh_attributes(self):
        """
        Ask the device in the specified COM port what it is. Check that
        it's a MFF10x filter wheel.
        """
        # Call MGMSG_HW_REQ_INFO:
        self.port.write(b'\x05\x00\x00\x00\x50\x01')
        response = self.port.read(90)
        assert self.port.in_waiting == 0 # Make sure we read everything.
        header = response[0:6]
        assert header == b'\x06\x00\x54\x00\x81\x50'
        serial_number = int.from_bytes(response[6:10], byteorder='little')
        assert len(str(serial_number)) == 8
        self.serial_number = serial_number
        
        model_number = response[10:18].decode('ascii').rstrip('\x00')
        assert model_number == 'MFF002' # Different from printed model number.
        self.model_number = model_number
        hardware_type = int.from_bytes(response[18:20],
                                       byteorder='little',
                                       signed=False)
        assert hardware_type == 16
        # I  a response of 16, but I expected a response of 44.
        self.hardware_type = hardware_type
        firmware_version_minor_revision =   str(response[20])
        firmware_version_interim_revision = str(response[21])
        firmware_version_major_revision =   str(response[22])
        firmware_version = '.'.join([firmware_version_major_revision,
                                     firmware_version_interim_revision,
                                     firmware_version_minor_revision])

        assert firmware_version == '1.0.3'
        self.firmware_version = firmware_version
        hardware_version = int.from_bytes(response[84:86], byteorder='little')
        assert hardware_version == 2 # I can't verify this is true.
        self.hardware_version = hardware_version
        mod_state = int.from_bytes(response[86:88])
        assert mod_state == 0 # I don't know what this is.
        self.mod_state = mod_state
        num_channels = int.from_bytes(response[88:90])
        assert num_channels == 256 # I don't know what this is.
        self.num_channels = num_channels
        return None

    def move(self, requested_position, finish_moving=True):
        """
        Send a request to the filter flipper to change its position.
        Check that the motor is finished moving.
        """
        if self.verbose:
            print('Moving filter flipper to position %d...'%(
                  requested_position),
                  end='')
        assert requested_position in (1, 2)
        requested_position_to_bits = {1: b'\x6A\x04\x00\x01\x50\x01',
                                      2: b'\x6A\x04\x00\x02\x50\x01'}
        # Call MGMSG_MOT_MOVE_JOG:
        self.port.write(requested_position_to_bits[requested_position])
        if finish_moving:
            self._finish_moving()
        if self.verbose:
            print(' done.')
        return None

    def _refresh_status(self):
        """
        Ask the filter flipper to return its status in raw bits. Use
        them to populate attributes.
        """
        # Call MGMSG_MOT_REQ_STATUSBITS:
        self.port.write(b'\x29\x04\x00\x00\x50\x01')
        response = self.port.read(12)
        assert self.port.in_waiting == 0
        header = response[0:6]
        assert header == b'\x2A\x04\x06\x00\x81\x50'
        chan_ident = int.from_bytes(response[6:8],
                                    byteorder='little',
                                    signed=False)
        assert chan_ident == 1 # There is only one motor in this flipper.
        status_bits = int.from_bytes(response[8:12],
                                     byteorder='little',
                                     signed=False)
        
        # Check if the motor is either at the CW or CCW limit.
        clockwise_limit = (
            status_bits & 0b00000000000000000000000000000001 != 0)
        counterclockwise_limit = (
            status_bits & 0b00000000000000000000000000000010 != 0)
        if clockwise_limit          and not counterclockwise_limit:
            self.position = 1
        elif counterclockwise_limit and not clockwise_limit:
            self.position = 2
        else:
            self.position = None # Weird but not impossible.

        # Check if the motor is in motion. Although the motor
        # physically moves both clockwise and counterclockwise, the
        # counterclockwise motion bit never appears to be True. In both
        # motion cases, only the clockwise motion appears True. Still,
        # the code should probably check for if the CCW motion is ever
        # true.
        in_motion_clockwise = (
            status_bits & 0b00000000000000000000000000010000 != 0) 
        in_motion_counterclockwise = (
            status_bits & 0b00000000000000000000000000100000 != 0)
        if in_motion_clockwise or in_motion_counterclockwise:
            self.moving = True
        else:
            self.moving = False
        return None

    def _finish_moving(self, max_polls=200):
        """
        A function to call to block the script from progressing until
        the filter flipper reports that it's finished moving. If it
        doesn't report that it's done moving after two seconds, raise a
        timeout exception.
        """
        for i in range(max_polls): # 200 polls is ~2 seconds.
            self._refresh_status()
            if not self.moving:
                break
        else:
            raise TimeoutError('ThorLabs MFF10x on %s failed to finish moving.'%
                               (self.which_port))
        return None

    def close(self):
        """
        Close the COM port.
        """
        if self.verbose:
            print("Closing ThorLabs MFF10x filter flipper on %s..."%(
                                self.which_port),
                                end='')
        self.port.close()
        if self.verbose: print(' done.')
        return None

if __name__ == '__main__':
    """
    Test block. Open a COM port, move the filter to position 1, then
    back to position 2 as soon as possible. Then close the COM port.
    """
    filter_flipper = MFF10x(which_port='COM4')
    filter_flipper.move(requested_position=1)
    filter_flipper.move(requested_position=2)
    filter_flipper.close()

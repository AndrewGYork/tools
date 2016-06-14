import time
import serial

class Stage:
    """Zaber stage(s), attached through the (USB?) serial port."""
    def __init__(
        self,
        port_name, # For example, 'COM3' on Windows
        timeout=1,
        verbose=True,
        very_verbose=False):
        """port_name: which serial port the stage is connected to, e.g. 'COM3'
        """
        self.verbose = verbose
        self.very_verbose = very_verbose
        try:
            self.serial = serial.Serial(
            port=port_name,
            baudrate=9600,
            bytesize=8,
            parity='N',
            stopbits=1,
            timeout=timeout)
        except serial.serialutil.SerialException:
            print('Failed to open serial port for Zaber stage(s).')
            print('Sometimes Windows is weird about this!')
            print('Consider trying again.')
            raise
        if self.verbose: print("Renumbering stages:")
        self.devices = self.renumber_all_devices()
        self.pending_moves = [False for d in self.devices]
        if self.verbose:
            for d in self.devices:
                print(' Axis:', d)
            print(' Done renumbering.')
        self.restore_settings()
        self.default_speed = min([r['speed'] for r in self.get_target_speed()])
        if verbose: print(" Default stage speed:", self.default_speed)
        self.move_home()

    def send(self, instruction):
        """Send an instruction to the Zaber stage.

        'instruction' must be a list of 6 integers, 0-255 (no error
        checking).
        See: http://www.zaber.com/wiki/Manuals/Binary_Protocol_Manual
        for a list of instructions.
        """
        assert len(instruction) == 6
        if self.very_verbose: print("Sending to stage:", instruction)
        serial_cmd = bytes(instruction) # 0 <= int(i) < 256 for i in instruction
        self.serial.write(serial_cmd)
        return None

    def receive(self, expected_command_ID=None):
        """Return 6 bytes from the serial port

        There must be 6 bytes to receive (no error checking).
        """
        response = self.serial.read(6)
        if len(response) != 6:
            raise UserWarning(
                "Zaber stage failed to respond. Is the timeout too short?\n" +
                "Is the stage plugged in?")
        response = {'device_number': response[0],
                    'command_ID': response[1],
                    'data': four_bytes_to_uint(response[2:6])}
        if expected_command_ID is not None:
            assert response['command_ID'] == expected_command_ID
        if self.very_verbose:
            print("Response from stage:\n", response)
        return response

    def get_position(self, axis='all'):
        if axis == 'all':
            axis = 0
            num_responses = len(self.devices)
        else:
            num_responses = 1
        assert axis in range(len(self.devices) + 1)
        self.send([axis, 60, 0, 0, 0, 0])
        responses = []
        for i in range(num_responses):
            responses.append(self.receive(expected_command_ID=60))
        axis_positions = {}
        for r in responses:
            axis_positions[r['device_number']] = r['data']
        return axis_positions

    def move(self, distance, movetype='absolute', response=True, axis='all'):
        distance = int(distance)
        if self.verbose:
            print("Moving axis: ", repr(axis),
                  " distance ", distance, " (", movetype, ")", sep='')
        if axis == 'all':
            axis = 0
            assert self.pending_moves == [False for d in self.devices]
        else:
            assert axis in [d['device_number'] for d in self.devices]
            assert self.pending_moves[(axis - 1)] == False
        if movetype == 'absolute':
            instruction = [axis, 20]
        elif movetype == 'relative':
            instruction = [axis, 21]
        else:
            raise UserWarning("Move type must be 'relative' or 'absolute'")
        # Data conversion and transfer:
        instruction.extend(uint_to_four_bytes(distance))
        self.send(instruction)
        if axis == 0:
            self.pending_moves = [True for d in self.devices]
        else:
            self.pending_moves[axis - 1] = True
        if response:
            return self.finish_moving()
        return None

    def finish_moving(self):
        response = []
        for i in range(len(self.devices)):
            if self.pending_moves[i]:
                response.append(self.receive())
                assert response[-1]['command_ID'] in (1, 20, 21)
        self.pending_moves = [False for d in self.devices]
        assert self.serial.inWaiting() == 0
        return response

    def move_home(self, response=True):
        if self.verbose: print("Moving stage(s) near home...")
        self.move(100)
        if self.verbose: print("Moving stage(s) home.")
        assert self.pending_moves == [False for d in self.devices]
        self.send([0, 1, 0, 0, 0, 0])
        self.pending_moves = [True for d in self.devices]
        if response:
            return self.finish_moving()
        return None

    def restore_settings(self):
        if self.verbose: print("Restoring stage(s) to default settings.")
        assert self.pending_moves == [False for d in self.devices]
        assert self.serial.inWaiting() == 0
        self.send([0, 36, 0, 0, 0, 0]) # Restore to default settings
        for d in self.devices:
            self.receive(expected_command_ID=36)
        self.send([0, 116, 1, 0, 0, 0]) # Disable manual move tracking
        for d in self.devices:
            self.receive(expected_command_ID=116)
        assert self.serial.inWaiting() == 0
        return None
            
    def renumber_all_devices(self):
        self.serial.flushInput()
        self.serial.flushOutput()
        self.send([0, 2, 0, 0, 0, 0])
        # We can't predict the length of the response, since we don't
        # yet know how many stages there are. Just wait a healthy amount
        # of time for the answer:
        time.sleep(.8) # Seems to take a little over 0.5 seconds.
        bytes_waiting = self.serial.inWaiting()
        assert bytes_waiting % 6 == 0 # Each stage responds with 6 bytes.
        num_stages = int(bytes_waiting / 6)
        stages = []
        for n in range(num_stages):
            r = self.receive()
            assert (r['device_number'] - 1) in range(num_stages)
            assert r.pop('command_ID') == 2
            r['device_ID'] = r.pop('data')
            assert r['device_ID'] in (# List of devices we've tested; add liberally.
                20053,
                )
            stages.append(r)
        assert self.serial.inWaiting() == 0
        return stages        
        
    def set_target_speed(self, speed, response=True):
        min_speed = int(self.default_speed * 0.01)
        max_speed = int(2*self.default_speed)
        speed = int(speed)
        assert min_speed <= speed < max_speed
        if self.verbose: print("Setting stage speed to", speed)
        inst = [0, 42]
        inst.extend(uint_to_four_bytes(speed))
        self.send(inst)
        if response:
            reply = [self.receive(expected_command_ID=42)
                     for d in self.devices]
            return reply

    def get_target_speed(self):
        inst = [0, 53, 42, 0, 0, 0]
        self.send(inst)
        reply = []
        for d in self.devices:
            reply.append(self.receive())
            assert reply[-1].pop('command_ID') == 42
            reply[-1]['speed'] = reply[-1].pop('data')
        return reply

    def close(self):
        self.move_home()
        self.serial.close()

def four_bytes_to_uint(x):
    assert len(x) == 4
    return int.from_bytes(x, byteorder='little')

def uint_to_four_bytes(x):
    assert 0 <= x < 4294967296
    return [x >> i & 0xff for i in (0, 8, 16, 24)]

if __name__ == '__main__':
    my_stage = Stage(port_name='COM3', verbose=True, very_verbose=False)
    try:
        my_stage.move(0, movetype='absolute', axis='all')
        for i in range(len(my_stage.devices)):
            my_stage.move(70000, movetype='absolute', axis=i+1)
            print("Stage postion:", my_stage.get_position())
            my_stage.move(0, movetype='absolute', axis=i+1)
            print("Stage postion:", my_stage.get_position())
        my_stage.set_target_speed(my_stage.default_speed * 1.3)
        my_stage.move(70000, movetype='absolute', axis='all')
        print("Stage postion:", my_stage.get_position())
        my_stage.move(0, movetype='absolute', axis='all')
        print("Stage postion:", my_stage.get_position())
        my_stage.set_target_speed(my_stage.default_speed)
    finally:
        my_stage.close()

import time
import ctypes as C
import numpy as np

class Edge:
    def __init__(self, pco_edge_type='4.2', verbose=True, very_verbose=False):
        assert pco_edge_type in ('4.2', '5.5')
        self.pco_edge_type = pco_edge_type
        self.verbose = verbose
        self.very_verbose = very_verbose
        self.camera_handle = C.c_void_p(0)
        if verbose: print("Opening pco.edge camera...")
        try:
            assert self.camera_handle.value is None
            dll.open_camera(self.camera_handle, 0)
            assert self.camera_handle.value is not None
        except (WindowsError, AssertionError):
            print("Failed to open pco.edge camera.")
            print(" *Is the camera on, and plugged into the computer?")
            print(" *Is CamWare running?")
            print(" *Is sc2_cl_me4.dll in the same directory as SC2_Cam.dll?")
            raise
        if self.verbose: print(" Camera open.")
        self.disarm()
        self._refresh_camera_setting_attributes()
        return None                            

    def close(self):
        if self.armed: self.disarm()
        if self.verbose: print("Closing pco.edge camera...")
        dll.close_camera(self.camera_handle)
        if self.verbose: print(" Camera closed.")
        return None

    def apply_settings(
        self,
        trigger='auto_trigger',
        exposure_time_microseconds=2200,
        region_of_interest={'left': 1,
                            'right': 2060,
                            'top': 1,
                            'bottom': 2048}
        ):
        """
        * 'trigger' can be 'auto_trigger' or 'external_trigger' See the
          comment block in _get_trigger_mode() for further details.
        * 'exposure_time_microseconds' can be as low as 107 and as high
          as 1000000.
        * 'region_of_interest' will be adjusted to match the nearest
          legal ROI that the camera supports. See _legalize_roi() for
          details.
        """    
        if self.armed: self.disarm()
        if self.verbose: print("Applying settings to camera...")
        """
        These settings matter, but we don't expose their functionality
        through apply_settings():
        """
        dll.reset_settings_to_default(self.camera_handle)        
        self._set_sensor_format('standard')
        self._set_acquire_mode('auto')
        self._set_pixel_rate({'4.2': 272250000,
                              '5.5': 286000000
                              }[self.pco_edge_type])
        """
        I think these settings don't matter for the pco.edge, but just
        in case...
        """
        self._set_storage_mode('recorder')
        self._set_recorder_submode('ring_buffer')
        """
        These settings change all the time:
        """
        self._set_trigger_mode(trigger)
        self._set_exposure_time(exposure_time_microseconds)
        self._set_roi(region_of_interest)
        """
        It's good to check the camera health periodically. Now's as good
        a time as any, especially since the expected result is
        predictable: it should all be zeros.
        """
        camera_health = self._get_camera_health()
        for v in camera_health.values():
            assert v == 0
        return None

    def arm(self, num_buffers=2):
        assert 1 <= num_buffers <= 16
        if self.armed:
            if self.verbose:
                print('Arm requested, but the pco.edge camera'
                      'is already armed. Disarming...')
                self.disarm()
        if self.verbose: print("Arming camera...") 
        dll.arm_camera(self.camera_handle)
        wXRes, wYRes, wXResMax, wYResMax = (
            C.c_uint16(), C.c_uint16(), C.c_uint16(), C.c_uint16())
        dll.get_sizes(self.camera_handle, wXRes, wYRes, wXResMax, wYResMax)
        self.width, self.height = wXRes.value, wYRes.value
        self.bytes_per_image = self.width * self.height * 2 #16 bit images
        if self.very_verbose:
            print(" Camera ROI dimensions:",
                  self.width, "(l/r) by", self.height, "(u/d)")
        """
        Allocate buffers that the camera will use to hold images.
        """
        self.buffer_pointers = []
        for i in range(num_buffers):
            buffer_number = C.c_int16(-1)
            self.buffer_pointers.append(C.POINTER(C.c_uint16)()) #Woo!
            buffer_event = C.c_void_p(0)
            dll.allocate_buffer(
                self.camera_handle,
                buffer_number,
                self.bytes_per_image,
                self.buffer_pointers[-1],
                buffer_event)
            assert buffer_number.value == i
            if self.very_verbose:
                print(" Buffer number ", i, " allocated, pointing to ",
                      self.buffer_pointers[-1].contents,
                      ", linked to event ", buffer_event.value, sep='')
        dll.set_image_parameters(self.camera_handle, self.width, self.height)
        dll.set_recording_state(self.camera_handle, 1)
        self.armed = True
        if self.verbose: print(" Camera armed.")
        """
        Add our allocated buffers to the camera's 'driver queue'
        """
        self.added_buffers = []
        for buf_num in range(len(self.buffer_pointers)):
            dll.add_buffer(
                self.camera_handle,
                0,
                0,
                buf_num,
                self.width,
                self.height,
                16)
            self.added_buffers.append(buf_num)
        self._dll_status = C.c_uint32()
        self._driver_status = C.c_uint32()
        self._image_datatype = C.c_uint16 * self.width * self.height
        return None

    def disarm(self):
        if self.verbose: print("Disarming camera...")
        dll.set_recording_state(self.camera_handle, 0)
        dll.remove_buffer(self.camera_handle)
        if hasattr(self, 'buffer_pointers'): #free allocated buffers
            for buf in range(len(self.buffer_pointers)):
                dll.free_buffer(self.camera_handle, buf)
            self.buffer_pointers = []
        self.armed = False
        if self.verbose: print(" Camera disarmed.")
        return None

    def record_to_memory(
        self,
        num_images,
        preframes=0,
        out=None,
        first_frame=0,
        poll_timeout=5e5,
        sleep_timeout=40,
        first_trigger_timeout_seconds=0,
        ):
        if not self.armed: self.arm()
        """
        We'll store our images in a numpy array. Did the user provide
        one, or should we allocate one ourselves?
        """
        if out is None:
            first_frame = 0
            out = np.ones((num_images - preframes, self.height, self.width),
                          dtype=np.uint16)
        try:
            assert len(out.shape) == 3
            assert (out.shape[0] - first_frame) >= (num_images - preframes)
            assert (out.shape[1], out.shape[2]) == (self.height, self.width)
            assert out.dtype == np.uint16
        except AssertionError:
            print("\nInput argument 'out' must have dimensions:")
            print("(>=num_images - preframes, y-resolution, x-resolution)")
            raise
        except AttributeError:
            print("\nInput argument 'out' must be a numpy array",
                  "(to hold our images)")
            raise
        """
        Try to record some images, and try to tolerate the many possible
        ways this can fail.
        """
        if self.verbose: print("Acquiring", num_images, "images...")
        num_acquired = 0
        for which_im in range(num_images):
            """
            Hassle the camera until it gives us a buffer. The only ways
            we exit this 'while' loop are by getting a buffer or running
            out of patience.
            """
            num_polls = 0
            num_sleeps = 0
            start_time = time.perf_counter()
            while True:
                """
                Check if a buffer is ready
                """
                num_polls += 1
                dll.get_buffer_status(
                    self.camera_handle,
                    self.added_buffers[0],
                    self._dll_status,
                    self._driver_status)
                if self._dll_status.value == 0xc0008000:
                    buffer_number = self.added_buffers.pop(0)#Removed from queue
                    if self.very_verbose:
                        print(" After", num_polls, "polls and", num_sleeps,
                              "sleeps, buffer", buffer_number, "is ready.")
                    break
                """
                The buffer isn't ready. How long should we wait to try
                again? For short exposures, we'd like to poll super
                frequently. For long exposures, we'll use time.sleep()
                to save CPU.
                """
                if self.exposure_time_microseconds > 30e3:
                    time.sleep(self.exposure_time_microseconds * 1e-6 * #seconds
                               2 / sleep_timeout) #Worst case
                    num_sleeps += 1
                """
                At some point we have to admit we probably missed a
                trigger, and give up. Give up after too many polls
                (likely triggered by short exposures) or too many sleeps
                (likely triggered by long exposures)
                """
                if num_polls > poll_timeout or num_sleeps > sleep_timeout:
                    elapsed_time = time.perf_counter() - start_time
                    if which_im == 0: # First image; maybe keep waiting...
                        if elapsed_time < first_trigger_timeout_seconds:
                            continue
                    raise TimeoutError(
                        "After %i polls,"%(num_polls) + 
                        " %i sleeps"%(num_sleeps) + 
                        " and %0.3f seconds,"%(elapsed_time) + 
                        " no buffer. (%i acquired)"%(num_acquired),
                        num_acquired=num_acquired)
            try:
                if self._driver_status.value == 0x0:
                    pass
                elif self._driver_status.value == 0x80332028:
                    # Zero the rest of the buffer
                    out[max(0, first_frame + (which_im - preframes)):, :, :].fill(0)
                    raise DMAError('DMA error during record_to_memory')
                else:
                    print("Driver status:", self._driver_status.value)
                    raise UserWarning("Buffer status error")
                if self.very_verbose:
                    print(" Record to memory result:",
                          hex(self._dll_status.value),
                          hex(self._driver_status.value))
                if which_im >= preframes:
                    """
                    http://stackoverflow.com/a/13481676
                    """
                    image = np.ctypeslib.as_array( #Temporary!
                        self._image_datatype.from_address(
                            C.addressof(
                                self.buffer_pointers[buffer_number].contents)))
                    out[first_frame + (which_im - preframes), :, :] = image
                    num_acquired += 1
            finally:
                dll.add_buffer(#Put the buffer back in the driver queue
                    self.camera_handle,
                    0,
                    0,
                    buffer_number,
                    self.width,
                    self.height,
                    16)
                self.added_buffers.append(buffer_number)
        if self.verbose: print("Done acquiring.")
        return out

    def _refresh_camera_setting_attributes(self):
        """
        There are two ways to access a camera setting:
        
         1. Ask the camera directly, using a self.get_*() - type method.
        
          This interrogates the camera via a DLL call, updates the
          relevant attribute(s) of the Edge object, and returns the
          relevant value(s). This is slower, because you have to wait for
          round-trip communication, but gets you up-to-date info.

         2. Access an attribute of the Edge object, e.g. self.roi

          This ignores the camera, which is very fast, but the resulting
          value could potentially be inconsistent with the camera's true
          setting (although I hope it isn't!)

        _refresh_camera_setting_attributes() is a convenience function
        to update all the camera attributes at once. Call it if you're
        nervous, I guess.
        """
        if self.verbose: print("Retrieving settings from camera...")
        self._get_sensor_format()
        self._get_trigger_mode()
        self._get_storage_mode()
        self._get_recorder_submode()
        self._get_acquire_mode()
        self._get_pixel_rate()
        self._get_exposure_time()
        self._get_roi()
        self._get_temperature()
        self._get_camera_health()
        return None

    def _get_sensor_format(self):
        wSensor = C.c_uint16(777) #777 is not an expected output
        dll.get_sensor_format(self.camera_handle, wSensor)
        assert wSensor.value in (0, 1) #wSensor.value should change
        mode_names = {0: "standard", 1: "extended"}
        if self.very_verbose:
            print(" Sensor format:", mode_names[wSensor.value])
        self.sensor_format = mode_names[wSensor.value]
        return self.sensor_format

    def _set_sensor_format(self, mode='standard'):
        mode_numbers = {"standard": 0, "extended": 1}
        if self.very_verbose:
            print(" Setting sensor format to:", mode)
        dll.set_sensor_format(self.camera_handle, mode_numbers[mode])
        assert self._get_sensor_format() == mode
        return self.sensor_format
    
    def _get_camera_health(self):
        dwWarn, dwErr, dwStatus = (
            C.c_uint32(), C.c_uint32(), C.c_uint32())
        dll.get_camera_health(self.camera_handle, dwWarn, dwErr, dwStatus)
        if self.verbose:
            print(" Camera health status:", end='')
            print("  Warnings:", dwWarn.value, end='')
            if dwWarn.value == 0:
                print(" (good)", end='')
            else:
                print("***BAD***")
            print(" / Errors:", dwErr.value, end='')
            if dwErr.value == 0:
                print(" (good)", end='')
            else:
                print("***BAD***")
            print(" / Status:", dwStatus.value)
        self.camera_health = {
            'warnings': dwWarn.value,
            'errors': dwErr.value,
            'status': dwStatus.value}
        return self.camera_health

    def _get_temperature(self):
        ccdtemp, camtemp, powtemp = (
            C.c_int16(), C.c_int16(), C.c_int16())
        dll.get_temperature(self.camera_handle, ccdtemp, camtemp, powtemp)
        if self.verbose:
            print(" Temperatures:",
                  "CCD", ccdtemp.value * 0.1, "C /",
                  "camera", camtemp.value, "C /",
                  "power supply", powtemp.value, "C ")
        self.temperature = {
            'ccd_temp': ccdtemp.value * 0.1,
            'camera_temp': camtemp.value,
            'power_supply_temp': powtemp.value}
        return self.temperature

    def _get_trigger_mode(self):
        """
        0x0000 = [auto trigger]
        A new image exposure is automatically started best possible
        compared to the readout of an image. If a CCD is used and the
        images are taken in a sequence, then exposures and sensor readout
        are started simultaneously. Signals at the trigger input (<exp
        trig>) are irrelevant.
        - 0x0001 = [software trigger]:
        An exposure can only be started by a force trigger command.
        - 0x0002 = [extern exposure & software trigger]:
        A delay / exposure sequence is started at the RISING or FALLING
        edge (depending on the DIP switch setting) of the trigger input
        (<exp trig>).
        - 0x0003 = [extern exposure control]:
        The exposure time is defined by the pulse length at the trigger
        input(<exp trig>). The delay and exposure time values defined by
        the set/request delay and exposure command are ineffective.
        (Exposure time length control is also possible for double image
        mode; exposure time of the second image is given by the readout
        time of the first image.)
        """
        trigger_mode_names = {0: "auto_trigger",
                              1: "software_trigger",
                              2: "external_trigger",
                              3: "external_exposure"}
        wTriggerMode = C.c_uint16()
        dll.get_trigger_mode(self.camera_handle, wTriggerMode)
        if self.verbose:
            print(" Trigger mode:", trigger_mode_names[wTriggerMode.value])
        self.trigger_mode = trigger_mode_names[wTriggerMode.value]
        return self.trigger_mode
    
    def _set_trigger_mode(self, mode="auto_trigger"):
        trigger_mode_numbers = {
            "auto_trigger": 0,
            "external_trigger": 2}
        if self.verbose: print(" Setting trigger mode to:", mode)
        dll.set_trigger_mode(self.camera_handle, trigger_mode_numbers[mode])
        assert self._get_trigger_mode() == mode
        return self.trigger_mode

    def _get_storage_mode(self):
        wStorageMode = C.c_uint16()
        dll.get_storage_mode(self.camera_handle, wStorageMode)
        storage_mode_names = {0: "recorder",
                              1: "FIFO_buffer"}
        if self.very_verbose:
            print(" Storage mode:", storage_mode_names[wStorageMode.value])
        self.storage_mode = storage_mode_names[wStorageMode.value]
        return self.storage_mode

    def _set_storage_mode(self, mode="recorder"):
        storage_mode_numbers = {"recorder": 0,
                                "FIFO_buffer": 1}
        if self.very_verbose: print(" Setting storage mode to:", mode)
        dll.set_storage_mode(self.camera_handle, storage_mode_numbers[mode])
        assert self._get_storage_mode() == mode
        return self.storage_mode

    def _get_recorder_submode(self):
        wRecSubmode = C.c_uint16(1)
        dll.get_recorder_submode(self.camera_handle, wRecSubmode)
        recorder_submode_names = {0: "sequence",
                                  1: "ring_buffer"}
        if self.very_verbose:
            print(" Recorder submode:",
                  recorder_submode_names[wRecSubmode.value])
        self.recorder_submode = recorder_submode_names[wRecSubmode.value]
        return self.recorder_submode

    def _set_recorder_submode(self, mode="ring_buffer"):
        recorder_mode_numbers = {
            "sequence": 0,
            "ring_buffer": 1}
        if self.very_verbose: print(" Setting recorder submode to:", mode)
        dll.set_recorder_submode(
            self.camera_handle, recorder_mode_numbers[mode])
        assert self._get_recorder_submode() == mode
        return self.recorder_submode

    def _get_acquire_mode(self):
        wAcquMode = C.c_uint16(0)
        dll.get_acquire_mode(self.camera_handle, wAcquMode)
        acquire_mode_names = {0: "auto",
                              1: "external_static",
                              2: "external_dynamic"}
        if self.very_verbose:
            print(" Acquire mode:", acquire_mode_names[wAcquMode.value])
        self.acquire_mode = acquire_mode_names[wAcquMode.value]
        return self.acquire_mode

    def _set_acquire_mode(self, mode='auto'):
        acquire_mode_numbers = {"auto": 0,
                                "external_static": 1,
                                "external_dynamic": 2}
        if self.very_verbose: print(" Setting acquire mode to:", mode)
        dll.set_acquire_mode(self.camera_handle, acquire_mode_numbers[mode])
        assert self._get_acquire_mode() == mode
        return self.acquire_mode

    def _get_pixel_rate(self):
        dwPixelRate = C.c_uint32(0)
        dll.get_pixel_rate(self.camera_handle, dwPixelRate)
        assert dwPixelRate.value != 0
        if self.very_verbose: print(" Pixel rate:", dwPixelRate.value)
        self.pixel_rate = dwPixelRate.value
        return self.pixel_rate

    def _set_pixel_rate(self, rate=272250000):
        if self.very_verbose: print(" Setting pixel rate to:", rate)
        dll.set_pixel_rate(self.camera_handle, rate)
        assert self._get_pixel_rate() == rate
        return self.pixel_rate

    def _get_exposure_time(self):
        dwDelay = C.c_uint32(0)
        wTimeBaseDelay = C.c_uint16(0)
        dwExposure = C.c_uint32(0)
        wTimeBaseExposure = C.c_uint16(1)
        dll.get_delay_exposure_time(
            self.camera_handle,
            dwDelay,
            dwExposure,
            wTimeBaseDelay,
            wTimeBaseExposure)
        time_base_mode_names = {0: "nanoseconds",
                                1: "microseconds",
                                2: "milliseconds"}
        if self.verbose:
            print(" Exposure:", dwExposure.value,
                  time_base_mode_names[wTimeBaseExposure.value])
        if self.very_verbose:
            print(" Delay:", dwDelay.value,
                  time_base_mode_names[wTimeBaseDelay.value])
        self.exposure_time_microseconds = (
            dwExposure.value * 10.**(3*wTimeBaseExposure.value - 3))
        self.delay_time = dwDelay.value
        return self.exposure_time_microseconds

    def _set_exposure_time(self, exposure_time_microseconds=2200):
        exposure_time_microseconds = int(exposure_time_microseconds)
        assert 1e2 <= exposure_time_microseconds <= 1e7
        if self.verbose:
            print(" Setting exposure time to", exposure_time_microseconds, "us")
        dll.set_delay_exposure_time(
            self.camera_handle, 0, exposure_time_microseconds, 1, 1)
        assert self._get_exposure_time() == exposure_time_microseconds
        return self.exposure_time_microseconds

    
    def _get_roi(self):
        wRoiX0, wRoiY0, wRoiX1, wRoiY1 = (
            C.c_uint16(), C.c_uint16(),
            C.c_uint16(), C.c_uint16())
        dll.get_roi(self.camera_handle, wRoiX0, wRoiY0, wRoiX1, wRoiY1)
        if self.verbose:
            print(" Camera ROI:");
            print("  From pixel", wRoiX0.value, "to pixel", wRoiX1.value, "(left/right)")
            print("  From pixel", wRoiY0.value, "to pixel", wRoiY1.value, "(up/down)")
        self.roi = {
            'left': wRoiX0.value,
            'top': wRoiY0.value,
            'right': wRoiX1.value,
            'bottom': wRoiY1.value}
        """
        How long do we expect the chip to spend rolling? Both the 4.2
        and the 5.5 take ~10 ms to roll the full chip. Calculate the
        fraction of the chip we're using and estimate the rolling time.
        """
        if self.pco_edge_type == '4.2':
            max_lines = 1024
        elif self.pco_edge_type == '5.5':
            max_lines = 1080
        full_chip_rolling_time = 1e4
        chip_fraction = max(wRoiY1.value - max_lines,
                            max_lines + 1 - wRoiY0.value) / max_lines
        self.rolling_time_microseconds =  full_chip_rolling_time * chip_fraction
        return self.roi

    def _legalize_roi(self, roi):
        """
        There are lots of ways a requested region of interest (ROI) can
        be illegal. This utility function returns a nearby legal ROI.

        Optionally, you can leave keys of 'roi' unspecified, and
        _legalize_roi() tries to return reasonable choices based on
        the current values in self.roi
        """
        left = roi.get('left')
        right = roi.get('right')
        bottom = roi.get('bottom')
        top = roi.get('top')
        if self.verbose:
            print(" Requested camera ROI:")
            print("  From pixel", left, "to pixel", right, "(left/right)")
            print("  From pixel", top, "to pixel", bottom, "(up/down)")
        min_lr, min_ud, min_height = 1, 1, 10
        if self.pco_edge_type == '4.2':
            max_lr, max_ud, min_width, step_lr = 2060, 2048, 40, 20
        elif self.pco_edge_type == '5.5':
            max_lr, max_ud, min_width, step_lr = 2560, 2160, 160, 160
        """
        Legalize left/right
        """
        if left is None and right is None:
            """
            User isn't trying to change l/r ROI; use existing ROI.
            """
            left, right = self.roi['left'], self.roi['right']
        elif left is not None:
            """
            'left' is specified, 'left' is the master.
            """
            if left < min_lr: #Legalize 'left'
                left = min_lr
            elif left > max_lr - min_width + 1:
                left = max_lr - min_width + 1
            else:
                left = 1 + step_lr*((left - 1) // step_lr)
            if right is None: #Now legalize 'right'
                right = self.roi['right']
            if right < left + min_width - 1:
                right = left + min_width - 1
            elif right > max_lr:
                right = max_lr
            else:
                right = left - 1 + step_lr*((right - (left - 1)) // step_lr)
        else:
            """
            'left' is unspecified, 'right' is specified. 'right' is the master.
            """
            if right > max_lr: #Legalize 'right'
                right = max_lr
            elif right < min_lr - 1 + min_width:
                right = min_width
            else:
                right = step_lr * (right  // step_lr)
            left = self.roi['left'] #Now legalize 'left'
            if left > right - min_width + 1:
                left = right - min_width + 1
            elif left < min_lr:
                left = min_lr
            else:
                left = right + 1 - step_lr * ((right - (left - 1)) // step_lr)
        assert min_lr <= left < left + min_width - 1 <= right <= max_lr
        """
        Legalize top/bottom
        """
        if top is None and bottom is None:
            """
            User isn't trying to change u/d ROI; use existing ROI.
            """
            top, bottom = self.roi['top'], self.roi['bottom']
        elif top is not None:
            """
            'top' is specified, 'top' is the master.
            """
            if top < min_ud: #Legalize 'top'
                top = min_ud
            if top > (max_ud - min_height)//2 + 1:
                top = (max_ud - min_height)//2 + 1
            bottom = max_ud - top + 1 #Now bottom is specified
        else:
            """
            'top' is unspecified, 'bottom' is specified, 'bottom' is the
            master.
            """
            if bottom > max_ud: #Legalize 'bottom'
                bottom = max_ud
            if bottom < (max_ud + min_height)//2:
                bottom = (max_ud + min_height)//2
            top = max_ud - bottom + 1 #Now 'top' is specified
        assert min_ud <= top < top + min_height - 1 <= bottom <= max_ud
        new_roi = {'left': left,
                   'top': top,
                   'right': right,
                   'bottom': bottom}
        if self.verbose and new_roi != roi:
            print(" ***Requested ROI must be adjusted to match the camera***")
        return new_roi

    def _set_roi(self, region_of_interest):
        roi = self._legalize_roi(region_of_interest)
        dll.set_roi(self.camera_handle,
                    roi['left'], roi['top'], roi['right'], roi['bottom'])
        assert self._get_roi() == roi
        return self.roi

def pco_edge_camera_child_process(
    data_buffers,
    buffer_shape,
    input_queue,
    output_queue,
    commands,
    pco_edge_type='4.2' #Change this if you're using a 5.5
    ):
    """For use with image_data_pipeline.py

    https://github.com/AndrewGYork/tools/blob/master/image_data_pipeline.py
    Debugged for the 4.2, but might work for the 5.5, with some TLC...
    """
    from image_data_pipeline import info, Q, sleep, clock
    try:
        import pco
    except ImportError:
        info("Failed to import pco.py; go get it from github:")
        info("https://github.com/AndrewGYork/tools/blob/master/pco.py")
        raise
    buffer_size = np.prod(buffer_shape)
    info("Initializing...")
    camera = pco.Edge(pco_edge_type=pco_edge_type, verbose=False)
    camera.apply_settings(trigger='auto_trigger')
    camera.arm(num_buffers=3)
    info("Done initializing")
    preframes = 3
    first_trigger_timeout_seconds = 0
    status = 'Normal'
    while True:
        if commands.poll():
            cmd, args = commands.recv()
            info("Command received: " + cmd)
            if cmd == 'apply_settings':
                result = camera.apply_settings(**args)
                camera.arm(num_buffers=3)
                commands.send(result)
            elif cmd == 'get_setting':
                setting = getattr(
                    camera, args['setting'], 'unrecognized_setting')
                commands.send(setting)
            elif cmd == 'set_buffer_shape':
                buffer_shape = args['shape']
                buffer_size = np.prod(buffer_shape)
                commands.send(buffer_shape)
            elif cmd == 'get_status':
                commands.send(status)
            elif cmd == 'reset_status':
                status = 'Normal'
                commands.send(status)
            elif cmd == 'get_preframes':
                commands.send(preframes)
            elif cmd == 'set_preframes':
                preframes = args['preframes']
                commands.send(preframes)
            elif cmd == 'get_first_trigger_timeout_seconds':
                commands.send(first_trigger_timeout_seconds)
            elif cmd == 'set_first_trigger_timeout_seconds':
                first_trigger_timeout_seconds = args[
                    'first_trigger_timeout_seconds']
                commands.send(first_trigger_timeout_seconds)
            else:
                info("Unrecognized command: " + cmd)
                commands.send("unrecognized_command")
                continue
        try:
            permission_slip = input_queue.get_nowait()
        except Q.Empty:
            sleep(0.001) #Non-deterministic sleep time :(
            continue
        if permission_slip is None: #This is how we signal "shut down"
            output_queue.put(permission_slip)
            break #We're done
        else:
            # Fill the data buffer with images from the camera
            time_received = clock()
            process_me = permission_slip['which_buffer']
            info("start buffer %i, acquiring %i frames and %i preframes"%(
                process_me, buffer_shape[0], preframes))
            with data_buffers[process_me].get_lock():
                a = np.frombuffer(data_buffers[process_me].get_obj(),
                                  dtype=np.uint16)[:buffer_size
                                                   ].reshape(buffer_shape)
                try:
                    camera.record_to_memory(
                        num_images=a.shape[0] + preframes,
                        preframes=preframes,
                        out=a,
                        first_trigger_timeout_seconds=(
                            first_trigger_timeout_seconds))
                except pco.TimeoutError as e:
                    info('TimeoutError: %s'%(e.value))
                    status = 'TimeoutError'
                    #FIXME: we can do better, probably. Keep trying?
                    #Should we zero the remainder of 'a'?
                except pco.DMAError:
                    info('DMAError')
                    status = 'DMAError'
                else:
                    status = 'Normal'
            info("end buffer %i, %06f seconds elapsed"%(
                process_me, clock() - time_received))
            output_queue.put(permission_slip)
    camera.close()
    return None

"""
A few types of exception we'll use during recording:
"""
class TimeoutError(Exception):
    def __init__(self, value, num_acquired=0):
        self.value = value
        self.num_acquired = num_acquired
    def __str__(self):
        return repr(self.value)

class DMAError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

"""
DLL management
"""
try:
    dll = C.oledll.LoadLibrary("SC2_Cam")
    """
    If you get a WindowsError, read PCO_err.h to decypher it.
    """
except WindowsError:
    print("Failed to load SC2_Cam.dll")
    print("You need this to run pco.py")
    raise

"""
This command opens the next pco camera; if you want to have multiple
cameras, and pick which one you're opening, I'd have to implement
PCO_OpenCameraEx, which would require me to understand PCO_OpenStruct.
"""
dll.open_camera = dll.PCO_OpenCamera
dll.open_camera.argtypes = [C.POINTER(C.c_void_p), C.c_uint16]

dll.close_camera = dll.PCO_CloseCamera
dll.close_camera.argtypes = [C.c_void_p]

dll.arm_camera = dll.PCO_ArmCamera
dll.arm_camera.argtypes = [C.c_void_p]

dll.allocate_buffer = dll.PCO_AllocateBuffer
dll.allocate_buffer.argtypes = [
    C.c_void_p,
    C.POINTER(C.c_int16),
    C.c_uint32,
    C.POINTER(C.POINTER(C.c_uint16)),
    C.POINTER(C.c_void_p)]

dll.add_buffer = dll.PCO_AddBufferEx
dll.add_buffer.argtypes = [
    C.c_void_p,
    C.c_uint32,
    C.c_uint32,
    C.c_int16,
    C.c_uint16,
    C.c_uint16,
    C.c_uint16]

dll.get_buffer_status = dll.PCO_GetBufferStatus
dll.get_buffer_status.argtypes = [
    C.c_void_p,
    C.c_int16,
    C.POINTER(C.c_uint32),
    C.POINTER(C.c_uint32)]

dll.set_image_parameters = dll.PCO_CamLinkSetImageParameters
dll.set_image_parameters.argtypes = [C.c_void_p, C.c_uint16, C.c_uint16]

dll.set_recording_state = dll.PCO_SetRecordingState
dll.set_recording_state.argtypes = [C.c_void_p, C.c_uint16]

dll.get_sizes = dll.PCO_GetSizes
dll.get_sizes.argtypes = [
    C.c_void_p,
    C.POINTER(C.c_uint16),
    C.POINTER(C.c_uint16),
    C.POINTER(C.c_uint16),
    C.POINTER(C.c_uint16)]

dll.get_sensor_format = dll.PCO_GetSensorFormat
dll.get_sensor_format.argtypes = [C.c_void_p, C.POINTER(C.c_uint16)]

dll.get_camera_health = dll.PCO_GetCameraHealthStatus
dll.get_camera_health.argtypes = [
    C.c_void_p,
    C.POINTER(C.c_uint32),
    C.POINTER(C.c_uint32),
    C.POINTER(C.c_uint32)]

dll.get_temperature = dll.PCO_GetTemperature
dll.get_temperature.argtypes = [
    C.c_void_p,
    C.POINTER(C.c_int16),
    C.POINTER(C.c_int16),
    C.POINTER(C.c_int16)]

dll.get_trigger_mode = dll.PCO_GetTriggerMode
dll.get_trigger_mode.argtypes = [C.c_void_p, C.POINTER(C.c_uint16)]

dll.get_storage_mode = dll.PCO_GetStorageMode
dll.get_storage_mode.argtypes = [C.c_void_p, C.POINTER(C.c_uint16)]

dll.get_recorder_submode = dll.PCO_GetRecorderSubmode
dll.get_recorder_submode.argtypes = [C.c_void_p, C.POINTER(C.c_uint16)]

dll.get_acquire_mode = dll.PCO_GetAcquireMode
dll.get_acquire_mode.argtypes = [C.c_void_p, C.POINTER(C.c_uint16)]

dll.get_pixel_rate = dll.PCO_GetPixelRate
dll.get_pixel_rate.argtypes = [C.c_void_p, C.POINTER(C.c_uint32)]

dll.set_pixel_rate = dll.PCO_SetPixelRate
dll.set_pixel_rate.argtypes = [C.c_void_p, C.c_uint32]

dll.get_delay_exposure_time = dll.PCO_GetDelayExposureTime
dll.get_delay_exposure_time.argtypes = [
    C.c_void_p,
    C.POINTER(C.c_uint32),
    C.POINTER(C.c_uint32),
    C.POINTER(C.c_uint16),
    C.POINTER(C.c_uint16)]

dll.set_delay_exposure_time = dll.PCO_SetDelayExposureTime
dll.set_delay_exposure_time.argtypes = [
    C.c_void_p,
    C.c_uint32,
    C.c_uint32,
    C.c_uint16,
    C.c_uint16]

dll.get_roi = dll.PCO_GetROI
dll.get_roi.argtypes = [
    C.c_void_p,
    C.POINTER(C.c_uint16),
    C.POINTER(C.c_uint16),
    C.POINTER(C.c_uint16),
    C.POINTER(C.c_uint16)]

dll.set_roi = dll.PCO_SetROI
dll.set_roi.argtypes = [
    C.c_void_p,
    C.c_uint16,
    C.c_uint16,
    C.c_uint16,
    C.c_uint16]

dll.reset_settings_to_default = dll.PCO_ResetSettingsToDefault
dll.reset_settings_to_default.argtypes = [C.c_void_p]

dll.set_recording_state = dll.PCO_SetRecordingState
dll.set_recording_state.argtypes = [C.c_void_p, C.c_uint16]

dll.remove_buffer = dll.PCO_RemoveBuffer
dll.remove_buffer.argtypes = [C.c_void_p]

dll.free_buffer = dll.PCO_FreeBuffer
dll.free_buffer.argtypes = [C.c_void_p, C.c_int16]

dll.set_sensor_format = dll.PCO_SetSensorFormat
dll.set_sensor_format.argtypes = [C.c_void_p, C.c_uint16]

dll.set_trigger_mode = dll.PCO_SetTriggerMode
dll.set_trigger_mode.argtypes = [C.c_void_p, C.c_uint16]

dll.set_recorder_submode = dll.PCO_SetRecorderSubmode
dll.set_recorder_submode.argtypes = [C.c_void_p, C.c_uint16]

dll.set_acquire_mode = dll.PCO_SetAcquireMode
dll.set_acquire_mode.argtypes = [C.c_void_p, C.c_uint16]

dll.set_storage_mode = dll.PCO_SetStorageMode
dll.set_storage_mode.argtypes = [C.c_void_p, C.c_uint16]

if __name__ == '__main__':
    """
    Half-assed testing; give randomized semi-garbage inputs, hope the
    plane don't crash.
    """
    camera = Edge()
    for i in range(10000):
        """
        Random exposure time, biased towards shorter exposures
        """
        exposure = min(np.random.randint(1e2, 1e7, size=40))
        """
        Random ROI, potentially with some/all limits unspecified.
        """
        roi = {
            'top': np.random.randint(low=-2000, high=3000),
            'bottom': np.random.randint(low=-2000, high=3000),
            'left': np.random.randint(low=-2000, high=3000),
            'right': np.random.randint(low=-2000, high=3000)}
        roi = {k: v for k, v in roi.items() if v > -10} #Delete some keys/vals
        camera.apply_settings(exposure_time_microseconds=exposure,
                              region_of_interest=roi)
        camera.arm(num_buffers=np.random.randint(1, 16))
        print("Allocating memory...")
        images = np.zeros((np.random.randint(1, 5), camera.height, camera.width),
                          dtype=np.uint16)
        print("Done allocating memory.")
        print("Expected time:",
              images.shape[0] *
              1e-6 * max(camera.rolling_time_microseconds,
                         camera.exposure_time_microseconds))
        start = time.perf_counter()
        camera.record_to_memory(num_images=images.shape[0], out=images)
        print("Elapsed time:", time.perf_counter() - start)
        
        print(images.min(), images.max(), images.shape)
        camera.disarm()
    camera.close()

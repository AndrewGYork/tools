import time
import multiprocessing as mp
import logging
import numpy as np
import image_data_pipeline
from theimagingsource import DMK_camera_child_process
import ni
import zaber

class Scanner:
    def __init__(
        self,
        zaber_stage_port_name, # For example, 'COM3'
        exposure_time_seconds=0.2,
        bin_size=None,
        ):
        # Image data pipeline, to get frames off the camera, display
        # them on-screen, and possibly save them to disk:
        self.idp = image_data_pipeline.Image_Data_Pipeline(
            num_buffers=1,
            buffer_shape=(1, 1944, 2592),
            camera_child_process=DMK_camera_child_process)
        self.idp.display.withdraw() # To prevent focus-theft
        self.idp.apply_camera_settings(
            exposure_time_microseconds=exposure_time_seconds * 1e6,
            trigger='external_trigger')
        self.idp.display.withdraw() # To prevent focus-theft
        self.idp.file_saving.commands.send(
            ('set_bin_size', {'bin_size': bin_size}))
        # Analog-out card, to trigger the camera and the illumination:
        self.analog_out = ni.Analog_Out(
            num_channels='all',
            rate=1e5,
            daq_type='9263',
            board_name='cDAQ1Mod1')
        self.set_exposure_and_dose(dose_duration_488_seconds=5,
                                   dose_duration_405_seconds=5,
                                   snap_voltage_488=4,
                                   snap_voltage_405=4,
                                   dose_voltage_488=4,
                                   dose_voltage_405=4)
        # XY stage, to scan multiple fields of view
        self.stage = zaber.Stage(port_name=zaber_stage_port_name, timeout=7)
        self.stage.min_x, self.stage.min_y = 0, 0
        self.stage.max_x, self.stage.max_y = 303118, 303118
        self.last_snap_x, self.last_snap_y = -1, -1
        return None

    def snap(
        self,
        illumination,
        x=None,
        y=None,
        filename=None,
        stage_cooldown_seconds=0.1):
        start_time = time.perf_counter()
        assert illumination in ('background', '488', '405')
        illumination = 'snap_' + illumination
        stage_start = time.perf_counter()
        stage_moves = False
        if x is not None and x != self.last_snap_x:
            assert self.stage.min_x <= x <= self.stage.max_x
            self.stage.move(x, axis=2, response=False)
            self.last_snap_x = x
            stage_moves = True
        if y is not None and y != self.last_snap_y:
            assert self.stage.min_y <= y <= self.stage.max_y
            self.stage.move(y, axis=1, response=False)
            self.last_snap_y = y
            stage_moves = True
        if filename is not None:
            file_saving_info = [{'filename': filename}]
        else:
            file_saving_info = None
        self.stage.finish_moving()
        stage_done_moving = time.perf_counter()
        slip_load_start = time.perf_counter()
        self.idp.load_permission_slips(
            num_slips=1,
            file_saving_info=file_saving_info,
            timeout=4)
        slip_load_end = time.perf_counter()
        if stage_moves: # Wait for the stage motion to finish vibrating
            while (time.perf_counter() - stage_start) < stage_cooldown_seconds:
                time.sleep(0.001)
        stage_done_cooling = time.perf_counter()
        if self.last_played_voltage == illumination:
            v = None # No need to re-load the voltages
        else:
            v = self.voltages[illumination]
            self.last_played_voltage = illumination
        while not self.idp.camera.input_queue.empty(): # Is the camera ready?
            time.sleep(0.001) # 'fraid not. Give it a (milli)sec.
        voltage_start_playing = time.perf_counter()
        self.analog_out.play_voltages(v)
        voltage_done_playing = time.perf_counter()
        print()
        print("Stage start moving:", stage_start - start_time)
        print("Stage done moving: ", stage_done_moving - start_time)
        print("Slip start loading:", slip_load_start - start_time)
        print("Slip done loading: ", slip_load_end - start_time)
        print("Stage done cooling:", stage_done_cooling - start_time)
        print("Voltage start playing:", voltage_start_playing - start_time)
        print("Voltage done playing:", voltage_done_playing - start_time)
        return None

    def dose(
        self,
        illumination,
        x=None,
        y=None
        ):
        assert illumination in ('488', '405')
        illumination = 'dose_' + illumination
        if x is not None:
            assert self.stage.min_x <= x <= self.stage.max_x
            self.stage.move(x, axis=2, response=False)
        if y is not None:
            assert self.stage.min_y <= y <= self.stage.max_y
            self.stage.move(y, axis=1, response=False)
        self.stage.finish_moving()
        if self.last_played_voltage == illumination:
            v = None # No need to re-load the voltages
        else:
            v = self.voltages[illumination]
            self.last_played_voltage = illumination
        self.analog_out.play_voltages(v)
        return None

    def set_exposure_and_dose(
        self,
        exposure_time_seconds=None,
        dose_duration_488_seconds=None,
        dose_duration_405_seconds=None,
        snap_voltage_488=None,
        snap_voltage_405=None,
        dose_voltage_488=None,
        dose_voltage_405=None,
        ):
        assert (exposure_time_seconds is not None or
                dose_duration_488_seconds is not None or
                dose_duration_405_seconds is not None or
                snap_voltage_488 is not None or
                snap_voltage_405 is not None or
                dose_voltage_488 is not None or
                dose_voltage_405 is not None)
        if exposure_time_seconds is not None:
            self.idp.apply_camera_settings(
                exposure_time_microseconds=exposure_time_seconds * 1e6)
        if dose_duration_488_seconds is not None:
            self.dose_duration_488_seconds = float(dose_duration_488_seconds)
        if dose_duration_405_seconds is not None:
            self.dose_duration_405_seconds = float(dose_duration_405_seconds)
        if dose_voltage_488 is not None:
            assert 0 < dose_voltage_488 < 5
            self.dose_voltage_488 = float(dose_voltage_488)
        if dose_voltage_405 is not None:
            assert 0 < dose_voltage_405 < 5
            self.dose_voltage_405 = float(dose_voltage_405)
        if snap_voltage_488 is not None:
            assert 0 < snap_voltage_488 < 5
            self.snap_voltage_488 = float(snap_voltage_488)
        if snap_voltage_405 is not None:
            assert 0 < snap_voltage_405 < 5
            self.snap_voltage_405 = float(snap_voltage_405)
        self._create_voltages()
        return None

    def _create_voltages(self):
        """Define useful voltages that we expect to reuse frequently.

        Re-run this if you change the camera exposure time, or if you
        want to change the duration of the 488/405 doses.
        """
        # Voltages for taking images:
        seconds_from_trigger_to_exposure = 936e-6
        seconds_per_line = 72.5e-6
        num_lines = self.idp.camera.get_setting('height')
        rolling_time = seconds_per_line * num_lines
        exposure_time = (
            1e-6 * self.idp.camera.get_setting('exposure_time_microseconds'))
        if exposure_time <= rolling_time:
            print("Camera exposure time:", exposure_time, "seconds")
            print("Camera rolling time:", rolling_time, "seconds")
            raise UserWarning(
                "Camera exposure time is too short for global operation" +
                " with rolling shutter.")
        start_time = (seconds_from_trigger_to_exposure + rolling_time)
        stop_time = (seconds_from_trigger_to_exposure + exposure_time)
        end_time = (seconds_from_trigger_to_exposure +
                    rolling_time + exposure_time) # Camera must "unroll", too
        start_pix = int(round(start_time * self.analog_out.rate))
        stop_pix = int(round(stop_time * self.analog_out.rate))
        end_pix = int(round(end_time * self.analog_out.rate))
        print("LED voltage is on from AO pixel", start_pix, "to", stop_pix)
        voltage_shape = (end_pix, self.analog_out.num_channels)
        voltages = {t: np.zeros(voltage_shape, dtype=np.float64)
                    for t in ('snap_background', 'snap_488', 'snap_405')}
        # Add triggers for the camera:
        for v in voltages.values():
            v[0:10, 0] = 4 # Camera trigger duration doesn't really matter
        # Add triggers for the LEDs:
        voltages['snap_488'][start_pix:stop_pix, 1] = self.snap_voltage_488
        voltages['snap_405'][start_pix:stop_pix, 2] = self.snap_voltage_405
        # Voltages for photoconversion, without imaging:
        voltages['dose_488'] = np.zeros(
            (self.dose_duration_488_seconds * self.analog_out.rate, #Autorounds
             self.analog_out.num_channels), dtype=np.float64)
        voltages['dose_488'][:-1, 1] = self.dose_voltage_488
        voltages['dose_405'] = np.zeros(
            (self.dose_duration_405_seconds * self.analog_out.rate, #Autorounds
             self.analog_out.num_channels), dtype=np.float64)
        voltages['dose_405'][:-1, 2] = self.dose_voltage_405
        self.voltages = voltages
        self.last_played_voltage = None
        return None

    def close(self):
        self.idp.close()
        self.analog_out.close()
        self.stage.close()

if __name__ == '__main__':
    logger = mp.log_to_stderr()
    logger.setLevel(logging.INFO)

    scanner = Scanner(zaber_stage_port_name='COM3')

    try:
        for x in range(1, 100000, 1000):
            print("New x position:", x)
            for i in range(3):
                scanner.snap('488', x=x, y=20000, stage_cooldown_seconds=0.1)
##                input("Hit enter...")
    finally:
        scanner.close()
    input("Hit enter to finish")

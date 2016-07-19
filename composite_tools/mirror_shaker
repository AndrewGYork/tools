import time
import os
import multiprocessing as mp
import logging
import numpy as np
from scipy.ndimage import filters
import image_data_pipeline
import ni
from theimagingsource import DMK_camera_child_process
import np_tif
        
n2c = {'camera': 0,
       'laser': 1,
       'mirror': 2,
       'dummy': 3} # Odd number of pix mandatory. Thanks, NI!

class Mirror_Shaker(object):
    def __init__(
        self,
        num_measurements=15,
        ):
        # Grab control of hardware
        self.analog_out = ni.Analog_Out(
            num_channels=len(n2c.keys()),
            num_digital_lines=0,
            rate=5e5,
            daq_type='6733',
            board_name='Dev1')
        self.idp = image_data_pipeline.Image_Data_Pipeline(
            num_buffers=1,
            buffer_shape=(num_measurements, 1944, 2592),
            camera_child_process=DMK_camera_child_process)
        self.idp.apply_camera_settings(trigger='external_trigger')
        self.idp.display.withdraw() # To prevent focus-theft
        self.mirror_voltage_shape = None
        self.measurement_start_pixel = None
        self.measurement_pixels = None
        self.illumination = None
        self._calibrate_laser()
        return None

    def snap_mirror_motion(
        self,
        mirror_voltage,
        measurement_start_pixel=0,
        measurement_pixels=100,
        ao_pixels_per_second = 5e5,
        filename='test.tif',
        illumination='sinusoidal',
        ):
        # Validate inputs
        assert (0 <=
                measurement_start_pixel <
                measurement_start_pixel + measurement_pixels <=
                mirror_voltage.shape[0])
        assert 0 < ao_pixels_per_second < self.analog_out.max_rate
        if isinstance(illumination, str):
            assert illumination in ('sinusoidal',)
        else:
            assert illumination.shape == (self.idp.buffer_shape[0],
                                          measurement_pixels)
        if (mirror_voltage.shape != self.mirror_voltage_shape or
            measurement_start_pixel != self.measurement_start_pixel or
            measurement_pixels != self.measurement_pixels or
            ao_pixels_per_second != self.analog_out.rate or
            not np.array_equal(illumination, self.illumination)):
            # We have to re-make the camera and modulator voltages:
            self.mirror_voltage_shape = mirror_voltage.shape
            self.measurement_start_pixel = measurement_start_pixel
            self.measurement_pixels = measurement_pixels
            self.analog_out.set_rate(ao_pixels_per_second)
            self.illumination = illumination
            self._reset_voltages()
        # Set the mirror-shaking voltages
        start = self.prevoltage_pixels - self.measurement_start_pixel
        self.voltages[:,
                      start:start + self.mirror_voltage_shape[0],
                      n2c['mirror']] = mirror_voltage
        # Flatten the 3D voltages into a 2D array
        v = self.voltages.reshape(
            self.idp.buffer_shape[0] * self.voltages.shape[1],
            self.analog_out.num_channels)
        # Play the voltages!
        while len(self.idp.idle_data_buffers) < self.idp.num_data_buffers:
            self.idp.collect_permission_slips()
        self.idp.load_permission_slips(
            num_slips=1,
            file_saving_info=[{'filename': filename}],
            timeout=4)
        while not self.idp.camera.input_queue.empty(): # Is the camera ready?
            time.sleep(0.001) # 'fraid not. Give it a (milli)sec.
        self.analog_out.play_voltages(v)
        while len(self.idp.idle_data_buffers) < self.idp.num_data_buffers:
            self.idp.collect_permission_slips() #Block 'til completion
        return None

    def close(self):
        self.analog_out.close()
        self.idp.close()
        return None

    def _reset_voltages(self):
        # Calculate the camera exposure time, to get "global" exposure
        # despite the rolling shutter.
        seconds_per_line = 72.5e-6
        rolling_time_seconds = (seconds_per_line *
                                self.idp.camera.get_setting('height'))
        measurement_time_seconds = (self.measurement_pixels /
                                    self.analog_out.rate)
        exposure_time_seconds = rolling_time_seconds + measurement_time_seconds
        self.idp.apply_camera_settings(
            exposure_time_microseconds=int(exposure_time_seconds * 1e6))
        exposure_time_seconds = (# Check we got what we wanted
            1e-6 * self.idp.camera.get_setting('exposure_time_microseconds'))
        # Calculate the length of the pre-measurement voltages
        seconds_from_trigger_to_exposure = 936e-6
        global_exposure_start_pix = int(np.round(self.analog_out.rate * (
            seconds_from_trigger_to_exposure + rolling_time_seconds)))
        self.prevoltage_pixels = max(global_exposure_start_pix,
                                     self.measurement_start_pixel)
        # Calculate the length of the post-measurement voltages
        seconds_per_readout = 100e-3
        camera_readout_pixels = int(np.round(self.analog_out.rate *
                                             seconds_per_readout))
        mirror_postvoltage_pixels = (self.mirror_voltage_shape[0] +
                                     -self.measurement_pixels +
                                     -self.measurement_start_pixel)
        postvoltage_pixels = max(camera_readout_pixels,
                                 mirror_postvoltage_pixels)
        # We're ready to allocate the array to hold our voltages
        self.voltages = np.zeros(
            (self.idp.buffer_shape[0], # Num measurements
             self.prevoltage_pixels+self.measurement_pixels+postvoltage_pixels,
             self.analog_out.num_channels),
            dtype=np.float64)
        # Trigger the camera
        self.voltages[
            :,
            self.prevoltage_pixels - global_exposure_start_pix:
            self.prevoltage_pixels,
            n2c['camera']] = 4
        # Modulate the laser
        if isinstance(self.illumination, str):
            if self.illumination == 'sinusoidal':
                modulation_amplitude = 0.4
                modulation_offset = 0.6
                cycles_per_measurement = (0.5, 1, 1.5)
                num_phases = 5
                assert (len(cycles_per_measurement) * num_phases ==
                        self.idp.buffer_shape[0])
                t = (np.arange(self.measurement_pixels) /
                     self.measurement_pixels)
                which_measurement = -1
                for cycles in cycles_per_measurement:
                    for phase in np.arange(0, 2*np.pi, 2*np.pi/num_phases):
                        which_measurement += 1
                        self.voltages[
                            which_measurement,
                            self.prevoltage_pixels:
                            self.prevoltage_pixels + self.measurement_pixels,
                            n2c['laser']
                            ] = (modulation_offset +
                                 modulation_amplitude * (
                                     np.cos(phase + cycles*2*np.pi*t)))
            else:
                raise UserWarning("Illumination not understood.")
        else:
            self.voltages[
                :,
                self.prevoltage_pixels:
                self.prevoltage_pixels + self.measurement_pixels,
                n2c['laser']
                ] = self.illumination
        return None

    def _calibrate_laser(self):
        if not os.path.exists('shaker_calibration.tif'):
            # Define simple mirror and laser voltages. The mirror sweeps
            # left-to-right, hopefully at constant speed. The modulator
            # turns on during (roughly) the middle of this sweep, at a
            # constant voltage. This voltage increases linearly over a
            # series of measurements.
            # The mirror voltage is easy:
            measurement_pixels = 80000
            mirror_voltage = np.linspace(0, 2, 3*measurement_pixels)
            # The modulator voltage is slightly trickier:
            desired_num_illuminations = 45
            num_snaps = max(1, int(np.round(desired_num_illuminations /
                                            self.idp.buffer_shape[0])))
            num_illuminations = num_snaps * self.idp.buffer_shape[0]
            modulator_max_voltage = 0.6
            modulator_voltage = np.linspace(
                0, modulator_max_voltage, num_illuminations)
            illuminations = (
                modulator_voltage.reshape(num_illuminations, 1) *
                np.ones(measurement_pixels).reshape(1, measurement_pixels)
                ).reshape(num_snaps,
                          self.idp.buffer_shape[0],
                          measurement_pixels)
            for s in range(num_snaps):
                self.snap_mirror_motion(
                    mirror_voltage,
                    measurement_start_pixel=(mirror_voltage.shape[0] -
                                             measurement_pixels) // 2,
                    measurement_pixels=measurement_pixels,
                    filename='calibration%i.tif'%s,
                    illumination=illuminations[s, :, :])
            data = []
            for s in range(num_snaps):
                data.append(np_tif.tif_to_array('calibration%i.tif'%s
                                                ).astype(np.float32))
                os.remove('calibration%i.tif'%s)
            data = np.concatenate(data, axis=0) # Lazy but effective
            variation = filters.median_filter(data.std(axis=0)**2 /
                                              data.mean(axis=0),
                                              size=3)
            mask = variation > 0.4 * variation.max()
            calibration = (data * mask.reshape((1,) + mask.shape)
                    ).sum(axis=-1).sum(axis=-1)
            calibration -= calibration[0]
            np_tif.array_to_tif(np.array([modulator_voltage,
                                          calibration]),
                                'shaker_calibration.tif')
        calibration = np_tif.tif_to_array('shaker_calibration.tif'
                                          ).astype(np.float64)
        self.laser_calibration = {
            'modulator_voltage': calibration[0, 0, :],
            'illumination_brightness': filters.gaussian_filter(
                calibration[0, 1, :], sigma=0.5)}
        return None

    def _voltage_to_brightness(self, voltage):
        voltage = np.asarray(voltage)
        assert voltage.min() >= 0
        assert voltage.max() <= 0.5
        assert hasattr(self, 'laser_calibration')
        return np.interp(
            voltage,
            xp=self.laser_calibration['modulator_voltage'],
            fp=self.laser_calibration['illumination_brightness'])

if __name__ == '__main__':
    logger = mp.log_to_stderr()
    logger.setLevel(logging.INFO)

    t = np.arange(5000)
    num_cycles = 4
    mirror_voltage = 1 * np.sin(num_cycles * 2*np.pi * t / t.shape[0])

    shaker = Mirror_Shaker()
##    shaker.snap_mirror_motion(
##        mirror_voltage,
##        measurement_start_pixel=0,
##        measurement_pixels=5000,
##        filename='test.tif')
##    input("Hit enter...")
    shaker.close()

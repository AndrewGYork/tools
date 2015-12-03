import queue as Q
import ctypes as C
import multiprocessing as mp #Careful about importing! See comment below.
from time import sleep
from time import perf_counter as clock
import numpy as np
"""
import pco # The "Camera" child process may try to make this import
import pyglet # The "Display" child process will try to make these imports
from arrayimage import ArrayInterfaceImage
from scipy import ndimage
import np_tif # The "File Saving" child process will make this import
"""

"""
Acquiring and displaying data from a camera is a common problem our lab
has to solve. This module provides a common framework for parallel
acquisition, display, and saving at their own paces, without enforced
synchronization.

                          ****** Careful! ******
image_data_pipeline.py uses python's multiprocessing module. This means
that any code that imports image_data_pipeline.py should be written with
multiprocessing in mind:
https://docs.python.org/3.4/library/multiprocessing.html#programming-guidelines
For example, your executing code should live inside one of these:
    if __name__ == '__main__':
(which is lame), and you should start your code with an incantation like:
    import multiprocessing as mp
    import logging
    logger = mp.log_to_stderr()
    logger.setLevel(logging.INFO)
...so that all the 'info' and 'debug' statments in
image_data_pipeline.py will work right.
"""
log = mp.get_logger()
info = log.info #Like a 'high priority' print statement
debug = log.debug #Like a 'low priority' print statement

class Image_Data_Pipeline:
    def __init__(
        self,
        num_buffers=10,
        buffer_shape=(60, 256, 512),
        camera_child_process='dummy',
        ):
        """
        Allocate a bunch of 16-bit buffers for image data, and a few
        8-bit buffers for display data.
        """
        self.buffer_shape = buffer_shape
        self.buffer_size = int(np.prod(buffer_shape))
        self.num_data_buffers = num_buffers        
        self.data_buffers = [mp.Array(C.c_uint16, self.buffer_size)
                             for _ in range(self.num_data_buffers)]
        self.idle_data_buffers = range(self.num_data_buffers)
        self.accumulation_buffers = [mp.Array(C.c_uint16, self.buffer_size)
                                     for _ in range(2)]
        projection_buffer_size = int(np.prod(buffer_shape[1:]))
        self.projection_buffers = [mp.Array(C.c_uint16, projection_buffer_size)
                                   for _ in range(2)]
        """
        Launch the child processes that make up the pipeline
        """
        self.camera = Data_Pipeline_Camera(
            data_buffers=self.data_buffers,
            buffer_shape=self.buffer_shape,
            camera_child_process=camera_child_process)
        self.accumulation = Data_Pipeline_Accumulation(
            data_buffers=self.data_buffers,
            buffer_shape=self.buffer_shape,
            accumulation_buffers=self.accumulation_buffers,
            input_queue=self.camera.output_queue)
        self.file_saving = Data_Pipeline_File_Saving(
            data_buffers=self.data_buffers,
            buffer_shape=self.buffer_shape,
            input_queue=self.accumulation.output_queue)
        """
        These processes are downstream of the accumulation process, but
        not in the same loop as the camera or file saving processes.
        """
        self.projection = Data_Pipeline_Projection(
            buffer_shape=self.buffer_shape,
            projection_buffers=self.projection_buffers,
            accumulation_buffers=self.accumulation_buffers,
            accumulation_buffer_input_queue=(
                self.accumulation.accumulation_buffer_output_queue),
            accumulation_buffer_output_queue=(
                self.accumulation.accumulation_buffer_input_queue))
        self.display = Data_Pipeline_Display(
            projection_buffers=self.projection_buffers,
            buffer_shape=self.buffer_shape,
            projection_buffer_input_queue=(
                self.projection.projection_buffer_output_queue),
            projection_buffer_output_queue=(
                self.projection.projection_buffer_input_queue))
        return None
    
##    def load_data_buffers(
##        self, N, file_saving_info=None, collect_buffers=True, timeout=0):
##        """
##        'file_saving_info' is None, or a list of dicts. Each dict is a
##        set of arguments to simple_tif.array_to_tif().
##        """
##        if file_saving_info is not None:
##            if len(file_saving_info) != N:
##                raise UserWarning(
##                    "If file saving info is provided, it must match the number" +
##                    " of buffers loaded.")
##        """
##        Feed the pipe!
##        """
##        for i in range(N):
##            """
##            Get an idle buffer
##            """
##            for tries in range(10):
##                try:
##                    idle_buffer = self.idle_data_buffers.pop(0)
##                    break
##                except IndexError:
##                    if collect_buffers:
##                        if tries > 0:
##                            sleep(timeout * 0.1)                            
##                        self.collect_data_buffers()
##            else:
##                raise UserWarning("Timeout exceeded, no buffer available")
##            """
##            Load the buffer into the queue, along with file saving
##            info if appropriate
##            """
##            permission_slip = {'which_buffer': idle_buffer}
##            if file_saving_info is not None:
##                permission_slip['file_info'] = file_saving_info.pop(0)
##            self.camera.input_queue.put(permission_slip)
##        return None
##
##    def collect_data_buffers(self):
##        while True:
##            try:
##                strip_me = self.file_saving.output_queue.get_nowait()
##            except Queue.Empty:
##                break
##            self.idle_data_buffers.append(strip_me['which_buffer'])
##            info("Buffer %i idle"%(self.idle_data_buffers[-1]))
##        return None
##    
##    def set_buffer_shape(self, buffer_shape):
##        """
##        You don't have to use the whole buffer! If you want to make a
##        buffer with more pixels than the original one, though, you
##        should make a new Image_Data_Pipeline.
##        """
##        assert len(buffer_shape) == 3
##        for s in buffer_shape:
##            assert s > 0
##        if np.prod(buffer_shape) > self.buffer_size:
##            raise UserWarning("If you want a buffer larger than the original" +
##                              " buffer size, close this Image_Data_Pipeline" +
##                              " and make a new one.")
##        self.buffer_shape = buffer_shape
##        while len(self.idle_data_buffers) < len(self.data_buffers):
##            self.collect_data_buffers()
##            sleep(0.01)
##        for p in (self.camera,
##                  self.accumulation,
##                  self.file_saving,
##                  self.projection,
##                  self.display):
##            p.commands.send(('set_buffer_shape', {'shape': buffer_shape}))
##            while True:
##                if p.commands.poll():
##                    p.commands.recv()
##                    break
##        return None
##    
##    def set_display_intensity_scaling(
##        self, scaling, display_min=None, display_max=None):
##        args = locals()
##        args.pop('self')
##        self.display.commands.send(('set_intensity_scaling', args))
##        return self.display.commands.recv()
##
##    def get_display_intensity_scaling(self):
##        return self.display.get_intensity_scaling()
##
##    def withdraw_display(self):
##        self.display.commands.send(('withdraw', {}))
##
##    def check_children(self):
##        return {'Camera': self.camera.child.is_alive(),
##                'Accumulation': self.accumulation.child.is_alive(),
##                'File Saving': self.file_saving.child.is_alive(),
##                'Projection': self.projection.child.is_alive(),
##                'Display': self.display.child.is_alive()}
##
##    def close(self):
##        self.camera.input_queue.put(None)
##        self.accumulation.input_queue.put(None)
##        self.file_saving.input_queue.put(None)
##        self.projection.display_buffer_input_queue.put(None)
##        self.projection.accumulation_buffer_input_queue.put(None)
##        self.display.display_buffer_input_queue.put(None)
##        self.camera.child.join()
##        self.accumulation.child.join()
##        self.file_saving.child.join()
##        self.projection.child.join()
##        self.display.child.join()
##        return None

class Data_Pipeline_Camera:
    def __init__(
        self,
        data_buffers,
        buffer_shape,
        input_queue=None,
        output_queue=None,
        camera_child_process='dummy',
        ):
        if input_queue is None:
            self.input_queue = mp.Queue()
        else:
            self.input_queue = input_queue
        if output_queue is None:
            self.output_queue = mp.Queue()
        else:
            self.output_queue = output_queue
        self.commands, self.child_commands = mp.Pipe()
        if camera_child_process is 'dummy':
            camera_child_process = dummy_camera_child_process
        elif camera_child_process is 'pco':
            camera_child_process = pco_edge_camera_child_process
        self.child = mp.Process(
            target=camera_child_process,
            args=(data_buffers, buffer_shape,
                  self.input_queue, self.output_queue,
                  self.child_commands),
            name='Camera')
        self.child.start()
        return None

def dummy_camera_child_process(
    data_buffers,
    buffer_shape,
    input_queue,
    output_queue,
    commands,
    ):
    """
    If you want to debug image_data_pipeline but you don't have a
    camera hooked up to the system, it's nice to have a 'dummy'
    process that pretends to be a camera and copies fake data into
    the appropriate input buffer. For educational purposes, this is
    lavishly commented, and hopefully illustrates the skeleton of
    how our child processes work.
    """
    info("Using dummy camera process, not a real camera.")
    buffer_size = np.prod(buffer_shape)
    fake_data = [np.zeros(buffer_size, dtype=np.uint16)
                 for i in data_buffers]
    for i, d in enumerate(fake_data):
        d.fill(int((2**16 - 1) * (i + 1.0) / len(fake_data)))
    data_idx = -1
    while True:
        """
        Respond to commands until we've emptied the command pipe.
        """
        if commands.poll():
            cmd, args = commands.recv()
            commands.send(None)
            if cmd == 'set_buffer_shape':
                buffer_shape = args['shape']
                buffer_size = np.prod(buffer_shape)
            continue
        """
        The command pipe is empty; check the input queue for
        permission slips.
        """
        try:
            permission_slip = input_queue.get_nowait()
        except Q.Empty:
            """
            Nothing in the command pipe, nothing in the input queue.
            Nothing to do! Start over, but after a small delay to
            avoid burning too much CPU.
            """
            sleep(0.001) #Semi-random sleep time :(
            continue
        """
        The command pipe was empty, and we've got a permission slip.
        Copy some fake data into the relevant data buffer to
        simulate a camera.
        """
        if permission_slip is None: #This is how we signal "shut down"
            output_queue.put(permission_slip)
            break #We're done
        else:
            """
            The permission slip hopefully refers appropriately to a
            data buffer. Fill that buffer with some fake data.
            """
            process_me = permission_slip['which_buffer']
            info("start buffer %i"%(process_me))
            with data_buffers[process_me].get_lock():
                """
                In this code block, we've locked the relevant
                buffer, and we'll automatically release it when
                we're done copying fake data into it.

                This incantation lets us treat a multiprocessing
                array like a numpy array:
                """
                a = np.frombuffer(data_buffers[process_me].get_obj(),
                                  dtype=np.uint16)[:buffer_size
                                                   ].reshape(buffer_shape)
                """
                Now we copy our fake data into the data buffer:
                """
                data_idx += 1
                data_idx = data_idx % len(fake_data)
                a[:, :, :] = fake_data[data_idx][:buffer_size
                                                 ].reshape(buffer_shape)
            """
            We're done copying fake data into the buffer. Wait a
            silly amount of time (to act a little more like a real
            camera), then pass the permission slip to the next guy:
            """
            sleep(0.010) #It'd be nice if this was 10 ms but it ain't
            info("end buffer %i"%(process_me))
            output_queue.put(permission_slip)
    return None

def pco_edge_camera_child_process(
    data_buffers,
    buffer_shape,
    input_queue,
    output_queue,
    commands,
    pco_edge_type='4.2' #Change this if you're using a 5.5
    ):
    """
    For the pco.edge camera. Debugged for the 4.2, but might work for
    the 5.5, with some TLC...
    """
    buffer_size = np.prod(buffer_shape)
    try:
        import pco
    except ImportError:
        info("Failed to import pco.py; go get it from github:")
        info("https://github.com/AndrewGYork/tools/blob/master/pco.py")
        raise
    camera = pco.Edge(pco_edge_type=pco_edge_type)
    camera.apply_settings(trigger='auto_trigger')
    camera.arm(num_buffers=3)
    preframes = 3
    status = 'Normal'
    while True:
        if commands.poll():
            cmd, args = commands.recv()
##            if cmd == 'set_buffer_shape':
##                buffer_shape = args['shape']
##                buffer_size = np.prod(buffer_shape)
##                commands.send(buffer_shape)
##            elif cmd == 'apply_settings':
##                settings = camera.apply_settings(**args)
##                camera.arm()
##                camera._prepare_to_record_to_memory()
##                commands.send(settings)
##            elif cmd == 'get_settings':
##                commands.send(camera.get_settings(**args))
##            elif cmd == 'get_status':
##                commands.send(status)
##            elif cmd == 'reset_status':
##                status = 'Normal'
##                commands.send(None)
##            elif cmd == 'get_preframes':
##                commands.send(preframes)
##            elif cmd == 'set_preframes':
##                preframes = args['preframes']
##                commands.send(preframes)
##            elif cmd == 'get_timeout':
##                commands.send(timeout)
##            elif cmd == 'set_timeout':
##                timeout = args['timeout']
##                commands.send(timeout)
            continue #ignore commands we don't recognize. Bad idea?
        try:
            permission_slip = input_queue.get_nowait()
        except Q.Empty:
            sleep(0.001) #Non-deterministic sleep time :(
            continue
        if permission_slip is None: #This is how we signal "shut down"
            output_queue.put(permission_slip)
            break #We're done
        else:
            """
            Fill the data buffer with images from the camera
            """
            time_received = clock()
            process_me = permission_slip['which_buffer']
            info("start buffer %i"%(process_me))
            with data_buffers[process_me].get_lock():
                a = np.frombuffer(data_buffers[process_me].get_obj(),
                                  dtype=np.uint16)[:buffer_size
                                                   ].reshape(buffer_shape)
                info('Start acquiring...')
                try:
                    camera.record_to_memory(
                        num_images=a.shape[0] + preframes,
                        preframes=preframes,
                        out=a)
                except pco.TimeoutError as e:
                    info('TimeoutError, %i acquired'%(e.num_acquired))
                    status = 'TimeoutError'
                    #FIXME: we can do better, probably. Keep trying?
                    #Should we zero the remainder of 'a'?
                except pco.DMAError:
                    info('DMAError')
                    status = 'DMAError'
                else:
                    status = 'Normal'
                info('Done acquiring: %06f seconds elapsed'%(
                    clock() - time_received))
            info("end buffer %i"%(process_me))
            output_queue.put(permission_slip)
    camera.close()
    return None    
    

class Data_Pipeline_Accumulation:
    def __init__(
        self,
        data_buffers,
        buffer_shape,
        accumulation_buffers,
        input_queue=None,
        output_queue=None,
        ):
        if input_queue is None:
            self.input_queue = mp.Queue()
        else:
            self.input_queue = input_queue
        if output_queue is None:
            self.output_queue = mp.Queue()
        else:
            self.output_queue = output_queue
        self.commands, self.child_commands = mp.Pipe()
        self.accumulation_buffer_input_queue = mp.Queue()
        self.accumulation_buffer_output_queue = mp.Queue()
        self.child = mp.Process(
            target=accumulation_child_process,
            args=(data_buffers, buffer_shape, accumulation_buffers,
                  self.input_queue, self.output_queue, self.child_commands,
                  self.accumulation_buffer_input_queue,
                  self.accumulation_buffer_output_queue),
            name='Accumulation')
        self.child.start()
        return None

def accumulation_child_process(
    data_buffers,
    buffer_shape,
    accumulation_buffers,
    data_buffer_input_queue,
    data_buffer_output_queue,
    commands,
    accumulation_buffer_input_queue,
    accumulation_buffer_output_queue,
    ):
    """
    The camera process might produce buffers far too fast for the
    display process to keep up. We'd like to make sure every frame of
    the accumulation buffer has a chance to put a pixel on the screen,
    so our data buffers come too fast, we mush multiple data buffers
    into a single "accumulation" buffer.
    """
    buffer_size = np.prod(buffer_shape)
    num_accumulated = 0
    current_accumulation_buffer = 0 #Buffer 0 is ready to go
    accumulation_buffer_input_queue.put(1) #Buffer 1 is next up to bat.
    accumulation_buffer_occupied = False
    while True:
        if commands.poll():
            cmd, args = commands.recv()
            if cmd == 'set_buffer_shape':
                buffer_shape = args['shape']
                buffer_size = np.prod(buffer_shape)
                commands.send(buffer_shape)
            continue
        if accumulation_buffer_occupied: #We'd prefer to copy to a fresh buffer
            try: #Check if a fresh, empty accumulation buffer is available
                switch_to_me = accumulation_buffer_input_queue.get_nowait()
            except Q.Empty: #No luck.
                pass  #Keep accumulating to the current buffer :(
            else: #We got one! Switch to using the fresh accumulation buffer
                accumulation_buffer_output_queue.put(
                    current_accumulation_buffer)
                current_accumulation_buffer = switch_to_me
                info("Sending accumulation buffer with %i timepoint(s)"%(
                    num_accumulated))
                accumulation_buffer_occupied = False
                num_accumulated = 0
        try: #Check for a pending data buffer
            permission_slip = data_buffer_input_queue.get_nowait()
        except Q.Empty: #Nothing pending. Back to square one.
            sleep(0.001) #Not sure how long this 1 ms sleep actually lasts
            continue
        if permission_slip is None: #Poison pill. Pass it on and quit!
            data_buffer_output_queue.put(permission_slip)
            accumulation_buffer_output_queue.put(None)
            break
        else:
            """
            The command pipe is empty, the data input queue was not. We
            tried to get a fresh accumulation buffer; if we succceeded,
            we'll copy our data buffer into it. If we failed, we'll
            max-project our data buffer into the already-filled
            accumulation buffer.
            """
            process_me = permission_slip['which_buffer']
            info("start buffer %i"%(process_me))
            with data_buffers[process_me].get_lock():
                data = np.frombuffer(
                    data_buffers[process_me].get_obj(),
                    dtype=np.uint16)[:buffer_size].reshape(buffer_shape)
                with accumulation_buffers[
                    current_accumulation_buffer].get_lock():
                    a_b = np.frombuffer(accumulation_buffers[
                        current_accumulation_buffer].get_obj(),
                        dtype=np.uint16)[:buffer_size].reshape(buffer_shape)
                    if accumulation_buffer_occupied: #Accumulate
                        np.maximum(data, a_b, out=a_b)
                    else: #First accumulation into a fresh buffer; copy.
                        a_b[:] = data
            accumulation_buffer_occupied = True
            num_accumulated += 1
            data_buffer_output_queue.put(permission_slip)
            info("end buffer %i"%(process_me))
    return None

class Data_Pipeline_Projection:
    def __init__(
        self,
        buffer_shape,
        projection_buffers,
        accumulation_buffers,
        accumulation_buffer_input_queue,
        accumulation_buffer_output_queue,
        ):
        self.accumulation_buffer_input_queue = accumulation_buffer_input_queue
        self.accumulation_buffer_output_queue = accumulation_buffer_output_queue
        self.commands, self.child_commands = mp.Pipe()
        self.projection_buffer_input_queue = mp.Queue()
        self.projection_buffer_output_queue = mp.Queue()
        self.child = mp.Process(
            target=projection_child_process,
            args=(buffer_shape, projection_buffers, accumulation_buffers,
                  self.child_commands,
                  self.projection_buffer_input_queue,
                  self.projection_buffer_output_queue,
                  self.accumulation_buffer_input_queue,
                  self.accumulation_buffer_output_queue),
            name='Projection')
        self.child.start()
        return None

def projection_child_process(
    buffer_shape,
    projection_buffers,
    accumulation_buffers,
    commands,
    projection_buffer_input_queue,
    projection_buffer_output_queue,
    accumulation_buffer_input_queue,
    accumulation_buffer_output_queue,
    ):
    """
    The 3D data buffers need to be smanged down to 2D before we can
    display them on screen. projection_child_process() takes in 3D
    accumulation buffers constructed from one or more data buffers, and
    spits out 2D projection buffers which hopefully give a half decent
    2D summary of the 3D object.
    """
    buffer_size = np.prod(buffer_shape)
    projection_buffer_size = np.prod(buffer_shape[1:])
    alive = True
    while alive:
        try: #Get a pending display buffer
            fill_me = projection_buffer_input_queue.get_nowait()
        except Q.Empty:
            sleep(0.001) #Don't trust this to be 1 ms
            continue #Don't bother with other stuff!
        info("Display buffer %i received"%(fill_me))
        """
        Code below this point knows we have exactly one projection
        buffer.
        """
        while True:
            if commands.poll():
                info("Command received")
                cmd, args = commands.recv()
                if cmd == 'set_buffer_shape':
                    buffer_shape = args['shape']
                    buffer_size = np.prod(buffer_shape)
                    projection_buffer_size = np.prod(buffer_shape[1:])
                    commands.send(buffer_shape)
                continue
            try: #Command pipe is empty; get an accumulation buffer
                project_me = accumulation_buffer_input_queue.get_nowait()
            except Q.Empty: #Nothing pending. Keep trying.
                sleep(0.001) #Again, don't trust this.
                continue
            if project_me is None: #Poison pill. Pass it on, and quit!
                projection_buffer_output_queue.put(None)
                alive = False #To break out of the while loop one level up
                break
            else:
                """
                We've got a 3D accumulation buffer and a 2D projection
                buffer, and the command pipe is empty. Project the
                accumulation buffer into the projection buffer.
                """
                info("start accumulation buffer %i"%(project_me))
                with accumulation_buffers[project_me].get_lock():
                    acc = np.frombuffer(
                        accumulation_buffers[project_me].get_obj(),
                        dtype=np.uint16)[:buffer_size].reshape(buffer_shape)
                    with projection_buffers[fill_me].get_lock():
                        proj = np.frombuffer(
                            projection_buffers[fill_me].get_obj(),
                            dtype=np.uint16)[:projection_buffer_size
                                             ].reshape(buffer_shape[1:])
                        np.amax(acc, axis=0, out=proj) #Project to 2D
                info("end accumulation buffer %i"%(project_me))
                accumulation_buffer_output_queue.put(project_me)
                info("Returning projection buffer %i"%(fill_me))
                projection_buffer_output_queue.put(fill_me)
                break #Go back and look for the next projection buffer
    return None

class Data_Pipeline_Display:
    def __init__(
        self,
        projection_buffers,
        buffer_shape,
        projection_buffer_input_queue,
        projection_buffer_output_queue,
        ):
        self.projection_buffer_input_queue = projection_buffer_input_queue
        self.projection_buffer_output_queue = projection_buffer_output_queue

        self.commands, self.child_commands = mp.Pipe()
##        self.display_intensity_scaling_queue = mp.Queue()
##        self.display_intensity_scaling = ('linear', 0, 2**16 - 1)

        self.child = mp.Process(
            target=display_child_process,
            args=(projection_buffers, buffer_shape,
                  self.projection_buffer_input_queue,
                  self.projection_buffer_output_queue,
                  self.child_commands,
                  #self.display_intensity_scaling_queue,
                  ),
            name='Display')
        self.child.start()
        return None

##    def get_intensity_scaling(self):
##        try:
##            self.display_intensity_scaling = (
##                self.display_intensity_scaling_queue.get_nowait())
##        except Queue.Empty:
##            pass
##        return self.display_intensity_scaling

def display_child_process(
    projection_buffers,
    buffer_shape,
    input_queue,
    output_queue,
    commands,
    #display_intensity_scaling_queue,
    ):
    args = locals()
    display = Display(**args)
    display.run()
    return None

class Display:
    def __init__(
        self,
        projection_buffers,
        buffer_shape,
        input_queue,
        output_queue,
        commands,
        #display_intensity_scaling_queue,
        ):
        import pyglet
        self.pyg = pyglet
        try:
            from arrayimage import ArrayInterfaceImage
        except ImportError:
            info("'arrayimage' not found. Go get it from:")
            info('https://github.com/AndrewGYork/tools/blob/master/arrayimage.py')
            info("or possibly from:")
            info("https://github.com/motmot/pygarrayimage/" +
                  "blob/master/pygarrayimage/arrayimage.py")
            raise
        self._array_to_image = ArrayInterfaceImage
        try:
            from scipy import ndimage #Median filtering autoscale, noncrucial
            self.ndimage = ndimage
        except ImportError:
            self.ndimage = None

        self.projection_buffers = projection_buffers
        self.buffer_shape = buffer_shape
        self.projection_buffer_size = np.prod(buffer_shape[1:])
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.commands = commands
        #self.display_intensity_scaling_queue = display_intensity_scaling_queue
        #self.set_intensity_scaling('linear', display_min=0, display_max=2**16-1)
        self.projection_buffers[1].get_lock().acquire()
        self.current_projection_buffer = 1
        self.switch_buffers(0)
        self.display_data = np.empty(self.buffer_shape[1:], dtype=np.uint8)
        self.display_min, self.display_max = 50, 150 #FIXME (temporary)
        self._make_linear_lookup_table() #FIXME (temporary)
        self.convert_to_8_bit()
        self.make_window()
        update_interval_seconds = 0.025
        self.pyg.clock.schedule_interval(self.update, update_interval_seconds)
        return None

    def run(self):
##        """
##        Eventually put code here to deal with closing and re-opening the window.
##        """
        self.pyg.app.run()
        return None

    def quit(self):
        self.pyg.app.exit()
        return None

    def update(self, dt):
        if self.commands.poll():
            self.execute_external_command()
            return None
        try:
            switch_to_me = self.input_queue.get_nowait()
        except Q.Empty:
            return None
        if switch_to_me is None: #Poison pill. Quit!
            self.quit()
        else:
            self.switch_buffers(switch_to_me)
            self.convert_to_8_bit()
        return None

    def make_window(self):
        screen_width, screen_height = self._get_screen_dimensions()

        self.window = self.pyg.window.Window(
            min(screen_width//2, screen_height),
            min(screen_width//2, screen_height),
            caption='Display',
            resizable=True)
##        self.window.set_location(int((screen_width * 0.95) // 2),
##                                 screen_height//20)
##        self.default_image_scale = min(
##            (screen_width//2) * 1.0 / self.image.width,
##            screen_height * 1.0 / self.image.height)
##        self.image_scale = self.default_image_scale
        self.image_x, self.image_y = 0, 0
        @self.window.event
        def on_draw():
            self.window.clear()
            self.image.blit(
                x=self.image_x,
                y=self.image_y,
##                height=int(self.image.height * self.image_scale),
##                width=int(self.image.width * self.image_scale)
                )

##        """
##        Allow the user to pan and zoom the image
##        """
##        @self.window.event
##        def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
##            if buttons == self.pyg.window.mouse.LEFT:
##                self.image_x += dx
##                self.image_y += dy
##            self._enforce_panning_limits()
##
##        @self.window.event
##        def on_mouse_scroll(x, y, scroll_x, scroll_y):
##            old_image_scale = self.image_scale
##            self.image_scale *= 1.3**(scroll_y)
##            """
##            No sense letting the user make the image underfill the window
##            """
##            while (self.image.width * self.image_scale < self.window.width and
##                   self.image.height * self.image_scale < self.window.height):
##                self.image_scale = min(
##                    self.window.width * 1.0 / self.image.width,
##                    self.window.height * 1.0 / self.image.height)
##            """
##            Might as well set some sane zoom limits, too.
##            """
##            if self.image_scale < 0.01:
##                self.image_scale = 0.01
##            if self.image_scale > 300:
##                self.image_scale = 300
##            """
##            Center the origin of the zoom on the mouse coordinate.
##            This was kinda thinky to figure out, don't fuck with this lightly.
##            """
##            zoom = self.image_scale * 1.0 / old_image_scale
##            self.image_x = self.image_x * zoom + x * (1 - zoom)
##            self.image_y = self.image_y * zoom + y * (1 - zoom)
##            self._enforce_panning_limits()
##        
##        @self.window.event
##        def on_mouse_release(x, y, button, modifiers):
##            self.last_mouse_release = (x, y, button, clock())
##
##        @self.window.event
##        def on_mouse_press(x, y, button, modifiers):
##            if hasattr(self, 'last_mouse_release'):
##                if (x, y, button) == self.last_mouse_release[:-1]:
##                    """Same place, same button"""
##                    if clock() - self.last_mouse_release[-1] < 0.2:
##                        """We got ourselves a double-click"""
##                        self.image_scale = self.default_image_scale
##                        self.image_x = 0
##                        self.image_y = 0
##                        w, h = self._get_screen_dimensions()
##                        edge_length = min(w//2, h)
##                        self.window.width = edge_length
##                        self.window.height = edge_length            
##        """
##        We don't want 'escape' or 'quit' to quit the pyglet
##        application, just withdraw it. The parent application should
##        control when pyglet quits.
##        """
##        @self.window.event
##        def on_key_press(symbol, modifiers):
##            if symbol == self.pyg.window.key.ESCAPE:
##                self.window.set_visible(False)
##                return self.pyg.event.EVENT_HANDLED
##        @self.window.event
##        def on_close():
##            self.window.set_visible(False)
##            return self.pyg.event.EVENT_HANDLED

    def execute_external_command(self):
        """
        The command should be a 2-tuple. The first element of the
        tuple is a string naming the command. The second element of
        the tuple is a dict of arguments to the command.
        """
        cmd, args = self.commands.recv()
##        if cmd == 'set_intensity_scaling':
##            response = self.set_intensity_scaling(**args)
##            self.commands.send(response)
##        elif cmd == 'get_intensity_scaling':
##            self.commands.send((self.intensity_scaling,
##                                self.display_min,
##                                self.display_max))
##        elif cmd == 'set_buffer_shape':
##            self.buffer_shape = args['shape']
##            self.display_buffer_size = np.prod(self.buffer_shape[1:])
##            if hasattr(self, 'display_data_16'):
##                self.display_data_16 = self.display_data_16[
##                    :self.buffer_shape[1],
##                    :self.buffer_shape[2]]
##            if hasattr(self, 'display_data_8'):
##                self.display_data_8 = np.empty(self.buffer_shape[1:],
##                                               dtype=np.uint8)
##            np.take(self.lut, self.display_data_16, out=self.display_data_8)
##            self.image = self._array_to_image(self.display_data_8,
##                                             allow_copy=False)
##            self.pyg.gl.glTexParameteri( #Reset to no interpolation
##                self.pyg.gl.GL_TEXTURE_2D,
##                self.pyg.gl.GL_TEXTURE_MAG_FILTER,
##                self.pyg.gl.GL_NEAREST)
##            self.window.set_size(self.image.width, self.image.height)
##            self.commands.send(self.buffer_shape)
##        elif cmd == 'withdraw':
##            self.window.set_visible(False)
##        else:
##            raise UserWarning("Command not recognized: %s, %s"%(
##                repr(cmd), repr(args)))
        return None

    def switch_buffers(self, switch_to_me):
        """
        Lock the new buffer, give up the old one.
        """
        info("Projection buffer %i received"%(switch_to_me))
        self.projection_buffers[switch_to_me].get_lock().acquire()
        self.projection_buffers[self.current_projection_buffer
                                ].get_lock().release()
        self.output_queue.put(int(self.current_projection_buffer))
        info("Projection buffer %i loaded to projection process"%(
            self.current_projection_buffer))
        self.current_projection_buffer = int(switch_to_me)
        self.projection_data = np.frombuffer(
            self.projection_buffers[self.current_projection_buffer].get_obj(),
            dtype=np.uint16)[:self.projection_buffer_size
                             ].reshape(self.buffer_shape[1:])
        return None

    def convert_to_8_bit(self):
        """
        Convert 16-bit projections to 8-bit display data using a lookup table.
        """
##        if self.intensity_scaling == 'autoscale':
##            self.display_min = self.display_data_16.min()
##            self.display_max = self.display_data_16.max()
##            self._make_linear_lookup_table()
##        elif self.intensity_scaling == 'median_filter_autoscale':
##            filtered_image = self.ndimage.filters.median_filter(
##                self.display_data_16, size=3, output=self.filtered_image)
##            self.display_min = self.filtered_image.min()
##            self.display_max = self.filtered_image.max()
##            self._make_linear_lookup_table()
        np.take(self.lut, self.projection_data, out=self.display_data)
        print(self._lut_start.min(), self._lut_start.max())
        print(self._lut_intermediate.min(), self._lut_intermediate.max())
        print(self.lut.min(), self.lut.max())
        print(self.projection_data.min(), self.projection_data.max())
        print(self.display_data.min(), self.display_data.max())
##        try:
##            self.display_intensity_scaling_queue.get_nowait()
##        except Queue.Empty:
##            pass
##        self.display_intensity_scaling_queue.put(
##            (self.intensity_scaling, self.display_min, self.display_max))
        self.image = self._array_to_image(self.display_data, allow_copy=False)
##        self.pyg.gl.glTexParameteri( #Reset to no interpolation
##                self.pyg.gl.GL_TEXTURE_2D,
##                self.pyg.gl.GL_TEXTURE_MAG_FILTER,
##                self.pyg.gl.GL_NEAREST)
##        if hasattr(self, 'window'):
##            if not self.window.visible:
##                self.window.set_visible(True)
        return None

##    def set_intensity_scaling(self, scaling, display_min, display_max):
##        if scaling == 'linear':
##            self.intensity_scaling = 'linear'
##            if display_min is not None and display_max is not None:
##                if display_min < 0:
##                    display_min = 0
##                if display_min > (2**16 - 2):
##                    display_min = (2**16 - 2)
##                if display_max <= display_min:
##                    display_max = display_min + 1
##                if display_max > (2**16 - 1):
##                    display_max = 2**16 - 1
##                self.display_min = display_min
##                self.display_max = display_max
##                self._make_linear_lookup_table()
##        elif scaling == 'autoscale':
##            self.intensity_scaling = 'autoscale'
##            self.display_min = self.display_data_16.min()
##            self.display_max = self.display_data_16.max()
##            self._make_linear_lookup_table()
##        elif scaling == 'median_filter_autoscale':
##            if self.ndimage is None:
##                info("Median filter autoscale requires Scipy. " +
##                     "Using min/max autoscale.")
##                self.intensity_scaling = 'autoscale'
##                self.display_min = self.display_data_16.min()
##                self.display_max = self.display_data_16.max()
##            else:
##                self.intensity_scaling = 'median_filter_autoscale'
##                if not hasattr(self, 'filtered_image'):
##                    self.filtered_image = np.empty(
##                        self.buffer_shape[1:], dtype=np.uint16)
##                filtered_image = self.ndimage.filters.median_filter(
##                    self.display_data_16, size=3, output=self.filtered_image)
##                self.display_min = self.filtered_image.min()
##                self.display_max = self.filtered_image.max()
##            self._make_linear_lookup_table()
##        else:
##            raise UserWarning("Scaling not recognized:, %s"%(repr(scaling)))
##        if hasattr(self, 'display_data_16'):
##            self.convert_to_8_bit()
##        return scaling, display_min, display_max
    
    def _make_linear_lookup_table(self):
        """
        Waaaaay faster than how I was doing it before.
        http://stackoverflow.com/q/14464449/513688
        """
        if not hasattr(self, '_lut_start'):
            self._lut_start = np.arange(2**16, dtype=np.uint16)
        if not hasattr(self, '_lut_intermediate'):
            self._lut_intermediate = self._lut_start.copy()
        if not hasattr(self, 'lut'):
            self.lut = np.empty(2**16, dtype=np.uint8)
        np.clip(self._lut_start, self.display_min, self.display_max,
                out=self._lut_intermediate)
        self._lut_intermediate -= self.display_min
        self._lut_intermediate //= (
            self.display_max - self.display_min + 1.) / 256.
        self.lut[:] = self._lut_intermediate.view(np.uint8)[::2] #Too sneaky?
        return None

    def _get_screen_dimensions(self):
        plat = self.pyg.window.Platform()
        disp = plat.get_default_display()
        screen = disp.get_default_screen()
        return screen.width, screen.height

##    def _enforce_panning_limits(self):
##        if self.image_x < (self.window.width -
##                           self.image.width*self.image_scale):
##            self.image_x = (self.window.width -
##                            self.image.width*self.image_scale)
##        if self.image_y < (self.window.height -
##                           self.image.height*self.image_scale):
##            self.image_y = (self.window.height -
##                            self.image.height*self.image_scale)
##        if self.image_x > 0:
##            self.image_x = 0
##        if self.image_y > 0:
##            self.image_y = 0
##        return None

class Data_Pipeline_File_Saving:
    def __init__(
        self,
        data_buffers,
        buffer_shape,
        input_queue=None,
        output_queue=None,
        ):
        if input_queue is None:
            self.input_queue = mp.Queue()
        else:
            self.input_queue = input_queue
        if output_queue is None:
            self.output_queue = mp.Queue()
        else:
            self.output_queue = output_queue
        self.commands, self.child_commands = mp.Pipe()
        self.child = mp.Process(
            target=file_saving_child_process,
            args=(data_buffers, buffer_shape,
                  self.input_queue, self.output_queue, self.child_commands),
            name='File Saving')
        self.child.start()
        return None

def file_saving_child_process(
    data_buffers,
    buffer_shape,
    input_queue,
    output_queue,
    commands,
    ):
    try:
        import np_tif
    except ImportError:
        info("Failed to import np_tif.py; go get it from github:")
        info("https://github.com/AndrewGYork/tools/blob/master/np_tif.py")
        raise
    buffer_size = np.prod(buffer_shape)
    while True:
        if commands.poll():
            cmd, args = commands.recv()
            if cmd == 'set_buffer_shape':
                buffer_shape = args['shape']
                buffer_size = np.prod(buffer_shape)
                commands.send(buffer_shape)
            continue
        try:
            permission_slip = input_queue.get_nowait()
        except Q.Empty:
            sleep(0.001) #Probably doesn't sleep for 1 ms :(
            continue
        if permission_slip is None: #Poison pill! Pass it on, then quit.
            output_queue.put(permission_slip)
            break
        else:
            process_me = permission_slip['which_buffer']
            info("start buffer %i"%(process_me))
            if 'file_info' in permission_slip:
                """
                We only save the data buffer to a file if we have 'file
                information' in the permission slip. The value
                associated with the 'file_info' key is a dict of
                arguments to pass to np_tif.array_to_tif(), specifying
                things like the file name.
                """
                info("saving buffer %i"%(process_me))
                """
                Save the buffer to disk as a TIF
                """
                file_info = permission_slip['file_info']
                with data_buffers[process_me].get_lock():
                    a = np.frombuffer(data_buffers[process_me].get_obj(),
                                      dtype=np.uint16)[:buffer_size
                                                       ].reshape(buffer_shape)
                    np_tif.array_to_tif(a, **file_info)
            info("end buffer %i"%(process_me))
            output_queue.put(permission_slip)
    return None

if __name__ == '__main__':
    import multiprocessing as mp
    import logging
    logger = mp.log_to_stderr()
    logger.setLevel(logging.INFO)

    idp = Image_Data_Pipeline(
        num_buffers=5,
        buffer_shape=(1, 2048, 2060),
        camera_child_process='pco')
##    idp.camera.commands.send(
##        ('apply_settings',
##         {'trigger': 'auto trigger',
##          'region_of_interest': (-1, -1, 10000, 10000),
##          'exposure_time_microseconds': 10000}))
##    idp.set_display_intensity_scaling(scaling='autoscale')
##    print idp.camera.commands.recv()
    while True:
        try:
##            print idp.check_children()
##            idp.collect_data_buffers()
##            idp.load_data_buffers(len(idp.idle_data_buffers))
            input()
            idp.camera.input_queue.put({'which_buffer': 0,
##                                        'file_info': {'filename': 'test.tif'}
                                        })
####            idp.set_buffer_shape((idp.buffer_shape[0],
####                                  idp.buffer_shape[1],
####                                  idp.buffer_shape[2] - 10))
        except KeyboardInterrupt:
            break
##    idp.close()

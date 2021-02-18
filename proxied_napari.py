#!/usr/bin/python3
# Printing from a child process is tricky:
import io
from contextlib import redirect_stdout
# Showing exceptions from a child process is tricky:
import traceback
# Napari/Qt stuff
import napari
from qtpy.QtCore import QTimer
# Our stuff
from proxy_objects import (
    ProxyManager, _dummy_function, _reconnect_shared_arrays, _SharedNumpyArray)

def display(proxy_manager=None, display_type=None):
    """Creates a simplified non-blocking napari viewer in a subprocess.

    If you don't know what you're doing, this is probably the only thing
    you should import from this module.

    If you're using a ProxyManager, pass it as `proxy_manager` so the
    display can use the same shared memory.

    If you're using a custom _NapariDisplay, pass it as `display_type`.
    """
    if proxy_manager is None:
        proxy_manager = ProxyManager()
    if display_type is None:
        display_type = _NapariDisplay
    display = proxy_manager.proxy_object(display_type,
                                         custom_loop=_napari_child_loop,
                                         close_method_name='close')
    return display

class _NapariDisplay:
    """This is a barebones example of a proxiable napari viewer.

    The idea is to expose a subset of napari's rich, deep API as a shallow,
    simple API that proxy_objects can handle.

    We encourage you to edit this class to suit your purposes.
    The only requirement is that it has a `close` method.
    """
    def __init__(self):
        self.viewer = napari.Viewer()

    def show_image(self, im):
        if not hasattr(self, 'image'):
            self.image = self.viewer.add_image(im)
        else:
            self.image.data = im

    def close(self):
        self.viewer.close()

# We're pretty confident that you shouldn't have to understand anything below
# this. If you want to extend the functionality of the proxied napari viewer
# this can probably be done by creating your own modified version of the
# _NapariDisplay  class above.

def _napari_child_loop(child_pipe, shared_arrays,
                       initializer, initargs, initkwargs,
                       close_method_name, closeargs, closekwargs):
    """Teach Qt's event loop how to act like a ProxyObject's child process."""

    # If any of the initargs are _SharedNumpyArrays, we have to show them where
    # to find shared memory:
    initargs, initkwargs = _reconnect_shared_arrays(
        initargs, initkwargs, shared_arrays)
    closeargs = tuple() if closeargs is None else closeargs
    closekwargs = dict() if closekwargs is None else closekwargs
    state = { # Mutable, to store the state of the child object.
        'keep_launching_napari': True,
        'initial_init': True}
    while state['keep_launching_napari']:
        # We'd ideally check that the pipe was still healthy, but we don't know
        # how to do that gracefully.
        with napari.gui_qt():
            # Initialize a napari-specific object
            printed_output = io.StringIO()
            try: # Create an instance of our napari object...
                with redirect_stdout(printed_output):
                    obj = initializer(*initargs, **initkwargs)
                    if close_method_name is not None:
                        close_method = getattr(obj, close_method_name)
                if state['initial_init']:
                    child_pipe.send(('Successfully initialized',
                                     printed_output.getvalue()))
                    state['initial_init'] = False
            except Exception as e: # If we fail to initialize, just give up.
                # If this isn't the initial init, this exception will substitute
                # for the response to the next method call of the proxied napari
                # object.
                print('Exception occurred while trying to initialize napari!')
                e.child_traceback_string = traceback.format_exc()
                child_pipe.send((e, printed_output.getvalue()))
                state['keep_launching_napari'] = False
                return None

            command_timer = QTimer()
            def communicate():
                """Execute commands from the main process.

                Periodically called by Qt's event loop.
                """
                printed_output = io.StringIO()
                try:
                    if not child_pipe.poll():
                        return
                except BrokenPipeError:
                    # Windows raises this if parent closes the pipe.
                    # If the pipe is closed, we should close too.
                    state['keep_launching_napari'] = False
                    if close_method_name is not None:
                        close_method(*closeargs, **closekwargs)
                    return
                try:
                    cmd = child_pipe.recv()
                except EOFError:
                    # If the pipe is closed and there is nothing to receive,
                    # we should close too.
                    state['keep_launching_napari'] = False
                    if close_method_name is not None:
                        close_method(*closeargs, **closekwargs)
                    return
                if cmd is None: # This is how the parent signals us to exit.
                    state['keep_launching_napari'] = False
                    if close_method_name is not None:
                        close_method(*closeargs, **closekwargs)
                    return
                method_name, args, kwargs = cmd
                args, kwargs = _reconnect_shared_arrays(
                    args, kwargs, shared_arrays)
                try:
                    with redirect_stdout(printed_output):
                        result = getattr(obj, method_name)(*args, **kwargs)
                    if callable(result):
                        result = _dummy_function # Cheaper than a real callable
                    if isinstance(result, _SharedNumpyArray):
                        result = result._disconnect()
                    child_pipe.send((result, printed_output.getvalue()))
                except Exception as e:
                    e.child_traceback_string = traceback.format_exc()
                    child_pipe.send((e, printed_output.getvalue()))

            command_timer.timeout.connect(communicate)
            command_timer_interval_ms = 1
            command_timer.start(command_timer_interval_ms)

        if state['keep_launching_napari']:
            try:
                # Blocks until something is in pipe or pipe is broken.
                child_pipe.poll(timeout=None)
            except BrokenPipeError:
                return

# A mocked "microscope" object for testing and demonstrating proxied
# Napari displays. Since this is just for testing our imports are ugly:
class _Microscope:
    def __init__(self):
        import queue
        import time
        self.pm = ProxyManager(shared_memory_sizes=(1*2048*2060*2,))
        self.data_buffers = [
            self.pm.shared_numpy_array(which_mp_array=0,
                                       shape=(1, 2048, 2060),
                                       dtype='uint16')
            for i in range(2)]
        self.data_buffer_queue = queue.Queue()
        for i in range(len(self.data_buffers)):
            self.data_buffer_queue.put(i)
        print("Displaying", self.data_buffers[0].shape,
              self.data_buffers[0].dtype, 'images.')
        self.camera = self.pm.proxy_object(_Camera)
        self.display = display(proxy_manager=self.pm)
        self.num_frames = 0
        self.initial_time = time.perf_counter()

    def snap(self):
        import time
        from proxy_objects import launch_custody_thread
        def snap_task(custody):
            custody.switch_from(None, to=self.camera)
            which_buffer = self.data_buffer_queue.get()
            data_buffer = self.data_buffers[which_buffer]
            self.camera.record(out=data_buffer)
            custody.switch_from(self.camera, to=self.display)
            self.display.show_image(data_buffer)
            custody.switch_from(self.display, to=None)
            self.data_buffer_queue.put(which_buffer)
            self.num_frames += 1
            if self.num_frames == 100:
                time_elapsed =  time.perf_counter() - self.initial_time
                print("%0.2f average FPS"%(self.num_frames / time_elapsed))
                self.num_frames = 0
                self.initial_time = time.perf_counter()
        th = launch_custody_thread(snap_task, first_resource=self.camera)
        return th

class _Camera:
    def record(self, out):
        import numpy as np
        out[:] = np.random.randint(
            0, 2**16, size=out.shape, dtype='uint16')

if __name__ == '__main__':
    scope = _Microscope()
    snap_threads = []
    import threading
    print("Launching a ton of 'snap' threads...")
    for i in range(50):
        try:
            th = scope.snap()
            snap_threads.append(th)
        except RuntimeError:
            print('running threads: ', threading.active_count())
            break
    print(len(snap_threads), "'snap' threads launched.")
    for th in snap_threads:
        th.join()
    print("All 'snap' threads finished execution.")
    input('Hit enter to close napari...')
    # This will just close the viewer, not the child process.
    scope.display.close()

    # If you want to kill the child process before the end of the script,
    # you have to do something like this:

    import proxy_objects
    proxy_objects._close(scope.display)

    # We'll look into a way to make this a little easier to do. In theory,
    # the child process should close when the proxy_object is garbage
    # collected, however this isn't happening when we expect it to. Something
    # for us to look into...

    input('Hit enter to run off the end of the script...')

    # Note -- calling methods of a closed proxy object will result in an
    # OSError.
    try:
        scope.display.close()
    except OSError:
        pass # This is the exception we expected!
    else:
        raise AssertionError('We did not get the error we expected')



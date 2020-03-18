import multiprocessing as mp
# Sharing memory between child processes is tricky:
import ctypes as C
try:
    import numpy as np
except ImportError:
    np = None
# Showing exceptions from a child process is tricky:
import sys
import traceback
# Making sure a child process closes when the parent exits is tricky:
import atexit
import signal
# Printing from a child process is tricky:
import io
from contextlib import redirect_stdout

"""
Sometimes we put a computationally demanding Python object in a
multiprocessing child process, but this usually leads to high mental
overhead. Using pipes and queues for calling methods, getting/setting
attributes, printing, handling exceptions, and cleanup can lean to ugly
code and confusion. Can we isolate most of this mental overhead to this
module? If we do it right, we'll be able to write code like:

from camera_module import Camera
cam = ProxyObject(Camera) # Camera object instance lives in a child process...
cam.set_fps(10000)        # ...but acts like it's in the parent process!
cam.set_trigger('external')
x = cam.record(num_images=10000) # Fast, demanding, performance-critical.
cam.close()

...without paying much attention to the fact that the instance of the
Camera object actually lives in a child process, and communicates over
pipes.

Note that the method calls to our proxied object still block the parent
process; the idea is, the parent process is now effectively IO-limited
rather than CPU-limited, so we can write clean(er)-looking threading
code in the parent if we want multiple things to happen at once in the
parent.

Also note that multiprocessing already has proxied objects via
"managers"; we're rolling our own to learn, and have complete control.
If at the end of the process, we don't like ours better, we'll switch to
multiprocessing proxies.

CURRENT LIMITATIONS:

If you use this module, you have to protect the "entry point" of your
program, like this:

from camera_module import Camera
def main():
    cam = ProxyObject(Camera)
    cam.set_fps(10000)
    cam.set_trigger('external')
    x = cam.record(num_images=10000)
    cam.close()

if __name__ == '__main__':
    main()
"""

class ProxyManager:
    """Allocates shared memory and spawns proxy objects

    If you want your ProxyObjects to share memory with the parent (and
    each other), allocate a ProxyManager first, and use it to spawn your
    ProxyObjects.
    """
    def __init__(self, shared_memory_sizes=tuple()):
        if np is None:
            raise ImportError("We failed to import Numpy," +
                              " so there's no point using a ProxyManager.")
        # Allocate shared memory as multiprocessing Arrays:
        self.shared_mp_arrays = tuple(mp.Array(C.c_uint8, sz)
                                      for sz in shared_memory_sizes)

    def proxy_object(self, initializer, *init_args, **init_kwargs):
        return ProxyObject(initializer, *init_args, **init_kwargs,
                           shared_mp_arrays=self.shared_mp_arrays)

    def shared_numpy_array(self, shape, dtype, which_mp_array):
        assert which_mp_array in range(len(self.shared_mp_arrays))
        return _SharedNumpyArray(
            shape, dtype, self.shared_mp_arrays, which_mp_array)

class _SharedNumpyArray:
    def __init__(self, shape, dtype, shared_mp_arrays, which_mp_array):
        assert all(s > 0 for s in shape)
        assert all(int(s) == s for s in shape)
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.which_mp_array = which_mp_array
        if shared_mp_arrays is not None:
            self._restore_from_stub(shared_mp_arrays)

    def __enter__(self):
        self.shared_mp_arrays[self.which_mp_array].get_lock().acquire()
        return self._as_numpy()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shared_mp_arrays[self.which_mp_array].get_lock().release()

    def _as_numpy(self):
        """View the underlying multiprocessing array as a numpy array
        """
        assert self.shared_mp_arrays is not None
        return np.frombuffer(
            self.shared_mp_arrays[self.which_mp_array].get_obj(),
            count=np.prod(self.shape), dtype=self.dtype).reshape(self.shape)

    def _to_stub(self):
        """A copy that doesn't reference any shared memory.

        Useful for passing down pipes, if the process on the other side
        reconnects it to the shared memory on the other end.
        """
        return _SharedNumpyArray(
            self.shape, self.dtype, None, self.which_mp_array)

    def _restore_from_stub(self, shared_mp_arrays):
        assert self.which_mp_array in range(len(shared_mp_arrays))
        if (np.prod(self.shape) * self.dtype.itemsize >
            len(shared_mp_arrays[self.which_mp_array])):
            raise ValueError("Multiprocessing shared memory array is too "+
                             "small to hold the requested Numpy array")
        self.shared_mp_arrays = shared_mp_arrays

class ProxyObject:
    def __init__(
        self,
        initializer,
        *initargs,
        shared_mp_arrays=tuple(),
        **initkwargs,
        ):
        """Make an object in a child process, that acts like it isn't.

        initializer -- callable that returns an instance of a Python object.
        init_args, init_kwargs --  Arguments to 'initializer'
        """
        # Put an instance of the Python object returned by 'initializer'
        # in a child process:
        initargs, initkwargs = _disconnect_shared_arrays(initargs, initkwargs)
        parent_pipe, child_pipe = mp.Pipe()
        child_process = mp.Process(
            target=_child_loop,
            name=initializer.__name__,
            args=(initializer, initargs, initkwargs,
                  child_pipe, shared_mp_arrays))
        # Attribute-setting looks weird because we override __setattr__:
        super().__setattr__('parent_pipe', parent_pipe)
        super().__setattr__('child_pipe', child_pipe)
        super().__setattr__('child_process', child_process)
        super().__setattr__('shared_mp_arrays', shared_mp_arrays)
        # Make sure the child process initialized successfully:
        self.child_process.start()
        assert _get_response(self) == 'Successfully initialized'
        # Try to ensure the child process closes when we exit:
        atexit.register(lambda: _close(self))
        signal.signal(signal.SIGTERM, lambda s, f: _close(self))

    def __getattr__(self, name):
        """Access attributes of the child-process object in the parent process.

        As much as possible, we want attribute access and method calls
        to *seem* like they're happening in the parent process, if
        possible, even though they actually involve asking the child
        process over a pipe.
        """
        self.parent_pipe.send(('__getattribute__', (name,), {}))
        attr = _get_response(self)
        if callable(attr):
            def attr(*args, **kwargs):
                args, kwargs = _disconnect_shared_arrays(args, kwargs)
                self.parent_pipe.send((name, args, kwargs))
                return _get_response(self)
        return attr

    def __setattr__(self, name, value):
        self.parent_pipe.send(('__setattr__', (name, value), {}))
        return _get_response(self)

def _disconnect_shared_arrays(args, kwargs):
    # Replaces shared arrays in the args and kwargs with new stubs
    new_args = tuple(a._to_stub() if isinstance(a, _SharedNumpyArray) else a
                     for a in args)
    new_kwargs = {}
    for k, v in kwargs.items():
        new_kwargs[k] = v._to_stub() if isinstance(v, _SharedNumpyArray) else v
    return new_args, new_kwargs

def _reconnect_shared_arrays(args, kwargs, shared_arrays):
    # Restores stubs in-place in the args and kwargs
    for a in args:
        if isinstance(a, _SharedNumpyArray):
            a._restore_from_stub(shared_arrays)
    for v in kwargs.values():
        if isinstance(v, _SharedNumpyArray):
            v._restore_from_stub(shared_arrays)
    return None

def _get_response(proxy_object):
    resp, printed_output = proxy_object.parent_pipe.recv()
    if len(printed_output) > 0:
        print(printed_output, end='')
    if isinstance(resp, Exception):
        raise resp
    return resp

def _close(proxy_object):
    if not proxy_object.child_process.is_alive():
        return
    proxy_object.parent_pipe.send(None)
    proxy_object.child_process.join()

def _child_loop(initializer, args, kwargs, child_pipe, shared_arrays):
    # If any of the input arguments are _SharedNumpyArrays, we have to
    # show them where to find shared memory:
    _reconnect_shared_arrays(args, kwargs, shared_arrays)
    # Initialization.
    printed_output = io.StringIO()
    try: # Create an instance of our object...
        with redirect_stdout(printed_output):
            obj = initializer(*args, **kwargs)
        child_pipe.send(('Successfully initialized', printed_output.getvalue()))
    except Exception as e: # If we fail to initialize, just give up.
        e.child_traceback_string = traceback.format_exc()
        child_pipe.send((e, printed_output.getvalue()))
        return None
    # Main loop:
    while True:
        printed_output = io.StringIO()
        try:
            cmd = child_pipe.recv()
        except EOFError: # This implies the parent is dead; exit.
            return None
        if cmd is None: # This is how the parent signals us to exit.
            return None
        attr_name, args, kwargs = cmd
        _reconnect_shared_arrays(args, kwargs, shared_arrays)
        try:
            with redirect_stdout(printed_output):
                result = getattr(obj, attr_name)(*args, **kwargs)
            if callable(result):
                result = _dummy_function # Cheaper than sending a real callable
            child_pipe.send((result, printed_output.getvalue()))
        except Exception as e:
            e.child_traceback_string = traceback.format_exc()
            child_pipe.send((e, printed_output.getvalue()))

# If we're trying to return a (presumably worthless) "callable" to
# the parent, it might as well be small and simple:
def _dummy_function():
    return None

# When an exception from the child process isn't handled by the parent
# process, we'd like the parent to print the child traceback. Overriding
# sys.excepthook seems to be the standard way to do this:
def _my_excepthook(t, v, tb):
    """Show a traceback when a child exception isn't handled by the parent.
    """
    if hasattr(v, 'child_traceback_string'):
        print(f'{" Child Process Traceback ":v^80s}\n',
              v.child_traceback_string,
              f'{" Child Process Traceback ":^^80s}\n',
              f'{" Main Process Traceback ":v^80s}')
    sys.__excepthook__(t, v, tb)

sys.excepthook = _my_excepthook

# Multiprocessing code works fairly differently depending whether you
# use 'spawn' or 'fork'. Since 'spawn' seems to be available on every
# platform we care about, and 'fork' is either missing or broken on some
# platforms, we'll always use 'spawn'. If your code calls
# mp.set_start_method() and sets it to anything other than 'spawn', this
# will crash with a RuntimeError. If you really need 'fork', or
# 'forkserver', then you probably know what you're doing better than us,
# and you shouldn't be using this module.
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn')


# Testing block.
class TestClass:
    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        for i, a in enumerate(args):
            setattr(self, f'arg_{i}', a)

    def printing_method(self, *args, **kwargs):
        print(*args, **kwargs)

    def printing_method2(self):
        print('Hello world 2', end='', flush=False)
        print(end='', flush=True)

    def test_method(self, *args, **kwargs):
        return (args, kwargs)

    def test_shared_numpy_input(self, shared_numpy_array):
        with shared_numpy_array as x:
            print(x)

    def nested_method(self, crash=False):
        self._nested_method(crash)

    def _nested_method(self, crash):
        if crash:
            raise ValueError('This error was supposed to be raised')

def main():
    import time

    pm = ProxyManager((10, 20))
    object_with_shared_memory = pm.proxy_object(TestClass)
    data = pm.shared_numpy_array(
        shape=(10,),
        dtype=np.uint8,
        which_mp_array=0)
    with data as x:
        x[3] = 7
    print("Testing printing a shared numpy array in the child process:")
    object_with_shared_memory.test_shared_numpy_input(data)

    a = ProxyObject(TestClass, 'attribute', x=4,)
    b = ProxyObject(TestClass, x=5)
    print("\nTesting printing from child process:")
    a.printing_method(a.x, '... ', end='', flush=False)
    b.printing_method(b.x)
    b.printing_method('Hello')
    a.printing_method('Done!')
    a.printing_method('A')
    a.x = 4
    a.printing_method('Hello', 'world', end='', flush=True)
    print("\n\nTesting an exception raised in the child process:")
    try:
        a.z
    except AttributeError as e:
        print("Attribute error handled by parent process:\n ", e)
    print("\nTesting overhead:")
    num_gets = 10000
    start = time.perf_counter()
    for i in range(num_gets):
        a.x
    end = time.perf_counter()
    print(" %0.2f \u03BCs per get-attribute."%(
        1e6*(end - start) / num_gets))
    num_sets = 10000
    start = time.perf_counter()
    for i in range(num_sets):
        a.x = 4
    end = time.perf_counter()
    print(" %0.2f \u03BCs per set-attribute."%(
        1e6*(end - start) / num_sets))
    num_calls = 10000
    start = time.perf_counter()
    for i in range(num_calls):
        a.test_method()
    end = time.perf_counter()
    print(" %0.2f \u03BCs per trivial method call."%(
        1e6*(end - start) / num_calls))
    num_exceptions = 10000
    start = time.perf_counter()
    for i in range(num_exceptions):
        try:
            a.z
        except AttributeError:
            pass
    end = time.perf_counter()
    print(" %0.2f \u03BCs per parent-handled exception."%(
        1e6*(end - start) / num_exceptions))


if __name__ == '__main__':
    main()


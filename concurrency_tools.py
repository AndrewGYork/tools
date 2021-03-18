# Multiprocessing to spread CPU load, threading for concurrency:
import multiprocessing as mp
import threading
# Printing from a child process is tricky:
import io
from contextlib import redirect_stdout
# Handling exceptions from a child process/thread is tricky:
import sys
import traceback
import inspect
# Making sure a child process closes when the parent exits is tricky:
import atexit
import signal
# Sharing memory between child processes is tricky:
try:
    from multiprocessing import shared_memory
    import numpy as np
except ImportError:
    shared_memory = None
    np = None

"""
Sometimes we put a computationally demanding Python object in a
multiprocessing child process, but this usually leads to high mental
overhead. Using pipes and queues for calling methods, getting/setting
attributes, printing, handling exceptions, and cleanup can lead to ugly
code and confusion. Can we isolate most of this mental overhead to this
module? If we do it right, we'll be able to write fairly sophisticated
code that's still fairly readable. Note that the following code is
effectively uncommented; can you still figure out what it's doing?

                ####################################################
                #  EXAMPLE CODE (copypaste into 'test.py' and run) #
                ####################################################
from concurrency_tools import (
    ObjectInSubprocess, CustodyThread, SharedNumpyArray)
from dummy_module import Camera, Preprocessor, Display

def main():
    data_buffers = [
        SharedNumpyArray(shape=(10, 2000, 2000), dtype='uint16'),
        SharedNumpyArray(shape=(10, 2000, 2000), dtype='uint16')]
    display_buffers = [
        SharedNumpyArray(shape=(2000, 2000), dtype='uint8'),
        SharedNumpyArray(shape=(2000, 2000), dtype='uint8')]

    camera = ObjectInSubprocess(Camera)
    preprocessor = ObjectInSubprocess(Preprocessor)
    display = ObjectInSubprocess(Display)

    def acquire_timelapse(data_buffer, display_buffer, custody):
        custody.switch_from(None, to=camera)
        camera.record(out=data_buffer)

        custody.switch_from(camera, to=preprocessor)
        preprocessor.process(data_buffer, out=display_buffer)

        custody.switch_from(preprocessor, to=display)
        display.show(display_buffer)

        custody.switch_from(display, to=None)

    for i in range(15):
        th0 = CustodyThread(first_resource=camera, target=acquire_timelapse,
                            args=(data_buffers[0], display_buffers[0])).start()
        if i > 0:
            th1.join()
        th1 = CustodyThread(first_resource=camera, target=acquire_timelapse,
                            args=(data_buffers[1], display_buffers[1])).start()
        th0.join()
    th1.join()

if __name__ == '__main__':
    main()
            #####################################################
            #  This code is imported by the example code above. #
            #       Copypaste it into 'dummy_module.py'         #
            #####################################################
class Camera:
    def record(self, out):
        out.fill(1)

class Preprocessor:
    def process(self, x, out):
        x.max(axis=0, out=out)

class Display:
    def show(self, image):
        pass
                            ######################
                            #  END EXAMPLE CODE  #
                            ######################

Notice how little attention this code is spending on the fact that the
instances of the Camera, Preprocessing, and Display objects actually
live in child processes, communicate over pipes, and synchronize access
to shared memory.

Note that the method calls to our objects-in-subprocesses still block the parent
process; the idea is, the parent process is now effectively IO-limited
rather than CPU-limited, so we can write clean(er)-looking threading
code in the parent if we want multiple things to happen at once in the
parent.

Also note that python's multiprocessing module already has
objects-in-subprocesses via "managers", called "proxy objects", and the Pyro
package (https://github.com/irmen/Pyro5) lets you "proxy" objects on different
machines. We're rolling our own to learn, and have complete control. If at the
end of the process, we don't like ours better, we'll switch to multiprocessing
proxies or Pyro.

CURRENT LIMITATIONS:

Like all python code that relies on multiprocessing, if you use this
module, you have to protect the "entry point" of your program. The
typical way to do this is by using an "if __name__ == '__main__':" block:

import numpy as np
from proxy_objects import ObjectInSubprocess
from dummy_module import Display

def main():
    disp = ObjectInSubprocess(Display)
    image = np.random.random((2000 2000))
    disp.show(image)

if __name__ == '__main__':
    main()
"""

class SharedNumpyArray(np.ndarray):
    """A numpy array that lives in shared memory

    Inputs and outputs to/from ObjectInSubprocess are 'serialized', which
    is pretty fast - except for large in-memory objects. The only large
    in-memory objects we regularly deal with are numpy arrays, so it
    makes sense to provide a way to pass large numpy arrays via shared memory
    (which avoids slow serialization).

    Maybe you wanted to write code that looks like this:

        data_buf = np.zeros((400, 2000, 2000), dtype='uint16')
        display_buf = np.zeros((2000, 2000), dtype='uint8')

        camera = Camera()
        preprocessor = Preprocessor()
        display = Display()

        camera.record(num_images=400, out=data_buf)
        preprocessor.process(in=data_buf, out=display_buf)
        display.show(display_buf)

    ...but instead you write code that looks like this:

        data_buf = SharedNumpyArray(shape=(400, 2000, 2000), dtype='uint16')
        display_buf = SharedNumpyArray(shape=(2000, 2000), dtype='uint8')

        camera = ObjectInSubprocess(Camera)
        preprocessor = ObjectInSubprocess(Preprocessor)
        display = ObjectInSubprocess(Display)

        camera.record(num_images=400, out=data_buf)
        preprocessor.process(in=data_buf, out=display_buf)
        display.show(display_buf)

    ...and your payoff is, each object gets its own CPU core, AND passing
    large numpy arrays between the processes is still really fast!
    """
    def __new__(cls, shape=None, dtype=float, shared_memory_name=None,
                offset=0, strides=None, order=None):
        if shared_memory_name is None:
            dtype = np.dtype(dtype)
            requested_bytes = np.prod(shape, dtype='uint64') * dtype.itemsize
            requested_bytes = int(requested_bytes) #
            shm = shared_memory.SharedMemory(create=True, size=requested_bytes)
            atexit.register(lambda: SharedNumpyArray._unlink(shm))
        else:
            shm = shared_memory.SharedMemory(
                name=shared_memory_name, create=False)
        obj = super(SharedNumpyArray, cls).__new__(
            cls, shape, dtype, shm.buf, offset, strides, order)
        obj.shared_memory = shm
        obj.offset = offset
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        if not isinstance(obj, SharedNumpyArray):
            raise ValueError(
                "You can't view non-shared memory as shared memory.")
        self.shared_memory = obj.shared_memory
        self.offset = obj.offset
        self.offset += (self.__array_interface__['data'][0] -
                         obj.__array_interface__['data'][0])

    def __reduce__(self):
        args = (
            self.shape, self.dtype, self.shared_memory.name, self.offset,
            self.strides, None)
        return (SharedNumpyArray, args)

    @staticmethod
    def _unlink(shm):
        try:
            shm.unlink()
        except FileNotFoundError:
            pass ## must have already been called?

class ResultThread(threading.Thread):
    """threading.Thread with all the simple features we wish it had.

    We added a 'get_result' method that returns values/raises exceptions.

    We changed the return value of 'start' from 'None' to 'self' -- just to
    trivially save us a line of code when launching threads.

    Example:
    ```
        def f(a):
            ''' A function that does something... '''
            return a.sum()

        ##
        ## Getting Results:
        ##
        a = np.ones((2,), dtype='uint8')

        # Our problem:
        th = threading.Thread(target=f, args=(a,))
        th.start()
        th.join() # We can't access the result of f(a) without redefining f!

        # Our solution:
        res_th = ResultThread(target=f, args=(a,)).start()
        res = res_th.get_result() # returns f(a)
        assert res == 2

        ##
        ## Error handling
        ##
        a = 1

        # Our problem:
        th = threading.Thread(target=f, args=(a,))
        th.start()
        th.join()
        # f(a) raised an unhandled exception. Our parent thread has no idea!

        # Our solution:
        res_th = ResultThread(target=f, args=(a,)).start()
        try:
            res = res_th.get_result()
        except AttributeError:
            print("AttributeError was raised in thread!")
        else:
            raise AssertionError(
                'We expected an AttributeError to be raised on join!')

        # Unhandled exceptions raised during evaluation of 'f' are reraised in
        # the parent thread when you call 'get_result'.
        # Tracebacks may print to STDERR when the exception occurs in
        # the child thread, but don't affect the parent thread (yet).
    ```
    NOTE: This module modifies threading.excepthook. You can't just copy/paste
    this class definition and expect it to work.
    """
    def __init__(self, group=None, target=None, name=None, args=(),
                 kwargs=None):
        super().__init__(group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def start(self):
        try:
            super().start()
        except RuntimeError as e:
            if e.args == ("can't start new thread",):
                print('*'*80)
                print('Failed to launch a thread.')
                print(threading.active_count(), 'threads are currently active.')
                print('You might have reached a limit of your system;')
                print('let some of your threads finish before launching more.')
                print('*'*80)
            raise
        return self

    def get_result(self, timeout=None):
        """ Either returns a value or raises an exception.

        Optionally accepts a timeout in seconds. If thread has not returned
        after timeout seconds, raises a TimeoutError.
        """
        super().join(timeout=timeout)
        if self.is_alive(): ## Thread could potentially not be done yet!
            return TimeoutError('Thread did not return!')
        if hasattr(self, 'exc_value'):
            raise self.exc_value
        return self._return

class CustodyThread(ResultThread):
    """Threads that can access shared resources in the order they were launched.

    See the docstring at the top of this module for examples.
    """
    def __init__(self, first_resource=None,
                 group=None, target=None, name=None, args=(), kwargs=None):
        if 'custody' not in inspect.signature(target).parameters:
            raise ValueError("The function 'target' passed to a CustodyThread"
            " must accept an argument named 'custody'")
        custody = _Custody() # Useful for synchronization in the launched thread
        if first_resource is not None:
            # Get in line for custody of the first resource the launched
            # thread will use, but don't *wait* in that line; the launched
            # thread should do the waiting, not the main thread:
            custody.switch_from(None, first_resource, wait=False)
        if kwargs is None: kwargs = {}
        if 'custody' in kwargs:
            raise ValueError(
                "CustodyThread will create and pass a keyword argument to"
                " 'target' named 'custody', so keyword arguments to a"
                " CustodyThread can't be named 'custody'")
        kwargs['custody'] = custody
        super().__init__(group, target, name, args, kwargs)
        self.custody = custody

_original_threading_excepthook = threading.excepthook

def _my_threading_excepthook(args):
    """Show a traceback when a child exception isn't handled by the parent.
    """
    if isinstance(args.thread, ResultThread):
        args.thread.exc_value = args.exc_value
        args.thread.exc_traceback = args.exc_traceback
        args.thread.exc_type = args.exc_type
    else:
        _try_to_print_child_traceback(args.exc_value)
    return _original_threading_excepthook(args)

threading.excepthook = _my_threading_excepthook

FancyThread = ResultThread # So Andy can refer to it like this.
PoliteThread = CustodyThread

class ObjectInSubprocess:
    def __init__(
        self,
        initializer,
        *initargs,
        custom_loop=None,
        close_method_name=None,
        closeargs=None,
        closekwargs=None,
        **initkwargs,
        ):
        """Make an object in a child process, that acts like it isn't.

        As much as possible, we try to make instances of ObjectInSubprocess
        behave as if they're an instance of the object living in the parent
        process. They're not, of course: they live in a child process. If you
        have spare cores on your machine, this turns CPU-bound operations
        (which  threading can't parallelize) into IO-bound operations (which
        threading CAN parallelize),  without too much mental overhead for the
        coder.

        initializer -- callable that returns an instance of a Python object
        initargs, initkwargs --  arguments to 'initializer'
        close_method_name -- string, optional, name of our object's method to
            be called automatically when the child process exits
        closeargs, closekwargs -- arguments to 'close_method'
        """
        # Put an instance of the Python object returned by 'initializer'
        # in a child process:
        parent_pipe, child_pipe = mp.Pipe()
        child_loop = _child_loop if custom_loop is None else custom_loop
        child_process = mp.Process(
            target=child_loop,
            name=initializer.__name__,
            args=(child_pipe, initializer, initargs, initkwargs,
                  close_method_name, closeargs, closekwargs))
        # Attribute-setting looks weird here because we override __setattr__,
        # and because we use a dummy object's namespace to hold our attributes
        # so we shadow as little of the object's namespace as possible:
        super().__setattr__('_', _DummyClass()) # Weird, but for a reason.
        self._.parent_pipe = parent_pipe
        self._.parent_pipe_lock = _ObjectInSubprocessPipeLock()
        self._.child_pipe = child_pipe
        self._.child_process = child_process
        self._.waiting_list = _WaitingList()
        # Make sure the child process initialized successfully:
        with self._.parent_pipe_lock:
            self._.child_process.start()
            assert _get_response(self) == 'Successfully initialized'
        # Try to ensure the child process closes when we exit:
        atexit.register(lambda: _close(self))
        try:
            signal.signal(signal.SIGTERM, lambda s, f: _close(self))
        except ValueError: # We are probably starting from a thread.
            pass # Signal handling can only happen from main thread

    def __getattr__(self, name):
        """Access attributes of the child-process object in the parent process.

        As much as possible, we want attribute access and method calls
        to *seem* like they're happening in the parent process, if
        possible, even though they actually involve asking the child
        process over a pipe.
        """
        with self._.parent_pipe_lock:
            self._.parent_pipe.send(('__getattribute__', (name,), {}))
            attr = _get_response(self)
        if callable(attr):
            def attr(*args, **kwargs):
                with self._.parent_pipe_lock:
                    self._.parent_pipe.send((name, args, kwargs))
                    return _get_response(self)
        return attr

    def __setattr__(self, name, value):
        with self._.parent_pipe_lock:
            self._.parent_pipe.send(('__setattr__', (name, value), {}))
            return _get_response(self)

    def __del__(self):
        _close(self)

def _get_response(object_in_subprocess):
    """Effectively a method of ObjectInSubprocess, but defined externally to
    minimize shadowing of the object's namespace"""
    resp, printed_output = object_in_subprocess._.parent_pipe.recv()
    if len(printed_output) > 0:
        print(printed_output, end='')
    if isinstance(resp, Exception):
        raise resp
    return resp

def _close(object_in_subprocess):
    """Effectively a method of ObjectInSubprocess, but defined externally to
    minimize shadowing of the object's namespace"""
    if not object_in_subprocess._.child_process.is_alive():
        return
    with object_in_subprocess._.parent_pipe_lock:
        object_in_subprocess._.parent_pipe.send(None)
        object_in_subprocess._.child_process.join()
        object_in_subprocess._.parent_pipe.close()

def _child_loop(child_pipe, initializer, initargs, initkwargs,
                close_method_name, closeargs, closekwargs):
    """The event loop of a ObjectInSubprocess's child process
    """
    # Initialization.
    printed_output = io.StringIO()
    try: # Create an instance of our object...
        with redirect_stdout(printed_output):
            obj = initializer(*initargs, **initkwargs)
            if close_method_name is not None:
                close_method = getattr(obj, close_method_name)
                closeargs = tuple() if closeargs is None else closeargs
                closekwargs = dict() if closekwargs is None else closekwargs
                atexit.register(lambda: close_method(*closeargs, **closekwargs))
                # Note: We don't know if print statements in the close method
                # will print in the main process.
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
        method_name, args, kwargs = cmd
        try:
            with redirect_stdout(printed_output):
                result = getattr(obj, method_name)(*args, **kwargs)
            if callable(result):
                result = _dummy_function # Cheaper than sending a real callable
            child_pipe.send((result, printed_output.getvalue()))
        except Exception as e:
            e.child_traceback_string = traceback.format_exc()
            child_pipe.send((e, printed_output.getvalue()))

# A minimal class that we use just to get another namespace:
class _DummyClass:
    pass

# If we're trying to return a (presumably worthless) "callable" to
# the parent, it might as well be small and simple:
def _dummy_function():
    return None

class _WaitingList:
    """For synchronization of one-thread-at-a-time shared resources

    Each ObjectInSubprocess has a _WaitingList; if you want to define your own
    _WaitingList-like objects that can interact with
    _Custody.switch_from() and _Custody._wait_in_line(), make sure they have
    a waiting_list = [] attribute, and a waiting_list_lock =
    threading.Lock() attribute.
    """
    def __init__(self):
        self.waiting_list = [] # Switch to a queue/deque if speed really matters
        self.waiting_list_lock = threading.Lock()

    def __enter__(self):
        self.waiting_list_lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.waiting_list_lock.release()

class _ObjectInSubprocessPipeLock:
    '''Raises an educational exception (rather than blocking) when you try
       to acquire a locked lock.'''
    def __init__(self):
        self.lock = threading.Lock()

    def __enter__(self):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError(
                "Two different threads tried to use the same "
                "ObjectInSubprocess at the same time! This is bad. Look at the "
                "docstring of object_in_subprocess.py to see an example of how "
                "to use a _Custody object to avoid this problem.")
        return self.lock

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock.release()

threading_lock_type = type(threading.Lock()) # Used for typechecking

def _get_list_and_lock(resource):
    """Convenience function.

    Expected input: An ObjectInSubprocess, a _WaitingList, or a
    _WaitingList-like object with 'waiting_list' and
    'waiting_list_lock' attributes.
    """
    if isinstance(resource, ObjectInSubprocess):
        waiting_list = resource._.waiting_list.waiting_list
        waiting_list_lock = resource._.waiting_list.waiting_list_lock
    else: # Either a _WaitingList, or a good enough impression
        waiting_list = resource.waiting_list
        waiting_list_lock = resource.waiting_list_lock
    assert isinstance(waiting_list_lock, threading_lock_type)
    return waiting_list, waiting_list_lock

class _Custody:
    def __init__(self):
        """For synchronization of single-thread-at-a-time shared resources.

        See the docstring at the start of this module for example usage.
        For _Custody() to be useful, at least some of the objects
        accessed by your launched thread must be ObjectInSubprocess()s,
        _WaitingList()s, or _WaitingList-like objects.
        """
        self.permission_slip = threading.Lock()
        self.permission_slip.acquire()
        self.has_custody = False
        self.target_resource = None

    def switch_from(self, resource, to=None, wait=True):
        """Get in line for a shared resource, then abandon your current resource

        If wait==True, also wait in that line until it's your turn to
        own the next shared resource.
        """
        assert resource is not None or to is not None
        if to is not None:
            to_waiting_list, to_waiting_list_lock = _get_list_and_lock(to)
            with to_waiting_list_lock: # Get in the line for the next lock...
                if self not in to_waiting_list: # ...unless you're already in it
                    to_waiting_list.append(self)
        if resource is not None:
            assert self.has_custody
            waiting_list, waiting_list_lock = _get_list_and_lock(resource)
            with waiting_list_lock:
                waiting_list.pop(0) # Remove ourselves from the current line
                if len(waiting_list) > 0: # If anyone's next...
                    waiting_list[0].permission_slip.release() # ...wake them up
        self.has_custody = False
        self.target_resource = to
        if wait and self.target_resource is not None:
            self._wait_in_line()

    def release(self):
        """ Release custody of the current shared resource.

        If you get custody of a shared resource and then raise an exception,
        the next-in-line might wait forever.

        'release' is useful while handling exceptions, if you want to pass
        custody of the resource to the next-in-line.

        This only works if you currently have custody, but it's hard to raise
        an exception while waiting in line.
        """
        if self.has_custody:
            self.switch_from(self.target_resource, to=None)
        else:
            if self.target_resource is None:
                return
            waiting_list, waiting_list_lock = _get_list_and_lock(resource)
            with waiting_list_lock:
                waiting_list.remove(self)

    def _wait_in_line(self):
        """Wait in line until it's your turn."""
        waiting_list, _ = _get_list_and_lock(self.target_resource)
        if self.has_custody:
            assert self is waiting_list[0]
            return
        # Wait for your number to be called
        if self is waiting_list[0] and self.permission_slip.locked():
            self.permission_slip.release() # We arrived to an empty waiting list
        self.permission_slip.acquire() # Blocks if we're not first in line
        self.has_custody = True

# When an exception from a child process isn't handled by the parent
# process, we'd like the parent to print the child traceback. Overriding
# sys.excepthook and threading.excepthook seems to be the standard way
# to do this:
def _try_to_print_child_traceback(v):
    if hasattr(v, 'child_traceback_string'):
        print(f'{" Child Process Traceback ":v^79s}\n',
              v.child_traceback_string,
              f'{" Child Process Traceback ":^^79s}\n',
              f'{" Main Process Traceback ":v^79s}')

def _my_excepthook(t, v, tb):
    """Show a traceback when a child exception isn't handled by the parent.
    """
    _try_to_print_child_traceback(v)
    return sys.__excepthook__(t, v, tb)

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
class _Tests():
    '''
    Method names that start with `test_` will be run.

    Use: _test_... method names for tests that are called as part
    of a larger group.
    '''
    class TestClass:
        """Toy class that can be pout in a subprocess for testing """
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
            return shared_numpy_array.shape

        def test_shared_numpy_return(self, shape=(5,5)):
            return SharedNumpyArray(shape=shape)

        def test_modify_array(self, a):
            a.fill(1)
            return a

        def test_return_array(self, a):
            return a

        def nested_method(self, crash=False):
            self._nested_method(crash)

        def _nested_method(self, crash):
            if crash:
                raise ValueError('This error was supposed to be raised')

    def __init__(self):
        print(f'{"#":#^80s}')
        print(f'{" Running Tests ":#^80s}')
        print(f'{"#":#^80s}')
        self.tests = 0
        self.passed = 0

    def test_create_object_in_subprocess(self):
        object_in_subprocess = ObjectInSubprocess(_Tests.TestClass)

    def test_reconnnecting_and_disconnecting_views(self):
        for i in range(1000):
            self._trial_slicing_of_shared_array()

    def _trial_slicing_of_shared_array(self):
        import pickle
        ri = np.random.randint # Just to get short lines
        dtype = np.dtype(np.random.choice(
            [np.uint16, np.uint8, float, np.float32, np.float64]))
        original_dimensions = tuple(
            ri(2, 100) for d in range(ri(2, 5)))
        slicer = tuple(
            slice(
                ri(0, a//2),
                ri(0, a//2)*-1,
                ri(1, min(6, a))
                )
            for a in original_dimensions)
        a = SharedNumpyArray(shape=original_dimensions, dtype=dtype)
        a.fill(0)
        b = a[slicer] ## should be a view
        b.fill(1)
        expected_total = int(b.sum())
        reloaded_total = pickle.loads(pickle.dumps(b)).sum()
        assert expected_total == reloaded_total, \
            f'Failed {dtype.name}/{original_dimensions}/{slicer}'

    def test_passing_normal_numpy_array(self):
        shape = (10, 10)
        dtype = int
        sz = int(np.prod(shape, dtype='uint64')*np.dtype(int).itemsize)
        a = np.zeros(shape, dtype)
        object_with_shared_memory = ObjectInSubprocess(_Tests.TestClass)
        object_with_shared_memory.test_shared_numpy_input(a)

    def test_passing_retrieving_shared_array(self):
        shape = (10, 10)
        dtype = int
        sz = int(np.prod(shape, dtype='uint64')*np.dtype(int).itemsize)
        object_with_shared_memory = ObjectInSubprocess(_Tests.TestClass)
        a = SharedNumpyArray(shape=shape, dtype=dtype)
        a.fill(0)
        a = object_with_shared_memory.test_modify_array(a)
        assert a.sum() == np.product(shape, dtype='uint64'), (
            'Contents of array not correct!')

    def test_raise_attribute_error(self):
        a = ObjectInSubprocess(_Tests.TestClass, 'attribute', x=4,)
        try:
            a.z
        except AttributeError as e: # Get __this__ specific error
            print("Attribute error handled by parent process:\n ", e)

    def test_printing_in_child_process(self):
        a = ObjectInSubprocess(_Tests.TestClass, 'attribute', x=4,)
        b = ObjectInSubprocess(_Tests.TestClass, x=5)
        b.printing_method('Hello')
        a.printing_method('A')
        a.printing_method('Hello', 'world', end='', flush=True)
        a.printing_method('')
        a.printing_method(a.x, '... ', end='', flush=False)
        b.printing_method(b.x)
        expected_output = 'Hello\nA\nHello world\n4 ... 5\n'
        return expected_output

    def test_setting_attribute_of_object_in_subprocess(self):
        a = ObjectInSubprocess(_Tests.TestClass, 'attribute', x=4,)
        a.z = 10
        assert a.z == 10
        setattr(a, 'z', 100)
        assert a.z == 100

    def test_getting_attribute_of_object_in_subprocess(self):
        a = ObjectInSubprocess(_Tests.TestClass, 'attribute', x=4)
        assert a.x == 4
        assert getattr(a, 'x') == 4

    def test_object_in_subprocess_overhead(self):
        print('Performance summary:')
        n_loops = 10000
        a = ObjectInSubprocess(_Tests.TestClass, 'attribute', x=4,)
        t = self.time_it(
            n_loops, a.test_method, timeout_us=100, name='Trivial method call')
        print(f" {t:.2f} \u03BCs per trivial method call.")
        t = self.time_it(
            n_loops, lambda: a.x, timeout_us=100, name='Attribute access')
        print(f" {t:.2f} \u03BCs per get-attribute.")
        a.x = 4 ## test set attribute with normal syntax
        t = self.time_it(n_loops, lambda: setattr(a, 'x', 5),
                         timeout_us=100, name='Attribute setting')
        print(f" {t:.2f} \u03BCs per set-attribute.")
        t = self.time_it(n_loops, lambda: a.z, fail=False, timeout_us=200,
                         name='Attribute error')
        print(f" {t:.2f} \u03BCs per parent-handled exception.")
        self._test_passing_array_performance()

    def _test_passing_array_performance(self):
        from itertools import product
        shape = (1000, 1000)
        dtype = np.uint8
        pass_methods = ['reference', 'serialization']
        method_names = ['test_shared_numpy_input', 'test_modify_array']
        shapes = [(10, 10), (1000, 1000)]
        for s, f, m in product(shapes, method_names, pass_methods):
            self._test_array_passing(m, f, s, dtype, 1000)

    def _test_array_passing(self, pass_by, method_name, shape, dtype, n_loops):
        dtype = np.dtype(dtype)
        sz = int(np.prod(shape, dtype='uint64')*np.dtype(int).itemsize)
        direction = '<->' if method_name == 'test_modify_array' else '->'
        name = f'{shape} array {direction} {pass_by}'
        object_with_shared_memory = ObjectInSubprocess(_Tests.TestClass)
        if pass_by == 'reference':
            a = SharedNumpyArray(shape, dtype=dtype)
            timeout_us = 5e3
        elif pass_by == 'serialization':
            a = np.zeros(shape=shape, dtype=dtype)
            timeout_us = 1e6
        func = getattr(object_with_shared_memory, method_name)
        t_per_loop = self.time_it(n_loops, func, (a,), timeout_us=timeout_us,
                                  name=name)
        print(f' {t_per_loop:.2f} \u03BCs per {name}')

    def test_lock_with_waitlist(self):
        import time
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None # No progress bars :(

        camera_lock = _WaitingList()
        display_lock = _WaitingList()

        def snap(i, custody):
            if not tqdm is None: pbars['camera'].update(1)
            if not tqdm is None: pbars['camera'].refresh()
            # We're already in line for the camera; wait until you're first in line
            custody.switch_from(None, camera_lock)
            # Use the resource
            time.sleep(0.02)
            order['camera'].append(i)
            if not tqdm is None: pbars['camera'].update(-1)
            if not tqdm is None: pbars['display'].update(1)
            if not tqdm is None: pbars['camera'].refresh()
            if not tqdm is None: pbars['display'].refresh()
            custody.switch_from(camera_lock, display_lock)
            # Use the resource
            time.sleep(0.05)
            order['display'].append(i)
            # Move to the next resource
            custody.switch_from(display_lock, None)
            if not tqdm is None: pbars['display'].update(-1)
            if not tqdm is None: pbars['display'].refresh()
            return None

        num_snaps = 100
        order = {'camera': [], 'display': []}
        if not tqdm is None:
            f = '{desc: <30}{n: 3d}-{bar:45}|'
            pbars = {n: tqdm(total=num_snaps, unit='th',
                             bar_format=f, desc=f'Threads waiting on {n}')
                     for n in order.keys()}
        threads = []
        for i in range(num_snaps):
            threads.append(CustodyThread(
                target=snap, first_resource=camera_lock, args=(i,)).start())
        for th in threads:
            th.get_result()

        if not tqdm is None:
            for pb in pbars.values(): pb.close()

        assert order['camera'] == list(range(num_snaps))
        assert order['display'] == list(range(num_snaps))

    def test_incorrect_thread_management(self):

        p = ObjectInSubprocess(_Tests.TestClass)
        p.x = 5
        exceptions = [1]
        def t():
            try:
                p.x
            except RuntimeError: ## Should raise this
                pass
            else:
                exceptions.append(1)
        for i in range(100):
            threading.Thread(target=t).start()
        if sum(exceptions) == 0:
            raise UserWarning('No exceptions raised.'
                              'Expected some RuntimeErrors')

    def test_object_with_lock_with_waitlist(self):
        import time
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None ## No progress bars :(

        def snap(i, custody):
            prev_res = None
            itr = enumerate(zip(resources, res_names, funcs))
            for ri, (res, name, resource_funcs) in itr:
                if not tqdm is None:
                    pbars[name].update(1) # Be careful to access the resource
                    pbars[name].refresh() # before you have control of it.
                custody.switch_from(prev_res, res)
                for fname in resource_funcs:
                    a = getattr(res, fname)(i)
                results[i].append(a)
                acq_order[res.name].append(i)
                if not tqdm is None:
                    pbars[res.name].update(-1)
                    pbars[res.name].refresh()
                prev_res = res
            custody.switch_from(res, None)

        NUM_STEPS = 4 # matches the number of steps for check results.
        num_snaps = 30  # Number of threads to start

        # Create objects in subprocesses that are mocked resources to use.
        # Each has a method that sleep for some amount of time and
        # returns `True`.
        camera = ObjectInSubprocess(_DummyCamera, name='camera')
        processor = ObjectInSubprocess(_DummyProcessor, name='processor')
        display = ObjectInSubprocess(_DummyGUI, name='display')
        disk = ObjectInSubprocess(_DummyFileSaver, name='disk')
        resources = [camera, processor, display, disk]
        res_names = [str(r.name) for r in resources]
        acq_order = {r.name:[] for r in resources}
        results = [[] for i in range(num_snaps)]
        funcs = [('record',), # methods to call for camera
                 ('process', ), # methods to call for processor
                 ('display', ), # methods to call for display
                 ('save', ) # methods to call for for disk.
            ]
        if not tqdm is None:
            f = '{desc: <30}{n: 3d}-{bar:45}|'
            pbars = {n: tqdm(total=num_snaps, unit='th',
                              bar_format=f, desc=f'Threads waiting on {n}')
                     for n in acq_order.keys()}
        threads = []
        for i in range(num_snaps):
            th = CustodyThread(target=snap, first_resource=camera, args=(i,)
                ).start()
            threads.append(th)
        for th in threads:
            th.get_result()

        if not tqdm is None:
            for pb in pbars.values(): pb.close()

        # Check results
        for i, a in enumerate(results):
            assert sum(a) == NUM_STEPS*i, f'{i}-{a}'
        for r, th_o in acq_order.items():
            assert sorted(th_o) == th_o,\
                f'Resource `{r}` was used out of order! -- {th_o}'

    def test_indexing_views_of_views(self):
        import pickle
        ri = np.random.randint # Just to get short lines

        original_dimensions = (3, 3, 3, 256, 256)
        a = SharedNumpyArray(shape=original_dimensions, dtype='uint8')
        c = ri(0, 255, original_dimensions, dtype='uint8')

        assert not np.allclose(a, c), 'Shouldnt be equal yet'
        a[:] = c # fill a with random values from 3
        assert np.allclose(a, c), 'Should be equal not'

        b = a[:, :, :, 8:, :] # create a view of a shared_numpy_array
        d = c[:, :, :, 8:, :] # create a view of a normal numpy array
        assert np.allclose(b, d), 'Views should be the same'
        assert np.allclose(
            b[0, :, 0, :, :],
            d[0, :, 0, :, :]
        ), 'Indexing into a view should be the same'

        assert np.allclose(
            pickle.loads(pickle.dumps(b)), d
        ), 'Should be the same after disonnecting and reconnecting'

        assert np.allclose(
            pickle.loads(pickle.dumps(b))[0, :, 0, :, :],
            d[0, :, 0, :, :]
        ), 'Indexing into a reconnected array/view should be the same'

        assert np.allclose(
            pickle.loads(pickle.dumps(b[0, :, 0, :, :])),
            d[0, :, 0, :, :]
        ), 'We should be able to serialize/reload a view of a view'

    def run(self, test_prefix='test_'):
        tests = [i for i in dir(self) if i.startswith(test_prefix)]
        self.tests = len(tests)
        for i, t in enumerate(tests):
            self._run_single_test(i, t)
        self._summarize_results()


    def _run_single_test(self, i, t):
        printed_output = io.StringIO()
        name = t[5:].replace('_', ' ')
        print(f'{f"     {i+1} of {self.tests} | Testing {name}    ":-^80s}')
        try:
            with redirect_stdout(printed_output):
                expected_output = getattr(self, t)()
            if expected_output is not None:
                o = printed_output.getvalue()
                assert expected_output == o, \
                    f'\n Returned result:\n'\
                    f'    `{repr(o)}`\n'\
                    f' Did not match expected output:\n'\
                    f'     "{repr(expected_output)}"\n'
        except Exception as e:
            print('v'*80)
            print(traceback.format_exc().strip('\n'))
            print('^'*80)
        else:
            self.passed += 1
            if printed_output.getvalue():
                for l in printed_output.getvalue().strip('\n').split('\n'):
                    print(f'   {l}')
            print(f'{f"> Success <":-^80s}')

    def _summarize_results(self):
        fill = '#' if self.passed == self.tests else '!'
        print(f'{fill}'*80)
        message = f"Completed Tests -- passed {self.passed} of {self.tests}"
        if fill == "#":
            print(f'{f"  {message}  ":#^80s}')
        else:
            print(f'{f"  {message}  ":!^80s}')
        print(f'{fill}'*80)


    def time_it(self, n_loops, func, args=None, kwargs=None, fail=True,
                timeout_us=None, name=None):
        import time
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None ## No progress bars :(

        start = time.perf_counter()
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        if tqdm is not None:
            f = '{desc: <38}{n: 7d}-{bar:17}|[{rate_fmt}]'
            pb = tqdm(total=n_loops, desc=name, bar_format=f)
        for i in range(n_loops):
            if tqdm is not None: pb.update(1)
            try:
                func(*args, **kwargs)
            except Exception as e:
                if fail:
                    raise e
                else:
                    pass
        if not tqdm is None: pb.close()
        end = time.perf_counter()
        time_per_loop_us = ((end-start) / n_loops)*1e6

        if timeout_us is not None:
            if time_per_loop_us > timeout_us:
                name = func.__name__ if name is None else name
                raise TimeoutError(
                    f'Timed out on {name}\n'
                    f'   args:{args}\n'
                    f'   kwargs: {kwargs}\n'
                    f' Each loop took {time_per_loop_us:.2f} \u03BCs'
                    f' (Allowed: {timeout_us:.2f} \u03BCs)')
        return time_per_loop_us

### TODO: Remove these if I can....
class _DummyObject:
    def __init__(self, name='generic_dummy_obj'):
        self.name=name

class _DummyCamera(_DummyObject):
    def record(self, a):
        import time
        time.sleep(.05)
        return a

class _DummyProcessor(_DummyObject):
    def process(self, a):
        import time
        time.sleep(.2)
        return a

class _DummyGUI(_DummyObject):
    def display(self, a):
        import time
        time.sleep(.002)
        return a

class _DummyFileSaver(_DummyObject):
    def save(self, a):
        import time
        time.sleep(.3)
        return a

if __name__ == '__main__':
    _Tests().run('test_')
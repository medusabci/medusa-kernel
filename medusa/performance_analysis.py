import functools, time


def perf_analysis(func):
    """Returns the run time of the decorated function"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        return value, {'run_time': run_time}
    return wrapper

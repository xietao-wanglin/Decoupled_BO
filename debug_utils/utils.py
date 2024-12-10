from functools import wraps


def record_io(enabled=True):
    def decorator(func):
        func.history = []

        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if enabled:
                call_data = {
                    "kwargs": kwargs,
                    "result": result
                }
                func.history.append(call_data)
            return result
        return wrapper
    return decorator

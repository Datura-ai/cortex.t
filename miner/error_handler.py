import functools
import bittensor as bt

def error_handler(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Execute the original function
            return func(*args, **kwargs)
        except Exception as e:
            # Handle any other exceptions
            bt.logging.exception(e)
            print(e)
            return None
    return wrapper

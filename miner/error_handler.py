import functools
import bittensor as bt


def error_handler(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Execute the original function
            return func(*args, **kwargs)
        except GeneratorExit:
            # Perform any necessary cleanup here
            bt.logging.error("Generator is closing, performing cleanup.")
        except Exception as err:
            # Handle any other exceptions
            bt.logging.exception(err)

    return wrapper

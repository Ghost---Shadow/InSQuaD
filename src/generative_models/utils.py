from functools import wraps
import time


def retry(max_retries=100, backoff_time=5):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_count = 0
            while retry_count < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(
                        f"Attempt {retry_count + 1} failed: {e}. Backoff for {backoff_time} seconds."
                    )
                    retry_count += 1
                    time.sleep(backoff_time)
            raise Exception("Max retries reached, unable to complete operation")

        return wrapper

    return decorator

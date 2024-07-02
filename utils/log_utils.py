import time
from asyncio.coroutines import iscoroutinefunction
from loguru import logger


def timer(message=None):
    def cost_time(func):
        def fun(*args, **kwargs):
            t = time.perf_counter()
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__}({message}) 耗时：{time.perf_counter() - t:.4f} s")
            return result

        async def func_async(*args, **kwargs):
            t = time.perf_counter()
            result = await func(*args, **kwargs)
            logger.debug(f"{func.__name__}({message}) 耗时：{time.perf_counter() - t:.4f}s")
            return result

        if iscoroutinefunction(func):
            return func_async
        else:
            return fun
    return cost_time

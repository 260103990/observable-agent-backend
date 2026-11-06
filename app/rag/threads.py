"""Run pure CPU/file work without abandoning its resources on repeated cancellation."""
import asyncio


async def run_cpu(function, *args):
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    cancellation = None
    while True:
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation = exc
            if task.done():
                # A worker itself may raise CancelledError. Consume its outcome.
                try:
                    task.result()
                except BaseException:
                    pass
                raise cancellation
            # Keep shielding on every cancellation. A bare await task here
            # would cancel the wrapper future while its actual thread runs on.
        except BaseException:
            if cancellation is not None:
                raise cancellation
            raise
        else:
            if cancellation is not None:
                raise cancellation
            return result

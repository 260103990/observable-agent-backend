import asyncio
import threading

import pytest


def test_repeated_cancellation_waits_until_actual_thread_finishes():
    from app.rag.service import _thread_work
    started,release,finished=threading.Event(),threading.Event(),threading.Event()
    def work():
        started.set()
        release.wait(3)
        finished.set()
    async def run():
        task=asyncio.create_task(_thread_work(work))
        assert await asyncio.to_thread(started.wait,2)
        try:
            task.cancel()
            for _ in range(3):
                await asyncio.sleep(0)  # schedule cancellation delivery, no time-based sleep
            task.cancel()
            for _ in range(3):
                await asyncio.sleep(0)
            finished_early=task.done()
            assert not finished.is_set()
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert not finished_early,"Request scope ended while its thread was still using the upload"
        assert finished.is_set()
    asyncio.run(run())


def test_store_cpu_snapshot_runs_in_thread_but_database_stays_on_original_loop(monkeypatch):
    from app.rag import store as module
    from tests.test_paper_store import sample
    original=module._snapshot_import
    cpu_threads=[]
    def snapshot(*args):
        cpu_threads.append(threading.get_ident())
        return original(*args)
    monkeypatch.setattr(module,"_snapshot_import",snapshot)
    async def run():
        loop=asyncio.get_running_loop()
        caller=threading.get_ident()
        class Transaction:
            async def __aenter__(self):
                assert asyncio.get_running_loop() is loop
            async def __aexit__(self,*args):
                return False
        class Connection:
            def transaction(self):
                return Transaction()
            async def execute(self,*args):
                assert asyncio.get_running_loop() is loop
            async def executemany(self,*args):
                assert asyncio.get_running_loop() is loop
        class Acquire:
            async def __aenter__(self):
                return Connection()
            async def __aexit__(self,*args):
                return False
        class Pool:
            def acquire(self):
                return Acquire()
        await module.PostgresPaperStore(Pool()).save(*sample())
        assert cpu_threads and all(thread!=caller for thread in cpu_threads)
    asyncio.run(run())

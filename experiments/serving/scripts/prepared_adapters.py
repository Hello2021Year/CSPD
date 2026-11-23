"""Bounded CPU lookahead; a cache miss invalidates a serving measurement.

Only synthetic data preparation happens here. The serving client must still
wait for arrival and admission before calling any GPU adapter-load endpoint.
"""
import asyncio


class PreparationStarvation(RuntimeError):
    pass


class PreparedAdapters:
    def __init__(self, live, arm, indices, capacity=64, workers=4):
        assert capacity > 0 and workers > 0 and len(indices) == len(set(indices))
        self.live, self.arm, self.indices = live, arm, list(indices)
        self.capacity = capacity
        self.space = asyncio.Semaphore(capacity)
        self.futures = {i: asyncio.get_running_loop().create_future() for i in indices}
        self.skipped, self.consumed = set(), set()
        self.released = set()
        self.cursor = 0
        self.tasks = []
        self.workers = workers
        self.misses = 0

    async def _worker(self):
        while self.cursor < len(self.indices):
            await self.space.acquire()
            # There is no await between selecting an index and advancing cursor.
            while self.cursor < len(self.indices) and self.indices[self.cursor] in self.skipped:
                self.cursor += 1
            if self.cursor == len(self.indices):
                self.space.release()
                return
            index = self.indices[self.cursor]
            self.cursor += 1
            try:
                item = await self.live.prepare(self.arm, index)
            except asyncio.CancelledError:
                self.space.release()
                raise
            except Exception as exc:
                if index in self.skipped:
                    self.space.release()
                else:
                    self.futures[index].set_exception(exc)
                continue  # The reserved slot is released by take/skip/close.
            if index in self.skipped:
                self.live.release(item)
                self.space.release()
            else:
                self.futures[index].set_result(item)

    async def start(self):
        self.tasks = [asyncio.create_task(self._worker()) for _ in range(self.workers)]
        # This bounded initial fill precedes the workload's time origin.
        await asyncio.gather(*(asyncio.shield(self.futures[i])
                               for i in self.indices[:self.capacity]))

    def take(self, index):
        future = self.futures[index]
        assert index not in self.consumed and index not in self.skipped
        if not future.done():
            self.misses += 1
            raise PreparationStarvation(f'CPU adapter {index} was not ready at admission')
        item = future.result()
        self.consumed.add(index)
        return item

    def release(self, item):
        index = item['adapter_id']
        assert index in self.consumed and index not in self.released
        self.live.release(item)
        self.released.add(index)
        # Reserve capacity until serving has unloaded the module. This bounds
        # active+ready tensors together and permits a full cold-start burst.
        self.space.release()

    def skip(self, index):
        if index in self.consumed or index in self.skipped:
            return
        self.skipped.add(index)
        future = self.futures[index]
        if future.done():
            if future.exception() is None:
                self.live.release(future.result())
            self.space.release()

    async def close(self):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        for index in self.indices:
            self.skip(index)

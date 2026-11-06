"""Bounded, event-loop-local notifications; the request owner performs all I/O."""
import asyncio
from dataclasses import dataclass

from app.tools.web_models import SearchBatch


@dataclass(frozen=True)
class FlightOutcome:
    batch: SearchBatch | None=None
    error: str | None=None


@dataclass
class _Flight:
    future: asyncio.Future
    waiters: int=0


@dataclass(frozen=True)
class FlightTicket:
    key: str
    flight: _Flight
    leader: bool

    @property
    def future(self):return self.flight.future


class SearchSingleFlight:
    def __init__(self,*,enabled=True,max_keys=64,max_waiters=32):
        if type(enabled) is not bool or any(type(v) is not int or v<=0 for v in (max_keys,max_waiters)):
            raise ValueError('Invalid Search coalescing bounds')
        self.enabled=enabled
        self._max_keys,self._max_waiters=max_keys,max_waiters
        self._flights={}

    def claim(self,key):
        if not self.enabled:return None
        flight=self._flights.get(key)
        if flight is not None:
            if flight.waiters>=self._max_waiters:return None
            flight.waiters+=1
            return FlightTicket(key,flight,False)
        if len(self._flights)>=self._max_keys:return None
        flight=_Flight(asyncio.get_running_loop().create_future())
        self._flights[key]=flight
        return FlightTicket(key,flight,True)

    def release_waiter(self,ticket):
        if not ticket.leader:
            ticket.flight.waiters=max(0,ticket.flight.waiters-1)

    def finish(self,ticket,*,batch=None,error=None):
        if not ticket.leader or self._flights.get(ticket.key) is not ticket.flight:return
        del self._flights[ticket.key]
        if not ticket.future.done():
            # Results carry safe codes, not exceptions or credential-bearing response bodies.
            ticket.future.set_result(FlightOutcome(batch,error))

    def clear(self):
        flights=list(self._flights.values())
        self._flights.clear()
        for flight in flights:
            if not flight.future.done():flight.future.set_result(FlightOutcome(error='invalidated'))

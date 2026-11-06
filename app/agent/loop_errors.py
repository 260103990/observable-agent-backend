"""Safe, provider-independent failures for the bounded loop."""


class LoopError(Exception):
    pass


class LoopConfigurationError(LoopError):
    pass


class LoopProtocolError(LoopError):
    pass


class LoopTimeoutError(LoopError):
    pass


class LoopConnectionError(LoopError):
    pass


class LoopContextLimitError(LoopError):
    pass

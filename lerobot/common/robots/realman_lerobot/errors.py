class DeviceNotConnectedError(RuntimeError):
    pass


class DeviceAlreadyConnectedError(RuntimeError):
    pass


class RobotAPIError(RuntimeError):
    """Raised when RealMan SDK returns non-zero status code."""
    pass
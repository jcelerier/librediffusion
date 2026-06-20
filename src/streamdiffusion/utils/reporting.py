from __future__ import annotations

import logging
from typing import Any


def report_error(
    msg: str,
    *args: Any,
    exc_info: Any | None = None,
    stack_info: bool = False,
    stacklevel: int = 2,
) -> None:
    """
    Log an error message and tag it as reportable upstream.

    It implements the same signature as logging.error, but adds extra={"report_error": True} so that ai-runner
    propagates the error to the end user.
    """
    logging.error(
        msg,
        *args,
        exc_info=exc_info,
        stack_info=stack_info,
        stacklevel=stacklevel,
        extra={"report_error": True},
    )



"""HTTP client to nerve's device API at :8111/tools/*."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

import httpx
import structlog

logger = structlog.get_logger().bind(domain="nerve_client")

_T = TypeVar("_T")

# Transient connection errors worth retrying on a cold first call. The nerve
# daemon intermittently drops the first keep-alive connection in a burst
# (Errno 54, reset by peer) before the socket is re-established. ``ConnectError``
# from a tool call is wrapped into the builtin ``ConnectionError`` by
# ``_call_tool``; ``ConnectionResetError`` is itself a ``ConnectionError``
# subclass, so catching ``ConnectionError`` covers both. Read-phase resets
# surface as ``httpx.ReadError`` / ``httpx.RemoteProtocolError``.
_TRANSIENT_CONN_ERRORS = (
    ConnectionError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
)


class UiTreeUnavailableError(RuntimeError):
    """The nerve daemon's UI element dump could not be read.

    Raised by ``NerveClient.elements_strict`` — never by the lenient
    ``elements()``, which still returns ``[]`` on the same failures. Callers
    that treat "no elements" as a meaningful assertion result (BVT assert
    steps) need this distinction: a failed dump is a HARNESS fault, not
    evidence the screen is genuinely empty.
    """


def find_reservation(reservations: list[dict], serial: str) -> dict | None:
    """Return the active reservation entry for *serial* in a nerve_pool payload, or None.

    Any reservation on the device blocks it — there is no override and no
    correlation against the caller's own session (owner/session-token
    passthrough is handled by linc-nerve).
    """
    for reservation in reservations or []:
        if reservation.get("device_serial") == serial:
            return reservation
    return None


class NerveClient:
    """HTTP client to nerve's device API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8111",
        *,
        max_retries: int = 3,
        retry_backoff: float = 0.25,
    ) -> None:
        self._base_url = base_url
        self._http = httpx.Client(trust_env=False)
        self._max_retries = max(1, max_retries)
        self._retry_backoff = retry_backoff

    def _with_retry(self, call: str, fn: Callable[[], _T]) -> _T:
        """Run ``fn``, retrying transient cold-connection resets.

        Retries up to ``max_retries`` attempts with linear backoff, logging
        each retry. A genuinely-down daemon raises on every attempt and the
        final error propagates once attempts are exhausted, so it is still
        reported unreachable (bounded retries).
        """
        for attempt in range(1, self._max_retries + 1):
            try:
                return fn()
            except _TRANSIENT_CONN_ERRORS as exc:
                if attempt >= self._max_retries:
                    raise
                logger.warning(
                    "nerve_cold_connection_retry",
                    call=call,
                    attempt=attempt,
                    max_attempts=self._max_retries,
                    error=str(exc),
                )
                time.sleep(self._retry_backoff * attempt)
        raise AssertionError("unreachable")  # pragma: no cover

    def _call_tool(self, tool: str, params: dict, timeout: int = 30) -> dict:
        """POST to {base_url}/tools/{tool} with JSON params.

        Raises ConnectionError on httpx.ConnectError.
        """
        try:
            resp = self._http.post(
                f"{self._base_url}/tools/{tool}",
                json=params,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            # Unwrap nerve's {"result": <tool_output>} envelope
            if isinstance(data, dict) and "result" in data:
                return data["result"]
            return data
        except httpx.ConnectError as exc:
            raise ConnectionError(f"Nerve unreachable at {self._base_url}: {exc}") from exc

    def discover(self) -> list[dict]:
        """Call nerve_discover, return devices list.

        Returns [] on any error (graceful degradation).
        """
        try:
            data = self._call_tool("nerve_discover", {})
            return data.get("devices", [])
        except Exception:
            return []

    def install(self, device: str, path: str) -> dict:
        """Call nerve_install with 120s timeout.

        Raises ConnectionError if nerve unreachable.
        """
        return self._call_tool("nerve_install", {"device": device, "path": path}, timeout=120)

    def launch(self, device: str, package: str, uri: str = "") -> dict:
        """Call nerve_launch, retrying transient cold-connection resets.

        The first launch in a BVT burst can hit a reset-by-peer before the
        daemon's keep-alive connection is established; retry a few times so a
        cold connection doesn't abort an otherwise-healthy run.

        Raises ConnectionError if nerve unreachable.
        """
        return self._with_retry(
            "launch",
            lambda: self._call_tool("nerve_launch", _launch_params(device, package, uri)),
        )

    def kill(self, device: str, package: str) -> dict:
        """Call nerve_kill.

        Raises ConnectionError if nerve unreachable.
        """
        return self._call_tool("nerve_kill", {"device": device, "package": package})

    def clear_app_data(self, device: str, package: str) -> dict:
        """Call nerve_clear_app_data (Android ``pm clear`` / iOS-sim uninstall)
        for a cold start.

        ``confirm=True`` is required by nerve's destructive-op gate:
        without it the daemon rejects the clear with ``confirmation_required``
        and — because a failed clear is a non-fatal reset warning — every
        cold_start silently degraded to a warm kill+relaunch while still
        reporting PASS. This client is the automated harness the confirmation
        exists to distinguish from accidental interactive calls.

        Raises ConnectionError if nerve unreachable.
        """
        return self._call_tool(
            "nerve_clear_app_data",
            {"device": device, "package": package, "confirm": True},
        )

    def grant_permission(self, device: str, package: str, permission: str) -> dict:
        """Call nerve_grant_permission (Android ``pm grant``).

        ``confirm=True`` is required by nerve's destructive-op gate,
        same as clear_app_data above — without it the daemon rejects the grant
        with ``confirmation_required`` and any BVT with a ``grant_permission``
        step fails at that step.

        Raises ConnectionError if nerve unreachable.
        """
        return self._call_tool(
            "nerve_grant_permission",
            {"device": device, "package": package, "permission": permission, "confirm": True},
        )

    def elements(self, device: str) -> list[dict]:
        """Call nerve_elements, return raw element list. Returns [] on error."""
        try:
            data = self._call_tool("nerve_elements", {"device": device})
            return data.get("elements", [])
        except Exception:
            return []

    def elements_strict(self, device: str) -> list[dict]:
        """Call nerve_elements, raising UiTreeUnavailableError when the dump
        could not be read.

        STRICT counterpart to ``elements()``: that lenient method swallows
        every failure to ``[]``, which is indistinguishable from a genuinely
        empty screen — fine for wait/poll and tap-by-text retry loops, wrong
        for a one-shot assertion (a failed dump must not silently pass a
        `*_not_present` assert or read as "0 elements" on a real failure).

        Raises UiTreeUnavailableError when: the call itself raises
        (ConnectionError from a dead daemon, or an HTTP error via
        ``raise_for_status``); the unwrapped result is a dict carrying an
        ``error`` or ``elements_error`` key; or the result lacks an
        ``elements`` key entirely. Returns the list otherwise — an empty list
        from ``{"elements": []}`` is a genuinely empty screen (nerve
        guarantees that shape), not swallowed as [] on failure.
        """
        try:
            data = self._call_tool("nerve_elements", {"device": device})
        except Exception as exc:
            raise UiTreeUnavailableError(str(exc)) from exc
        detail = isinstance(data, dict) and (data.get("error") or data.get("elements_error"))
        if detail:
            raise UiTreeUnavailableError(str(detail))
        if isinstance(data, dict) and "elements" in data:
            return data["elements"]
        raise UiTreeUnavailableError(f"nerve_elements response missing 'elements' key: {data!r}")

    def inspect(self, device: str, package: str) -> dict:
        """Call nerve_inspect for non-invasive process state (pid, running,
        memory, crash signature). Works on Android and iOS simulators.

        Raises ConnectionError if nerve unreachable.
        """
        return self._call_tool("nerve_inspect", {"device": device, "package": package})

    def screen_state(self, device: str) -> dict:
        """Call nerve_state for composite device state. Returns {} on error."""
        try:
            return self._call_tool("nerve_state", {"device": device})
        except Exception:
            return {}

    def screenshot(self, device: str) -> bytes:
        """Call nerve_screenshot and read the resulting PNG.

        The daemon writes to a file and returns its path. We read the bytes
        so callers don't need filesystem access to the daemon's tmp dir.

        Raises ConnectionError if nerve unreachable.
        Raises RuntimeError if daemon returns no path/screenshot key.
        """
        import pathlib

        result = self._call_tool("nerve_screenshot", {"device": device})
        path = result.get("path") or result.get("screenshot")
        if path:
            return pathlib.Path(path).read_bytes()
        raise RuntimeError(f"nerve_screenshot returned no path: {result}")

    def tap(self, device: str, x: int, y: int) -> dict:
        """Call nerve_tap at screen coordinates.

        Raises ConnectionError if nerve unreachable.
        """
        return self._call_tool("nerve_tap", {"device": device, "x": x, "y": y})

    def type_text(self, device: str, text: str) -> dict:
        """Call nerve_type — type text into the currently focused field.

        Raises ConnectionError if nerve unreachable.
        """
        return self._call_tool("nerve_type", {"device": device, "text": text})

    def long_press(self, device: str, x: int, y: int, duration_ms: int = 500) -> dict:
        """Call nerve_long_press at screen coordinates.

        Raises ConnectionError if nerve unreachable.
        """
        return self._call_tool(
            "nerve_long_press",
            {"device": device, "x": x, "y": y, "duration_ms": duration_ms},
        )

    def swipe(
        self, device: str, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300
    ) -> dict:
        """Call nerve_swipe — a two-point gesture from (x1,y1) to (x2,y2).

        Used for swipe/scroll/drag gestures. Maps onto the daemon's
        ``start_x``/``start_y``/``end_x``/``end_y`` tool parameters.

        Raises ConnectionError if nerve unreachable.
        """
        return self._call_tool(
            "nerve_swipe",
            {
                "device": device,
                "start_x": x1,
                "start_y": y1,
                "end_x": x2,
                "end_y": y2,
                "duration_ms": duration_ms,
            },
        )

    def density(self, device: str) -> dict:
        """Call nerve_density for the dp→px factor. Returns {} on any error.

        Density is consumed by the BVT runner's dp size assertions, which
        hard-fail when the factor is missing — so swallowing errors to {} here
        surfaces a clean assertion failure rather than an exception.
        """
        try:
            return self._call_tool("nerve_density", {"device": device})
        except Exception:
            return {}

    def navigate(self, device: str, uri: str) -> dict:
        """Open a deep-link URI on the device via nerve_launch.

        The daemon dispatches `am start -W -a android.intent.action.VIEW
        -d <uri>` for Android or `xcrun simctl openurl` for iOS.

        Raises ConnectionError if nerve unreachable.
        """
        return self._call_tool("nerve_launch", {"device": device, "package": "", "uri": uri})

    def metrics(self, device: str, package: str) -> dict | None:
        """Call nerve_metrics for a running package. Returns None on any error.

        Metrics are best-effort — callers (e.g., BVT runner) skip
        the perf snapshot when None comes back rather than failing the test.
        """
        try:
            return self._call_tool("nerve_metrics", {"device": device, "package": package})
        except Exception:
            return None

    def pool(self) -> dict:
        """Call nerve_pool for the device-reservation ledger.

        Returns the daemon's raw ``{reservations: [...], count, session_id}``
        payload (linc-nerve ``mcp/discovery_tools.py`` / ``core/device_manager.py``).

        Raises ConnectionError if nerve unreachable — mirrors ``install`` /
        ``launch`` rather than swallowing, so callers checking a reservation
        as a soft precondition can choose to degrade gracefully (skip the
        check) instead of silently treating "unreachable" as "unreserved".
        """
        return self._call_tool("nerve_pool", {})

    def health(self) -> dict:
        """GET /health — daemon liveness probe.

        Retries transient cold-connection resets so the first probe in a
        burst doesn't false-negative as "unreachable".

        Raises ConnectionError if nerve unreachable.
        """

        def _probe() -> dict:
            try:
                resp = self._http.get(f"{self._base_url}/health", timeout=5)
                resp.raise_for_status()
                return resp.json()
            except httpx.ConnectError as exc:
                raise ConnectionError(f"Nerve unreachable at {self._base_url}: {exc}") from exc

        return self._with_retry("health", _probe)

    @property
    def available(self) -> bool:
        """True if the nerve daemon's /health endpoint responds. Never raises."""
        try:
            self.health()
            return True
        except Exception:
            return False


def _launch_params(device: str, package: str, uri: str) -> dict[str, str]:
    """Build the nerve_launch request payload, including uri only when non-empty."""
    params: dict[str, str] = {"device": device, "package": package}
    if uri:
        params["uri"] = uri
    return params

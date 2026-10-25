"""Custom stdio transport — native asyncio to avoid anyio.wrap_file deadlock.

The MCP SDK's stdio_server uses anyio.wrap_file() which deadlocks after ~2 tool
calls when tools use anyio.to_thread. Note: MCP SDK v1.26.0 calls sync tools
directly on the event loop without anyio.to_thread wrapping.
This module uses asyncio.connect_read_pipe/connect_write_pipe for native async
I/O that doesn't compete with anyio's thread pool.

Ported from linc-nerve.
"""

import asyncio
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import anyio
import anyio.lowlevel
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage


@asynccontextmanager
async def stdio_server() -> AsyncGenerator[
    tuple[
        MemoryObjectReceiveStream[SessionMessage | Exception],
        MemoryObjectSendStream[SessionMessage],
    ],
    None,
]:
    """Server transport using native asyncio streams (deadlock-free).

    Drop-in replacement for mcp.server.stdio.stdio_server. Uses
    asyncio.connect_read_pipe / connect_write_pipe instead of
    anyio.wrap_file to avoid thread pool contention with
    anyio.to_thread.run_sync.

    Yields:
        Tuple of (read_stream, write_stream) for the MCP server.
    """
    loop = asyncio.get_event_loop()

    # Native asyncio reader for stdin — no threads involved.
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin.buffer)

    # Native asyncio writer for stdout.
    w_transport, _ = await loop.connect_write_pipe(asyncio.BaseProtocol, sys.stdout.buffer)

    read_stream_writer, read_stream = anyio.create_memory_object_stream[
        SessionMessage | Exception
    ](0)
    write_stream, write_stream_reader = anyio.create_memory_object_stream[SessionMessage](0)

    async def stdin_reader() -> None:
        try:
            async with read_stream_writer:
                while True:
                    line = await reader.readline()
                    if not line:
                        break
                    try:
                        message = JSONRPCMessage.model_validate_json(line)
                    except Exception as exc:
                        await read_stream_writer.send(exc)
                        continue
                    await read_stream_writer.send(SessionMessage(message))
        except anyio.ClosedResourceError:
            await anyio.lowlevel.checkpoint()

    async def stdout_writer() -> None:
        try:
            async with write_stream_reader:
                async for session_message in write_stream_reader:
                    json_str = session_message.message.model_dump_json(
                        by_alias=True, exclude_none=True
                    )
                    w_transport.write((json_str + "\n").encode("utf-8"))
        except anyio.ClosedResourceError:
            await anyio.lowlevel.checkpoint()

    async with anyio.create_task_group() as tg:
        tg.start_soon(stdin_reader)
        tg.start_soon(stdout_writer)
        try:
            yield read_stream, write_stream
        finally:
            # The caller's `async with stdio_server()` block is exiting —
            # close write_stream's *send* side here (idempotent even if the
            # caller already closed it) so stdout_writer's `async for` drains
            # anything the caller queued during its own block and then
            # returns. Without this, a caller that never closes write_stream
            # itself leaves this task group hanging forever at context exit,
            # waiting on a writer that only stops once write_stream is closed.
            # Doing the close *after* the yield returns — i.e.
            # only once the caller's own body has finished running — avoids
            # racing an in-flight `write_stream.send()` the caller is still
            # making inside its block.
            await write_stream.aclose()

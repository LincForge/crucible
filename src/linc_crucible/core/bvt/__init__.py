"""BVT (Build Verification Test) runner — deterministic device testing.

Pure Python test runner that uses linc-nerve APIs to deploy, interact with,
and verify apps on real devices. No AI tokens required for basic assertions.

Usage:
    from linc_crucible.core.bvt import BVTRunner, load_bvt_dir
    suite = load_bvt_dir(Path("myapp/.bvt"))
    runner = BVTRunner(nerve=NerveAdapter(), output_dir=Path("/tmp/bvt"))
    result = runner.run_suite(suite, device="emulator-5554")

For the MCP surface: see linc_crucible.mcp_server.crucible_bvt.
"""

from linc_crucible.core.bvt.assertions import (
    AssertResult,
    assert_element,
    assert_text,
)
from linc_crucible.core.bvt.nerve_adapter import NerveAdapter
from linc_crucible.core.bvt.runner import (
    BVTRunner,
    BVTRunResult,
    BVTStepResult,
    BVTTestResult,
)
from linc_crucible.core.bvt.schema import (
    BVTStep,
    BVTSuite,
    BVTTest,
    load_bvt_dir,
)

__all__ = [
    "AssertResult",
    "BVTRunResult",
    "BVTRunner",
    "BVTStep",
    "BVTStepResult",
    "BVTSuite",
    "BVTTest",
    "BVTTestResult",
    "NerveAdapter",
    "assert_element",
    "assert_text",
    "load_bvt_dir",
]

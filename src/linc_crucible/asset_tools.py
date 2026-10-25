import asyncio
import json

from linc_crucible.core.asset_config import AssetExtractorConfig
from linc_crucible.core.asset_extractor import extract_assets
from linc_crucible.core.tool_args import extract_assets_args_error
from linc_crucible.server_context import _observed_tool, server


async def _extract_assets_json(video_path: str, max_assets: int) -> str:
    """Run extract_assets off-thread, mapping known errors to a JSON error payload.

    Kept separate from the thin MCP tool so the tool body stays a single delegate
    call (the error-mapping branch lives here, not on the decorated boundary).
    """
    args_error = extract_assets_args_error(video_path, max_assets)
    if args_error:
        return args_error
    config = AssetExtractorConfig(max_assets=max_assets)
    try:
        result = await asyncio.to_thread(extract_assets, video_path, config)
    except (FileNotFoundError, ValueError) as exc:
        return json.dumps({"error": str(exc), "assets": [], "fallback_used": False})
    return json.dumps(result)


@server.tool()
@_observed_tool
async def crucible_extract_assets(video_path: str, max_assets: int = 5) -> str:
    """Extract AI-ranked marketing screenshots (with variants) from a video.

    Samples frames from a screen recording, has the AI adapter rank the best
    marketing-grade shots, and writes each pick plus social/store-sized
    variants and a manifest.json into an assets directory next to the video.

    USE FOR: turning a nerve screen recording into app-store / social assets,
        seeding visual-baseline candidates from a walkthrough video.
    NOT FOR: golden-baseline management (use crucible_baseline), or grabbing a
        single live screenshot — capture that directly via nerve instead.
    TIP: `fallback_used: true` in the result means AI ranking was unavailable
        and evenly-spaced frames were picked instead — re-run once the AI
        adapter is reachable if you want curated picks.

    Args:
        video_path: Path to the source video (e.g. an .mp4 from nerve_record_stop).
        max_assets: Maximum number of screenshots to extract (default 5).

    Returns:
        JSON string: {assets: [{filename, rationale, variants}], fallback_used,
        manifest_path} on success; {error, assets: [], fallback_used: false}
        when the video is missing or unreadable.
    """
    return await _extract_assets_json(video_path, max_assets)

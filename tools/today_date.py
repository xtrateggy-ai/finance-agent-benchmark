# tools/today_date.py
import datetime
from zoneinfo import ZoneInfo

async def get_today_date(
    date_format: str = "iso",
    timezone: str = "UTC"
) -> dict:
    """
    Returns today's date in various formats.
    
    Args:
        format (str): 
            "iso"      → 2025-11-23
            "full"     → Sunday, November 23, 2025
            "ymd"      → 20251123
            "mdy"      → 11/23/2025
            "timestamp"→ 2025-11-23T14:30:22.123456+00:00
        timezone (str): IANA timezone, e.g. "America/New_York", "Europe/London", "UTC"
    
    Returns:
        dict with all formats + metadata
    """
    try:
        tz = ZoneInfo(timezone)
    except Exception:
        tz = ZoneInfo("UTC")  # fallback

    now = datetime.datetime.now(tz)

    # Pre-compute common formats
    iso_date = now.date().isoformat()                   # 2025-11-23
    full_date = now.strftime("%A, %B %d, %Y")           # Sunday, November 23, 2025
    ymd = now.strftime("%Y%m%d")                        # 20251123
    mdy = now.strftime("%m/%d/%Y")                      # 11/23/2025
    timestamp = now.isoformat(timespec='microseconds')  # 2025-11-23T14:30:22.123456+00:00

    result = {
        "today_iso": iso_date,
        "today_full": full_date,
        "today_ymd": ymd,
        "today_mdy": mdy,
        "timestamp": timestamp,
        "timezone": str(tz),
        "utc_offset": now.utcoffset().total_seconds() / 3600 if now.utcoffset() else 0,
    }

    # Return only the requested format if specified
    if date_format == "iso":
        return {"today": result["today_iso"]}
    elif date_format == "full":
        return {"today": result["today_full"]}
    elif date_format == "ymd":
        return {"today": result["today_ymd"]}
    elif date_format == "mdy":
        return {"today": result["today_mdy"]}
    elif date_format == "timestamp":
        return {"today": result["timestamp"]}

    # Default: return everything
    return result
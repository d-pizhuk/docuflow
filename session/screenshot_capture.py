from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import mss
import mss.tools


@dataclass
class CapturedScreenshot:
    path: Path
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ScreenshotCapture:
    def __init__(self, output_dir: Path):
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._index = 0

    def capture(self) -> CapturedScreenshot:
        ts = datetime.now(timezone.utc)
        filename = f"screenshot_{self._index:03d}_{ts.strftime('%Y%m%d_%H%M%S')}.png"
        path = self._output_dir / filename
        self._index += 1

        with mss.mss() as sct:
            shot = sct.grab(sct.monitors[0])
            mss.tools.to_png(shot.rgb, shot.size, output=str(path))

        return CapturedScreenshot(path=path, timestamp=ts)
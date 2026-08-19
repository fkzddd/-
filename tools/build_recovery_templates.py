"""从用户提供的 1920x1080 MuMu 截图中裁剪自启动恢复流程模板。"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


EMU_LEFT = 325
EMU_TOP = 170
EMU_WIDTH = 1920
EMU_HEIGHT = 1080


def read_image(path: Path) -> np.ndarray:
    encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取截图：{path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError(f"无法编码模板：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())


def crop_emulator_template(
    source: Path,
    output: Path,
    rect: tuple[int, int, int, int],
) -> None:
    screenshot = read_image(source)
    emulator = screenshot[
        EMU_TOP : EMU_TOP + EMU_HEIGHT,
        EMU_LEFT : EMU_LEFT + EMU_WIDTH,
    ]
    if emulator.shape[:2] != (EMU_HEIGHT, EMU_WIDTH):
        raise RuntimeError(
            f"截图中的 MuMu 区域尺寸异常：{source} -> {emulator.shape[1]}x{emulator.shape[0]}"
        )

    x, y, width, height = rect
    template = emulator[y : y + height, x : x + width]
    if template.shape[:2] != (height, width):
        raise RuntimeError(f"模板裁剪区域越界：{output} rect={rect}")
    write_image(output, template)
    print(f"已生成 {output.name}: {width}x{height}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--start-page", type=Path, required=True)
    parser.add_argument("--character", type=Path, required=True)
    parser.add_argument("--main-ui", type=Path, required=True)
    parser.add_argument("--commercial-page", type=Path, required=True)
    parser.add_argument("--shop-page", type=Path, required=True)
    parser.add_argument("--server-character-page", type=Path)
    args = parser.parse_args()

    crops = (
        (args.launcher, "launcher_game_icon.png", (1550, 490, 190, 170)),
        (args.character, "character_start_button.png", (1680, 850, 235, 225)),
        (args.main_ui, "commercial_street_button.png", (1010, 15, 190, 165)),
        (args.commercial_page, "shop_tab.png", (1790, 350, 130, 230)),
        (args.shop_page, "my_following_tab.png", (20, 270, 340, 175)),
    )
    for source, filename, rect in crops:
        crop_emulator_template(source, args.output_dir / filename, rect)
    if args.server_character_page is not None:
        crop_emulator_template(
            args.server_character_page,
            args.output_dir / "server_first_character.png",
            (620, 220, 200, 220),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

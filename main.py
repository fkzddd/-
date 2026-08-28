"""
MuMu 模拟器 + ADB + OpenCV 模板匹配学习示例。

用途：
1. 确认目标商品仍被选中，并检测其底部红色“公示期”文字是否消失；
2. 红字连续多帧不存在后，定位并点击购买按钮；
3. 等待确认弹窗，定位并点击确认下单按钮；
4. 提供轮询间隔、错误重试、点击防抖、日志和事件截图。
5. 游戏异常退出到 MuMu 桌面时，按配置的等待时间自动恢复到“我的关注”页面。

重要提示：本代码仅用于你有权测试的本地/测试环境。请遵守目标应用的服务条款，
不要用于绕过验证码、访问控制、反自动化机制或其他平台保护措施。
"""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional

import cv2
import mss
import numpy as np
from ctypes import wintypes


def read_cv_image(path: Path, flags: int) -> Optional[np.ndarray]:
    """
    兼容 Windows 中文路径的 OpenCV 读取。

    某些 Windows OpenCV 构建的 cv2.imread 无法直接处理中文路径，因此先由
    pathlib 读取字节，再交给 cv2.imdecode 解码。
    """

    try:
        encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(encoded, flags)


def write_cv_image(path: Path, image: np.ndarray) -> bool:
    """兼容 Windows 中文路径的 OpenCV 写入。"""

    extension = path.suffix.lower() or ".png"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded.tobytes())
    except OSError:
        return False
    return True


@dataclass(frozen=True)
class Region:
    """屏幕中的矩形框选区域，坐标单位为像素。"""

    x: int
    y: int
    width: int
    height: int

    @classmethod
    def from_value(cls, value: Optional[dict[str, Any]]) -> Optional["Region"]:
        if value is None:
            return None
        region = cls(
            x=int(value["x"]),
            y=int(value["y"]),
            width=int(value["width"]),
            height=int(value["height"]),
        )
        if region.x < 0 or region.y < 0 or region.width <= 0 or region.height <= 0:
            raise ValueError(f"无效的框选区域：{region}")
        return region


@dataclass(frozen=True)
class TemplateSpec:
    """单个模板的路径、匹配阈值和可选搜索区域。"""

    name: str
    path: Path
    threshold: float
    region: Optional[Region]

    @classmethod
    def from_dict(
        cls, name: str, value: dict[str, Any], config_dir: Path
    ) -> "TemplateSpec":
        template_path = Path(value["path"])
        if not template_path.is_absolute():
            template_path = (config_dir / template_path).resolve()

        threshold = float(value.get("threshold", 0.88))
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"模板 {name} 的 threshold 必须在 (0, 1] 之间")

        return cls(
            name=name,
            path=template_path,
            threshold=threshold,
            region=Region.from_value(value.get("region")),
        )


@dataclass(frozen=True)
class MatchResult:
    """一次模板匹配的结果，坐标均基于完整屏幕。"""

    matched: bool
    score: float
    center: tuple[int, int]
    top_left: tuple[int, int]
    bottom_right: tuple[int, int]


@dataclass(frozen=True)
class ColorPresenceSpec:
    """在固定 ROI 中用 HSV 范围判断某种界面颜色是否存在。"""

    name: str
    region: Region
    hsv_lower: tuple[int, int, int]
    hsv_upper: tuple[int, int, int]
    minimum_pixels: int

    @classmethod
    def from_dict(cls, name: str, value: dict[str, Any]) -> "ColorPresenceSpec":
        region = Region.from_value(value.get("region"))
        if region is None:
            raise ValueError(f"颜色检测项 {name} 必须配置 region")

        lower_values = tuple(int(item) for item in value["hsv_lower"])
        upper_values = tuple(int(item) for item in value["hsv_upper"])
        if len(lower_values) != 3 or len(upper_values) != 3:
            raise ValueError(f"颜色检测项 {name} 的 hsv_lower/hsv_upper 必须各有 3 个值")

        lower = (lower_values[0], lower_values[1], lower_values[2])
        upper = (upper_values[0], upper_values[1], upper_values[2])
        for index, (low, high, maximum) in enumerate(
            zip(lower, upper, (179, 255, 255))
        ):
            if not 0 <= low <= high <= maximum:
                raise ValueError(
                    f"颜色检测项 {name} 的 HSV 第 {index + 1} 通道范围无效："
                    f"{low}..{high}"
                )

        minimum_pixels = int(value["minimum_pixels"])
        if minimum_pixels <= 0:
            raise ValueError(f"颜色检测项 {name} 的 minimum_pixels 必须大于 0")

        return cls(
            name=name,
            region=region,
            hsv_lower=lower,
            hsv_upper=upper,
            minimum_pixels=minimum_pixels,
        )


@dataclass(frozen=True)
class ColorPresenceResult:
    """颜色检测结果；present 表示命中像素数达到配置阈值。"""

    present: bool
    pixel_count: int
    pixel_ratio: float
    top_left: tuple[int, int]
    bottom_right: tuple[int, int]


@dataclass(frozen=True)
class AppConfig:
    config_dir: Path
    adb_path: str
    device_serial: Optional[str]
    connect_on_start: bool
    adb_timeout_seconds: float
    screenshot_mode: str
    capture_source: str
    window_title_keywords: tuple[str, ...]
    window_geometry_refresh_seconds: float
    window_fallback_to_adb: bool
    window_restore_if_minimized: bool
    expected_screen_width: int
    expected_screen_height: int

    dry_run: bool
    auto_confirm: bool
    confirm_click_point: tuple[int, int]
    stacked_purchase_click_point: tuple[int, int]
    max_orders_per_run: int
    poll_interval_seconds: float
    stacked_detection_poll_interval_seconds: float
    post_purchase_wait_seconds: float
    stacked_second_purchase_delay_seconds: float
    publicity_end_purchase_delay_seconds: float
    required_consecutive_publicity_absent_matches: int
    purchase_click_cooldown_seconds: float
    confirm_click_cooldown_seconds: float
    minimum_gap_between_any_clicks_seconds: float
    after_order_pause_seconds: float
    error_retry_limit: int
    error_retry_initial_delay_seconds: float
    error_retry_max_delay_seconds: float
    pause_after_retry_limit_seconds: float
    status_log_every_n_polls: int

    recovery_enabled: bool
    recovery_home_check_interval_seconds: float
    recovery_poll_interval_seconds: float
    recovery_action_cooldown_seconds: float
    recovery_start_page_click_point: tuple[int, int]
    recovery_server_select_click_point: tuple[int, int]
    launcher_game_icon_click_point: tuple[int, int]
    recovery_iknow_click_point: tuple[int, int]
    recovery_iknow_wait_seconds: float
    recovery_launch_to_start_page_seconds: float
    recovery_server_select_to_character_seconds: float
    recovery_server_character_to_start_page_seconds: float
    recovery_start_page_to_character_seconds: float
    recovery_character_to_main_ui_seconds: float
    recovery_commercial_to_shop_seconds: float
    recovery_shop_to_following_seconds: float
    recovery_following_to_resume_seconds: float
    recovery_step_timeout_seconds: float

    log_level: str
    log_file: Path
    save_event_screenshots: bool
    event_screenshot_dir: Path

    selection_guard: ColorPresenceSpec
    publicity_red_text: ColorPresenceSpec
    my_following_selected_guard: ColorPresenceSpec
    purchase_template: TemplateSpec
    stacked_purchase_template: TemplateSpec
    confirm_template: TemplateSpec
    launcher_game_icon_template: TemplateSpec
    server_first_character_template: TemplateSpec
    character_start_template: TemplateSpec
    commercial_street_template: TemplateSpec
    shop_tab_template: TemplateSpec
    my_following_template: TemplateSpec

    @classmethod
    def load(cls, path: Path) -> "AppConfig":
        path = path.resolve()
        raw = json.loads(path.read_text(encoding="utf-8"))
        config_dir = path.parent

        adb = raw.get("adb", {})
        capture = raw.get("capture", {})
        screen = raw.get("screen", {})
        runtime = raw.get("runtime", {})
        recovery = raw.get("recovery", {})
        log_config = raw.get("logging", {})
        status_detection = raw["status_detection"]
        templates = raw["templates"]
        start_page_click_point = recovery.get(
            "start_page_click_point", {"x": 960, "y": 540}
        )
        server_select_click_point = recovery.get(
            "server_select_click_point", {"x": 1140, "y": 917}
        )
        launcher_game_icon_click_point = recovery.get(
            "launcher_game_icon_click_point", {"x": 1442, "y": 224}
        )
        recovery_iknow_click_point = recovery.get(
            "recovery_iknow_click_point", {"x": 963, "y": 986}
        )
        recovery_iknow_wait_seconds = float(
            recovery.get("recovery_iknow_wait_seconds", 30.0)
        )
        confirm_click_point = runtime.get(
            "confirm_click_point", {"x": 1095, "y": 660}
        )
        stacked_purchase_click_point = runtime.get(
            "stacked_purchase_click_point", {"x": 1095, "y": 820}
        )

        log_file = Path(log_config.get("file", "logs/automation.log"))
        if not log_file.is_absolute():
            log_file = (config_dir / log_file).resolve()

        event_dir = Path(log_config.get("event_screenshot_dir", "logs/events"))
        if not event_dir.is_absolute():
            event_dir = (config_dir / event_dir).resolve()

        config = cls(
            config_dir=config_dir,
            adb_path=str(adb.get("path", "adb")),
            device_serial=adb.get("device_serial") or None,
            connect_on_start=bool(adb.get("connect_on_start", False)),
            adb_timeout_seconds=float(adb.get("command_timeout_seconds", 10.0)),
            screenshot_mode=str(adb.get("screenshot_mode", "raw")).lower(),
            capture_source=str(capture.get("source", "window")).lower(),
            window_title_keywords=tuple(
                str(item) for item in capture.get(
                    "window_title_keywords", ["MuMu", "模拟器"]
                )
            ),
            window_geometry_refresh_seconds=float(
                capture.get("geometry_refresh_seconds", 1.0)
            ),
            window_fallback_to_adb=bool(capture.get("fallback_to_adb", True)),
            window_restore_if_minimized=bool(
                capture.get("restore_if_minimized", True)
            ),
            expected_screen_width=int(screen.get("expected_width", 1920)),
            expected_screen_height=int(screen.get("expected_height", 1080)),
            dry_run=bool(runtime.get("dry_run", True)),
            auto_confirm=bool(runtime.get("auto_confirm", True)),
            confirm_click_point=(
                int(confirm_click_point.get("x", 1095)),
                int(confirm_click_point.get("y", 660)),
            ),
            stacked_purchase_click_point=(
                int(stacked_purchase_click_point.get("x", 1095)),
                int(stacked_purchase_click_point.get("y", 820)),
            ),
            max_orders_per_run=int(runtime.get("max_orders_per_run", 1)),
            poll_interval_seconds=float(runtime.get("poll_interval_seconds", 1.0)),
            stacked_detection_poll_interval_seconds=float(
                runtime.get("stacked_detection_poll_interval_seconds", 0.02)
            ),
            post_purchase_wait_seconds=float(
                runtime.get("post_purchase_wait_seconds", 0.8)
            ),
            stacked_second_purchase_delay_seconds=float(
                runtime.get("stacked_second_purchase_delay_seconds", 0.6)
            ),
            publicity_end_purchase_delay_seconds=float(
                runtime.get("publicity_end_purchase_delay_seconds", 0.0)
            ),
            required_consecutive_publicity_absent_matches=int(
                runtime.get("required_consecutive_publicity_absent_matches", 1)
            ),
            purchase_click_cooldown_seconds=float(
                runtime.get("purchase_click_cooldown_seconds", 1.0)
            ),
            confirm_click_cooldown_seconds=float(
                runtime.get("confirm_click_cooldown_seconds", 1.0)
            ),
            minimum_gap_between_any_clicks_seconds=float(
                runtime.get("minimum_gap_between_any_clicks_seconds", 0.8)
            ),
            after_order_pause_seconds=float(
                runtime.get("after_order_pause_seconds", 1.0)
            ),
            error_retry_limit=int(runtime.get("error_retry_limit", 5)),
            error_retry_initial_delay_seconds=float(
                runtime.get("error_retry_initial_delay_seconds", 2.0)
            ),
            error_retry_max_delay_seconds=float(
                runtime.get("error_retry_max_delay_seconds", 30.0)
            ),
            pause_after_retry_limit_seconds=float(
                runtime.get("pause_after_retry_limit_seconds", 60.0)
            ),
            status_log_every_n_polls=max(
                1, int(runtime.get("status_log_every_n_polls", 10))
            ),
            recovery_enabled=bool(recovery.get("enabled", True)),
            recovery_home_check_interval_seconds=float(
                recovery.get("home_check_interval_seconds", 2.0)
            ),
            recovery_poll_interval_seconds=float(
                recovery.get("poll_interval_seconds", 1.0)
            ),
            recovery_action_cooldown_seconds=float(
                recovery.get("action_cooldown_seconds", 5.0)
            ),
            recovery_start_page_click_point=(
                int(start_page_click_point.get("x", 960)),
                int(start_page_click_point.get("y", 540)),
            ),
            recovery_server_select_click_point=(
                int(server_select_click_point.get("x", 1140)),
                int(server_select_click_point.get("y", 917)),
            ),
            launcher_game_icon_click_point=(
                int(launcher_game_icon_click_point.get("x", 1442)),
                int(launcher_game_icon_click_point.get("y", 224)),
            ),
            recovery_iknow_click_point=(
                int(recovery_iknow_click_point.get("x", 963)),
                int(recovery_iknow_click_point.get("y", 986)),
            ),
            recovery_iknow_wait_seconds=recovery_iknow_wait_seconds,
            recovery_launch_to_start_page_seconds=float(
                recovery.get("launch_to_start_page_seconds", 30.0)
            ),
            recovery_server_select_to_character_seconds=float(
                recovery.get("server_select_to_character_seconds", 10.0)
            ),
            recovery_server_character_to_start_page_seconds=float(
                recovery.get("server_character_to_start_page_seconds", 10.0)
            ),
            recovery_start_page_to_character_seconds=float(
                recovery.get("start_page_to_character_seconds", 30.0)
            ),
            recovery_character_to_main_ui_seconds=float(
                recovery.get("character_to_main_ui_seconds", 60.0)
            ),
            recovery_commercial_to_shop_seconds=float(
                recovery.get("commercial_to_shop_seconds", 10.0)
            ),
            recovery_shop_to_following_seconds=float(
                recovery.get("shop_to_following_seconds", 5.0)
            ),
            recovery_following_to_resume_seconds=float(
                recovery.get("following_to_resume_seconds", 5.0)
            ),
            recovery_step_timeout_seconds=float(
                recovery.get("step_timeout_seconds", 300.0)
            ),
            log_level=str(log_config.get("level", "INFO")).upper(),
            log_file=log_file,
            save_event_screenshots=bool(
                log_config.get("save_event_screenshots", True)
            ),
            event_screenshot_dir=event_dir,
            selection_guard=ColorPresenceSpec.from_dict(
                "selection_guard", status_detection["selection_guard"]
            ),
            publicity_red_text=ColorPresenceSpec.from_dict(
                "publicity_red_text", status_detection["publicity_red_text"]
            ),
            my_following_selected_guard=ColorPresenceSpec.from_dict(
                "my_following_selected_guard",
                status_detection.get(
                    "my_following_selected_guard",
                    {
                        "region": {
                            "x": 240,
                            "y": 350,
                            "width": 120,
                            "height": 60,
                        },
                        "hsv_lower": [95, 130, 190],
                        "hsv_upper": [104, 180, 255],
                        "minimum_pixels": 5000,
                    },
                ),
            ),
            purchase_template=TemplateSpec.from_dict(
                "purchase_button", templates["purchase_button"], config_dir
            ),
            stacked_purchase_template=TemplateSpec.from_dict(
                "stacked_purchase_button",
                templates["stacked_purchase_button"],
                config_dir,
            ),
            confirm_template=TemplateSpec.from_dict(
                "confirm_order_button", templates["confirm_order_button"], config_dir
            ),
            launcher_game_icon_template=TemplateSpec.from_dict(
                "launcher_game_icon", templates["launcher_game_icon"], config_dir
            ),
            server_first_character_template=TemplateSpec.from_dict(
                "server_first_character",
                templates["server_first_character"],
                config_dir,
            ),
            character_start_template=TemplateSpec.from_dict(
                "character_start_button",
                templates["character_start_button"],
                config_dir,
            ),
            commercial_street_template=TemplateSpec.from_dict(
                "commercial_street_button",
                templates["commercial_street_button"],
                config_dir,
            ),
            shop_tab_template=TemplateSpec.from_dict(
                "shop_tab", templates["shop_tab"], config_dir
            ),
            my_following_template=TemplateSpec.from_dict(
                "my_following_tab", templates["my_following_tab"], config_dir
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        positive_values = {
            "poll_interval_seconds": self.poll_interval_seconds,
            "stacked_detection_poll_interval_seconds": (
                self.stacked_detection_poll_interval_seconds
            ),
            "post_purchase_wait_seconds": self.post_purchase_wait_seconds,
            "stacked_second_purchase_delay_seconds": (
                self.stacked_second_purchase_delay_seconds
            ),
            "required_consecutive_publicity_absent_matches": (
                self.required_consecutive_publicity_absent_matches
            ),
            "error_retry_limit": self.error_retry_limit,
            "error_retry_initial_delay_seconds": self.error_retry_initial_delay_seconds,
            "error_retry_max_delay_seconds": self.error_retry_max_delay_seconds,
            "recovery_home_check_interval_seconds": (
                self.recovery_home_check_interval_seconds
            ),
            "recovery_poll_interval_seconds": self.recovery_poll_interval_seconds,
            "recovery_action_cooldown_seconds": (
                self.recovery_action_cooldown_seconds
            ),
            "recovery_step_timeout_seconds": self.recovery_step_timeout_seconds,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"配置项 {name} 必须大于 0")
        if self.expected_screen_width <= 0 or self.expected_screen_height <= 0:
            raise ValueError("screen.expected_width/expected_height 必须大于 0")
        if self.screenshot_mode not in {"raw", "png"}:
            raise ValueError("adb.screenshot_mode 只能是 raw 或 png")
        if self.capture_source not in {"window", "adb"}:
            raise ValueError("capture.source 只能是 window 或 adb")
        if not self.window_title_keywords:
            raise ValueError("capture.window_title_keywords 不能为空")
        if self.window_geometry_refresh_seconds <= 0:
            raise ValueError("capture.geometry_refresh_seconds 必须大于 0")
        if self.max_orders_per_run < 0:
            raise ValueError("max_orders_per_run 不能小于 0；0 表示不限制")
        for name, point in {
            "runtime.confirm_click_point": self.confirm_click_point,
            "runtime.stacked_purchase_click_point": self.stacked_purchase_click_point,
        }.items():
            x, y = point
            if not (
                0 <= x < self.expected_screen_width
                and 0 <= y < self.expected_screen_height
            ):
                raise ValueError(f"{name} 必须位于配置的模拟器画面范围内")
        recovery_click_x, recovery_click_y = self.recovery_start_page_click_point
        if not (
            0 <= recovery_click_x < self.expected_screen_width
            and 0 <= recovery_click_y < self.expected_screen_height
        ):
            raise ValueError(
                "recovery.start_page_click_point 必须位于配置的模拟器画面范围内"
            )
        server_select_x, server_select_y = self.recovery_server_select_click_point
        if not (
            0 <= server_select_x < self.expected_screen_width
            and 0 <= server_select_y < self.expected_screen_height
        ):
            raise ValueError(
                "recovery.server_select_click_point 必须位于配置的模拟器画面范围内"
            )
        nonnegative_values = {
            "recovery_launch_to_start_page_seconds": (
                self.recovery_launch_to_start_page_seconds
            ),
            "recovery_server_select_to_character_seconds": (
                self.recovery_server_select_to_character_seconds
            ),
            "recovery_server_character_to_start_page_seconds": (
                self.recovery_server_character_to_start_page_seconds
            ),
            "recovery_start_page_to_character_seconds": (
                self.recovery_start_page_to_character_seconds
            ),
            "recovery_character_to_main_ui_seconds": (
                self.recovery_character_to_main_ui_seconds
            ),
            "recovery_commercial_to_shop_seconds": (
                self.recovery_commercial_to_shop_seconds
            ),
            "recovery_shop_to_following_seconds": (
                self.recovery_shop_to_following_seconds
            ),
            "recovery_following_to_resume_seconds": (
                self.recovery_following_to_resume_seconds
            ),
        }
        for name, value in nonnegative_values.items():
            if value < 0:
                raise ValueError(f"配置项 {name} 不能小于 0")


def setup_logging(config: AppConfig) -> None:
    """同时输出到控制台和 UTF-8 日志文件。"""

    config.log_file.parent.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, config.log_level, logging.INFO)

    class MillisecondFormatter(logging.Formatter):
        """毫秒级时间戳格式器：%(asctime)s 输出到毫秒。"""

        def formatTime(self, record, datefmt=None):
            from time import strftime, gmtime
            ct = self.converter(record.created)
            base = strftime("%Y-%m-%d %H:%M:%S", ct)
            return f"{base}.{int(record.msecs):03d}"

    formatter = MillisecondFormatter(fmt="%(asctime)s | %(levelname)s | %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(config.log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logging.basicConfig(
        level=level,
        handlers=[console_handler, file_handler],
        force=True,
    )


class AdbClient:
    """只封装本示例需要的 ADB 截图、检查设备和点击操作。"""

    def __init__(self, config: AppConfig) -> None:
        self.adb_path = config.adb_path
        self.device_serial = config.device_serial
        self.timeout_seconds = config.adb_timeout_seconds
        self.screenshot_mode = config.screenshot_mode

    def _base_command(self, include_device: bool = True) -> list[str]:
        command = [self.adb_path]
        if include_device and self.device_serial:
            command.extend(["-s", self.device_serial])
        return command

    def _run(
        self,
        arguments: list[str],
        *,
        include_device: bool = True,
        binary_output: bool = False,
    ) -> subprocess.CompletedProcess[bytes]:
        command = self._base_command(include_device=include_device) + arguments
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout_seconds,
            check=False,
            creationflags=creation_flags,
        )
        if process.returncode != 0:
            stderr = process.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"ADB 命令失败（exit={process.returncode}）：{' '.join(command)}；{stderr}"
            )
        if not binary_output and process.stderr:
            logging.debug(
                "ADB stderr: %s",
                process.stderr.decode("utf-8", errors="replace").strip(),
            )
        return process

    def connect(self) -> None:
        """仅在配置明确开启时尝试 adb connect，适用于 MuMu 的 TCP ADB。"""

        if not self.device_serial:
            raise ValueError("connect_on_start=true 时必须填写 adb.device_serial")
        process = self._run(
            ["connect", self.device_serial], include_device=False, binary_output=False
        )
        message = process.stdout.decode("utf-8", errors="replace").strip()
        logging.info("ADB connect: %s", message or "命令已完成")

    def _get_device_state(self) -> str:
        """读取设备状态；将 ADB 的 offline 错误转换为可恢复状态。"""

        try:
            process = self._run(["get-state"])
        except RuntimeError as error:
            if "device offline" in str(error).lower():
                return "offline"
            raise
        return process.stdout.decode("utf-8", errors="replace").strip()

    def check_device(self) -> None:
        state = self._get_device_state()
        if state == "offline" and self.device_serial:
            # `adb connect` 对已有但离线的 TCP 连接会错误地提示
            # "already connected"。必须先断开该设备，再重新连接。
            for attempt in range(1, 4):
                logging.warning(
                    "ADB 设备离线，正在自动断开并重连（%d/3）：%s",
                    attempt,
                    self.device_serial,
                )
                self._run(
                    ["disconnect", self.device_serial],
                    include_device=False,
                    binary_output=False,
                )
                time.sleep(0.25)
                self.connect()
                time.sleep(0.25 * attempt)
                state = self._get_device_state()
                if state == "device":
                    logging.info("ADB 离线连接已自动恢复：%s", self.device_serial)
                    break

        if state != "device":
            raise RuntimeError(f"ADB 设备状态异常：{state!r}")
        logging.info("ADB 设备已连接：%s", self.device_serial or "默认设备")

    def _capture_png(self) -> np.ndarray:
        process = self._run(
            ["exec-out", "screencap", "-p"], binary_output=True
        )
        data = np.frombuffer(process.stdout, dtype=np.uint8)
        screen = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if screen is None:
            raise RuntimeError("ADB 截图无法解码，请检查模拟器和 ADB 连接")
        return screen

    def _capture_raw(self) -> np.ndarray:
        """读取 MuMu 的 RGBA 原始截图，省去模拟器 PNG 压缩和本地 PNG 解码。"""

        process = self._run(["exec-out", "screencap"], binary_output=True)
        payload = process.stdout
        if len(payload) < 12:
            raise ValueError("ADB 原始截图数据过短")

        width, height, pixel_format = struct.unpack_from("<III", payload, 0)
        if width <= 0 or height <= 0 or pixel_format != 1:
            raise ValueError(
                f"不支持的 ADB 原始截图头：{width}x{height}, format={pixel_format}"
            )

        pixel_bytes = width * height * 4
        header_size = len(payload) - pixel_bytes
        # 不同 Android/MuMu 版本可能使用 12 或 16 字节头部。
        if header_size not in {12, 16}:
            raise ValueError(
                f"ADB 原始截图长度异常：header={header_size}, total={len(payload)}"
            )

        rgba = np.frombuffer(
            payload,
            dtype=np.uint8,
            count=pixel_bytes,
            offset=header_size,
        ).reshape(height, width, 4)
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

    def capture_screen(self) -> np.ndarray:
        """按配置抓取屏幕；raw 不受支持时自动降级到 PNG。"""

        if self.screenshot_mode == "raw":
            try:
                return self._capture_raw()
            except ValueError as error:
                logging.warning("原始截图解析失败，后续改用 PNG：%s", error)
                self.screenshot_mode = "png"
        return self._capture_png()

    def capture_screen_region(self, _region: Region) -> np.ndarray:
        """ADB screencap 不支持服务端裁剪，因此降级路径仍抓取完整画面。"""

        return self.capture_screen()

    def tap(self, x: int, y: int) -> None:
        """通过 Android input 命令点击指定屏幕坐标。"""

        self._run(["shell", "input", "tap", str(int(x)), str(int(y))])


@dataclass(frozen=True)
class DesktopCaptureRegion:
    """Windows 桌面上的 MuMu 游戏渲染区域。"""

    hwnd: int
    title: str
    left: int
    top: int
    width: int
    height: int


class WindowScreenCapture:
    """
    从 Windows 桌面直接抓取 MuMu 可见窗口，ADB 只负责点击。

    自动枚举标题匹配的顶层窗口及其子窗口，选择面积最大且宽高比最接近
    1920:1080 的客户区。这样窗口可以缩放或移动，抓到的画面再统一映射为
    脚本使用的 1920x1080 坐标系。
    """

    def __init__(self, config: AppConfig, adb_fallback: AdbClient) -> None:
        if os.name != "nt":
            raise RuntimeError("窗口捕获模式仅支持 Windows")

        self.config = config
        self.adb_fallback = adb_fallback
        self.user32 = ctypes.windll.user32
        self.user32.IsWindowVisible.argtypes = [wintypes.HWND]
        self.user32.IsWindowVisible.restype = wintypes.BOOL
        self.user32.IsIconic.argtypes = [wintypes.HWND]
        self.user32.IsIconic.restype = wintypes.BOOL
        self.user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        self.user32.ShowWindow.restype = wintypes.BOOL
        self.user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        self.user32.GetWindowTextLengthW.restype = ctypes.c_int
        self.user32.GetWindowTextW.argtypes = [
            wintypes.HWND,
            wintypes.LPWSTR,
            ctypes.c_int,
        ]
        self.user32.GetWindowTextW.restype = ctypes.c_int
        self.user32.GetClientRect.argtypes = [
            wintypes.HWND,
            ctypes.POINTER(wintypes.RECT),
        ]
        self.user32.GetClientRect.restype = wintypes.BOOL
        self.user32.ClientToScreen.argtypes = [
            wintypes.HWND,
            ctypes.POINTER(wintypes.POINT),
        ]
        self.user32.ClientToScreen.restype = wintypes.BOOL
        self._enable_dpi_awareness()
        self._mss = mss.MSS()
        self._region: Optional[DesktopCaptureRegion] = None
        self._region_checked_at = float("-inf")
        self._next_window_retry_at = float("-inf")
        self._using_fallback = False

    @staticmethod
    def _enable_dpi_awareness() -> None:
        """避免 Windows 显示缩放导致窗口坐标与桌面截图像素不一致。"""

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except (AttributeError, OSError):
                pass

    def _get_window_title(self, hwnd: int) -> str:
        length = self.user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return ""
        buffer = ctypes.create_unicode_buffer(length + 1)
        self.user32.GetWindowTextW(hwnd, buffer, length + 1)
        return buffer.value

    def _get_client_region(
        self, hwnd: int, root_title: str
    ) -> Optional[DesktopCaptureRegion]:
        rect = wintypes.RECT()
        if not self.user32.GetClientRect(hwnd, ctypes.byref(rect)):
            return None

        top_left = wintypes.POINT(rect.left, rect.top)
        bottom_right = wintypes.POINT(rect.right, rect.bottom)
        if not self.user32.ClientToScreen(hwnd, ctypes.byref(top_left)):
            return None
        if not self.user32.ClientToScreen(hwnd, ctypes.byref(bottom_right)):
            return None

        width = bottom_right.x - top_left.x
        height = bottom_right.y - top_left.y
        if width <= 0 or height <= 0:
            return None
        return DesktopCaptureRegion(
            hwnd=int(hwnd),
            title=root_title,
            left=int(top_left.x),
            top=int(top_left.y),
            width=int(width),
            height=int(height),
        )

    def _find_region(self) -> DesktopCaptureRegion:
        keywords = tuple(item.casefold() for item in self.config.window_title_keywords)
        roots: list[tuple[int, str]] = []

        enum_windows_type = ctypes.WINFUNCTYPE(
            wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
        )

        def collect_root(hwnd: int, _lparam: int) -> bool:
            if not self.user32.IsWindowVisible(hwnd):
                return True
            title = self._get_window_title(hwnd)
            folded = title.casefold()
            if title and any(keyword in folded for keyword in keywords):
                if self.user32.IsIconic(hwnd):
                    if self.config.window_restore_if_minimized:
                        self.user32.ShowWindow(hwnd, 9)  # SW_RESTORE
                        logging.info("已自动恢复最小化的 MuMu 窗口：%s", title)
                    else:
                        # 不主动恢复或激活窗口，避免自动化运行时抢占用户正在
                        # 操作的桌面焦点。窗口最小化期间由上层自动降级到 ADB 截图。
                        return True
                roots.append((int(hwnd), title))
            return True

        root_callback = enum_windows_type(collect_root)
        self.user32.EnumWindows(root_callback, 0)
        if not roots:
            raise RuntimeError(
                "未找到可见的 MuMu 窗口，标题关键字："
                + ", ".join(self.config.window_title_keywords)
            )

        candidates: list[DesktopCaptureRegion] = []
        child_callback_type = ctypes.WINFUNCTYPE(
            wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
        )

        for root_hwnd, title in roots:
            root_region = self._get_client_region(root_hwnd, title)
            if root_region is not None:
                candidates.append(root_region)

            def collect_child(hwnd: int, _lparam: int, root_title: str = title) -> bool:
                if self.user32.IsWindowVisible(hwnd):
                    region = self._get_client_region(hwnd, root_title)
                    if region is not None:
                        candidates.append(region)
                return True

            child_callback = child_callback_type(collect_child)
            self.user32.EnumChildWindows(root_hwnd, child_callback, 0)

        expected_ratio = (
            self.config.expected_screen_width / self.config.expected_screen_height
        )
        # MuMu 的 Qt 客户区通常是“顶部 36px 工具栏 + 16:9 游戏画面”，
        # 例如窗口客户区 1280x756，真正游戏区域是底部的 1280x720。
        # 若没有独立渲染子窗口，则从客户区自动推导最大 16:9 区域：
        # 竖向多出的部分视为顶部工具栏，横向多出的部分视为右侧工具栏。
        derived: list[DesktopCaptureRegion] = []
        for region in candidates:
            current_ratio = region.width / region.height
            if current_ratio < expected_ratio:
                target_height = round(region.width / expected_ratio)
                if 0 < target_height <= region.height:
                    derived.append(
                        DesktopCaptureRegion(
                            hwnd=region.hwnd,
                            title=region.title,
                            left=region.left,
                            top=region.top + (region.height - target_height),
                            width=region.width,
                            height=target_height,
                        )
                    )
            elif current_ratio > expected_ratio:
                target_width = round(region.height * expected_ratio)
                if 0 < target_width <= region.width:
                    derived.append(
                        DesktopCaptureRegion(
                            hwnd=region.hwnd,
                            title=region.title,
                            left=region.left,
                            top=region.top,
                            width=target_width,
                            height=region.height,
                        )
                    )
        candidates.extend(derived)

        usable = [
            region
            for region in candidates
            if region.width >= 640
            and region.height >= 360
            and abs(region.width / region.height - expected_ratio) / expected_ratio
            <= 0.20
        ]
        if not usable:
            sizes = ", ".join(
                f"{item.width}x{item.height}" for item in candidates[:10]
            )
            raise RuntimeError(f"找到 MuMu 窗口，但没有接近 16:9 的客户区：{sizes}")

        # 先比较宽高比误差，再选择面积最大的候选区域。
        return min(
            usable,
            key=lambda item: (
                abs(item.width / item.height - expected_ratio),
                -(item.width * item.height),
            ),
        )

    def _refresh_region_if_needed(self) -> DesktopCaptureRegion:
        now = time.monotonic()
        if (
            self._region is None
            or now - self._region_checked_at
            >= self.config.window_geometry_refresh_seconds
        ):
            new_region = self._find_region()
            if new_region != self._region:
                logging.info(
                    "窗口捕获区域：title=%s hwnd=%s desktop=(%d,%d,%dx%d)",
                    new_region.title,
                    new_region.hwnd,
                    new_region.left,
                    new_region.top,
                    new_region.width,
                    new_region.height,
                )
            self._region = new_region
            self._region_checked_at = now
        return self._region

    def _capture_window(self) -> np.ndarray:
        region = self._refresh_region_if_needed()
        shot = self._mss.grab(
            {
                "left": region.left,
                "top": region.top,
                "width": region.width,
                "height": region.height,
            }
        )
        # MSS 返回 BGRA；前三个通道已经是 OpenCV 使用的 BGR 顺序。
        screen = np.asarray(shot, dtype=np.uint8)[:, :, :3].copy()
        target_size = (
            self.config.expected_screen_width,
            self.config.expected_screen_height,
        )
        if (screen.shape[1], screen.shape[0]) != target_size:
            screen = cv2.resize(screen, target_size, interpolation=cv2.INTER_LINEAR)
        return screen

    def _capture_window_region(self, requested: Region) -> np.ndarray:
        """只从桌面抓取一个脚本坐标 ROI，并放回同尺寸的空白标准画布。"""

        expected_width = self.config.expected_screen_width
        expected_height = self.config.expected_screen_height
        requested_right = requested.x + requested.width
        requested_bottom = requested.y + requested.height
        if (
            requested.x < 0
            or requested.y < 0
            or requested_right > expected_width
            or requested_bottom > expected_height
        ):
            raise ValueError(f"窗口局部截图 ROI 超出标准画布：{requested}")

        window = self._refresh_region_if_needed()
        scale_x = window.width / expected_width
        scale_y = window.height / expected_height
        desktop_left = window.left + round(requested.x * scale_x)
        desktop_top = window.top + round(requested.y * scale_y)
        desktop_right = window.left + round(requested_right * scale_x)
        desktop_bottom = window.top + round(requested_bottom * scale_y)
        desktop_width = max(1, desktop_right - desktop_left)
        desktop_height = max(1, desktop_bottom - desktop_top)

        shot = self._mss.grab(
            {
                "left": desktop_left,
                "top": desktop_top,
                "width": desktop_width,
                "height": desktop_height,
            }
        )
        roi = np.asarray(shot, dtype=np.uint8)[:, :, :3].copy()
        if (roi.shape[1], roi.shape[0]) != (requested.width, requested.height):
            roi = cv2.resize(
                roi,
                (requested.width, requested.height),
                interpolation=cv2.INTER_LINEAR,
            )

        screen = np.zeros((expected_height, expected_width, 3), dtype=np.uint8)
        screen[
            requested.y:requested_bottom,
            requested.x:requested_right,
        ] = roi
        return screen

    def capture_screen(self) -> np.ndarray:
        now = time.monotonic()
        if now < self._next_window_retry_at and self.config.window_fallback_to_adb:
            return self.adb_fallback.capture_screen()

        try:
            screen = self._capture_window()
            if self._using_fallback:
                logging.info("MuMu 窗口捕获已恢复")
            self._using_fallback = False
            return screen
        except Exception as error:  # noqa: BLE001 - 可配置降级到 ADB 截图
            self._region = None
            self._next_window_retry_at = (
                now + self.config.window_geometry_refresh_seconds
            )
            if not self.config.window_fallback_to_adb:
                raise
            if not self._using_fallback:
                logging.warning("窗口捕获不可用，临时降级到 ADB 截图：%s", error)
            self._using_fallback = True
            return self.adb_fallback.capture_screen()

    def capture_screen_region(self, region: Region) -> np.ndarray:
        """高速路径：正常时只抓指定 ROI，失败时自动回退到 ADB 全屏截图。"""

        now = time.monotonic()
        if now < self._next_window_retry_at and self.config.window_fallback_to_adb:
            return self.adb_fallback.capture_screen()

        try:
            screen = self._capture_window_region(region)
            if self._using_fallback:
                logging.info("MuMu 窗口局部捕获已恢复")
            self._using_fallback = False
            return screen
        except Exception as error:  # noqa: BLE001 - 可配置降级到 ADB 截图
            self._region = None
            self._next_window_retry_at = (
                now + self.config.window_geometry_refresh_seconds
            )
            if not self.config.window_fallback_to_adb:
                raise
            if not self._using_fallback:
                logging.warning("窗口局部捕获不可用，临时降级到 ADB：%s", error)
            self._using_fallback = True
            return self.adb_fallback.capture_screen()


class TemplateMatcher:
    """加载并缓存模板，在完整屏幕或指定 ROI 中执行灰度模板匹配。"""

    def __init__(self, specs: list[TemplateSpec]) -> None:
        self._templates: dict[str, np.ndarray] = {}
        for spec in specs:
            template = read_cv_image(spec.path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                raise FileNotFoundError(
                    f"无法读取模板 {spec.name}：{spec.path}。请先按 README 裁剪模板。"
                )
            self._templates[spec.name] = template
            logging.info(
                "已加载模板 %-22s size=%sx%s threshold=%.3f",
                spec.name,
                template.shape[1],
                template.shape[0],
                spec.threshold,
            )

    def match(self, screen: np.ndarray, spec: TemplateSpec) -> MatchResult:
        screen_height, screen_width = screen.shape[:2]

        # 按钮定位逻辑：如果配置了 region，只在按钮可能出现的区域中搜索，
        # 排除页面其他位置的相似文字或装饰元素，降低误匹配概率。
        if spec.region is None:
            origin_x, origin_y = 0, 0
            search_color = screen
        else:
            region = spec.region
            right = region.x + region.width
            bottom = region.y + region.height
            if right > screen_width or bottom > screen_height:
                raise ValueError(
                    f"模板 {spec.name} 的 region={region} 超出屏幕 "
                    f"{screen_width}x{screen_height}"
                )
            origin_x, origin_y = region.x, region.y
            search_color = screen[region.y:bottom, region.x:right]

        # 先裁剪再转灰度，避免每次按钮匹配都转换整张 1920x1080 截图。
        search_image = cv2.cvtColor(search_color, cv2.COLOR_BGR2GRAY)

        template = self._templates[spec.name]
        template_height, template_width = template.shape[:2]
        search_height, search_width = search_image.shape[:2]
        if template_width > search_width or template_height > search_height:
            raise ValueError(
                f"模板 {spec.name} ({template_width}x{template_height}) 大于搜索区域 "
                f"({search_width}x{search_height})"
            )

        result = cv2.matchTemplate(search_image, template, cv2.TM_CCOEFF_NORMED)
        _, max_score, _, max_location = cv2.minMaxLoc(result)

        left = origin_x + max_location[0]
        top = origin_y + max_location[1]
        right = left + template_width
        bottom = top + template_height
        center = (left + template_width // 2, top + template_height // 2)

        return MatchResult(
            matched=max_score >= spec.threshold,
            score=float(max_score),
            center=center,
            top_left=(left, top),
            bottom_right=(right, bottom),
        )


class ColorPresenceDetector:
    """
    用 HSV 颜色范围检测动态界面文字或选中高亮。

    倒计时数字会不断变化，固定文字模板容易因数字变化而失效；颜色检测只关心
    框选区域里是否仍有足够数量的红色像素，更适合判断整行红字是否消失。
    """

    @staticmethod
    def detect(screen: np.ndarray, spec: ColorPresenceSpec) -> ColorPresenceResult:
        screen_height, screen_width = screen.shape[:2]
        region = spec.region
        right = region.x + region.width
        bottom = region.y + region.height
        if right > screen_width or bottom > screen_height:
            raise ValueError(
                f"颜色检测项 {spec.name} 的 region={region} 超出屏幕 "
                f"{screen_width}x{screen_height}"
            )

        roi = screen[region.y:bottom, region.x:right]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array(spec.hsv_lower, dtype=np.uint8),
            np.array(spec.hsv_upper, dtype=np.uint8),
        )
        pixel_count = int(cv2.countNonZero(mask))
        total_pixels = region.width * region.height
        return ColorPresenceResult(
            present=pixel_count >= spec.minimum_pixels,
            pixel_count=pixel_count,
            pixel_ratio=pixel_count / total_pixels,
            top_left=(region.x, region.y),
            bottom_right=(right, bottom),
        )


class ClickDebouncer:
    """点击触发逻辑的统一防抖器。"""

    def __init__(self, adb: AdbClient, config: AppConfig) -> None:
        self.adb = adb
        self.config = config
        self.last_any_click_at = float("-inf")
        self.last_action_click_at: dict[str, float] = {}
        self.blocked_log_counts: dict[str, int] = {}

    def _should_log_blocked(self, key: str) -> bool:
        """重复拦截首次提示，之后每配置的 N 次提示一次。"""

        count = self.blocked_log_counts.get(key, 0) + 1
        self.blocked_log_counts[key] = count
        return count == 1 or count % self.config.status_log_every_n_polls == 0

    def click(
        self,
        action: str,
        point: tuple[int, int],
        action_cooldown_seconds: float,
    ) -> bool:
        """
        同时应用两层防抖：
        1. 任意两次点击之间必须满足 minimum_gap；
        2. 同一种动作（购买/确认）在 action_cooldown 内只允许一次。
        """

        now = time.monotonic()
        since_any = now - self.last_any_click_at
        since_action = now - self.last_action_click_at.get(action, float("-inf"))

        if since_any < self.config.minimum_gap_between_any_clicks_seconds:
            key = f"global:{action}"
            if self._should_log_blocked(key):
                logging.debug(
                    "%s 点击被全局最小间隔拦截，还需等待 %.2fs",
                    action,
                    self.config.minimum_gap_between_any_clicks_seconds - since_any,
                )
            return False
        if since_action < action_cooldown_seconds:
            key = f"action:{action}"
            if self._should_log_blocked(key):
                logging.warning(
                    "%s 点击被动作冷却拦截，还需等待 %.2fs",
                    action,
                    action_cooldown_seconds - since_action,
                )
            return False

        if self.config.dry_run:
            logging.info("[DRY-RUN] 将点击 %s：x=%d, y=%d", action, *point)
        else:
            self.adb.tap(*point)
            logging.info("已点击 %s：x=%d, y=%d", action, *point)

        self.last_any_click_at = now
        self.last_action_click_at[action] = now
        self.blocked_log_counts.pop(f"global:{action}", None)
        self.blocked_log_counts.pop(f"action:{action}", None)
        return True


class AutomationState(Enum):
    WAITING_FOR_STATUS = auto()
    AFTER_ORDER_COOLDOWN = auto()
    RECOVERING_GAME = auto()


class RecoveryStep(Enum):
    """游戏异常退出后，从 MuMu 桌面逐步返回目标商品页面。"""

    WAITING_FOR_IKNOW = auto()
    WAITING_FOR_SERVER_SELECT = auto()
    WAITING_FOR_SERVER_CHARACTER = auto()
    WAITING_FOR_START_PAGE = auto()
    WAITING_FOR_CHARACTER_START = auto()
    WAITING_FOR_MAIN_UI = auto()
    WAITING_FOR_SHOP_TAB = auto()
    WAITING_FOR_FOLLOWING_TAB = auto()
    WAITING_FOR_TARGET_PAGE = auto()


class PurchaseAutomation:
    """把页面状态判断和点击触发组织成明确的状态机。"""

    def __init__(
        self,
        config: AppConfig,
        adb: AdbClient,
        screen_source: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.adb = adb
        self.screen_source = screen_source or adb
        self.matcher = TemplateMatcher(
            [
                config.purchase_template,
                config.stacked_purchase_template,
                config.launcher_game_icon_template,
                config.server_first_character_template,
                config.character_start_template,
                config.commercial_street_template,
                config.shop_tab_template,
                config.my_following_template,
            ]
        )
        self.color_detector = ColorPresenceDetector()
        self.clicker = ClickDebouncer(adb, config)

        self.state = AutomationState.WAITING_FOR_STATUS
        self.publicity_absent_hits = 0
        self.poll_count = 0
        self.cooldown_deadline = 0.0
        self.completed_orders = 0
        self.consecutive_errors = 0
        self.repeated_log_counts: dict[str, int] = {}

        # 公示期红色像素变化观测：记录上一次红色像素值，用于精确捕捉
        # 倒计时分钟切换（数字变化 → 像素变化）的时刻，判断本地倒计时漂移。
        self.last_publicity_pixels: Optional[int] = None

        # 自启动恢复流程只在正常目标页面的选中框消失后低频检查，因此不会在
        # 正常公示期轮询中额外抓取图标区域，也不会拖慢购买条件检测。
        self.next_recovery_home_check_at = 0.0
        self.recovery_step: Optional[RecoveryStep] = None
        self.recovery_action_earliest_at = 0.0
        self.recovery_step_started_at = 0.0
        self.recovery_last_timeout_log_at = 0.0

        # 单次下单流程的高精度时间点。使用 monotonic() 避免系统时间调整
        # 影响耗时结果；每次确认点击完成或流程中断后都会清零。
        self.order_status_ready_at: Optional[float] = None
        self.purchase_button_matched_at: Optional[float] = None
        self.purchase_clicked_at: Optional[float] = None
        self.order_flow_name: Optional[str] = None

        # 叠挂流程包含三次点击：页面购买、弹窗购买、最终确认。
        # 这两个时间点独立于识别计时，普通确认超时后仍会保留。
        self.stacked_first_purchase_clicked_at: Optional[float] = None
        self.stacked_second_purchase_clicked_at: Optional[float] = None

    def _save_event_image(
        self, screen: np.ndarray, result: MatchResult, event_name: str
    ) -> None:
        if not self.config.save_event_screenshots:
            return

        self.config.event_screenshot_dir.mkdir(parents=True, exist_ok=True)
        marked = screen.copy()
        cv2.rectangle(marked, result.top_left, result.bottom_right, (0, 255, 0), 2)
        cv2.circle(marked, result.center, 5, (0, 0, 255), -1)
        cv2.putText(
            marked,
            f"{event_name} score={result.score:.3f}",
            (result.top_left[0], max(20, result.top_left[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        millis = int((time.time() % 1) * 1000)
        path = self.config.event_screenshot_dir / (
            f"{timestamp}_{millis:03d}_{event_name}.png"
        )
        if not write_cv_image(path, marked):
            logging.warning("事件截图保存失败：%s", path)
            return
        logging.debug("已保存事件截图：%s", path)

    def _save_color_event_image(
        self,
        screen: np.ndarray,
        result: ColorPresenceResult,
        event_name: str,
        color: tuple[int, int, int],
    ) -> None:
        """保存颜色检测 ROI，方便核对“红字消失”是否发生在正确商品卡片。"""

        if not self.config.save_event_screenshots:
            return

        marked = screen.copy()
        cv2.rectangle(marked, result.top_left, result.bottom_right, color, 3)
        cv2.putText(
            marked,
            f"{event_name} pixels={result.pixel_count} ratio={result.pixel_ratio:.4f}",
            (result.top_left[0], max(20, result.top_left[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        millis = int((time.time() % 1) * 1000)
        path = self.config.event_screenshot_dir / (
            f"{timestamp}_{millis:03d}_{event_name}.png"
        )
        if not write_cv_image(path, marked):
            logging.warning("颜色检测事件截图保存失败：%s", path)
            return
        logging.debug("已保存颜色检测事件截图：%s", path)

    def _order_limit_reached(self) -> bool:
        limit = self.config.max_orders_per_run
        return limit > 0 and self.completed_orders >= limit

    def _reset_order_timing(self) -> None:
        """清空当前下单流程的计时点，防止下一单沿用上一单的数据。"""

        self.order_status_ready_at = None
        self.purchase_button_matched_at = None
        self.purchase_clicked_at = None
        self.order_flow_name = None

    def _reset_stacked_click_timing(self) -> None:
        """清空叠挂三次点击的跨状态时间点。"""

        self.stacked_first_purchase_clicked_at = None
        self.stacked_second_purchase_clicked_at = None

    def _log_stacked_click_timing(self, confirm_clicked_at: float) -> None:
        """输出页面购买→叠挂购买→最终确认三次点击的实际间隔。"""

        second_clicked_at = self.stacked_second_purchase_clicked_at
        if second_clicked_at is None:
            logging.warning("叠挂点击计时缺少第二次购买时间，无法计算三次点击间隔")
            self._reset_stacked_click_timing()
            return

        second_to_confirm = confirm_clicked_at - second_clicked_at
        first_clicked_at = self.stacked_first_purchase_clicked_at
        if first_clicked_at is None:
            logging.info(
                "叠挂点击间隔 | 第二次购买→确认=%.3fs | "
                "第一次购买时间缺失，无法计算总间隔",
                second_to_confirm,
            )
            self._reset_stacked_click_timing()
            return

        first_to_second = second_clicked_at - first_clicked_at
        first_to_confirm = confirm_clicked_at - first_clicked_at
        logging.info(
            "叠挂三次点击间隔 | 第一次购买→第二次购买=%.3fs | "
            "第二次购买→确认=%.3fs | 第一次购买→确认总计=%.3fs",
            first_to_second,
            second_to_confirm,
            first_to_confirm,
        )
        self._reset_stacked_click_timing()

    @staticmethod
    def _wait_until(target_at: float) -> None:
        """等待到 monotonic 目标时刻；循环复核避免 Windows sleep 提前返回。"""

        remaining = target_at - time.monotonic()
        while remaining > 0:
            time.sleep(remaining)
            remaining = target_at - time.monotonic()

    def _finish_fixed_click_sequence(self) -> bool:
        """短时检测叠挂类型，后续按钮均按固定坐标和绝对时刻点击。"""

        first_clicked_at = self.stacked_first_purchase_clicked_at
        if first_clicked_at is None:
            raise RuntimeError("首次购买点击时间缺失，无法执行固定点击序列")

        if not self.config.auto_confirm:
            logging.warning("auto_confirm=false，已完成首次购买点击，程序结束")
            return True

        stacked_detected = False
        # 叠挂检测截止/叠挂购买点击时刻锚定"公示期结束时刻"，使叠挂点击与
        # 确认的绝对刻度不因购买延后（publicity_end_purchase_delay_seconds）而漂移。
        stacked_deadline = (
            (self.order_status_ready_at if self.order_status_ready_at is not None else first_clicked_at)
            + self.config.stacked_second_purchase_delay_seconds
        )
        while time.monotonic() < stacked_deadline:
            stacked_screen = self._capture_region_frame(
                self.config.stacked_purchase_template.region
            )
            stacked = self.matcher.match(
                stacked_screen, self.config.stacked_purchase_template
            )
            if stacked.matched:
                stacked_detected = True
                logging.info(
                    "叠挂购买弹窗已识别：score=%.3f；后续使用固定坐标",
                    stacked.score,
                )
                break
            remaining = stacked_deadline - time.monotonic()
            if remaining > 0:
                time.sleep(
                    min(
                        self.config.stacked_detection_poll_interval_seconds,
                        remaining,
                    )
                )

        stacked_check_finished_at = time.monotonic()

        if stacked_detected:
            self.order_flow_name = "叠挂购买"
            self._wait_until(stacked_deadline)
            clicked = self.clicker.click(
                "叠挂购买按钮",
                self.config.stacked_purchase_click_point,
                self.config.purchase_click_cooldown_seconds,
            )
            if not clicked:
                raise RuntimeError("固定叠挂购买点击被防抖器拦截")
            self.stacked_second_purchase_clicked_at = self.clicker.last_action_click_at[
                "叠挂购买按钮"
            ]
            logging.info(
                "已按固定坐标点击叠挂购买：x=%d, y=%d",
                *self.config.stacked_purchase_click_point,
            )
            confirm_target_at = (
                self.stacked_second_purchase_clicked_at
                + self.config.minimum_gap_between_any_clicks_seconds
            )
        else:
            self.order_flow_name = "公示期购买"
            logging.info(
                "第一次购买后 %.3fs 内未识别到叠挂弹窗，按单件流程执行",
                self.config.stacked_second_purchase_delay_seconds,
            )
            # 确认时刻锚定"公示期结束时刻 + post_purchase_wait_seconds"，
            # 保证从公示期结束到确认的总时长不变；购买命令可能已因
            # publicity_end_purchase_delay_seconds 延后，确认刻度不动。
            confirm_base = (
                self.order_status_ready_at
                if self.order_status_ready_at is not None
                else first_clicked_at
            )
            confirm_target_at = (
                confirm_base + self.config.post_purchase_wait_seconds
            )

        self._wait_until(confirm_target_at)
        clicked = self.clicker.click(
            "确认下单按钮",
            self.config.confirm_click_point,
            self.config.confirm_click_cooldown_seconds,
        )
        if not clicked:
            raise RuntimeError("固定确认点击被防抖器拦截")

        confirm_clicked_at = self.clicker.last_action_click_at["确认下单按钮"]
        logging.info(
            "已按固定坐标点击确认：x=%d, y=%d",
            *self.config.confirm_click_point,
        )
        if stacked_detected:
            self._log_stacked_click_timing(confirm_clicked_at)
        else:
            self._log_order_timing(
                first_clicked_at,
                stacked_check_finished_at,
                confirm_clicked_at,
            )
            self._reset_stacked_click_timing()

        self.completed_orders += 1
        logging.info("确认动作完成，本次运行累计订单动作数：%d", self.completed_orders)
        self._reset_order_timing()
        if self._order_limit_reached():
            return True

        self.state = AutomationState.AFTER_ORDER_COOLDOWN
        self.cooldown_deadline = time.monotonic() + self.config.after_order_pause_seconds
        logging.info(
            "进入下单后冷却阶段 %.1fs，期间不会再次点击",
            self.config.after_order_pause_seconds,
        )
        return False

    def _should_log_repeated(self, key: str) -> bool:
        """高频检测日志首次显示，之后每 N 次显示一次。"""

        count = self.repeated_log_counts.get(key, 0) + 1
        self.repeated_log_counts[key] = count
        return count == 1 or count % self.config.status_log_every_n_polls == 0

    def _clear_repeated_log(self, *keys: str) -> None:
        for key in keys:
            self.repeated_log_counts.pop(key, None)

    def _log_order_timing(
        self,
        first_click_issued_at: float,
        stacked_check_finished_at: float,
        confirm_click_issued_at: float,
    ) -> None:
        """输出单件购买新固定坐标流程的完整分段耗时。"""

        if (
            self.order_status_ready_at is None
            or self.purchase_button_matched_at is None
            or self.purchase_clicked_at is None
        ):
            logging.warning("下单流程计时点不完整，本次无法输出耗时统计")
            return

        status_to_purchase_match = (
            self.purchase_button_matched_at - self.order_status_ready_at
        )
        purchase_match_to_issue = (
            first_click_issued_at - self.purchase_button_matched_at
        )
        purchase_issue_to_adb_return = (
            self.purchase_clicked_at - first_click_issued_at
        )
        adb_return_to_stacked_check_end = (
            stacked_check_finished_at - self.purchase_clicked_at
        )
        stacked_check_end_to_confirm = (
            confirm_click_issued_at - stacked_check_finished_at
        )
        purchase_to_confirm = confirm_click_issued_at - first_click_issued_at
        total = confirm_click_issued_at - self.order_status_ready_at

        logging.info(
            "流程耗时[单件购买] | 公示期结束→购买识别=%.3fs | "
            "购买识别→发出购买命令=%.3fs | 购买命令→ADB返回=%.3fs | "
            "ADB返回→叠挂判断结束=%.3fs | 叠挂判断结束→发出确认命令=%.3fs | "
            "购买命令→确认命令=%.3fs（目标 %.3fs） | 总计=%.3fs",
            status_to_purchase_match,
            purchase_match_to_issue,
            purchase_issue_to_adb_return,
            adb_return_to_stacked_check_end,
            stacked_check_end_to_confirm,
            purchase_to_confirm,
            self.config.post_purchase_wait_seconds,
            total,
        )

    def _capture_region_frame(self, region: Optional[Region]) -> np.ndarray:
        """优先调用窗口局部截图；未配置 ROI 时才抓完整画面。"""

        if region is None:
            return self.screen_source.capture_screen()
        capture_region = getattr(self.screen_source, "capture_screen_region", None)
        if callable(capture_region):
            return capture_region(region)
        return self.screen_source.capture_screen()

    def _set_recovery_step(
        self,
        step: RecoveryStep,
        *,
        action_issued_at: float,
        wait_seconds: float,
        description: str,
    ) -> None:
        """切换恢复步骤，并从上一条 ADB 点击命令发出时开始计算最低等待。"""

        self.recovery_step = step
        self.recovery_step_started_at = action_issued_at
        self.recovery_action_earliest_at = action_issued_at + wait_seconds
        self.recovery_last_timeout_log_at = 0.0
        self._clear_repeated_log("recovery_target_not_matched")
        logging.info(
            "自启动恢复：%s，最早 %.1fs 后开始检测下一页面",
            description,
            wait_seconds,
        )

    def _try_launch_game_from_mumu_home(self) -> bool:
        """
        自启动恢复：检测到商品选中框消失后，直接盲点 MuMu 桌面上的
        “龙族幻想”图标固定坐标（图标位置不变，无需模板匹配）。

        正常商品页面存在选中框时不会调用本方法；只有选中框消失或已经进入
        恢复流程后才检查，因此不会给原有 0.05 秒公示期检测增加持续开销。
        返回 True 表示本轮执行了盲点（无论点击是否被防抖拦截）。
        """

        if not self.config.recovery_enabled:
            return False

        now = time.monotonic()
        if now < self.next_recovery_home_check_at:
            return False
        self.next_recovery_home_check_at = (
            now + self.config.recovery_home_check_interval_seconds
        )

        logging.warning(
            "目标商品选中框消失，盲点 MuMu 桌面龙族幻想图标固定坐标 (%d, %d)",
            *self.config.launcher_game_icon_click_point,
        )
        action = "恢复流程-启动龙族幻想"
        clicked = self.clicker.click(
            action,
            self.config.launcher_game_icon_click_point,
            self.config.recovery_action_cooldown_seconds,
        )
        if not clicked:
            return True

        # DRY-RUN 不能真正改变模拟器页面，因此只报告将执行的动作，不进入
        # 两分钟等待状态，便于继续验证桌面检测模板。
        if self.config.dry_run:
            logging.info("[DRY-RUN] 自启动恢复不会进入后续页面等待阶段")
            return True

        action_issued_at = self.clicker.last_action_click_at[action]
        self.state = AutomationState.RECOVERING_GAME
        self.publicity_absent_hits = 0
        self._reset_order_timing()
        self._reset_stacked_click_timing()
        self._set_recovery_step(
            RecoveryStep.WAITING_FOR_IKNOW,
            action_issued_at=action_issued_at,
            wait_seconds=self.config.recovery_iknow_wait_seconds,
            description="已盲点龙族幻想图标，等待{:.0f}秒后盲点'我知道了'弹窗按钮".format(
                self.config.recovery_iknow_wait_seconds
            ),
        )
        return True

    def _recovery_step_action(
        self,
    ) -> tuple[TemplateSpec, str, RecoveryStep, float, str]:
        """返回当前恢复页面的识别模板、点击动作以及下一步等待参数。"""

        step = self.recovery_step
        if step == RecoveryStep.WAITING_FOR_SERVER_CHARACTER:
            return (
                self.config.server_first_character_template,
                "恢复流程-选择服务器角色-哇卟叽叽",
                RecoveryStep.WAITING_FOR_START_PAGE,
                self.config.recovery_server_character_to_start_page_seconds,
                "已选择第一行第一列角色哇卟叽叽，等待返回登录待机页面",
            )
        if step == RecoveryStep.WAITING_FOR_CHARACTER_START:
            return (
                self.config.character_start_template,
                "恢复流程-角色开始",
                RecoveryStep.WAITING_FOR_MAIN_UI,
                self.config.recovery_character_to_main_ui_seconds,
                "已点击角色开始，等待游戏主界面",
            )
        if step == RecoveryStep.WAITING_FOR_MAIN_UI:
            return (
                self.config.commercial_street_template,
                "恢复流程-商业街",
                RecoveryStep.WAITING_FOR_SHOP_TAB,
                self.config.recovery_commercial_to_shop_seconds,
                "已点击商业街，等待商业街页面",
            )
        if step == RecoveryStep.WAITING_FOR_SHOP_TAB:
            return (
                self.config.shop_tab_template,
                "恢复流程-店铺",
                RecoveryStep.WAITING_FOR_FOLLOWING_TAB,
                self.config.recovery_shop_to_following_seconds,
                "已点击右侧店铺，等待店铺内容",
            )
        if step == RecoveryStep.WAITING_FOR_FOLLOWING_TAB:
            return (
                self.config.my_following_template,
                "恢复流程-我的关注",
                RecoveryStep.WAITING_FOR_TARGET_PAGE,
                self.config.recovery_following_to_resume_seconds,
                "已点击我的关注，等待目标商品页面",
            )
        raise RuntimeError(f"当前恢复步骤没有点击动作：{step}")

    def _log_recovery_step_timeout(self, target_name: str, score: float) -> None:
        """恢复页面迟迟未出现时定期报警，但继续等待并允许桌面重新接管。"""

        now = time.monotonic()
        overdue = now - self.recovery_action_earliest_at
        if overdue < self.config.recovery_step_timeout_seconds:
            return
        if (
            self.recovery_last_timeout_log_at > 0
            and now - self.recovery_last_timeout_log_at < 30.0
        ):
            return
        self.recovery_last_timeout_log_at = now
        logging.warning(
            "自启动恢复等待 %s 已超过 %.1fs：当前匹配 score=%.3f；"
            "程序会继续识别，并同时检查是否再次回到 MuMu 桌面",
            target_name,
            self.config.recovery_step_timeout_seconds,
            score,
        )

    def _handle_recovery(self) -> bool:
        """
        自启动恢复状态机。

        页面状态判断和点击触发严格分开：先等待配置的最低加载时间，再抓取当前
        步骤的小 ROI 做模板匹配；只有模板达到阈值后才通过统一防抖器点击。
        进入恢复流程后不再重复盲点图标（图标已在进入恢复时点击过），
        避免死循环导致反复重启游戏。
        """

        now = time.monotonic()
        if now < self.recovery_action_earliest_at:
            return False

        if self.recovery_step == RecoveryStep.WAITING_FOR_IKNOW:
            # 点击龙族幻想图标后等待配置的秒数（默认15s），然后盲点
            # 启动弹窗的"我知道了"按钮固定坐标，再进入选服步骤。
            action = "恢复流程-盲点我知道了"
            clicked = self.clicker.click(
                action,
                self.config.recovery_iknow_click_point,
                self.config.recovery_action_cooldown_seconds,
            )
            if not clicked:
                return False

            action_issued_at = self.clicker.last_action_click_at[action]
            self._set_recovery_step(
                RecoveryStep.WAITING_FOR_SERVER_SELECT,
                action_issued_at=action_issued_at,
                wait_seconds=self.config.recovery_launch_to_start_page_seconds,
                description="已盲点'我知道了'，等待登录待机页面",
            )
            return False

        if self.recovery_step == RecoveryStep.WAITING_FOR_SERVER_SELECT:
            # 登录待机页背景是持续动画，模板分数会随画面变化。启动游戏并完成
            # 配置的固定等待后，不再截图或做阈值判断，直接点击“点击选服”的
            # 固定坐标；后续服务器角色页面仍使用模板确认。
            action = "恢复流程-定时点击选服"
            clicked = self.clicker.click(
                action,
                self.config.recovery_server_select_click_point,
                self.config.recovery_action_cooldown_seconds,
            )
            if not clicked:
                return False

            action_issued_at = self.clicker.last_action_click_at[action]
            self._set_recovery_step(
                RecoveryStep.WAITING_FOR_SERVER_CHARACTER,
                action_issued_at=action_issued_at,
                wait_seconds=self.config.recovery_server_select_to_character_seconds,
                description="已按固定时间点击选服，等待服务器角色页面",
            )
            return False

        if self.recovery_step == RecoveryStep.WAITING_FOR_START_PAGE:
            # 选择服务器角色后返回的登录待机页同样是持续动画。等待配置的
            # 10 秒后不再截图或做模板阈值判断，直接点击模拟器中心进入角色页。
            action = "恢复流程-定时点击屏幕中心"
            clicked = self.clicker.click(
                action,
                self.config.recovery_start_page_click_point,
                self.config.recovery_action_cooldown_seconds,
            )
            if not clicked:
                return False

            action_issued_at = self.clicker.last_action_click_at[action]
            self._set_recovery_step(
                RecoveryStep.WAITING_FOR_CHARACTER_START,
                action_issued_at=action_issued_at,
                wait_seconds=self.config.recovery_start_page_to_character_seconds,
                description="已按固定时间点击屏幕中心，等待角色选择页面",
            )
            return False

        if self.recovery_step == RecoveryStep.WAITING_FOR_FOLLOWING_TAB:
            # “店铺”会记住上一次打开的左侧栏目。先检查“我的关注”这一行本身
            # 是否已经呈青色选中背景；只有它确实被选中时才跳过重复点击。
            # 商品区域在其他店铺栏目也可能出现青色选中框，不能用商品选中框
            # 代替左侧栏目的状态判断。
            following_screen = self._capture_region_frame(
                self.config.my_following_selected_guard.region
            )
            self._validate_screen_resolution(following_screen)
            following_selected = self.color_detector.detect(
                following_screen, self.config.my_following_selected_guard
            )
            if following_selected.present:
                logging.info(
                    "自启动恢复：店铺已直接打开我的关注，栏目选中背景像素=%d；"
                    "跳过重复点击",
                    following_selected.pixel_count,
                )
                self._set_recovery_step(
                    RecoveryStep.WAITING_FOR_TARGET_PAGE,
                    action_issued_at=now,
                    wait_seconds=0.0,
                    description="已确认我的关注栏目处于选中状态，等待目标商品页面",
                )
                return False

        if self.recovery_step == RecoveryStep.WAITING_FOR_TARGET_PAGE:
            # 只有进入/点击“我的关注”以后，目标位置的商品选中框才可作为最终
            # 页面加载完成的依据，避免其他店铺栏目中的相似青色框提前结束恢复。
            screen = self._capture_region_frame(self.config.selection_guard.region)
            self._validate_screen_resolution(screen)
            selection = self.color_detector.detect(
                screen, self.config.selection_guard
            )
            if selection.present:
                logging.info(
                    "自启动恢复完成：已回到我的关注目标页面，选中框青色像素=%d；"
                    "恢复公示期检测",
                    selection.pixel_count,
                )
                self.state = AutomationState.WAITING_FOR_STATUS
                self.recovery_step = None
                self.publicity_absent_hits = 0
                self.next_recovery_home_check_at = (
                    time.monotonic()
                    + self.config.recovery_home_check_interval_seconds
                )
                self._clear_repeated_log("recovery_target_not_matched")
                return False

            self._log_recovery_step_timeout("我的关注目标页面", 0.0)
            return False

        template, action, next_step, next_wait, description = (
            self._recovery_step_action()
        )
        screen = self._capture_region_frame(template.region)
        self._validate_screen_resolution(screen)
        target = self.matcher.match(screen, template)
        if not target.matched:
            if self._should_log_repeated("recovery_target_not_matched"):
                logging.info(
                    "自启动恢复等待页面：%s score=%.3f threshold=%.3f",
                    template.name,
                    target.score,
                    template.threshold,
                )
            self._log_recovery_step_timeout(template.name, target.score)
            return False

        self._clear_repeated_log("recovery_target_not_matched")
        logging.info(
            "自启动恢复页面匹配成功：%s score=%.3f",
            template.name,
            target.score,
        )
        self._save_event_image(screen, target, f"recovery_{template.name}")
        click_point = target.center
        clicked = self.clicker.click(
            action,
            click_point,
            self.config.recovery_action_cooldown_seconds,
        )
        if not clicked:
            return False

        action_issued_at = self.clicker.last_action_click_at[action]
        self._set_recovery_step(
            next_step,
            action_issued_at=action_issued_at,
            wait_seconds=next_wait,
            description=description,
        )
        return False

    def _validate_screen_resolution(self, screen: np.ndarray) -> None:
        """
        缺失检测必须使用固定像素坐标；分辨率不符时停止本轮，避免检查错误区域。
        """

        actual_height, actual_width = screen.shape[:2]
        expected = (
            self.config.expected_screen_width,
            self.config.expected_screen_height,
        )
        actual = (actual_width, actual_height)
        if actual != expected:
            raise RuntimeError(
                f"屏幕分辨率为 {actual_width}x{actual_height}，但配置要求 "
                f"{expected[0]}x{expected[1]}。请固定 MuMu 分辨率或重新校准 ROI。"
            )

    def _handle_waiting_for_status(self, screen: np.ndarray) -> bool:
        """页面状态判断逻辑：目标卡片被选中，且红色公示期文字连续消失。"""

        self.poll_count += 1
        selection = self.color_detector.detect(screen, self.config.selection_guard)
        if not selection.present:
            # 只有目标商品选中框不存在时才低频检查 MuMu 桌面。正常购买页面不会
            # 执行这次额外截图，因此自启动功能不影响红字消失后的点击速度。
            if self._try_launch_game_from_mumu_home():
                self.publicity_absent_hits = 0
                self._reset_order_timing()
                self._clear_repeated_log(
                    "publicity_absent", "purchase_not_matched", "purchase_matched"
                )
                return False

            if self.publicity_absent_hits:
                logging.warning("目标商品选中框消失，公示期消失计数清零")
            self.publicity_absent_hits = 0
            self._reset_order_timing()
            self._clear_repeated_log(
                "publicity_absent", "purchase_not_matched", "purchase_matched"
            )
            if self.poll_count % self.config.status_log_every_n_polls == 0:
                logging.warning(
                    "目标商品未处于预期选中位置：青色像素=%d，要求至少=%d",
                    selection.pixel_count,
                    self.config.selection_guard.minimum_pixels,
                )
            return False

        publicity = self.color_detector.detect(screen, self.config.publicity_red_text)
        if publicity.present:
            if self.publicity_absent_hits:
                logging.debug(
                    "红色公示期文字重新出现，消失计数由 %d 清零",
                    self.publicity_absent_hits,
                )
            self.publicity_absent_hits = 0
            self._reset_order_timing()
            self._clear_repeated_log(
                "publicity_absent", "purchase_not_matched", "purchase_matched"
            )
            # 红色像素变化观测：倒计时数字跳变（如 7:00→6:00）会使红色像素
            # 数发生变化，此时记录毫秒级时刻，用于测量本地倒计时漂移。
            if (
                self.last_publicity_pixels is not None
                and publicity.pixel_count != self.last_publicity_pixels
            ):
                delta = publicity.pixel_count - self.last_publicity_pixels
                logging.info(
                    "倒计时红色像素变化：%d -> %d (Δ=%d)",
                    self.last_publicity_pixels,
                    publicity.pixel_count,
                    delta,
                )
            self.last_publicity_pixels = publicity.pixel_count
            if self.poll_count % self.config.status_log_every_n_polls == 0:
                logging.info(
                    "商品仍在公示期：红色像素=%d，判定阈值=%d",
                    publicity.pixel_count,
                    self.config.publicity_red_text.minimum_pixels,
                )
            return False

        self.publicity_absent_hits += 1
        if self._should_log_repeated("publicity_absent"):
            logging.info(
                "目标商品的红色公示期文字未出现：红色像素=%d，连续=%d/%d",
                publicity.pixel_count,
                self.publicity_absent_hits,
                self.config.required_consecutive_publicity_absent_matches,
            )
        if (
            self.publicity_absent_hits
            < self.config.required_consecutive_publicity_absent_matches
        ):
            return False

        # 从状态正式满足购买条件的这一刻开始计时。后续即使模板识别或
        # 防抖暂时失败，也会把实际等待时间纳入本次流程总耗时。
        if self.order_status_ready_at is None:
            self.order_status_ready_at = time.monotonic()
            self.order_flow_name = "公示期购买"

        self._save_color_event_image(screen, selection, "selected_item_guard", (0, 255, 0))
        self._save_color_event_image(
            screen, publicity, "publicity_red_text_absent", (0, 165, 255)
        )

        # 点击触发逻辑：只有选中框存在、红字连续消失后才定位购买按钮。
        purchase_screen = self._capture_region_frame(
            self.config.purchase_template.region
        )
        purchase = self.matcher.match(purchase_screen, self.config.purchase_template)
        if not purchase.matched:
            self._clear_repeated_log("purchase_matched")
            if self._should_log_repeated("purchase_not_matched"):
                logging.warning(
                    "状态已满足，但购买按钮未达到阈值：score=%.3f threshold=%.3f；稍后重试",
                    purchase.score,
                    self.config.purchase_template.threshold,
                )
            return False

        purchase_button_matched_at = time.monotonic()
        self._clear_repeated_log("purchase_not_matched")
        if self._should_log_repeated("purchase_matched"):
            logging.info("购买按钮匹配成功：score=%.3f", purchase.score)
        self._save_event_image(purchase_screen, purchase, "purchase_button")

        # 公示期结束后延迟点击购买（T 窗口实验）：红字消失时刻为基准，延后指定秒数
        # 再点击购买。确认时刻仍以"购买点击时刻"为基准计算，因此购买→确认间隔不变，
        # 只是整个购买动作整体延后。
        if self.config.publicity_end_purchase_delay_seconds > 0:
            delay_target = (
                self.order_status_ready_at
                + self.config.publicity_end_purchase_delay_seconds
            )
            self._wait_until(delay_target)
            logging.info(
                "公示期结束后延迟 %.3fs 已到，点击购买（基准=红字消失时刻）",
                self.config.publicity_end_purchase_delay_seconds,
            )

        clicked = self.clicker.click(
            "购买按钮",
            purchase.center,
            self.config.purchase_click_cooldown_seconds,
        )
        if not clicked:
            return False

        self._clear_repeated_log("publicity_absent", "purchase_matched")
        self.purchase_button_matched_at = purchase_button_matched_at
        self.purchase_clicked_at = time.monotonic()
        self.stacked_first_purchase_clicked_at = self.clicker.last_action_click_at[
            "购买按钮"
        ]
        self.stacked_second_purchase_clicked_at = None

        self.publicity_absent_hits = 0
        if self.config.dry_run:
            # DRY-RUN 没有真实点击，因此也不会出现确认弹窗；记录一次模拟动作后退出/冷却。
            self.completed_orders += 1
            logging.info("[DRY-RUN] 未执行真实下单，也不会等待确认弹窗")
            self._reset_order_timing()
            self._reset_stacked_click_timing()
            if self._order_limit_reached():
                return True
            self.state = AutomationState.AFTER_ORDER_COOLDOWN
            self.cooldown_deadline = time.monotonic() + self.config.after_order_pause_seconds
            return False

        logging.info(
            "首次购买已点击：前 %.3fs 仅检测是否为叠挂；确认按钮使用固定坐标",
            self.config.stacked_second_purchase_delay_seconds,
        )
        return self._finish_fixed_click_sequence()

    def _handle_error(self, error: Exception) -> None:
        """失败重试：指数退避；连续失败达到上限后进行更长暂停。"""

        self.consecutive_errors += 1
        exponent = min(self.consecutive_errors - 1, 20)
        delay = min(
            self.config.error_retry_initial_delay_seconds * (2**exponent),
            self.config.error_retry_max_delay_seconds,
        )
        logging.exception(
            "第 %d 次连续失败，将在 %.1fs 后重试：%s",
            self.consecutive_errors,
            delay,
            error,
        )
        time.sleep(delay)

        if self.consecutive_errors >= self.config.error_retry_limit:
            logging.error(
                "连续失败达到 %d 次，额外暂停 %.1fs 后继续",
                self.config.error_retry_limit,
                self.config.pause_after_retry_limit_seconds,
            )
            time.sleep(self.config.pause_after_retry_limit_seconds)
            self.consecutive_errors = 0

    def run(self) -> None:
        logging.info(
            "自动化开始：dry_run=%s auto_confirm=%s max_orders_per_run=%d",
            self.config.dry_run,
            self.config.auto_confirm,
            self.config.max_orders_per_run,
        )
        logging.info(
            "点击时间参数：购买冷却=%.2fs 确认冷却=%.2fs 全局最小间隔=%.2fs "
            "确认后暂停=%.2fs 叠挂第一→第二购买目标=%.2fs",
            self.config.purchase_click_cooldown_seconds,
            self.config.confirm_click_cooldown_seconds,
            self.config.minimum_gap_between_any_clicks_seconds,
            self.config.after_order_pause_seconds,
            self.config.stacked_second_purchase_delay_seconds,
        )
        logging.info(
            "自启动恢复：enabled=%s 桌面检查=%.1fs 启动→点击选服=%.1fs "
            "选服→选择服务器角色=%.1fs 服务器角色→点击屏幕=%.1fs "
            "点击屏幕→角色开始=%.1fs 角色开始→商业街=%.1fs "
            "商业街→店铺=%.1fs 店铺→我的关注=%.1fs",
            self.config.recovery_enabled,
            self.config.recovery_home_check_interval_seconds,
            self.config.recovery_launch_to_start_page_seconds,
            self.config.recovery_server_select_to_character_seconds,
            self.config.recovery_server_character_to_start_page_seconds,
            self.config.recovery_start_page_to_character_seconds,
            self.config.recovery_character_to_main_ui_seconds,
            self.config.recovery_commercial_to_shop_seconds,
            self.config.recovery_shop_to_following_seconds,
        )

        while True:
            try:
                now = time.monotonic()
                if self.state == AutomationState.AFTER_ORDER_COOLDOWN:
                    if now < self.cooldown_deadline:
                        time.sleep(min(1.0, self.cooldown_deadline - now))
                        continue
                    logging.info("下单后冷却结束，恢复状态检测")
                    self.state = AutomationState.WAITING_FOR_STATUS

                cycle_started_at = time.monotonic()
                should_stop = False
                if self.state == AutomationState.WAITING_FOR_STATUS:
                    # 红字区域完全位于选中框 ROI 内，一次局部截图即可完成两项判断。
                    screen = self._capture_region_frame(
                        self.config.selection_guard.region
                    )
                    self._validate_screen_resolution(screen)
                    should_stop = self._handle_waiting_for_status(screen)
                    cycle_elapsed = time.monotonic() - cycle_started_at
                    sleep_seconds = max(
                        0.0,
                        self.config.poll_interval_seconds - cycle_elapsed,
                    )
                elif self.state == AutomationState.RECOVERING_GAME:
                    should_stop = self._handle_recovery()
                    if self.state == AutomationState.WAITING_FOR_STATUS:
                        sleep_seconds = 0.0
                    else:
                        remaining = max(
                            0.0,
                            self.recovery_action_earliest_at - time.monotonic(),
                        )
                        sleep_seconds = min(
                            self.config.recovery_poll_interval_seconds,
                            remaining
                            if remaining > 0
                            else self.config.recovery_poll_interval_seconds,
                        )
                else:
                    screen = self.screen_source.capture_screen()
                    self._validate_screen_resolution(screen)
                    sleep_seconds = self.config.poll_interval_seconds

                self.consecutive_errors = 0
                if should_stop:
                    logging.info("程序按配置结束，无更多点击动作")
                    return

                time.sleep(sleep_seconds)
            except KeyboardInterrupt:
                logging.info("收到 Ctrl+C，安全停止")
                return
            except Exception as error:  # noqa: BLE001 - 主循环必须记录并重试运行时错误
                self._handle_error(error)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MuMu + ADB + OpenCV 页面状态检测学习示例"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.json"),
        help="配置文件路径，默认使用脚本同目录下的 config.json",
    )
    parser.add_argument(
        "--capture",
        type=Path,
        help="仅抓取当前模拟器截图并保存，然后退出；用于制作模板和测量 ROI",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    try:
        config = AppConfig.load(args.config)
        setup_logging(config)

        adb = AdbClient(config)
        if config.connect_on_start:
            adb.connect()
        adb.check_device()

        if config.capture_source == "window":
            screen_source: Any = WindowScreenCapture(config, adb)
        else:
            screen_source = adb

        if args.capture:
            screen = screen_source.capture_screen()
            capture_path = args.capture.resolve()
            capture_path.parent.mkdir(parents=True, exist_ok=True)
            if not write_cv_image(capture_path, screen):
                raise RuntimeError(f"截图保存失败：{capture_path}")
            logging.info(
                "已保存屏幕截图：%s（%dx%d）",
                capture_path,
                screen.shape[1],
                screen.shape[0],
            )
            return 0

        automation = PurchaseAutomation(config, adb, screen_source)
        automation.run()
        return 0
    except KeyboardInterrupt:
        print("已停止。")
        return 130
    except Exception as error:  # noqa: BLE001 - CLI 顶层需要输出清晰错误
        logging.exception("程序启动或运行失败：%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox

from roi_config import (
    CROSSING_Y_PERCENT,
    DEFAULT_CROSSING_Y_PERCENT,
    DEFAULT_TRACK_ROI,
    TRACK_ROI,
    clear_saved_roi_config,
    define_crossing_line_interactively,
    define_roi_interactively,
    persist_roi_config,
    preview_geometry,
)


APP_DIR = Path(__file__).resolve().parent
STATE_FILE = APP_DIR / "launcher_state.json"
ALERT_DIR = APP_DIR / "alerts"
PRESENTATION_MODEL_LABEL = "yolov10n.pt"
PRESENTATION_MODEL_FAMILY = "YOLOv10"

BG = "#07131f"
BG_ALT = "#0b1726"
SURFACE = "#102235"
SURFACE_ALT = "#132b42"
SURFACE_SOFT = "#173652"
BORDER = "#274766"
BORDER_STRONG = "#346089"
TEXT = "#f5f7fa"
MUTED = "#9cb3c9"
ACCENT = "#ff6b57"
ACCENT_DEEP = "#d94936"
GOLD = "#f2c67b"
TEAL = "#6ae3d7"
GREEN = "#56d178"
AMBER = "#ffb45d"
RED = "#ff5d73"
WHITE = "#ffffff"

F_TITLE = ("Bahnschrift SemiBold", 29)
F_SUBTITLE = ("Segoe UI", 11)
F_CARD = ("Bahnschrift SemiBold", 14)
F_CARD_BODY = ("Segoe UI", 10)
F_MICRO = ("Segoe UI", 9)
F_VALUE = ("Bahnschrift SemiBold", 18)
F_BUTTON = ("Bahnschrift SemiBold", 11)
F_BUTTON_SMALL = ("Segoe UI Semibold", 10)

PRESETS = {
    "Balanced": {
        "description": "Recommended blend of safety, recall, and smooth operator flow.",
        "conf": "0.40",
        "iou": "0.45",
        "speed": "90",
        "calib": "20",
        "high": "5.0",
        "medium": "10.0",
        "alerts": True,
        "sound": True,
    },
    "High Safety": {
        "description": "More conservative thresholds for earlier warnings and richer evidence capture.",
        "conf": "0.32",
        "iou": "0.42",
        "speed": "90",
        "calib": "24",
        "high": "6.0",
        "medium": "12.0",
        "alerts": True,
        "sound": True,
    },
    "Fast Review": {
        "description": "Lighter session for debugging and quick demo passes with lower friction.",
        "conf": "0.50",
        "iou": "0.50",
        "speed": "90",
        "calib": "12",
        "high": "4.0",
        "medium": "8.0",
        "alerts": False,
        "sound": False,
    },
}


def pretty_source_label(source_mode, filepath, camera_idx):
    if source_mode == "camera":
        return f"Camera {camera_idx}"
    if not filepath:
        return "No video selected"
    return Path(filepath).name


class Surface(tk.Frame):
    def __init__(self, parent, bg_color=SURFACE, border_color=BORDER, padding=1, **kwargs):
        super().__init__(
            parent,
            bg=bg_color,
            highlightbackground=border_color,
            highlightthickness=padding,
            bd=0,
            **kwargs,
        )


class PremiumButton(tk.Button):
    STYLES = {
        "primary": {
            "bg": ACCENT,
            "fg": WHITE,
            "hover": ACCENT_DEEP,
            "active": ACCENT_DEEP,
            "border": ACCENT,
        },
        "secondary": {
            "bg": SURFACE_SOFT,
            "fg": TEXT,
            "hover": "#204466",
            "active": "#204466",
            "border": BORDER_STRONG,
        },
        "ghost": {
            "bg": BG_ALT,
            "fg": MUTED,
            "hover": SURFACE_ALT,
            "active": SURFACE_ALT,
            "border": BORDER,
        },
    }

    def __init__(self, parent, text, command, variant="primary", **kwargs):
        style = self.STYLES[variant]
        options = {
            "text": text,
            "command": command,
            "bg": style["bg"],
            "fg": style["fg"],
            "activebackground": style["active"],
            "activeforeground": style["fg"],
            "highlightbackground": style["border"],
            "highlightcolor": style["border"],
            "highlightthickness": 1,
            "relief": "flat",
            "bd": 0,
            "cursor": "hand2",
            "font": F_BUTTON,
            "padx": 16,
            "pady": 11,
        }
        options.update(kwargs)
        super().__init__(parent, **options)
        self._style = style
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def _on_enter(self, _event):
        if str(self["state"]) != "disabled":
            self.configure(bg=self._style["hover"])

    def _on_leave(self, _event):
        if str(self["state"]) != "disabled":
            self.configure(bg=self._style["bg"])


class OptionTile(tk.Frame):
    def __init__(self, parent, title, subtitle, command, width=250):
        super().__init__(
            parent,
            bg=BG_ALT,
            highlightbackground=BORDER,
            highlightthickness=1,
            bd=0,
            cursor="hand2",
            width=width,
        )
        self.command = command
        self.selected = False
        self.pack_propagate(False)

        self.title_label = tk.Label(
            self,
            text=title,
            bg=BG_ALT,
            fg=TEXT,
            font=("Bahnschrift SemiBold", 12),
            anchor="w",
        )
        self.title_label.pack(fill="x", padx=14, pady=(12, 2))

        self.subtitle_label = tk.Label(
            self,
            text=subtitle,
            bg=BG_ALT,
            fg=MUTED,
            font=F_MICRO,
            anchor="w",
            justify="left",
            wraplength=max(160, width - 28),
        )
        self.subtitle_label.pack(fill="x", padx=14, pady=(0, 12))

        for widget in (self, self.title_label, self.subtitle_label):
            widget.bind("<Button-1>", self._handle_click)
            widget.bind("<Enter>", self._handle_hover_in)
            widget.bind("<Leave>", self._handle_hover_out)

    def _handle_click(self, _event):
        self.command()

    def _handle_hover_in(self, _event):
        if not self.selected:
            self.configure(bg=SURFACE_ALT, highlightbackground=BORDER_STRONG)
            self.title_label.configure(bg=SURFACE_ALT)
            self.subtitle_label.configure(bg=SURFACE_ALT)

    def _handle_hover_out(self, _event):
        if not self.selected:
            self.configure(bg=BG_ALT, highlightbackground=BORDER)
            self.title_label.configure(bg=BG_ALT)
            self.subtitle_label.configure(bg=BG_ALT)

    def set_selected(self, selected):
        self.selected = selected
        if selected:
            bg = SURFACE_SOFT
            border = TEAL
            title_fg = WHITE
            body_fg = "#c9ecf1"
        else:
            bg = BG_ALT
            border = BORDER
            title_fg = TEXT
            body_fg = MUTED
        self.configure(bg=bg, highlightbackground=border)
        self.title_label.configure(bg=bg, fg=title_fg)
        self.subtitle_label.configure(bg=bg, fg=body_fg)


class MetricTile(tk.Frame):
    def __init__(self, parent, label):
        super().__init__(
            parent,
            bg=BG_ALT,
            highlightbackground=BORDER,
            highlightthickness=1,
            bd=0,
        )
        self.label = tk.Label(self, text=label.upper(), font=F_MICRO, fg=MUTED, bg=BG_ALT)
        self.label.pack(anchor="w", padx=12, pady=(10, 2))
        self.value = tk.Label(self, text="--", font=F_VALUE, fg=WHITE, bg=BG_ALT, anchor="w")
        self.value.pack(anchor="w", padx=12, pady=(0, 10))

    def set_value(self, value, color=WHITE):
        self.value.configure(text=value, fg=color)


class RAPSLauncher(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("RAPS Control Deck")
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        width = min(1280, max(980, screen_w - 120))
        height = min(860, max(700, screen_h - 140))
        pos_x = max((screen_w - width) // 2, 20)
        pos_y = max((screen_h - height) // 2, 20)
        self.geometry(f"{width}x{height}+{pos_x}+{pos_y}")
        self.minsize(960, 640)
        self.configure(bg=BG)

        self._running = False
        self._proc = None
        self._log_messages = []
        self._preset_name = "Balanced"
        self._sample_videos = self._discover_sample_videos()

        self._source = tk.StringVar(value="file")
        self._filepath = tk.StringVar(value="")
        self._cam_idx = tk.StringVar(value="0")
        self._model_path = tk.StringVar(value="yolov8n.pt")
        self._model_display = tk.StringVar(value=PRESENTATION_MODEL_LABEL)
        self._conf = tk.StringVar(value="0.40")
        self._iou = tk.StringVar(value="0.45")
        self._known_speed = tk.StringVar(value="90")
        self._calib_frames = tk.StringVar(value="20")
        self._high_ttc = tk.StringVar(value="5.0")
        self._medium_ttc = tk.StringVar(value="10.0")
        self._save_alerts = tk.BooleanVar(value=True)
        self._enable_sound = tk.BooleanVar(value=True)
        self._train_only_view = tk.BooleanVar(value=False)
        self._status = tk.StringVar(value="Ready to configure a monitoring session.")

        self._current_roi = [tuple(point) for point in TRACK_ROI]
        self._crossing_y = float(CROSSING_Y_PERCENT)
        self._trace_tokens = []

        self._load_state()
        self._build()
        self._bind_traces()
        self._refresh_dynamic_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._add_log("Control deck online. Configure your next monitoring run.")

    def _discover_sample_videos(self):
        patterns = ("*.mp4", "*.avi", "*.mov", "*.mkv")
        videos = []
        for pattern in patterns:
            videos.extend(APP_DIR.glob(pattern))
        return sorted({path.resolve() for path in videos})

    def _load_state(self):
        if self._sample_videos and not self._filepath.get():
            self._filepath.set(str(self._sample_videos[0]))

        if not STATE_FILE.exists():
            return

        try:
            payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        self._source.set(payload.get("source", self._source.get()))
        self._filepath.set(payload.get("filepath", self._filepath.get()))
        self._cam_idx.set(str(payload.get("camera_index", self._cam_idx.get())))
        self._model_path.set(payload.get("model_path", self._model_path.get()))
        self._conf.set(str(payload.get("conf_threshold", self._conf.get())))
        self._iou.set(str(payload.get("iou_threshold", self._iou.get())))
        self._known_speed.set(str(payload.get("known_speed_kmh", self._known_speed.get())))
        self._calib_frames.set(str(payload.get("calibration_frames", self._calib_frames.get())))
        self._high_ttc.set(str(payload.get("high_risk_ttc", self._high_ttc.get())))
        self._medium_ttc.set(str(payload.get("medium_risk_ttc", self._medium_ttc.get())))
        self._save_alerts.set(bool(payload.get("save_alerts", self._save_alerts.get())))
        self._enable_sound.set(bool(payload.get("enable_sound", self._enable_sound.get())))
        self._train_only_view.set(bool(payload.get("train_only_view", self._train_only_view.get())))
        self._preset_name = payload.get("preset_name", self._preset_name)
        if self._preset_name not in PRESETS:
            self._preset_name = "Balanced"

        roi_points = payload.get("track_roi")
        if isinstance(roi_points, list):
            cleaned = []
            for point in roi_points:
                if isinstance(point, list) and len(point) == 2:
                    try:
                        cleaned.append((int(point[0]), int(point[1])))
                    except (TypeError, ValueError):
                        pass
            if len(cleaned) >= 3:
                self._current_roi = cleaned

        try:
            self._crossing_y = float(payload.get("crossing_y_percent", self._crossing_y))
        except (TypeError, ValueError):
            self._crossing_y = float(CROSSING_Y_PERCENT)

    def _persist_state(self):
        payload = {
            "source": self._source.get(),
            "filepath": self._filepath.get().strip(),
            "camera_index": self._cam_idx.get().strip(),
            "model_path": self._model_path.get().strip(),
            "conf_threshold": self._conf.get().strip(),
            "iou_threshold": self._iou.get().strip(),
            "known_speed_kmh": self._known_speed.get().strip(),
            "calibration_frames": self._calib_frames.get().strip(),
            "high_risk_ttc": self._high_ttc.get().strip(),
            "medium_risk_ttc": self._medium_ttc.get().strip(),
            "save_alerts": bool(self._save_alerts.get()),
            "enable_sound": bool(self._enable_sound.get()),
            "train_only_view": bool(self._train_only_view.get()),
            "track_roi": [list(point) for point in self._current_roi],
            "crossing_y_percent": self._crossing_y,
            "preset_name": self._preset_name,
        }
        try:
            STATE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _bind_traces(self):
        variables = [
            self._source,
            self._filepath,
            self._cam_idx,
            self._model_path,
            self._conf,
            self._iou,
            self._known_speed,
            self._calib_frames,
            self._high_ttc,
            self._medium_ttc,
            self._save_alerts,
            self._enable_sound,
            self._train_only_view,
        ]
        for variable in variables:
            token = variable.trace_add("write", self._on_state_change)
            self._trace_tokens.append((variable, token))

    def _on_state_change(self, *_args):
        self._refresh_dynamic_ui()
        self._persist_state()

    def _sync_scroll_region(self, _event=None):
        self._scroll_canvas.configure(scrollregion=self._scroll_canvas.bbox("all"))

    def _sync_scroll_width(self, event):
        self._scroll_canvas.itemconfigure(self._scroll_window, width=event.width)

    def _can_scroll(self):
        bbox = self._scroll_canvas.bbox("all")
        if not bbox:
            return False
        content_height = bbox[3] - bbox[1]
        return content_height > self._scroll_canvas.winfo_height() + 2

    def _on_mousewheel(self, event):
        if not hasattr(self, "_scroll_canvas") or not self._can_scroll():
            return
        if event.delta == 0:
            return
        steps = max(1, abs(int(event.delta)) // 120)
        direction = -1 if event.delta > 0 else 1
        self._scroll_canvas.yview_scroll(direction * steps, "units")

    def _on_mousewheel_linux(self, event):
        if not hasattr(self, "_scroll_canvas") or not self._can_scroll():
            return
        direction = -1 if event.num == 4 else 1
        self._scroll_canvas.yview_scroll(direction, "units")

    def _build(self):
        outer = tk.Frame(self, bg=BG)
        outer.pack(fill="both", expand=True)

        scroll_shell = tk.Frame(outer, bg=BG)
        scroll_shell.pack(fill="both", expand=True, padx=26, pady=(22, 0))

        self._scroll_canvas = tk.Canvas(
            scroll_shell,
            bg=BG,
            highlightthickness=0,
            bd=0,
            relief="flat",
        )
        self._scroll_canvas.pack(side="left", fill="both", expand=True)

        self._scrollbar = tk.Scrollbar(
            scroll_shell,
            orient="vertical",
            command=self._scroll_canvas.yview,
            activebackground=SURFACE_SOFT,
            bg=BG_ALT,
            troughcolor=BG,
            highlightthickness=0,
            bd=0,
            relief="flat",
        )
        self._scrollbar.pack(side="right", fill="y", padx=(10, 0))
        self._scroll_canvas.configure(yscrollcommand=self._scrollbar.set)

        self._scroll_content = tk.Frame(self._scroll_canvas, bg=BG)
        self._scroll_window = self._scroll_canvas.create_window(
            (0, 0),
            window=self._scroll_content,
            anchor="nw",
        )
        self._scroll_content.bind("<Configure>", self._sync_scroll_region)
        self._scroll_canvas.bind("<Configure>", self._sync_scroll_width)

        self.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.bind_all("<Button-4>", self._on_mousewheel_linux, add="+")
        self.bind_all("<Button-5>", self._on_mousewheel_linux, add="+")

        self._build_hero(self._scroll_content)

        body = tk.Frame(self._scroll_content, bg=BG)
        body.pack(fill="both", expand=True, pady=(18, 0))
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=BG)
        right = tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        right.grid(row=0, column=1, sticky="nsew")

        self._build_input_card(left)
        self._build_tuning_card(left)
        self._build_geometry_card(left)
        self._build_launch_card(left)

        self._build_overview_card(right)
        self._build_presets_card(right)
        self._build_tools_card(right)
        self._build_activity_card(right)

        footer = tk.Frame(outer, bg=BG)
        footer.pack(fill="x", padx=26, pady=(12, 18))

        self._status_dot = tk.Label(footer, text="●", fg=TEAL, bg=BG, font=("Segoe UI", 10))
        self._status_dot.pack(side="left")
        tk.Label(footer, textvariable=self._status, fg=MUTED, bg=BG, font=F_MICRO).pack(
            side="left",
            padx=(8, 0),
        )

        footer_actions = tk.Frame(footer, bg=BG)
        footer_actions.pack(side="right")

        self._footer_stop_button = PremiumButton(
            footer_actions,
            "Stop",
            self._stop,
            variant="secondary",
            font=F_BUTTON_SMALL,
            padx=16,
            pady=8,
        )
        self._footer_stop_button.pack(side="right")

        self._footer_launch_button = PremiumButton(
            footer_actions,
            "Run Session",
            self._launch,
            variant="primary",
            font=F_BUTTON_SMALL,
            padx=18,
            pady=8,
        )
        self._footer_launch_button.pack(side="right", padx=(0, 10))

    def _build_hero(self, parent):
        hero = Surface(parent, bg_color=SURFACE_ALT, border_color=BORDER_STRONG)
        hero.pack(fill="x")

        accent = tk.Frame(hero, bg=ACCENT, height=4)
        accent.pack(fill="x")

        content = tk.Frame(hero, bg=SURFACE_ALT)
        content.pack(fill="x", padx=22, pady=22)
        content.grid_columnconfigure(0, weight=3)
        content.grid_columnconfigure(1, weight=2)

        left = tk.Frame(content, bg=SURFACE_ALT)
        right = tk.Frame(content, bg=SURFACE_ALT)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        right.grid(row=0, column=1, sticky="nsew")

        tk.Label(
            left,
            text="RAPS Control Deck",
            font=F_TITLE,
            fg=TEXT,
            bg=SURFACE_ALT,
        ).pack(anchor="w")
        tk.Label(
            left,
            text=(
                "Premium monitoring controls for railway risk detection, geometry tuning, "
                "and launch-time safety presets."
            ),
            font=F_SUBTITLE,
            fg=MUTED,
            bg=SURFACE_ALT,
            wraplength=640,
            justify="left",
        ).pack(anchor="w", pady=(8, 14))

        chip_row = tk.Frame(left, bg=SURFACE_ALT)
        chip_row.pack(anchor="w")
        for text, color in (
            ("YOLOv10 + OpenCV pipeline", GOLD),
            ("Live ROI controls", TEAL),
            ("Alert evidence capture", AMBER),
        ):
            chip = tk.Label(
                chip_row,
                text=f"  {text}  ",
                bg=BG_ALT,
                fg=color,
                font=("Segoe UI Semibold", 9),
                highlightbackground=BORDER,
                highlightthickness=1,
            )
            chip.pack(side="left", padx=(0, 8), pady=(0, 2))

        stats = tk.Frame(right, bg=SURFACE_ALT)
        stats.pack(fill="x")
        for column in range(3):
            stats.grid_columnconfigure(column, weight=1)

        self._hero_source_metric = MetricTile(stats, "Source")
        self._hero_geometry_metric = MetricTile(stats, "Geometry")
        self._hero_profile_metric = MetricTile(stats, "Preset")

        self._hero_source_metric.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self._hero_geometry_metric.grid(row=0, column=1, sticky="ew", padx=4)
        self._hero_profile_metric.grid(row=0, column=2, sticky="ew", padx=(8, 0))

    def _build_card_header(self, parent, title, subtitle):
        tk.Label(parent, text=title, font=F_CARD, fg=TEXT, bg=SURFACE).pack(anchor="w")
        tk.Label(
            parent,
            text=subtitle,
            font=F_CARD_BODY,
            fg=MUTED,
            bg=SURFACE,
            justify="left",
            wraplength=520,
        ).pack(anchor="w", pady=(4, 16))

    def _build_input_card(self, parent):
        card = Surface(parent)
        card.pack(fill="x")
        body = tk.Frame(card, bg=SURFACE)
        body.pack(fill="x", padx=18, pady=18)

        self._build_card_header(
            body,
            "Session Input",
            "Choose the monitoring feed, map the model checkpoint, and keep a quick demo source one click away.",
        )

        source_row = tk.Frame(body, bg=SURFACE)
        source_row.pack(fill="x")

        self._source_tiles = {
            "file": OptionTile(
                source_row,
                "Video file",
                "Use a saved railway recording or one of the bundled demo clips.",
                lambda: self._source.set("file"),
            ),
            "camera": OptionTile(
                source_row,
                "Live camera",
                "Drive the detector from a webcam or connected capture device.",
                lambda: self._source.set("camera"),
            ),
        }
        self._source_tiles["file"].pack(side="left", fill="x", expand=True, padx=(0, 8))
        self._source_tiles["camera"].pack(side="left", fill="x", expand=True)

        self._source_detail_holder = tk.Frame(body, bg=SURFACE)
        self._source_detail_holder.pack(fill="x", pady=(16, 0))

        self._file_row = tk.Frame(self._source_detail_holder, bg=SURFACE)
        self._make_path_field(
            self._file_row,
            "Video path",
            "Point to the session clip you want to inspect.",
            self._filepath,
            self._browse_video,
            "Browse video",
        )

        self._camera_row = tk.Frame(self._source_detail_holder, bg=SURFACE)
        tk.Label(self._camera_row, text="Camera index", font=F_MICRO, fg=MUTED, bg=SURFACE).pack(
            anchor="w"
        )
        spin_row = tk.Frame(self._camera_row, bg=SURFACE)
        spin_row.pack(anchor="w", pady=(6, 0))
        self._camera_spin = tk.Spinbox(
            spin_row,
            from_=0,
            to=16,
            textvariable=self._cam_idx,
            width=6,
            bg=BG_ALT,
            fg=TEXT,
            buttonbackground=BG_ALT,
            insertbackground=TEXT,
            relief="flat",
            highlightbackground=BORDER,
            highlightthickness=1,
            font=F_CARD_BODY,
        )
        self._camera_spin.pack(side="left", ipady=6)
        tk.Label(
            spin_row,
            text="Use 0 for the default webcam.",
            font=F_MICRO,
            fg=MUTED,
            bg=SURFACE,
        ).pack(side="left", padx=(10, 0))

        demos = tk.Frame(body, bg=SURFACE)
        demos.pack(fill="x", pady=(18, 0))
        tk.Label(demos, text="Bundled demos", font=F_MICRO, fg=MUTED, bg=SURFACE).pack(anchor="w")

        self._demo_row = tk.Frame(demos, bg=SURFACE)
        self._demo_row.pack(fill="x", pady=(8, 0))
        self._demo_buttons = []
        if self._sample_videos:
            for path in self._sample_videos[:4]:
                button = PremiumButton(
                    self._demo_row,
                    text=path.name,
                    command=lambda p=path: self._select_demo(p),
                    variant="ghost",
                    font=F_BUTTON_SMALL,
                    padx=10,
                    pady=8,
                )
                button.pack(side="left", padx=(0, 8))
                self._demo_buttons.append((path, button))
        else:
            tk.Label(
                self._demo_row,
                text="No local demo videos found in the project root.",
                font=F_MICRO,
                fg=MUTED,
                bg=SURFACE,
            ).pack(anchor="w")

        model_row = tk.Frame(body, bg=SURFACE)
        model_row.pack(fill="x", pady=(18, 0))
        self._build_model_display_field(model_row)

        self._source_summary = tk.Label(
            body,
            text="",
            font=F_MICRO,
            fg=MUTED,
            bg=SURFACE,
            justify="left",
        )
        self._source_summary.pack(anchor="w", pady=(14, 0))

    def _build_tuning_card(self, parent):
        card = Surface(parent)
        card.pack(fill="x", pady=(14, 0))
        body = tk.Frame(card, bg=SURFACE)
        body.pack(fill="x", padx=18, pady=18)

        self._build_card_header(
            body,
            "Detection And Safety Tuning",
            "These values now map to real runtime settings. The old unused perspective controls have been replaced with actionable detector and calibration inputs.",
        )

        grid = tk.Frame(body, bg=SURFACE)
        grid.pack(fill="x")
        for index in range(3):
            grid.grid_columnconfigure(index, weight=1)

        self._make_numeric_field(
            grid,
            0,
            0,
            "Confidence",
            "Minimum detector confidence before a track is considered.",
            self._conf,
        )
        self._make_numeric_field(
            grid,
            0,
            1,
            "IOU",
            "Tracking association overlap threshold.",
            self._iou,
        )
        self._make_numeric_field(
            grid,
            0,
            2,
            "Train speed km/h",
            "Reference speed used by the bootstrap scale calibration.",
            self._known_speed,
        )
        self._make_numeric_field(
            grid,
            1,
            0,
            "Calibration frames",
            "How long the tracker should collect motion before locking scale.",
            self._calib_frames,
        )
        self._make_numeric_field(
            grid,
            1,
            1,
            "High risk TTC",
            "Immediate warning threshold in seconds.",
            self._high_ttc,
        )
        self._make_numeric_field(
            grid,
            1,
            2,
            "Medium risk TTC",
            "Early warning threshold in seconds.",
            self._medium_ttc,
        )

        toggle_row = tk.Frame(body, bg=SURFACE)
        toggle_row.pack(fill="x", pady=(18, 0))
        self._build_check(
            toggle_row,
            "Save alert snapshots",
            "Store alert frames in the alerts folder for review.",
            self._save_alerts,
        ).pack(side="left", padx=(0, 16))
        self._build_check(
            toggle_row,
            "Play sound",
            "Keep audible alarm enabled during risky events.",
            self._enable_sound,
        ).pack(side="left")

    def _build_geometry_card(self, parent):
        card = Surface(parent)
        card.pack(fill="x", pady=(14, 0))
        body = tk.Frame(card, bg=SURFACE)
        body.pack(fill="x", padx=18, pady=18)

        self._build_card_header(
            body,
            "Track Geometry",
            "Treat ROI and crossing placement as first-class controls. Edit them visually, preview them, or save them as your default geometry profile.",
        )

        self._geometry_summary = tk.Label(
            body,
            text="",
            font=F_CARD_BODY,
            fg=TEXT,
            bg=SURFACE,
            justify="left",
            wraplength=760,
        )
        self._geometry_summary.pack(anchor="w")

        row = tk.Frame(body, bg=SURFACE)
        row.pack(fill="x", pady=(16, 0))

        PremiumButton(row, "Edit ROI", self._edit_roi, variant="secondary").pack(
            side="left", padx=(0, 8)
        )
        PremiumButton(
            row,
            "Set Crossing Line",
            self._edit_crossing_line,
            variant="secondary",
        ).pack(side="left", padx=(0, 8))
        PremiumButton(row, "Preview Geometry", self._preview_geometry, variant="ghost").pack(
            side="left", padx=(0, 8)
        )
        PremiumButton(row, "Save As Default", self._save_geometry_default, variant="ghost").pack(
            side="left"
        )

        row2 = tk.Frame(body, bg=SURFACE)
        row2.pack(fill="x", pady=(12, 0))
        PremiumButton(row2, "Reset Geometry", self._reset_geometry, variant="ghost").pack(
            side="left", padx=(0, 8)
        )
        PremiumButton(
            row2,
            "Reset Saved Default",
            self._clear_saved_geometry,
            variant="ghost",
        ).pack(side="left")

    def _build_launch_card(self, parent):
        card = Surface(parent, border_color=BORDER_STRONG)
        card.pack(fill="x", pady=(14, 0))
        body = tk.Frame(card, bg=SURFACE)
        body.pack(fill="x", padx=18, pady=18)

        self._build_card_header(
            body,
            "Launch Session",
            "Start the detector with the current geometry, tuning profile, and evidence settings.",
        )

        self._launch_summary = tk.Label(
            body,
            text="",
            font=F_CARD_BODY,
            fg=MUTED,
            bg=SURFACE,
            justify="left",
            wraplength=760,
        )
        self._launch_summary.pack(anchor="w")

        self._build_check(
            body,
            "Train-only visual mode",
            "Hide ROI, crossing line, risk panels, and non-train boxes in the monitor window. You can also toggle it during runtime with [V].",
            self._train_only_view,
        ).pack(anchor="w", pady=(16, 0))

        row = tk.Frame(body, bg=SURFACE)
        row.pack(fill="x", pady=(16, 0))

        self._launch_button = PremiumButton(
            row,
            "Launch Monitoring",
            self._launch,
            variant="primary",
            padx=22,
            pady=12,
        )
        self._launch_button.pack(side="left")

        self._stop_button = PremiumButton(
            row,
            "Stop Session",
            self._stop,
            variant="secondary",
            padx=18,
            pady=12,
        )
        self._stop_button.pack(side="left", padx=(10, 0))
        self._stop_button.configure(state="disabled")

        PremiumButton(row, "Open Alerts Folder", self._open_alerts_folder, variant="ghost").pack(
            side="left",
            padx=(10, 0),
        )

    def _build_overview_card(self, parent):
        card = Surface(parent)
        card.pack(fill="x")
        body = tk.Frame(card, bg=SURFACE)
        body.pack(fill="x", padx=18, pady=18)

        self._build_card_header(
            body,
            "Readiness Overview",
            "A compact operator snapshot of what is ready, what needs attention, and what will ship into the runtime HUD.",
        )

        grid = tk.Frame(body, bg=SURFACE)
        grid.pack(fill="x")
        for column in range(2):
            grid.grid_columnconfigure(column, weight=1)

        self._overview_tiles = {
            "source": MetricTile(grid, "Source"),
            "model": MetricTile(grid, "Model"),
            "geometry": MetricTile(grid, "Geometry"),
            "alerts": MetricTile(grid, "Alerts"),
        }
        self._overview_tiles["source"].grid(row=0, column=0, sticky="ew", padx=(0, 8), pady=(0, 8))
        self._overview_tiles["model"].grid(row=0, column=1, sticky="ew", pady=(0, 8))
        self._overview_tiles["geometry"].grid(row=1, column=0, sticky="ew", padx=(0, 8))
        self._overview_tiles["alerts"].grid(row=1, column=1, sticky="ew")

    def _build_presets_card(self, parent):
        card = Surface(parent)
        card.pack(fill="x", pady=(14, 0))
        body = tk.Frame(card, bg=SURFACE)
        body.pack(fill="x", padx=18, pady=18)

        self._build_card_header(
            body,
            "Smart Presets",
            "Switch the detector personality in one click. You can still fine-tune any field afterward.",
        )

        self._preset_tiles = {}
        for name, config in PRESETS.items():
            tile = OptionTile(
                body,
                name,
                config["description"],
                lambda preset=name: self._apply_preset(preset),
                width=340,
            )
            tile.pack(fill="x", pady=(0, 10))
            self._preset_tiles[name] = tile

    def _build_tools_card(self, parent):
        card = Surface(parent)
        card.pack(fill="x", pady=(14, 0))
        body = tk.Frame(card, bg=SURFACE)
        body.pack(fill="x", padx=18, pady=18)

        self._build_card_header(
            body,
            "Quick Tools",
            "A few quality-of-life actions that make the project feel less like a prototype and more like an operator console.",
        )

        PremiumButton(body, "Load Recommended Demo", self._load_recommended_demo, variant="secondary").pack(
            fill="x"
        )
        PremiumButton(body, "Restore Session Defaults", self._restore_session_defaults, variant="ghost").pack(
            fill="x",
            pady=(10, 0),
        )

        self._tools_note = tk.Label(
            body,
            text="",
            font=F_MICRO,
            fg=MUTED,
            bg=SURFACE,
            justify="left",
            wraplength=420,
        )
        self._tools_note.pack(anchor="w", pady=(14, 0))

    def _build_activity_card(self, parent):
        card = Surface(parent)
        card.pack(fill="both", expand=True, pady=(14, 0))
        body = tk.Frame(card, bg=SURFACE)
        body.pack(fill="both", expand=True, padx=18, pady=18)

        self._build_card_header(
            body,
            "Activity Feed",
            "Recent actions and runtime signals surface here so you do not need to babysit the terminal.",
        )

        self._activity_rows = []
        for _ in range(7):
            label = tk.Label(
                body,
                text="",
                font=F_MICRO,
                fg=MUTED,
                bg=BG_ALT,
                justify="left",
                anchor="w",
                padx=12,
                pady=10,
                wraplength=420,
                highlightbackground=BORDER,
                highlightthickness=1,
            )
            label.pack(fill="x", pady=(0, 8))
            self._activity_rows.append(label)

    def _make_path_field(self, parent, title, subtitle, variable, browse_command, button_label):
        tk.Label(parent, text=title, font=F_MICRO, fg=MUTED, bg=SURFACE).pack(anchor="w")
        tk.Label(
            parent,
            text=subtitle,
            font=F_MICRO,
            fg=MUTED,
            bg=SURFACE,
            justify="left",
        ).pack(anchor="w", pady=(2, 8))

        row = tk.Frame(parent, bg=SURFACE)
        row.pack(fill="x")
        entry = tk.Entry(
            row,
            textvariable=variable,
            bg=BG_ALT,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            highlightbackground=BORDER,
            highlightthickness=1,
            font=F_CARD_BODY,
        )
        entry.pack(side="left", fill="x", expand=True, ipady=7)
        PremiumButton(
            row,
            button_label,
            browse_command,
            variant="ghost",
            font=F_BUTTON_SMALL,
            padx=12,
            pady=8,
        ).pack(side="left", padx=(10, 0))

    def _build_model_display_field(self, parent):
        tk.Label(parent, text="Presentation model", font=F_MICRO, fg=MUTED, bg=SURFACE).pack(anchor="w")
        tk.Label(
            parent,
            text=(
                "The UI shows your project as YOLOv10 for presentation. "
                "Actual runtime weights stay configured internally."
            ),
            font=F_MICRO,
            fg=MUTED,
            bg=SURFACE,
            justify="left",
        ).pack(anchor="w", pady=(2, 8))

        row = tk.Frame(parent, bg=SURFACE)
        row.pack(fill="x")
        entry = tk.Entry(
            row,
            textvariable=self._model_display,
            state="readonly",
            readonlybackground=BG_ALT,
            fg=TEXT,
            relief="flat",
            highlightbackground=BORDER,
            highlightthickness=1,
            font=F_CARD_BODY,
        )
        entry.pack(side="left", fill="x", expand=True, ipady=7)
        PremiumButton(
            row,
            "Actual Weights...",
            self._browse_model,
            variant="ghost",
            font=F_BUTTON_SMALL,
            padx=12,
            pady=8,
        ).pack(side="left", padx=(10, 0))

        self._runtime_model_note = tk.Label(
            parent,
            text="",
            font=F_MICRO,
            fg=MUTED,
            bg=SURFACE,
            justify="left",
        )
        self._runtime_model_note.pack(anchor="w", pady=(8, 0))

    def _make_numeric_field(self, parent, row, column, title, subtitle, variable):
        frame = tk.Frame(parent, bg=SURFACE)
        frame.grid(row=row, column=column, sticky="nsew", padx=(0 if column == 0 else 8, 8), pady=(0, 14))

        tk.Label(frame, text=title, font=F_MICRO, fg=MUTED, bg=SURFACE).pack(anchor="w")
        tk.Label(
            frame,
            text=subtitle,
            font=F_MICRO,
            fg=MUTED,
            bg=SURFACE,
            justify="left",
            wraplength=210,
        ).pack(anchor="w", pady=(2, 8))
        entry = tk.Entry(
            frame,
            textvariable=variable,
            bg=BG_ALT,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            highlightbackground=BORDER,
            highlightthickness=1,
            font=F_CARD_BODY,
            width=12,
        )
        entry.pack(anchor="w", ipady=7)

    def _build_check(self, parent, title, subtitle, variable):
        frame = tk.Frame(parent, bg=SURFACE)
        check = tk.Checkbutton(
            frame,
            text=title,
            variable=variable,
            onvalue=True,
            offvalue=False,
            bg=SURFACE,
            fg=TEXT,
            activebackground=SURFACE,
            activeforeground=TEXT,
            selectcolor=BG_ALT,
            highlightthickness=0,
            font=("Segoe UI Semibold", 10),
        )
        check.pack(anchor="w")
        tk.Label(frame, text=subtitle, font=F_MICRO, fg=MUTED, bg=SURFACE, justify="left").pack(
            anchor="w",
            padx=(22, 0),
        )
        return frame

    def _refresh_dynamic_ui(self):
        source_mode = self._source.get()
        filepath = self._filepath.get().strip()
        source_label = pretty_source_label(source_mode, filepath, self._cam_idx.get().strip() or "0")
        model_path = self._model_path.get().strip()

        for name, tile in self._source_tiles.items():
            tile.set_selected(name == source_mode)

        if source_mode == "file":
            self._camera_row.pack_forget()
            self._file_row.pack(fill="x")
        else:
            self._file_row.pack_forget()
            self._camera_row.pack(fill="x")

        for path, button in self._demo_buttons:
            try:
                active = source_mode == "file" and filepath and Path(filepath).resolve() == path
            except OSError:
                active = False
            base_bg = SURFACE_SOFT if active else BG_ALT
            hover_bg = "#245177" if active else SURFACE_ALT
            button._style["bg"] = base_bg
            button._style["hover"] = hover_bg
            button._style["active"] = hover_bg
            button.configure(
                bg=base_bg,
                fg=TEXT if active else MUTED,
                activebackground=hover_bg,
                activeforeground=TEXT if active else MUTED,
                highlightbackground=TEAL if active else BORDER,
            )

        source_ready = source_mode == "camera" or (filepath and Path(filepath).exists())
        model_ready = bool(model_path) and Path(model_path).exists()
        geometry_ready = len(self._current_roi) >= 3
        alerts_text = "Enabled" if self._save_alerts.get() else "Disabled"
        visual_mode = "Train only" if self._train_only_view.get() else "Full overlays"

        self._hero_source_metric.set_value(source_label, TEAL if source_ready else RED)
        self._hero_geometry_metric.set_value(
            f"{len(self._current_roi)} pts / {self._crossing_y:.0%}",
            GOLD if geometry_ready else RED,
        )
        self._hero_profile_metric.set_value(self._preset_name, ACCENT)

        self._source_summary.configure(
            text=(
                f"Current source: {source_label}\n"
                f"Presentation model: {PRESENTATION_MODEL_LABEL}"
            )
        )
        self._runtime_model_note.configure(
            text=(
                "Actual runtime weights are ready."
                if model_ready
                else "Actual runtime weights are missing. Use 'Actual Weights...' if you need to fix the internal checkpoint."
            ),
            fg=MUTED if model_ready else RED,
        )
        self._geometry_summary.configure(
            text=(
                f"Current ROI has {len(self._current_roi)} points. Crossing line sits at "
                f"{self._crossing_y:.1%} of frame height. Runtime hotkeys in the monitor window "
                f"will also let you redraw ROI with [R], re-place the crossing line with [C], "
                f"and toggle train-only view with [V]."
            )
        )
        self._launch_summary.configure(
            text=(
                f"Session profile: {self._preset_name}. Risk thresholds are {self._high_ttc.get().strip()}s high / "
                f"{self._medium_ttc.get().strip()}s medium. Alerts are {alerts_text.lower()} and sound is "
                f"{'on' if self._enable_sound.get() else 'off'}. Visual mode starts in {visual_mode.lower()}."
            )
        )
        self._tools_note.configure(
            text=(
                "Tip: geometry defaults are stored separately from session defaults, so you can keep one trusted track layout "
                "while experimenting with detection presets."
            )
        )

        self._overview_tiles["source"].set_value("Ready" if source_ready else "Missing", GREEN if source_ready else RED)
        self._overview_tiles["model"].set_value(
            PRESENTATION_MODEL_FAMILY if model_ready else "Check weights",
            GREEN if model_ready else RED,
        )
        self._overview_tiles["geometry"].set_value(
            f"{len(self._current_roi)} pts",
            GOLD if geometry_ready else RED,
        )
        self._overview_tiles["alerts"].set_value(
            alerts_text,
            AMBER if self._save_alerts.get() else MUTED,
        )

        for name, tile in self._preset_tiles.items():
            tile.set_selected(name == self._preset_name)

        self._stop_button.configure(state="normal" if self._running else "disabled")
        self._launch_button.configure(state="disabled" if self._running else "normal")
        self._footer_stop_button.configure(state="normal" if self._running else "disabled")
        self._footer_launch_button.configure(state="disabled" if self._running else "normal")
        self._status_dot.configure(fg=GREEN if self._running else TEAL)

    def _browse_video(self):
        path = filedialog.askopenfilename(
            title="Select video source",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._source.set("file")
            self._filepath.set(path)
            self._add_log(f"Selected video source: {Path(path).name}")

    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="Select YOLO checkpoint",
            filetypes=[
                ("Model weights", "*.pt *.onnx"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._model_path.set(path)
            self._add_log(f"Switched model checkpoint to {Path(path).name}")

    def _select_demo(self, path):
        self._source.set("file")
        self._filepath.set(str(path))
        self._add_log(f"Loaded bundled demo: {path.name}")

    def _load_recommended_demo(self):
        if not self._sample_videos:
            messagebox.showinfo("No demos", "No bundled demo videos were found in the project root.")
            return
        self._select_demo(self._sample_videos[0])

    def _restore_session_defaults(self):
        defaults = PRESETS["Balanced"]
        self._preset_name = "Balanced"
        self._source.set("file")
        if self._sample_videos:
            self._filepath.set(str(self._sample_videos[0]))
        self._cam_idx.set("0")
        self._model_path.set("yolov8n.pt")
        self._conf.set(defaults["conf"])
        self._iou.set(defaults["iou"])
        self._known_speed.set(defaults["speed"])
        self._calib_frames.set(defaults["calib"])
        self._high_ttc.set(defaults["high"])
        self._medium_ttc.set(defaults["medium"])
        self._save_alerts.set(defaults["alerts"])
        self._enable_sound.set(defaults["sound"])
        self._train_only_view.set(False)
        self._add_log("Restored the recommended session defaults.")

    def _resolve_source_for_geometry(self):
        if self._source.get() == "camera":
            camera_value = self._cam_idx.get().strip()
            if not camera_value:
                camera_value = "0"
                self._cam_idx.set(camera_value)
            try:
                return int(camera_value)
            except ValueError:
                messagebox.showerror("Invalid camera", "Camera index must be an integer.")
                return None

        path = self._filepath.get().strip()
        if not path:
            messagebox.showwarning("No video selected", "Choose a video file before editing geometry.")
            return None
        if not Path(path).exists():
            messagebox.showerror("Missing video", f"Cannot find the selected video:\n{path}")
            return None
        return path

    def _edit_roi(self):
        source = self._resolve_source_for_geometry()
        if source is None:
            return
        self._add_log("Opening ROI editor...")
        updated = define_roi_interactively(source, "TRACK", self._current_roi)
        if len(updated) >= 3:
            self._current_roi = [tuple(point) for point in updated]
            self._persist_state()
            self._refresh_dynamic_ui()
            self._add_log(f"Updated ROI with {len(self._current_roi)} points.")

    def _edit_crossing_line(self):
        source = self._resolve_source_for_geometry()
        if source is None:
            return
        self._add_log("Opening crossing-line editor...")
        updated = define_crossing_line_interactively(
            source,
            roi_points=self._current_roi,
            initial_percent=self._crossing_y,
        )
        self._crossing_y = float(updated)
        self._persist_state()
        self._refresh_dynamic_ui()
        self._add_log(f"Crossing line moved to {self._crossing_y:.1%} of frame height.")

    def _preview_geometry(self):
        source = self._resolve_source_for_geometry()
        if source is None:
            return
        self._add_log("Showing geometry preview window.")
        preview_geometry(
            source,
            roi_points=self._current_roi,
            crossing_y_percent=self._crossing_y,
        )

    def _save_geometry_default(self):
        try:
            persist_roi_config(self._current_roi, self._crossing_y)
        except OSError as exc:
            messagebox.showerror("Save failed", f"Could not save geometry defaults.\n\n{exc}")
            return
        self._add_log("Saved current ROI and crossing line as the default geometry.")
        messagebox.showinfo("Geometry saved", "Current ROI and crossing line are now the default profile.")

    def _clear_saved_geometry(self):
        clear_saved_roi_config()
        self._add_log("Cleared saved default geometry profile.")
        messagebox.showinfo(
            "Geometry defaults cleared",
            "Saved geometry defaults were removed. The built-in defaults will be used next time.",
        )

    def _reset_geometry(self):
        self._current_roi = [tuple(point) for point in DEFAULT_TRACK_ROI]
        self._crossing_y = float(DEFAULT_CROSSING_Y_PERCENT)
        self._persist_state()
        self._refresh_dynamic_ui()
        self._add_log("Reset the working geometry to the built-in defaults.")

    def _apply_preset(self, name):
        config = PRESETS[name]
        self._preset_name = name
        self._conf.set(config["conf"])
        self._iou.set(config["iou"])
        self._known_speed.set(config["speed"])
        self._calib_frames.set(config["calib"])
        self._high_ttc.set(config["high"])
        self._medium_ttc.set(config["medium"])
        self._save_alerts.set(config["alerts"])
        self._enable_sound.set(config["sound"])
        self._add_log(f"Applied preset: {name}.")

    def _open_alerts_folder(self):
        ALERT_DIR.mkdir(exist_ok=True)
        os.startfile(str(ALERT_DIR))
        self._add_log("Opened the alerts folder.")

    def _parse_numeric_settings(self):
        try:
            conf = float(self._conf.get().strip())
            iou = float(self._iou.get().strip())
            speed = float(self._known_speed.get().strip())
            calib = int(float(self._calib_frames.get().strip()))
            high_ttc = float(self._high_ttc.get().strip())
            medium_ttc = float(self._medium_ttc.get().strip())
        except ValueError:
            raise ValueError("Numeric tuning fields must contain valid numbers.") from None

        if not 0.0 < conf <= 1.0:
            raise ValueError("Confidence must be between 0 and 1.")
        if not 0.0 < iou <= 1.0:
            raise ValueError("IOU must be between 0 and 1.")
        if speed <= 0:
            raise ValueError("Train speed must be a positive number.")
        if calib < 3:
            raise ValueError("Calibration frames must be at least 3.")
        if high_ttc <= 0 or medium_ttc <= 0:
            raise ValueError("Risk TTC values must be positive.")
        if medium_ttc <= high_ttc:
            raise ValueError("Medium-risk TTC must be greater than high-risk TTC.")

        return conf, iou, speed, calib, high_ttc, medium_ttc

    def _build_runtime_patch(self, source_value, numeric_settings):
        conf, iou, speed, calib, high_ttc, medium_ttc = numeric_settings
        source_label = pretty_source_label(
            self._source.get(),
            self._filepath.get().strip(),
            self._cam_idx.get().strip() or "0",
        )
        overrides = {
            "VIDEO_SOURCE": source_value,
            "MODEL_PATH": self._model_path.get().strip(),
            "CONF_THRESHOLD": conf,
            "IOU_THRESHOLD": iou,
            "KNOWN_TRAIN_SPEED_KMH": speed,
            "AUTO_CALIB_FRAMES": calib,
            "HIGH_RISK_TTC": high_ttc,
            "MEDIUM_RISK_TTC": medium_ttc,
            "SAVE_ALERTS": bool(self._save_alerts.get()),
            "ENABLE_SOUND": bool(self._enable_sound.get()),
            "TRAIN_ONLY_VIEW": bool(self._train_only_view.get()),
            "TRACK_ROI": self._current_roi,
            "CROSSING_Y_PERCENT": self._crossing_y,
            "SESSION_LABEL": source_label,
        }
        lines = ["import json", "import main as _m"]
        for key, value in overrides.items():
            serialized = json.dumps(value)
            lines.append(f"_m.{key} = json.loads({serialized!r})")
        lines.append("_m.main()")
        return "\n".join(lines)

    def _launch(self):
        if self._running:
            return

        source_mode = self._source.get()
        filepath = self._filepath.get().strip()
        if source_mode == "file":
            if not filepath:
                messagebox.showwarning("No video", "Choose a video file before launching the detector.")
                return
            if not Path(filepath).exists():
                messagebox.showerror("Missing video", f"Cannot find the selected video:\n{filepath}")
                return
            source_value = filepath
        else:
            camera_value = self._cam_idx.get().strip()
            try:
                source_value = int(camera_value)
            except ValueError:
                messagebox.showerror("Invalid camera", "Camera index must be an integer.")
                return

        model_path = self._model_path.get().strip()
        if not model_path:
            messagebox.showerror("Missing model", "Select a YOLO checkpoint before launching.")
            return
        if not Path(model_path).exists():
            messagebox.showerror("Missing model", f"Cannot find the selected model checkpoint:\n{model_path}")
            return

        if len(self._current_roi) < 3:
            messagebox.showerror("Invalid ROI", "ROI must contain at least 3 points.")
            return

        try:
            numeric_settings = self._parse_numeric_settings()
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        runtime_patch = self._build_runtime_patch(source_value, numeric_settings)
        self._running = True
        self._proc = None
        self._refresh_dynamic_ui()
        self._set_status("Monitoring session running in the OpenCV window.", GREEN)
        self._add_log("Launching monitoring session...")

        def _run():
            self._proc = subprocess.Popen(
                [sys.executable, "-c", runtime_patch],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(APP_DIR),
            )
            if self._proc.stdout is not None:
                for line in self._proc.stdout:
                    cleaned = line.strip()
                    if cleaned:
                        self.after(0, self._handle_runtime_line, cleaned)
            return_code = self._proc.wait()
            self.after(0, self._on_process_done, return_code)

        threading.Thread(target=_run, daemon=True).start()

    def _handle_runtime_line(self, line):
        muted_line = line.replace("[INFO]", "").strip()
        self._add_log(muted_line or line)

    def _stop(self):
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            self._add_log("Stop signal sent to the monitoring session.")
            self._set_status("Stopping the running session...", AMBER)
        except Exception as exc:
            messagebox.showerror("Stop failed", f"Could not stop the running process.\n\n{exc}")

    def _on_process_done(self, return_code):
        self._running = False
        self._proc = None
        self._refresh_dynamic_ui()
        if return_code == 0:
            self._set_status("Session ended cleanly.", TEAL)
            self._add_log("Monitoring session finished.")
        else:
            self._set_status(f"Session exited with code {return_code}.", AMBER)
            self._add_log(f"Monitoring session exited with code {return_code}.")

    def _set_status(self, message, color=TEAL):
        self._status.set(message)
        self._status_dot.configure(fg=color)

    def _add_log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        entry = f"{timestamp}  {message}"
        self._log_messages.insert(0, entry)
        self._log_messages = self._log_messages[: len(self._activity_rows)]
        for idx, row in enumerate(self._activity_rows):
            row.configure(text=self._log_messages[idx] if idx < len(self._log_messages) else "")

    def _on_close(self):
        self._persist_state()
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass
        self.destroy()


if __name__ == "__main__":
    app = RAPSLauncher()
    app.mainloop()

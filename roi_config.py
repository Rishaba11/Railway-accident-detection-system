import json
from pathlib import Path

import cv2
import numpy as np


CONFIG_PATH = Path(__file__).with_name("roi_preset.json")

DEFAULT_TRACK_ROI = [
    (0, 500),
    (0, 360),
    (1580, 500),
    (1420, 1400),
]
DEFAULT_CROSSING_Y_PERCENT = 0.55

EDITOR_MAX_WIDTH = 1500
EDITOR_MAX_HEIGHT = 900
POINT_HIT_RADIUS = 18


def _sanitize_roi(points):
    cleaned = []
    for point in points or []:
        if len(point) != 2:
            continue
        try:
            x = int(point[0])
            y = int(point[1])
        except (TypeError, ValueError):
            continue
        cleaned.append((x, y))
    return cleaned


def _sanitize_crossing(value):
    try:
        crossing = float(value)
    except (TypeError, ValueError):
        return DEFAULT_CROSSING_Y_PERCENT
    return min(0.95, max(0.05, crossing))


def _load_saved_geometry():
    if not CONFIG_PATH.exists():
        return list(DEFAULT_TRACK_ROI), DEFAULT_CROSSING_Y_PERCENT

    try:
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return list(DEFAULT_TRACK_ROI), DEFAULT_CROSSING_Y_PERCENT

    roi_points = _sanitize_roi(payload.get("track_roi"))
    if len(roi_points) < 3:
        roi_points = list(DEFAULT_TRACK_ROI)

    crossing = _sanitize_crossing(payload.get("crossing_y_percent"))
    return roi_points, crossing


TRACK_ROI, CROSSING_Y_PERCENT = _load_saved_geometry()


def normalize_video_source(video_source):
    if isinstance(video_source, str):
        candidate = video_source.strip()
        if candidate.isdigit() and not Path(candidate).exists():
            return int(candidate)
        return candidate
    return video_source


def capture_reference_frame(video_source="railway_video.mp4"):
    source = normalize_video_source(video_source)
    cap = cv2.VideoCapture(source)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"[ERROR] Could not open video source: {video_source}")
        return None
    return frame


def _prepare_display_frame(frame):
    height, width = frame.shape[:2]
    scale = min(
        1.0,
        EDITOR_MAX_WIDTH / float(width),
        EDITOR_MAX_HEIGHT / float(height),
    )
    if scale == 1.0:
        return frame.copy(), scale

    display = cv2.resize(
        frame,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_LINEAR,
    )
    return display, scale


def _scale_points(points, scale):
    scaled = []
    for x, y in points:
        scaled.append((int(round(x * scale)), int(round(y * scale))))
    return scaled


def _clip_point(point, frame_shape):
    height, width = frame_shape[:2]
    x = max(0, min(width - 1, int(round(point[0]))))
    y = max(0, min(height - 1, int(round(point[1]))))
    return (x, y)


def _source_point_from_display(x, y, scale, frame_shape):
    if scale <= 0:
        return _clip_point((x, y), frame_shape)
    return _clip_point((x / scale, y / scale), frame_shape)


def _nearest_point_index(display_point, points, scale, max_distance=POINT_HIT_RADIUS):
    if not points:
        return None

    best_index = None
    best_distance = max_distance * max_distance
    display_x, display_y = display_point

    for index, (px, py) in enumerate(_scale_points(points, scale)):
        dx = px - display_x
        dy = py - display_y
        distance = dx * dx + dy * dy
        if distance <= best_distance:
            best_distance = distance
            best_index = index

    return best_index


def _draw_top_panel(canvas, title, footer_lines):
    footer_lines = footer_lines if isinstance(footer_lines, list) else [footer_lines]
    panel_height = 82 + (max(0, len(footer_lines) - 1) * 22)
    panel = canvas.copy()
    cv2.rectangle(panel, (12, 12), (canvas.shape[1] - 12, 12 + panel_height), (7, 15, 27), -1)
    cv2.addWeighted(panel, 0.78, canvas, 0.22, 0, canvas)
    cv2.rectangle(canvas, (12, 12), (canvas.shape[1] - 12, 12 + panel_height), (76, 106, 138), 1)
    cv2.putText(
        canvas,
        title,
        (26, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.90,
        (245, 247, 250),
        2,
    )
    y = 72
    for line in footer_lines:
        cv2.putText(
            canvas,
            line,
            (26, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (192, 205, 221),
            1,
        )
        y += 22


def _render_geometry_view(
    frame,
    title,
    footer_lines,
    roi_points=None,
    crossing_y_percent=None,
    selected_index=None,
    guide_points=None,
):
    display, scale = _prepare_display_frame(frame)

    if guide_points and len(guide_points) >= 3:
        guide_display = _scale_points(guide_points, scale)
        guide_np = np.array(guide_display, dtype=np.int32)
        guide_overlay = display.copy()
        cv2.fillPoly(guide_overlay, [guide_np], (28, 48, 71))
        cv2.addWeighted(guide_overlay, 0.10, display, 0.90, 0, display)
        cv2.polylines(display, [guide_np], True, (135, 153, 177), 1)

    if roi_points and len(roi_points) >= 3:
        roi_display = _scale_points(roi_points, scale)
        roi_np = np.array(roi_display, dtype=np.int32)
        overlay = display.copy()
        cv2.fillPoly(overlay, [roi_np], (34, 82, 138))
        cv2.addWeighted(overlay, 0.24, display, 0.76, 0, display)
        cv2.polylines(display, [roi_np], True, (104, 226, 223), 3)
    elif roi_points and len(roi_points) > 1:
        roi_display = _scale_points(roi_points, scale)
        cv2.polylines(display, [np.array(roi_display, dtype=np.int32)], False, (104, 226, 223), 3)

    if roi_points:
        for index, (px, py) in enumerate(_scale_points(roi_points, scale)):
            active = selected_index == index
            circle_color = (84, 118, 255) if active else (255, 233, 160)
            cv2.circle(display, (px, py), 8 if active else 6, circle_color, -1)
            cv2.circle(display, (px, py), 10 if active else 8, (255, 255, 255), 1)
            cv2.putText(
                display,
                str(index + 1),
                (px + 10, py - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                circle_color,
                2,
            )

    if crossing_y_percent is not None:
        display_y = int(round(_sanitize_crossing(crossing_y_percent) * display.shape[0]))
        label_y = max(30, min(display.shape[0] - 16, display_y - 10))
        cv2.line(display, (0, display_y), (display.shape[1], display_y), (255, 196, 110), 3)
        cv2.putText(
            display,
            "CROSSING LINE",
            (18, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 224, 165),
            2,
        )

    _draw_top_panel(display, title, footer_lines)
    return display, scale


def preview_geometry_on_frame(
    frame,
    roi_points=None,
    crossing_y_percent=None,
    window_name="Geometry Preview",
):
    roi_points = _sanitize_roi(roi_points if roi_points is not None else TRACK_ROI)
    crossing = (
        _sanitize_crossing(crossing_y_percent)
        if crossing_y_percent is not None
        else CROSSING_Y_PERCENT
    )
    display, _scale = _render_geometry_view(
        frame,
        "Track geometry preview",
        [
            "Press any key to close this preview window.",
        ],
        roi_points=roi_points,
        crossing_y_percent=crossing,
    )

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display.shape[1], display.shape[0])
    cv2.imshow(window_name, display)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)
    cv2.waitKey(1)
    return True


def preview_geometry(
    video_source="railway_video.mp4",
    roi_points=None,
    crossing_y_percent=None,
    window_name="Geometry Preview",
):
    frame = capture_reference_frame(video_source)
    if frame is None:
        return False
    return preview_geometry_on_frame(frame, roi_points, crossing_y_percent, window_name)


def edit_roi_on_frame(
    frame,
    roi_name="TRACK",
    initial_points=None,
    crossing_y_percent=None,
):
    state = {
        "points": _sanitize_roi(initial_points),
        "selected_index": 0 if initial_points else None,
        "dragging": False,
    }
    original_points = list(state["points"])
    window_name = f"Set {roi_name} ROI"

    def render():
        display, _scale = _render_geometry_view(
            frame,
            f"{roi_name} ROI editor",
            [
                "Drag existing points to edit the current ROI.",
                "Right click adds a point. Backspace/U removes selected. N clears. Enter saves. Esc cancels.",
            ],
            roi_points=state["points"],
            crossing_y_percent=crossing_y_percent,
            selected_index=state["selected_index"],
        )
        cv2.putText(
            display,
            f"Points: {len(state['points'])} (minimum 3)",
            (26, display.shape[0] - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.64,
            (255, 240, 190),
            2,
        )
        cv2.imshow(window_name, display)

    def on_mouse(event, x, y, flags, param):
        _display_frame, scale = _prepare_display_frame(frame)
        source_point = _source_point_from_display(x, y, scale, frame.shape)

        if event == cv2.EVENT_LBUTTONDOWN:
            hit_index = _nearest_point_index((x, y), state["points"], scale)
            if hit_index is not None:
                state["selected_index"] = hit_index
                state["dragging"] = True
                state["points"][hit_index] = source_point
                render()
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"] and state["selected_index"] is not None:
            state["points"][state["selected_index"]] = source_point
            render()
        elif event == cv2.EVENT_LBUTTONUP and state["dragging"]:
            if state["selected_index"] is not None:
                state["points"][state["selected_index"]] = source_point
            state["dragging"] = False
            render()
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["points"].append(source_point)
            state["selected_index"] = len(state["points"]) - 1
            render()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    initial_display, _initial_scale = _render_geometry_view(
        frame,
        f"{roi_name} ROI editor",
        [""],
        roi_points=state["points"],
        crossing_y_percent=crossing_y_percent,
    )
    cv2.resizeWindow(window_name, initial_display.shape[1], initial_display.shape[0])
    cv2.setMouseCallback(window_name, on_mouse)
    render()

    confirmed = False
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 13 and len(state["points"]) >= 3:
            confirmed = True
            break
        if key == 27:
            break
        if key in (8, 127, ord("u")) and state["points"]:
            remove_index = state["selected_index"] if state["selected_index"] is not None else len(state["points"]) - 1
            remove_index = max(0, min(remove_index, len(state["points"]) - 1))
            state["points"].pop(remove_index)
            if state["points"]:
                state["selected_index"] = min(remove_index, len(state["points"]) - 1)
            else:
                state["selected_index"] = None
            render()
        if key in (ord("n"), ord("c")):
            state["points"].clear()
            state["selected_index"] = None
            render()

    cv2.destroyWindow(window_name)
    cv2.waitKey(1)

    if confirmed:
        return _sanitize_roi(state["points"])
    if original_points:
        return original_points
    return None


def define_roi_interactively(
    video_source="railway_video.mp4",
    roi_name="TRACK",
    initial_points=None,
):
    frame = capture_reference_frame(video_source)
    if frame is None:
        return list(initial_points or TRACK_ROI)

    points = edit_roi_on_frame(
        frame,
        roi_name=roi_name,
        initial_points=initial_points if initial_points is not None else TRACK_ROI,
    )
    return list(points or initial_points or TRACK_ROI)


def edit_crossing_line_on_frame(
    frame,
    roi_points=None,
    initial_percent=DEFAULT_CROSSING_Y_PERCENT,
):
    state = {
        "crossing_percent": _sanitize_crossing(initial_percent),
        "dragging": False,
    }
    roi_points = _sanitize_roi(roi_points if roi_points is not None else TRACK_ROI)
    window_name = "Set Crossing Line"

    def render():
        display, _scale = _render_geometry_view(
            frame,
            "Crossing line editor",
            [
                "Drag or click the crossing line to reposition it.",
                "Enter saves the new line. Esc cancels.",
            ],
            roi_points=roi_points,
            crossing_y_percent=state["crossing_percent"],
        )
        cv2.putText(
            display,
            f"Crossing line: {state['crossing_percent']:.1%} of frame height",
            (26, display.shape[0] - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.64,
            (255, 240, 190),
            2,
        )
        cv2.imshow(window_name, display)

    def on_mouse(event, x, y, flags, param):
        _display_frame, scale = _prepare_display_frame(frame)
        source_point = _source_point_from_display(x, y, scale, frame.shape)
        crossing_percent = source_point[1] / float(frame.shape[0])

        if event == cv2.EVENT_LBUTTONDOWN:
            state["dragging"] = True
            state["crossing_percent"] = _sanitize_crossing(crossing_percent)
            render()
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            state["crossing_percent"] = _sanitize_crossing(crossing_percent)
            render()
        elif event == cv2.EVENT_LBUTTONUP:
            state["dragging"] = False
            state["crossing_percent"] = _sanitize_crossing(crossing_percent)
            render()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    initial_display, _initial_scale = _render_geometry_view(
        frame,
        "Crossing line editor",
        [""],
        roi_points=roi_points,
        crossing_y_percent=state["crossing_percent"],
    )
    cv2.resizeWindow(window_name, initial_display.shape[1], initial_display.shape[0])
    cv2.setMouseCallback(window_name, on_mouse)
    render()

    confirmed = False
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 13:
            confirmed = True
            break
        if key == 27:
            break

    cv2.destroyWindow(window_name)
    cv2.waitKey(1)

    if confirmed:
        return _sanitize_crossing(state["crossing_percent"])
    return None


def define_crossing_line_interactively(
    video_source="railway_video.mp4",
    roi_points=None,
    initial_percent=DEFAULT_CROSSING_Y_PERCENT,
):
    frame = capture_reference_frame(video_source)
    if frame is None:
        return _sanitize_crossing(initial_percent)

    crossing = edit_crossing_line_on_frame(
        frame,
        roi_points=roi_points,
        initial_percent=initial_percent,
    )
    if crossing is None:
        return _sanitize_crossing(initial_percent)
    return crossing


def persist_roi_config(track_roi=None, crossing_y_percent=None, config_path=CONFIG_PATH):
    payload = {
        "track_roi": _sanitize_roi(track_roi if track_roi is not None else TRACK_ROI),
        "crossing_y_percent": _sanitize_crossing(
            crossing_y_percent if crossing_y_percent is not None else CROSSING_Y_PERCENT
        ),
    }
    Path(config_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def clear_saved_roi_config(config_path=CONFIG_PATH):
    path = Path(config_path)
    if path.exists():
        path.unlink()


if __name__ == "__main__":
    print("=== ROI Calibration Tool ===")
    points = define_roi_interactively("railway_video.mp4", "TRACK", TRACK_ROI)
    crossing = define_crossing_line_interactively(
        "railway_video.mp4",
        roi_points=points,
        initial_percent=CROSSING_Y_PERCENT,
    )
    print(f"TRACK_ROI = {points}")
    print(f"CROSSING_Y_PERCENT = {crossing:.4f}")

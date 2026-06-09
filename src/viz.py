"""Bird's-eye-view (BEV) matplotlib renderer.

This is the only visualization in the project (no camera image, no LiDAR).
Coordinate convention:
  - ego frame: +x forward, +y left, +z up.
  - on screen we map (ego_x, ego_y) -> (screen_x, screen_y) = (-ego_y, ego_x)
    so that "forward" points up and "left" stays on the left.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle

from src.object_list import DetectedObject, EgoState
from src.reasoning.arbiter import Decision

# Class -> (face color, edge color)
_CLASS_STYLE: dict[str, tuple[str, str]] = {
    "VEHICLE":       ("#4F86E0", "#1F4FA8"),
    "LARGE_VEHICLE": ("#23408E", "#0E2050"),
    "PEDESTRIAN":    ("#D9534F", "#7A2120"),
    "CYCLIST":       ("#F0AD4E", "#A0651E"),
    "STATIC":        ("#9A9A9A", "#5A5A5A"),
}

# Decision action -> banner color (used in title bar)
ACTION_COLORS: dict[str, str] = {
    "BRAKE":               "#D9534F",
    "STOP":                "#D9534F",
    "YIELD":               "#F0AD4E",
    "FOLLOW":              "#F0AD4E",
    "INHIBIT_LANE_CHANGE": "#5BC0DE",
    "CRUISE":              "#5CB85C",
}

# Default plot extents (metres). Forward bias because most objects of
# interest are ahead.
DEFAULT_X_RANGE = (-25.0, 80.0)   # forward
DEFAULT_Y_RANGE = (-20.0, 20.0)   # lateral

# Theme palettes. "light" keeps the default behavior so existing tests
# (and CLI demos) stay unchanged. "dark" matches the futurist UI.
_THEMES = {
    "light": dict(
        fig_bg="white", ax_bg="#f6f6f6",
        grid="#dddddd", tick="#222222",
        ego_lane="#cfe2ff", adj_lane="#e6eef9", lane_alpha=0.55,
        ped_caution="#d99494", ped_stop="#b04848",
        text="#222222", text_bg="white", text_border="#cccccc",
        title_default="#444444",
    ),
    "dark": dict(
        fig_bg="#04070B", ax_bg="#0A1420",
        grid="#15202B", tick="#7A8C99",
        ego_lane="#0E2A45", adj_lane="#0B1F33", lane_alpha=0.70,
        ped_caution="#7B3A3A", ped_stop="#A85050",
        text="#D5E0E8", text_bg="#0E1B26", text_border="#1F3441",
        title_default="#D5E0E8",
    ),
}


def _ego_to_screen(x: float, y: float) -> tuple[float, float]:
    """ego (forward, left) -> screen (right, up)."""
    return (-y, x)


def _box_corners_screen(obj: DetectedObject) -> np.ndarray:
    """Return 4 corners of the object's box in screen coords (4x2)."""
    w, l, _h = obj.size
    local = np.array(
        [
            [+l / 2.0, +w / 2.0],
            [+l / 2.0, -w / 2.0],
            [-l / 2.0, -w / 2.0],
            [-l / 2.0, +w / 2.0],
        ]
    )
    c, s = np.cos(obj.yaw), np.sin(obj.yaw)
    R = np.array([[c, -s], [s, c]])
    corners_ego = local @ R.T + np.array([obj.x, obj.y])
    # Map each (ego_x, ego_y) -> (-ego_y, ego_x)
    return np.column_stack([-corners_ego[:, 1], corners_ego[:, 0]])


def _draw_lane_bands(ax, rules: dict[str, Any], theme: dict) -> None:
    half = float(rules["lane_half_width_m"])
    adj = float(rules["adjacent_lane_max_m"])
    y_lo, y_hi = ax.get_ylim()
    ax.add_patch(
        Rectangle(
            (-half, y_lo), 2 * half, y_hi - y_lo,
            facecolor=theme["ego_lane"], edgecolor="none",
            alpha=theme["lane_alpha"], zorder=0,
        )
    )
    for x0 in (-adj, half):
        ax.add_patch(
            Rectangle(
                (x0, y_lo), adj - half, y_hi - y_lo,
                facecolor=theme["adj_lane"], edgecolor="none",
                alpha=theme["lane_alpha"], zorder=0,
            )
        )


def _draw_ped_radii(ax, rules: dict[str, Any], theme: dict) -> None:
    """Faint circles at the pedestrian caution and stop radii."""
    from matplotlib.patches import Circle
    for r, color in (
        (float(rules["ped_caution_radius_m"]), theme["ped_caution"]),
        (float(rules["ped_stop_radius_m"]), theme["ped_stop"]),
    ):
        ax.add_patch(
            Circle(
                (0.0, 0.0), r, fill=False, linestyle=":",
                edgecolor=color, linewidth=1.0, alpha=0.6, zorder=1,
            )
        )


def _draw_ego(ax, theme: dict) -> None:
    """Ego triangle at origin pointing up (screen +y)."""
    tri = np.array([[0.0, 2.2], [-1.0, -1.4], [+1.0, -1.4]])
    edge = "white" if theme["ax_bg"] != "#0A1420" else "#3FE0C5"
    face = "#222" if theme["ax_bg"] != "#0A1420" else "#0E1B26"
    ax.add_patch(
        Polygon(tri, closed=True, facecolor=face, edgecolor=edge,
                linewidth=1.6, zorder=5)
    )
    ax.plot([0.0], [0.0], "o", color=edge, markersize=3, zorder=6)


def _draw_object(
    ax,
    obj: DetectedObject,
    *,
    label_distances: bool,
    screen_bounds: tuple[float, float, float, float],
    theme: dict,
) -> None:
    """``screen_bounds`` = (sx_lo, sx_hi, sy_lo, sy_hi) for label clipping."""
    face, edge = _CLASS_STYLE.get(obj.cls, _CLASS_STYLE["STATIC"])
    alpha = 0.85 if obj.state == "MOVING" else 0.60
    corners = _box_corners_screen(obj)
    ax.add_patch(
        Polygon(
            corners, closed=True, facecolor=face, edgecolor=edge,
            linewidth=1.0, alpha=alpha, zorder=3,
        )
    )
    sx, sy = _ego_to_screen(obj.x, obj.y)
    fx, fy = np.cos(obj.yaw), np.sin(obj.yaw)
    fex, fey = _ego_to_screen(obj.x + 0.6 * fx, obj.y + 0.6 * fy)
    ax.plot([sx, fex], [sy, fey], "-", color=edge, linewidth=1.2, zorder=4)

    if not label_distances:
        return
    sx_lo, sx_hi, sy_lo, sy_hi = screen_bounds
    if not (sx_lo + 1 < sx < sx_hi - 1 and sy_lo + 1 < sy < sy_hi - 1):
        return
    if obj.distance > 40.0:
        return
    ax.text(
        sx + 0.6, sy + 0.4,
        f"{obj.distance:.0f}m",
        fontsize=7, color=theme["text"],
        ha="left", va="bottom", zorder=6,
        clip_on=True,
    )


def _draw_legend(ax, theme: dict) -> None:
    handles = []
    for cls in ("VEHICLE", "LARGE_VEHICLE", "PEDESTRIAN", "CYCLIST", "STATIC"):
        face, edge = _CLASS_STYLE[cls]
        handles.append(
            plt.Line2D(
                [0], [0], marker="s", color="none",
                markerfacecolor=face, markeredgecolor=edge,
                markersize=10, label=cls,
            )
        )
    leg = ax.legend(
        handles=handles, loc="lower right", fontsize=8,
        framealpha=0.85, facecolor=theme["text_bg"],
        edgecolor=theme["text_border"], labelcolor=theme["text"],
    )


def plot_bev(
    ego: EgoState,
    objects: Iterable[DetectedObject],
    decision: Optional[Decision],
    rules: dict[str, Any],
    *,
    x_range: tuple[float, float] = DEFAULT_X_RANGE,
    y_range: tuple[float, float] = DEFAULT_Y_RANGE,
    label_distances: bool = True,
    figsize: tuple[float, float] = (6.5, 8.5),
    theme: str = "light",
    show_title: bool = True,
) -> Figure:
    """Render a BEV figure. Caller is responsible for closing/displaying.

    The screen y-axis is the ego forward direction; screen x-axis is the
    ego left (positive ego_y → negative screen_x).

    ``theme`` is ``"light"`` (default, used by tests and CLI) or ``"dark"``
    (used by the Streamlit app).
    """
    t = _THEMES.get(theme, _THEMES["light"])
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(t["fig_bg"])
    ax.set_facecolor(t["ax_bg"])

    screen_x_lim = (-y_range[1], -y_range[0])
    screen_y_lim = x_range
    ax.set_xlim(*screen_x_lim)
    ax.set_ylim(*screen_y_lim)
    ax.set_aspect("equal", adjustable="box")

    ax.grid(True, color=t["grid"], linestyle="-", linewidth=0.6)
    ax.set_xticks(np.arange(np.ceil(screen_x_lim[0] / 10) * 10,
                            screen_x_lim[1] + 1, 10))
    ax.set_yticks(np.arange(np.ceil(screen_y_lim[0] / 10) * 10,
                            screen_y_lim[1] + 1, 10))
    ax.tick_params(labelsize=8, colors=t["tick"])
    for spine in ax.spines.values():
        spine.set_edgecolor(t["grid"])
    ax.set_xlabel("← right    lateral (m)    left →", fontsize=9, color=t["tick"])
    ax.set_ylabel("forward (m) ↑", fontsize=9, color=t["tick"])

    _draw_lane_bands(ax, rules, t)
    _draw_ped_radii(ax, rules, t)
    _draw_ego(ax, t)

    screen_bounds = (screen_x_lim[0], screen_x_lim[1], screen_y_lim[0], screen_y_lim[1])
    for obj in objects:
        _draw_object(
            ax, obj,
            label_distances=label_distances,
            screen_bounds=screen_bounds,
            theme=t,
        )

    if decision is not None and show_title:
        color = ACTION_COLORS.get(decision.action, t["title_default"])
        ax.set_title(
            f"DECISION: {decision.action}  (priority {decision.priority})  "
            f"— {decision.num_objects} objects",
            fontsize=11, fontweight="bold", color=color, pad=10,
        )

    _draw_legend(ax, t)
    fig.tight_layout()
    return fig

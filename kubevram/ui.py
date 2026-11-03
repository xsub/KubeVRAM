"""
KubeVRAM Terminal UI Rendering Engine.

Provides zero-dependency ANSI formatting, progress bars, severity badges,
Unicode box-drawing tables, and callout panels.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from typing import List, Optional

# Color support detection (honor NO_COLOR standard)
_NO_COLOR = bool(os.environ.get("NO_COLOR"))

if _NO_COLOR:
    BOLD = ""
    DIM = ""
    RED = ""
    GREEN = ""
    YELLOW = ""
    BLUE = ""
    MAGENTA = ""
    CYAN = ""
    WHITE = ""
    RESET = ""
else:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    RESET = "\033[0m"

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def strip_ansi(text: str) -> str:
    """Strips ANSI escape sequences from text for accurate character-width calculation."""
    return ANSI_ESCAPE_RE.sub("", text)


def colorize(text: str, *styles: str) -> str:
    """Wraps text in ANSI style codes and resets afterwards."""
    if not styles or _NO_COLOR:
        return text
    return "".join(styles) + text + RESET


def vram_bar(used_mib: int, total_mib: int, width: int = 18) -> str:
    """
    Renders a color-coded progress bar (e.g., '[██████████░░░░] 99.4%').
    - >= 92% is RED (critical memory pressure / pre-OOM)
    - >= 75% is YELLOW (high allocation / KV cache threshold)
    - else GREEN (healthy memory headroom)
    """
    if total_mib <= 0:
        ratio = 0.0
        pct = 0.0
    else:
        ratio = min(max(used_mib / total_mib, 0.0), 1.0)
        pct = (used_mib / total_mib) * 100.0

    filled = int(round(ratio * width))
    filled = min(width, max(0, filled))
    empty = width - filled
    bar = "█" * filled + "░" * empty

    if pct >= 92.0:
        color = RED
    elif pct >= 75.0:
        color = YELLOW
    else:
        color = GREEN

    return f"{color}[{bar}] {pct:.1f}%{RESET}"


def severity_badge(severity: str) -> str:
    """Returns a color-coded severity badge ([CRITICAL], [HIGH], [MEDIUM], [LOW], [OK])."""
    s = (severity or "").upper().strip()
    if s == "CRITICAL":
        return f"{RED}{BOLD}[CRITICAL]{RESET}"
    elif s == "HIGH":
        return f"{RED}[HIGH]{RESET}"
    elif s == "MEDIUM":
        return f"{YELLOW}[MEDIUM]{RESET}"
    elif s == "LOW":
        return f"{CYAN}[LOW]{RESET}"
    elif s in ("OK", "INFO"):
        return f"{GREEN}[OK]{RESET}"
    else:
        return f"{DIM}[{s or 'UNKNOWN'}]{RESET}"


def render_table(
    headers: List[str],
    rows: List[List[str]],
    title: Optional[str] = None,
) -> str:
    """
    Renders a clean Unicode box-drawing table (┌─┬─┐, │ │ │, ├─┼─┤, └─┴─┘),
    stripping ANSI escape codes when calculating column widths so alignment is always pixel-perfect.
    """
    if not headers:
        return ""

    num_cols = len(headers)
    col_widths = [len(strip_ansi(h)) for h in headers]

    for row in rows:
        for i in range(num_cols):
            cell_str = str(row[i]) if i < len(row) else ""
            col_widths[i] = max(col_widths[i], len(strip_ansi(cell_str)))

    total_inner_width = sum(w + 2 for w in col_widths) + max(0, num_cols - 1)

    if title:
        title_len = len(strip_ansi(title))
        if title_len + 2 > total_inner_width:
            diff = (title_len + 2) - total_inner_width
            col_widths[-1] += diff
            total_inner_width = sum(w + 2 for w in col_widths) + max(0, num_cols - 1)

    lines: List[str] = []

    # Title or Top border
    if title:
        lines.append("┌" + "─" * total_inner_width + "┐")
        title_clean_len = len(strip_ansi(title))
        pad_right = total_inner_width - 2 - title_clean_len
        lines.append("│ " + BOLD + title + RESET + (" " * max(0, pad_right)) + " │")
        lines.append("├" + "┬".join("─" * (w + 2) for w in col_widths) + "┤")
    else:
        lines.append("┌" + "┬".join("─" * (w + 2) for w in col_widths) + "┐")

    # Header row
    header_cells = []
    for h, w in zip(headers, col_widths):
        pad = w - len(strip_ansi(h))
        header_cells.append(f" {BOLD}{h}{RESET}{' ' * pad} ")
    lines.append("│" + "│".join(header_cells) + "│")

    # Separator
    lines.append("├" + "┼".join("─" * (w + 2) for w in col_widths) + "┤")

    # Data rows
    for row in rows:
        row_cells = []
        for idx, w in enumerate(col_widths):
            cell = str(row[idx]) if idx < len(row) else ""
            pad = w - len(strip_ansi(cell))
            row_cells.append(f" {cell}{' ' * pad} ")
        lines.append("│" + "│".join(row_cells) + "│")

    # Bottom border
    lines.append("└" + "┴".join("─" * (w + 2) for w in col_widths) + "┘")

    return "\n".join(lines)


def render_panel(
    title: str,
    lines: List[str],
    border_color: str = CYAN,
) -> str:
    """Renders a boxed callout panel with Unicode borders and an optional colored title."""
    flat_lines: List[str] = []
    for line in lines:
        for subline in str(line).split("\n"):
            flat_lines.append(subline)

    title_clean = strip_ansi(title)
    max_content_len = max(
        [len(strip_ansi(l)) for l in flat_lines] + [len(title_clean) + 4],
        default=40,
    )
    term_width = shutil.get_terminal_size((80, 24)).columns
    panel_width = min(term_width, max(max_content_len + 4, 60))
    content_width = panel_width - 4

    out: List[str] = []
    if title_clean:
        title_disp = f" {title} "
        title_len = len(strip_ansi(title_disp))
        fill_len = max(0, panel_width - 3 - title_len)
        out.append(
            f"{border_color}┌─{RESET}{BOLD}{title_disp}{RESET}{border_color}{'─' * fill_len}┐{RESET}"
        )
    else:
        out.append(f"{border_color}┌{'─' * (panel_width - 2)}┐{RESET}")

    for line in flat_lines:
        vis_len = len(strip_ansi(line))
        if vis_len <= content_width:
            pad = " " * (content_width - vis_len)
            out.append(f"{border_color}│{RESET} {line}{pad} {border_color}│{RESET}")
        else:
            out.append(f"{border_color}│{RESET} {line} {border_color}│{RESET}")

    out.append(f"{border_color}└{'─' * (panel_width - 2)}┘{RESET}")
    return "\n".join(out)

"""PhaseWise AI dashboard design system.

Presentation only. Every component renders values the service has already
established; no business rule lives here.

Principles
- One accent colour (brand green). Status colour is used only to carry meaning.
- One type scale, one radius scale, one border colour.
- Icons are inline SVG (Lucide geometry, 24px grid, 1.75 stroke). No emoji.
- Every piece of text that reaches HTML goes through escape().
"""

from __future__ import annotations

import html
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import streamlit as st

from config import config


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

# name -> (background, foreground, border)
TONES: dict[str, tuple[str, str, str]] = {
    "green": ("#EDF6F1", "#156B45", "#CFE6D9"),
    "red": ("#FCEFED", "#A12A1F", "#F3D2CD"),
    "orange": ("#FCF3E8", "#924A10", "#F1DCC2"),
    "yellow": ("#FBF6E2", "#6B5304", "#EEE2B4"),
    "blue": ("#EDF3FB", "#1F4E8C", "#D3E0F2"),
    "violet": ("#F3EFFA", "#5934A0", "#DFD5F1"),
    "gray": ("#F3F4F2", "#4A5451", "#E3E6E2"),
    "grey": ("#F3F4F2", "#4A5451", "#E3E6E2"),
    "primary": ("#E9F2EE", "#0E5E4E", "#CDE1D9"),
}

_STAT_ACCENT_TONES = {
    "pw-accent": "primary",
    "pw-accent-amber": "orange",
    "pw-accent-red": "red",
    "pw-accent-violet": "violet",
}

_CALLOUTS = {
    "info": ("gray", "info"),
    "success": ("green", "circle-check"),
    "warning": ("orange", "alert-triangle"),
    "error": ("red", "alert-circle"),
}

# Lucide icon geometry (ISC licence), 24 x 24 grid.
ICON_PATHS: dict[str, str] = {
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "x": '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
    "help": '<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/>',
    "zap": '<path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/>',
    "undo": '<path d="M9 14 4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11"/>',
    "flag": '<path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/><path d="M4 22v-7"/>',
    "alert-triangle": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/>'
                      '<path d="M12 9v4"/><path d="M12 17h.01"/>',
    "alert-circle": '<circle cx="12" cy="12" r="10"/><path d="M12 8v4"/><path d="M12 16h.01"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "circle-check": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    "inbox": '<path d="M22 12h-6l-2 3h-4l-2-3H2"/><path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 '
             '2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/>',
    "file": '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "layers": '<path d="m12 2 10 5-10 5L2 7z"/><path d="m2 17 10 5 10-5"/><path d="m2 12 10 5 10-5"/>',
    "flame": '<path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.07-2.14-.22-4.05 2-6 .5 2.5 2 4.9 4 '
             '6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.15.43-2.29 1-3a2.5 2.5 0 0 0 2.5 2.5z"/>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/>'
           '<path d="M15 2v2M9 2v2M15 20v2M9 20v2M2 15h2M2 9h2M20 15h2M20 9h2"/>',
    "hash": '<path d="M4 9h16M4 15h16M10 3 8 21M16 3l-2 18"/>',
}


_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --pw-ink: #16201D;
  --pw-ink-2: #3A4643;
  --pw-muted: #6A7471;
  --pw-faint: #9AA29F;

  --pw-canvas: #F3F6F4;
  --pw-surface: #FFFFFF;
  --pw-subtle: #F2F3F0;

  --pw-line: #E4E6E1;
  --pw-line-strong: #D2D6D0;

  --pw-brand: #0B6B57;
  --pw-brand-hover: #0B4C40;
  --pw-brand-subtle: #E9F2EE;

  --pw-radius-sm: 6px;
  --pw-radius: 10px;
  --pw-radius-lg: 12px;
  --pw-shadow: 0 1px 2px rgba(16, 24, 22, 0.04);

  --pw-font: "Inter", ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --pw-font-mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}

/* ---------- Shell ---------- */

html, body, .stApp, .stMarkdown, input, textarea, select, button {
  font-family: var(--pw-font);
  font-feature-settings: "cv11", "ss01";
  -webkit-font-smoothing: antialiased;
}
.stApp { background: var(--pw-canvas); color: var(--pw-ink); }
.block-container { max-width: 1320px; padding-top: 2.1rem; padding-bottom: 5rem; }
header[data-testid="stHeader"] { background: transparent; }
.stAppDeployButton, footer { display: none !important; }
p { color: var(--pw-ink-2); }

/* ---------- Sidebar ---------- */

section[data-testid="stSidebar"] {
  background: var(--pw-surface);
  border-right: 1px solid var(--pw-line);
}
[data-testid="stSidebarNav"] { padding-top: 12px; }
[data-testid="stSidebarNavLink"] svg { color: var(--pw-muted); }
[data-testid="stSidebarNavLink"] {
  border-radius: var(--pw-radius-sm);
  margin: 1px 10px;
  padding: 6px 10px;
}
[data-testid="stSidebarNavLink"]:hover { background: var(--pw-subtle); }
[data-testid="stSidebarNavLink"][aria-current="page"] { background: var(--pw-brand-subtle); }
[data-testid="stSidebarNavLink"][aria-current="page"] span { color: var(--pw-brand); font-weight: 600; }

/* ---------- Page header & sections ---------- */

.pw-header { padding: 4px 0 22px; margin-bottom: 24px; border-bottom: 1px solid var(--pw-line); }
.pw-eyebrow { color: var(--pw-brand); font-size: 12.5px; font-weight: 600; letter-spacing: 0.01em; margin-bottom: 6px; }
.pw-title { color: var(--pw-ink); font-size: 26px; font-weight: 650; letter-spacing: -0.022em; line-height: 1.2; }
.pw-subtitle { color: var(--pw-muted); font-size: 14.5px; line-height: 1.55; margin-top: 6px; max-width: 760px; }
.pw-runtime-status { display:flex; align-items:center; gap:7px; color:var(--pw-muted); font-size:11.5px; margin:14px 10px 4px; }
.pw-runtime-status span { width:7px; height:7px; border-radius:50%; display:inline-block; }
.pw-runtime-status small { display:block; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; opacity:.7; margin-left:auto; max-width:110px; }

.pw-section { margin: 28px 0 12px; }
.pw-section-title { color: var(--pw-ink); font-size: 15px; font-weight: 600; letter-spacing: -0.01em; }
.pw-section-caption { color: var(--pw-muted); font-size: 13px; margin-top: 2px; line-height: 1.5; }
.pw-subheading { color: var(--pw-ink-2); font-size: 13px; font-weight: 600; margin: 18px 0 8px; }

/* ---------- Text utilities ---------- */

.pw-strong { color: var(--pw-ink); font-weight: 600; }
.pw-text { color: var(--pw-ink-2); }
.pw-muted { color: var(--pw-muted); }
.pw-mono { font-family: var(--pw-font-mono); font-size: 12.5px; }
.pw-num { font-variant-numeric: tabular-nums; }
.pw-icon { flex: none; display: inline-block; vertical-align: middle; }

/* ---------- Badges ---------- */

.pw-row { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin: 4px 0 8px; }
.pw-row-meta { color: var(--pw-muted); font-size: 13px; margin-left: 4px; }
.pw-badge {
  display: inline-flex; align-items: center; gap: 6px;
  height: 22px; padding: 0 8px;
  border: 1px solid; border-radius: var(--pw-radius-sm);
  font-size: 12px; font-weight: 500; line-height: 1;
  white-space: nowrap; font-variant-numeric: tabular-nums;
}

/* ---------- Stat cards ---------- */

.pw-stat {
  display: flex; flex-direction: column;
  min-height: 116px; padding: 16px 18px;
  background: var(--pw-surface);
  border: 1px solid var(--pw-line);
  border-radius: var(--pw-radius-lg);
  box-shadow: var(--pw-shadow);
}
.pw-stat-label { display: flex; align-items: center; gap: 8px; color: var(--pw-muted); font-size: 13px; font-weight: 500; }
.pw-stat-icon { display: inline-flex; align-items: center; justify-content: center; width: 22px; height: 22px; border-radius: var(--pw-radius-sm); }
.pw-stat-value { color: var(--pw-ink); font-size: 28px; font-weight: 600; letter-spacing: -0.025em; line-height: 1.1; margin-top: 12px; font-variant-numeric: tabular-nums; }
.pw-stat-hint { color: var(--pw-muted); font-size: 12.5px; line-height: 1.45; margin-top: auto; padding-top: 10px; }

/* ---------- Key / value grid ---------- */

.pw-kv {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 14px 24px;
  margin: 4px 0 0; padding: 14px 18px;
  background: var(--pw-surface); border: 1px solid var(--pw-line); border-radius: var(--pw-radius-lg);
}
.pw-kv dt { color: var(--pw-muted); font-size: 12px; font-weight: 500; margin: 0 0 3px; }
.pw-kv dd { color: var(--pw-ink); font-size: 14px; margin: 0; line-height: 1.45; word-break: break-word; }

/* ---------- Callouts & empty states ---------- */

.pw-callout {
  display: flex; gap: 10px; align-items: flex-start;
  padding: 11px 14px; margin: 10px 0;
  border: 1px solid; border-radius: var(--pw-radius);
  font-size: 14px; line-height: 1.5;
}
.pw-callout .pw-icon { margin-top: 2px; }
.pw-callout-title { color: var(--pw-ink); font-weight: 600; margin-bottom: 1px; }
.pw-callout-body { color: var(--pw-ink-2); }

.pw-empty {
  display: flex; flex-direction: column; align-items: center; text-align: center;
  padding: 44px 24px; margin: 8px 0;
  background: var(--pw-surface); border: 1px dashed var(--pw-line-strong); border-radius: var(--pw-radius-lg);
}
.pw-empty-icon { display: inline-flex; align-items: center; justify-content: center; width: 40px; height: 40px; border-radius: var(--pw-radius); background: var(--pw-subtle); color: var(--pw-muted); }
.pw-empty-title { color: var(--pw-ink); font-size: 15px; font-weight: 600; margin-top: 12px; }
.pw-empty-body { color: var(--pw-muted); font-size: 13.5px; margin-top: 4px; max-width: 420px; line-height: 1.5; }

/* ---------- Tables ---------- */

.pw-table-wrap { width: 100%; overflow-x: auto; background: var(--pw-surface); border: 1px solid var(--pw-line); border-radius: var(--pw-radius-lg); box-shadow: var(--pw-shadow); }
table.pw-table { width: 100%; min-width: 620px; border-collapse: collapse; font-size: 14px; }
table.pw-table th {
  padding: 10px 14px; background: var(--pw-subtle);
  color: var(--pw-muted); font-size: 12.5px; font-weight: 500;
  text-align: left; white-space: nowrap; border-bottom: 1px solid var(--pw-line);
}
table.pw-table td { padding: 12px 14px; color: var(--pw-ink); line-height: 1.5; vertical-align: top; border-top: 1px solid var(--pw-line); }
table.pw-table tbody tr:first-child td { border-top: none; }
table.pw-table tbody tr:hover { background: #FAFBF9; }

/* ---------- Documents & sentences ---------- */

.pw-doc-head { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; color: var(--pw-muted); margin-bottom: 4px; }
.pw-doc-title { color: var(--pw-ink); font-size: 14.5px; font-weight: 600; }
.pw-doc-count { margin-left: auto; color: var(--pw-muted); font-size: 12.5px; }

.pw-sentence { padding: 12px 14px; margin: 8px 0; background: var(--pw-surface); border: 1px solid var(--pw-line); border-left: 3px solid var(--pw-line-strong); border-radius: var(--pw-radius); }
.pw-sentence.pw-flagged { background: #FFFAF9; border-color: #F3D2CD; border-left-color: #C0392B; }
.pw-sentence-top { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }
.pw-sentence-text { color: var(--pw-ink); font-size: 14px; line-height: 1.6; }
.pw-sentence-meta { color: var(--pw-muted); font-size: 12.5px; margin-top: 8px; line-height: 1.45; }

/* ---------- Notices ---------- */

.pw-notice {
  padding: 14px 16px; background: var(--pw-surface);
  border: 1px solid var(--pw-line); border-radius: var(--pw-radius);
  color: var(--pw-ink); font-size: 14px; line-height: 1.65; white-space: pre-wrap;
}

/* ---------- Brand ---------- */

.pw-brandmark { display: inline-flex; align-items: center; justify-content: center; background: var(--pw-brand); color: #FFFFFF; border-radius: 9px; }
.pw-signin-brand { display: flex; align-items: center; gap: 12px; margin-bottom: 28px; }
.pw-signin-name { color: var(--pw-ink); font-size: 18px; font-weight: 650; letter-spacing: -0.015em; }
.pw-signin-sub { color: var(--pw-muted); font-size: 13.5px; margin-top: 1px; }
.pw-form-title { color: var(--pw-ink); font-size: 20px; font-weight: 650; letter-spacing: -0.02em; }
.pw-form-caption { color: var(--pw-muted); font-size: 13.5px; margin: 4px 0 6px; }
.pw-footnote { color: var(--pw-faint); font-size: 12.5px; line-height: 1.5; text-align: center; margin-top: 18px; }

/* ---------- Selection bar ---------- */

.pw-selection { display: flex; align-items: center; gap: 10px; color: var(--pw-ink-2); font-size: 14px; }
.pw-selection .pw-icon { color: var(--pw-brand); }

/* ---------- Streamlit refinements ---------- */

.stTabs [data-baseweb="tab-list"] { gap: 20px; border-bottom: 1px solid var(--pw-line); }
.stTabs [data-baseweb="tab"] { padding: 10px 2px; font-size: 14px; font-weight: 500; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 18px; }

div[data-testid="stForm"] { padding: 20px; background: var(--pw-surface); border: 1px solid var(--pw-line); border-radius: var(--pw-radius-lg); box-shadow: var(--pw-shadow); }
div[data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--pw-line) !important; border-radius: var(--pw-radius-lg) !important; }
div[data-testid="stExpander"] details { background: var(--pw-surface); border: 1px solid var(--pw-line); border-radius: var(--pw-radius-lg); }
div[data-testid="stExpander"] summary { font-size: 14px; font-weight: 500; }

.stButton > button, .stFormSubmitButton > button { min-height: 38px; border-radius: 8px; font-weight: 500; font-size: 14px; box-shadow: none; }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"],
button[data-testid^="stBaseButton-primary"] {
  background: var(--pw-brand) !important; border-color: var(--pw-brand) !important; color: #FFFFFF !important;
  -webkit-text-fill-color: #FFFFFF !important;
}
.stButton > button[kind="primary"] *, .stFormSubmitButton > button[kind="primary"] *,
button[data-testid^="stBaseButton-primary"] * {
  color: #FFFFFF !important; fill: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important;
}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover,
button[data-testid^="stBaseButton-primary"]:hover {
  background: var(--pw-brand-hover) !important; border-color: var(--pw-brand-hover) !important; color: #FFFFFF !important;
  -webkit-text-fill-color: #FFFFFF !important;
}
.stButton > button[kind="primary"]:active, .stFormSubmitButton > button[kind="primary"]:active,
.stButton > button[kind="primary"]:focus, .stFormSubmitButton > button[kind="primary"]:focus,
button[data-testid^="stBaseButton-primary"]:active, button[data-testid^="stBaseButton-primary"]:focus {
  background: var(--pw-brand-hover) !important; color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF !important;
}
.stButton > button[kind="primary"]:hover *, .stFormSubmitButton > button[kind="primary"]:hover * {
  color: #FFFFFF !important;
}
.stButton > button[kind="secondary"], .stFormSubmitButton > button[kind="secondary"] {
  background: #FFFFFF !important; border: 1px solid var(--pw-line-strong) !important;
  color: var(--pw-ink) !important;
}
.stButton > button[kind="secondary"] *, .stFormSubmitButton > button[kind="secondary"] * { color:var(--pw-ink) !important; }
.stButton > button:disabled, .stFormSubmitButton > button:disabled { opacity:.55; cursor:not-allowed; }

[data-testid="stWidgetLabel"] p { color: var(--pw-ink-2); font-size: 13px; font-weight: 500; }
div[data-testid="stChatMessage"] { background: var(--pw-surface); border: 1px solid var(--pw-line); border-radius: var(--pw-radius-lg); padding: 14px 16px; }
[data-testid="stDataFrame"] { border: 1px solid var(--pw-line); border-radius: var(--pw-radius-lg); overflow: hidden; }
.stSegmentedControl { background: var(--pw-surface); border-radius: var(--pw-radius); padding: 3px; }
.stSegmentedControl [aria-checked="true"] { background: var(--pw-brand-subtle); color: var(--pw-brand); }
.stProgress > div > div > div > div { background: var(--pw-brand); }

button:focus-visible, input:focus-visible, textarea:focus-visible, select:focus-visible, a:focus-visible {
  outline: 2px solid var(--pw-brand) !important; outline-offset: 2px !important;
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { transition-duration: 0.01ms !important; animation-duration: 0.01ms !important; }
}

@media (max-width: 768px) {
  .block-container { padding: 1.25rem 1rem 3rem; }
  .pw-title { font-size: 22px; }
  .pw-stat-value { font-size: 24px; }
  .pw-doc-count { margin-left: 0; }
}
.st-key-pw_signin_shell {
  max-width: 500px;
  margin: 0 auto;
  padding: 34px 38px 28px;
  background: var(--pw-surface);
  border: 1px solid var(--pw-line);
  border-radius: 18px;
  box-shadow: 0 18px 48px rgba(22, 32, 29, .08);
}
.stHorizontalBlock:has(.st-key-pw_signin_shell) { justify-content:center; }
.stHorizontalBlock:has(.st-key-pw_signin_shell) > .stColumn:not(:has(.st-key-pw_signin_shell)) { display:none; }
.stHorizontalBlock:has(.st-key-pw_signin_shell) > .stColumn:has(.st-key-pw_signin_shell) {
  flex:0 1 500px !important;
  width:min(500px, 100%) !important;
}
.pw-signin-brand { display:flex; align-items:center; gap:13px; margin-bottom:32px; }
.pw-brandmark { width:42px; height:42px; border-radius:12px; }
.pw-signin-name { font-size:17px; font-weight:700; letter-spacing:-.025em; }
.pw-signin-sub { font-size:12.5px; margin-top:3px; }
.pw-signin-intro { margin-bottom:20px; }
.pw-form-title { font-size:25px; font-weight:700; letter-spacing:-.035em; line-height:1.2; }
.pw-form-caption { font-size:14px; line-height:1.5; margin-top:7px; }
.st-key-pw_signin_shell div[data-testid="stForm"] {
  padding: 0;
  background: transparent;
  border: 0;
  border-radius: 0;
  box-shadow: none;
}
.st-key-pw_signin_shell [data-testid="stWidgetLabel"] p { color:var(--pw-ink-2); font-size:13px; font-weight:600; }
.st-key-pw_signin_shell .stTextInput input {
  min-height:48px;
  border-radius:9px;
  background:#F7F9F8;
  border-color:#DCE4E0;
  color:var(--pw-ink);
}
.st-key-pw_signin_shell .stTextInput input::placeholder { color:#8A9591; opacity:1; }
.st-key-pw_signin_shell .stTextInput input:focus {
  background:#FFFFFF;
  border-color:var(--pw-brand);
  box-shadow:0 0 0 3px rgba(11,107,87,.12);
}
.st-key-pw_signin_shell [data-testid="stFormSubmitButton"] button {
  min-height:48px;
  margin-top:7px;
  border-radius:9px;
  background:#0B6B57 !important;
  border:1px solid #0B6B57 !important;
  color:#FFFFFF !important;
  -webkit-text-fill-color:#FFFFFF !important;
  font-size:14px;
  font-weight:650;
  letter-spacing:.005em;
  transition:background .16s ease, border-color .16s ease, transform .16s ease;
}
.st-key-pw_signin_shell [data-testid="stFormSubmitButton"] button * {
  color:#FFFFFF !important;
  -webkit-text-fill-color:#FFFFFF !important;
}
.st-key-pw_signin_shell [data-testid="stFormSubmitButton"] button:hover {
  background:#084F43 !important;
  border-color:#084F43 !important;
  transform:translateY(-1px);
}
.st-key-pw_signin_shell [data-testid="stFormSubmitButton"] button:active { transform:translateY(0); }
.pw-footnote { display:flex; align-items:flex-start; gap:10px; color:var(--pw-muted); font-size:12.5px;
  line-height:1.55; text-align:left; margin-top:22px; padding-top:18px; border-top:1px solid var(--pw-line); }
.pw-footnote-mark { display:inline-flex; flex:none; align-items:center; justify-content:center; width:20px; height:20px;
  border-radius:50%; background:#EAF4EF; color:var(--pw-brand); font-size:12px; font-weight:700; }
@media (max-width: 640px) {
  .st-key-pw_signin_shell { padding:28px 24px 22px; border-radius:15px; }
  .pw-signin-brand { margin-bottom:26px; }
  .pw-form-title { font-size:23px; }
}
</style>
"""


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def inject_theme() -> None:
    """Apply the shared theme to the current page."""
    st.markdown(_THEME_CSS, unsafe_allow_html=True)


def escape(value: Any) -> str:
    """Return a value escaped for safe inclusion in generated HTML."""
    return html.escape("" if value is None else str(value), quote=True)


def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def _tone(name: str) -> tuple[str, str, str]:
    return TONES.get(name, TONES["gray"])


def icon(name: str, size: int = 16, stroke: float = 1.75) -> str:
    """Return inline SVG for a named icon, or an empty string if unknown."""
    paths = ICON_PATHS.get(name)
    if not paths:
        return ""
    return (f'<svg class="pw-icon" xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
            f'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="{stroke}" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{paths}</svg>')


svg_icon = icon  # alias for scopes where a parameter is named `icon`


def brand_mark(size: int = 32) -> str:
    """Return the PhaseWise logomark."""
    return (f'<span class="pw-brandmark" style="width:{size}px;height:{size}px">'
            f'{icon("activity", round(size * 0.55), 2)}</span>')


def spacer(height: int = 12) -> None:
    """Insert vertical space."""
    _html(f'<div aria-hidden="true" style="height:{max(0, min(int(height), 120))}px"></div>')


# ---------------------------------------------------------------------------
# Badges
# ---------------------------------------------------------------------------

def badge_html(text: Any, colour: str = config.NEUTRAL_BADGE_COLOUR, icon_name: str | None = None) -> str:
    """Return HTML for one badge."""
    background, foreground, border = _tone(colour)
    return (f'<span class="pw-badge" style="background:{background};color:{foreground};border-color:{border}">'
            f'{icon(icon_name, 12, 2) if icon_name else ""}{escape(text)}</span>')


def badge_row(badges: Iterable[tuple[Any, str]], meta: Any | None = None) -> None:
    """Render a row of badges with optional muted metadata."""
    markup = "".join(badge_html(text, colour) for text, colour in badges)
    meta_markup = (f'<span class="pw-row-meta">{escape(meta)}</span>'
                   if meta is not None and str(meta).strip() else "")
    _html(f'<div class="pw-row">{markup}{meta_markup}</div>')


# ---------------------------------------------------------------------------
# Page structure
# ---------------------------------------------------------------------------

def page_header(title: Any, subtitle: Any | None = None, eyebrow: Any = config.DASHBOARD_TITLE) -> None:
    """Render the page title block."""
    subtitle_markup = (f'<div class="pw-subtitle">{escape(subtitle)}</div>'
                       if subtitle is not None and str(subtitle).strip() else "")
    _html(f'<div class="pw-header"><div class="pw-eyebrow">{escape(eyebrow)}</div>'
          f'<div class="pw-title" role="heading" aria-level="1">{escape(title)}</div>{subtitle_markup}</div>')


def section(title: Any, caption: Any | None = None) -> None:
    """Render a section heading with an optional one-line caption."""
    caption_markup = f'<div class="pw-section-caption">{escape(caption)}</div>' if caption else ""
    _html(f'<div class="pw-section"><div class="pw-section-title" role="heading" aria-level="2">'
          f'{escape(title)}</div>{caption_markup}</div>')


def subheading(title: Any) -> None:
    """Render a small heading inside a section or expander."""
    _html(f'<div class="pw-subheading" role="heading" aria-level="3">{escape(title)}</div>')


def key_values(items: Sequence[tuple[Any, str]]) -> None:
    """Render a grid of labelled values. Values are pre-built safe HTML."""
    cells = "".join(f"<div><dt>{escape(label)}</dt><dd>{value}</dd></div>" for label, value in items)
    _html(f'<dl class="pw-kv">{cells}</dl>')


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

def callout(body: Any, tone: str = "info", title: Any | None = None, body_is_html: bool = False) -> None:
    """Render an inline message. tone is info, success, warning or error."""
    colour, icon_name = _CALLOUTS.get(tone, _CALLOUTS["info"])
    background, foreground, border = _tone(colour)
    title_markup = f'<div class="pw-callout-title">{escape(title)}</div>' if title else ""
    body_markup = body if body_is_html else escape(body)
    _html(f'<div class="pw-callout" role="status" style="background:{background};border-color:{border}">'
          f'<span style="color:{foreground}">{icon(icon_name, 16, 2)}</span>'
          f'<div>{title_markup}<div class="pw-callout-body">{body_markup}</div></div></div>')


def empty_state(title: Any, body: Any | None = None, icon_name: str = "inbox") -> None:
    """Render a calm placeholder for an empty list."""
    body_markup = f'<div class="pw-empty-body">{escape(body)}</div>' if body else ""
    _html(f'<div class="pw-empty"><span class="pw-empty-icon">{icon(icon_name, 20)}</span>'
          f'<div class="pw-empty-title">{escape(title)}</div>{body_markup}</div>')


# ---------------------------------------------------------------------------
# Metric cards
# ---------------------------------------------------------------------------

def stat_card(label: Any, value: Any, hint_html: str = "", accent: str = "pw-accent",
              icon: str = "", tone: str | None = None) -> None:
    """Render one metric card. `icon` is an icon name from ICON_PATHS."""
    background, foreground, _ = _tone(tone or _STAT_ACCENT_TONES.get(accent, "primary"))
    icon_markup = (f'<span class="pw-stat-icon" style="background:{background};color:{foreground}">'
                   f'{svg_icon(icon, 14, 2)}</span>' if icon in ICON_PATHS else "")
    hint_markup = f'<div class="pw-stat-hint">{hint_html}</div>' if hint_html else ""
    _html(f'<div class="pw-stat"><div class="pw-stat-label">{icon_markup}{escape(label)}</div>'
          f'<div class="pw-stat-value">{escape(value)}</div>{hint_markup}</div>')


def stat_row(cards: Sequence[Mapping[str, Any]], max_per_row: int = 5) -> None:
    """Render metric cards in balanced rows (6 cards -> 3 + 3, never 4 + 2)."""
    if not cards:
        return
    rows = math.ceil(len(cards) / max_per_row)
    per_row = math.ceil(len(cards) / rows)
    for start in range(0, len(cards), per_row):
        group = cards[start:start + per_row]
        for column, card in zip(st.columns(per_row), group):
            with column:
                stat_card(**dict(card))
        if start + per_row < len(cards):
            spacer(4)


# ---------------------------------------------------------------------------
# Tables, sentences, notices
# ---------------------------------------------------------------------------

def html_table(headers: Sequence[Any], rows: Sequence[Sequence[str]]) -> None:
    """Render a styled table. Headers are escaped; cells are pre-built safe HTML."""
    if not headers:
        return
    head = "".join(f'<th scope="col">{escape(header)}</th>' for header in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    _html(f'<div class="pw-table-wrap"><table class="pw-table"><thead><tr>{head}</tr></thead>'
          f'<tbody>{body}</tbody></table></div>')


def muted(text: Any) -> str:
    """Return muted inline text."""
    return f'<span class="pw-muted">{escape(text)}</span>'


def strong(text: Any) -> str:
    """Return emphasised inline text."""
    return f'<span class="pw-strong">{escape(text)}</span>'


def mono(text: Any) -> str:
    """Return monospaced inline text, for identifiers."""
    return f'<span class="pw-mono">{escape(text)}</span>'


def sentence_card(text: Any, badges_html: str, meta_html: str = "", flagged: bool = False) -> None:
    """Render one quoted sentence from a statement, with its classification and notes."""
    meta = f'<div class="pw-sentence-meta">{meta_html}</div>' if meta_html else ""
    _html(f'<div class="pw-sentence{" pw-flagged" if flagged else ""}">'
          f'<div class="pw-sentence-top">{badges_html}</div>'
          f'<div class="pw-sentence-text">{escape(text)}</div>{meta}</div>')


def notice_box(text: Any) -> None:
    """Render service-produced text verbatim."""
    _html(f'<div class="pw-notice">{escape(text)}</div>')


# ---------------------------------------------------------------------------
# Presentation mappings
# ---------------------------------------------------------------------------

def decision_colour(decision: str) -> str:
    return config.DECISION_BADGE_COLOURS.get(decision, config.NEUTRAL_BADGE_COLOUR)


def status_colour(status: str) -> str:
    return config.STATUS_BADGE_COLOURS.get(status, config.NEUTRAL_BADGE_COLOUR)


def band_colour(band: str) -> str:
    return config.LIKELIHOOD_BAND_BADGE_COLOURS.get(band, config.NEUTRAL_BADGE_COLOUR)


def result_colour(result: str) -> str:
    return config.REQUIREMENT_RESULT_BADGE_COLOURS.get(result, config.NEUTRAL_BADGE_COLOUR)


def chat_type_colour(question_type: str) -> str:
    return config.CHAT_TYPE_BADGE_COLOURS.get(question_type, config.NEUTRAL_BADGE_COLOUR)


def answer_label(column: str) -> str:
    return config.FORM_ANSWER_LABELS.get(column, column.replace("_", " "))


def format_amps(values: Iterable[float]) -> str:
    """Join current values to the configured precision."""
    return ", ".join(f"{amps:.{config.AMPS_DECIMALS}f}" for amps in values)


def delay_badge(duration: Mapping[str, Any]) -> str:
    """Render the delay status the service established."""
    delay_expected = duration.get("delay_expected")
    planned = duration.get("planned_working_days")
    if delay_expected is None:
        return badge_html("No plan given", config.NEUTRAL_BADGE_COLOUR)
    if delay_expected:
        return badge_html(f"Delay expected · {planned} planned", config.DELAY_COLOUR)
    return badge_html(f"On time · {planned} planned", config.NEUTRAL_BADGE_COLOUR)

#!/usr/bin/env python3
"""
US Congress Bill Monitor
Watches House and Senate bills via Congress.gov API and emails a formatted
summary when new bills are detected.  Uses Claude Sonnet 4.6 to generate
plain-English bill descriptions.

Usage:
  python monitor.py              # Start daemon (checks every 24 h)
  python monitor.py --show       # One-time snapshot, no daemon
  python monitor.py --test-email # Send test email and exit
  python monitor.py --reset      # Clear seen-bills history and exit
"""

import argparse
import json
import logging
import os
import re
import smtplib
import sys
import time
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# Windows UTF-8 fix — must happen BEFORE Rich Console is created so that
# emoji in status labels don't crash the legacy cp125x encoder when stdout
# is redirected to a log file.
# ---------------------------------------------------------------------------
import io as _io
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import requests
import schedule
from rich.console import Console
from rich.table import Table
from rich import box

# Anthropic is optional — only needed for bill descriptions
try:
    import anthropic as _anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR        = Path(__file__).parent
CONFIG_PATH     = BASE_DIR / "config.json"
DATA_DIR        = BASE_DIR / "data"
SEEN_BILLS_PATH = DATA_DIR / "seen_bills.json"

# ---------------------------------------------------------------------------
# Logging + console
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log     = logging.getLogger("congress_monitor")
console = Console()

# ---------------------------------------------------------------------------
# Default config  (auto-written to config.json on first run)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: dict = {
    "congress_api_key":       "DEMO_KEY",   # free key → api.congress.gov/sign-up
    "anthropic_api_key":      "",           # or set ANTHROPIC_API_KEY env var
    "check_interval_minutes": 1440,         # 24 hours
    "bills_per_check":        50,
    "congress":               119,          # 119th Congress 2025-2027
    "track_recent_days":      3,
    "detail_fetch_limit":     100,          # max per-bill Congress.gov calls per run
    "keywords":               [],
    "email": {
        "enabled":   False,
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
        "username":  "",
        "password":  "",
        "from":      "",
        "to":        "",
    },
}

# ---------------------------------------------------------------------------
# Status detection  (from latestAction.text)
# ---------------------------------------------------------------------------
STATUS_RULES = [
    (["became public law", "enacted into law"],            "✅ Became Law"),
    (["signed by the president", "signed by president"],   "✅ Signed by President"),
    (["vetoed"],                                           "❌ Vetoed"),
    (["passed senate", "agreed to in senate"],             "🔵 Passed Senate"),
    (["passed house", "agreed to in house"],               "🔵 Passed House"),
    (["reported"],                                         "🟡 Reported by Committee"),
    (["referred to"],                                      "🟡 In Committee"),
]
STATUS_DEFAULT = "⬜ Introduced"


def detect_status(action_text: str) -> str:
    if not action_text:
        return STATUS_DEFAULT
    lower = action_text.lower()
    for keywords, label in STATUS_RULES:
        if any(kw in lower for kw in keywords):
            return label
    return STATUS_DEFAULT


# ---------------------------------------------------------------------------
# Stage tracker helpers
# ---------------------------------------------------------------------------

def _stage_labels(origin: str) -> List[str]:
    other = "Senate" if origin == "House" else "House"
    return [
        "Introduced",
        "In Committee",
        "Reported",
        f"Passed {origin}",
        f"Passed {other}",
        "To President",
        "Enacted",
    ]


def derive_stages(bill: dict) -> List[Dict]:
    """Infer completed stages from the bill's current status (no extra API calls)."""
    status = bill.get("status", STATUS_DEFAULT)
    origin = bill.get("originChamber", "House")
    other  = "Senate" if origin == "House" else "House"
    labels = _stage_labels(origin)

    if   "Became Law"          in status or "Enacted" in status: n = 7
    elif "Signed by President" in status:                         n = 6
    elif "Vetoed"              in status:                         n = 6
    elif f"Passed {other}"     in status:                         n = 5
    elif f"Passed {origin}"    in status:                         n = 4
    elif "Reported"            in status:                         n = 3
    elif "In Committee"        in status:                         n = 2
    else:                                                         n = 1

    return [{"stage": labels[i], "done": i < n, "date": ""} for i in range(7)]


def build_stage_tracker(actions: list, bill: dict) -> List[Dict]:
    """Build stage tracker from the full /actions list for a bill."""
    origin = bill.get("originChamber", "House")
    other  = "Senate" if origin == "House" else "House"
    labels = _stage_labels(origin)

    completed: Dict[str, str] = {}
    completed["Introduced"] = bill.get("introducedDate", "")

    for action in actions:
        text  = action.get("text", "").lower()
        date  = action.get("actionDate", "")
        atype = action.get("type", "")

        if atype == "IntroReferral" or "referred to" in text:
            completed.setdefault("In Committee", date)

        if "reported" in text:
            completed.setdefault("Reported", date)

        for phrase in ["passed house", "agreed to in the house",
                       "passed by the house", "the house passed"]:
            if phrase in text:
                key = f"Passed {origin}" if origin == "House" else f"Passed {other}"
                completed.setdefault(key, date)
                break

        for phrase in ["passed senate", "agreed to in the senate",
                       "passed by the senate", "the senate passed"]:
            if phrase in text:
                key = f"Passed {origin}" if origin == "Senate" else f"Passed {other}"
                completed.setdefault(key, date)
                break

        for phrase in ["presented to the president", "submitted to the president",
                       "sent to the president"]:
            if phrase in text:
                completed.setdefault("To President", date)
                break

        for phrase in ["signed by the president", "became public law", "enacted into law"]:
            if phrase in text:
                completed.setdefault("Enacted", date)
                break

    return [
        {"stage": labels[i], "done": labels[i] in completed,
         "date": completed.get(labels[i], "")}
        for i in range(len(labels))
    ]


def _stage_compact(tracker: List[Dict]) -> str:
    """7-char dot string for the terminal table: ●●●○○○○"""
    if not tracker:
        return "○○○○○○○"
    return "".join("●" if s["done"] else "○" for s in tracker)


def _stage_abbrev(label: str) -> str:
    mapping = {
        "Introduced":   "Intro",
        "In Committee": "Comm",
        "Reported":     "Rptd",
        "To President": "Pres",
        "Enacted":      "Law",
    }
    if label in mapping:
        return mapping[label]
    if label.startswith("Passed "):
        return "P." + label[7:]   # "P.House" / "P.Senate"
    return label[:6]


def _stage_html(tracker: List[Dict]) -> str:
    """Render stage tracker as inline HTML pill-chain for the email."""
    if not tracker:
        return ""
    parts = []
    for i, s in enumerate(tracker):
        display    = _stage_abbrev(s["stage"])
        bg         = "#4caf50" if s["done"] else "#e0e0e0"
        fg         = "white"   if s["done"] else "#999"
        title_attr = f' title="{s["date"]}"' if s.get("date") else ""
        parts.append(
            f'<span{title_attr} style="background:{bg};color:{fg};border-radius:3px;'
            f'padding:2px 6px;font-size:10px;white-space:nowrap;">{display}</span>'
        )
        if i < len(tracker) - 1:
            arrow_color = "#4caf50" if s["done"] else "#ccc"
            parts.append(
                f'<span style="color:{arrow_color};margin:0 1px;font-size:10px;">→</span>'
            )
    return (
        '<div style="display:flex;align-items:center;flex-wrap:wrap;gap:1px;line-height:1.8;">'
        + "".join(parts) + "</div>"
    )


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    if not CONFIG_PATH.exists():
        log.info("config.json not found — creating defaults.")
        save_config(DEFAULT_CONFIG)
        return dict(DEFAULT_CONFIG)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        user_cfg = json.load(f)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(user_cfg)
    cfg["email"] = {**DEFAULT_CONFIG["email"], **user_cfg.get("email", {})}
    return cfg


def save_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


# ---------------------------------------------------------------------------
# Seen-bills persistence
# ---------------------------------------------------------------------------

def load_seen_bills() -> Set[str]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not SEEN_BILLS_PATH.exists():
        return set()
    with open(SEEN_BILLS_PATH, "r", encoding="utf-8") as f:
        return set(json.load(f))


def save_seen_bills(seen: Set[str]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SEEN_BILLS_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(seen), f, indent=2)


def reset_seen_bills() -> None:
    if SEEN_BILLS_PATH.exists():
        SEEN_BILLS_PATH.unlink()
    log.info("Seen-bills history cleared.")


# ---------------------------------------------------------------------------
# Congress.gov API
# ---------------------------------------------------------------------------
API_BASE   = "https://api.congress.gov/v3"
BILL_TYPES = ["hr", "s", "hjres", "sjres", "hres", "sres"]


def _bill_url(congress: int, bill_type: str, number) -> str:
    mapping = {
        "hr":    "house-bill",
        "s":     "senate-bill",
        "hjres": "house-joint-resolution",
        "sjres": "senate-joint-resolution",
        "hres":  "house-resolution",
        "sres":  "senate-resolution",
    }
    path = mapping.get(bill_type.lower(), "house-bill")
    return f"https://congress.gov/bill/{congress}th-congress/{path}/{number}"


def fetch_bills(cfg: dict, since_date: Optional[datetime] = None) -> List[dict]:
    """Fetch the recent bill list (6 API calls — no sponsors at this stage)."""
    bills: List[dict] = []
    for bill_type in BILL_TYPES:
        url    = f"{API_BASE}/bill/{cfg['congress']}/{bill_type}"
        params = {
            "api_key": cfg["congress_api_key"],
            "sort":    "updateDate desc",
            "limit":   cfg["bills_per_check"],
            "format":  "json",
        }
        data = _fetch_json(url, params)
        if not data:
            log.warning("No data returned for %s bill list.", bill_type.upper())
            time.sleep(1)
            continue

        for raw in data.get("bills", []):
            bill = _parse_bill_from_list(raw, cfg["congress"], bill_type)
            if since_date:
                dt = _parse_date(bill.get("updateDate", ""))
                if dt and dt < since_date:
                    continue
            bills.append(bill)

        time.sleep(1)   # 1-second pause between bill-type calls → stays well under rate limits

    return bills


def _parse_bill_from_list(raw: dict, congress: int, bill_type: str) -> dict:
    latest = raw.get("latestAction", {})
    number = raw.get("number", "")
    return {
        "id":              f"{bill_type.upper()}{number}-{congress}",
        "number":          number,
        "type":            bill_type.upper(),
        "title":           raw.get("title", ""),
        "originChamber":   raw.get("originChamber", ""),
        "sponsor":         "",   # populated by enrich_bills()
        "party":           "",
        "state":           "",
        "status":          detect_status(latest.get("text", "")),
        "latestActionText": latest.get("text", ""),
        "actionDate":      latest.get("actionDate", ""),
        "introducedDate":  raw.get("introducedDate", ""),
        "updateDate":      raw.get("updateDate", ""),
        "url":             _bill_url(congress, bill_type, number),
        "description":     "",   # populated by generate_descriptions()
        "stageTracker":    [],   # populated by enrich_bills()
    }


def _parse_date(date_str: str) -> Optional[datetime]:
    if not date_str:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _fetch_json(url: str, params: dict, retries: int = 3) -> dict:
    """GET with exponential back-off on 429 / 5xx."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 429:
                wait = 60 * (attempt + 1)
                log.warning("Rate-limited (429) — waiting %d s before retry %d/%d…",
                            wait, attempt + 1, retries)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            log.debug("Request failed %s (attempt %d): %s", url, attempt + 1, exc)
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
    return {}


def _strip_html(text: str, max_chars: int = 250) -> str:
    """Strip HTML tags, decode common entities, and truncate."""
    text = re.sub(r"<[^>]+>", " ", text)
    for entity, char in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                          ("&nbsp;", " "), ("&#39;", "'"), ("&quot;", '"')]:
        text = text.replace(entity, char)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
    return text


def enrich_bills(cfg: dict, bills: List[dict]) -> None:
    """
    For each bill fetch:
      • /bill/{congress}/{type}/{number}          → sponsor name, party, state
      • /bill/{congress}/{type}/{number}/actions  → full stage tracker
      • /bill/{congress}/{type}/{number}/summaries → CRS plain-English description

    Skipped for DEMO_KEY (rate limit: 30 req/hour) — stages derived from status instead.
    """
    if cfg["congress_api_key"] == "DEMO_KEY":
        log.info(
            "DEMO_KEY detected — per-bill detail fetch skipped "
            "(stages inferred from status). Register a free key at "
            "api.congress.gov/sign-up to enable real sponsor data."
        )
        for bill in bills:
            bill["stageTracker"] = derive_stages(bill)
        return

    limit    = min(int(cfg.get("detail_fetch_limit", 100)), len(bills))
    to_fetch = bills[:limit]
    rest     = bills[limit:]

    log.info(
        "Enriching %d bill(s) with sponsor + action history (Congress.gov)…",
        len(to_fetch),
    )
    base_params = {"api_key": cfg["congress_api_key"], "format": "json"}

    for i, bill in enumerate(to_fetch):
        btype  = bill["type"].lower()
        number = bill["number"]
        cong   = cfg["congress"]

        # ── Sponsor ────────────────────────────────────────────────────
        detail = _fetch_json(
            f"{API_BASE}/bill/{cong}/{btype}/{number}", base_params
        ).get("bill", {})
        sponsors = detail.get("sponsors", [])
        if sponsors:
            sp = sponsors[0]
            bill["sponsor"] = sp.get("fullName", sp.get("name", ""))
            bill["party"]   = sp.get("party", "")
            bill["state"]   = sp.get("state", "")
        time.sleep(0.15)

        # ── Actions / stage tracker ────────────────────────────────────
        actions_data = _fetch_json(
            f"{API_BASE}/bill/{cong}/{btype}/{number}/actions",
            {**base_params, "limit": 250},
        )
        actions = actions_data.get("actions", [])
        bill["stageTracker"] = (
            build_stage_tracker(actions, bill) if actions else derive_stages(bill)
        )
        time.sleep(0.15)

        # ── CRS Summary / description ──────────────────────────────────
        summ_data = _fetch_json(
            f"{API_BASE}/bill/{cong}/{btype}/{number}/summaries", base_params
        )
        summaries = summ_data.get("summaries", [])
        if summaries:
            raw = summaries[-1].get("text", "")
            if raw:
                bill["description"] = _strip_html(raw)
        time.sleep(0.15)

        if (i + 1) % 20 == 0:
            log.info("  … enriched %d / %d", i + 1, len(to_fetch))

    for bill in rest:
        bill["stageTracker"] = derive_stages(bill)


# ---------------------------------------------------------------------------
# Claude Sonnet 4.6 — bill descriptions
# ---------------------------------------------------------------------------

def generate_descriptions(bills: List[dict], cfg: dict) -> Dict[str, str]:
    """
    Batch-generate 1-sentence plain-English descriptions via Claude Sonnet 4.6.
    Uses ANTHROPIC_API_KEY env var or 'anthropic_api_key' from config.
    Returns {bill_id: description}.
    """
    if not _ANTHROPIC_AVAILABLE:
        log.info("anthropic package not installed — skipping descriptions (pip install anthropic).")
        return {}

    api_key = cfg.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        log.info(
            "No Anthropic API key found — skipping descriptions. "
            "Set 'anthropic_api_key' in config.json or export ANTHROPIC_API_KEY."
        )
        return {}

    client       = _anthropic.Anthropic(api_key=api_key)
    descriptions: Dict[str, str] = {}
    BATCH        = 50

    for start in range(0, len(bills), BATCH):
        batch = bills[start : start + BATCH]
        lines = "\n".join(
            f"[{b['id']}] {b['type']} {b['number']}: {b['title']}"
            for b in batch
        )
        prompt = (
            "You are a nonpartisan congressional analyst. For each US Congress bill "
            "listed below, write exactly one plain-English sentence (max 25 words) "
            "describing what the bill would do. Be specific and factual.\n\n"
            f"Bills:\n{lines}\n\n"
            "Respond ONLY with a valid JSON object mapping each bill ID to its summary. "
            'Example: {"HR1-119": "Requires federal agencies to reduce paper waste by 50%."}'
        )
        try:
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip()
            # Strip optional markdown fences
            m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw)
            if m:
                raw = m.group(1)
            elif not raw.startswith("{"):
                s, e = raw.find("{"), raw.rfind("}") + 1
                if 0 <= s < e:
                    raw = raw[s:e]
            descriptions.update(json.loads(raw))
            log.info("Claude descriptions generated for %d bills.", len(batch))
        except Exception as exc:
            log.warning("Claude description batch %d failed: %s", start // BATCH + 1, exc)

    return descriptions


# ---------------------------------------------------------------------------
# Keyword filtering
# ---------------------------------------------------------------------------

def matches_keywords(bill: dict, keywords: List[str]) -> bool:
    if not keywords:
        return True
    title  = bill.get("title", "").lower()
    action = bill.get("latestActionText", "").lower()
    return any(kw.lower() in title or kw.lower() in action for kw in keywords)


# ---------------------------------------------------------------------------
# Rich terminal table
# ---------------------------------------------------------------------------

def display_table(bills: List[dict], title: str = "Congress Bills") -> None:
    if not bills:
        console.print("[dim]No bills to display.[/dim]")
        return

    table = Table(
        title=title, box=box.ROUNDED, show_lines=True,
        header_style="bold cyan", expand=False,
    )
    table.add_column("#",           style="dim", width=3,  justify="right")
    table.add_column("Bill",        style="bold", width=9)
    table.add_column("Title",                    width=34, overflow="fold")
    table.add_column("Description",              width=30, overflow="fold")
    table.add_column("Sponsor",                  width=26, overflow="fold")
    table.add_column("Status",                   width=24)
    table.add_column("Stages",                   width=9,  justify="center")
    table.add_column("Date",                     width=12)
    table.add_column("Link",                     width=6)

    for i, bill in enumerate(bills, 1):
        sponsor = bill.get("sponsor", "")
        party   = bill.get("party",   "")
        state   = bill.get("state",   "")

        if sponsor:
            suffix       = "-".join(x for x in [party, state] if x)
            sponsor_cell = f"{sponsor} ({suffix})" if suffix else sponsor
            if   party == "R": sponsor_cell = f"[red]{sponsor_cell}[/red]"
            elif party == "D": sponsor_cell = f"[blue]{sponsor_cell}[/blue]"
        else:
            sponsor_cell = "[dim]—[/dim]"

        desc   = bill.get("description", "") or "[dim]—[/dim]"
        stages = _stage_compact(bill.get("stageTracker", []))
        date   = bill.get("actionDate") or bill.get("introducedDate", "")
        url    = bill.get("url", "")
        link   = f"[link={url}]View[/link]" if url else ""

        table.add_row(
            str(i),
            f"{bill['type']} {bill['number']}",
            bill.get("title", ""),
            desc,
            sponsor_cell,
            bill.get("status", STATUS_DEFAULT),
            stages,
            date,
            link,
        )

    console.print(table)


# ---------------------------------------------------------------------------
# HTML email
# ---------------------------------------------------------------------------

def build_html_email(
    bills: List[dict], date_str: str, headline: str = "New Bills Detected"
) -> str:
    rows = ""
    for bill in bills:
        url   = bill.get("url", "#")
        date  = bill.get("actionDate") or bill.get("introducedDate", "")
        bid   = f"{bill['type']} {bill['number']}"

        sponsor = bill.get("sponsor", "")
        party   = bill.get("party",   "")
        state   = bill.get("state",   "")
        if sponsor:
            suffix  = "-".join(x for x in [party, state] if x)
            sp_html = f"{sponsor} ({suffix})" if suffix else sponsor
        else:
            sp_html = "<em style='color:#aaa'>—</em>"

        desc       = bill.get("description", "") or "<em style='color:#aaa'>—</em>"
        stage_html = _stage_html(bill.get("stageTracker", []))

        rows += f"""
        <tr style="vertical-align:top;">
          <td style="padding:8px 10px;border-bottom:1px solid #f0f0f0;white-space:nowrap;">
            <a href="{url}" style="color:#1a73e8;text-decoration:none;font-weight:bold;">{bid}</a>
          </td>
          <td style="padding:8px 10px;border-bottom:1px solid #f0f0f0;">
            <div style="font-weight:bold;margin-bottom:3px;">{bill.get('title','')}</div>
            <div style="color:#555;font-size:12px;font-style:italic;">{desc}</div>
          </td>
          <td style="padding:8px 10px;border-bottom:1px solid #f0f0f0;font-size:12px;">{sp_html}</td>
          <td style="padding:8px 10px;border-bottom:1px solid #f0f0f0;font-size:12px;">{bill.get('status', STATUS_DEFAULT)}</td>
          <td style="padding:8px 10px;border-bottom:1px solid #f0f0f0;">{stage_html}</td>
          <td style="padding:8px 10px;border-bottom:1px solid #f0f0f0;font-size:12px;white-space:nowrap;">{date}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  body  {{ font-family:Arial,sans-serif;color:#333;max-width:1100px;margin:0 auto;padding:20px; }}
  h2    {{ color:#1a73e8;border-bottom:2px solid #1a73e8;padding-bottom:8px; }}
  table {{ width:100%;border-collapse:collapse;font-size:13px; }}
  th    {{ background:#1a73e8;color:white;padding:8px 10px;text-align:left; }}
  tr:nth-child(even) {{ background:#f8f9fa; }}
</style></head>
<body>
  <h2>Congress Monitor — {headline}</h2>
  <p style="color:#555;margin-bottom:16px;">
    <strong>{len(bills)}</strong> bill(s) as of <strong>{date_str}</strong> |
    Source: <a href="https://congress.gov" style="color:#1a73e8;">Congress.gov</a> |
    Descriptions: Claude Sonnet&nbsp;4.6
  </p>
  <table>
    <thead><tr>
      <th style="width:70px;">Bill</th>
      <th>Title &amp; Description</th>
      <th style="width:170px;">Sponsor</th>
      <th style="width:155px;">Status</th>
      <th style="width:295px;">Stage Tracker</th>
      <th style="width:90px;">Date</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <p style="margin-top:20px;font-size:11px;color:#aaa;">
    Congress Monitor — 119th Congress (2025–2027) — running on your machine
  </p>
</body></html>"""


def send_email(
    cfg: dict,
    bills: List[dict],
    subject: Optional[str] = None,
    headline: str = "New Bills Detected",
) -> bool:
    ecfg = cfg.get("email", {})
    if not ecfg.get("enabled", False):
        log.info("Email not enabled — skipping.")
        return False

    missing = [k for k in ["smtp_host", "smtp_port", "username", "password", "from", "to"]
               if not ecfg.get(k)]
    if missing:
        log.warning("Email config incomplete — missing: %s", ", ".join(missing))
        return False

    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    if subject is None:
        subject = f"Congress Monitor — {len(bills)} new bill(s) [{date_str}]"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = ecfg["from"]
    msg["To"]      = ecfg["to"]
    msg.attach(MIMEText(build_html_email(bills, date_str, headline=headline), "html"))

    try:
        with smtplib.SMTP(ecfg["smtp_host"], int(ecfg["smtp_port"])) as srv:
            srv.ehlo()
            srv.starttls()
            srv.login(ecfg["username"], ecfg["password"])
            srv.sendmail(ecfg["from"], ecfg["to"], msg.as_string())
        log.info("Email sent → %s", ecfg["to"])
        return True
    except smtplib.SMTPException as exc:
        log.error("Email failed: %s", exc)
        return False


def send_test_email(cfg: dict) -> None:
    console.print("[cyan]Sending test email…[/cyan]")
    dummy_tracker = [
        {"stage": "Introduced",   "done": True,  "date": "2025-01-01"},
        {"stage": "In Committee", "done": True,  "date": "2025-01-05"},
        {"stage": "Reported",     "done": False, "date": ""},
        {"stage": "Passed House", "done": False, "date": ""},
        {"stage": "Passed Senate","done": False, "date": ""},
        {"stage": "To President", "done": False, "date": ""},
        {"stage": "Enacted",      "done": False, "date": ""},
    ]
    dummy = {
        "id": "HR1-119", "number": "1", "type": "HR",
        "title": "Test Bill — Congress Monitor is working!",
        "originChamber": "House",
        "sponsor": "Rep. Jane Smith", "party": "D", "state": "CA",
        "status": "🟡 In Committee",
        "description": "A test bill to verify email delivery from Congress Monitor.",
        "stageTracker": dummy_tracker,
        "actionDate": datetime.now().strftime("%Y-%m-%d"),
        "introducedDate": datetime.now().strftime("%Y-%m-%d"),
        "updateDate": datetime.now().strftime("%Y-%m-%d"),
        "url": "https://congress.gov", "latestActionText": "",
    }
    ok = send_email(cfg, [dummy], subject="Congress Monitor — Test Email", headline="Test Email")
    if ok:
        console.print("[green]Test email sent successfully.[/green]")
    else:
        console.print("[red]Test email failed — check config.json.[/red]")


# ---------------------------------------------------------------------------
# Core check logic
# ---------------------------------------------------------------------------

def run_check(
    cfg: dict,
    seen_bills: Set[str],
    show_only: bool = False,
    force_email: bool = False,
) -> List[dict]:
    since = datetime.now(timezone.utc) - timedelta(days=cfg["track_recent_days"])
    log.info(
        "Checking bills updated since %s (119th Congress)…",
        since.strftime("%Y-%m-%d"),
    )

    all_bills = fetch_bills(cfg, since_date=since)
    keywords  = cfg.get("keywords", [])
    filtered  = [b for b in all_bills if matches_keywords(b, keywords)]

    if show_only:
        enrich_bills(cfg, filtered)
        descs = generate_descriptions(filtered, cfg)
        for b in filtered:
            b["description"] = b.get("description") or descs.get(b["id"], "")
        display_table(
            filtered,
            title=f"Bills — last {cfg['track_recent_days']} days ({len(filtered)} total)",
        )
        return filtered

    if force_email:
        bills_to_notify = filtered
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        subj     = f"Congress Monitor — Initial Snapshot: {len(filtered)} bill(s) [{date_str}]"
        headline = "Initial Bill Snapshot"
        log.info("First run — preparing snapshot of %d bill(s).", len(filtered))
    else:
        bills_to_notify = [b for b in filtered if b["id"] not in seen_bills]
        subj     = None
        headline = "New Bills Detected"

    if bills_to_notify:
        enrich_bills(cfg, bills_to_notify)
        descs = generate_descriptions(bills_to_notify, cfg)
        for b in bills_to_notify:
            b["description"] = b.get("description") or descs.get(b["id"], "")

        # Email first — display errors must never block the notification
        ok = send_email(cfg, bills_to_notify, subject=subj, headline=headline)
        if not ok and cfg["email"].get("enabled"):
            log.warning("Email delivery failed — bills printed below.")

        label = "Snapshot" if force_email else "New Bills"
        display_table(bills_to_notify, title=f"{label} ({len(bills_to_notify)})")
    else:
        log.info("No new bills since last check.")

    for b in filtered:
        seen_bills.add(b["id"])

    return bills_to_notify


# ---------------------------------------------------------------------------
# Daemon
# ---------------------------------------------------------------------------

def start_daemon(cfg: dict) -> None:
    seen_bills = load_seen_bills()
    interval   = cfg["check_interval_minutes"]
    has_claude = bool(
        cfg.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY")
    )

    console.print("[bold green]Congress Monitor daemon started.[/bold green]")
    console.print(
        f"  Interval: [bold]{interval} min ({interval // 60}h)[/bold] | "
        f"Congress: [bold]{cfg['congress']}[/bold] | "
        f"Email: [bold]{'on' if cfg['email'].get('enabled') else 'off'}[/bold] | "
        f"Claude descriptions: [bold]{'yes' if has_claude else 'no (set anthropic_api_key)'}[/bold]"
    )
    console.print("  Press Ctrl+C to stop.\n")

    is_first_run = [True]

    def job() -> None:
        force            = is_first_run[0]
        is_first_run[0]  = False
        try:
            run_check(cfg, seen_bills, force_email=force)
            save_seen_bills(seen_bills)
        except Exception as exc:
            log.error("Unexpected error: %s", exc, exc_info=True)

    job()  # immediate first run
    schedule.every(interval).minutes.do(job)

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        console.print("\n[yellow]Daemon stopped.[/yellow]")
        save_seen_bills(seen_bills)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="US Congress Bill Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--show",       action="store_true", help="One-time snapshot, no daemon")
    parser.add_argument("--test-email", action="store_true", help="Send test email and exit")
    parser.add_argument("--reset",      action="store_true", help="Clear seen-bills history and exit")
    args = parser.parse_args()
    cfg  = load_config()

    if args.reset:
        reset_seen_bills()
        console.print("[green]Seen-bills history reset.[/green]")
        return

    if args.test_email:
        send_test_email(cfg)
        return

    if args.show:
        since = datetime.now(timezone.utc) - timedelta(days=cfg["track_recent_days"])
        log.info("Fetching bills since %s…", since.strftime("%Y-%m-%d"))
        all_bills = fetch_bills(cfg, since_date=since)
        filtered  = [b for b in all_bills if matches_keywords(b, cfg.get("keywords", []))]
        enrich_bills(cfg, filtered)
        descs = generate_descriptions(filtered, cfg)
        for b in filtered:
            b["description"] = b.get("description") or descs.get(b["id"], "")
        display_table(
            filtered,
            title=f"Bills — last {cfg['track_recent_days']} days ({len(filtered)} total)",
        )
        return

    start_daemon(cfg)


if __name__ == "__main__":
    main()

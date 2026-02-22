import os
from datetime import datetime, timedelta, timezone

import streamlit as st

from monitor import (
    DEFAULT_CONFIG,
    fetch_bills,
    enrich_bills,
    generate_descriptions,
    matches_keywords,
)


st.set_page_config(page_title="Congress Bill Watchdog", layout="wide")

st.title("Congress Bill Watchdog")
st.caption("Interactive snapshot of recent U.S. congressional bills.")


def _build_config():
    cfg = dict(DEFAULT_CONFIG)
    secret_congress_key = st.secrets.get("CONGRESS_API_KEY", "") if hasattr(st, "secrets") else ""
    secret_anthropic_key = st.secrets.get("ANTHROPIC_API_KEY", "") if hasattr(st, "secrets") else ""
    cfg.update(
        {
            "congress_api_key": st.session_state.get("congress_api_key", "") or secret_congress_key,
            "congress": st.session_state.get("congress", cfg["congress"]),
            "track_recent_days": st.session_state.get("track_recent_days", cfg["track_recent_days"]),
            "bills_per_check": st.session_state.get("bills_per_check", cfg["bills_per_check"]),
            "detail_fetch_limit": st.session_state.get("detail_fetch_limit", cfg["detail_fetch_limit"]),
            "keywords": st.session_state.get("keywords", []),
        }
    )
    cfg["email"] = dict(cfg.get("email", {}))
    cfg["email"]["enabled"] = False
    if st.session_state.get("use_anthropic"):
        cfg["anthropic_api_key"] = st.session_state.get("anthropic_api_key", "") or secret_anthropic_key
    else:
        cfg["anthropic_api_key"] = ""
    return cfg


@st.cache_data(show_spinner=False)
def _cached_fetch(cfg: dict, since_iso: str):
    since = datetime.fromisoformat(since_iso)
    bills = fetch_bills(cfg, since_date=since)
    keywords = cfg.get("keywords", [])
    return [b for b in bills if matches_keywords(b, keywords)]


with st.sidebar:
    st.subheader("Settings")
    st.text_input(
        "Congress API key",
        key="congress_api_key",
        value=os.environ.get("CONGRESS_API_KEY", "")
        or (st.secrets.get("CONGRESS_API_KEY", "") if hasattr(st, "secrets") else ""),
    )
    st.number_input("Congress #", min_value=93, max_value=120, step=1, key="congress", value=DEFAULT_CONFIG["congress"])
    st.slider("Track recent days", min_value=1, max_value=30, key="track_recent_days", value=DEFAULT_CONFIG["track_recent_days"])
    st.number_input("Bills per check", min_value=10, max_value=250, step=10, key="bills_per_check", value=DEFAULT_CONFIG["bills_per_check"])
    st.number_input("Detail fetch limit", min_value=10, max_value=250, step=10, key="detail_fetch_limit", value=DEFAULT_CONFIG["detail_fetch_limit"])
    st.text_input("Keywords (comma-separated)", key="keywords_raw", value="")
    st.divider()
    st.checkbox("Generate descriptions (Anthropic)", key="use_anthropic", value=False)
    st.text_input(
        "Anthropic API key",
        key="anthropic_api_key",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", "")
        or (st.secrets.get("ANTHROPIC_API_KEY", "") if hasattr(st, "secrets") else ""),
    )


if st.button("Fetch latest bills"):
    raw = st.session_state.get("keywords_raw", "").strip()
    keywords = [k.strip() for k in raw.split(",") if k.strip()]
    st.session_state["keywords"] = keywords

    cfg = _build_config()
    if not cfg["congress_api_key"]:
        st.error("Please provide a Congress API key (free at api.congress.gov).")
        st.stop()

    since = datetime.now(timezone.utc) - timedelta(days=cfg["track_recent_days"])
    since_iso = since.isoformat()

    with st.spinner("Fetching bills from Congress.gov..."):
        bills = _cached_fetch(cfg, since_iso)
        enrich_bills(cfg, bills)
        descs = generate_descriptions(bills, cfg)
        for b in bills:
            b["description"] = b.get("description") or descs.get(b["id"], "")

    st.success(f"Loaded {len(bills)} bills updated since {since.strftime('%Y-%m-%d')}")

    rows = []
    for b in bills:
        sponsor = b.get("sponsor", "")
        party = b.get("party", "")
        state = b.get("state", "")
        if sponsor and (party or state):
            sponsor = f"{sponsor} ({'-'.join(x for x in [party, state] if x)})"
        rows.append(
            {
                "Bill": f"{b['type']} {b['number']}",
                "Title": b.get("title", ""),
                "Description": b.get("description", ""),
                "Sponsor": sponsor,
                "Status": b.get("status", ""),
                "Date": b.get("actionDate") or b.get("introducedDate", ""),
                "Link": b.get("url", ""),
            }
        )

    st.dataframe(rows, use_container_width=True, hide_index=True)

    if bills:
        st.markdown("### Links")
        for b in bills[:50]:
            st.markdown(f"- [{b['type']} {b['number']}]({b.get('url', '')})")

else:
    st.info("Configure settings in the sidebar, then click **Fetch latest bills**.")

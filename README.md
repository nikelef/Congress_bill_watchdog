# Congress Bill Watchdog

Congress Bills Monitoring.

## Overview

This repository monitors U.S. congressional bills and records updates for downstream analysis and alerting.

## Requirements

- Python 3

## Quick Start

```bash
pip install -r requirements.txt
python monitor.py
```

## Streamlit App

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The Streamlit app reads the Congress API key from the sidebar. You can also set:

- `CONGRESS_API_KEY` in your environment
- `ANTHROPIC_API_KEY` if you enable descriptions

## Notes

- Configuration lives in `config.json`. A safe template is provided in `config.example.json`.
- Collected data is stored under `data/`.

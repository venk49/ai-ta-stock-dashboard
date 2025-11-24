# AI-Powered Technical Stock Analysis Dashboard

This repository contains a Streamlit-based dashboard for technical stock analysis. It fetches data from Yahoo Finance using `yfinance`, renders interactive Plotly charts, and optionally uses Google Gemini (via `google-generativeai`) for AI-driven chart commentary when you provide a Gemini API key.

## Files included
- `ai-ta-stock-dashboard.py` — main Streamlit app entrypoint (patched for safe Kaleido use on Streamlit Cloud).
- `requirements.txt` — Python dependencies.
- `.gitignore` — sensible git ignores for a Python/Streamlit project.
- `README.md` — this file.

> The patched app disables Kaleido image export on Streamlit Cloud (because Streamlit Cloud cannot install Chrome). Interactive Plotly charts still work normally. If you deploy to a host where Chrome + Kaleido are available, the app will export images as intended.

## Quick setup (local)

1. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate    # Windows (PowerShell)
```

2. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. Run the app locally:
```bash
streamlit run ai-ta-stock-dashboard.py
```

> **Note:** Kaleido requires a working Chrome installation for exporting images. If image export fails locally, install Chrome and ensure `kaleido` is installed. On Streamlit Cloud, image export is disabled automatically.

## Deploy to Streamlit Community Cloud (recommended)

1. Create a new GitHub repository and push these files.
2. On Streamlit Cloud (https://share.streamlit.io) click **New app** → choose your GitHub repo → branch `main` → file `ai-ta-stock-dashboard.py` → Deploy.
3. Add your Gemini API key (optional) in the app's **Settings → Secrets** as `GOOGLE_API_KEY` to enable AI analysis.

## Environment / Secrets
- `GOOGLE_API_KEY` — (optional) Google Gemini API key. Add under Streamlit secrets or as environment variable if running locally.

## Troubleshooting
- If you see: `Kaleido requires Google Chrome to be installed` — this means the host doesn't provide Chrome. The app will not crash; image exports will be skipped and interactive charts remain available.
- If Gemini analysis fails, ensure the `GOOGLE_API_KEY` is valid and network egress is allowed on your host.

## License
Use this project as you wish — add a license file if you plan to distribute.

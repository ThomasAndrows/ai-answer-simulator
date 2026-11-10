# 🔎 AI Answer Simulator

**Would ChatGPT or Perplexity cite your website? Find out, and fix it, for a few cents.**

🎥 **Demo video:** https://youtu.be/x7jBgwOZmr0

Built by team **Super AI Visibility** at OpenRouter x Meta Build Day (Nov 9, 2026) for the **$1 Challenge**, using **Meta Muse Ember** via OpenRouter.

## What it does

1. **Reads your site** (homepage + a few internal pages)
2. **Generates real buyer questions** people ask AI assistants about your kind of business
3. **Simulates an AI answer engine** on every question, in parallel: would it cite your site, and if not, what was missing?
4. **Writes an AEO fix pack**: FAQ answers, FAQPage JSON-LD schema, and an `llms.txt` file (unknown facts become `[ADD: ...]` placeholders, never invented)
5. **Re-tests** the same questions with the fixes and shows the **before/after citation score**
6. **Live cost meter** the whole time

## Why Muse Ember

AI visibility (AEO) audits are repetitive agent work: dozens of simulated answers per site, every time content changes. That is exactly what a small, efficient model is for. A full audit + fix + re-test cycle costs a fraction of a dollar, so $1 covers many complete site audits.

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env   # add your OpenRouter key + Muse Ember model ID
streamlit run app.py
```

## Results

| Site | Questions | Cited before | Cited after | Cost |
|---|---|---|---|---|
| _fill in after your run_ | 20 | | | $ |

## License

MIT, see LICENSE

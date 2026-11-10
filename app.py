"""
AI Answer Simulator - a test suite for your website's AI visibility.
Powered by Meta Muse Ember via OpenRouter. Built for the $1 Challenge.

Think of it like unit tests:
  - Test cases  = real questions buyers ask ChatGPT / Perplexity, per buyer persona
  - Test runner = Muse Ember acting as an AI answer engine ("would I cite this site?")
  - Fixer       = Muse Ember writing the missing content (FAQ, schema, llms.txt)
  - Re-run      = same tests against the fixed content -> red to green
"""

import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse

import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = os.getenv("MODEL", "meta/muse-ember")  # confirmed on openrouter.ai/meta/muse-ember
PRICE_IN = float(os.getenv("PRICE_IN", "0.10"))   # USD per 1M input tokens (fallback only)
PRICE_OUT = float(os.getenv("PRICE_OUT", "0.50"))  # USD per 1M output tokens (fallback only)
MOCK = os.getenv("MOCK") == "1"  # offline UI testing only, never for the demo
MAX_WORKERS = 10

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY") or "missing")

PERSONAS = {
    "🧑‍💼 Marketing leader (CMO)": "a CMO or head of marketing comparing vendors and looking for ROI and trust",
    "🚀 Startup founder": "a busy startup founder with a small budget who wants quick, practical answers",
    "👩‍💻 Developer / engineer": "a developer who wants technical specifics, integrations and implementation detail",
    "🏪 Small business owner": "a non-technical small business owner who wants plain-language help and pricing",
}


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------
class CostMeter:
    def __init__(self):
        self.cost = 0.0
        self.calls = 0
        self.tokens = 0

    def add(self, usage):
        if usage is None:
            return
        self.calls += 1
        p = getattr(usage, "prompt_tokens", 0) or 0
        c = getattr(usage, "completion_tokens", 0) or 0
        self.tokens += p + c
        real = getattr(usage, "cost", None)
        if real is None and getattr(usage, "model_extra", None):
            real = usage.model_extra.get("cost")
        self.cost += float(real) if real is not None else p / 1e6 * PRICE_IN + c / 1e6 * PRICE_OUT


class _MockUsage:
    def __init__(self):
        self.prompt_tokens = random.randint(2000, 4000)
        self.completion_tokens = random.randint(80, 300)
        self.cost = None


def ask(prompt, meter, max_tokens=2500, mock=None):
    if MOCK:
        time.sleep(random.uniform(0.2, 0.6))
        meter.add(_MockUsage())
        return mock
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are precise and honest. Reply with valid JSON only, no markdown."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
        extra_body={"usage": {"include": True}, "reasoning": {"effort": "low"}},
    )
    meter.add(resp.usage)
    data = parse_json(resp.choices[0].message.content or "")
    if isinstance(data, list):
        return {"questions": data}
    return data if isinstance(data, dict) else None


def parse_json(text):
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    return None


# ---------------------------------------------------------------------------
# Site reading
# ---------------------------------------------------------------------------
HEADERS = {"User-Agent": "Mozilla/5.0 (AI-Answer-Simulator)"}


def page_text(url):
    r = requests.get(url, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")
    links = [urljoin(url, a["href"]) for a in soup.find_all("a", href=True)]
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ")).strip(), links


def read_site(url, max_pages=5, max_chars=14000):
    if not url.startswith("http"):
        url = "https://" + url
    if MOCK:
        return f"[PAGE] {url}\nMock content for offline UI testing."
    domain = urlparse(url).netloc
    text, links = page_text(url)
    chunks, seen = [f"[PAGE] {url}\n{text[:5000]}"], {url.rstrip("/")}
    for link in links:
        clean = link.split("#")[0].rstrip("/")
        if urlparse(clean).netloc != domain or clean in seen or re.search(r"\.(pdf|jpg|png|zip|svg)$", clean, re.I):
            continue
        seen.add(clean)
        try:
            t, _ = page_text(clean)
            chunks.append(f"[PAGE] {clean}\n{t[:3000]}")
        except Exception:
            pass
        if len(chunks) >= max_pages:
            break
    return "\n\n".join(chunks)[:max_chars]


# ---------------------------------------------------------------------------
# Agent steps
# ---------------------------------------------------------------------------
def gen_questions(business, persona, desc, n, meter):
    prompt = f"""Business type: {business}
Buyer persona: {desc}

Write {n} realistic questions this persona would type into ChatGPT or Perplexity while looking
for this kind of business or service. Mix comparisons, pricing, "best X for Y", how-to and trust
questions, in this persona's own voice. Do not name any specific company.

Return JSON: {{"questions": ["...", "..."]}}"""
    mock = {"questions": [f"Mock question {i + 1} from {persona}?" for i in range(n)]}
    data = ask(prompt, meter, mock=mock) or {}
    return [{"persona": persona, "question": q} for q in (data.get("questions") or [])[:n]]


def simulate(item, site_url, content, meter, fixed=False):
    prompt = f"""You are an AI answer engine (like ChatGPT search or Perplexity).
A user ({PERSONAS.get(item['persona'], '')}) asked: "{item['question']}"

You found this website: {site_url}
Website content:
\"\"\"{content}\"\"\"

How real answer engines cite: an answer usually cites 3-6 sources. Vendor and company pages
ARE regularly cited when they contain specific, extractable facts relevant to the question
(services, process, timelines, pricing, case results, comparisons, how-to steps), even for
"best X" or "how much" questions. They are NOT cited when they only have vague marketing copy.
{{placeholder_rule}}
Score how likely you are to cite this website as one of your sources:
0-3 = would not cite, 4 = borderline, 5-10 = would cite. Use the full range; a page with several relevant specific facts deserves 5+.

Return JSON: {{"score": 0-10, "reason": "one short sentence",
"missing": "the specific info the site would need to earn (or strengthen) the citation"}}"""
    prompt = prompt.replace("{placeholder_rule}", "Text marked [ADD: ...] is a placeholder the owner will fill with real, specific facts: treat it as if that specific fact is present." if fixed else "")
    mock = {"score": random.randint(6, 9) if fixed else random.randint(1, 7), "reason": "Mock reason.", "missing": "Mock missing info."}
    data = ask(prompt, meter, max_tokens=1200, mock=mock) or {}
    raw = data.get("score", data.get("cited", 0))
    if isinstance(raw, bool):
        score = 8 if raw else 2
    else:
        m = re.search(r"\d+(\.\d+)?", str(raw))
        score = int(float(m.group())) if m else 0
    return {**item, "score": score, "cited": score >= 5, "reason": data.get("reason") or ("No valid reply from model" if not data else ""), "missing": data.get("missing", "")}


def run_tests(items, site_url, content, meter, on_progress, fixed=False):
    results = []
    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        futures = {ex.submit(simulate, it, site_url, content, meter, fixed): it for it in items}
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                results.append({**futures[f], "score": 0, "cited": False, "reason": f"Error: {e}", "missing": ""})
            on_progress(len(results), len(items))
    order = {(i["persona"], i["question"]): k for k, i in enumerate(items)}
    return sorted(results, key=lambda r: order.get((r["persona"], r["question"]), 999))


def gen_fixes(business, site_url, content, misses, meter):
    gaps = "\n".join(f"- Q: {m['question']}\n  Missing: {m['missing']}" for m in misses)
    prompt = f"""Business: {business}
Website: {site_url}
Current site content:
\"\"\"{content[:8000]}\"\"\"

AI answer engines did NOT cite this site for these questions:
{gaps}

Write AEO fixes:
1. "faq": one entry per question above. Answer in 2-4 clear, factual, extractable sentences.
   Use ONLY facts from the site content. Where a fact is unknown, write a placeholder like
   [ADD: your starting price] so the owner fills in real data. Never invent facts.
2. "llms_txt": a complete llms.txt file (markdown) for AI crawlers: H1 name, one-line summary
   in a blockquote, key services, who it is for, key pages.

Return JSON: {{"faq": [{{"q": "...", "a": "..."}}], "llms_txt": "..."}}"""
    mock = {"faq": [{"q": m["question"], "a": "Mock answer. [ADD: real detail]"} for m in misses],
            "llms_txt": "# Mock\n> Mock summary"}
    return ask(prompt, meter, max_tokens=6000, mock=mock) or {"faq": [], "llms_txt": ""}


def faq_schema(faq):
    return {"@context": "https://schema.org", "@type": "FAQPage", "mainEntity": [
        {"@type": "Question", "name": f["q"], "acceptedAnswer": {"@type": "Answer", "text": f["a"]}} for f in faq]}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Answer Simulator", page_icon="🔎", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700;900&family=Roboto+Mono&display=swap');
html, body, .stApp, .stMarkdown, p, li, label, input, button, h1, h2, h3, h4 { font-family: 'Roboto', sans-serif !important; }
code, pre { font-family: 'Roboto Mono', monospace !important; }
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero {padding: 34px 38px; border-radius: 22px; background: radial-gradient(circle at 85% 20%, #3b1d6e 0%, #1b2a4a 45%, #0b0f19 100%); color:#fff; margin-bottom: 22px;}
.hero h1 {color:#fff; font-size: 46px; font-weight: 900; margin: 10px 0 6px 0; line-height:1.1;}
.hero p {color:#c9d3ff; font-size: 18px; margin: 0;}
.pill {display:inline-block; background:#c6ff3d; color:#0b0f19; font-weight:700; padding:4px 12px; border-radius:999px; font-size:12px; margin-right:6px;}
.card {border:1px solid rgba(128,128,128,.25); border-radius:16px; padding:18px 20px; height:100%;}
.card h4 {margin:6px 0 6px 0; font-size:17px;}
.card p {font-size:14px; opacity:.8; margin:0;}
.step {font-size:12px; font-weight:700; color:#7c5cff; letter-spacing:1px; text-transform:uppercase;}
.meter {background:#0b0f19; color:#c6ff3d; border-radius:14px; padding:14px 20px; font-size:22px; font-weight:700; margin: 8px 0 16px 0;}
.meter span {color:#9aa4c7; font-size:14px; font-weight:400; margin-left:10px;}
.score {border-radius:18px; padding:22px 10px; text-align:center; color:#fff;}
.score .n {font-size:56px; font-weight:900; line-height:1;}
.score .l {font-size:12px; opacity:.9; margin-top:8px; text-transform:uppercase; letter-spacing:1px;}
.bad {background: linear-gradient(135deg,#7a1f2b,#c0392b);}
.good {background: linear-gradient(135deg,#0f6b3a,#1fae5b);}
.dark {background: linear-gradient(135deg,#0b0f19,#2b2f45); color:#c6ff3d;}
.callout {border-left:4px solid #7c5cff; padding:10px 16px; background:rgba(124,92,255,.08); border-radius:8px; margin: 10px 0;}
</style>""", unsafe_allow_html=True)

st.markdown("""<div class="hero"><span class="pill">$1 Challenge</span><span class="pill">Meta Muse Ember</span><span class="pill">via OpenRouter</span>
<h1>🔎 AI Answer Simulator <span style="font-size:14px;opacity:.5">v6</span></h1>
<p>A test suite for your website's visibility in ChatGPT, Perplexity and other AI answer engines. Find the failing tests, auto-fix them, re-run. For cents.</p></div>""",
            unsafe_allow_html=True)

if MOCK:
    st.warning("MOCK MODE: fake data for UI testing. Turn off MOCK in .env for real Muse Ember results.")

# ---- Walkthrough for engineers
with st.expander("🧠 New to AEO? The 60-second explainer for engineers", expanded=True):
    a, b = st.columns(2)
    a.markdown("""**The problem**

Search used to be 10 blue links. Now people ask ChatGPT or Perplexity, and the AI answers
with **2-3 cited sources**. If your site isn't one of them, you're invisible.

**AEO** (Answer Engine Optimization) = writing content that AI can *extract and cite*:
clear facts, direct answers, structured data.""")
    b.markdown("""**The engineering analogy**

| Testing | This app |
|---|---|
| Test cases | Questions real buyers ask AI |
| Test runner | Muse Ember playing ChatGPT |
| ❌ Failing test | Site not cited, with the reason |
| Patch | Auto-written FAQ + schema + llms.txt |
| Re-run | Same questions, red → green |""")
    st.markdown('<div class="callout">💡 Why Muse Ember: this is repetitive agent work (dozens of simulated answers per site, re-run every time content changes). A small, efficient model makes a full audit cost cents instead of a consultant\'s day rate.</div>',
                unsafe_allow_html=True)

st.markdown("#### How it works")
cols = st.columns(4)
steps = [
    ("Step 1", "🕷️ Read the site", "Crawls your homepage and key pages into plain text."),
    ("Step 2", "🎭 Persona questions", "Ember role-plays buyer personas and writes the questions each would ask AI."),
    ("Step 3", "🤖 Simulate the AI", "Ember plays ChatGPT on every question, in parallel: cite this site, yes or no, and why?"),
    ("Step 4", "🛠️ Fix and re-test", "Ember writes the missing answers + schema + llms.txt, then re-runs every test."),
]
for c, (s, t, d) in zip(cols, steps):
    c.markdown(f'<div class="card"><div class="step">{s}</div><h4>{t}</h4><p>{d}</p></div>', unsafe_allow_html=True)

st.write("")
st.markdown("#### Run it")
c1, c2 = st.columns([2, 3])
url = c1.text_input("Website URL", "waftr.com")
business = c2.text_input("What does the business do? (one line)", "Google-native analytics consultancy: GA4, GTM, BigQuery, Looker Studio")
c3, c4 = st.columns([3, 1])
personas = c3.multiselect("Buyer personas", list(PERSONAS), default=list(PERSONAS))
per = c4.number_input("Questions per persona", 2, 10, 5)

if st.button("▶️  Run the test suite", type="primary", width="stretch"):
    if not MOCK and not os.getenv("OPENROUTER_API_KEY"):
        st.error("Add OPENROUTER_API_KEY to your .env file, then restart.")
        st.stop()
    if not personas:
        st.error("Pick at least one persona.")
        st.stop()

    meter = CostMeter()
    cost_box = st.empty()

    def show_cost():
        cost_box.markdown(
            f'<div class="meter">💸 ${meter.cost:.4f} spent<span>{meter.calls} Muse Ember calls · '
            f'{meter.tokens:,} tokens · {meter.cost * 100:.2f}% of the $1 budget</span></div>',
            unsafe_allow_html=True)

    show_cost()

    with st.status("Step 1 · Reading the website...") as s:
        try:
            content = read_site(url)
        except Exception as e:
            st.error(f"Could not read {url}: {e}")
            st.stop()
        s.update(label=f"Step 1 · Read {content.count('[PAGE]')} pages ({len(content):,} characters)", state="complete")

    with st.status("Step 2 · Personas are writing their questions...") as s:
        with ThreadPoolExecutor(len(personas)) as ex:
            batches = list(ex.map(lambda p: gen_questions(business, p, PERSONAS[p], per, meter), personas))
        items = [i for b in batches for i in b]
        show_cost()
        s.update(label=f"Step 2 · {len(items)} questions from {len(personas)} personas", state="complete")
    if not items:
        st.error("No questions came back. Check the MODEL id and your API key in .env.")
        st.stop()

    st.markdown("##### Step 3 · Round 1: your site as it is today")
    bar = st.progress(0.0)

    def prog(done, total):
        bar.progress(done / total, text=f"Muse Ember is answering as ChatGPT · {done}/{total}")
        show_cost()

    before = run_tests(items, url, content, meter, prog)
    misses = [r for r in before if not r["cited"]]

    with st.status(f"Step 4 · Writing fixes for {len(misses)} failing tests...") as s:
        fixes = (gen_fixes(business, url, content, misses, meter) if misses else None) or {}
        fixes["faq"] = [f for f in (fixes.get("faq") or []) if isinstance(f, dict) and f.get("q") and f.get("a")]
        fixes["llms_txt"] = fixes.get("llms_txt") or ""
        show_cost()
        s.update(label=f"Step 4 · Wrote {len(fixes.get('faq', []))} FAQ answers + FAQPage schema + llms.txt", state="complete")

    st.markdown("##### Step 4 · Round 2: same questions, with the fixes")
    st.caption("Fixes were written for these exact questions, so Round 2 checks that the fix pack covers them. It simulates AI citation, it does not guarantee real ChatGPT rankings. [ADD: ...] placeholders still need real facts.")
    bar2 = st.progress(0.0)

    def prog2(done, total):
        bar2.progress(done / total, text=f"Re-running the tests · {done}/{total}")
        show_cost()

    faq_text = "\n".join(f"Q: {f['q']}\nA: {f['a']}" for f in fixes.get("faq", []))
    improved = content + "\n\n[PAGE] FAQ\n" + faq_text + "\n\n[llms.txt]\n" + fixes.get("llms_txt", "")
    after = run_tests(items, url, improved, meter, prog2, fixed=True)
    show_cost()

    # ---- Results
    total = len(items)
    b_ok, a_ok = sum(r["cited"] for r in before), sum(r["cited"] for r in after)
    st.divider()
    st.markdown("## 📊 Results")
    k = st.columns(4)
    k[0].markdown(f'<div class="score bad"><div class="n">{b_ok}/{total}</div><div class="l">Cited before ({b_ok / total:.0%})</div></div>', unsafe_allow_html=True)
    k[1].markdown(f'<div class="score good"><div class="n">{a_ok}/{total}</div><div class="l">Cited after ({a_ok - b_ok:+d})</div></div>', unsafe_allow_html=True)
    k[2].markdown(f'<div class="score dark"><div class="n">${meter.cost:.4f}</div><div class="l">Total cost</div></div>', unsafe_allow_html=True)
    k[3].markdown(f'<div class="score dark"><div class="n">{(1 / meter.cost):,.0f}</div><div class="l">Full audits per $1</div></div>'
                  if meter.cost else '<div class="score dark"><div class="n">∞</div><div class="l">Full audits per $1</div></div>', unsafe_allow_html=True)

    st.markdown("#### Visibility by persona")
    rows = []
    for p in personas:
        bp = [r for r in before if r["persona"] == p]
        ap = [r for r in after if r["persona"] == p]
        if bp:
            rows.append({"Persona": p, "Before": sum(r["cited"] for r in bp) / len(bp) * 100,
                         "After": sum(r["cited"] for r in ap) / max(len(ap), 1) * 100})
    if rows:
        st.bar_chart({"Before %": {r["Persona"]: r["Before"] for r in rows},
                      "After %": {r["Persona"]: r["After"] for r in rows}}, color=["#c0392b", "#1fae5b"], stack=False)
        worst = min(rows, key=lambda r: r["Before"])
        st.markdown(f'<div class="callout">🔦 Biggest blind spot: <b>{worst["Persona"]}</b>, cited in only '
                    f'{worst["Before"]:.0f}% of their questions before the fixes.</div>', unsafe_allow_html=True)

    st.markdown("#### Test results")
    st.dataframe([{
        "Persona": b["persona"], "Question": b["question"],
        "Before": f"{'✅' if b['cited'] else '❌'} {b['score']}/10",
        "After": f"{('🟢' if not b['cited'] else '✅') if a['cited'] else '❌'} {a['score']}/10",
        "Why it failed": "" if b["cited"] else b["reason"],
        "What was missing": "" if b["cited"] else b["missing"],
    } for b, a in zip(before, after)], width="stretch", hide_index=True)

    st.markdown("#### 📦 Your AEO fix pack (ready to ship)")
    schema = json.dumps(faq_schema(fixes.get("faq", [])), indent=2)
    t1, t2, t3 = st.tabs(["FAQ content", "FAQPage schema (JSON-LD)", "llms.txt"])
    with t1:
        st.caption("Facts the site doesn't state become [ADD: ...] placeholders. Ember never invents facts.")
        for f in fixes.get("faq", []):
            st.markdown(f"**{f['q']}**  \n{f['a']}")
    with t2:
        st.code(schema, language="json")
        st.download_button("Download faq-schema.json", schema, "faq-schema.json", on_click="ignore")
    with t3:
        st.code(fixes.get("llms_txt", ""), language="markdown")
        st.download_button("Download llms.txt", fixes["llms_txt"], "llms.txt", on_click="ignore")

    st.success(f"Full AI-visibility audit, fixes and re-test for **${meter.cost:.4f}** with Muse Ember. "
               f"Citations went from {b_ok} to {a_ok} out of {total}.")

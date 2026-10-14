"""Gradio demo for the Arabic Legal RAG (redesigned).

Run from the project root:
    uv run python -m legal_rag.ui

What's new vs. the old version:
  * Modern RTL design (Cairo font, hero header, cards, colored metric bars).
  * Faithfulness + Answer-relevancy scores, judged by the SAME model that
    generates the answer (GENERATOR_MODEL) through Ollama.
  * Speed: the answer is shown immediately and the metrics are computed
    AFTER it (so you never wait for them to read the answer); both scores
    come from ONE short JSON call (temperature 0, ~60 tokens, trimmed
    context, model kept warm with keep_alive).
"""
from __future__ import annotations

import json
import os
import re
import time
from html import escape
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import gradio as gr

# Import only from config: importing legal_rag.rag would load the reranker,
# the search stack and Langfuse inside the UI process for no reason.
from legal_rag.config import (
    CANDIDATE_K,
    ENABLE_VERIFICATION,
    GENERATOR_MODEL as GENERATOR_MODEL_NAME,
    INSUFFICIENT_CONTEXT,
    TOP_K,
    VERIFIER_MODEL as VERIFIER_MODEL_NAME,
)

API_URL = os.getenv(
    "LEGAL_RAG_API_URL",
    "http://127.0.0.1:8001",
).rstrip("/")

# Ollama is used only for the quality metrics (same model as generation).
# Change the env var if your Ollama runs elsewhere.
OLLAMA_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434",
).rstrip("/")

ENABLE_EVAL = os.getenv("LEGAL_RAG_ENABLE_EVAL", "1") != "0"
EVAL_TIMEOUT = int(os.getenv("LEGAL_RAG_EVAL_TIMEOUT", "120"))
# Trimming the context is the biggest time saver for the judge call.
EVAL_MAX_CONTEXT_CHARS = int(os.getenv("LEGAL_RAG_EVAL_MAX_CTX", "5000"))


# --------------------------------------------------------------------------
# Design
# --------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');

:root {
    --lr-bg: #faf6f0;
    --lr-card: #fffaf3;
    --lr-card-2: #f4ebdf;
    --lr-border: #e6d8c5;
    --lr-text: #4a3a2c;
    --lr-muted: #8a7664;
    --lr-accent: #b08968;
    --lr-accent-2: #9c7452;
    --lr-good: #6a9a63;
    --lr-mid: #c9953c;
    --lr-bad: #b8604f;
}

body, .gradio-container {
    background: var(--lr-bg) !important;
    color: var(--lr-text);
}

.gradio-container {
    max-width: 1240px !important;
    margin: 0 auto !important;
    font-family: 'Cairo', 'Segoe UI', Tahoma, sans-serif !important;
}

/* ---------- RTL ---------- */
.rtl textarea, .rtl input, .rtl .prose, .rtl .md, .rtl p, .rtl li {
    direction: rtl;
    text-align: right;
    font-family: 'Cairo', 'Segoe UI', Tahoma, sans-serif !important;
    font-size: 1.08rem;
    line-height: 2;
    color: var(--lr-text);
}
.rtl-block { direction: rtl; text-align: right; }

/* ---------- Hero ---------- */
.hero {
    direction: rtl;
    text-align: right;
    padding: 26px 30px;
    border-radius: 20px;
    color: #4a3426;
    background: linear-gradient(135deg, #f3e6d3 0%, #e4cdae 100%);
    border: 1px solid #dcc6a8;
    box-shadow: 0 8px 24px rgba(150, 110, 70, 0.15);
    margin-bottom: 14px;
}
.hero h1 {
    margin: 0 0 6px 0;
    font-size: 2rem;
    font-weight: 800;
    color: #4a3426;
}
.hero p { margin: 0 0 14px 0; color: #6b5441; font-size: 1.02rem; }
.hero .chips { display: flex; gap: 8px; flex-wrap: wrap; }
.hero .chip {
    background: rgba(255, 255, 255, .6);
    border: 1px solid #d9c3a5;
    color: #6b5441;
    border-radius: 999px;
    padding: 4px 14px;
    font-size: .85rem;
    direction: ltr;
}

/* ---------- Section titles ---------- */
.section-title {
    direction: rtl;
    text-align: right;
    font-weight: 700;
    font-size: 1.1rem;
    color: #6b5441;
    margin: 6px 0 2px 0;
}

/* ---------- Answer ---------- */
#answer-box {
    border: 1px solid var(--lr-border);
    border-right: 5px solid var(--lr-accent);
    border-radius: 16px;
    padding: 14px 18px;
    min-height: 90px;
    background: var(--lr-card);
}

/* ---------- Primary button ---------- */
#ask-btn {
    background: linear-gradient(135deg, #bb9473, #9c7452) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 700;
    font-size: 1.05rem;
    border-radius: 12px !important;
    box-shadow: 0 6px 16px rgba(156, 116, 82, .28);
    transition: transform .12s ease, box-shadow .12s ease;
}
#ask-btn:hover { transform: translateY(-1px); box-shadow: 0 10px 22px rgba(156, 116, 82, .36); }

/* ---------- Live timer ---------- */
.timer {
    direction: rtl;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 10px 18px;
    border-radius: 14px;
    border: 1px solid var(--lr-border);
    background: var(--lr-card-2);
}
.timer-label { display: flex; align-items: center; gap: 8px; font-weight: 700; color: var(--lr-muted); }
.timer-sub { font-size: .8rem; color: var(--lr-muted); opacity: .85; margin-top: 2px; }
.timer .dot { width: 10px; height: 10px; border-radius: 50%; background: #cdbba6; }
.timer.running .dot { background: var(--lr-accent); animation: lrpulse 1s infinite; }
.timer.running .timer-label { color: var(--lr-accent-2); }
.timer.done .dot { background: var(--lr-good); }
.timer.done .timer-label { color: var(--lr-good); }
.timer-val {
    direction: ltr;
    font-variant-numeric: tabular-nums;
    font-weight: 800;
    font-size: 1.8rem;
    color: var(--lr-text);
}
.timer.running .timer-val { color: var(--lr-accent-2); }
@keyframes lrpulse {
    0%   { box-shadow: 0 0 0 0 rgba(176, 137, 104, .55); }
    70%  { box-shadow: 0 0 0 9px rgba(176, 137, 104, 0); }
    100% { box-shadow: 0 0 0 0 rgba(176, 137, 104, 0); }
}

/* ---------- Status + timings ---------- */
.status-row { direction: rtl; display: flex; flex-wrap: wrap; gap: 8px; }
.pill {
    border-radius: 999px;
    padding: 4px 14px;
    font-size: .86rem;
    font-weight: 600;
    border: 1px solid var(--lr-border);
    background: var(--lr-card);
    color: var(--lr-text);
}
.pill.ok   { color: var(--lr-good); border-color: var(--lr-good); }
.pill.bad  { color: var(--lr-bad);  border-color: var(--lr-bad); }
.pill.none { opacity: .8; }
.pill.time { direction: ltr; font-variant-numeric: tabular-nums; }

/* ---------- Metrics ---------- */
.metrics { direction: rtl; display: grid; gap: 12px; }
.metric {
    border: 1px solid var(--lr-border);
    border-radius: 14px;
    padding: 12px 14px;
    background: var(--lr-card);
}
.metric .top { display: flex; justify-content: space-between; align-items: baseline; }
.metric .name { font-weight: 700; color: var(--lr-text); }
.metric .val  { font-weight: 800; font-size: 1.5rem; direction: ltr; }
.metric .hint { font-size: .8rem; color: var(--lr-muted); margin-top: 4px; }
.bar { height: 9px; border-radius: 999px; background: #eadfcf; overflow: hidden; margin-top: 8px; }
.bar > span { display: block; height: 100%; border-radius: 999px; transition: width .6s ease; }
.metric.good .val { color: var(--lr-good); } .metric.good .bar > span { background: var(--lr-good); }
.metric.mid  .val { color: var(--lr-mid);  } .metric.mid  .bar > span { background: var(--lr-mid); }
.metric.bad  .val { color: var(--lr-bad);  } .metric.bad  .bar > span { background: var(--lr-bad); }
.metric.na   .val { opacity: .5; }
.metric-note { direction: rtl; font-size: .85rem; color: var(--lr-muted); }

footer { display: none !important; }
"""


EXAMPLES = [
    ["ما هو سن الرشد في القانون المدني المصري؟"],
    ["متى يكون العقد باطلا؟"],
    ["ما هي شروط صحة التعاقد؟"],
    ["ما أحكام التقادم المسقط للالتزام؟"],
]


BROWN = gr.themes.Color(
    name="brown",
    c50="#fbf7f2",
    c100="#f5ece1",
    c200="#eadbc8",
    c300="#dcc6a9",
    c400="#c9a884",
    c500="#b38b65",
    c600="#99714d",
    c700="#7d5a3c",
    c800="#634730",
    c900="#4d3825",
    c950="#3a2a1b",
)

THEME = gr.themes.Soft(
    primary_hue=BROWN,
    secondary_hue=BROWN,
    neutral_hue="stone",
    radius_size="lg",
).set(
    body_background_fill="#faf6f0",
    body_text_color="#4a3a2c",
    background_fill_primary="#fffaf3",
    background_fill_secondary="#f4ebdf",
    border_color_primary="#e6d8c5",
    block_label_text_color="#6b5441",
    block_title_text_color="#6b5441",
    input_background_fill="#fffdf9",
    button_secondary_background_fill="#f1e5d6",
    button_secondary_text_color="#6b5441",
    table_even_background_fill="#fffaf3",
    table_odd_background_fill="#f7efe4",
)


# --------------------------------------------------------------------------
# HTML helpers
# --------------------------------------------------------------------------
def hero_html() -> str:
    return f"""
    <div class="hero">
        <h1>⚖️ مساعد القانون المدني المصري</h1>
        <p>اطرح سؤالك القانوني، واحصل على إجابة مستندة إلى مواد القانون المدني المصري، مع عرض المصادر ذات الصلة ومؤشرات تقييم جودة الإجابة.</p>
        <div class="chips">
            <span class="chip">🧠 Generator: {escape(str(GENERATOR_MODEL_NAME))}</span>
            <span class="chip">🔍 Verifier: {escape(str(VERIFIER_MODEL_NAME))}</span>
        </div>
    </div>
    """


def status_html(verified, timings: dict, total: float, eval_time: float | None = None) -> str:
    if verified is True:
        badge = '<span class="pill ok">✅ تم التحقق</span>'
    elif verified is False:
        badge = '<span class="pill bad">❌ فشل التحقق</span>'
    elif verified is None:
        badge = '<span class="pill none">— بدون تحقق</span>'
    else:
        badge = '<span class="pill none">— حالة التحقق غير معروفة</span>'

    pills = [badge]
    for key, value in (timings or {}).items():
        pills.append(f'<span class="pill time">{escape(str(key))}: {escape(str(value))}s</span>')
    if eval_time is not None:
        pills.append(f'<span class="pill time">eval: {eval_time:.2f}s</span>')
    pills.append(f'<span class="pill time">total: {total:.2f}s</span>')
    return f'<div class="status-row">{"".join(pills)}</div>'


def loading_html(text: str) -> str:
    return f'<div class="metric-note">⏳ {escape(text)}</div>'


def _level(score: float | None) -> str:
    if score is None:
        return "na"
    if score >= 0.8:
        return "good"
    if score >= 0.5:
        return "mid"
    return "bad"


def _metric_card(name: str, icon: str, hint: str, score: float | None) -> str:
    level = _level(score)
    value = "—" if score is None else f"{round(score * 100)}%"
    width = 0 if score is None else round(score * 100)
    return f"""
    <div class="metric {level}">
        <div class="top"><span class="name">{icon} {escape(name)}</span><span class="val">{value}</span></div>
        <div class="bar"><span style="width:{width}%"></span></div>
        <div class="hint">{escape(hint)}</div>
    </div>
    """


def metrics_html(faithfulness: float | None, relevancy: float | None, note: str = "") -> str:
    cards = (
        _metric_card(
            "الأمانة (Faithfulness)",
            "🛡️",
            "هل تستند جميع الادعاءات في الإجابة إلى النصوص القانونية المسترجعة؟",
            faithfulness,
        )
        + _metric_card(
            "الصلة (Relevancy)",
            "🎯",
            "هل تجيب الإجابة عن السؤال المطروح بشكل مباشر وملائم؟",
            relevancy,
        )
    )
    extra = f'<div class="metric-note">{escape(note)}</div>' if note else ""
    return f'<div class="metrics">{cards}</div>{extra}'


EMPTY_STATS = '<div class="metric-note">النتائج     .</div>'
EMPTY_METRICS = '<div class="metric-note">Faithfulness and Relevancy</div>'


# --------------------------------------------------------------------------
# API call (same behaviour and error handling as before)
# --------------------------------------------------------------------------
def call_api(payload: dict) -> dict:
    request = Request(
        url=f"{API_URL}/ask",
        data=json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    # Diagnostic message: confirms Gradio is sending the request.
    print(f"[Gradio] Sending question to: {API_URL}/ask", flush=True)

    try:
        # generation + verification can each take up to OLLAMA_TIMEOUT (300s)
        with urlopen(request, timeout=660) as response:
            print(
                f"[Gradio] API response status: {response.status}",
                flush=True,
            )
            return json.loads(
                response.read().decode("utf-8")
            )

    except HTTPError as exc:
        try:
            error_body = json.loads(
                exc.read().decode("utf-8")
            )
            detail = error_body.get("detail", str(exc))
        except (ValueError, AttributeError):
            detail = str(exc)

        print(
            f"[Gradio] API HTTP error: {exc.code} - {detail}",
            flush=True,
        )
        raise gr.Error(
            f"خطأ من RAG API ({exc.code}): {detail}"
        ) from exc

    except (URLError, TimeoutError, OSError) as exc:
        print(
            f"[Gradio] API connection error: {exc}",
            flush=True,
        )
        raise gr.Error(
            f"تعذر الاتصال بـ FastAPI على {API_URL}. "
            "تأكدي أن السيرفر يعمل على المنفذ 8001. "
            f"التفاصيل: {exc}"
        ) from exc

    except (json.JSONDecodeError, KeyError) as exc:
        print(
            f"[Gradio] Invalid API response: {exc}",
            flush=True,
        )
        raise gr.Error(
            f"استجابة غير متوقعة من RAG API: {exc}"
        ) from exc


# --------------------------------------------------------------------------
# Faithfulness + Relevancy (LLM-as-judge, same model as the generator)
# --------------------------------------------------------------------------
JUDGE_SYSTEM = (
    "You are a strict evaluator for a legal RAG system. "
    "Reply with a single JSON object only, no extra text."
)

JUDGE_TEMPLATE = """Evaluate the ANSWER using the QUESTION and the CONTEXT (legal articles).

Give two integer scores from 0 to 10:
- "faithfulness": 10 = every claim in the ANSWER is supported by the CONTEXT; 0 = the ANSWER is unsupported or contradicts the CONTEXT.
- "relevancy": 10 = the ANSWER directly and completely addresses the QUESTION; 0 = it does not address the QUESTION.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}

Return exactly: {{"faithfulness": <0-10>, "relevancy": <0-10>}}"""


def _to_unit(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, number / 10.0))


def evaluate_quality(question: str, answer: str, context: str):
    """One short Ollama call -> (faithfulness, relevancy) in [0, 1].

    Uses the generator model so Ollama doesn't have to swap models
    (swapping is the slowest part on a small GPU).
    """
    prompt = JUDGE_TEMPLATE.format(
        question=question.strip(),
        context=context[:EVAL_MAX_CONTEXT_CHARS],
        answer=answer.strip(),
    )

    body = {
        "model": GENERATOR_MODEL_NAME,
        "stream": False,
        "format": "json",
        "think": False,          # skip reasoning tokens on thinking models
        "keep_alive": "30m",     # keep the model loaded between requests
        "options": {
            "temperature": 0,
            "num_predict": 60,   # the JSON answer is tiny
        },
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    }

    request = Request(
        url=f"{OLLAMA_URL}/api/chat",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=EVAL_TIMEOUT) as response:
        data = json.loads(response.read().decode("utf-8"))

    content = (data.get("message") or {}).get("content", "")
    # Some models wrap JSON in text or <think> blocks; grab the first {...}.
    match = re.search(r"\{.*?\}", content, flags=re.DOTALL)
    if not match:
        raise ValueError(f"no JSON in judge output: {content[:120]!r}")

    scores = json.loads(match.group(0))
    return _to_unit(scores.get("faithfulness")), _to_unit(scores.get("relevancy"))


# --------------------------------------------------------------------------
# Main handler (generator: shows the answer first, metrics right after)
# --------------------------------------------------------------------------
def run(
    question: str,
    top_k: int,
    candidate_k: int,
    verify: bool,
    evaluate: bool,
):
    if not question or not question.strip():
        raise gr.Error("اكتبي سؤالك الأول.")

    top_k = int(top_k)
    candidate_k = max(int(candidate_k), top_k)

    payload = {
        "question": question.strip(),
        "top_k": top_k,
        "candidate_k": candidate_k,
        "verify": verify,
    }

    start = time.perf_counter()

    # Immediate feedback while the API works.
    yield (
        "⏳ جاري البحث وتوليد الإجابة...",
        [],
        "",
        loading_html("بنجهز الإجابة..."),
        EMPTY_METRICS,
    )

    result = call_api(payload)

    answer = result.get("answer", "")
    if not answer:
        raise gr.Error("الـ API رجّعت إجابة فارغة.")

    insufficient = answer == INSUFFICIENT_CONTEXT
    if insufficient:
        answer = f"⚠️ {answer}"

    rows = [
        [
            source.get("article_number", ""),
            source.get("citation", ""),
            source.get("source_page") or "",
        ]
        for source in result.get("sources", [])
    ]

    context_text = result.get("context") or ""
    verified = result.get("verified")
    timings = result.get("timings", {})

    api_total = time.perf_counter() - start

    will_eval = evaluate and ENABLE_EVAL and not insufficient and bool(context_text.strip())

    # 1) Show the answer NOW; metrics are computed after it is on screen.
    yield (
        answer,
        rows,
        context_text or "(لا يوجد سياق)",
        status_html(verified, timings, api_total),
        loading_html("جاري تقييم الأمانة والصلة...") if will_eval else (
            metrics_html(None, None, "التقييم متوقف أو لا يوجد سياق كافٍ للتقييم.")
        ),
    )

    if not will_eval:
        print(f"[Gradio] Request completed in {api_total:.2f}s", flush=True)
        return

    # 2) Metrics (same model as the generator).
    eval_start = time.perf_counter()
    try:
        faithfulness, relevancy = evaluate_quality(question, result.get("answer", ""), context_text)
        note = ""
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[Gradio] Eval failed: {exc}", flush=True)
        faithfulness, relevancy = None, None
        note = f"تعذر حساب التقييم: {exc}"
    eval_time = time.perf_counter() - eval_start

    total = time.perf_counter() - start
    print(
        f"[Gradio] Request completed in {total:.2f}s (eval {eval_time:.2f}s)",
        flush=True,
    )

    yield (
        answer,
        rows,
        context_text or "(لا يوجد سياق)",
        status_html(verified, timings, total, eval_time),
        metrics_html(faithfulness, relevancy, note),
    )


# --------------------------------------------------------------------------
# Live timer (runs in the browser, starts the moment you press "اسأل")
# --------------------------------------------------------------------------
TIMER_IDLE_HTML = (
    '<div id="lr-timer" class="timer idle">'
    '<div class="timer-info">'
    '<div class="timer-label"><span class="dot"></span>'
    '<span id="lr-timer-state">جاهز</span></div>'
    '<div class="timer-sub">الوقت من لحظة الإرسال</div>'
    '</div>'
    '<div id="lr-timer-val" class="timer-val">00:00.0</div>'
    '</div>'
)

START_TIMER_JS = """
(...args) => {
    if (window.__lrStop) { window.__lrStop(true); }
    if (!args[0] || !String(args[0]).trim()) { return args; }
    const box = document.getElementById('lr-timer');
    const val = document.getElementById('lr-timer-val');
    const state = document.getElementById('lr-timer-state');
    const t0 = performance.now();
    const fmt = (ms) => {
        const s = ms / 1000;
        const m = Math.floor(s / 60);
        const r = s - m * 60;
        return String(m).padStart(2, '0') + ':' + r.toFixed(1).padStart(4, '0');
    };
    if (box) { box.className = 'timer running'; }
    if (state) { state.textContent = 'شغّال...'; }
    const id = setInterval(() => {
        if (val) { val.textContent = fmt(performance.now() - t0); }
    }, 100);
    window.__lrStop = (silent) => {
        clearInterval(id);
        window.__lrStop = null;
        if (val) { val.textContent = fmt(performance.now() - t0); }
        if (!silent) {
            if (box) { box.className = 'timer done'; }
            if (state) { state.textContent = 'تم'; }
        }
    };
    return args;
}
"""

STOP_TIMER_JS = """
() => { if (window.__lrStop) { window.__lrStop(false); } }
"""

RESET_TIMER_JS = """
() => {
    if (window.__lrStop) { window.__lrStop(true); }
    const box = document.getElementById('lr-timer');
    const val = document.getElementById('lr-timer-val');
    const state = document.getElementById('lr-timer-state');
    if (box) { box.className = 'timer idle'; }
    if (val) { val.textContent = '00:00.0'; }
    if (state) { state.textContent = 'جاهز'; }
    return [];
}
"""

# The palette is designed for light mode only.
FORCE_LIGHT_JS = """
() => { document.body.classList.remove('dark'); }
"""


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
with gr.Blocks(title="Arabic Legal RAG", css=CSS, theme=THEME) as demo:
    gr.HTML(hero_html())

    with gr.Row(equal_height=False):
        # ---------------- main column ----------------
        with gr.Column(scale=3):
            question = gr.Textbox(
                label="السؤال القانوني",
                placeholder="اكتب سؤالك هنا... مثال: متى يكون العقد باطلا؟",
                lines=3,
                elem_classes="rtl",
            )

            with gr.Row():
                submit = gr.Button("اسأل ⚖️", variant="primary", elem_id="ask-btn", scale=3)
                clear = gr.ClearButton(value="مسح", scale=1)

            gr.HTML(TIMER_IDLE_HTML)

            gr.HTML('<div class="section-title">💡 الإجابة</div>')
            answer = gr.Markdown(
                value="",
                elem_id="answer-box",
                elem_classes="rtl",
            )

            gr.HTML('<div class="section-title">📚 المصادر</div>')
            sources = gr.Dataframe(
                headers=["رقم المادة", "المرجع", "الصفحة"],
                interactive=False,
                wrap=True,
            )

        # ---------------- side column ----------------
        with gr.Column(scale=1, min_width=300):
            gr.HTML('<div class="section-title">📊 جودة الإجابة</div>')
            metrics = gr.HTML(EMPTY_METRICS)

            gr.HTML('<div class="section-title">⏱️ الحالة والأزمنة</div>')
            stats = gr.HTML(EMPTY_STATS)

            with gr.Accordion("⚙️ إعدادات", open=True):
                top_k = gr.Slider(
                    1,
                    10,
                    value=TOP_K,
                    step=1,
                    label="عدد المواد (top_k)",
                )

                candidate_k = gr.Slider(
                    5,
                    50,
                    value=CANDIDATE_K,
                    step=1,
                    label="المرشحين قبل الـ rerank",
                )

                verify = gr.Checkbox(
                    value=ENABLE_VERIFICATION,
                    label="مراجعة الإجابة (محلي)",
                )

                evaluate = gr.Checkbox(
                    value=ENABLE_EVAL,
                    label="تقييم الأمانة والصلة (Faithfulness / Relevancy)",
                )

    with gr.Accordion("🧾 السياق المُرسل للموديل", open=False):
        context = gr.Textbox(
            lines=12,
            show_label=False,
            interactive=False,
            elem_classes="rtl",
        )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[question],
        label="أمثلة",
    )

    inputs = [question, top_k, candidate_k, verify, evaluate]
    outputs = [answer, sources, context, stats, metrics]

    for trigger in (submit.click, question.submit):
        trigger(
            fn=run,
            inputs=inputs,
            outputs=outputs,
            js=START_TIMER_JS,
        ).then(fn=None, js=STOP_TIMER_JS)

    clear.add([question, answer, sources, context])
    clear.click(
        fn=lambda: (EMPTY_STATS, EMPTY_METRICS),
        inputs=None,
        outputs=[stats, metrics],
        js=RESET_TIMER_JS,
    )

    demo.load(fn=None, js=FORCE_LIGHT_JS)


if __name__ == "__main__":
    print(f"[Gradio] API target: {API_URL}", flush=True)
    print(f"[Gradio] Ollama (metrics): {OLLAMA_URL} / model: {GENERATOR_MODEL_NAME}", flush=True)

    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=7860,
    )
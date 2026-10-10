#!/usr/bin/env python3
"""
تحويل ملف PDF للقانون المدني المصري (إنجليزي يسار / عربي يمين) إلى ملف JSON.

التشغيل:
    pip install pdfplumber
    python pdf_to_json.py input.pdf egyptian_civil_code.json

الخطوات:
  1) قراءة حروف الـ PDF بمواقعها (pdfplumber) وتقسيم الصفحة لعمودين:
     يسار = إنجليزي ، يمين = عربي (الخط الفاصل x = 297).
  2) إعادة بناء السطور؛ العربي بيتقرا من اليمين لليسار، وأرقام الأعداد
     المتعددة الخانات بتتعكس في الـ PDF فبنرجّعها لوضعها الصحيح.
  3) تقسيم النص لمواد (Article N / مادة N) وتتبّع العناوين
     (Part / Chapter / Section / Topic) من الخطوط البولد.
  4) ربط النص العربي بالإنجليزي برقم المادة وكتابة الـ JSON.
"""
import json
import re
import sys

import pdfplumber

SPLIT_X = 297.0                      # الحد الفاصل بين العمودين
AD = '٠١٢٣٤٥٦٧٨٩'                    # الأرقام العربية
OUT_KEYS = ['article_number', 'book', 'chapter', 'section', 'topic',
            'ar_text', 'text_en', 'is_repealed', 'source_page', 'citation']


def a2i(s):
    return int(''.join(str(AD.index(c)) for c in s))


def isbold(line):
    return 'Bold' in line['f']


# ======================================================================
# 1) استخراج السطور من الـ PDF
# ======================================================================
def line_text(chars, rtl):
    chars = sorted(chars, key=lambda c: c['x0'], reverse=rtl)
    toks = [c['text'] for c in chars]
    if not rtl:
        return ''.join(toks)
    # الأرقام المتعددة بتطلع معكوسة -> نعكس كل مجموعة أرقام متتالية
    out, i = [], 0
    while i < len(toks):
        if len(toks[i]) == 1 and toks[i] in AD:
            j = i
            while j < len(toks) and len(toks[j]) == 1 and toks[j] in AD:
                j += 1
            out.extend(reversed(toks[i:j]))
            i = j
        else:
            out.append(toks[i])
            i += 1
    return ''.join(out)


def cluster_lines(chars):
    chars = sorted(chars, key=lambda c: c['top'])
    lines, cur, ref = [], [], None
    for c in chars:
        if ref is None or abs(c['top'] - ref) <= 3.0:
            cur.append(c)
            ref = c['top'] if ref is None else ref
        else:
            lines.append(cur)
            cur, ref = [c], c['top']
    if cur:
        lines.append(cur)
    return lines


def extract_lines(pdf_path):
    out = []
    with pdfplumber.open(pdf_path) as pdf:
        for pn, page in enumerate(pdf.pages, 1):
            left = [c for c in page.chars if (c['x0'] + c['x1']) / 2 < SPLIT_X]
            right = [c for c in page.chars if (c['x0'] + c['x1']) / 2 >= SPLIT_X]
            for col, cs, rtl in (('en', left, False), ('ar', right, True)):
                for ln in cluster_lines(cs):
                    t = line_text(ln, rtl).strip()
                    if t:
                        out.append({
                            'p': pn, 'col': col,
                            'y': round(min(c['top'] for c in ln), 1),
                            'x0': round(min(c['x0'] for c in ln), 1),
                            'x1': round(max(c['x1'] for c in ln), 1),
                            't': t, 'sz': round(ln[0]['size'], 1),
                            'f': ln[0]['fontname'],
                        })
    return out


# ======================================================================
# 2) العمود الإنجليزي: المواد + العناوين
# ======================================================================
SMALL = {'or', 'of', 'the', 'and', 'in', 'to', 'a', 'an', 'as', 'for', 'on', 'by', 'at'}


def titlecase(s):
    ws = s.lower().split()
    return ' '.join(w if (i and w in SMALL) else w.capitalize() for i, w in enumerate(ws))


def clean_title(s):
    return re.sub(r'\s+', ' ', s).strip().rstrip(':.').strip()


ART_RE = re.compile(r'^Article\s*(\d+)\s*$')


def normalize_article_markers(en):
    """يتعرف على Article N سواء في سطر لوحده أو متبوع بنص، وحتى لو مكتوب rticle."""
    mark_re = re.compile(r'^\s*A?rticle\s*(\d+)\b\s*(.*)$')
    res, last = [], 0
    for l in en:
        m = mark_re.match(l['t'])
        if m and not isbold(l):
            N, rest = int(m.group(1)), m.group(2).strip()
            ok = (N == last + 1) or (rest == '' and N > last and (N - last <= 3 or N in (81, 418)))
            if ok:
                last = N
                l1 = dict(l); l1['t'] = 'Article %d' % N
                res.append(l1)
                if rest:
                    l2 = dict(l); l2['t'] = rest
                    res.append(l2)
                continue
        res.append(l)
    return res


def parse_english(lines):
    en = sorted([l for l in lines if l['col'] == 'en'], key=lambda l: (l['p'], l['y']))
    en = normalize_article_markers(en)

    state = dict(part='Preliminary Title: General Provisions',
                 chapter=None, section=None, main=None, sub=None)
    prelim = True
    order, repeal_notes = [], []
    cur = None
    i, n = 0, len(en)

    def meta():
        if state['main'] and state['sub']:
            topic = f"{state['main']} — {state['sub']}"
        else:
            topic = state['main'] or state['sub']
        return dict(book=state['part'], chapter=state['chapter'],
                    section=state['section'], topic=topic)

    while i < n:
        l = en[i]
        m = ART_RE.match(l['t'])
        if m:
            cur = dict(num=int(m.group(1)), page=l['p'], lines=[], **meta())
            order.append(cur)
            i += 1
            continue

        if isbold(l):
            cur = None
            toks = []
            while i < n and isbold(en[i]) and not ART_RE.match(en[i]['t']):
                x = en[i]; txt = x['t']
                # عنوان طويل اتقسم على سطرين -> ندمجهم
                while (x['x0'] >= 35.9 and x['x1'] > 240 and i + 1 < n and isbold(en[i + 1])
                       and en[i + 1]['p'] == x['p'] and en[i + 1]['y'] - x['y'] < 17
                       and not ART_RE.match(en[i + 1]['t'])):
                    i += 1; x = en[i]; txt += ' ' + x['t']
                toks.append(clean_title(txt)); i += 1

            expect = None
            for tk in toks:
                if re.search(r'repealed', tk, re.IGNORECASE):
                    mm = re.search(r'(\d+)\s*[-–]\s*(\d+)', tk)
                    repeal_notes.append((int(mm.group(1)), int(mm.group(2)), l['p'], tk))
                    expect = None
                    continue
                if expect:
                    if expect == 'part':
                        state.update(part=titlecase(tk), chapter=None, section=None, main=None, sub=None)
                        prelim = False
                    elif expect == 'chapter':
                        state.update(chapter=tk, section=None, main=None, sub=None)
                    elif expect == 'section':
                        if prelim: state.update(chapter=tk, section=None, main=None, sub=None)
                        else:      state.update(section=tk, main=None, sub=None)
                    expect = None            # (expect == 'book' يتجاهل الاسم)
                    continue
                if re.match(r'^(FIRST|SECOND) PART$', tk):      expect = 'part';    continue
                if re.match(r'^BOOK [IVX]+$', tk):              expect = 'book';    continue
                if re.match(r'^chapter [IVX]+\.?$', tk, re.IGNORECASE):  expect = 'chapter'; continue
                if re.match(r'^section [IVX]+\.?$', tk, re.IGNORECASE):  expect = 'section'; continue
                mm = re.match(r'^section [IVX]+\.?\s+(.+)$', tk, re.IGNORECASE)
                if mm:
                    if prelim: state.update(chapter=mm.group(1), section=None, main=None, sub=None)
                    else:      state.update(section=mm.group(1), main=None, sub=None)
                    continue
                mm = re.match(r'^\d+\s*[.\-–]\s*(.+)$', tk)           # "1. Elements of Contracts"
                if mm:
                    if prelim: state.update(section=clean_title(mm.group(1)), main=None, sub=None)
                    else:      state.update(main=clean_title(mm.group(1)), sub=None)
                    continue
                if prelim: state.update(main=tk, sub=None)            # عنوان فرعي بدون رقم
                else:      state['sub'] = tk

            if expect:   # العنوان نفسه مش بولد (زي صفحة 1) -> السطر اللي بعده هو العنوان
                tl = en[i]['t']
                if expect == 'section':
                    if prelim: state.update(chapter=tl, section=None, main=None, sub=None)
                    else:      state.update(section=tl, main=None, sub=None)
                elif expect == 'chapter':
                    state.update(chapter=tl, section=None, main=None, sub=None)
                i += 1
            continue

        if cur is not None:
            cur['lines'].append(l)
        i += 1

    return order, repeal_notes


# ======================================================================
# 3) العمود العربي
# ======================================================================
MK = re.compile(r'^مادة\s*\(?\s*([٠-٩]+)\s*[\(\)]?\s*$')
RNG = re.compile(r'^المواد من ([٠-٩]+) إلى ([٠-٩]+)')


def parse_arabic(lines):
    ar = sorted([l for l in lines if l['col'] == 'ar'], key=lambda l: (l['p'], l['y']))
    ar_list, ar_ranges, acur = [], [], None
    for l in ar:
        t = l['t']
        mr = RNG.match(t)
        if mr and not isbold(l):                       # ملاحظات المواد الملغاة
            acur = dict(num=None, rng=(a2i(mr.group(1)), a2i(mr.group(2))),
                        page=l['p'], lines=[l])
            ar_ranges.append(acur)
            continue
        m = MK.match(t)
        if m:
            acur = dict(num=a2i(m.group(1)), page=l['p'], lines=[], my=l['y'])
            ar_list.append(acur)
            continue
        if isbold(l):
            # مادة نصها بخط بولد (زي مادة 558): أول بولد بعد الرقم هو نص المادة
            if acur is not None and (acur.get('bold_body') or
                                     (not acur['lines'] and l['p'] == acur['page']
                                      and l['y'] - acur.get('my', 0) < 20)):
                if not acur['lines'] or l['y'] - acur['lines'][-1]['y'] < 15:
                    acur['bold_body'] = True
                    acur['lines'].append(l)
                    continue
            acur = None
            continue
        if acur is not None:
            acur['lines'].append(l)

    # قانون الإصدار فيه مادتين برقم 1 و 2 قبل النصوص الفعلية -> نتجاهلهم
    ones = [k for k, a in enumerate(ar_list) if a['num'] == 1]
    return ar_list[ones[1]:], ar_ranges


# ======================================================================
# 4) تنظيف النصوص وتجميع الـ JSON
# ======================================================================
CL_START = re.compile(r'^[\(\)]\s*[٠-٩]+\s*[\(\)]|^[\(\)]\s*[أ-ي]\s*[\(\)]|^[أ-ي]\s*[-–]\s')


def ar_join(lines):
    paras = []
    for l in lines:
        if CL_START.match(l['t']) or not paras:
            paras.append(l['t'])
        else:
            paras[-1] += ' ' + l['t']
    txt = '\n'.join(paras)
    txt = re.sub(r'[\(\)]\s*([٠-٩]+)\s*[\(\)]', r'(\1)', txt)          # (١)
    txt = re.sub(r'[\(\)]\s*([أ-ي])\s*[\(\)]', r'(\1)', txt)           # (أ)
    txt = re.sub(r'(?<=\s)\.([٠-٩][٠-٩/]*)', r'\1.', txt)              # نقطة تايهة قبل رقم
    txt = re.sub(r'^\.([٠-٩][٠-٩/]*)', r'\1.', txt)
    txt = re.sub(r'(\S) (اً|لاً)(?=[\s.،:؛)]|$)', r'\1\2', txt)        # مسافة زيادة قبل تنوين/لا
    txt = re.sub(r'[ \t]+', ' ', txt)
    txt = re.sub(r' *\n *', '\n', txt).strip()
    return txt


def en_join(lines):
    return re.sub(r'\s+', ' ', ' '.join(l['t'] for l in lines)).strip()


def build_records(order, ar_list, ar_ranges):
    ar_by = {a['num']: a for a in ar_list}

    # النص العربي بيدمج مادتي 1021 و 1022 تحت رقم واحد -> نفصلهم عند الفقرة (٢)
    a = ar_by[1021]
    k = [i for i, l in enumerate(a['lines']) if re.match(r'^[\(\)]\s*٢\s*[\(\)]', l['t'])][0]
    ar_by[1022] = dict(num=1022, page=a['page'], lines=a['lines'][k:])
    ar_by[1021] = dict(num=1021, page=a['page'], lines=a['lines'][:k])

    out = []
    for a in order:
        n = a['num']
        rec = dict(article_number=n, book=a['book'], chapter=a['chapter'],
                   section=a['section'], topic=a['topic'])
        if n == 54:                                        # المواد 54-80 ملغاة
            r = [x for x in ar_ranges if x['rng'] == (54, 80)][0]
            rec.update(ar_text=ar_join(r['lines'][1:]),
                       text_en='Articles 54-80 have been repealed by Presidential Decree.',
                       is_repealed=True, source_page=a['page'])
            out.append(rec)
            for m in range(55, 81):
                out.append(dict(rec, article_number=m))
            continue

        ar = ar_by.get(n)
        rec.update(ar_text=ar_join(ar['lines']) if ar else None,
                   text_en=en_join(a['lines']), is_repealed=False, source_page=a['page'])
        out.append(rec)

        if n == 388:                                       # المواد 389-417 ملغاة
            r = [x for x in ar_ranges if x['rng'] == (389, 417)][0]
            for m in range(389, 418):
                out.append(dict(article_number=m, book=a['book'], chapter='Proof of Obligations',
                                section=None, topic=None, ar_text=ar_join(r['lines']),
                                text_en='Articles 389-417 repealed', is_repealed=True,
                                source_page=r['page']))

    out.sort(key=lambda r: r['article_number'])
    for r in out:
        r['citation'] = f"Egyptian Civil Code, Article {r['article_number']}"
    return [{k: r[k] for k in OUT_KEYS} for r in out]


def main():
    if len(sys.argv) != 3:
        sys.exit('usage: python pdf_to_json.py input.pdf output.json')
    pdf_path, out_path = sys.argv[1], sys.argv[2]

    lines = extract_lines(pdf_path)
    order, _repeal = parse_english(lines)
    ar_list, ar_ranges = parse_arabic(lines)
    records = build_records(order, ar_list, ar_ranges)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f'wrote {len(records)} articles -> {out_path}')


if __name__ == '__main__':
    main()
"""Independent re-scoring of a larplarpsahur results.json on a PC.

    pip install sacrebleu
    python tools/rescore_results.py <folder with results.json> [out.json]

Three views of the same run, per track:
  strict   the app's own profile (NFC, lowercase, punctuation stripped; digits spelled
           out only for normalized reference tables). Corpus BLEU/chrF here must match
           eval_mt_summary.csv; if they do not, the app's metric code has a bug.
  lenient  format-neutral: digits spelled out on both sides (vi/en), times "6:30"/"6h30",
           Chinese numerals, 她/他/它 and 的/得/地 merged (same sound), English contractions
           expanded, fillers (uh, um, ờ, ừm…) and stutter repetitions removed.
  clean    lenient, minus samples whose reference is known to be wrong (TRUNCATED_VI_REF).
It also splits MT quality by whether the ASR transcript was exactly right (lenient).
"""
import collections
import json
import re
import statistics
import sys
import unicodedata

from sacrebleu.metrics import BLEU, CHRF

# Round 3 Vie-Man rows whose vie_text holds only the last phrase of what the audio says
# (the man_text and both recordings have the full sentence). Found by the vi/zh reference
# length ratio: these four are below 1.2, the other 176 rows are 1.6-5.0.
TRUNCATED_VI_REF = {'VM-5201', 'VM-0383', 'VM-1989', 'VM-5942'}
TRACKS = ['VE-vi', 'VM-vi', 'VM-zh', 'EV-en']


def strip(text, lang):
    s = unicodedata.normalize('NFC', text or '').lower()
    s = ''.join(c for c in s if not unicodedata.category(c).startswith('P'))
    s = re.sub(r'\s+', ' ', s).strip()
    return s.replace(' ', '') if lang == 'zh' else s


# ---- numbers -------------------------------------------------------------------------
VI = ['không', 'một', 'hai', 'ba', 'bốn', 'năm', 'sáu', 'bảy', 'tám', 'chín']
EN1 = ('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen '
       'fifteen sixteen seventeen eighteen nineteen').split()
EN10 = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
ZH = '零一二三四五六七八九'


def _vi_group(n, padded):
    h, t, o = n // 100, n // 10 % 10, n % 10
    out = [VI[h], 'trăm'] if h or padded else []
    if t == 0 and o == 0:
        return ' '.join(out)
    if t == 0:
        return ' '.join(out + (['linh'] if out else []) + [VI[o]])
    out.append('mười' if t == 1 else VI[t] + ' mươi')
    if o:
        out.append('mốt' if o == 1 and t > 1 else 'lăm' if o == 5 else VI[o])
    return ' '.join(out)


def _en_group(n, with_and):
    out = [EN1[n // 100], 'hundred'] if n >= 100 else []
    r = n % 100
    if r == 0:
        return ' '.join(out)
    if out or with_and:
        out.append('and')
    out.append(EN1[r] if r < 20 else EN10[r // 10] + ('' if r % 10 == 0 else ' ' + EN1[r % 10]))
    return ' '.join(out)


def _spell_int(digits, lang):
    ones = VI if lang == 'vi' else EN1
    if (len(digits) > 1 and digits[0] == '0') or len(digits) > 12:
        return ' '.join(ones[int(c)] for c in digits)          # codes, phone numbers
    n = int(digits)
    if n == 0:
        return ones[0]
    out, higher = [], False
    for value, vi_name, en_name in [(10**9, 'tỷ', 'billion'), (10**6, 'triệu', 'million'),
                                    (1000, 'nghìn', 'thousand'), (1, '', '')]:
        g = n // value % 1000
        if not g:
            continue
        out.append(_vi_group(g, higher) if lang == 'vi' else _en_group(g, higher and value == 1))
        if value > 1:
            out.append(vi_name if lang == 'vi' else en_name)
        higher = True
    return ' '.join(out)


def spell(text, lang):
    t = (text or '').lower()
    if lang == 'vi':
        t = re.sub(r'(\d{1,2}):00\b', r'\1 giờ', t)
        t = re.sub(r'(\d{1,2})[:h](\d{2})\b', r'\1 giờ \2', t)
        t = re.sub(r'(\d{1,2})h\b', r'\1 giờ', t)
        t = re.sub(r'(\d)\.(\d{3})\b', r'\1\2', t)
    else:
        t = re.sub(r'(\d{1,2}):00\b', r'\1', t)
        t = re.sub(r'(\d{1,2}):(\d{2})\b', r'\1 \2', t)
        t = re.sub(r'(\d),(\d{3})\b', r'\1\2', t)
        t = t.replace('%', ' percent')
    t = re.sub(r'\d+', lambda m: ' ' + _spell_int(m.group(), lang) + ' ', t)
    if lang == 'vi':
        t = re.sub(r'\blẻ\b', 'linh', t)
        t = re.sub(r'\bngàn\b', 'nghìn', t)
        t = re.sub(r'mươi tư\b', 'mươi bốn', t)
    return t


def _zh_under_10k(n, leading):
    out, zero = '', False
    for value, unit in [(1000, '千'), (100, '百'), (10, '十'), (1, '')]:
        d = n // value % 10
        if d:
            if zero:
                out += '零'
            out += ('' if value == 10 and d == 1 and leading and not out else ZH[d]) + unit
            zero = False
        elif out:
            zero = True
    return out


def _zh_int(digits):
    n = int(digits)
    if n == 0:
        return '零'
    if n >= 10000:
        hi, lo = divmod(n, 10000)
        tail = ('零' if 0 < lo < 1000 else '') + _zh_under_10k(lo, False) if lo else ''
        return _zh_under_10k(hi, True) + '万' + tail
    return _zh_under_10k(n, True)


def zh_neutral(text):
    t = re.sub(r'(\d{4})(?=年)', lambda m: ''.join(ZH[int(c)] for c in m.group(1)), text or '')
    t = re.sub(r'\d+', lambda m: _zh_int(m.group()), t)
    # Same pronunciation (tā / de): the audio cannot tell them apart.
    return t.replace('她', '他').replace('它', '他').replace('得', '的').replace('地', '的')


FILLERS = {'en': r'\b(uh|um|uhm|er|ah|hmm|mm|uh huh)\b', 'vi': r'(?<!\w)(ờ|ừm|ừ|à|ơ|ạ)(?!\w)'}
EN_FORMS = [(r"\bi'm\b", 'i am'), (r"\b(it|that|there|he|she|what|where)'s\b", r'\1 is'),
            (r"\bcan't\b", 'cannot'), (r"\bwon't\b", 'will not'), (r"n't\b", ' not'),
            (r"'re\b", ' are'), (r"'ll\b", ' will'), (r"'ve\b", ' have'), (r"'d\b", ' would'),
            (r'\balright\b', 'all right'), (r'\bok\b', 'okay'), (r'\bokie doke\b', 'okay')]


def lenient(text, lang):
    if lang == 'zh':
        return strip(zh_neutral(text), 'zh')
    s = strip(spell(text, lang), lang)
    s = re.sub(FILLERS[lang], ' ', s)
    if lang == 'en':
        for pattern, repl in EN_FORMS:
            s = re.sub(pattern, repl, s)
        s = re.sub(r'\b(\w+)( \1\b)+', r'\1', s)
    return re.sub(r'\s+', ' ', s).strip()


# ---- scoring -------------------------------------------------------------------------
def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j - 1] + (x != y), prev[j] + 1, cur[j - 1] + 1))
        prev = cur
    return prev[-1]


def asr_scores(pairs, lang):
    ce = cl = we = wl = exact = 0
    for ref, hyp in pairs:
        ce += levenshtein(ref, hyp)
        cl += len(ref)
        exact += ref == hyp
        if lang != 'zh':
            we += levenshtein(ref.split(), hyp.split())
            wl += len(ref.split())
    return {'cer': ce / cl, 'wer': we / wl if wl else None, 'exact': exact / len(pairs), 'n': len(pairs)}


def mt_scores(samples, norm, tgt):
    refs = [norm(s['ref_trans']) for s in samples]
    hyps = [norm(s['mt_hyp'] or '') for s in samples]
    tok = 'zh' if tgt == 'zh' else '13a'
    return {'bleu': BLEU(tokenize=tok).corpus_score(hyps, [refs]).score,
            'chrf': CHRF().corpus_score(hyps, [refs]).score, 'n': len(samples)}


def main(folder, out_path=None):
    data = json.load(open(folder + '/results.json', encoding='utf8'))
    by_track = collections.defaultdict(list)
    for s in data['samples']:
        if s.get('ref_text') and s.get('ref_trans') and s.get('asr_done'):
            by_track[s['track']].append(s)
    report = {}
    for track in TRACKS:
        samples = by_track.get(track)
        if not samples:
            continue
        src, tgt = samples[0]['reference_language'], samples[0]['tgt_lang']
        normalized = bool(samples[0].get('ref_normalized'))

        def strict_src(x):
            return strip(spell(x, src), src) if normalized else strip(x, src)

        def strict_tgt(x):
            return strip(spell(x, tgt), tgt) if normalized else strip(x, tgt)

        valid_asr = [s for s in samples if not (track == 'VM-vi' and s['id'] in TRUNCATED_VI_REF)]
        valid_mt = [s for s in samples if not (track == 'VM-zh' and s['id'] in TRUNCATED_VI_REF)]
        asr_ok = [s for s in valid_mt if lenient(s['ref_text'], src) == lenient(s['asr_hyp'], src)]
        asr_bad = [s for s in valid_mt if lenient(s['ref_text'], src) != lenient(s['asr_hyp'], src)]
        lat = lambda key: sorted(s[key] for s in samples)
        report[track] = {
            'asr': {
                'strict': asr_scores([(strict_src(s['ref_text']), strict_src(s['asr_hyp'])) for s in samples], src),
                'lenient': asr_scores([(lenient(s['ref_text'], src), lenient(s['asr_hyp'], src)) for s in samples], src),
                'clean': asr_scores([(lenient(s['ref_text'], src), lenient(s['asr_hyp'], src)) for s in valid_asr], src),
            },
            'mt': {
                'strict': mt_scores(samples, strict_tgt, tgt),
                'clean': mt_scores(valid_mt, lambda x: lenient(x, tgt), tgt),
                'asr_exact': mt_scores(asr_ok, lambda x: lenient(x, tgt), tgt) if asr_ok else None,
                'asr_errors': mt_scores(asr_bad, lambda x: lenient(x, tgt), tgt) if asr_bad else None,
            },
            'latency_ms': {
                'asr_median': statistics.median(lat('asr_ms')), 'asr_p90': lat('asr_ms')[int(.9 * len(samples))],
                'mt_median': statistics.median(lat('mt_ms')), 'mt_p90': lat('mt_ms')[int(.9 * len(samples))],
                'rtf_asr': sum(s['asr_ms'] for s in samples) / 1000 / sum(s['duration_s'] for s in samples),
            },
        }
    if out_path:
        json.dump(report, open(out_path, 'w', encoding='utf8'), ensure_ascii=False, indent=1)
    row = '{:<6} {:<8} {:>7} {:>7} {:>7} {:>5}'
    print('ASR'); print(row.format('track', 'view', 'CER%', 'WER%', 'exact%', 'n'))
    for track, r in report.items():
        for view, a in r['asr'].items():
            wer = '-' if a['wer'] is None else '%.2f' % (100 * a['wer'])
            print(row.format(track, view, '%.2f' % (100 * a['cer']), wer, '%.1f' % (100 * a['exact']), a['n']))
    print('\nMT'); print(row.format('track', 'view', 'BLEU', 'chrF', '', 'n'))
    for track, r in report.items():
        for view, m in r['mt'].items():
            if m:
                print(row.format(track, view, '%.2f' % m['bleu'], '%.2f' % m['chrf'], '', m['n']))
    return report


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)

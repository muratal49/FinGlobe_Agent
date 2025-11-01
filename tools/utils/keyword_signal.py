
def countHits(text, terms):
    t = text.lower()
    c, hits = 0, []
    for w in terms:
        n = t.count(w.lower())
        if n > 0:
            c += n
            hits.append((w, n))
    return c, hits

def keywordScorer(text):
    
    # keyword panels
    panel_A1 = ['inflation expectation','interest rate','bank rate','fund rate','price','economic activity','inflation','employment']
    panel_B1 = ['unemployment','growth','exchange rate','productivity','deficit','demand','job market','monetary policy']
    panel_A2 = ['anchor','cut','subdue','decline','decrease','reduce','low','drop','fall','fell','decelerate','slow','pause','pausing','stable','nonaccelerating','downward','tighten']
    panel_B2 = ['ease','easing','rise','rising','increase','expand','improve','strong','upward','raise','high','rapid']

    # domain mapping: 'tighten' is hawkish
    hawk_terms = set(panel_B2 + ['tighten'])
    dove_terms = set([w for w in panel_A2 if w != 'tighten'])
    
    if not text:
        return 0.0, "neutral", "No text found for keyword analysis."
    hc, hh = countHits(text, hawk_terms)
    dc, dh = countHits(text, dove_terms)
    denom = hc + dc
    score = 0.0 if denom == 0 else max(-1.0, min(1.0, (hc - dc) / denom))
    stance = "neutral"
    if score > 0.10: stance = "hawkish"
    elif score < -0.10: stance = "dovish"
    rationale = (
        f"Keyword rationale → hawkish_hits={hc}, dovish_hits={dc}. "
        f"Hawkish: {', '.join([f'{w}×{n}' for w,n in hh]) or 'none'}; "
        f"Dovish: {', '.join([f'{w}×{n}' for w,n in dh]) or 'none'}."
    )
    return score, stance, rationale

# Vendored evolutionary chassis — pinned provenance

## OpenEvolve (chassis; forked/vendored, not a pip dependency)
- Source: https://github.com/algorithmicsuperintelligence/openevolve
- Pinned commit: 411fb59c886c18704caaffb611e17cf9e7d824d2
- Vendored: 2026-07-31 (shallow clone; .git stripped to avoid a nested repo)
- Role: island MAP-Elites + cascade + prompt-sampler chassis (spec D-5).
  We build FunWake-2's three custom parts (cascade evaluator, MAP-elites
  archive on our frozen descriptors, LLM engines) against its interfaces.

## ShinkaEvolve (grafted mechanisms — NOT vendored wholesale)
- Source: https://github.com/SakanaAI/ShinkaEvolve (Apache-2.0), arXiv 2509.19349
- Grafted (re-implemented as ~100-line additions in funwake2/controller/novelty.py):
    1. code-novelty rejection-sampling (embedding + cheap-LLM dedup BEFORE a
       stage-B eval)
    2. fitness/novelty-aware parent sampling
- Only the two mechanisms are grafted; the full repo is not required at runtime.

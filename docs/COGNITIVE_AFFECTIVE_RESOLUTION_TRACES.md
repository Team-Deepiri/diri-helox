# Cognitive-Affective Resolution Traces (CART)

Status: **Proposal — seeking design review before implementation**

## 1. What this is

A Deepiri-original training dataset that captures real technical disagreements
between collaborators end-to-end: not just the resolved decision, but the
emotional/interpersonal state that shaped how the disagreement was navigated
and resolved.

Every existing "reasoning" dataset (papers, docs, cleaned-up post-hoc
interviews) shows only the polished output — the false starts, doubt, ego,
and persuasion that actually produced the answer are edited out. Every
existing "EQ" dataset is scripted role-play with no real stakes. Nobody has
captured the place where cognition and affect are the same event: a real
person, mid-disagreement, whose confidence, trust, and stubbornness visibly
shape which technical path gets taken.

Deepiri is one of the only orgs positioned to capture this at all: a
cross-domain community (fusion, music, cosmology, VR, AI, ...) already having
real technical disagreements on GitHub issues/PRs and Plaky cards, with a
platform and membership agreement that can bake in capture + licensing
structurally instead of per-episode.

## 2. Relationship to other dataset proposals

This repo/PR also documents two companion ideas so reviewers can weigh them
against each other:

- **Live Reasoning Traces (LRT)** — solo capture, right after a contributor
  resolves a hard technical problem: what they tried, what failed, what
  changed their mind, what the answer is. Pure cognitive trace, no
  interpersonal axis.
- **Cross-Domain Translation Corpus (CDTC)** — paired members from unrelated
  domains explain a concept from their field using the other's vocabulary,
  and mark where the analogy breaks down.
- **Cognitive-Affective Resolution Traces (CART, this doc)** — the flagship:
  a real two-person disagreement, captured with both the technical resolution
  and the emotional/relational arc that produced it. LRT becomes the
  technical-only subset of this schema; CDTC becomes a smaller companion
  corpus using the same tagging/licensing pipeline.

## 3. Capture schema

Trigger: a bot/webhook fires a short paired debrief prompt when two
contributors resolve a real disagreement (a PR review thread reaches
resolution, a Plaky card moves out of a "blocked/disputed" state, a Discord
thread a moderator flags as a resolved technical debate). Each participant
independently answers a short structured prompt (voice or text) **before**
comparing notes with the other side, to avoid post-hoc narrative smoothing.

```json
{
  "trace_id": "uuid",
  "domain": "fusion | music | cosmology | vr | ai | ...",
  "participants": ["member_id_a", "member_id_b"],
  "source_ref": {"platform": "github|plaky|discord", "url": "..."},
  "state_before": [
    {
      "participant": "member_id_a",
      "position": "what they believed technically",
      "confidence": "self-reported 1-5",
      "affect": "free text: how they felt going in"
    }
  ],
  "friction_point": {
    "technical_claim": "...",
    "interpersonal_read": "felt dismissed | defensive | curious | ..."
  },
  "turn": {
    "trigger": "the specific evidence/argument that shifted someone",
    "why_it_landed": "logical reason AND emotional/trust reason"
  },
  "resolution": {
    "technical_outcome": "...",
    "relational_outcome": "trust increased | decreased | neutral",
    "conceded_by": "member_id or 'mutual'",
    "genuine_or_fatigue": "..."
  },
  "consent": {
    "license": "per membership agreement clause X",
    "anonymized": true
  }
}
```

## 4. Where this lives

- **Schema + validation + anonymization**: `diri_helox/data_sources/` in
  *this* repo (diri-helox), as a new `CognitiveAffectiveSource` implementing
  the existing `DataSource` ABC (`data_sources/base.py`) alongside
  `self_feedback_source.py` and `composite_source.py`. Helox already owns
  data source adapters and the ingestion contract into training — this
  belongs here, not a new repo.
- **Capture tooling (Discord bot trigger, Plaky webhook, debrief prompt UI)**:
  belongs in `deepiri-control-plane` (or a platform service under it) since
  that's where the bot/webhook integrations for the platform already live —
  NOT in helox. Helox should only ever receive already-structured JSON
  through a defined ingestion endpoint/queue, the same way
  `stream_source.py` / `postgres_source.py` consume upstream data today.
- **Raw voice/text debrief storage + consent records**: needs its own
  storage decision (likely Postgres via `postgres_source.py` conventions,
  or object storage for voice) — flagged as an open question for the design
  session below, since it has real privacy/consent weight (dual-participant
  disagreement data about real people).

Recommendation: **no new repo**. Capture/trigger tooling → control-plane.
Schema, dedup, weighting, ingestion → helox (this repo), reusing
`data_management/semantic_deduplication_engine.py` and
`data_management/domain_weighting_engine.py` once volume exists.

## 5. Open questions for the design session

1. Consent/anonymization model: opt-in per capture vs. blanket membership
   consent — legal review needed (`deepiri-legal`).
2. Where does raw audio live vs. structured JSON only — retention policy.
3. Trigger mechanism: bot heuristic for "this was a real disagreement" needs
   a first pass (false positives waste contributor time).
4. Bootstrap volume: which 2-3 domains pilot this first.
5. How CART, LRT, and CDTC schemas share a common base so they can be mixed
   at training time via `composite_source.py`.

## 6. Next step

Ricardo — please put time on the calendar for a design session to work
through the open questions above and turn this into an implementation plan.
Connor, Sean, Nathan — flagged on the PR for input on the schema, the
control-plane vs. helox split, and anything this proposal is missing.

"""
LLM-as-judge scoring.

Deterministic scorers (``exact_match``, ``contains``, ``token_f1``) only work
when the right answer can be written down in advance. Open-ended responses --
explanations, refactors, summaries -- have many acceptable forms, and string
overlap scores them badly: a correct answer phrased differently looks wrong,
and a fluent wrong answer looks right.

This module scores those responses with a second model. The judge is any
:class:`~.subjects.ResponseGenerator`, so it shares the retry, cache, and cost
machinery already built for evaluation subjects, and can be swapped or run
locally without touching the harness.

The judge is a measuring instrument, so it is built to fail loudly: a verdict
it cannot parse raises :class:`JudgeParseError` rather than returning zero.
A silent zero is indistinguishable from a genuinely bad answer, which would
turn a broken judge into a fake regression.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from .subjects import ResponseGenerator

# Registered alongside the deterministic scorers in ``_VALID_TEST_TYPES`` so a
# suite can mix judged and exact-matched cases in one file.
JUDGE_TEST_TYPE = "llm_judge"

DEFAULT_RUBRIC = (
    "Score how well the response answers the question.\n"
    "5 - fully correct, complete, and directly responsive\n"
    "4 - correct with minor omissions\n"
    "3 - partially correct, or correct but incomplete\n"
    "2 - largely incorrect, with some relevant content\n"
    "1 - incorrect, irrelevant, or empty"
)

_PROMPT_TEMPLATE = """You are grading a response. Apply the rubric exactly.

{rubric}

Judge only on the rubric. Length is not quality: a longer response is not
better unless the extra content is correct and relevant, and a short response
that fully answers the question scores at the top of the scale.

[QUESTION]
{question}
{reference_block}
[RESPONSE]
{response}

Reply with JSON only, no prose around it:
{{"score": <integer {low}-{high}>, "reasoning": "<one sentence>"}}"""

_REFERENCE_TEMPLATE = """
[REFERENCE ANSWER]
{reference}
"""

_COMPARE_TEMPLATE = """You are comparing two responses to the same question.

{rubric}

Judge only on the rubric. Length is not quality: a longer response is not
better unless the extra content is correct and relevant.

[QUESTION]
{question}
{reference_block}
[RESPONSE A]
{first}

[RESPONSE B]
{second}

Reply with JSON only, no prose around it:
{{"winner": "A" | "B" | "tie", "reasoning": "<one sentence>"}}"""

# The judge is asked for bare JSON, but models routinely wrap it in prose or a
# ```json fence. Locating the object is more robust than demanding clean output.
_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)
_SCORE_RE = re.compile(r'"?score"?\s*[:=]\s*(-?\d+(?:\.\d+)?)', re.IGNORECASE)
_WINNER_RE = re.compile(r'"?winner"?\s*[:=]\s*"?(A|B|tie)"?', re.IGNORECASE)


class JudgeParseError(ValueError):
    """Raised when a judge verdict contains no readable score."""


class LlmJudge:
    """
    Scores free-form responses with a model instead of string matching.

    Args:
        subject: The generator asked to grade. Prefer a different model from
            the one under test -- see the self-preference guard in the bias
            controls.
        rubric: The grading criteria shown to the judge. The default is a
            5-point correctness scale.
        scale: Inclusive ``(low, high)`` bounds of the raw score. Verdicts are
            normalised to ``[0, 1]`` before reaching the harness, which assumes
            every score is dimensionless.
        max_new_tokens: Generation budget for the verdict. A judge that runs
            out of tokens mid-JSON produces an unparseable verdict, so this
            wants headroom over the reasoning sentence.
    """

    def __init__(
        self,
        subject: ResponseGenerator,
        rubric: str = DEFAULT_RUBRIC,
        scale: tuple[int, int] = (1, 5),
        max_new_tokens: int = 200,
    ) -> None:
        low, high = scale
        if high <= low:
            raise ValueError(f"scale must be increasing, got {scale!r}")
        self.subject = subject
        self.rubric = rubric
        self.low = low
        self.high = high
        self.max_new_tokens = max_new_tokens

    @property
    def name(self) -> str:
        """Identity of the grading model, recorded in run provenance."""
        return self.subject.name

    def build_prompt(self, question: str, response: str, reference: str = "") -> str:
        """Render the grading prompt sent to the judge."""
        reference_block = ""
        if reference:
            reference_block = _REFERENCE_TEMPLATE.format(reference=reference)
        return _PROMPT_TEMPLATE.format(
            rubric=self.rubric,
            question=question or "(no question provided)",
            reference_block=reference_block,
            response=response if response.strip() else "(empty response)",
            low=self.low,
            high=self.high,
        )

    def parse_verdict(self, raw: str) -> Dict[str, Any]:
        """
        Extract ``{"score", "reasoning", "raw_score"}`` from a judge reply.

        Tries strict JSON first, then a bare ``score:`` field, so a judge that
        adds a preamble is still usable. Out-of-range scores are clamped rather
        than rejected: a 7 on a 1-5 scale is a sloppy verdict, not an
        unreadable one, and clamping keeps the ``[0, 1]`` invariant.
        """
        text = (raw or "").strip()
        if not text:
            raise JudgeParseError("judge returned an empty verdict")

        reasoning = ""
        raw_score: Optional[float] = None

        for match in _JSON_OBJECT_RE.finditer(text):
            try:
                payload = json.loads(match.group())
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and "score" in payload:
                try:
                    raw_score = float(payload["score"])
                except (TypeError, ValueError):
                    continue
                reasoning = str(payload.get("reasoning", ""))
                break

        if raw_score is None:
            fallback = _SCORE_RE.search(text)
            if fallback is None:
                raise JudgeParseError(f"no score found in judge verdict: {text[:200]!r}")
            raw_score = float(fallback.group(1))

        clamped = max(self.low, min(self.high, raw_score))
        return {
            "score": (clamped - self.low) / (self.high - self.low),
            "raw_score": raw_score,
            "reasoning": reasoning,
        }

    def judge(self, question: str, response: str, reference: str = "") -> Dict[str, Any]:
        """Grade one response, returning the parsed verdict."""
        prompt = self.build_prompt(question, response, reference)
        raw = self.subject.generate(prompt, max_new_tokens=self.max_new_tokens)
        return self.parse_verdict(raw)

    def score(self, question: str, response: str, reference: str = "") -> float:
        """Grade one response, returning only the normalised score."""
        return float(self.judge(question, response, reference)["score"])

    # ------------------------------------------------------------------
    # Bias controls
    # ------------------------------------------------------------------

    def parse_comparison(self, raw: str) -> Dict[str, Any]:
        """Extract ``{"winner", "reasoning"}`` from a pairwise verdict."""
        text = (raw or "").strip()
        if not text:
            raise JudgeParseError("judge returned an empty comparison")

        for match in _JSON_OBJECT_RE.finditer(text):
            try:
                payload = json.loads(match.group())
            except json.JSONDecodeError:
                continue
            winner = payload.get("winner") if isinstance(payload, dict) else None
            if isinstance(winner, str) and winner.strip().lower() in {"a", "b", "tie"}:
                return {
                    "winner": winner.strip().upper() if winner.strip().lower() != "tie" else "tie",
                    "reasoning": str(payload.get("reasoning", "")),
                }

        fallback = _WINNER_RE.search(text)
        if fallback is None:
            raise JudgeParseError(f"no winner found in judge verdict: {text[:200]!r}")
        winner = fallback.group(1)
        return {
            "winner": "tie" if winner.lower() == "tie" else winner.upper(),
            "reasoning": "",
        }

    def compare(
        self,
        question: str,
        response_a: str,
        response_b: str,
        reference: str = "",
    ) -> Dict[str, Any]:
        """
        Compare two responses, controlling for position bias.

        Judges have a well-documented habit of favouring whichever answer they
        read first, so a single-order comparison partly measures the ordering
        rather than the answers. This asks twice with the order swapped and
        only declares a winner when both passes agree; a disagreement is
        reported as a tie with ``position_bias`` set, which is the honest
        result -- the judge could not separate these two answers.

        Costs two judge calls per comparison. That is the price of the
        control, and it is cheaper than shipping on a coin flip.
        """
        first = self._compare_once(question, response_a, response_b, reference)
        # Second pass: B is shown first, so its verdict is inverted back to
        # A/B terms before the two passes are compared.
        second_raw = self._compare_once(question, response_b, response_a, reference)
        second = dict(second_raw)
        second["winner"] = {"A": "B", "B": "A", "tie": "tie"}[second_raw["winner"]]

        agreed = first["winner"] == second["winner"]
        return {
            "winner": first["winner"] if agreed else "tie",
            "position_bias": not agreed,
            "first_pass": first,
            "second_pass": second,
        }

    def _compare_once(
        self,
        question: str,
        first: str,
        second: str,
        reference: str,
    ) -> Dict[str, Any]:
        """Run one ordering of a pairwise comparison."""
        reference_block = ""
        if reference:
            reference_block = _REFERENCE_TEMPLATE.format(reference=reference)
        prompt = _COMPARE_TEMPLATE.format(
            rubric=self.rubric,
            question=question or "(no question provided)",
            reference_block=reference_block,
            first=first if first.strip() else "(empty response)",
            second=second if second.strip() else "(empty response)",
        )
        return self.parse_comparison(
            self.subject.generate(prompt, max_new_tokens=self.max_new_tokens)
        )

    def self_preference_risk(self, subject: ResponseGenerator) -> bool:
        """
        Whether the judge is being asked to grade its own output.

        Models score their own output higher than a neutral grader does, so a
        run where judge and subject share a model is measuring loyalty as much
        as quality. Matching is on model identity rather than the subject
        label, since the same model wrapped under two names is the same risk.
        """
        judge_model = self._model_identity(self.subject)
        return bool(judge_model) and judge_model == self._model_identity(subject)

    @staticmethod
    def _model_identity(subject: ResponseGenerator) -> str:
        """Best available model identity for a subject, lowercased."""
        model = getattr(subject, "model", None) or subject.name
        return str(model).strip().lower()

    def config(self) -> Dict[str, Any]:
        """
        Judge identity for run provenance.

        The rubric is hashed into the config alongside the model, because
        editing the rubric changes what the scores mean -- comparing runs
        across a rubric change would report a phantom regression.
        """
        return {
            "judge": self.subject.name,
            "judge_type": type(self.subject).__name__,
            "rubric": self.rubric,
            "scale": [self.low, self.high],
        }

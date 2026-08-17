"""
Validating the judge itself.

A judge is a measuring instrument, and an uncalibrated instrument is worse
than none: it produces confident numbers that move for reasons unrelated to
model quality. The usual calibration is a hand-labelled gold set -- a few
hundred responses graded by people, compared against the judge. That is a
labelling project, and until it exists most teams simply ship an unvalidated
judge.

This module takes the cheaper route. Every check here runs off suites that
already exist, because it builds its own answers rather than needing new
labels:

* **Discrimination** -- a case's own reference answer must outscore another
  case's reference answer. The wrong answer is fluent, well-formed, and
  plainly off-topic, so a judge that cannot separate them is not reading.
* **Length invariance** -- padding an answer with filler must not raise its
  score. This is the single most documented judge failure.
* **Position invariance** -- swapping which answer is shown first must not
  change the winner.
* **Stability** -- the same input graded twice must land in the same place.
* **Agreement** -- on cases where a deterministic scorer is provably right,
  the judge must agree with it, measured by Cohen's kappa.

None of this measures whether the judge shares *your* taste; only a human
comparison can do that. What it does measure is whether the judge is reading
the content at all, which is the failure that actually ships.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

from .judge import LlmJudge

logger = logging.getLogger(__name__)

# Content-free padding. It must not make an answer more correct, so any score
# it moves is the judge responding to bulk rather than substance.
DEFAULT_FILLER = (
    " It is worth noting that this is an important topic with many facets, "
    "and there are several considerations to keep in mind. Broadly speaking, "
    "context matters a great deal here, and different situations may call for "
    "different approaches depending on the circumstances involved."
)

# Kappa bands, from the LLM-as-judge literature. A judge below the floor is
# not agreeing with the reference more than chance would explain.
KAPPA_SHIP = 0.8
KAPPA_USABLE = 0.6


class JudgeValidator:
    """
    Runs label-free invariant probes against an :class:`LlmJudge`.

    Args:
        judge: The judge under test.
        tolerance: How much a score may move before a probe that expects no
            movement counts it as a failure. Judges are stochastic, so a hard
            equality check would fail on sampling noise alone.
        min_margin: How far a correct answer must outscore a mismatched one
            for the judge to count as discriminating. A judge that rates both
            at 0.8 is not usefully separating them even though it ranks them
            in the right order.
    """

    def __init__(
        self,
        judge: LlmJudge,
        tolerance: float = 0.1,
        min_margin: float = 0.25,
    ) -> None:
        self.judge = judge
        self.tolerance = tolerance
        self.min_margin = min_margin

    # ------------------------------------------------------------------
    # Case preparation
    # ------------------------------------------------------------------

    @staticmethod
    def cases_from_suite(tests: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Build probe cases from an existing suite.

        Only tests carrying both a prompt and a reference answer are usable --
        the probes need a known-good response to perturb, and a suite entry
        without one has nothing to offer.
        """
        cases = []
        for test in tests:
            prompt = str(test.get("prompt") or test.get("text") or "").strip()
            expected = str(test.get("expected", "")).strip()
            if prompt and expected:
                cases.append({"prompt": prompt, "expected": expected})
        return cases

    @staticmethod
    def _mismatch_for(cases: Sequence[Dict[str, str]], index: int) -> str:
        """
        Another case's reference answer, used as a fluent wrong answer.

        Rotating by one keeps the pairing deterministic, so a failing probe
        reproduces instead of depending on which shuffle ran.
        """
        return cases[(index + 1) % len(cases)]["expected"]

    # ------------------------------------------------------------------
    # Probes
    # ------------------------------------------------------------------

    def check_discrimination(self, cases: Sequence[Dict[str, str]]) -> Dict[str, Any]:
        """A correct answer must outscore another question's answer."""
        margins = []
        failures = []
        for index, case in enumerate(cases):
            prompt, reference = case["prompt"], case["expected"]
            correct = self.judge.score(prompt, reference, reference)
            wrong = self.judge.score(prompt, self._mismatch_for(cases, index), reference)
            margins.append(correct - wrong)
            if correct - wrong < self.min_margin:
                failures.append({"prompt": prompt, "correct": correct, "mismatched": wrong})
        return self._report("discrimination", margins, failures, len(cases))

    def check_length_invariance(
        self,
        cases: Sequence[Dict[str, str]],
        filler: str = DEFAULT_FILLER,
    ) -> Dict[str, Any]:
        """Padding an answer with content-free filler must not raise its score."""
        deltas = []
        failures = []
        for case in cases:
            plain = self.judge.score(case["prompt"], case["expected"], case["expected"])
            padded = self.judge.score(case["prompt"], case["expected"] + filler, case["expected"])
            deltas.append(padded - plain)
            if padded - plain > self.tolerance:
                failures.append({"prompt": case["prompt"], "plain": plain, "padded": padded})
        return self._report("length_invariance", deltas, failures, len(cases))

    def check_position_invariance(self, cases: Sequence[Dict[str, str]]) -> Dict[str, Any]:
        """Swapping the two answers must not change which one wins."""
        biased = []
        for index, case in enumerate(cases):
            verdict = self.judge.compare(
                case["prompt"],
                case["expected"],
                self._mismatch_for(cases, index),
                reference=case["expected"],
            )
            if verdict["position_bias"]:
                biased.append({"prompt": case["prompt"], "verdict": verdict})
        rate = len(biased) / len(cases) if cases else 0.0
        return {
            "probe": "position_invariance",
            "cases": len(cases),
            "passed": not biased,
            "bias_rate": rate,
            "failures": biased,
        }

    def check_stability(self, cases: Sequence[Dict[str, str]], repeats: int = 2) -> Dict[str, Any]:
        """The same input graded repeatedly must land in the same place."""
        spreads = []
        failures = []
        for case in cases:
            scores = [
                self.judge.score(case["prompt"], case["expected"], case["expected"])
                for _ in range(repeats)
            ]
            spread = max(scores) - min(scores)
            spreads.append(spread)
            if spread > self.tolerance:
                failures.append({"prompt": case["prompt"], "scores": scores})
        return self._report("stability", spreads, failures, len(cases))

    def check_agreement(self, cases: Sequence[Dict[str, str]]) -> Dict[str, Any]:
        """
        Agreement with a deterministic scorer, as Cohen's kappa.

        Each case yields two responses whose correctness is not a matter of
        opinion: the reference answer itself (right) and another case's
        reference (wrong). The judge's normalised score is binarised at the
        midpoint and compared against that ground truth.

        Kappa rather than raw accuracy, because the labels here are balanced
        by construction and a judge that says "good" every time would still
        score 50% accuracy while carrying no information at all.
        """
        truth: List[int] = []
        predicted: List[int] = []
        for index, case in enumerate(cases):
            for response, label in (
                (case["expected"], 1),
                (self._mismatch_for(cases, index), 0),
            ):
                score = self.judge.score(case["prompt"], response, case["expected"])
                truth.append(label)
                predicted.append(1 if score >= 0.5 else 0)

        kappa = cohen_kappa(truth, predicted)
        return {
            "probe": "agreement",
            "cases": len(cases),
            "comparisons": len(truth),
            "kappa": kappa,
            "accuracy": (
                sum(t == p for t, p in zip(truth, predicted)) / len(truth) if truth else 0.0
            ),
            "verdict": self._kappa_verdict(kappa),
            "passed": kappa >= KAPPA_USABLE,
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def validate(
        self,
        cases: Sequence[Dict[str, str]],
        include_agreement: bool = True,
    ) -> Dict[str, Any]:
        """
        Run every probe and report whether the judge is fit to score with.

        Raises if fewer than two cases are supplied: every probe here needs a
        second case to source a wrong answer from, and a one-case run would
        silently compare an answer against itself and pass.
        """
        if len(cases) < 2:
            raise ValueError(f"judge validation needs at least 2 cases, got {len(cases)}")

        probes = [
            self.check_discrimination(cases),
            self.check_length_invariance(cases),
            self.check_position_invariance(cases),
            self.check_stability(cases),
        ]
        if include_agreement:
            probes.append(self.check_agreement(cases))

        failed = [probe["probe"] for probe in probes if not probe["passed"]]
        if failed:
            logger.warning(
                "judge %r failed validation probes: %s — scores from this judge "
                "are not trustworthy until these are fixed",
                self.judge.name,
                ", ".join(failed),
            )
        return {
            "judge": self.judge.name,
            "cases": len(cases),
            "passed": not failed,
            "failed_probes": failed,
            "probes": {probe["probe"]: probe for probe in probes},
        }

    @staticmethod
    def _kappa_verdict(kappa: float) -> str:
        """Plain-language band for an agreement score."""
        if kappa >= KAPPA_SHIP:
            return "strong"
        if kappa >= KAPPA_USABLE:
            return "usable"
        return "unreliable"

    @staticmethod
    def _report(
        probe: str,
        values: List[float],
        failures: List[Dict[str, Any]],
        case_count: int,
    ) -> Dict[str, Any]:
        """Shared shape for the numeric probes."""
        return {
            "probe": probe,
            "cases": case_count,
            "passed": not failures,
            "mean": sum(values) / len(values) if values else 0.0,
            "worst": (min(values) if probe == "discrimination" else max(values)) if values else 0.0,
            "failures": failures,
        }


def cohen_kappa(truth: Sequence[int], predicted: Sequence[int]) -> float:
    """
    Cohen's kappa between two sets of binary labels.

    Returns 0.0 for an empty input and for the degenerate case where chance
    agreement is total -- both raters constant. That is the honest reading:
    a rater that never varies carries no information, so it agrees with
    nothing beyond chance.
    """
    if not truth or len(truth) != len(predicted):
        return 0.0

    from sklearn.metrics import cohen_kappa_score

    score = float(cohen_kappa_score(list(truth), list(predicted)))
    # sklearn returns NaN when both raters are constant; there is no agreement
    # beyond chance to report, so it is reported as none.
    return 0.0 if score != score else score


def validate_judge(
    judge: LlmJudge,
    tests: Sequence[Dict[str, Any]],
    tolerance: float = 0.1,
    min_margin: float = 0.25,
    include_agreement: bool = True,
) -> Dict[str, Any]:
    """Convenience wrapper: validate ``judge`` against an existing suite."""
    validator = JudgeValidator(judge, tolerance=tolerance, min_margin=min_margin)
    cases = validator.cases_from_suite(tests)
    return validator.validate(cases, include_agreement=include_agreement)

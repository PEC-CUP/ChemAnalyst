from __future__ import annotations

import unittest

from app.agents.evidence_sufficiency_benchmark import run_evidence_sufficiency_benchmark


class EvidenceSufficiencyBenchmarkTests(unittest.TestCase):
    def test_default_benchmark_cases_pass(self) -> None:
        result = run_evidence_sufficiency_benchmark()
        self.assertEqual(result["case_count"], 5)
        self.assertEqual(result["passed_count"], result["case_count"])
        self.assertEqual(result["pass_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()

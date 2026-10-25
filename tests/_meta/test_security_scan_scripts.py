"""Behavior of the repository's security scan scripts, without network or LLM calls.

`scan_skills.py` writes docs/security-report.{md,json} and reuses unchanged
skills' findings from the previous JSON report. `scan_pr_skills.py` writes the
pull-request comment and decides whether the scan blocks the PR. These tests
drive both with stand-in scanners, so they need no API key.
"""

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from skill_scanner.core.loader import SkillLoadError
from skill_scanner.core.models import Finding, Report, ScanResult, Severity, ThreatCategory

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    # scan_pr_skills imports scan_skills by name, so register it first.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


scan_skills = _load("scan_skills")
scan_pr_skills = _load("scan_pr_skills")


def make_finding(severity=Severity.HIGH, rule_id="RULE_1", **overrides):
    fields = dict(
        id="f-1",
        rule_id=rule_id,
        category=ThreatCategory.DATA_EXFILTRATION,
        severity=severity,
        title="Sends data to a remote host",
        description="The script posts environment variables.",
        file_path="scripts/run.py",
        line_number=12,
        snippet="requests.post(url, data=os.environ)",
        remediation="Do not send environment variables.",
        analyzer="behavioral",
        metadata={"source": "test"},
    )
    fields.update(overrides)
    return Finding(**fields)


def make_result(name, findings=(), duration=1.25):
    return ScanResult(
        skill_name=name,
        skill_directory=f"skills/{name}",
        findings=list(findings),
        scan_duration_seconds=duration,
        analyzers_used=["behavioral"],
    )


def make_report(*results):
    report = Report()
    for result in results:
        report.add_scan_result(result)
    return report


def write_skill(base, name, body="Body."):
    skill = base / name
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(f"---\nname: {name}\ndescription: Test skill.\n---\n\n{body}\n")
    return skill


# ---------------------------------------------------------------- content_hash


def test_content_hash_tracks_bytes_and_paths_but_not_pycache(tmp_path):
    skill = write_skill(tmp_path, "alpha")
    original = scan_skills.content_hash(skill)

    (skill / "__pycache__").mkdir()
    (skill / "__pycache__" / "x.pyc").write_bytes(b"\x00")
    assert scan_skills.content_hash(skill) == original

    (skill / "notes.md").write_text("x")
    added = scan_skills.content_hash(skill)
    assert added != original

    (skill / "notes.md").rename(skill / "renamed.md")
    assert scan_skills.content_hash(skill) != added

    (skill / "renamed.md").unlink()
    assert scan_skills.content_hash(skill) == original


# ----------------------------------------------------------- JSON round trip


def test_json_report_round_trips_through_result_from_dict():
    finding = make_finding()
    report = make_report(make_result("alpha", [finding]), make_result("beta"))

    data = scan_skills.generate_json_report(
        report,
        skill_meta={"alpha": {"content_hash": "h1", "last_scanned": "2026-01-01T00:00:00+00:00"}},
        run_meta={"model": "m"},
    )
    data = json.loads(json.dumps(data))  # what the next run actually reads

    assert [s["name"] for s in data["skills"]] == ["alpha", "beta"]
    alpha = data["skills"][0]
    assert alpha["content_hash"] == "h1"
    assert alpha["max_severity"] == "HIGH"
    assert alpha["is_safe"] is False
    assert data["totals"]["high"] == 1
    assert data["totals"]["safe_skills"] == 1

    rebuilt = scan_skills._result_from_dict(alpha)
    assert rebuilt.skill_name == "alpha"
    assert rebuilt.max_severity == Severity.HIGH
    assert rebuilt.scan_duration_seconds == 1.25
    assert len(rebuilt.findings) == 1
    got = rebuilt.findings[0]
    for field in ("id", "rule_id", "category", "severity", "title", "description",
                  "file_path", "line_number", "snippet", "remediation", "analyzer",
                  "metadata"):
        assert getattr(got, field) == getattr(finding, field), field


def test_json_skills_are_sorted_by_name():
    report = make_report(make_result("zeta"), make_result("alpha"), make_result("mid"))

    data = scan_skills.generate_json_report(report)

    assert [s["name"] for s in data["skills"]] == ["alpha", "mid", "zeta"]


# ----------------------------------------------------- cache invalidation


@pytest.fixture
def current_scanner(monkeypatch):
    monkeypatch.setattr(scan_skills, "scanner_version", lambda: "2.0.0")
    monkeypatch.setenv("SKILL_SCANNER_LLM_MODEL", "model-a")


def previous_report(**run):
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    base = {"scanner_version": "2.0.0", "model": "model-a", "last_full_scan": stamp}
    base.update(run)
    return {"generated": stamp, "run": base}


@pytest.mark.usefixtures("current_scanner")
@pytest.mark.parametrize(
    ("previous", "expected"),
    [
        (None, "no previous report"),
        (previous_report(), None),
        (previous_report(scanner_version="1.9.0"), "scanner version changed (1.9.0 -> 2.0.0)"),
        (previous_report(model="model-b"), "model changed (model-b -> model-a)"),
        (previous_report(last_full_scan="yesterday"), "unparseable timestamp ('yesterday')"),
    ],
)
def test_cache_invalidation_reason(previous, expected):
    assert scan_skills.cache_invalidation_reason(previous, max_age_days=30) == expected


@pytest.mark.usefixtures("current_scanner")
def test_cache_expires_after_max_age_and_accepts_naive_timestamps():
    old = (datetime.now(timezone.utc) - timedelta(days=31)).replace(tzinfo=None)
    previous = previous_report(last_full_scan=old.isoformat())

    reason = scan_skills.cache_invalidation_reason(previous, max_age_days=30)

    assert reason == "last full scan was 31 days ago (limit 30)"


@pytest.mark.usefixtures("current_scanner")
def test_cache_age_falls_back_to_generated():
    previous = previous_report(last_full_scan=None)
    previous["generated"] = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()

    reason = scan_skills.cache_invalidation_reason(previous, max_age_days=30)

    assert reason.startswith("last full scan was 40 days ago")


def test_load_previous_report(tmp_path, capsys):
    assert scan_skills.load_previous_report(str(tmp_path / "missing.json")) is None

    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert scan_skills.load_previous_report(str(bad)) is None
    assert "Previous report unusable" in capsys.readouterr().out

    good = tmp_path / "good.json"
    good.write_text('{"run": {}}')
    assert scan_skills.load_previous_report(str(good)) == {"run": {}}


# ------------------------------------------------ incremental scanning


class FakeLoader:
    def load_skill(self, skill_dir):
        if (Path(skill_dir) / "BROKEN").exists():
            raise SkillLoadError("cannot load")
        return Path(skill_dir).name


class FakePolicy:
    disabled_rules = set()


class FakeScanner:
    """Stands in for SkillScanner; records which skills were really scanned."""

    def __init__(self, findings_by_skill=None):
        self.loader = FakeLoader()
        self.policy = FakePolicy()
        self.findings_by_skill = findings_by_skill or {}
        self.scanned = []

    def _scan_single_skill(self, skill, skill_dir):
        self.scanned.append(skill)
        return make_result(skill, self.findings_by_skill.get(skill, []))

    def _check_description_overlap(self, skills):
        return []

    def _apply_severity_overrides(self, findings):
        pass


def test_scan_with_progress_reuses_only_unchanged_skills(tmp_path, monkeypatch):
    base = tmp_path / "skills"
    unchanged = write_skill(base, "unchanged")
    write_skill(base, "changed")
    write_skill(base, "new")
    broken = write_skill(base, "broken")
    (broken / "BROKEN").write_text("")

    worker = FakeScanner({"changed": [make_finding(Severity.MEDIUM)]})
    monkeypatch.setattr(scan_skills, "build_scanner", lambda: worker)
    previous = {
        "skills": [
            {
                "name": "unchanged",
                "directory": "skills/unchanged",
                "content_hash": scan_skills.content_hash(unchanged),
                "last_scanned": "2026-01-01T00:00:00+00:00",
                "scan_duration_seconds": 3.0,
                "findings": [scan_skills._finding_dict(make_finding(Severity.LOW))],
            },
            {"name": "changed", "directory": "skills/changed", "content_hash": "stale"},
        ]
    }

    report, skill_meta, run_meta = scan_skills.scan_with_progress(
        FakeScanner(), str(base), workers=2, previous=previous
    )

    assert sorted(worker.scanned) == ["changed", "new"]
    assert run_meta["skills_reused"] == 1
    assert run_meta["skills_rescanned"] == 3
    by_name = {r.skill_name: r for r in report.scan_results}
    assert sorted(by_name) == ["changed", "new", "unchanged"]
    assert by_name["unchanged"].max_severity == Severity.LOW
    assert by_name["changed"].max_severity == Severity.MEDIUM
    assert skill_meta["unchanged"] == {
        "content_hash": scan_skills.content_hash(unchanged),
        "last_scanned": "2026-01-01T00:00:00+00:00",
        "reused": True,
    }
    assert skill_meta["changed"]["reused"] is False
    assert skill_meta["changed"]["content_hash"] != "stale"
    assert [s["skill"] for s in report.skills_skipped] == [str(broken)]
    assert report.skills_skipped[0]["reason"].startswith("SKIP (")


def test_scan_with_progress_rescans_results_whose_llm_analysis_failed(tmp_path, monkeypatch):
    base = tmp_path / "skills"
    skills = {name: write_skill(base, name) for name in ("llm-error", "llm-marker", "static-error")}

    def entry(name, **fields):
        return {"name": name, "directory": f"skills/{name}",
                "content_hash": scan_skills.content_hash(skills[name]), **fields}

    failed = make_finding(Severity.INFO, rule_id="LLM_ANALYSIS_FAILED", analyzer="llm")
    previous = {"skills": [
        entry("llm-error", analyzers_failed=[{"analyzer": "llm_analyzer", "error": "429"}]),
        entry("llm-marker", findings=[scan_skills._finding_dict(failed)]),
        # A deterministic contract failure would recur on a rescan, so it is reused.
        entry("static-error", analyzers_failed=[{"analyzer": "behavioral", "error": "contract"}]),
    ]}
    worker = FakeScanner()
    monkeypatch.setattr(scan_skills, "build_scanner", lambda: worker)

    _, skill_meta, run_meta = scan_skills.scan_with_progress(
        FakeScanner(), str(base), workers=1, previous=previous
    )

    assert sorted(worker.scanned) == ["llm-error", "llm-marker"]
    assert skill_meta["static-error"]["reused"] is True
    assert run_meta["skills_reused"] == 1


def test_scan_with_progress_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_skills.scan_with_progress(FakeScanner(), str(tmp_path / "nope"))


# ------------------------------------------------------- markdown report


def test_generate_report_orders_by_severity_and_details_findings():
    report = make_report(
        make_result("safe-skill"),
        make_result("medium-skill", [make_finding(Severity.MEDIUM, rule_id="R_MED")]),
        make_result("critical-skill", [make_finding(Severity.CRITICAL, rule_id="R_CRIT")]),
    )

    md = scan_skills.generate_report(
        report, run_meta={"scanner_version": "2.0.0", "model": "m", "skills_rescanned": 1,
                          "skills_reused": 2, "full_scan": False},
    )

    rows = [line for line in md.splitlines() if line.startswith("| ") and "skill |" in line]
    assert [row.split(" | ")[0] for row in rows] == [
        "| critical-skill", "| medium-skill", "| safe-skill",
    ]
    assert "**This run:** 1 skill(s) rescanned; 2 unchanged" in md
    assert "### critical-skill — 🔴 CRITICAL" in md
    assert "- **🔴 CRITICAL** `R_CRIT` — Sends data to a remote host" in md
    assert "  > File: `scripts/run.py:12`" in md
    assert "### safe-skill" not in md


def test_generate_report_without_findings():
    md = scan_skills.generate_report(make_report(make_result("a")), run_meta={
        "skills_rescanned": 1, "full_scan": True})

    assert "No findings to report — all skills passed." in md
    assert "**This run:** full rescan of all 1 skill(s)." in md


def cross_skill_report():
    report = make_report(make_result("alpha"), make_result("beta"))
    report.add_cross_skill_findings([
        make_finding(Severity.MEDIUM, rule_id="CROSS_SKILL_SHARED_URL", file_path=None,
                     line_number=None, title="Skills share a URL"),
        make_finding(Severity.HIGH, rule_id="CROSS_SKILL_DATA_RELAY", file_path=None,
                     line_number=None, title="Potential data relay"),
    ])
    return report


def test_generate_report_lists_the_cross_skill_findings_it_counts():
    report = cross_skill_report()

    md = scan_skills.generate_report(report)

    assert "**Total findings:** 2" in md
    assert "No per-skill findings. Cross-skill findings are listed below." in md
    assert "No findings to report" not in md
    assert "<details><summary>2 cross-skill findings</summary>" in md
    high = md.index("- **🟠 HIGH** `CROSS_SKILL_DATA_RELAY` — Potential data relay")
    medium = md.index("- **🟡 MEDIUM** `CROSS_SKILL_SHARED_URL` — Skills share a URL")
    assert high < medium


def test_severity_badge_unknown_value():
    assert scan_skills.severity_badge("HIGH") == "🟠 HIGH"
    assert scan_skills.severity_badge("WEIRD") == "⚫ WEIRD"


# --------------------------------------------------------- PR comment


def test_format_comment_without_scanned_dirs():
    md = scan_pr_skills.format_comment(Report(), [])

    assert md.startswith(scan_pr_skills.COMMENT_MARKER + "\n")
    assert "nothing to scan" in md


def test_format_comment_lists_findings_and_skips(tmp_path):
    report = make_report(
        make_result("ok"),
        make_result("risky", [make_finding(Severity.HIGH, rule_id="R_HIGH", line_number=None)]),
    )
    report.skills_skipped.append({"skill": "skills/broken", "reason": "bad frontmatter"})

    md = scan_pr_skills.format_comment(report, [tmp_path])

    assert md.startswith(scan_pr_skills.COMMENT_MARKER)
    assert "| `risky` | 🟠 HIGH | 1 | ❌ |" in md
    assert "| `ok` | 🟢 SAFE | 0 | ✅ |" in md
    assert md.index("`risky`") < md.index("`ok`")
    assert "<summary><code>risky</code> — 🟠 HIGH (1 finding)</summary>" in md
    assert "  > File: `scripts/run.py`" in md  # no line number
    assert "- `skills/broken`: bad frontmatter" in md


@pytest.mark.parametrize(
    ("fail_on", "blocked"),
    [("CRITICAL", False), ("HIGH", False), ("MEDIUM", True), ("LOW", True), ("NEVER", False)],
)
def test_should_block_uses_worst_severity(fail_on, blocked):
    report = make_report(make_result("ok"), make_result("m", [make_finding(Severity.MEDIUM)]))

    got, reason = scan_pr_skills._should_block(report, fail_on)

    assert got is blocked
    assert reason == ("MEDIUM finding(s) in m" if blocked else None)


def test_format_comment_lists_cross_skill_findings(tmp_path):
    md = scan_pr_skills.format_comment(cross_skill_report(), [tmp_path])

    assert "### Cross-skill findings" in md
    assert "<summary>Across the scanned skills — 🟠 HIGH (2 findings)</summary>" in md
    assert "- **🟠 HIGH** `CROSS_SKILL_DATA_RELAY` — Potential data relay" in md


@pytest.mark.parametrize(
    ("fail_on", "reason"),
    [
        ("CRITICAL", None),
        ("HIGH", "HIGH cross-skill finding(s) (CROSS_SKILL_DATA_RELAY)"),
        ("MEDIUM", "MEDIUM cross-skill finding(s) (CROSS_SKILL_SHARED_URL)"),
        ("NEVER", None),
    ],
)
def test_should_block_counts_cross_skill_findings(fail_on, reason):
    got = scan_pr_skills._should_block(cross_skill_report(), fail_on)

    assert got == (reason is not None, reason)


def test_pr_main_without_api_key_writes_skip_comment(tmp_path, monkeypatch):
    skill = write_skill(tmp_path, "alpha")
    output = tmp_path / "comment.md"
    monkeypatch.delenv("SKILL_SCANNER_LLM_API_KEY", raising=False)
    monkeypatch.setattr(scan_pr_skills, "build_scanner",
                        lambda: pytest.fail("scanner must not be built"))
    monkeypatch.setattr(sys, "argv", ["scan_pr_skills.py", "--output", str(output), str(skill)])

    assert scan_pr_skills.main() == 0
    assert output.read_text() == scan_pr_skills.SKIP_NO_API_KEY_COMMENT


def test_pr_main_skips_non_skill_directories(tmp_path, monkeypatch, capsys):
    output = tmp_path / "comment.md"
    (tmp_path / "not-a-skill").mkdir()
    monkeypatch.setattr(sys, "argv", [
        "scan_pr_skills.py", "--output", str(output),
        str(tmp_path / "not-a-skill"), str(tmp_path / "missing"),
    ])

    assert scan_pr_skills.main() == 0
    out = capsys.readouterr().out
    assert "(no SKILL.md)" in out
    assert "(not a directory)" in out
    assert "nothing to scan" in output.read_text()


def test_pr_main_blocks_on_fail_on_severity(tmp_path, monkeypatch):
    alpha = write_skill(tmp_path, "alpha")
    beta = write_skill(tmp_path, "beta")
    output = tmp_path / "comment.md"
    scanner = FakeScanner({"beta": [make_finding(Severity.CRITICAL)]})
    monkeypatch.setenv("SKILL_SCANNER_LLM_API_KEY", "test-key")
    monkeypatch.setattr(scan_pr_skills, "build_scanner", lambda: scanner)
    monkeypatch.setattr(scanner, "list_analyzers", lambda: ["fake"], raising=False)
    monkeypatch.setattr(sys, "argv", [
        "scan_pr_skills.py", "--output", str(output), str(alpha), str(beta),
    ])

    assert scan_pr_skills.main() == 1
    assert sorted(scanner.scanned) == ["alpha", "beta"]
    comment = output.read_text()
    assert "**Skills scanned:** 2" in comment
    assert "| `beta` | 🔴 CRITICAL | 1 | ❌ |" in comment


# ------------------------------------------------- multi-line finding text

MULTILINE = dict(
    title="Sends data\nto a remote host",
    description="First paragraph.\n\nSecond paragraph.\n- not a list item",
    remediation="Step 1.\r\nStep 2.",
    line_number=None,
)
MULTILINE_LINES = [
    "- **🟠 HIGH** `RULE_1` — Sends data to a remote host",
    "  > First paragraph.",
    "  >",
    "  > Second paragraph.",
    "  > - not a list item",
    "  > File: `scripts/run.py`",
    "  > **Remediation:** Step 1.",
    "  > Step 2.",
]


def _finding_block(md):
    lines = md.splitlines()
    start = lines.index(MULTILINE_LINES[0])
    return lines[start:start + len(MULTILINE_LINES)]


def test_generate_report_keeps_multiline_text_inside_the_finding():
    report = make_report(make_result("risky", [make_finding(**MULTILINE)]))

    md = scan_skills.generate_report(report)

    assert _finding_block(md) == MULTILINE_LINES


def test_format_comment_keeps_multiline_text_inside_the_finding(tmp_path):
    report = make_report(make_result("risky", [make_finding(**MULTILINE)]))

    md = scan_pr_skills.format_comment(report, [tmp_path])

    assert _finding_block(md) == MULTILINE_LINES


# --------------------------------------- fields carried by reused results


def test_json_round_trip_keeps_suppressed_findings_and_llm_usage():
    result = make_result("alpha", [make_finding()])
    result.suppressed_findings = [make_finding(Severity.LOW, rule_id="R_SUPPRESSED", id="f-2")]
    result.llm_usage = {"input_tokens": 1200, "output_tokens": 340}

    data = json.loads(json.dumps(scan_skills.generate_json_report(make_report(result))))
    rebuilt = scan_skills._result_from_dict(data["skills"][0])

    assert [f.rule_id for f in rebuilt.suppressed_findings] == ["R_SUPPRESSED"]
    assert rebuilt.suppressed_findings[0].severity == Severity.LOW
    assert rebuilt.llm_usage == {"input_tokens": 1200, "output_tokens": 340}
    # Suppressed findings stay out of the verdict.
    assert [f.rule_id for f in rebuilt.findings] == ["RULE_1"]


def test_json_round_trip_without_suppressed_findings_or_llm_usage():
    data = scan_skills.generate_json_report(make_report(make_result("alpha")))
    entry = data["skills"][0]

    assert entry["suppressed_findings"] == []
    assert entry["llm_usage"] is None
    rebuilt = scan_skills._result_from_dict(entry)
    assert rebuilt.suppressed_findings == []
    assert rebuilt.llm_usage is None


def test_result_from_older_report_entry_without_new_fields():
    entry = {"name": "alpha", "directory": "skills/alpha", "findings": []}

    rebuilt = scan_skills._result_from_dict(entry)

    assert rebuilt.suppressed_findings == []
    assert rebuilt.llm_usage is None


def test_reused_skill_without_last_scanned_falls_back_to_previous_report_date(tmp_path, monkeypatch):
    base = tmp_path / "skills"
    skill = write_skill(base, "alpha")
    monkeypatch.setattr(scan_skills, "build_scanner", lambda: pytest.fail("nothing to rescan"))
    previous = {
        "generated": "2026-02-03T04:05:06+00:00",
        "skills": [{
            "name": "alpha",
            "directory": "skills/alpha",
            "content_hash": scan_skills.content_hash(skill),
            "findings": [],
        }],
    }

    _, skill_meta, _ = scan_skills.scan_with_progress(FakeScanner(), str(base), previous=previous)

    assert skill_meta["alpha"]["reused"] is True
    assert skill_meta["alpha"]["last_scanned"] == "2026-02-03T04:05:06+00:00"

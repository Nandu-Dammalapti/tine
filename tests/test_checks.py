import numpy as np
import pandas as pd
import pytest

from tine.checks import check_matches, prepare_checks, read_checks


def tasks():
    return pd.DataFrame({"tenant": ["team-a", "team-a"], "task_id": ["NA", "001"]})


def test_check_csv_preserves_identifiers_and_nullable_numeric_labels(tmp_path):
    path = tmp_path / "checks.csv"
    path.write_text("task_id,model,passed\nNA,cheap,1.0\n001,cheap,\n")
    frame = read_checks(path)
    assert frame.task_id.tolist() == ["NA", "001"]
    passes, report = prepare_checks(frame, tasks(), ["cheap", "frontier"])
    assert passes.all()
    assert report["known_checks"] == 1
    assert report["unknown_checks"] == 3
    assert report["per_model"]["cheap"]["known"] == 1


def test_checks_are_scoped_by_tenant_before_alignment():
    frame = pd.DataFrame({"tenant": ["team-a", "team-b"], "task_id": ["NA", "NA"],
                          "model": ["cheap", "cheap"], "passed": [True, False]})
    passes, report = prepare_checks(frame, tasks(), ["cheap"])
    assert passes.tolist() == [[True], [True]]
    assert report["ignored_rows"] == 1
    assert report["identity_scope"] == "tenant/task/model"


@pytest.mark.parametrize("prefix", ["", "\ufeff\n \n"])
def test_check_csv_rejects_duplicate_headers_before_pandas_renames_them(tmp_path, prefix):
    path = tmp_path / "checks.csv"
    path.write_text(prefix + "task_id,model,passed,passed\nNA,cheap,true,false\n")
    with pytest.raises(ValueError, match="unique columns"):
        read_checks(path)


@pytest.mark.parametrize("row", ["ignored,NA,cheap,true", "NA,cheap,true,extra", "NA,cheap"])
def test_check_csv_rejects_extra_or_missing_fields(tmp_path, row):
    path = tmp_path / "checks.csv"
    path.write_text("task_id,model,passed\n" + row + "\n")
    with pytest.raises(ValueError, match="CSV.*fields"):
        read_checks(path)


def test_check_csv_rejects_malformed_quotes(tmp_path):
    path = tmp_path / "checks.csv"
    path.write_text('task_id,model,passed\n"NA"unexpected,cheap,true\n')
    with pytest.raises(ValueError, match="Malformed CSV"):
        read_checks(path)


def test_check_csv_keeps_quoted_fields_and_explicit_unknowns(tmp_path):
    path = tmp_path / "checks.csv"
    path.write_text('\ufeff\n \ntask_id,model,passed\n"task,one","model\nname",false\n\n001,cheap,\n')
    frame = read_checks(path)
    assert frame.task_id.tolist() == ["task,one", "001"]
    assert frame.model.tolist() == ["model\nname", "cheap"]
    assert frame.passed.tolist() == ["false", ""]


@pytest.mark.parametrize("value", ["ERROR", "unknown", 2, -1])
def test_invalid_labels_cannot_silently_turn_into_caught_failures(value):
    frame = pd.DataFrame({"task_id": ["NA"], "model": ["cheap"], "passed": [value]})
    with pytest.raises(ValueError, match="passed"):
        prepare_checks(frame, tasks(), ["cheap"])


def test_duplicate_checks_require_explicit_run_selection():
    frame = pd.DataFrame({"task_id": ["NA", "NA"], "model": ["cheap", "cheap"], "passed": [True, False]})
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_checks(frame, tasks(), ["cheap"])


def test_known_false_is_distinct_from_missing_or_unknown():
    frame = pd.DataFrame({"task_id": ["NA", "001"], "model": ["cheap", "cheap"], "passed": [False, np.nan]})
    passes, report = prepare_checks(frame, tasks(), ["cheap"])
    assert passes.tolist() == [[False], [True]]
    assert report["known_checks"] == report["unknown_checks"] == 1


def test_multitenant_evidence_requires_separate_experiments():
    mixed = tasks()
    mixed.loc[1, "tenant"] = "team-b"
    checks = pd.DataFrame(columns=["task_id", "model", "passed"])
    with pytest.raises(ValueError, match="one nonempty tenant"):
        prepare_checks(checks, mixed, ["cheap"])


def test_check_matches_reads_the_optional_column_for_the_same_rows_and_keeps_unknowns_apart():
    frame = pd.DataFrame({"tenant": ["team-a", "team-a", "team-a", "team-b"], "task_id": ["NA", "001", "NA", "001"],
                          "model": ["cheap", "cheap", "other", "cheap"], "passed": [True, False, True, True],
                          "matches_frontier": ["false", "", "1.0", "true"]})
    matches = check_matches(frame, tasks(), ["cheap", "other", "frontier"])
    # Aligned like the pass matrix; another tenant's row and a missing row say nothing.
    assert matches.shape == prepare_checks(frame, tasks(), ["cheap", "other", "frontier"])[0].shape
    assert matches.tolist() == [[False, True, None], [None, None, None]]
    typed = frame.assign(matches_frontier=[True, None, np.nan, False])
    assert check_matches(typed, tasks(), ["cheap", "other"]).tolist() == [[True, None], [None, None]]
    # Checks written before the column existed have none, and say so.
    assert check_matches(frame.drop(columns="matches_frontier"), tasks(), ["cheap"]) is None
    with pytest.raises(ValueError, match="matches_frontier"):
        check_matches(frame.assign(matches_frontier="maybe"), tasks(), ["cheap"])
    with pytest.raises(ValueError, match="Duplicate"):
        check_matches(pd.concat([frame, frame]), tasks(), ["cheap"])
    # The pass matrix does not depend on the column.
    assert (prepare_checks(frame, tasks(), ["cheap", "other"])[0]
            == prepare_checks(frame.drop(columns="matches_frontier"), tasks(), ["cheap", "other"])[0]).all()

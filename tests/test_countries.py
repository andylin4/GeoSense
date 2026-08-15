"""Tests for canonical country codes and cross-dataset name reconciliation."""

from __future__ import annotations

import pytest

from geoguessr.data.countries import (
    OTHER,
    STREET_VIEW_COUNTRIES,
    audit_coverage,
    class_index,
    class_names,
    display_name,
    normalize,
    to_code,
)


class TestClassList:
    def test_size_matches_decision_4(self):
        # Exactly 100 after the 2026-08-13 OSV-5M reconciliation.
        assert len(STREET_VIEW_COUNTRIES) == 100

    def test_other_is_not_a_trainable_class(self):
        # OTHER is a filter sentinel; the head is 100-way, not 101-way.
        assert OTHER not in class_names()
        assert len(class_names()) == len(STREET_VIEW_COUNTRIES)

    def test_other_can_be_opted_into_and_is_pinned_last(self):
        names = class_names(include_other=True)
        assert names[-1] == OTHER
        assert class_index(include_other=True)[OTHER] == len(names) - 1

    def test_dropped_micro_territories_are_gone(self):
        # Zero OSV-5M rows -> dead softmax columns.
        for code in ("AD", "AS", "GI", "GU", "MC", "MO", "MP", "SM"):
            assert code not in STREET_VIEW_COUNTRIES

    def test_ordering_is_stable(self):
        assert class_names() == class_names()
        assert class_names() == sorted(STREET_VIEW_COUNTRIES)

    def test_no_duplicates(self):
        names = class_names()
        assert len(set(names)) == len(names)

    def test_no_alias_points_at_a_dropped_code(self):
        # Guards against an alias resolving to a code that no longer exists.
        for name in ("Macao", "Vatican", "Macao SAR China"):
            code = to_code(name)
            assert code == OTHER or code in STREET_VIEW_COUNTRIES

    def test_index_inverts_names(self):
        names, idx = class_names(), class_index()
        assert all(names[i] == code for code, i in idx.items())

    def test_codes_are_alpha2_uppercase(self):
        assert all(len(c) == 2 and c.isupper() for c in STREET_VIEW_COUNTRIES)


class TestNormalize:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Côte d'Ivoire", "cote d ivoire"),
            ("  UNITED   STATES  ", "united states"),
            ("Türkiye", "turkiye"),
            ("Bosnia & Herzegovina", "bosnia herzegovina"),
        ],
    )
    def test_folds_accents_case_and_punctuation(self, raw, expected):
        assert normalize(raw) == expected


class TestToCode:
    @pytest.mark.parametrize(
        "name,code",
        [
            ("United States of America", "US"),
            ("USA", "US"),
            ("Czech Republic", "CZ"),
            ("Czechia", "CZ"),
            ("Russian Federation", "RU"),
            ("Korea, Republic of", "KR"),
            ("South Korea", "KR"),
            ("Türkiye", "TR"),
            ("Turkey", "TR"),
            ("Swaziland", "SZ"),
            ("Holland", "NL"),
            ("The Netherlands", "NL"),
            ("Macedonia", "MK"),
            ("Viet Nam", "VN"),
        ],
    )
    def test_aliases_resolve(self, name, code):
        assert to_code(name) == code

    def test_accepts_bare_alpha2(self):
        assert to_code("FR") == "FR"
        assert to_code("fr") == "FR"

    def test_uk_constituent_countries_fold_to_gb(self):
        for part in ("England", "Scotland", "Wales", "Northern Ireland"):
            assert to_code(part) == "GB"

    def test_unknown_falls_into_other_rather_than_raising(self):
        # Keeping the label space closed is the whole point of decision #4.
        assert to_code("Atlantis") == OTHER
        assert to_code("Cote d Ivoire") == OTHER  # no Street View coverage
        assert to_code("") == OTHER

    def test_custom_default(self):
        assert to_code("Atlantis", default="ZZ") == "ZZ"

    def test_every_display_name_round_trips(self):
        for code, name in STREET_VIEW_COUNTRIES.items():
            assert to_code(name) == code


class TestDisplayName:
    def test_known_code(self):
        assert display_name("JP") == "Japan"

    def test_other_is_human_readable(self):
        assert display_name(OTHER) == "Other"

    def test_unknown_code_passes_through(self):
        assert display_name("ZZ") == "ZZ"


class TestAuditCoverage:
    def test_splits_matched_from_unmapped(self):
        report = audit_coverage(["France", "Japan", "Atlantis", "Narnia"])
        assert report["matched"] == ["FR", "JP"]
        assert report["unmapped"] == ["Atlantis", "Narnia"]

    def test_unused_lists_classes_with_no_data(self):
        report = audit_coverage(["France"])
        assert "JP" in report["unused"]
        assert "FR" not in report["unused"]

    def test_deduplicates(self):
        report = audit_coverage(["France", "France", "Atlantis", "Atlantis"])
        assert report["matched"] == ["FR"]
        assert report["unmapped"] == ["Atlantis"]

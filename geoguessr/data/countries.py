"""Canonical country list and name reconciliation.

Three datasets spell countries three different ways -- OSV-5M, Natural Earth,
and GeoGuessr-50k -- so ISO 3166-1 alpha-2 codes are the canonical identifier
here and display names are only ever a presentation detail.

The class list implements design decision #4: exactly 100 classes covering
countries with real Google Street View coverage.

``OTHER`` ("XX") is a *filter sentinel*, not a trainable class. :func:`to_code`
returns it for anything outside the list, and the manifest builder drops those
rows. This follows the 2026-08-13 review: OSV-5M is Mapillary-sourced and
covers 119 countries Google does not (22.6% of its rows -- China, Iran,
Ethiopia, Belarus...). Those locations essentially never appear in a GeoGuessr
round, so training a softmax class for them spends capacity on an answer that
can never be correct.

The membership was reconciled against OSV-5M on 2026-08-13; eight micro-
territories with zero rows (Andorra, Monaco, San Marino, Gibraltar, Macau,
Guam, American Samoa, Northern Marianas) were dropped as dead classes. Re-run
``audit_coverage`` when a new dataset lands.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "OTHER",
    "STREET_VIEW_COUNTRIES",
    "class_names",
    "class_index",
    "normalize",
    "to_code",
    "display_name",
    "audit_coverage",
]

OTHER = "XX"

# ISO alpha-2 -> display name. Countries with meaningful Google Street View
# driving coverage, which is the population this model is ever asked about.
STREET_VIEW_COUNTRIES: dict[str, str] = {
    # --- Europe ---
    "AL": "Albania",
    "AT": "Austria",
    "BE": "Belgium",
    "BG": "Bulgaria",
    "HR": "Croatia",
    "CZ": "Czechia",
    "DK": "Denmark",
    "EE": "Estonia",
    "FO": "Faroe Islands",
    "FI": "Finland",
    "FR": "France",
    "DE": "Germany",
    "GR": "Greece",
    "HU": "Hungary",
    "IS": "Iceland",
    "IE": "Ireland",
    "IM": "Isle of Man",
    "IT": "Italy",
    "LV": "Latvia",
    "LT": "Lithuania",
    "LU": "Luxembourg",
    "MT": "Malta",
    "ME": "Montenegro",
    "NL": "Netherlands",
    "MK": "North Macedonia",
    "NO": "Norway",
    "PL": "Poland",
    "PT": "Portugal",
    "RO": "Romania",
    "RU": "Russia",
    "RS": "Serbia",
    "SK": "Slovakia",
    "SI": "Slovenia",
    "ES": "Spain",
    "SE": "Sweden",
    "CH": "Switzerland",
    "UA": "Ukraine",
    "GB": "United Kingdom",
    # --- Asia ---
    "BD": "Bangladesh",
    "BT": "Bhutan",
    "KH": "Cambodia",
    "HK": "Hong Kong",
    "IN": "India",
    "ID": "Indonesia",
    "IL": "Israel",
    "JP": "Japan",
    "JO": "Jordan",
    "KZ": "Kazakhstan",
    "KG": "Kyrgyzstan",
    "LA": "Laos",
    "MY": "Malaysia",
    "MN": "Mongolia",
    "NP": "Nepal",
    "PS": "Palestine",
    "PH": "Philippines",
    "QA": "Qatar",
    "SG": "Singapore",
    "KR": "South Korea",
    "LK": "Sri Lanka",
    "TW": "Taiwan",
    "TH": "Thailand",
    "TR": "Turkey",
    "AE": "United Arab Emirates",
    "VN": "Vietnam",
    # --- Africa ---
    "BW": "Botswana",
    "EG": "Egypt",
    "SZ": "Eswatini",
    "GH": "Ghana",
    "KE": "Kenya",
    "LS": "Lesotho",
    "MG": "Madagascar",
    "NG": "Nigeria",
    "RE": "Reunion",
    "RW": "Rwanda",
    "SN": "Senegal",
    "ZA": "South Africa",
    "TN": "Tunisia",
    "UG": "Uganda",
    # --- Americas ---
    "AR": "Argentina",
    "BO": "Bolivia",
    "BR": "Brazil",
    "CA": "Canada",
    "CL": "Chile",
    "CO": "Colombia",
    "CR": "Costa Rica",
    "CW": "Curacao",
    "DO": "Dominican Republic",
    "EC": "Ecuador",
    "GL": "Greenland",
    "GT": "Guatemala",
    "MX": "Mexico",
    "PA": "Panama",
    "PE": "Peru",
    "PR": "Puerto Rico",
    "US": "United States",
    "UY": "Uruguay",
    # --- Oceania ---
    "AU": "Australia",
    "NZ": "New Zealand",
    "PG": "Papua New Guinea",
    "WS": "Samoa",
}

# Alternate spellings seen across datasets. Keys are normalized forms.
_ALIASES: dict[str, str] = {
    "united states of america": "US",
    "usa": "US",
    "us": "US",
    "u s a": "US",
    "america": "US",
    "united kingdom of great britain and northern ireland": "GB",
    "great britain": "GB",
    "uk": "GB",
    "britain": "GB",
    "england": "GB",
    "scotland": "GB",
    "wales": "GB",
    "northern ireland": "GB",
    "czech republic": "CZ",
    "russian federation": "RU",
    "korea republic of": "KR",
    "republic of korea": "KR",
    "korea south": "KR",
    "s korea": "KR",
    "turkiye": "TR",
    "swaziland": "SZ",
    "macedonia": "MK",
    "the former yugoslav republic of macedonia": "MK",
    "holland": "NL",
    "the netherlands": "NL",
    "viet nam": "VN",
    "lao peoples democratic republic": "LA",
    "lao pdr": "LA",
    "bolivia plurinational state of": "BO",
    "taiwan province of china": "TW",
    "chinese taipei": "TW",
    "palestinian territory": "PS",
    "state of palestine": "PS",
    "west bank": "PS",
    "hong kong sar china": "HK",
    "hong kong s a r": "HK",
    "united arab emirates the": "AE",
    "reunion": "RE",
    "la reunion": "RE",
    "curacao": "CW",
    "republic of ireland": "IE",
    "eire": "IE",
    "kyrgyz republic": "KG",
    "slovak republic": "SK",
    "east timor": OTHER,
    "myanmar": OTHER,
    "burma": OTHER,
}


def normalize(name: str) -> str:
    """Fold a country name to a comparison key.

    Strips accents, punctuation, and case so "Côte d'Ivoire", "Cote d Ivoire",
    and "COTE DIVOIRE" all collapse to the same key.
    """
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    lowered = ascii_only.lower().strip()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", lowered)).strip()


# Built once: normalized display name -> code.
_BY_NAME: dict[str, str] = {
    normalize(name): code for code, name in STREET_VIEW_COUNTRIES.items()
}


def to_code(name_or_code: str, *, default: str = OTHER) -> str:
    """Resolve any spelling (or an alpha-2 code) to a canonical class code.

    Anything not in the Street View list resolves to ``OTHER`` rather than
    raising, which is what keeps the label space closed per decision #4.
    """
    if not name_or_code:
        return default

    raw = name_or_code.strip()
    if len(raw) == 2 and raw.upper() in STREET_VIEW_COUNTRIES:
        return raw.upper()

    key = normalize(raw)
    if key in _BY_NAME:
        return _BY_NAME[key]
    if key in _ALIASES:
        return _ALIASES[key]
    return default


def display_name(code: str) -> str:
    """Human-readable name for a class code."""
    if code == OTHER:
        return "Other"
    return STREET_VIEW_COUNTRIES.get(code.upper(), code)


def class_names(*, include_other: bool = False) -> list[str]:
    """Ordered class codes. Index in this list is the model's class index.

    Sorted so the ordering is stable across runs and machines. ``OTHER`` is
    excluded by default because it is a filter sentinel rather than a class;
    pass ``include_other=True`` only if you deliberately want a rejection
    class, in which case it is pinned last so adding a country never renumbers
    it.
    """
    codes = sorted(STREET_VIEW_COUNTRIES)
    return [*codes, OTHER] if include_other else codes


def class_index(*, include_other: bool = False) -> dict[str, int]:
    """Class code -> integer index, the inverse of :func:`class_names`."""
    return {code: i for i, code in enumerate(class_names(include_other=include_other))}


def audit_coverage(dataset_names: list[str]) -> dict[str, list[str]]:
    """Compare a dataset's country names against this class list.

    Run this once GeoGuessr-50k and OSV-5M are on disk to reconcile decision #4:
    ``unmapped`` names are falling into the OTHER bucket and may deserve a class
    or an alias; ``unused`` classes have no data behind them at all.
    """
    seen: set[str] = set()
    unmapped: list[str] = []

    for name in dataset_names:
        code = to_code(name)
        if code == OTHER:
            unmapped.append(name)
        else:
            seen.add(code)

    return {
        "unmapped": sorted(set(unmapped)),
        "unused": sorted(set(STREET_VIEW_COUNTRIES) - seen),
        "matched": sorted(seen),
    }

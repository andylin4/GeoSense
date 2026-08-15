"""Tests for the crop specification."""

from __future__ import annotations

import pytest
from PIL import Image

from geoguessr.data.crop import (
    CROP_PRESETS,
    FULL_FRAME,
    GEOGUESSR_16_9,
    CropSpec,
    get_preset,
)


@pytest.fixture
def frame():
    return Image.new("RGB", (1920, 1080), "green")


class TestValidation:
    @pytest.mark.parametrize("bad", [-0.1, 1.5])
    def test_fractions_must_be_in_range(self, bad):
        with pytest.raises(ValueError, match="fraction in"):
            CropSpec(name="bad", left=bad)

    def test_left_must_precede_right(self):
        with pytest.raises(ValueError, match="must be <"):
            CropSpec(name="bad", left=0.8, right=0.2)

    def test_top_must_precede_bottom(self):
        with pytest.raises(ValueError, match="must be <"):
            CropSpec(name="bad", top=0.9, bottom=0.1)

    def test_degenerate_rectangle_rejected(self):
        with pytest.raises(ValueError, match="must be <"):
            CropSpec(name="bad", top=0.5, bottom=0.5)


class TestGeometry:
    def test_box_scales_to_resolution(self):
        spec = CropSpec(name="half", top=0.0, bottom=0.5)
        assert spec.box(1920, 1080) == (0, 0, 1920, 540)
        assert spec.box(1280, 720) == (0, 0, 1280, 360)

    def test_same_spec_gives_proportional_boxes(self):
        # The point of fractional coords: one spec, any display size. Rounding
        # happens independently per resolution, so allow a single pixel.
        small = GEOGUESSR_16_9.box(1920, 1080)
        large = GEOGUESSR_16_9.box(3840, 2160)
        for got, expected in zip(large, (v * 2 for v in small)):
            assert abs(got - expected) <= 1

    def test_kept_area(self):
        assert CropSpec(name="q", right=0.5, bottom=0.5).kept_area == pytest.approx(0.25)
        assert FULL_FRAME.kept_area == pytest.approx(1.0)

    def test_apply_produces_expected_size(self, frame):
        cropped = CropSpec(name="mid", top=0.25, bottom=0.75).apply(frame)
        assert cropped.size == (1920, 540)

    def test_apply_removes_the_intended_region(self):
        # Red top strip, green below. Cropping the strip must remove all red.
        img = Image.new("RGB", (100, 100), "green")
        img.paste(Image.new("RGB", (100, 10), "red"), (0, 0))

        cropped = CropSpec(name="cut_top", top=0.10).apply(img)
        assert cropped.size == (100, 90)
        assert (255, 0, 0) not in cropped.convert("RGB").getcolors()


class TestIdentity:
    def test_full_frame_is_identity(self):
        assert FULL_FRAME.is_identity

    def test_identity_returns_the_same_object(self, frame):
        # Cheap enough to matter when embedding millions of training images.
        assert FULL_FRAME.apply(frame) is frame

    def test_geoguessr_preset_is_not_identity(self):
        assert not GEOGUESSR_16_9.is_identity


class TestFingerprintAndSerialization:
    def test_fingerprint_changes_with_geometry(self):
        a = CropSpec(name="x", top=0.06)
        b = CropSpec(name="x", top=0.07)
        assert a.fingerprint() != b.fingerprint()

    def test_fingerprint_is_stable(self):
        assert GEOGUESSR_16_9.fingerprint() == GEOGUESSR_16_9.fingerprint()

    def test_json_round_trip(self):
        restored = CropSpec.from_json(GEOGUESSR_16_9.to_json())
        assert restored == GEOGUESSR_16_9


class TestPresets:
    def test_geoguessr_preset_cuts_top_and_bottom_only(self):
        # Full width is deliberate: HUD spans both bottom corners.
        assert GEOGUESSR_16_9.left == 0.0
        assert GEOGUESSR_16_9.right == 1.0
        assert GEOGUESSR_16_9.top > 0.0
        assert GEOGUESSR_16_9.bottom < 1.0

    def test_geoguessr_preset_keeps_most_of_the_scene(self):
        assert 0.3 < GEOGUESSR_16_9.kept_area < 0.6

    def test_preset_is_flagged_verified(self):
        # Guard against a future regression being quietly treated as validated.
        assert "Verified" in GEOGUESSR_16_9.note
        assert "NOT re-validated against a live" in GEOGUESSR_16_9.note

    def test_lookup(self):
        assert get_preset("full_frame") is FULL_FRAME
        assert set(CROP_PRESETS) == {"full_frame", "geoguessr_16_9"}

    def test_unknown_preset_lists_options(self):
        with pytest.raises(KeyError, match="geoguessr_16_9"):
            get_preset("nope")


class TestPreview:
    def test_preview_does_not_mutate_the_original(self, frame):
        before = frame.tobytes()
        GEOGUESSR_16_9.preview(frame)
        assert frame.tobytes() == before

    def test_preview_returns_same_size_image(self, frame):
        assert GEOGUESSR_16_9.preview(frame).size == frame.size

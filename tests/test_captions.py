"""Unit tests for the deterministic caption compiler (no models, no CUDA)."""

from __future__ import annotations

import pytest

from media_pipeline.captions import (
    AlignedToken,
    AlignmentError,
    AlignmentMismatchError,
    Caption,
    build_captions,
    compile_srt,
    format_timestamp,
    parse_alignment,
    render_srt,
)


def _tokens(*entries: tuple[str, float, float]) -> list[AlignedToken]:
    return parse_alignment(
        [{"text": text, "start": start, "end": end} for text, start, end in entries]
    )


# --- timestamps ------------------------------------------------------------


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "00:00:00,000"),
        (1.36, "00:00:01,360"),
        (7.92, "00:00:07,920"),
        (61.5, "00:01:01,500"),
        (3661.5, "01:01:01,500"),
        (359999.999, "99:59:59,999"),
        (360000.0, "100:00:00,000"),
    ],
)
def test_format_timestamp(seconds: float, expected: str) -> None:
    assert format_timestamp(seconds) == expected


def test_format_timestamp_rounds_half_up() -> None:
    assert format_timestamp(0.0005) == "00:00:00,001"
    assert format_timestamp(1.2345) == "00:00:01,235"


def test_format_timestamp_rejects_negative() -> None:
    with pytest.raises(ValueError):
        format_timestamp(-0.001)


# --- parsing / validation --------------------------------------------------


def test_parse_alignment_normalizes_valid_items() -> None:
    tokens = parse_alignment([{"text": " 你 ", "start": 0, "end": 0.5}])
    assert tokens == [AlignedToken(text="你", start=0.0, end=0.5)]


def test_parse_alignment_drops_blank_tokens() -> None:
    assert parse_alignment([{"text": "   ", "start": 0, "end": 0.5}]) == []


@pytest.mark.parametrize(
    ("item", "match"),
    [
        ({"text": "a", "start": 0}, "missing field 'end'"),
        ("a", "must be an object"),
        ({"text": 1, "start": 0, "end": 1}, "must be a string"),
        ({"text": "a", "start": True, "end": 1}, "must be a number"),
        ({"text": "a", "start": 0, "end": float("inf")}, "must be finite"),
        ({"text": "a", "start": 0, "end": float("nan")}, "must be finite"),
        ({"text": "a", "start": -1, "end": 0}, "starts before zero"),
        ({"text": "a", "start": 1, "end": 0.5}, "ends before it starts"),
    ],
)
def test_parse_alignment_rejects_malformed_items(item: object, match: str) -> None:
    with pytest.raises(AlignmentError, match=match):
        parse_alignment([item])


def test_parse_alignment_rejects_overlapping_tokens() -> None:
    with pytest.raises(AlignmentError, match="before the previous item ends"):
        parse_alignment(
            [
                {"text": "a", "start": 0.0, "end": 1.0},
                {"text": "b", "start": 0.5, "end": 1.5},
            ]
        )


def test_parse_alignment_allows_touching_tokens() -> None:
    tokens = parse_alignment(
        [
            {"text": "a", "start": 0.0, "end": 0.5},
            {"text": "b", "start": 0.5, "end": 1.0},
        ]
    )
    assert len(tokens) == 2


# --- text/alignment matching ----------------------------------------------


def test_punctuation_is_ignored_when_matching() -> None:
    captions = build_captions("你好。", _tokens(("你", 0.0, 0.5), ("好", 0.5, 1.0)))
    assert captions == [Caption(start=0.0, end=1.0, text="你好。")]


def test_untimed_leading_text_attaches_to_next_caption() -> None:
    captions = build_captions("「你好」", _tokens(("你", 0.0, 0.5), ("好", 0.5, 1.0)))
    assert captions == [Caption(start=0.0, end=1.0, text="「你好」")]


def test_multi_character_token_spans_original_characters() -> None:
    captions = build_captions("hello", _tokens(("hello", 0.0, 0.5)))
    assert captions == [Caption(start=0.0, end=0.5, text="hello")]


def test_mismatched_alignment_raises() -> None:
    with pytest.raises(AlignmentMismatchError, match="does not match"):
        build_captions("你", _tokens(("他", 0.0, 0.5)))


def test_leftover_alignment_token_raises() -> None:
    with pytest.raises(AlignmentMismatchError, match="not present"):
        build_captions("你", _tokens(("你", 0.0, 0.5), ("好", 0.5, 1.0)))


def test_content_without_alignment_raises() -> None:
    with pytest.raises(AlignmentMismatchError, match="has no aligned token"):
        build_captions("你好", _tokens(("你", 0.0, 0.5)))


def test_partially_consumed_token_raises() -> None:
    with pytest.raises(AlignmentMismatchError, match="only partially"):
        build_captions("你", _tokens(("你好", 0.0, 1.0)))


def test_punctuation_only_text_produces_no_captions() -> None:
    assert build_captions("。……", []) == []


# --- segmentation ----------------------------------------------------------


def test_sentence_punctuation_splits_captions() -> None:
    captions = build_captions(
        "你好。再见。",
        _tokens(("你", 0.0, 0.5), ("好", 0.5, 1.0), ("再", 1.5, 2.0), ("见", 2.0, 2.5)),
    )
    assert captions == [
        Caption(start=0.0, end=1.0, text="你好。"),
        Caption(start=1.5, end=2.5, text="再见。"),
    ]


def test_newline_is_a_hard_break() -> None:
    captions = build_captions("你\n好", _tokens(("你", 0.0, 0.5), ("好", 0.5, 1.0)))
    assert [caption.text for caption in captions] == ["你", "好"]


def test_crlf_is_a_single_hard_break() -> None:
    captions = build_captions("你\r\n好", _tokens(("你", 0.0, 0.5), ("好", 0.5, 1.0)))
    assert [caption.text for caption in captions] == ["你", "好"]


def test_max_chars_splits_without_orphaning_punctuation() -> None:
    captions = build_captions(
        "你好，世界",
        _tokens(("你", 0.0, 0.5), ("好", 0.5, 1.0), ("世", 1.0, 1.5), ("界", 1.5, 2.0)),
        max_chars=2,
    )
    assert captions == [
        Caption(start=0.0, end=1.0, text="你好，"),
        Caption(start=1.0, end=2.0, text="世界"),
    ]


def test_max_duration_splits_long_captions() -> None:
    captions = build_captions(
        "abcd",
        _tokens(("a", 0.0, 0.4), ("b", 0.4, 0.8), ("c", 0.8, 1.2), ("d", 1.2, 1.6)),
        max_duration=1.0,
    )
    assert captions == [
        Caption(start=0.0, end=0.8, text="ab"),
        Caption(start=0.8, end=1.6, text="cd"),
    ]


def test_break_after_can_be_overridden() -> None:
    tokens = _tokens(("你", 0.0, 0.5), ("好", 0.5, 1.0), ("再", 1.0, 1.5), ("见", 1.5, 2.0))
    captions = build_captions("你好，再见。", tokens, break_after="，")
    assert [caption.text for caption in captions] == ["你好，", "再见。"]


def test_caption_text_preserves_original_text() -> None:
    original = "真正限制本地人工智能模型使用体验的，\n往往并不只是模型本身的参数规模。"
    content = [char for char in original if char not in "，。\n"]
    tokens = [
        AlignedToken(text=char, start=index * 0.1, end=(index + 1) * 0.1)
        for index, char in enumerate(content)
    ]
    captions = build_captions(original, tokens)
    joined = "".join(caption.text for caption in captions)
    assert "".join(joined.split()) == "".join(original.split())


def test_invalid_options_are_rejected() -> None:
    with pytest.raises(ValueError, match="max_chars"):
        build_captions("a", [], max_chars=0)
    with pytest.raises(ValueError, match="max_duration"):
        build_captions("a", [], max_duration=0)


# --- rendering -------------------------------------------------------------


def test_render_srt_is_standard() -> None:
    captions = [
        Caption(start=1.36, end=4.88, text="第一句，"),
        Caption(start=5.2, end=7.92, text="第二句。"),
    ]
    assert render_srt(captions) == (
        "1\n"
        "00:00:01,360 --> 00:00:04,880\n"
        "第一句，\n"
        "\n"
        "2\n"
        "00:00:05,200 --> 00:00:07,920\n"
        "第二句。\n"
    )


def test_render_srt_without_captions_is_empty() -> None:
    assert render_srt([]) == ""


def test_compile_srt_composes_build_and_render() -> None:
    srt = compile_srt("你好。", _tokens(("你", 0.0, 0.5), ("好", 0.5, 1.0)))
    assert srt == "1\n00:00:00,000 --> 00:00:01,000\n你好。\n"


# --- trailing closing punctuation ------------------------------------------


def test_closing_quote_stays_with_the_sentence_it_closes() -> None:
    captions = build_captions(
        "他说：“你好。”再见。",
        _tokens(
            ("他", 0.0, 0.1),
            ("说", 0.1, 0.2),
            ("你", 0.2, 0.3),
            ("好", 0.3, 0.4),
            ("再", 0.4, 0.5),
            ("见", 0.5, 0.6),
        ),
    )
    assert [caption.text for caption in captions] == ["他说：“你好。”", "再见。"]


def test_closing_bracket_stays_with_the_sentence_it_closes() -> None:
    captions = build_captions(
        "（第一句。）第二句。",
        _tokens(
            ("第", 0.0, 0.1),
            ("一", 0.1, 0.2),
            ("句", 0.2, 0.3),
            ("第", 0.3, 0.4),
            ("二", 0.4, 0.5),
            ("句", 0.5, 0.6),
        ),
    )
    assert [caption.text for caption in captions] == ["（第一句。）", "第二句。"]


def test_repeated_sentence_marks_stay_together() -> None:
    captions = build_captions(
        "真的吗？！然后呢。",
        _tokens(
            ("真", 0.0, 0.1),
            ("的", 0.1, 0.2),
            ("吗", 0.2, 0.3),
            ("然", 0.3, 0.4),
            ("后", 0.4, 0.5),
            ("呢", 0.5, 0.6),
        ),
    )
    assert [caption.text for caption in captions] == ["真的吗？！", "然后呢。"]


def test_closing_quote_before_newline_stays_attached() -> None:
    captions = build_captions(
        "他说：“你好。”\n再见。",
        _tokens(
            ("他", 0.0, 0.1),
            ("说", 0.1, 0.2),
            ("你", 0.2, 0.3),
            ("好", 0.3, 0.4),
            ("再", 0.4, 0.5),
            ("见", 0.5, 0.6),
        ),
    )
    assert [caption.text for caption in captions] == ["他说：“你好。”", "再见。"]


def test_opening_quote_starts_the_next_caption() -> None:
    captions = build_captions(
        "他说。“你好”。",
        _tokens(
            ("他", 0.0, 0.1),
            ("说", 0.1, 0.2),
            ("你", 0.2, 0.3),
            ("好", 0.3, 0.4),
        ),
    )
    assert [caption.text for caption in captions] == ["他说。", "“你好”。"]


# --- pause-aware breaks ----------------------------------------------------


def test_pause_gap_splits_without_punctuation_or_newline() -> None:
    captions = build_captions(
        "abcd",
        _tokens(("a", 0.0, 0.5), ("b", 0.5, 1.0), ("c", 1.4, 1.9), ("d", 1.9, 2.4)),
    )
    assert captions == [
        Caption(start=0.0, end=1.0, text="ab"),
        Caption(start=1.4, end=2.4, text="cd"),
    ]


def test_pause_threshold_is_configurable() -> None:
    tokens = _tokens(("a", 0.0, 0.5), ("b", 0.5, 1.0), ("c", 1.4, 1.9), ("d", 1.9, 2.4))
    captions = build_captions("abcd", tokens, pause_threshold=0.5)
    assert captions == [Caption(start=0.0, end=2.4, text="abcd")]


def test_invalid_pause_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="pause_threshold"):
        build_captions("a", [], pause_threshold=-0.1)


# --- natural candidate breakpoints when a budget is exceeded ----------------


def test_budget_prefers_recent_soft_breakpoint() -> None:
    captions = build_captions(
        "一二三，四五六七八",
        _tokens(
            ("一", 0.0, 0.1),
            ("二", 0.1, 0.2),
            ("三", 0.2, 0.3),
            ("四", 0.3, 0.4),
            ("五", 0.4, 0.5),
            ("六", 0.5, 0.6),
            ("七", 0.6, 0.7),
            ("八", 0.7, 0.8),
        ),
        max_chars=6,
    )
    assert [caption.text for caption in captions] == ["一二三，", "四五六七八"]


def test_budget_prefers_recent_pause_candidate() -> None:
    captions = build_captions(
        "abcde",
        _tokens(
            ("a", 0.0, 0.4),
            ("b", 0.4, 0.8),
            ("c", 0.9, 1.3),
            ("d", 1.3, 1.7),
            ("e", 1.7, 2.1),
        ),
        max_chars=3,
    )
    assert [caption.text for caption in captions] == ["ab", "cde"]


def test_duration_budget_prefers_candidate_breakpoint() -> None:
    captions = build_captions(
        "ab，cde",
        _tokens(
            ("a", 0.0, 0.5),
            ("b", 0.5, 1.0),
            ("c", 1.0, 1.5),
            ("d", 1.5, 2.0),
            ("e", 2.0, 2.5),
        ),
        max_duration=1.6,
    )
    assert captions == [
        Caption(start=0.0, end=1.0, text="ab，"),
        Caption(start=1.0, end=2.5, text="cde"),
    ]


def test_budget_hard_cuts_when_no_candidate_exists() -> None:
    captions = build_captions(
        "一二三四五六七八",
        _tokens(
            ("一", 0.0, 0.1),
            ("二", 0.1, 0.2),
            ("三", 0.2, 0.3),
            ("四", 0.3, 0.4),
            ("五", 0.4, 0.5),
            ("六", 0.5, 0.6),
            ("七", 0.6, 0.7),
            ("八", 0.7, 0.8),
        ),
        max_chars=6,
    )
    assert [caption.text for caption in captions] == ["一二三四五六", "七八"]

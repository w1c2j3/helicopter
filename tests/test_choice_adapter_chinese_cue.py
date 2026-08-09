from helicopter_cli.lighteval_answer_adapters import extract_choice_answer


def test_choice_adapter_extracts_chinese_answer_cue() -> None:
    assert extract_choice_answer("答案：(C)") == " C"
    assert extract_choice_answer("推理\n最终答案：D") == " D"

"""parse/dialects.py: aliases, stub dialects, text-call routing."""
import pytest

from localbot.config import get_profile
from localbot.parse import dialects
from localbot.parse.dialects import canonical_name, parse_text_calls, unparsed_dialect

CORE4 = get_profile("core4").tools.names
LAB6 = get_profile("parity_lab6").tools.names


@pytest.mark.parametrize("alias,expected", [
    ("read_file", "read"),
    ("write_file", "write"),
    ("patch", "edit"),
    ("terminal", "bash"),
    ("bash_exec", "bash"),
])
def test_aliases_map_onto_core4(alias, expected):
    assert canonical_name(alias, CORE4) == expected


def test_core4_names_unchanged():
    for name in CORE4:
        assert canonical_name(name, CORE4) == name


def test_parity_keeps_its_own_tool_names():
    """bash_exec IS a lab tool: parity must not rename it to bash."""
    for name in LAB6:
        assert canonical_name(name, LAB6) == name
    assert canonical_name("search_files", LAB6) == "grep"
    assert canonical_name("read_file", LAB6) == "read"


def test_alias_to_missing_tool_is_left_alone():
    """core4 has no grep tool: search_files stays unknown so the loop reports it."""
    assert canonical_name("search_files", CORE4) == "search_files"
    assert canonical_name("made_up", CORE4) == "made_up"


def test_dialect_registry_is_scaffold_only():
    assert {d.name for d in dialects.DIALECTS} == {"gemma", "mistral", "llama", "deepseek"}
    assert not any(d.implemented for d in dialects.DIALECTS)


@pytest.mark.parametrize("text,name", [
    ('```tool_code\nread(path="notes.txt")\n```', "gemma"),
    ('[TOOL_CALLS] [{"name": "read", "arguments": {"path": "a"}}]', "mistral"),
    ('<|python_tag|>{"name": "read", "parameters": {"path": "a"}}', "llama"),
    ('<｜DSML｜tool invoke="read">', "deepseek"),
])
def test_stub_dialects_detect_but_do_not_parse(text, name):
    assert unparsed_dialect(text) == name
    d = next(d for d in dialects.DIALECTS if d.name == name)
    assert d.detect(text) and d.extract(text) == []


def test_plain_text_is_no_dialect():
    assert unparsed_dialect("All done, the file is updated.") is None
    assert unparsed_dialect("") is None
    assert parse_text_calls("All done.") == ([], None)


def test_text_calls_go_through_vendored_parser():
    calls, source = parse_text_calls('<tool_call>\n{"name": "read", "arguments": {"path": "a.py"}}\n</tool_call>')
    assert source == "fn_call"
    assert [(c.name, c.arguments) for c in calls] == [("read", {"path": "a.py"})]


def test_qwen_xml_with_think_block():
    text = "<think>look first</think>\n<tool_call>\n<function=read>\n<parameter=path>a.py</parameter>\n</function>\n</tool_call>"
    calls, source = parse_text_calls(text)
    assert source == "fn_call" and calls[0].name == "read" and calls[0].arguments == {"path": "a.py"}


@pytest.mark.parametrize("text,expected", [
    ('{"name": "Alice", "arguments": ["late", "rain"]}', False),  # an answer, not a call (L-7)
    ('```json\n{"name": "my-app", "parameters": {"debug": true}}\n```', False),
    ('{"name": "read", "arguments": {"path": "a.txt"}}', True),  # names a core4 tool
    ('{"name": "read_file", "arguments": {"path": "a.txt"}}', True),  # an alias of one
    ('<tool_call>{"name": "Alice", "arguments": {}}</tool_call>', True),  # taught syntax: always
    ('```tool_code\nlist_files(path=".")\n```', True),
    ("All done.", False),
])
def test_call_attempt_needs_a_known_tool_for_bare_json(text, expected):
    assert dialects.call_attempt(text, CORE4) is expected


C4_SCHEMAS = list(get_profile("core4").tools.schemas)


def _qwen_write(content: str) -> str:
    return ("<tool_call>\n<function=write>\n<parameter=path>n.json</parameter>\n"
            f"<parameter=content>{content}</parameter>\n</function>\n</tool_call>")


@pytest.mark.parametrize("raw", ["21", '{"a": 1}', "true", '"quoted"', "1e3", "[1, 2]", "null"])
def test_l5_qwen_xml_keeps_raw_strings_for_string_params(raw):
    """L-5: the vendored parser JSON-decodes every <parameter=…> value (21 -> int 21), and the arg
    gate then refuses a write whose content is not a string. With the schemas, a string-typed
    parameter keeps the raw text exactly."""
    calls, source = parse_text_calls(_qwen_write(raw), C4_SCHEMAS)
    assert source == "fn_call" and calls[0].arguments == {"path": "n.json", "content": raw}


def test_l5_non_string_params_and_no_schemas_keep_the_vendored_coercion():
    tools = [{"type": "function", "function": {"name": "sleep", "parameters": {
        "type": "object", "properties": {"seconds": {"type": "integer"}, "why": {"type": "string"}}}}}]
    text = ("<tool_call>\n<function=sleep>\n<parameter=seconds>5</parameter>\n"
            "<parameter=why>7</parameter>\n</function>\n</tool_call>")
    assert parse_text_calls(text, tools)[0][0].arguments == {"seconds": 5, "why": "7"}
    assert parse_text_calls(_qwen_write("21"))[0][0].arguments["content"] == 21  # vendored, unchanged


def test_l5_alias_names_use_the_target_tools_schema():
    text = ("<tool_call>\n<function=write_file>\n<parameter=path>n.txt</parameter>\n"
            "<parameter=content>42</parameter>\n</function>\n</tool_call>")
    assert parse_text_calls(text, C4_SCHEMAS)[0][0].arguments == {"path": "n.txt", "content": "42"}


def test_gemma_note_records_probe():
    gemma = next(d for d in dialects.DIALECTS if d.name == "gemma")
    assert "native tool_calls" in gemma.notes and "probe_native_tools_20261026T163037Z" in gemma.notes

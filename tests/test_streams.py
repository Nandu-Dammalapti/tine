import pytest

from tine.streams import StreamAssembler, replay, unstreamed, wants_usage

ANTHROPIC_REPLY = {
    "id": "msg_1", "type": "message", "role": "assistant", "model": "cheap",
    "content": [{"type": "text", "text": "Answer: B"},
                {"type": "tool_use", "id": "t1", "name": "lookup", "input": {"q": "x"}}],
    "stop_reason": "tool_use", "stop_sequence": None,
    "usage": {"input_tokens": 12, "output_tokens": 7},
}
OPENAI_REPLY = {
    "id": "c1", "object": "chat.completion", "created": 1, "model": "cheap",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Answer: B"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 12, "completion_tokens": 7},
}


def assemble(provider, data, size):
    assembler = StreamAssembler(provider)
    for start in range(0, len(data), size):
        assembler.feed(data[start:start + size])
    return assembler.payload()


@pytest.mark.parametrize("size", [1, 5, 10_000])
def test_an_anthropic_reply_replays_and_assembles_back_to_itself(size):
    assert assemble("anthropic", replay("anthropic", ANTHROPIC_REPLY), size) == ANTHROPIC_REPLY


@pytest.mark.parametrize("size", [1, 5, 10_000])
def test_an_openai_reply_replays_and_assembles_back_to_itself(size):
    data = replay("openai", OPENAI_REPLY, include_usage=True)
    assert data.endswith(b"data: [DONE]\n\n")
    assert assemble("openai", data, size) == OPENAI_REPLY


def test_without_include_usage_no_usage_chunk_is_written():
    data = replay("openai", OPENAI_REPLY)
    assert b'"usage"' not in data
    assert "usage" not in assemble("openai", data, 64)


def test_a_cut_off_stream_or_unknown_block_is_not_assembled_or_replayed():
    data = replay("anthropic", ANTHROPIC_REPLY)
    assert assemble("anthropic", data[: len(data) // 2], 7) is None
    assert replay("anthropic", {**ANTHROPIC_REPLY, "content": [{"type": "image"}]}) is None


def test_the_unstreamed_request_drops_only_the_stream_fields():
    body = {"messages": [], "stream": True, "stream_options": {"include_usage": True}, "max_tokens": 5}
    assert unstreamed(body) == {"messages": [], "max_tokens": 5}
    assert wants_usage(body) and not wants_usage(unstreamed(body))

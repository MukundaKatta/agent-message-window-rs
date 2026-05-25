# agent-message-window

Sliding window of recent LLM conversation turns with paired-protection: never drop a `tool_use` message without also dropping its `tool_result` sibling.

The Anthropic API rejects conversations where a `tool_result` block references a `tool_use` ID that is no longer in the message history. A naive "drop the oldest" window breaks the next request silently. This crate fixes that.

## Usage

```rust
use agent_message_window::MessageWindow;
use serde_json::json;

let mut win = MessageWindow::new(20, true);

win.add(json!({"role": "user", "content": "search for X"})).unwrap();
win.add(json!({
    "role": "assistant",
    "content": [
        {"type": "tool_use", "id": "u1", "name": "search", "input": {"q": "X"}}
    ]
})).unwrap();
win.add(json!({
    "role": "user",
    "content": [
        {"type": "tool_result", "tool_use_id": "u1", "content": "results..."}
    ]
})).unwrap();

// Pass win.messages_owned() to the next API call.
```

## Features

- `max_turns` cap with oldest-first eviction
- Paired protection: evicting a `tool_use` cascades to orphaned `tool_result` messages
- `paired_protect = false` for raw oldest-first eviction
- Zero dependencies beyond `serde_json`

## License

MIT OR Apache-2.0

# agent-message-window

[![CI](https://github.com/MukundaKatta/agent-message-window-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/MukundaKatta/agent-message-window-rs/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/agent-message-window.svg)](https://crates.io/crates/agent-message-window)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

Sliding window of recent LLM conversation turns with **paired-protection**: never
drop a `tool_use` message without also dropping its `tool_result` sibling (and
vice versa).

## Why

The Anthropic Messages API rejects a conversation when a `tool_result` block
references a `tool_use` id that is no longer present in the message history (and
likewise rejects a `tool_use` with no following `tool_result`). A naive
"drop the oldest message" window will eventually split such a pair and the
*next* request fails — often confusingly, because the message that was dropped
is not the one the error points at.

`MessageWindow` keeps the last `max_messages` messages, but when it evicts an
old message that participates in a tool pair, it cascades the eviction to the
paired message so the remaining window is always a valid request body.

## Install

Add it to your `Cargo.toml`:

```toml
[dependencies]
agent-message-window = "0.1"
serde_json = "1"
```

## Usage

```rust
use agent_message_window::MessageWindow;
use serde_json::json;

// Keep at most 20 messages in context.
let mut win = MessageWindow::new(20);

// Plain turns.
win.push(json!({"role": "user", "content": "search for X"}));

// An assistant turn that calls a tool (Anthropic content-block format).
win.push(json!({
    "role": "assistant",
    "content": [
        {"type": "tool_use", "id": "u1", "name": "search", "input": {"q": "X"}}
    ]
}));

// The matching tool result.
win.push(json!({
    "role": "user",
    "content": [
        {"type": "tool_result", "tool_use_id": "u1", "content": "results..."}
    ]
}));

assert_eq!(win.len(), 3);

// Pass the current window straight to the next API call.
let body = win.messages();
assert_eq!(body.len(), 3);
```

When the window overflows, paired messages are evicted together:

```rust
use agent_message_window::MessageWindow;
use serde_json::json;

let tool_use = json!({
    "role": "assistant",
    "content": [{"type": "tool_use", "id": "u1", "name": "search", "input": {}}]
});
let tool_result = json!({
    "role": "user",
    "content": [{"type": "tool_result", "tool_use_id": "u1", "content": "ok"}]
});

let mut win = MessageWindow::new(3);
win.push(tool_use);                                       // index 0
win.push(tool_result);                                   // index 1
win.push(json!({"role": "user", "content": "next"}));    // index 2
win.push(json!({"role": "user", "content": "again"}));   // overflow -> trim

// The oldest message is the `tool_use`. Evicting it would orphan the
// `tool_result`, so both are dropped together. No orphan is ever left behind.
assert_eq!(win.len(), 2);
for m in win.messages() {
    assert_eq!(m["role"], json!("user"));
}
```

## API

| Method | Description |
| --- | --- |
| `MessageWindow::new(max_messages)` | Create a window capped at `max_messages` (clamped to a minimum of 1). |
| `push(msg)` | Append a message and trim the window, keeping tool pairs intact. |
| `messages() -> &[Value]` | Borrow the current messages, ready to send to the API. |
| `reset(messages)` | Replace all messages, then trim to `max_messages`. |
| `clear()` | Remove every message. |
| `len() -> usize` | Number of messages currently held. |
| `is_empty() -> bool` | `true` when the window holds no messages. |

### Supported message shapes

`MessageWindow` recognises the tool pairing used by the Anthropic Messages API:

- **`tool_use`** — an `assistant` message whose `content` array contains a block
  with `{"type": "tool_use", "id": "..."}`.
- **`tool_result`** — a `user` message whose `content` array contains a block
  with `{"type": "tool_result", "tool_use_id": "..."}`, **or** a message with
  `{"role": "tool", "tool_use_id": "..."}` (the OpenAI-style flat form).

Messages with a plain string `content` (ordinary user/assistant turns) are
treated as unpaired and evicted oldest-first.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

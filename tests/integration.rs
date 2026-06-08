//! Integration tests exercising the public `MessageWindow` API the way a
//! consumer would, including the no-orphan invariant that the crate exists to
//! guarantee.

use agent_message_window::MessageWindow;
use serde_json::{json, Value};
use std::collections::HashSet;

fn user(content: &str) -> Value {
    json!({ "role": "user", "content": content })
}

fn tool_use(id: &str) -> Value {
    json!({
        "role": "assistant",
        "content": [{ "type": "tool_use", "id": id, "name": "search", "input": {} }]
    })
}

/// Anthropic content-block form of a tool result.
fn tool_result_block(id: &str) -> Value {
    json!({
        "role": "user",
        "content": [{ "type": "tool_result", "tool_use_id": id, "content": "ok" }]
    })
}

/// OpenAI-style flat form: a top-level `role: "tool"` message.
fn tool_result_flat(id: &str) -> Value {
    json!({ "role": "tool", "tool_use_id": id, "content": "ok" })
}

/// Collect every `tool_use` id present in the window.
fn use_ids(msgs: &[Value]) -> HashSet<String> {
    let mut out = HashSet::new();
    for m in msgs {
        if let Some(arr) = m.get("content").and_then(|v| v.as_array()) {
            for b in arr {
                if b.get("type").and_then(|v| v.as_str()) == Some("tool_use") {
                    if let Some(id) = b.get("id").and_then(|v| v.as_str()) {
                        out.insert(id.to_string());
                    }
                }
            }
        }
    }
    out
}

/// Collect every `tool_use_id` referenced by a tool result (either form).
fn result_ids(msgs: &[Value]) -> HashSet<String> {
    let mut out = HashSet::new();
    for m in msgs {
        if m.get("role").and_then(|v| v.as_str()) == Some("tool") {
            if let Some(id) = m.get("tool_use_id").and_then(|v| v.as_str()) {
                out.insert(id.to_string());
            }
        }
        if let Some(arr) = m.get("content").and_then(|v| v.as_array()) {
            for b in arr {
                if b.get("type").and_then(|v| v.as_str()) == Some("tool_result") {
                    if let Some(id) = b.get("tool_use_id").and_then(|v| v.as_str()) {
                        out.insert(id.to_string());
                    }
                }
            }
        }
    }
    out
}

/// The core invariant: every tool_result in the window has its tool_use present.
fn assert_no_orphans(msgs: &[Value]) {
    let uses = use_ids(msgs);
    for r in result_ids(msgs) {
        assert!(
            uses.contains(&r),
            "orphan tool_result for id {r:?} left in window: {msgs:?}"
        );
    }
}

#[test]
fn evicting_tool_use_co_evicts_anthropic_result() {
    let mut win = MessageWindow::new(3);
    win.push(tool_use("u1"));
    win.push(tool_result_block("u1"));
    win.push(user("next"));
    win.push(user("again")); // overflow -> trim must not orphan the result

    assert_eq!(win.len(), 2);
    assert_no_orphans(win.messages());
    // Both surviving messages are the plain user turns.
    assert_eq!(win.messages()[0]["content"], json!("next"));
    assert_eq!(win.messages()[1]["content"], json!("again"));
}

#[test]
fn evicting_tool_use_co_evicts_flat_openai_result() {
    let mut win = MessageWindow::new(3);
    win.push(tool_use("u9"));
    win.push(tool_result_flat("u9"));
    win.push(user("next"));
    win.push(user("again"));

    assert_eq!(win.len(), 2);
    assert_no_orphans(win.messages());
}

#[test]
fn many_interleaved_pairs_never_orphan() {
    // Push tool_use / tool_result / plain in a repeating cycle for a range of
    // window sizes and assert the invariant holds at every step.
    for cap in 1..=8 {
        let mut win = MessageWindow::new(cap);
        for i in 0..40u32 {
            match i % 3 {
                0 => win.push(tool_use(&format!("id{i}"))),
                1 => win.push(tool_result_block(&format!("id{}", i - 1))),
                _ => win.push(user(&format!("plain{i}"))),
            }
            assert!(
                win.len() <= cap,
                "window exceeded cap {cap} (len {})",
                win.len()
            );
            assert_no_orphans(win.messages());
        }
    }
}

#[test]
fn reset_then_trim_keeps_invariant() {
    let mut win = MessageWindow::new(2);
    win.reset(vec![
        tool_use("r1"),
        tool_result_block("r1"),
        user("a"),
        user("b"),
        user("c"),
    ]);
    assert_eq!(win.len(), 2);
    assert_no_orphans(win.messages());
    assert_eq!(win.messages()[0]["content"], json!("b"));
    assert_eq!(win.messages()[1]["content"], json!("c"));
}

#[test]
fn clear_resets_window() {
    let mut win = MessageWindow::new(5);
    win.push(user("a"));
    win.push(user("b"));
    assert!(!win.is_empty());
    win.clear();
    assert!(win.is_empty());
    assert_eq!(win.len(), 0);
}

#[test]
fn zero_cap_is_clamped_to_one() {
    let mut win = MessageWindow::new(0);
    win.push(user("a"));
    win.push(user("b"));
    assert_eq!(win.len(), 1);
    assert_eq!(win.messages()[0]["content"], json!("b"));
}

#[test]
fn unpaired_messages_evict_oldest_first() {
    let mut win = MessageWindow::new(3);
    for i in 0..6u32 {
        win.push(user(&i.to_string()));
    }
    assert_eq!(win.len(), 3);
    assert_eq!(win.messages()[0]["content"], json!("3"));
    assert_eq!(win.messages()[2]["content"], json!("5"));
}

/*!
agent-message-window: sliding message window with paired tool-call protection.

Keeps the last `max_messages` messages for an LLM context window. When
evicting old messages, paired `tool_use`/`tool_result` blocks are never
separated — if a `tool_use` would be dropped, its paired `tool_result`
is dropped too (and vice versa).

```rust
use agent_message_window::MessageWindow;
use serde_json::json;

let mut w = MessageWindow::new(4);
w.push(json!({"role": "user", "content": "hello"}));
w.push(json!({"role": "assistant", "content": "hi"}));
assert_eq!(w.len(), 2);
```
*/

use serde_json::Value;

fn role(msg: &Value) -> Option<&str> {
    msg.get("role").and_then(|v| v.as_str())
}

/// Returns the tool_use_id if the message is a tool_result block (or
/// contains one in a content array).
fn tool_use_ids_in_result(msg: &Value) -> Vec<String> {
    let mut ids = Vec::new();
    if role(msg) == Some("tool") {
        if let Some(id) = msg.get("tool_use_id").and_then(|v| v.as_str()) {
            ids.push(id.to_owned());
        }
    }
    // Anthropic content-array format: role=user, content=[{type:tool_result}]
    if role(msg) == Some("user") {
        if let Some(arr) = msg.get("content").and_then(|v| v.as_array()) {
            for block in arr {
                if block.get("type").and_then(|v| v.as_str()) == Some("tool_result") {
                    if let Some(id) = block.get("tool_use_id").and_then(|v| v.as_str()) {
                        ids.push(id.to_owned());
                    }
                }
            }
        }
    }
    ids
}

/// Returns tool_use IDs emitted in this message.
fn tool_use_ids_in_use(msg: &Value) -> Vec<String> {
    let mut ids = Vec::new();
    // Anthropic content-array: role=assistant, content=[{type:tool_use, id:...}]
    if let Some(arr) = msg.get("content").and_then(|v| v.as_array()) {
        for block in arr {
            if block.get("type").and_then(|v| v.as_str()) == Some("tool_use") {
                if let Some(id) = block.get("id").and_then(|v| v.as_str()) {
                    ids.push(id.to_owned());
                }
            }
        }
    }
    ids
}

/// A capped sliding window of LLM messages.
///
/// When the window is full, the oldest messages are evicted while keeping
/// `tool_use`/`tool_result` pairs intact.
pub struct MessageWindow {
    max_messages: usize,
    messages: Vec<Value>,
}

impl MessageWindow {
    /// Create a window capped at `max_messages` entries.
    pub fn new(max_messages: usize) -> Self {
        Self {
            max_messages: max_messages.max(1),
            messages: Vec::new(),
        }
    }

    /// Push a message, evicting old messages as needed.
    pub fn push(&mut self, msg: Value) {
        self.messages.push(msg);
        self.trim();
    }

    /// Current messages.
    pub fn messages(&self) -> &[Value] {
        &self.messages
    }

    /// Number of messages currently in the window.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// True when the window is empty.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Evict from the front until we are at or below `max_messages`,
    /// without splitting any tool_use / tool_result pair.
    fn trim(&mut self) {
        while self.messages.len() > self.max_messages {
            // Collect tool_use IDs that appear in messages[0] and their paired
            // result IDs that would need co-eviction.
            let use_ids = tool_use_ids_in_use(&self.messages[0]);
            let result_ids = tool_use_ids_in_result(&self.messages[0]);

            // Drop messages[0].
            self.messages.remove(0);

            // If the evicted message contained tool_use blocks, also evict
            // any following tool_result messages for those IDs.
            if !use_ids.is_empty() {
                self.messages.retain(|m| {
                    let result_ids_here = tool_use_ids_in_result(m);
                    !result_ids_here.iter().any(|id| use_ids.contains(id))
                });
            }

            // If the evicted message was a tool_result, also evict the
            // tool_use message that emitted those IDs.
            if !result_ids.is_empty() {
                self.messages.retain(|m| {
                    let use_ids_here = tool_use_ids_in_use(m);
                    !use_ids_here.iter().any(|id| result_ids.contains(id))
                });
            }
        }
    }

    /// Replace all messages (ignores max; use push() for capped insertion).
    pub fn reset(&mut self, messages: Vec<Value>) {
        self.messages = messages;
        self.trim();
    }

    /// Remove all messages.
    pub fn clear(&mut self) {
        self.messages.clear();
    }
}

impl std::fmt::Debug for MessageWindow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MessageWindow")
            .field("max_messages", &self.max_messages)
            .field("len", &self.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn user(content: &str) -> Value {
        json!({"role": "user", "content": content})
    }

    fn assistant(content: &str) -> Value {
        json!({"role": "assistant", "content": content})
    }

    fn tool_use_msg(id: &str) -> Value {
        json!({
            "role": "assistant",
            "content": [{"type": "tool_use", "id": id, "name": "search", "input": {}}]
        })
    }

    fn tool_result_msg(id: &str) -> Value {
        json!({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": id, "content": "result"}]
        })
    }

    #[test]
    fn push_under_limit() {
        let mut w = MessageWindow::new(5);
        w.push(user("hi"));
        w.push(assistant("hello"));
        assert_eq!(w.len(), 2);
    }

    #[test]
    fn push_at_limit() {
        let mut w = MessageWindow::new(2);
        w.push(user("1"));
        w.push(user("2"));
        assert_eq!(w.len(), 2);
    }

    #[test]
    fn evicts_oldest_when_full() {
        let mut w = MessageWindow::new(2);
        w.push(user("a"));
        w.push(user("b"));
        w.push(user("c"));
        assert_eq!(w.len(), 2);
        assert_eq!(w.messages()[0]["content"], json!("b"));
        assert_eq!(w.messages()[1]["content"], json!("c"));
    }

    #[test]
    fn is_empty_initially() {
        let w = MessageWindow::new(5);
        assert!(w.is_empty());
    }

    #[test]
    fn not_empty_after_push() {
        let mut w = MessageWindow::new(5);
        w.push(user("hi"));
        assert!(!w.is_empty());
    }

    #[test]
    fn clear_empties_window() {
        let mut w = MessageWindow::new(5);
        w.push(user("hi"));
        w.clear();
        assert!(w.is_empty());
    }

    #[test]
    fn tool_pair_not_split_evict_use_evicts_result_too() {
        let mut w = MessageWindow::new(3);
        // Messages: tool_use, tool_result, user, user
        // After 4th push, oldest 2 (tool_use + tool_result) should both be evicted
        // to avoid leaving an orphan tool_result.
        w.push(tool_use_msg("id1"));
        w.push(tool_result_msg("id1"));
        w.push(user("a"));
        w.push(user("b")); // triggers trim: evicts tool_use, co-evicts tool_result
        // Should end up with just ["a", "b"]
        let msgs = w.messages();
        assert!(!msgs.iter().any(|m| tool_use_ids_in_use(m).contains(&"id1".to_string())));
        assert!(!msgs.iter().any(|m| tool_use_ids_in_result(m).contains(&"id1".to_string())));
    }

    #[test]
    fn reset_applies_trim() {
        let mut w = MessageWindow::new(2);
        w.reset(vec![user("a"), user("b"), user("c")]);
        assert_eq!(w.len(), 2);
    }

    #[test]
    fn max_1_keeps_latest() {
        let mut w = MessageWindow::new(1);
        w.push(user("a"));
        w.push(user("b"));
        assert_eq!(w.len(), 1);
        assert_eq!(w.messages()[0]["content"], json!("b"));
    }

    #[test]
    fn debug_format() {
        let w = MessageWindow::new(5);
        let s = format!("{:?}", w);
        assert!(s.contains("MessageWindow"));
    }

    #[test]
    fn messages_returns_current_slice() {
        let mut w = MessageWindow::new(5);
        w.push(user("hello"));
        assert_eq!(w.messages().len(), 1);
        assert_eq!(w.messages()[0]["content"], json!("hello"));
    }

    #[test]
    fn multiple_pairs_independent() {
        let mut w = MessageWindow::new(4);
        w.push(tool_use_msg("a"));
        w.push(tool_result_msg("a"));
        w.push(tool_use_msg("b"));
        w.push(tool_result_msg("b"));
        assert_eq!(w.len(), 4);
    }

    #[test]
    fn evict_many_simple_messages() {
        let mut w = MessageWindow::new(3);
        for i in 0..10u32 {
            w.push(user(&i.to_string()));
        }
        assert_eq!(w.len(), 3);
        assert_eq!(w.messages()[0]["content"], json!("7"));
        assert_eq!(w.messages()[2]["content"], json!("9"));
    }
}

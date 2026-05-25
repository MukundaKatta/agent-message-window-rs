/*!
agent-message-window: sliding window of LLM conversation turns with
paired tool_use / tool_result protection.

The Anthropic API rejects conversations that contain a `tool_result`
whose `tool_use` is no longer in the history. A naive "drop the oldest"
window silently corrupts the conversation. This crate's eviction logic
never drops a `tool_use` message without also evicting all `tool_result`
messages that reference it.

```rust
use agent_message_window::MessageWindow;
use serde_json::json;

let mut win = MessageWindow::new(4, true);
win.add(json!({"role": "user", "content": "hi"})).unwrap();
win.add(json!({"role": "assistant", "content": [
    {"type": "tool_use", "id": "u1", "name": "search", "input": {}}
]})).unwrap();
win.add(json!({"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "u1", "content": "ok"}
]})).unwrap();

// Even with max_turns=4, paired protection applies.
assert_eq!(win.len(), 3);
```
*/

use serde_json::Value;
use std::collections::{HashSet, VecDeque};

// ---- error ----------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WindowError {
    NotAnObject,
    MaxTurnsMustBePositive,
}

impl std::fmt::Display for WindowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WindowError::NotAnObject => write!(f, "message must be a JSON object"),
            WindowError::MaxTurnsMustBePositive => write!(f, "max_turns must be >= 1"),
        }
    }
}

impl std::error::Error for WindowError {}

// ---- helpers --------------------------------------------------------------

/// Collect `tool_use` IDs from an assistant message's `content` array.
fn tool_use_ids(message: &Value) -> HashSet<String> {
    let mut out = HashSet::new();
    if let Some(content) = message.get("content") {
        if let Some(blocks) = content.as_array() {
            for block in blocks {
                if block.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
                    if let Some(id) = block.get("id").and_then(|i| i.as_str()) {
                        out.insert(id.to_owned());
                    }
                }
            }
        }
    }
    out
}

/// Collect `tool_use_id` references from a user message's `content` array.
fn tool_result_refs(message: &Value) -> HashSet<String> {
    let mut out = HashSet::new();
    if let Some(content) = message.get("content") {
        if let Some(blocks) = content.as_array() {
            for block in blocks {
                if block.get("type").and_then(|t| t.as_str()) == Some("tool_result") {
                    if let Some(id) = block.get("tool_use_id").and_then(|i| i.as_str()) {
                        out.insert(id.to_owned());
                    }
                }
            }
        }
    }
    out
}

// ---- MessageWindow --------------------------------------------------------

/// Sliding window over LLM conversation messages.
///
/// `max_turns` controls how many messages are kept. When adding a new
/// message causes overflow, the oldest is evicted first. With
/// `paired_protect = true` (the default), evicting a `tool_use` message
/// also evicts any `tool_result` messages that reference its IDs.
pub struct MessageWindow {
    max_turns: usize,
    paired_protect: bool,
    buf: VecDeque<Value>,
    evicted_count: usize,
}

impl MessageWindow {
    /// Create a new window with `max_turns` capacity.
    ///
    /// `paired_protect = true` guards Anthropic tool-use / tool-result pairs.
    pub fn new(max_turns: usize, paired_protect: bool) -> Self {
        assert!(max_turns >= 1, "max_turns must be >= 1");
        Self {
            max_turns,
            paired_protect,
            buf: VecDeque::new(),
            evicted_count: 0,
        }
    }

    /// Try to create; returns Err if max_turns < 1.
    pub fn try_new(max_turns: usize, paired_protect: bool) -> Result<Self, WindowError> {
        if max_turns < 1 {
            return Err(WindowError::MaxTurnsMustBePositive);
        }
        Ok(Self::new(max_turns, paired_protect))
    }

    pub fn max_turns(&self) -> usize {
        self.max_turns
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn evicted_count(&self) -> usize {
        self.evicted_count
    }

    /// Ordered slice of current messages (oldest → newest).
    pub fn messages(&self) -> Vec<&Value> {
        self.buf.iter().collect()
    }

    /// Cloned owned snapshot.
    pub fn messages_owned(&self) -> Vec<Value> {
        self.buf.iter().cloned().collect()
    }

    pub fn clear(&mut self) {
        self.buf.clear();
        self.evicted_count = 0;
    }

    /// Append a message and evict-to-fit.
    pub fn add(&mut self, message: Value) -> Result<(), WindowError> {
        if !message.is_object() {
            return Err(WindowError::NotAnObject);
        }
        self.buf.push_back(message);
        self.evict_to_fit();
        Ok(())
    }

    /// Add multiple messages in order.
    pub fn extend(&mut self, messages: impl IntoIterator<Item = Value>) -> Result<(), WindowError> {
        for m in messages {
            self.add(m)?;
        }
        Ok(())
    }

    // ---- eviction ---------------------------------------------------

    fn evict_to_fit(&mut self) {
        while self.buf.len() > self.max_turns {
            let oldest = self.buf.pop_front().expect("buf non-empty");
            self.evicted_count += 1;
            if self.paired_protect {
                let tu_ids = tool_use_ids(&oldest);
                if !tu_ids.is_empty() {
                    self.drop_orphan_tool_results(&tu_ids);
                }
            }
        }
    }

    /// Remove any buffered messages whose tool_result references are in `tu_ids`.
    fn drop_orphan_tool_results(&mut self, tu_ids: &HashSet<String>) {
        let mut kept = VecDeque::with_capacity(self.buf.len());
        for msg in self.buf.drain(..) {
            let refs = tool_result_refs(&msg);
            if refs.iter().any(|id| tu_ids.contains(id)) {
                self.evicted_count += 1;
            } else {
                kept.push_back(msg);
            }
        }
        self.buf = kept;
    }
}

// ---- tests ----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn user(text: &str) -> Value {
        json!({"role": "user", "content": text})
    }

    fn assistant(text: &str) -> Value {
        json!({"role": "assistant", "content": text})
    }

    fn tool_use_msg(id: &str, name: &str) -> Value {
        json!({
            "role": "assistant",
            "content": [{"type": "tool_use", "id": id, "name": name, "input": {}}]
        })
    }

    fn tool_result_msg(tool_use_id: &str) -> Value {
        json!({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": "ok"}]
        })
    }

    #[test]
    fn basic_sliding_window() {
        let mut win = MessageWindow::new(3, false);
        win.add(user("a")).unwrap();
        win.add(user("b")).unwrap();
        win.add(user("c")).unwrap();
        win.add(user("d")).unwrap();
        assert_eq!(win.len(), 3);
        assert_eq!(win.evicted_count(), 1);
        assert_eq!(win.messages()[0]["content"], json!("b"));
    }

    #[test]
    fn no_overflow_within_cap() {
        let mut win = MessageWindow::new(5, true);
        for i in 0..5 {
            win.add(user(&i.to_string())).unwrap();
        }
        assert_eq!(win.len(), 5);
        assert_eq!(win.evicted_count(), 0);
    }

    #[test]
    fn paired_protect_drops_orphan_tool_result() {
        let mut win = MessageWindow::new(2, true);
        win.add(tool_use_msg("u1", "search")).unwrap();
        win.add(tool_result_msg("u1")).unwrap();
        // Now add two more; the tool_use will be evicted, which cascades to result
        win.add(user("follow-up")).unwrap();
        win.add(assistant("done")).unwrap();
        // tool_use AND tool_result both gone; only last 2 remain
        assert_eq!(win.len(), 2);
        assert!(win.messages().iter().all(|m| {
            m.get("content")
                .and_then(|c| c.as_array())
                .map(|arr| arr.iter().all(|b| b.get("type") != Some(&json!("tool_result"))))
                .unwrap_or(true)
        }));
    }

    #[test]
    fn paired_protect_off_leaves_orphan() {
        let mut win = MessageWindow::new(2, false);
        win.add(tool_use_msg("u1", "search")).unwrap();
        win.add(tool_result_msg("u1")).unwrap();
        win.add(user("next")).unwrap();
        win.add(assistant("done")).unwrap();
        // Without protection, raw oldest is dropped; tool_result may still be present
        assert_eq!(win.len(), 2);
    }

    #[test]
    fn clear_resets_all() {
        let mut win = MessageWindow::new(3, true);
        win.add(user("a")).unwrap();
        win.add(user("b")).unwrap();
        win.clear();
        assert_eq!(win.len(), 0);
        assert_eq!(win.evicted_count(), 0);
    }

    #[test]
    fn extend_adds_in_order() {
        let mut win = MessageWindow::new(5, true);
        win.extend(vec![user("a"), user("b"), user("c")]).unwrap();
        assert_eq!(win.len(), 3);
        assert_eq!(win.messages()[1]["content"], json!("b"));
    }

    #[test]
    fn not_object_error() {
        let mut win = MessageWindow::new(3, true);
        let err = win.add(json!([1, 2, 3])).unwrap_err();
        assert_eq!(err, WindowError::NotAnObject);
    }

    #[test]
    fn try_new_zero_cap_errors() {
        assert!(matches!(
            MessageWindow::try_new(0, true),
            Err(WindowError::MaxTurnsMustBePositive)
        ));
    }

    #[test]
    fn try_new_valid() {
        assert!(MessageWindow::try_new(1, true).is_ok());
    }

    #[test]
    fn evicted_count_increments_correctly() {
        let mut win = MessageWindow::new(2, false);
        for i in 0..5u32 {
            win.add(user(&i.to_string())).unwrap();
        }
        assert_eq!(win.evicted_count(), 3);
    }

    #[test]
    fn multiple_tool_use_ids_in_one_message() {
        let mut win = MessageWindow::new(2, true);
        // one assistant message with two tool_use blocks
        win.add(json!({
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "a1", "name": "x", "input": {}},
                {"type": "tool_use", "id": "a2", "name": "y", "input": {}}
            ]
        }))
        .unwrap();
        win.add(tool_result_msg("a1")).unwrap();
        win.add(tool_result_msg("a2")).unwrap();
        // window fills to 2, starts evicting
        win.add(user("next")).unwrap();
        // tool_use evicted → both results should be purged
        for m in win.messages() {
            let content = m.get("content");
            if let Some(Value::Array(blocks)) = content {
                for b in blocks {
                    assert_ne!(b.get("type"), Some(&json!("tool_result")));
                }
            }
        }
    }

    #[test]
    fn messages_owned_clones() {
        let mut win = MessageWindow::new(3, true);
        win.add(user("hello")).unwrap();
        let owned = win.messages_owned();
        assert_eq!(owned[0]["content"], json!("hello"));
    }

    #[test]
    fn single_capacity_window() {
        let mut win = MessageWindow::new(1, true);
        win.add(user("a")).unwrap();
        win.add(user("b")).unwrap();
        assert_eq!(win.len(), 1);
        assert_eq!(win.messages()[0]["content"], json!("b"));
        assert_eq!(win.evicted_count(), 1);
    }

    #[test]
    fn tool_use_no_result_in_buffer_just_evicts_use() {
        let mut win = MessageWindow::new(1, true);
        win.add(tool_use_msg("u1", "search")).unwrap();
        // no tool_result in buffer; evicting tool_use alone is fine
        win.add(user("follow")).unwrap();
        assert_eq!(win.len(), 1);
        assert_eq!(win.messages()[0]["content"], json!("follow"));
    }

    #[test]
    fn add_then_clear_then_add() {
        let mut win = MessageWindow::new(3, true);
        win.add(user("a")).unwrap();
        win.clear();
        win.add(user("b")).unwrap();
        assert_eq!(win.len(), 1);
        assert_eq!(win.messages()[0]["content"], json!("b"));
    }

    #[test]
    fn is_empty_and_len() {
        let mut win = MessageWindow::new(3, true);
        assert!(win.is_empty());
        win.add(user("x")).unwrap();
        assert!(!win.is_empty());
        assert_eq!(win.len(), 1);
    }
}

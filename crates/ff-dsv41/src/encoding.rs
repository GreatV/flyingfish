//! The DeepSeek-V4.1 prompt format.
//!
//! Mirrors `encoding/encoding.py` in the checkpoint for the single-turn
//! text and image forms the first cut supports.

pub const BOS_TOKEN: &str = "<｜begin▁of▁sentence｜>";
pub const EOS_TOKEN: &str = "<｜end▁of▁sentence｜>";
pub const SYSTEM_SP_TOKEN: &str = "<｜System｜>";
pub const USER_SP_TOKEN: &str = "<｜User｜>";
pub const ASSISTANT_SP_TOKEN: &str = "<｜Assistant｜>";
pub const THINKING_START_TOKEN: &str = "<think>";
pub const THINKING_END_TOKEN: &str = "</think>";
pub const IMAGE_PLACEHOLDER: &str = "<｜deepseek_image｜>";

const REASONING_EFFORT_TEMPLATE: &str = "Reasoning Effort: {budget} (range 1-100, the higher the value, the more thorough the reasoning)\n\n";

/// The numeric reasoning budget, rendered only in thinking mode at the start
/// of the conversation. String aliases map per `encoding/README.md`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReasoningEffort {
    Low,
    High,
    Max,
    Budget(u8),
}

impl ReasoningEffort {
    pub fn budget(self) -> u8 {
        match self {
            Self::Low => 50,
            Self::High => 75,
            Self::Max => 100,
            Self::Budget(value) => value,
        }
    }

    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "low" => Ok(Self::Low),
            "high" => Ok(Self::High),
            "max" => Ok(Self::Max),
            _ => value
                .parse::<u8>()
                .ok()
                .filter(|budget| (1..=100).contains(budget))
                .map(Self::Budget)
                .ok_or_else(|| {
                    format!(
                        "invalid reasoning effort {value:?}; use low, high, max, or an integer 1-100"
                    )
                }),
        }
    }
}

/// Encode one turn: optional system message, user content, and the assistant
/// generation header. Chat mode closes the thinking block immediately;
/// thinking mode leaves it open and prefixes the reasoning-effort budget.
pub fn chat_prompt(
    prompt: &str,
    system: Option<&str>,
    thinking: bool,
    effort: ReasoningEffort,
) -> String {
    let mut encoded = String::from(BOS_TOKEN);
    let effort_prefix = if thinking {
        REASONING_EFFORT_TEMPLATE.replace("{budget}", &effort.budget().to_string())
    } else {
        String::new()
    };
    if !effort_prefix.is_empty() || system.is_some() {
        encoded.push_str(SYSTEM_SP_TOKEN);
        encoded.push_str(&effort_prefix);
        if let Some(system) = system {
            encoded.push_str(system);
        }
    }
    encoded.push_str(USER_SP_TOKEN);
    encoded.push_str(prompt);
    encoded.push_str(ASSISTANT_SP_TOKEN);
    encoded.push_str(if thinking {
        THINKING_START_TOKEN
    } else {
        THINKING_END_TOKEN
    });
    encoded
}

/// The same, with one image placeholder block before the prompt text.
pub fn chat_prompt_with_image(
    prompt: &str,
    system: Option<&str>,
    thinking: bool,
    effort: ReasoningEffort,
) -> String {
    let image_block = format!("\n\n{IMAGE_PLACEHOLDER}\n\n");
    chat_prompt(&format!("{image_block}{prompt}"), system, thinking, effort)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_mode_matches_the_reference_encoding() {
        let prompt = chat_prompt("What is 2+2?", None, false, ReasoningEffort::High);
        assert_eq!(
            prompt,
            "<｜begin▁of▁sentence｜><｜User｜>What is 2+2?<｜Assistant｜></think>"
        );
    }

    #[test]
    fn thinking_mode_renders_the_effort_prefix_inside_the_system_block() {
        let prompt = chat_prompt(
            "What is 2+2?",
            Some("You are a helpful assistant."),
            true,
            ReasoningEffort::Budget(75),
        );
        assert_eq!(
            prompt,
            "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 \
             (range 1-100, the higher the value, the more thorough the reasoning)\n\n\
             You are a helpful assistant.<｜User｜>What is 2+2?<｜Assistant｜><think>"
        );
    }

    #[test]
    fn image_prompts_place_the_placeholder_before_the_text() {
        let prompt = chat_prompt_with_image("有什么内容？", None, false, ReasoningEffort::High);
        assert_eq!(
            prompt,
            "<｜begin▁of▁sentence｜><｜User｜>\n\n<｜deepseek_image｜>\n\n有什么内容？\
             <｜Assistant｜></think>"
        );
    }

    #[test]
    fn effort_aliases_and_bounds() {
        assert_eq!(ReasoningEffort::Low.budget(), 50);
        assert_eq!(ReasoningEffort::High.budget(), 75);
        assert_eq!(ReasoningEffort::Max.budget(), 100);
        assert_eq!(
            ReasoningEffort::parse("42").unwrap(),
            ReasoningEffort::Budget(42)
        );
        assert!(ReasoningEffort::parse("0").is_err());
        assert!(ReasoningEffort::parse("101").is_err());
        assert!(ReasoningEffort::parse("turbo").is_err());
    }
}

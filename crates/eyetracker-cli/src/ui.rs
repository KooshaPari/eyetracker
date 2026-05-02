//! UI module
//!
//! Shared UI utilities.

use ratatui::style::{Color, Style};

/// Create a styled text widget
pub fn styled_text(text: &str, color: Color) -> ratatui::widgets::Paragraph<'_> {
    ratatui::widgets::Paragraph::new(text.to_string())
        .style(Style::new().fg(color))
        .alignment(ratatui::layout::Alignment::Center)
}

/// Create a status indicator
pub fn status_indicator(active: bool) -> &'static str {
    if active {
        "●"
    } else {
        "○"
    }
}

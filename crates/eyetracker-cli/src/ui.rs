//! UI module
//!
//! Shared UI utilities.

use ratatui::{
    style::{Color, Style},
    widgets::Widget,
};

/// Create a styled text widget
pub fn styled_text(text: &str, color: Color) -> ratatui::widgets::Paragraph<'static> {
    ratatui::widgets::Paragraph::new(text)
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

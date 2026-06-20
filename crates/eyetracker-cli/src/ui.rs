//! Terminal UI rendering with ratatui

use anyhow::Result;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Gauge, Paragraph, Sparkline};
use ratatui::Terminal;
use std::io::Stdout;
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

/// Data to display on the TUI dashboard
pub struct DashboardData {
    pub fps: f64,
    pub processing_ms: f64,
    pub frame_number: u64,
    pub gaze_vector: Option<String>,
    pub smoothed_gaze: Option<String>,
    pub confidence: String,
    pub face_detected: bool,
    pub resolution: String,
    pub events: String,
}

/// Run the TUI event loop with ratatui
pub fn run_event_loop<T>(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    rx: &Receiver<T>,
    data_fn: impl Fn(&T) -> DashboardData,
    duration_secs: u64,
) -> Result<()> {
    use crossterm::event::{self, Event, KeyCode};

    let start = Instant::now();
    let mut fps_history: Vec<f64> = Vec::with_capacity(60);
    let mut last_frame_time: Vec<f64> = Vec::with_capacity(120);
    let mut frame_count: u64 = 0;

    // Enable raw mode for crossterm
    crossterm::terminal::enable_raw_mode()?;
    let mut result = Ok(());

    loop {
        // Check duration
        if duration_secs > 0 && start.elapsed().as_secs() >= duration_secs {
            break;
        }

        // Check for key press (non-blocking)
        if event::poll(Duration::from_millis(16))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break,
                    KeyCode::Char('r') => {
                        fps_history.clear();
                        last_frame_time.clear();
                    }
                    _ => {}
                }
            }
        }

        // Try to receive latest frame data
        let mut latest_data: Option<DashboardData> = None;
        while let Ok(data) = rx.try_recv() {
            latest_data = Some(data_fn(&data));
            frame_count += 1;
        }

        if let Some(ref data) = latest_data {
            fps_history.push(data.fps);
            if fps_history.len() > 60 {
                fps_history.remove(0);
            }
            last_frame_time.push(data.processing_ms);
            if last_frame_time.len() > 120 {
                last_frame_time.remove(0);
            }
        }

        // Render UI
        let render_result = terminal.draw(|f| {
            let area = f.size();
            draw_dashboard(
                f,
                area,
                &latest_data,
                &fps_history,
                &last_frame_time,
                start.elapsed(),
                frame_count,
            );
        });

        if let Err(e) = render_result {
            result = Err(anyhow::anyhow!("Render error: {}", e));
            break;
        }
    }

    crossterm::terminal::disable_raw_mode()?;
    result
}

/// Draw the dashboard
fn draw_dashboard(
    f: &mut ratatui::Frame,
    area: Rect,
    data: &Option<DashboardData>,
    fps_history: &[f64],
    time_history: &[f64],
    _elapsed: Duration,
    _frame_count: u64,
) {
    // Main layout
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),   // Title
            Constraint::Length(8),   // Stats row
            Constraint::Length(9),   // Gaze visualization
            Constraint::Min(3),      // Sparklines
            Constraint::Length(3),   // Controls help
        ])
        .split(area);

    // Title
    let title = Block::default()
        .title(" Eye Tracker ".bold())
        .title_alignment(Alignment::Center)
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::Cyan));
    f.render_widget(title, chunks[0]);

    // Stats row
    let stats_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ])
        .split(chunks[1]);

    if let Some(ref d) = data {
        // FPS gauge
        let fps_block = Block::default()
            .title(" FPS ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(if d.fps > 30.0 { Color::Green } else { Color::Red }));
        let fps_gauge = Gauge::default()
            .block(fps_block)
            .gauge_style(
                Style::default()
                    .fg(Color::White)
                    .bg(if d.fps > 30.0 { Color::Green } else { Color::Red })
                    .add_modifier(Modifier::BOLD),
            )
            .percent((d.fps.min(120.0) / 120.0 * 100.0) as u16)
            .label(format!("{:.1}", d.fps));
        f.render_widget(fps_gauge, stats_chunks[0]);

        // Processing time
        let proc_block = Block::default()
            .title(" Process ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(if d.processing_ms < 33.0 { Color::Green } else { Color::Red }));
        let proc_gauge = Gauge::default()
            .block(proc_block)
            .gauge_style(
                Style::default()
                    .fg(Color::White)
                    .bg(if d.processing_ms < 33.0 { Color::Green } else { Color::Red })
                    .add_modifier(Modifier::BOLD),
            )
            .percent((d.processing_ms.min(100.0) / 100.0 * 100.0) as u16)
            .label(format!("{:.1}ms", d.processing_ms));
        f.render_widget(proc_gauge, stats_chunks[1]);

        // Gaze vector
        let gaze_display = d
            .gaze_vector
            .as_ref()
            .map(|s| s.as_str())
            .unwrap_or("No gaze");
        let gaze_block = Block::default()
            .title(" Gaze ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(if d.face_detected { Color::Green } else { Color::Yellow }));
        let gaze_para = Paragraph::new(Line::from(Span::styled(
            gaze_display,
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        )))
        .block(gaze_block)
        .alignment(Alignment::Center);
        f.render_widget(gaze_para, stats_chunks[2]);

        // Status block
        let smoothed_str = d
            .smoothed_gaze
            .as_ref()
            .map(|s| s.as_str())
            .unwrap_or("-");
        let events_str = if d.events.is_empty() {
            String::from("-")
        } else {
            d.events.clone()
        };
        let status_lines = vec![
            Line::from(format!("Face: {}", if d.face_detected { "✓" } else { "✗" })),
            Line::from(format!("Confidence: {}", d.confidence)),
            Line::from(format!("Smoothed: {}", smoothed_str)),
            Line::from(format!("Events: {}", events_str)),
            Line::from(format!("Res: {}", d.resolution)),
            Line::from(format!("Frame: {}", d.frame_number)),
        ];
        let status_block = Block::default()
            .title(" Status ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan));
        let status_para = Paragraph::new(status_lines)
            .block(status_block)
            .alignment(Alignment::Left);
        f.render_widget(status_para, stats_chunks[3]);
    } else {
        // No data yet
        let no_data = Paragraph::new("Waiting for camera...")
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::Gray));
        for chunk in stats_chunks.iter() {
            f.render_widget(no_data.clone(), *chunk);
        }
    }

    // Gaze direction visualization (simplified crosshair)
    let viz_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[2]);

    // Left: crosshair visualization
    let crosshair_block = Block::default()
        .title(" Gaze Direction ")
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::Cyan));
    let crosshair_area = crosshair_block.inner(viz_chunks[0]);
    f.render_widget(crosshair_block, viz_chunks[0]);

    // Draw crosshair
    let cx = crosshair_area.x + crosshair_area.width / 2;
    let cy = crosshair_area.y + crosshair_area.height / 2;
    // Crosshair lines (unused currently, kept for reference)
    let _crosshair_chars = [
        ((cx as i32, crosshair_area.y as i32 - 1), "│"),
        ((cx as i32, crosshair_area.y as i32 + crosshair_area.height as i32 + 1), "│"),
        ((crosshair_area.x as i32 - 1, cy as i32), "─"),
        ((crosshair_area.x as i32 + crosshair_area.width as i32 + 1, cy as i32), "─"),
    ];

    // Draw gaze point if we have data
    if let Some(ref d) = data {
        if let Some(ref _gaze_str) = d.gaze_vector {
            let gaze_point = (
                (cx as f32 + d.processing_ms as f32 * 0.1) as u16,
                cy,
            );
            let gaze_marker = Paragraph::new("●")
                .style(Style::default().fg(Color::Red).add_modifier(Modifier::BOLD));
            f.render_widget(gaze_marker, Rect::new(
                gaze_point.0.saturating_sub(1),
                gaze_point.1.saturating_sub(1),
                3,
                3,
            ));
        }
    }

    // Right: calibration status placeholder
    let cal_block = Block::default()
        .title(" Calibration ")
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::Yellow));
    let cal_text = Paragraph::new(vec![
        Line::from("Not calibrated"),
        Line::from(""),
        Line::from("Run: eyetracker --calibrate"),
    ])
    .block(cal_block)
    .alignment(Alignment::Center);
    f.render_widget(cal_text, viz_chunks[1]);

    // Sparklines for FPS history
    let sparkline_area = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[3]);

    // FPS sparkline
    let fps_spark_block = Block::default()
        .title(" FPS History (60s) ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Green));
    let fps_data: Vec<u64> = fps_history.iter().map(|v| *v as u64).collect();
    let fps_spark = Sparkline::default()
        .block(fps_spark_block)
        .data(&fps_data)
        .style(Style::default().fg(Color::Green))
        .max(fps_history.iter().cloned().fold(0.0, f64::max).max(60.0) as u64);
    f.render_widget(fps_spark, sparkline_area[0]);

    // Processing time sparkline
    let proc_spark_block = Block::default()
        .title(" Processing Time (ms) ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Magenta));
    let proc_data: Vec<u64> = time_history.iter().map(|v| *v as u64).collect();
    let proc_spark = Sparkline::default()
        .block(proc_spark_block)
        .data(&proc_data)
        .style(Style::default().fg(Color::Magenta));
    f.render_widget(proc_spark, sparkline_area[1]);

    // Help bar
    let help_text = Paragraph::new(Line::from(vec![
        Span::styled(" [q] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Quit  "),
        Span::styled(" [r] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Reset  "),
        Span::styled(" [Esc] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Exit"),
    ]))
    .alignment(Alignment::Center)
    .style(Style::default().fg(Color::DarkGray));
    f.render_widget(help_text, chunks[4]);
}

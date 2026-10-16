use std::{
    io::{self, BufRead},
    sync::mpsc::{self, RecvTimeoutError},
    thread,
    time::Duration,
};

use tokio::sync::mpsc::UnboundedSender;

const PASTE_BURST_WINDOW: Duration = Duration::from_millis(50);

enum InputEvent {
    Line(io::Result<String>),
    Boundary,
    Closed,
}

#[derive(Default)]
struct PasteFramer {
    lines: Vec<String>,
}

impl PasteFramer {
    fn is_collecting(&self) -> bool {
        !self.lines.is_empty()
    }

    fn accept(&mut self, event: InputEvent) -> Vec<io::Result<String>> {
        match event {
            InputEvent::Line(Ok(line)) => {
                self.lines.push(line);
                Vec::new()
            }
            InputEvent::Line(Err(error)) => {
                let mut output = self.finish().into_iter().map(Ok).collect::<Vec<_>>();
                output.push(Err(error));
                output
            }
            InputEvent::Boundary | InputEvent::Closed => {
                self.finish().into_iter().map(Ok).collect()
            }
        }
    }

    fn finish(&mut self) -> Option<String> {
        if self.lines.is_empty() {
            return None;
        }
        let framed = self.lines.join("\n");
        self.lines.clear();
        Some(framed.trim_end_matches('\n').to_owned())
    }
}

pub fn spawn(destination: UnboundedSender<io::Result<String>>) {
    let (raw_sender, raw_receiver) = mpsc::channel();
    thread::spawn(move || {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            if raw_sender.send(line).is_err() {
                break;
            }
        }
    });
    thread::spawn(move || {
        let mut framer = PasteFramer::default();
        loop {
            let event = if framer.is_collecting() {
                match raw_receiver.recv_timeout(PASTE_BURST_WINDOW) {
                    Ok(line) => InputEvent::Line(line),
                    Err(RecvTimeoutError::Timeout) => InputEvent::Boundary,
                    Err(RecvTimeoutError::Disconnected) => InputEvent::Closed,
                }
            } else {
                match raw_receiver.recv() {
                    Ok(line) => InputEvent::Line(line),
                    Err(_) => InputEvent::Closed,
                }
            };
            let closed = matches!(event, InputEvent::Closed);
            for framed in framer.accept(event) {
                if destination.send(framed).is_err() {
                    return;
                }
            }
            if closed {
                return;
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(events: impl IntoIterator<Item = InputEvent>) -> Vec<String> {
        let mut framer = PasteFramer::default();
        events
            .into_iter()
            .flat_map(|event| framer.accept(event))
            .map(|result| result.unwrap())
            .collect()
    }

    #[test]
    fn pasted_lines_are_one_submission_without_trailing_newlines() {
        assert_eq!(
            frame([
                InputEvent::Line(Ok("first line".into())),
                InputEvent::Line(Ok("second line".into())),
                InputEvent::Line(Ok(String::new())),
                InputEvent::Boundary,
            ]),
            ["first line\nsecond line"]
        );
    }

    #[test]
    fn burst_boundaries_keep_commands_and_guidance_independent() {
        assert_eq!(
            frame([
                InputEvent::Line(Ok("/stop".into())),
                InputEvent::Boundary,
                InputEvent::Line(Ok("please check the second file".into())),
                InputEvent::Boundary,
                InputEvent::Line(Ok("/interrupt".into())),
                InputEvent::Closed,
            ]),
            ["/stop", "please check the second file", "/interrupt"]
        );
    }

    #[test]
    fn end_of_input_flushes_the_final_paste() {
        assert_eq!(
            frame([
                InputEvent::Line(Ok("alpha".into())),
                InputEvent::Line(Ok("beta".into())),
                InputEvent::Closed,
            ]),
            ["alpha\nbeta"]
        );
    }
}

use std::{collections::BTreeSet, fs::File, io::BufReader, path::Path};

use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct DatasetItem {
    #[allow(dead_code)]
    question_id: String,
    haystack_dates: Vec<String>,
    haystack_session_ids: Vec<String>,
    haystack_sessions: Vec<Vec<Turn>>,
}

#[derive(Debug, Deserialize)]
struct Turn {
    role: String,
    content: String,
    #[serde(default)]
    has_answer: Option<bool>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct LongMemChunk {
    pub session_id: String,
    pub session_index: usize,
    pub date: String,
    pub chunk_index: usize,
    pub has_answer: bool,
    pub text: String,
}

#[derive(Debug)]
pub(crate) struct LongMemCorpus {
    pub question_count: usize,
    pub session_count: usize,
    pub chunks: Vec<LongMemChunk>,
}

pub(crate) fn load(path: &Path) -> Result<LongMemCorpus, Box<dyn std::error::Error>> {
    let items: Vec<DatasetItem> = serde_json::from_reader(BufReader::new(File::open(path)?))?;
    build(items)
}

fn build(items: Vec<DatasetItem>) -> Result<LongMemCorpus, Box<dyn std::error::Error>> {
    let mut seen = BTreeSet::new();
    let mut chunks = Vec::new();

    for item in &items {
        if item.haystack_session_ids.len() != item.haystack_sessions.len()
            || item.haystack_dates.len() != item.haystack_sessions.len()
        {
            return Err(format!(
                "LongMemEval item {} has misaligned session IDs, dates, and sessions",
                item.question_id
            )
            .into());
        }
        for ((session_id, date), turns) in item
            .haystack_session_ids
            .iter()
            .zip(&item.haystack_dates)
            .zip(&item.haystack_sessions)
        {
            if !seen.insert(session_id.clone()) {
                continue;
            }
            let session_index = seen.len() - 1;
            let mut turn_index = 0;
            let mut chunk_index = 0;
            while turn_index < turns.len() {
                let pair_end = if turns[turn_index].role.eq_ignore_ascii_case("user")
                    && turns
                        .get(turn_index + 1)
                        .is_some_and(|turn| turn.role.eq_ignore_ascii_case("assistant"))
                {
                    turn_index + 2
                } else {
                    turn_index + 1
                };
                let pair = &turns[turn_index..pair_end];
                let text = pair
                    .iter()
                    .map(|turn| format!("[{}]: {}", turn.role, turn.content))
                    .collect::<Vec<_>>()
                    .join("\n");
                chunks.push(LongMemChunk {
                    session_id: session_id.clone(),
                    session_index,
                    date: date.clone(),
                    chunk_index,
                    has_answer: pair.iter().any(|turn| turn.has_answer.unwrap_or(false)),
                    text: format!("[date: {date}] {text}"),
                });
                chunk_index += 1;
                turn_index = pair_end;
            }
        }
    }

    Ok(LongMemCorpus {
        question_count: items.len(),
        session_count: seen.len(),
        chunks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deduplicates_sessions_and_pairs_turns() {
        let items = r#"[
          {
            "question_id":"q1",
            "haystack_dates":["2025/01/01"],
            "haystack_session_ids":["s1"],
            "haystack_sessions":[[
              {"role":"user","content":"hello","has_answer":true},
              {"role":"assistant","content":"hi","has_answer":false},
              {"role":"user","content":"last","has_answer":false}
            ]]
          },
          {
            "question_id":"q2",
            "haystack_dates":["2025/01/01"],
            "haystack_session_ids":["s1"],
            "haystack_sessions":[[{"role":"user","content":"duplicate"}]]
          }
        ]"#;
        let parsed: Vec<DatasetItem> = serde_json::from_str(items).unwrap();
        let corpus = build(parsed).unwrap();

        assert_eq!(corpus.question_count, 2);
        assert_eq!(corpus.session_count, 1);
        assert_eq!(corpus.chunks.len(), 2);
        assert_eq!(
            corpus.chunks[0].text,
            "[date: 2025/01/01] [user]: hello\n[assistant]: hi"
        );
        assert!(corpus.chunks[0].has_answer);
        assert_eq!(corpus.chunks[0].session_index, 0);
        assert_eq!(corpus.chunks[1].chunk_index, 1);
    }

    #[test]
    fn only_pairs_user_turns_with_the_following_assistant_turn() {
        let items = r#"[{
          "question_id":"q1",
          "haystack_dates":["2025/01/01"],
          "haystack_session_ids":["s1"],
          "haystack_sessions":[[
            {"role":"assistant","content":"preface"},
            {"role":"user","content":"first"},
            {"role":"user","content":"second"},
            {"role":"assistant","content":"reply"}
          ]]
        }]"#;
        let parsed: Vec<DatasetItem> = serde_json::from_str(items).unwrap();
        let corpus = build(parsed).unwrap();

        assert_eq!(corpus.chunks.len(), 3);
        assert_eq!(
            corpus.chunks[0].text,
            "[date: 2025/01/01] [assistant]: preface"
        );
        assert_eq!(corpus.chunks[1].text, "[date: 2025/01/01] [user]: first");
        assert_eq!(
            corpus.chunks[2].text,
            "[date: 2025/01/01] [user]: second\n[assistant]: reply"
        );
    }
}

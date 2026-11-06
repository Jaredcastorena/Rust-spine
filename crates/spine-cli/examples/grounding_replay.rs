//! Offline replay of fixed response/evidence fixtures through production grounding.
use std::{fs, time::Instant};

use serde::Deserialize;
use serde_json::json;
use spine_heart::{GroundingEvidence, GroundingVerdict};

#[allow(dead_code)]
mod grounding {
    include!("../src/grounding.rs");
}

#[derive(Deserialize)]
struct Case {
    id: String,
    group: String,
    response: String,
    evidence: Vec<String>,
    expected: String,
    should_repair: bool,
    note: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() < 4 {
        return Err("usage: grounding_replay MODEL_DIR OUTPUT_JSON CASES_JSON...".into());
    }
    let gate = grounding::GroundingGate::load(&args[1])?;
    let mut rows = Vec::new();
    for path in &args[3..] {
        let cases: Vec<Case> = serde_json::from_slice(&fs::read(path)?)?;
        for case in cases {
            let evidence = case
                .evidence
                .iter()
                .enumerate()
                .map(|(i, text)| GroundingEvidence {
                    id: format!("fixture-{}-{i}", case.id),
                    text: text.clone(),
                })
                .collect::<Vec<_>>();
            let started = Instant::now();
            let result = gate.verify(&case.response, &evidence, 0.65);
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            let (repair, label, report, error) = match result {
                Ok(decision) => {
                    let label = if decision.claim_count == 0 {
                        "no_claim"
                    } else if decision
                        .report
                        .claims
                        .iter()
                        .any(|c| c.verdict == GroundingVerdict::Conflicting)
                    {
                        "conflicting"
                    } else if decision
                        .report
                        .claims
                        .iter()
                        .any(|c| c.verdict == GroundingVerdict::Contradicted)
                    {
                        "contradicted"
                    } else if decision.report.needs_repair() {
                        "unverified"
                    } else {
                        "supported"
                    };
                    (decision.needs_repair, label, Some(decision.report), None)
                }
                Err(error) => (true, "unverified", None, Some(error.to_string())),
            };
            let row = json!({
                "id":case.id, "group":case.group, "response":case.response,
                "evidence":case.evidence, "expected":case.expected,
                "expected_repair":case.should_repair, "note":case.note,
                "actual":label, "needs_repair":repair, "report":report, "error":error,
                "action_correct":repair == case.should_repair,
                "label_correct":label == case.expected, "elapsed_ms":elapsed_ms,
            });
            println!(
                "{}",
                json!({"id":row["id"],"actual":label,"repair":repair,"action_correct":row["action_correct"],"label_correct":row["label_correct"],"error":row["error"]})
            );
            rows.push(row);
            // Preserve partial results if a later model evaluation fails.
            fs::write(
                &args[2],
                serde_json::to_vec_pretty(&json!({"complete":false,"rows":rows}))?,
            )?;
        }
    }
    let summary = json!({
        "cases":rows.len(),
        "correct_actions":rows.iter().filter(|r|r["action_correct"] == true).count(),
        "correct_labels":rows.iter().filter(|r|r["label_correct"] == true).count(),
        "missed_problems":rows.iter().filter(|r|r["expected_repair"] == true && r["needs_repair"] == false).count(),
        "unnecessary_flags":rows.iter().filter(|r|r["expected_repair"] == false && r["needs_repair"] == true).count(),
        "inference_errors":rows.iter().filter(|r|!r["error"].is_null()).count(),
    });
    println!("{summary}");
    fs::write(
        &args[2],
        serde_json::to_vec_pretty(&json!({"complete":true,"summary":summary,"rows":rows}))?,
    )?;
    Ok(())
}

use super::*;
use serde_json::json;

async fn read_page(
    root: &std::path::Path,
    arguments: serde_json::Value,
    budget: usize,
) -> ToolResult {
    let mut context = ToolContext::default();
    context
        .metadata
        .insert("max_tool_result_chars".into(), budget.to_string());
    FileReadTool {
        cwd: root.to_path_buf(),
    }
    .execute(
        &ToolCall {
            id: "page".into(),
            name: "file_read".into(),
            arguments,
        },
        &context,
    )
    .await
    .expect("tool execution")
}

#[tokio::test]
async fn pagination_reconstructs_large_unicode_files_within_model_budget() {
    let directory = tempfile::tempdir().unwrap();
    for content in ["a".repeat(60_000), "é🦀中\n".repeat(15_000)] {
        fs::write(directory.path().join("input.txt"), &content).unwrap();
        for budget in [512, 16_384] {
            let mut offset = 0;
            let mut reconstructed = String::new();
            loop {
                let result = read_page(
                    directory.path(),
                    json!({"path":"input.txt", "offset":offset}),
                    budget,
                )
                .await;
                assert!(result.success, "{:?}", result.error);
                assert!(result.output.chars().count() <= budget);
                assert_eq!(result.model_text(budget), result.output);
                let (header, page) = result.output.split_once("\n=== Content ===\n").unwrap();
                reconstructed.push_str(page);
                let next = header
                    .lines()
                    .find_map(|line| line.strip_prefix("next_offset: "))
                    .unwrap();
                if next == "null" {
                    break;
                }
                let next: usize = next.parse().unwrap();
                assert_eq!(next, offset + page.chars().count());
                assert!(next > offset);
                offset = next;
            }
            assert_eq!(reconstructed, content);
        }
    }
}

#[tokio::test]
async fn pagination_handles_empty_files_eof_and_single_character_pages() {
    let directory = tempfile::tempdir().unwrap();
    fs::write(directory.path().join("empty.txt"), "").unwrap();
    fs::write(directory.path().join("input.txt"), "é🦀中").unwrap();
    for arguments in [
        json!({"path":"empty.txt"}),
        json!({"path":"input.txt", "offset":3}),
    ] {
        let result = read_page(directory.path(), arguments, 16_384).await;
        assert!(result.success, "{:?}", result.error);
        assert!(result.output.contains("next_offset: null\n"));
        assert!(result.output.ends_with("=== Content ===\n"));
    }
    let result = read_page(
        directory.path(),
        json!({"path":"input.txt", "offset":1, "max_chars":1}),
        16_384,
    )
    .await;
    assert!(result.success);
    assert!(result.output.contains("next_offset: 2\n"));
    assert!(result.output.ends_with("=== Content ===\n🦀"));
}

#[tokio::test]
async fn pagination_rejects_invalid_arguments_and_insufficient_budgets() {
    let directory = tempfile::tempdir().unwrap();
    fs::write(directory.path().join("input.txt"), "abc").unwrap();
    for extra in [
        json!({"offset":4}),
        json!({"offset":-1}),
        json!({"offset":1.5}),
        json!({"max_chars":0}),
        json!({"max_chars":12001}),
        json!({"max_chars":"3"}),
    ] {
        let mut arguments = json!({"path":"input.txt"});
        arguments
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        let result = read_page(directory.path(), arguments, 16_384).await;
        assert!(!result.success, "unexpected success: {}", result.output);
    }
    let result = read_page(directory.path(), json!({"path":"input.txt"}), 1).await;
    assert!(!result.success);
    assert!(result.error.unwrap().contains("budget"));
}

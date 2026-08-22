use proptest::prelude::*;
use spine_runtime::ToolResult;

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 2_000,
        max_shrink_iters: 10_000,
        .. ProptestConfig::default()
    })]

    #[test]
    fn unicode_tool_results_truncate_only_at_character_boundaries(
        text in any::<String>(),
        maximum_chars in 1_usize..1_024,
        failure in any::<bool>(),
    ) {
        let result = if failure {
            ToolResult::failure(text.clone())
        } else {
            ToolResult::success(text.clone())
        };
        let rendered = result.model_text(maximum_chars);
        if text.chars().count() <= maximum_chars {
            prop_assert_eq!(rendered, text);
        } else {
            prop_assert!(rendered.ends_with("\n[truncated]"));
            let prefix = rendered.strip_suffix("\n[truncated]").unwrap();
            prop_assert_eq!(prefix.chars().count(), maximum_chars);
            prop_assert!(text.starts_with(prefix));
        }
    }
}

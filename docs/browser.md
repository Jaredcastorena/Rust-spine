# Optional browser interaction

Spine's `web_search`, `web_fetch`, `web_navigate`, `web_back`, and `web_forward`
read HTTP content as text. They do not run page JavaScript or operate forms.
For rendered pages and form interaction, connect an existing W3C WebDriver
service by setting its base URL before starting Spine:

```sh
export SPINE_WEBDRIVER_URL=http://127.0.0.1:4444
```

A service with a path prefix is also supported, for example
`http://127.0.0.1:4444/wd/hub`. Spine does not download, install, or start a
browser or driver. With the variable unset, only the existing text web tools
are registered. A malformed configured URL fails startup with a clear error.

Optional `SPINE_WEBDRIVER_CAPABILITIES` supplies a JSON object for W3C
`alwaysMatch` capabilities. For example, a Firefox-capable driver can use:

```sh
export SPINE_WEBDRIVER_CAPABILITIES='{"browserName":"firefox","moz:firefoxOptions":{"args":["-headless"]}}'
```

Browser-specific executable paths and arguments belong in that object when
needed by your driver. Spine enforces a 15-second page-load timeout, a
5-second script timeout, no implicit element wait, and 20-second HTTP request
timeouts. Dialogs are dismissed and reported rather than silently accepted.

| Tool | Behavior |
| --- | --- |
| `browser_open` | Open an HTTP(S) URL; create a new session on first use. |
| `browser_snapshot` | Read rendered text and visible controls, optionally scoped by CSS selector. |
| `browser_click` | Click the single element matched by a CSS selector. |
| `browser_type` | Replace an editable element's text; `clear: false` appends. |
| `browser_select` | Choose a native select option by exact `value`, firing input/change events. |
| `browser_submit` | Submit a form or a control's associated form, honoring validation and submit handlers. |
| `browser_close` | Delete the session and clear local state; call when finished. |

Each interaction returns the current URL, title, bounded page text, and CSS
selectors for controls. Selectors must match exactly one element. Snapshots
omit input values, show up to 40 visible controls and 20 options per select,
and are limited to 12,000 characters. Scope a snapshot to a section to inspect
more controls. When navigation temporarily removes the document or it is still loading,
Spine retries only the snapshot, at most ten reads within three seconds. It
never repeats a click or submission during this wait. If observation remains
unavailable, the action may already have completed: inspect with
`browser_snapshot` before repeating it. For asynchronous page updates, take
another snapshot after the update; a readable document does not prove that
all background activity has finished. These tools operate the current top-level document; they do not expose
arbitrary script execution, screenshots, iframe switching, or file uploads.
All tools except `browser_snapshot` are mutating actions under Spine's tool
policy; clicks and submissions can change a website's data.

Each harness has its own session, even when agents share a tool registry.
Actions are serialized and at most 32 harness sessions can remain open; close
sessions when finished to release capacity. Elements are located again for every action. Dead or
uncertain sessions are deleted before reuse; an interrupted action is cleaned
up before the next use. Failed deletion retains the session ID for a retry
before another session can be created. A new `browser_open` can restart after
cleanup.
If the process or service stops during session creation, a remote session may
outlive Spine; configure the service's own session expiry as appropriate.

Use a dedicated browser profile. Authentication, network access, profile
isolation, and protection of the external WebDriver service are your
responsibility. Driver URLs with embedded credentials, queries, or fragments
are rejected. Driver error messages are not echoed. Snapshot omission does not prevent
Spine from recording tool arguments, including text supplied to `browser_type`,
in its normal tool history. Page text and URLs are
visible to the model, so use only accounts and pages appropriate to the task.

The implementation follows the [W3C WebDriver protocol](https://www.w3.org/TR/webdriver2/),
including session creation/deletion, CSS element lookup, element interactions,
and fixed internal scripts. Browser/vendor setup remains external to Spine.

For maintainers, protocol tests use local fake driver responses. An additional
ignored test, `browser_tools::tests::real_browser_javascript_and_form_roundtrip`,
uses `SPINE_WEBDRIVER_TEST_URL` and the same optional capabilities variable to
exercise a local JavaScript-generated form against an explicitly supplied
real driver. It creates no driver process and does not contact public pages.

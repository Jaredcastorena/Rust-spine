//! A lexical risk heuristic, not a shell parser or a security boundary.
//!
//! Inspect executable positions and explicit shell/SQL payloads rather than
//! searching prose arguments. Substitutions execute even inside double quotes.
//! Malformed quoting, excessive nesting and dynamically constructed executable
//! names are conservative. This cannot resolve aliases/functions, variables,
//! sourced files, arbitrary interpreter programs, or the behavior of binaries.
//! Shell grammar such as case patterns and arithmetic is only approximated and
//! can be conservatively classified. Explicit PowerShell payloads use its quote
//! and call-operator rules; cmd payloads have only basic command-word coverage.
//! The caller must continue to enforce the tool's ordinary mutation policy.

#[derive(Clone, Copy, PartialEq)]
enum Dialect {
    Posix,
    PowerShell,
}

#[derive(Clone, Default, Debug)]
struct Word {
    text: String,
    dynamic: bool,
    quoted: bool,
}

enum Token {
    Word(Word),
    Boundary(bool), // PowerShell's call operator invokes a quoted command name.
    Redirect(bool, Word),
    HereInput(Word),
}

const MAX_DEPTH: usize = 24;

pub(crate) fn is_destructive(command: &str) -> bool {
    inspect(
        command,
        if cfg!(windows) {
            Dialect::PowerShell
        } else {
            Dialect::Posix
        },
        0,
    )
}

fn inspect(command: &str, dialect: Dialect, depth: usize) -> bool {
    if depth > MAX_DEPTH {
        return true;
    }
    let mut lexer = Lexer {
        input: command.chars().collect(),
        position: 0,
        dialect,
        depth,
        dangerous: false,
    };
    match lexer.tokens(None) {
        Ok(tokens) => lexer.dangerous || commands(&tokens, dialect, depth),
        Err(()) => true,
    }
}

struct Lexer {
    input: Vec<char>,
    position: usize,
    dialect: Dialect,
    depth: usize,
    dangerous: bool,
}

impl Lexer {
    fn current(&self) -> Option<char> {
        self.input.get(self.position).copied()
    }

    fn next(&self) -> Option<char> {
        self.input.get(self.position + 1).copied()
    }

    fn tokens(&mut self, closing: Option<char>) -> Result<Vec<Token>, ()> {
        if self.depth > MAX_DEPTH {
            return Err(());
        }
        let mut tokens = Vec::new();
        let mut heredocs: Vec<(String, bool, bool, usize)> = Vec::new();
        while let Some(c) = self.current() {
            if Some(c) == closing {
                self.position += 1;
                return Ok(tokens);
            }
            match c {
                ' ' | '\t' | '\r' => self.position += 1,
                '#' => {
                    while self.current().is_some_and(|c| c != '\n') {
                        self.position += 1;
                    }
                }
                '\n' => {
                    self.position += 1;
                    tokens.push(Token::Boundary(false));
                    for (delimiter, expand, strip_tabs, index) in heredocs.drain(..) {
                        let text = self.heredoc(&delimiter, expand, strip_tabs)?;
                        tokens[index] = Token::HereInput(Word {
                            text,
                            ..Word::default()
                        });
                    }
                }
                ';' | '|' | '&' if !(c == '&' && self.next() == Some('>')) => {
                    self.position += 1;
                    tokens.push(Token::Boundary(c == '&'));
                }
                '(' => {
                    // Preserve the traditional fork-bomb guard without
                    // matching its spelling in prose arguments.
                    if matches!(tokens.last(), Some(Token::Word(word)) if word.text == ":") {
                        self.dangerous = true;
                    }
                    self.position += 1;
                    self.depth += 1;
                    let nested = self.tokens(Some(')'))?;
                    self.depth -= 1;
                    self.dangerous |= commands(&nested, self.dialect, self.depth + 1);
                    tokens.push(Token::Boundary(false));
                }
                ')' => return Err(()),
                '{' | '}'
                    if self.dialect == Dialect::PowerShell
                        || (self
                            .next()
                            .is_none_or(|c| c.is_whitespace() || ";|&".contains(c))
                            && (c == '}'
                                || tokens.is_empty()
                                || matches!(tokens.last(), Some(Token::Boundary(_))))) =>
                {
                    self.position += 1;
                    tokens.push(Token::Boundary(false));
                }
                '=' if self.dialect == Dialect::PowerShell => {
                    self.position += 1;
                    tokens.push(Token::Word(Word {
                        text: "=".into(),
                        ..Word::default()
                    }));
                }
                '<' if self.next() == Some('#') && self.dialect == Dialect::PowerShell => {
                    self.position += 2;
                    while !(self.current() == Some('#') && self.next() == Some('>')) {
                        self.current().ok_or(())?;
                        self.position += 1;
                    }
                    self.position += 2;
                }
                '<' | '>' | '&' => {
                    // An adjacent numeric prefix is a file descriptor, not
                    // the executable (`2>log rm file`).
                    if self.position > 0
                        && self.input[self.position - 1].is_ascii_digit()
                        && matches!(tokens.last(), Some(Token::Word(word)) if !word.quoted && word.text.chars().all(|c| c.is_ascii_digit()))
                    {
                        tokens.pop();
                    }
                    let output = c != '<';
                    self.position += 1;
                    if c == '&' {
                        self.position += 1;
                    }
                    // Process substitutions are executable, not file names.
                    if self.current() == Some('(') {
                        self.position += 1;
                        self.depth += 1;
                        let nested = self.tokens(Some(')'))?;
                        self.depth -= 1;
                        self.dangerous |= commands(&nested, self.dialect, self.depth + 1);
                        continue;
                    }
                    let heredoc = !output && self.current() == Some('<');
                    if self.current() == Some(c) || matches!(self.current(), Some('|' | '&')) {
                        self.position += 1;
                    }
                    let here_string = heredoc && self.current() == Some('<');
                    if here_string {
                        self.position += 1;
                    }
                    let strip_tabs = heredoc && self.current() == Some('-');
                    if strip_tabs {
                        self.position += 1;
                    }
                    while matches!(self.current(), Some(' ' | '\t')) {
                        self.position += 1;
                    }
                    let target = self.word()?;
                    if heredoc && !here_string {
                        heredocs.push((
                            target.text.clone(),
                            !target.quoted,
                            strip_tabs,
                            tokens.len(),
                        ));
                        tokens.push(Token::HereInput(Word::default()));
                    } else if here_string {
                        tokens.push(Token::HereInput(target));
                    } else {
                        tokens.push(Token::Redirect(output, target));
                    }
                }
                _ => tokens.push(Token::Word(self.word()?)),
            }
        }
        if closing.is_some() || !heredocs.is_empty() {
            Err(())
        } else {
            Ok(tokens)
        }
    }

    fn word(&mut self) -> Result<Word, ()> {
        let start = self.position;
        let mut word = Word::default();
        while let Some(c) = self.current() {
            if c.is_whitespace()
                || ";|&()<>".contains(c)
                || (c == '=' && self.dialect == Dialect::PowerShell && word.text.starts_with('$'))
            {
                break;
            }
            match c {
                '\'' | '"' => {
                    word.quoted = true;
                    self.quote(c, &mut word)?;
                }
                '$' => self.dollar(&mut word)?,
                '\\' if self.dialect == Dialect::Posix => {
                    word.quoted = true;
                    self.position += 1;
                    let escaped = self.current().ok_or(())?;
                    if escaped != '\n' {
                        word.text.push(escaped);
                    }
                    self.position += 1;
                }
                '`' if self.dialect == Dialect::Posix => self.backticks(&mut word)?,
                '`' => {
                    self.position += 1;
                    let escaped = self.current().ok_or(())?;
                    if escaped != '\n' {
                        word.text.push(escaped);
                    }
                    self.position += 1;
                }
                '#' if self.dialect == Dialect::PowerShell => break,
                '*' | '?' | '[' if self.dialect == Dialect::Posix => {
                    word.dynamic = true;
                    word.text.push(c);
                    self.position += 1;
                }
                '{' | '}' if self.dialect == Dialect::Posix => {
                    word.dynamic = true;
                    word.text.push(c);
                    self.position += 1;
                }
                _ => {
                    word.text.push(c);
                    self.position += 1;
                }
            }
        }
        if self.position == start {
            Err(())
        } else {
            Ok(word)
        }
    }

    fn quote(&mut self, quote: char, word: &mut Word) -> Result<(), ()> {
        self.position += 1;
        while let Some(c) = self.current() {
            if c == quote {
                self.position += 1;
                if quote == '\''
                    && self.dialect == Dialect::PowerShell
                    && self.current() == Some('\'')
                {
                    word.text.push('\'');
                    self.position += 1;
                    continue;
                }
                return Ok(());
            }
            if quote == '"' {
                if c == '$' {
                    self.dollar(word)?;
                    continue;
                }
                if c == '`' && self.dialect == Dialect::Posix {
                    self.backticks(word)?;
                    continue;
                }
                let escape = if self.dialect == Dialect::Posix {
                    '\\'
                } else {
                    '`'
                };
                if c == escape {
                    let next = self.next().ok_or(())?;
                    if self.dialect == Dialect::PowerShell || "$`\"\\\n".contains(next) {
                        self.position += 2;
                        if next != '\n' {
                            word.text.push(next);
                        }
                        continue;
                    }
                }
            }
            word.text.push(c);
            self.position += 1;
        }
        Err(())
    }

    fn dollar(&mut self, word: &mut Word) -> Result<(), ()> {
        if self.depth > MAX_DEPTH {
            return Err(());
        }
        self.position += 1;
        match self.current() {
            Some('(') => {
                self.position += 1;
                self.depth += 1;
                let nested = self.tokens(Some(')'))?;
                self.depth -= 1;
                self.dangerous |= commands(&nested, self.dialect, self.depth + 1);
                word.dynamic = true;
            }
            Some('{') => {
                self.position += 1;
                self.depth += 1;
                // Scan parameter expansions for nested executable expansions.
                while self.current() != Some('}') {
                    match self.current().ok_or(())? {
                        '$' => self.dollar(word)?,
                        '`' if self.dialect == Dialect::Posix => self.backticks(word)?,
                        '\'' | '"' => self.quote(self.current().unwrap(), word)?,
                        _ => self.position += 1,
                    }
                }
                self.position += 1;
                self.depth -= 1;
                word.dynamic = true;
            }
            Some('\'') if self.dialect == Dialect::Posix => {
                // ANSI-C quoting can synthesize a command name. Treat that
                // executable as unknown, but retain literal prose arguments.
                word.quoted = true;
                word.dynamic = true;
                self.position += 1;
                loop {
                    let c = self.current().ok_or(())?;
                    self.position += 1;
                    if c == '\'' {
                        break;
                    }
                    if c == '\\' {
                        self.current().ok_or(())?;
                        self.position += 1;
                    } else {
                        word.text.push(c);
                    }
                }
            }
            Some(c) if c.is_alphanumeric() || "_@*#?$!-".contains(c) => {
                word.dynamic = true;
                word.text.push('$');
            }
            _ => word.text.push('$'),
        }
        Ok(())
    }

    fn backticks(&mut self, word: &mut Word) -> Result<(), ()> {
        self.position += 1;
        let mut source = String::new();
        loop {
            let c = self.current().ok_or(())?;
            self.position += 1;
            if c == '`' {
                break;
            }
            if c == '\\' && self.current().is_some_and(|c| "$`\\".contains(c)) {
                source.push(self.current().unwrap());
                self.position += 1;
            } else {
                source.push(c);
            }
        }
        self.dangerous |= inspect(&source, self.dialect, self.depth + 1);
        word.dynamic = true;
        Ok(())
    }

    fn heredoc(&mut self, delimiter: &str, expand: bool, strip_tabs: bool) -> Result<String, ()> {
        let mut body = String::new();
        loop {
            let start = self.position;
            while self.current().is_some_and(|c| c != '\n') {
                self.position += 1;
            }
            let line: String = self.input[start..self.position].iter().collect();
            if self.current() == Some('\n') {
                self.position += 1;
            }
            if (if strip_tabs {
                line.trim_start_matches('\t')
            } else {
                &line
            }) == delimiter
            {
                break;
            }
            if self.position == self.input.len() {
                return Err(());
            }
            body.push_str(&line);
            body.push('\n');
        }
        if expand {
            // Heredoc bodies are data, but unquoted delimiters allow command
            // substitutions. Quotes inside the body do not disable them.
            let mut nested = Lexer {
                input: body.chars().collect(),
                position: 0,
                dialect: self.dialect,
                depth: self.depth + 1,
                dangerous: false,
            };
            let mut ignored = Word::default();
            while let Some(c) = nested.current() {
                match c {
                    '$' => nested.dollar(&mut ignored)?,
                    '`' => nested.backticks(&mut ignored)?,
                    '\\' if nested.next().is_some_and(|c| "$`\\\n".contains(c)) => {
                        nested.position += 2
                    }
                    _ => nested.position += 1,
                }
            }
            self.dangerous |= nested.dangerous;
        }
        Ok(body)
    }
}

fn commands(tokens: &[Token], dialect: Dialect, depth: usize) -> bool {
    let mut words = Vec::new();
    let mut invoke = false;
    let mut input = None;
    for token in tokens {
        match token {
            Token::Word(word) => words.push(word),
            Token::HereInput(body) => input = Some(body),
            Token::Redirect(output, target) => {
                if *output && raw_device(&target.text) {
                    return true;
                }
            }
            Token::Boundary(call) => {
                if command(&words, dialect, depth, invoke, input) {
                    return true;
                }
                words.clear();
                invoke = *call;
                input = None;
            }
        }
    }
    command(&words, dialect, depth, invoke, input)
}

fn assignment(text: &str) -> bool {
    text.split_once('=').is_some_and(|(name, _)| {
        !name.is_empty()
            && name
                .chars()
                .enumerate()
                .all(|(i, c)| c == '_' || c.is_ascii_alphabetic() || (i > 0 && c.is_ascii_digit()))
    })
}

fn basename(text: &str) -> String {
    text.rsplit(['/', '\\'])
        .next()
        .unwrap_or(text)
        .to_ascii_lowercase()
        .trim_end_matches(".exe")
        .to_owned()
}

fn raw_device(text: &str) -> bool {
    let text = text.to_ascii_lowercase();
    [
        "/dev/sd",
        "/dev/hd",
        "/dev/nvme",
        "/dev/mmcblk",
        "\\\\.\\physicaldrive",
    ]
    .iter()
    .any(|prefix| text.starts_with(prefix))
}

fn command(
    words: &[&Word],
    dialect: Dialect,
    depth: usize,
    invoke: bool,
    input: Option<&Word>,
) -> bool {
    if depth > MAX_DEPTH {
        return true;
    }
    let Some(first) = words.first() else {
        return false;
    };
    if assignment(&first.text)
        || matches!(
            first.text.as_str(),
            "if" | "then" | "elif" | "else" | "while" | "until" | "do" | "!"
        )
    {
        return command(&words[1..], dialect, depth + 1, invoke, input);
    }
    if matches!(
        first.text.as_str(),
        "for" | "select" | "case" | "function" | "foreach" | "switch"
    ) {
        return false;
    }
    if dialect == Dialect::PowerShell {
        if first.text.starts_with('$') && words.get(1).is_some_and(|w| w.text == "=") {
            return command(&words[2..], dialect, depth + 1, false, input);
        }
        if first.text == "." {
            return command(&words[1..], dialect, depth + 1, true, input);
        }
        if (first.quoted || first.dynamic) && !invoke {
            return false;
        }
    }
    if first.dynamic {
        return true;
    }
    let name = basename(&first.text);
    let args: Vec<String> = words[1..]
        .iter()
        .map(|w| w.text.to_ascii_lowercase())
        .collect();
    let dynamic = words[1..].iter().any(|w| w.dynamic);
    match name.as_str() {
        "rm" | "rmdir" | "unlink" | "wipefs" | "truncate" | "shred" | "shutdown" | "reboot"
        | "poweroff" | "remove-item" | "ri" | "rd" | "clear-disk" | "format-volume"
        | "initialize-disk" | "remove-partition" | "stop-computer" | "restart-computer"
        | "diskpart" => true,
        name if name == "mkfs" || name.starts_with("mkfs.") => true,
        "kill" | "pkill" | "killall" => {
            dynamic
                || args.iter().any(|a| {
                    matches!(
                        a.as_str(),
                        "-9" | "-kill"
                            | "-sigkill"
                            | "--signal=9"
                            | "--signal=kill"
                            | "--signal=sigkill"
                    )
                })
                || args.windows(2).any(|pair| {
                    matches!(pair[0].as_str(), "-s" | "--signal")
                        && matches!(pair[1].as_str(), "9" | "kill" | "sigkill")
                })
        }
        "chmod" => words[1..]
            .iter()
            .find(|w| !w.text.starts_with('-'))
            .is_some_and(|mode| {
                mode.dynamic || (!mode.text.is_empty() && mode.text.chars().all(|c| c == '0'))
            }),
        "crontab" => {
            dynamic
                || args
                    .iter()
                    .any(|a| a.starts_with('-') && !a.starts_with("--") && a.contains('r'))
        }
        "dd" => {
            dynamic
                || args
                    .iter()
                    .any(|a| a.starts_with("if=") || a.strip_prefix("of=").is_some_and(raw_device))
        }
        "del" | "erase" => {
            dialect == Dialect::PowerShell || args.iter().any(|a| a.starts_with('/'))
        }
        "git" => git_risk(&words[1..]),
        "drop" => args
            .first()
            .is_some_and(|a| a == "table" || a == "database"),
        "psql" | "mysql" | "mariadb" | "sqlite3" | "sqlcmd" => {
            words[1..].iter().any(|a| sql_risk(&a.text))
        }
        "sh" | "bash" | "dash" | "zsh" | "ksh" | "ash" => shell_command(&words[1..], input, depth),
        "powershell" | "pwsh" => {
            if args
                .iter()
                .any(|a| a.starts_with("-enc") || a == "-e" || a == "-ec")
            {
                return true;
            }
            args.iter()
                .position(|a| matches!(a.as_str(), "-command" | "-c" | "-commandwithargs"))
                .is_some_and(|i| {
                    let payload = &words[i + 2..];
                    payload.iter().any(|w| w.dynamic)
                        || inspect(
                            &payload
                                .iter()
                                .map(|w| w.text.as_str())
                                .collect::<Vec<_>>()
                                .join(" "),
                            Dialect::PowerShell,
                            depth + 1,
                        )
                })
        }
        "cmd" => args
            .iter()
            .position(|a| a == "/c" || a == "/k")
            .is_some_and(|i| {
                let payload = &words[i + 2..];
                payload.iter().any(|w| w.dynamic)
                    || inspect(
                        &payload
                            .iter()
                            .map(|w| w.text.as_str())
                            .collect::<Vec<_>>()
                            .join(" "),
                        Dialect::Posix,
                        depth + 1,
                    )
            }),
        "eval" | "invoke-expression" | "iex" => {
            dynamic
                || inspect(
                    &words[1..]
                        .iter()
                        .map(|w| w.text.as_str())
                        .collect::<Vec<_>>()
                        .join(" "),
                    dialect,
                    depth + 1,
                )
        }
        "command" if args.first().is_some_and(|a| a == "-v") => false,
        "env" | "sudo" | "doas" | "command" | "builtin" | "exec" | "nohup" | "nice" | "timeout"
        | "setsid" | "xargs" | "time" => wrapper(&name, &words[1..], dialect, depth + 1, input),
        "find" => words.iter().enumerate().any(|(i, word)| {
            matches!(word.text.as_str(), "-exec" | "-execdir" | "-ok" | "-okdir")
                && command(&words[i + 1..], dialect, depth + 1, true, input)
        }),
        _ => false,
    }
}

fn shell_command(words: &[&Word], input: Option<&Word>, depth: usize) -> bool {
    let mut index = 0;
    let mut stdin_script = false;
    while let Some(word) = words.get(index) {
        if word.dynamic {
            return true;
        }
        let arg = word.text.as_str();
        if arg == "--" {
            index += 1;
            break;
        }
        if arg == "-" {
            stdin_script = true;
            index += 1;
            break;
        }
        if !arg.starts_with('-') && !arg.starts_with('+') {
            break;
        }
        if arg.starts_with('-') && !arg.starts_with("--") && arg[1..].contains('c') {
            index += 1;
            if words.get(index).is_some_and(|word| word.text == "--") {
                index += 1;
            }
            // -c consumes the command argument; redirected stdin remains
            // data for that command and is not itself a shell script.
            return words.get(index).is_some_and(|payload| {
                payload.dynamic || inspect(&payload.text, Dialect::Posix, depth + 1)
            });
        }
        if arg.starts_with('-') && !arg.starts_with("--") && arg[1..].contains('s') {
            stdin_script = true;
        }
        index += if matches!(arg, "-o" | "+o" | "-O" | "+O" | "--rcfile" | "--init-file") {
            2
        } else {
            1
        };
    }
    // A positional script filename consumes the shell's program slot unless
    // -s explicitly requests stdin; its file contents are outside this scan.
    if !stdin_script && index < words.len() {
        return false;
    }
    input
        .is_some_and(|payload| payload.dynamic || inspect(&payload.text, Dialect::Posix, depth + 1))
}

fn git_risk(words: &[&Word]) -> bool {
    let mut i = 0;
    while let Some(word) = words.get(i) {
        if word.dynamic {
            return true;
        }
        match word.text.as_str() {
            "-C" | "-c" | "--git-dir" | "--work-tree" | "--namespace" => i += 2,
            "--" => {
                i += 1;
                break;
            }
            text if text.starts_with('-') => i += 1,
            _ => break,
        }
    }
    match words.get(i).map(|w| w.text.as_str()) {
        Some("reset") => words[i + 1..]
            .iter()
            .any(|w| w.dynamic || w.text == "--hard" || w.text.starts_with("--hard=")),
        Some("clean") => words[i + 1..].iter().any(|w| {
            w.dynamic
                || w.text == "--force"
                || (w.text.starts_with('-') && !w.text.starts_with("--") && w.text.contains('f'))
        }),
        _ => false,
    }
}

fn sql_risk(text: &str) -> bool {
    let words: Vec<_> = text
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .filter(|w| !w.is_empty())
        .map(str::to_ascii_lowercase)
        .collect();
    words
        .windows(2)
        .any(|pair| pair[0] == "drop" && matches!(pair[1].as_str(), "database" | "table"))
}

fn wrapper(
    name: &str,
    words: &[&Word],
    dialect: Dialect,
    depth: usize,
    input: Option<&Word>,
) -> bool {
    let mut i = 0;
    let mut shell_mode = false;
    while let Some(word) = words.get(i) {
        let arg = word.text.as_str();
        if name == "env" && assignment(arg) {
            i += 1;
            continue;
        }
        if word.dynamic {
            return true;
        }
        if arg == "--" {
            i += 1;
            break;
        }
        if !arg.starts_with('-') {
            break;
        }
        if name == "command" && !arg.starts_with("--") && (arg.contains('v') || arg.contains('V')) {
            return false;
        }
        if name == "sudo"
            && (matches!(arg, "--shell" | "--login")
                || (!arg.starts_with("--") && (arg.contains('s') || arg.contains('i'))))
        {
            shell_mode = true;
        }
        if name == "env" && matches!(arg, "-S" | "--split-string") {
            return words.get(i + 1).is_some_and(|w| {
                w.dynamic || split_command(&w.text, &words[i + 2..], dialect, depth + 1, input)
            });
        }
        if name == "env" && arg.starts_with("--split-string=") {
            return split_command(&arg[15..], &words[i + 1..], dialect, depth + 1, input);
        }
        if name == "env" && arg.starts_with("-S") && arg.len() > 2 {
            return split_command(&arg[2..], &words[i + 1..], dialect, depth + 1, input);
        }
        let value = match name {
            "env" => matches!(arg, "-u" | "--unset" | "-C" | "--chdir"),
            "sudo" | "doas" => matches!(
                arg,
                "-u" | "-g"
                    | "-h"
                    | "-p"
                    | "-C"
                    | "-T"
                    | "-R"
                    | "-D"
                    | "-r"
                    | "-t"
                    | "--user"
                    | "--group"
                    | "--host"
                    | "--prompt"
                    | "--chdir"
                    | "--chroot"
            ),
            "nice" => matches!(arg, "-n" | "--adjustment"),
            "exec" => arg == "-a",
            "time" => matches!(arg, "-o" | "--output" | "-f" | "--format"),
            "timeout" => matches!(arg, "-s" | "--signal" | "-k" | "--kill-after"),
            "xargs" => matches!(
                arg,
                "-a" | "--arg-file"
                    | "-E"
                    | "-I"
                    | "-L"
                    | "-n"
                    | "-P"
                    | "-s"
                    | "-d"
                    | "--delimiter"
                    | "--max-args"
                    | "--max-procs"
                    | "--max-chars"
                    | "--max-lines"
            ),
            _ => false,
        };
        i += if value { 2 } else { 1 };
    }
    if name == "timeout" {
        i += 1;
    }
    if shell_mode {
        return inspect(
            &words
                .get(i..)
                .unwrap_or_default()
                .iter()
                .map(|w| w.text.as_str())
                .collect::<Vec<_>>()
                .join(" "),
            dialect,
            depth + 1,
        );
    }
    command(
        words.get(i..).unwrap_or_default(),
        dialect,
        depth + 1,
        true,
        input,
    )
}

fn split_command(
    source: &str,
    trailing: &[&Word],
    dialect: Dialect,
    depth: usize,
    input: Option<&Word>,
) -> bool {
    let mut lexer = Lexer {
        input: source.chars().collect(),
        position: 0,
        dialect,
        depth,
        dangerous: false,
    };
    match lexer.tokens(None) {
        Ok(mut tokens) => {
            tokens.extend(trailing.iter().map(|word| Token::Word((*word).clone())));
            if let Some(input) = input {
                tokens.push(Token::HereInput(input.clone()));
            }
            lexer.dangerous || commands(&tokens, dialect, depth)
        }
        Err(()) => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(cases: &[&str], expected: bool, dialect: Dialect) {
        for source in cases {
            assert_eq!(inspect(source, dialect, 0), expected, "{source}");
        }
    }

    #[test]
    fn quoted_prose_comments_and_read_commands_are_not_executables() {
        check(
            &[
                r#"echo "confirm repo""#,
                r#"echo "form something""#,
                r#"echo "alarm clock""#,
                r#"echo "xrm y""#,
                r#"echo "confirm""#,
                r#"echo "platform""#,
                r#"echo "term""#,
                r#"echo "rm""#,
                r#"echo "rm harmless-text""#,
                r#"echo "/tmp/alarm clock.txt""#,
                "printf '%s\\n' 'git reset --hard'",
                "echo shutdown reboot mkfs.ext4 Remove-Item",
                "echo ok # rm victim",
                "# rm victim\nprintf ok",
                "echo foo#rm",
                "printf '%s' '$(rm victim)'",
                "echo 'x > /dev/sda'",
                "X=rm printf ok",
                "env PATH=\"$PATH\" printf ok",
                "command -v rm",
                "command -V rm",
                "command -p -v rm",
                "command -pv rm",
                "git -C repo status",
                "git clean -nd",
                "python -c 'print(\"confirm repo\")'",
                "cat rm",
                "grep 'drop table' file",
                "echo $(printf 'confirm repo')",
                "echo $'rm victim'",
                "echo \"$HOME\"",
                "cat <<'EOF'\nrm victim\nEOF\n",
                "cat <<EOF\nrm victim\nEOF\n",
                "cat <<\\EOF\n$(rm victim)\nEOF\n",
                "kill 9",
                "kill -TERM 9",
                "chmod 755 000",
                "git status \"$path\"",
                "echo foo{rm}bar",
                "echo { rm victim }",
            ],
            false,
            Dialect::Posix,
        );
    }

    #[test]
    fn executable_names_operators_and_wrappers_preserve_risk() {
        check(
            &[
                "rm victim",
                "rm",
                "rmdir victim",
                "r\\m victim",
                "r\"m\" victim",
                "'rm' victim",
                "r\\\nm victim",
                "/bin/rm victim",
                "$'\\x72m' victim",
                "$COMMAND victim",
                "printf ok;rm victim",
                "true && /bin/rm victim",
                "false || rmdir victim",
                "echo ok | rm victim",
                "(rm victim)",
                "{ rm victim; }",
                "if true; then rm victim; fi",
                "for f in x; do rm victim; done",
                "env X=1 /bin/rm victim",
                "sudo -u root -- rm victim",
                "sudo -s 'rm victim'",
                "env -u HOME -- rm victim",
                "env -S 'rm victim'",
                "env -S 'sh -c' 'rm victim'",
                "env -Srm victim",
                "command -- rm victim",
                "nohup nice -n 5 rm victim",
                "timeout 3 rm victim",
                "printf x | xargs -I{} rm '{}'",
                "find . -exec rm '{}' ';'",
                "sh -c 'rm victim'",
                "bash -lc 'rm victim'",
                "bash -O extglob -c 'rm victim'",
                "eval 'r'\"m victim\"",
                ":(){ :|:& };:",
            ],
            true,
            Dialect::Posix,
        );
    }

    #[test]
    fn expansions_and_device_writes_are_executable() {
        check(
            &[
                "echo $(rm victim)",
                "echo \"$(rm victim)\"",
                "echo `rm victim`",
                "X=$(rm victim) true",
                "echo ${x:-$(rm victim)}",
                "cat <(rm victim)",
                "echo x >'/dev/sda'",
                "echo x 2>/dev/sda",
                "echo x >> /dev/nvme0n1",
                "cat <<EOF\n$(rm victim)\nEOF\n",
                "echo $(echo $(rm victim))",
                "printf ok > /dev/sda",
                "2>/tmp/log rm victim",
                "2>&1 rm victim",
                "dd of=/dev/sda if=input",
            ],
            true,
            Dialect::Posix,
        );
    }

    #[test]
    fn known_destructive_argument_forms() {
        check(
            &[
                "kill -9 123",
                "pkill -KILL name",
                "killall --signal=9 name",
                "kill -s KILL 123",
                "mkfs.ext4 device",
                "wipefs device",
                "git -C repo reset --hard HEAD",
                "git -c advice.foo=false clean -dfx",
                "truncate file",
                "shred file",
                "chmod -R 000 dir",
                "crontab -r",
                "shutdown now",
                "reboot",
                "poweroff",
                "drop database foo",
                "psql -c 'DROP TABLE things'",
                "mysql -e 'DROP DATABASE things'",
            ],
            true,
            Dialect::Posix,
        );
    }

    #[test]
    fn powershell_uses_its_own_quoting_and_invocation_rules() {
        check(
            &[
                "Write-Output 'Remove-Item victim'",
                "Get-Content file",
                "'Remove-Item'",
                "Write-Output '$(Remove-Item victim)'",
                "Write-Output \"confirm repo\"",
                "# Remove-Item victim\nGet-Content file",
                "$x = 'Remove-Item victim'",
                "$x='Remove-Item victim'",
                "$x = Get-Content file",
                "$x",
                "<# Remove-Item victim #> Get-Content file",
            ],
            false,
            Dialect::PowerShell,
        );
        check(
            &[
                "Remove-Item victim",
                "Remove`-Item victim",
                "& 'Remove-Item' victim",
                "& { Remove-Item victim }",
                "Write-Output \"$(Remove-Item victim)\"",
                "Microsoft.PowerShell.Management\\Remove-Item victim",
                "Clear-Disk 0",
                "Get-Content file | Remove-Item",
                "& $command victim",
                "$result = Remove-Item victim",
                "$result=Remove-Item victim",
                ". Remove-Item victim",
                ". 'Remove-Item' victim",
                "del victim",
            ],
            true,
            Dialect::PowerShell,
        );
        check(
            &[
                "powershell -NoProfile -Command 'Remove-Item victim'",
                "pwsh -c 'Remove-Item victim'",
                "pwsh -EncodedCommand AAAA",
                "cmd /c 'del /f victim'",
                "cmd /c '\"del\" /f victim'",
            ],
            true,
            Dialect::Posix,
        );
        check(
            &["pwsh -c 'Get-Content file'", "sh -c 'printf ok'"],
            false,
            Dialect::Posix,
        );
    }

    #[test]
    fn malformed_or_excessively_nested_syntax_is_conservative() {
        check(
            &["echo 'unfinished", "echo $(printf ok", "echo trailing\\"],
            true,
            Dialect::Posix,
        );
        let deep = format!("{}printf ok{}", "$(".repeat(30), ")".repeat(30));
        assert!(is_destructive(&deep));
    }

    #[test]
    fn shell_stdin_and_wrapper_options_keep_the_executable_position() {
        check(
            &[
                "sh <<'EOF'\nrm victim\nEOF\n",
                "bash <<EOF\nrm victim\nEOF\n",
                "env bash -s -- argument <<'EOF'\nrm victim\nEOF\n",
                "env -S 'bash -s' <<'EOF'\nrm victim\nEOF\n",
                "bash -s <<< 'rm victim'",
                "sh -c -- 'rm victim'",
                "bash -lc -- 'rm victim'",
                "exec -a label rm victim",
                "time -p rm victim",
                "/usr/bin/time -f elapsed -o timing.txt rm victim",
            ],
            true,
            Dialect::Posix,
        );
        check(
            &[
                "cat <<'EOF'\nrm victim\nEOF\n",
                "cat <<EOF\nrm victim\nEOF\n",
                "sh <<'EOF'\necho \"rm victim\"\nEOF\n",
                "bash <<EOF\nprintf '%s' 'confirm repo'\nEOF\n",
                "env bash -c 'cat' <<'EOF'\nrm victim\nEOF\n",
                "sh read-script.sh <<'EOF'\nrm victim\nEOF\n",
                "bash -- read-script.sh <<'EOF'\nrm victim\nEOF\n",
                "cat <<< 'rm victim'",
                "sh -c -- 'echo \"rm victim\"'",
                "exec -a rm printf '%s\\n' harmless",
                "time -p printf '%s\\n' 'rm victim'",
                "/usr/bin/time -f rm -o timing.txt printf ok",
            ],
            false,
            Dialect::Posix,
        );
    }
}

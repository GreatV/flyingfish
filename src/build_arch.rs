pub fn parse(text: &str) -> Result<Vec<u32>, String> {
    let mut result = Vec::new();
    for token in text.split(|c: char| c.is_whitespace() || c == ';' || c == ',') {
        if token.is_empty() {
            continue;
        }
        if token.contains("+PTX") || token.ends_with("-virtual") || token.starts_with("compute_") {
            return Err(format!(
                "PTX target {token} is unsupported; specify native architectures"
            ));
        }
        let token = token.strip_prefix("sm_").unwrap_or(token);
        let token = token.strip_suffix("-real").unwrap_or(token);
        let value = if let Some((major, minor)) = token.split_once('.') {
            let major: u32 = major
                .parse()
                .map_err(|_| format!("invalid architecture {token}"))?;
            let minor: u32 = minor
                .parse()
                .map_err(|_| format!("invalid architecture {token}"))?;
            if !(8..=99).contains(&major) || minor > 9 {
                return Err(format!("unsupported architecture {token}"));
            }
            major * 10 + minor
        } else {
            token
                .parse::<u32>()
                .map_err(|_| format!("invalid architecture {token}"))?
        };
        if !(80..=999).contains(&value) {
            return Err(format!(
                "FA2 requires compute capability >= 8.0; found {token}"
            ));
        }
        result.push(value);
    }
    result.sort_unstable();
    result.dedup();
    if result.is_empty() {
        return Err("CUDA architecture list is empty".into());
    }
    Ok(result)
}

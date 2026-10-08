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

#[cfg(test)]
mod tests {
    use super::parse;

    #[test]
    fn formats_and_duplicate_devices() {
        assert_eq!(parse("8.9;8.6 8.9").unwrap(), vec![86, 89]);
        assert_eq!(parse("86-real,sm_89").unwrap(), vec![86, 89]);
        assert_eq!(parse("8.6\n8.6\n").unwrap(), vec![86]);
        assert_eq!(parse("10.0;120").unwrap(), vec![100, 120]);
    }

    #[test]
    fn rejects_virtual_and_invalid_targets() {
        for text in [
            "",
            "8.9+PTX",
            "86-virtual",
            "compute_86",
            "8.10",
            "7.5",
            "Ada",
        ] {
            assert!(parse(text).is_err(), "{text}");
        }
    }
}

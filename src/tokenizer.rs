#[cfg(any(feature = "cuda", test))]
use anyhow::{Context, bail, ensure};
use anyhow::{Result, anyhow};
use std::path::Path;
use tokenizers::Tokenizer;

pub(crate) fn load(dir: &Path) -> Result<Tokenizer> {
    Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("tokenizer load: {e}"))
}

pub fn encode(dir: &Path, text: &str) -> Result<Vec<u32>> {
    let tokenizer = load(dir)?;
    Ok(tokenizer
        .encode(text, true)
        .map_err(|e| anyhow!("tokenizer encode: {e}"))?
        .get_ids()
        .to_vec())
}

pub fn decode(dir: &Path, ids: &[u32]) -> Result<String> {
    decode_with(&load(dir)?, ids)
}

pub(crate) fn decode_with(tokenizer: &Tokenizer, ids: &[u32]) -> Result<String> {
    tokenizer
        .decode(ids, true)
        .map_err(|e| anyhow!("tokenizer decode: {e}"))
}

/// Sentencepiece `tokenizer.model` pieces in id order. The `ModelProto`
/// carries the pieces as repeated field 1, each a message with the piece
/// bytes as its field 1 (`sentencepiece_model.proto`).
#[cfg(any(feature = "cuda", test))]
pub(crate) fn sentencepiece_pieces(bytes: &[u8]) -> Result<Vec<String>> {
    let mut pieces = Vec::new();
    let mut reader = ProtoReader { data: bytes };
    while let Some((field, wire)) = reader.tag()? {
        if field == 1 && wire == 2 {
            let mut entry = ProtoReader {
                data: reader.bytes()?,
            };
            while let Some((entry_field, entry_wire)) = entry.tag()? {
                if entry_field == 1 && entry_wire == 2 {
                    pieces.push(String::from_utf8(entry.bytes()?.to_vec())?);
                } else if !entry.skip(entry_wire)? {
                    bail!("sentencepiece piece field {entry_field} wire {entry_wire}");
                }
            }
        } else if !reader.skip(wire)? {
            bail!("sentencepiece proto field {field} wire {wire}");
        }
    }
    ensure!(!pieces.is_empty(), "sentencepiece model has no pieces");
    Ok(pieces)
}

#[cfg(any(feature = "cuda", test))]
struct ProtoReader<'a> {
    data: &'a [u8],
}

#[cfg(any(feature = "cuda", test))]
impl<'a> ProtoReader<'a> {
    /// Reads one protobuf tag `(field number, wire type)`; `None` at the end.
    fn tag(&mut self) -> Result<Option<(u32, u8)>> {
        if self.data.is_empty() {
            return Ok(None);
        }
        let key = self.varint().context("proto tag")?;
        Ok(Some((
            u32::try_from(key >> 3).context("proto field number")?,
            (key & 7) as u8,
        )))
    }

    fn varint(&mut self) -> Result<u64> {
        let mut value = 0u64;
        for index in 0..10 {
            let byte = *self.data.first().context("truncated proto varint")?;
            self.data = &self.data[1..];
            value |= u64::from(byte & 0x7f) << (index * 7);
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        bail!("proto varint exceeds 10 bytes")
    }

    /// A length-delimited field's payload.
    fn bytes(&mut self) -> Result<&'a [u8]> {
        let length = usize::try_from(self.varint()?).context("proto field length")?;
        self.take(length)
    }

    fn skip(&mut self, wire: u8) -> Result<bool> {
        match wire {
            0 => {
                self.varint()?;
                Ok(true)
            }
            1 => {
                self.take(8)?;
                Ok(true)
            }
            2 => {
                self.bytes()?;
                Ok(true)
            }
            5 => {
                self.take(4)?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        let bytes = self
            .data
            .get(..count)
            .with_context(|| format!("truncated proto field of {count} bytes"))?;
        self.data = &self.data[count..];
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn varint(value: u64) -> Vec<u8> {
        let mut out = Vec::new();
        let mut rest = value;
        loop {
            let byte = (rest & 0x7f) as u8;
            rest >>= 7;
            if rest == 0 {
                out.push(byte);
                return out;
            }
            out.push(byte | 0x80);
        }
    }

    fn piece(text: &str, kind: u64) -> Vec<u8> {
        let mut entry = vec![0x0a, u8::try_from(text.len()).unwrap()];
        entry.extend_from_slice(text.as_bytes());
        entry.extend_from_slice(&[0x15]);
        entry.extend_from_slice(&(-0.5f32).to_le_bytes());
        entry.push(0x18);
        entry.extend_from_slice(&varint(kind));
        let mut field = vec![0x0a, u8::try_from(entry.len()).unwrap()];
        field.extend_from_slice(&entry);
        field
    }

    #[test]
    fn sentencepiece_pieces_read_the_id_order() -> Result<()> {
        let mut proto = Vec::new();
        for (text, kind) in [("<unk>", 2u64), ("▁a", 6), ("b", 6), ("▁the", 6)] {
            proto.extend_from_slice(&piece(text, kind));
        }
        proto.extend_from_slice(&[0x12, 0x02, 0x08, 0x01]);
        let pieces = sentencepiece_pieces(&proto)?;
        assert_eq!(pieces, ["<unk>", "▁a", "b", "▁the"]);
        Ok(())
    }

    #[test]
    fn sentencepiece_pieces_reject_truncated_models() {
        let error = sentencepiece_pieces(&[0x0a, 0x10, 0x0a]).unwrap_err();
        assert!(error.to_string().contains("truncated"), "{error}");
    }
}

//! GGUF container reader: header, metadata and tensor index over a read-only mmap.
use anyhow::{Context, Result, bail, ensure};
use memmap2::{Mmap, MmapOptions};
use std::{
    collections::{BTreeMap, HashSet},
    fs::File,
    path::{Path, PathBuf},
};

const MAGIC: u32 = 0x4655_4747;
const DEFAULT_ALIGNMENT: u32 = 32;
const MIN_HEADER_BYTES: u64 = 24;
const MAX_TENSOR_NAME_BYTES: usize = 64;
const MAX_TENSOR_DIMS: u32 = 4;
const MIN_TENSOR_RECORD_BYTES: usize = 24; // name length + dimension count + type + offset
const MAX_META_NESTING: u32 = 16;

pub struct GgufFile {
    path: PathBuf,
    map: Mmap,
    header: GgufHeader,
    tensors: Vec<GgufTensor>,
    data_start: u64,
    orig_shapes: BTreeMap<String, Vec<u32>>,
}

#[derive(Debug)]
pub struct GgufHeader {
    pub version: u32,
    pub tensor_count: usize,
    pub alignment: u32,
    pub architecture: Option<String>,
}

#[derive(Debug)]
pub struct GgufTensor {
    pub name: String,
    pub ggml_type: GgmlType,
    pub dims: Vec<u64>,
    pub offset: u64,
    pub nbytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum GgmlType {
    F32,
    F16,
    Bf16,
    Q4_0,
    Q8_0,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Other(u32),
}

impl GgmlType {
    pub fn block_size(self) -> Option<usize> {
        match self {
            Self::F32 | Self::F16 | Self::Bf16 => Some(1),
            Self::Q4_0 | Self::Q8_0 => Some(32),
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K => Some(256),
            Self::Other(_) => None,
        }
    }

    pub fn block_bytes(self) -> Option<usize> {
        match self {
            Self::F32 => Some(4),
            Self::F16 | Self::Bf16 => Some(2),
            Self::Q4_0 => Some(18),
            Self::Q8_0 => Some(34),
            Self::Q2K => Some(84),
            Self::Q3K => Some(110),
            Self::Q4K => Some(144),
            Self::Q5K => Some(176),
            Self::Q6K => Some(210),
            Self::Other(_) => None,
        }
    }

    fn from_u32(code: u32) -> Self {
        match code {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            8 => Self::Q8_0,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            30 => Self::Bf16,
            other => Self::Other(other),
        }
    }
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let len = file
            .metadata()
            .with_context(|| format!("stat {}", path.display()))?
            .len();
        ensure!(len >= MIN_HEADER_BYTES, "truncated GGUF: {len} bytes");
        let map = unsafe { MmapOptions::new().map(&file)? };
        let parsed = parse(&map)?;
        Ok(Self {
            path: path.to_owned(),
            map,
            header: parsed.header,
            tensors: parsed.tensors,
            data_start: parsed.data_start,
            orig_shapes: parsed.orig_shapes,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn data(&self) -> &[u8] {
        &self.map
    }

    /// Absolute file offset of the first tensor byte.
    pub fn data_start(&self) -> u64 {
        self.data_start
    }

    /// The `comfy.gguf.orig_shape.<tensor>` value: the original shape of a packed tensor.
    pub fn orig_shape(&self, tensor: &str) -> Option<&[u32]> {
        self.orig_shapes.get(tensor).map(Vec::as_slice)
    }

    pub fn header(&self) -> &GgufHeader {
        &self.header
    }

    pub fn tensor(&self, name: &str) -> Option<&GgufTensor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn tensors(&self) -> impl Iterator<Item = &GgufTensor> {
        self.tensors.iter()
    }
}

enum MetaValue {
    U32(u32),
    I32(i32),
    I32Array(Vec<i32>),
    Str(String),
    Other,
}

struct SliceReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> SliceReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn pos(&self) -> usize {
        self.pos
    }

    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    fn take(&mut self, n: usize, what: &str) -> Result<&'a [u8]> {
        let left = self.buf.len() - self.pos;
        ensure!(
            n <= left,
            "truncated GGUF: {what} needs {n} bytes, {left} left"
        );
        let out = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }

    fn u8(&mut self, what: &str) -> Result<u8> {
        Ok(self.take(1, what)?[0])
    }

    fn u16(&mut self, what: &str) -> Result<u16> {
        let b = self.take(2, what)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self, what: &str) -> Result<u32> {
        let b = self.take(4, what)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn i32(&mut self, what: &str) -> Result<i32> {
        Ok(self.u32(what)? as i32)
    }

    fn u64(&mut self, what: &str) -> Result<u64> {
        let b = self.take(8, what)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn f32(&mut self, what: &str) -> Result<f32> {
        let b = self.take(4, what)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn f64(&mut self, what: &str) -> Result<f64> {
        let b = self.take(8, what)?;
        Ok(f64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn string(&mut self, what: &str) -> Result<String> {
        let len = usize::try_from(self.u64(what)?).context("GGUF string length overflow")?;
        let bytes = self.take(len, what)?;
        String::from_utf8(bytes.to_vec()).with_context(|| format!("invalid UTF-8 in {what}"))
    }
}

fn ensure_capacity(
    remaining: usize,
    count: usize,
    bytes_per_item: usize,
    what: &str,
) -> Result<()> {
    ensure!(
        count <= remaining / bytes_per_item,
        "truncated GGUF: {what} needs at least {} bytes, {remaining} left",
        count.saturating_mul(bytes_per_item)
    );
    Ok(())
}

struct ParsedGguf {
    header: GgufHeader,
    tensors: Vec<GgufTensor>,
    data_start: u64,
    orig_shapes: BTreeMap<String, Vec<u32>>,
}

fn parse(buf: &[u8]) -> Result<ParsedGguf> {
    let mut r = SliceReader::new(buf);
    let magic = r.u32("magic")?;
    ensure!(magic == MAGIC, "bad GGUF magic {magic:#010x}");
    let version = r.u32("version")?;
    ensure!(
        version == 2 || version == 3,
        "unsupported GGUF version {version}"
    );
    let tensor_count =
        usize::try_from(r.u64("tensor count")?).context("GGUF tensor count overflow")?;
    let kv_count =
        usize::try_from(r.u64("metadata count")?).context("GGUF metadata count overflow")?;
    ensure!(
        tensor_count <= buf.len(),
        "tensor count {tensor_count} exceeds the {}-byte file",
        buf.len()
    );
    ensure!(
        kv_count <= buf.len(),
        "metadata count {kv_count} exceeds the {}-byte file",
        buf.len()
    );

    let mut alignment = DEFAULT_ALIGNMENT;
    let mut architecture = None;
    let mut orig_shapes = BTreeMap::new();
    for _ in 0..kv_count {
        let key = r.string("metadata key")?;
        let value_type = r.u32("metadata value type")?;
        let value = read_meta_value(&mut r, value_type, &key, 0)?;
        match value {
            MetaValue::U32(v) if key == "general.alignment" => alignment = v,
            MetaValue::Str(s) if key == "general.architecture" => architecture = Some(s),
            MetaValue::I32(v) => {
                if let Some(name) = key.strip_prefix("comfy.gguf.orig_shape.") {
                    ensure!(v > 0, "metadata {key:?} has non-positive value {v}");
                    orig_shapes.insert(name.to_owned(), vec![v as u32]);
                }
            }
            MetaValue::I32Array(values) => {
                if let Some(name) = key.strip_prefix("comfy.gguf.orig_shape.") {
                    ensure!(
                        values.iter().all(|&v| v > 0),
                        "metadata {key:?} has a non-positive dimension"
                    );
                    orig_shapes.insert(name.to_owned(), values.iter().map(|&v| v as u32).collect());
                }
            }
            _ => {}
        }
    }

    ensure!(
        alignment > 0 && alignment.is_multiple_of(8),
        "invalid GGUF alignment {alignment}"
    );

    ensure_capacity(
        r.remaining(),
        tensor_count,
        MIN_TENSOR_RECORD_BYTES,
        "tensor index",
    )?;
    let mut tensors = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = r.string("tensor name")?;
        ensure!(
            name.len() <= MAX_TENSOR_NAME_BYTES,
            "tensor name {name:?} is {} bytes, over the {MAX_TENSOR_NAME_BYTES}-byte limit",
            name.len()
        );
        let n_dimensions = r.u32("tensor dimension count")?;
        ensure!(
            n_dimensions <= MAX_TENSOR_DIMS,
            "tensor {name:?} has {n_dimensions} dimensions, at most {MAX_TENSOR_DIMS} allowed"
        );
        ensure_capacity(
            r.remaining(),
            n_dimensions as usize,
            8,
            "tensor dimension list",
        )?;
        let mut dims = Vec::with_capacity(n_dimensions as usize);
        for _ in 0..n_dimensions {
            dims.push(r.u64("tensor dimension")?);
        }
        let ggml_type = GgmlType::from_u32(r.u32("tensor type")?);
        let offset = r.u64("tensor offset")?;
        let nbytes = tensor_nbytes(&name, ggml_type, &dims)?;
        ensure!(
            offset.is_multiple_of(u64::from(alignment)),
            "tensor {name:?} offset {offset} is not a multiple of alignment {alignment}"
        );
        tensors.push(GgufTensor {
            name,
            ggml_type,
            dims,
            offset,
            nbytes,
        });
    }

    let pos = r.pos() as u64;
    let data_start =
        pos + (u64::from(alignment) - pos % u64::from(alignment)) % u64::from(alignment);
    ensure!(
        data_start <= buf.len() as u64,
        "truncated GGUF: tensor data starts at byte {data_start} of {}",
        buf.len()
    );
    for t in &tensors {
        let start = data_start
            .checked_add(t.offset)
            .context("GGUF tensor offset overflow")?;
        let end = start
            .checked_add(t.nbytes as u64)
            .context("GGUF tensor range overflow")?;
        ensure!(
            end <= buf.len() as u64,
            "tensor {:?} range {start}..{end} is outside the {}-byte file",
            t.name,
            buf.len()
        );
    }

    ensure_capacity(
        buf.len(),
        tensors.len(),
        MIN_TENSOR_RECORD_BYTES,
        "tensor name set",
    )?;
    let mut names = HashSet::with_capacity(tensors.len());
    for t in &tensors {
        ensure!(
            names.insert(t.name.as_str()),
            "duplicate tensor name {:?}",
            t.name
        );
    }

    Ok(ParsedGguf {
        header: GgufHeader {
            version,
            tensor_count,
            alignment,
            architecture,
        },
        tensors,
        data_start,
        orig_shapes,
    })
}

fn read_meta_value(
    r: &mut SliceReader,
    value_type: u32,
    key: &str,
    depth: u32,
) -> Result<MetaValue> {
    ensure!(
        depth <= MAX_META_NESTING,
        "metadata {key:?} nests deeper than {MAX_META_NESTING} arrays"
    );
    Ok(match value_type {
        0 | 1 => {
            r.u8(key)?;
            MetaValue::Other
        }
        2 | 3 => {
            r.u16(key)?;
            MetaValue::Other
        }
        4 => MetaValue::U32(r.u32(key)?),
        5 => MetaValue::I32(r.i32(key)?),
        6 => {
            r.f32(key)?;
            MetaValue::Other
        }
        7 => {
            let b = r.u8(key)?;
            ensure!(
                b == 0 || b == 1,
                "metadata {key:?} has invalid bool value {b}"
            );
            MetaValue::Other
        }
        8 => MetaValue::Str(r.string(key)?),
        9 => {
            let element_type = r.u32("array element type")?;
            let len =
                usize::try_from(r.u64("array length")?).context("GGUF array length overflow")?;
            if element_type == 5 {
                ensure_capacity(r.remaining(), len, 4, "metadata i32 array")?;
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(r.i32(key)?);
                }
                MetaValue::I32Array(values)
            } else {
                for _ in 0..len {
                    read_meta_value(r, element_type, key, depth + 1)?;
                }
                MetaValue::Other
            }
        }
        10 | 11 => {
            r.u64(key)?;
            MetaValue::Other
        }
        12 => {
            r.f64(key)?;
            MetaValue::Other
        }
        other => bail!("metadata {key:?} has unknown value type {other}"),
    })
}

fn tensor_nbytes(name: &str, ggml_type: GgmlType, dims: &[u64]) -> Result<usize> {
    let (block_size, block_bytes) = match (ggml_type.block_size(), ggml_type.block_bytes()) {
        (Some(size), Some(bytes)) => (size as u128, bytes as u128),
        _ => bail!("tensor {name:?} has unsupported type {ggml_type:?}"),
    };
    let elements = dims
        .iter()
        .try_fold(1u128, |acc, &d| acc.checked_mul(u128::from(d)))
        .context("GGUF tensor element count overflow")?;
    ensure!(
        elements.is_multiple_of(block_size),
        "tensor {name:?} has {elements} elements, not a multiple of block size {block_size}"
    );
    let nbytes = elements / block_size * block_bytes;
    ensure!(
        nbytes <= usize::MAX as u128,
        "tensor {name:?} is {nbytes} bytes, too large"
    );
    Ok(nbytes as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicUsize, Ordering},
    };

    fn put_u32(b: &mut Vec<u8>, v: u32) {
        b.extend_from_slice(&v.to_le_bytes());
    }

    fn put_u64(b: &mut Vec<u8>, v: u64) {
        b.extend_from_slice(&v.to_le_bytes());
    }

    fn put_str(b: &mut Vec<u8>, s: &str) {
        put_u64(b, s.len() as u64);
        b.extend_from_slice(s.as_bytes());
    }

    fn synthetic() -> Vec<u8> {
        let mut b = Vec::new();
        put_u32(&mut b, MAGIC);
        put_u32(&mut b, 3);
        put_u64(&mut b, 2);
        put_u64(&mut b, 2);
        put_str(&mut b, "general.alignment");
        put_u32(&mut b, 4);
        put_u32(&mut b, 32);
        put_str(&mut b, "general.architecture");
        put_u32(&mut b, 8);
        put_str(&mut b, "test");
        put_str(&mut b, "f32.tensor");
        put_u32(&mut b, 2);
        put_u64(&mut b, 4);
        put_u64(&mut b, 8);
        put_u32(&mut b, 0);
        put_u64(&mut b, 0);
        put_str(&mut b, "q4.tensor");
        put_u32(&mut b, 2);
        put_u64(&mut b, 32);
        put_u64(&mut b, 2);
        put_u32(&mut b, 2);
        put_u64(&mut b, 128);
        while b.len() % 32 != 0 {
            b.push(0);
        }
        b.extend_from_slice(&[0xAB; 128]);
        b.extend_from_slice(&[0xCD; 36]);
        b
    }

    struct TempGguf(PathBuf);

    impl TempGguf {
        fn new(bytes: &[u8]) -> Result<Self> {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "flyingfish-gguf-{}-{}.gguf",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::write(&path, bytes)?;
            Ok(Self(path))
        }
    }

    impl Drop for TempGguf {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }

    #[test]
    fn synthetic_container_parses() -> Result<()> {
        let bytes = synthetic();
        let parsed = parse(&bytes)?;
        let header = &parsed.header;
        let tensors = &parsed.tensors;
        ensure!(header.version == 3, "version {}", header.version);
        ensure!(
            header.tensor_count == 2,
            "tensor_count {}",
            header.tensor_count
        );
        ensure!(header.alignment == 32, "alignment {}", header.alignment);
        ensure!(
            header.architecture.as_deref() == Some("test"),
            "architecture {:?}",
            header.architecture
        );
        ensure!(tensors.len() == 2, "tensors {}", tensors.len());

        let f32t = tensors
            .iter()
            .find(|t| t.name == "f32.tensor")
            .context("f32.tensor missing")?;
        ensure!(
            f32t.ggml_type == GgmlType::F32,
            "f32 type {:?}",
            f32t.ggml_type
        );
        ensure!(f32t.dims == [4, 8], "f32 dims {:?}", f32t.dims);
        ensure!(f32t.offset == 0, "f32 offset {}", f32t.offset);
        ensure!(f32t.nbytes == 128, "f32 nbytes {}", f32t.nbytes);

        let q4 = tensors
            .iter()
            .find(|t| t.name == "q4.tensor")
            .context("q4.tensor missing")?;
        ensure!(q4.ggml_type == GgmlType::Q4_0, "q4 type {:?}", q4.ggml_type);
        ensure!(q4.dims == [32, 2], "q4 dims {:?}", q4.dims);
        ensure!(q4.offset == 128, "q4 offset {}", q4.offset);
        ensure!(q4.nbytes == 36, "q4 nbytes {}", q4.nbytes);
        Ok(())
    }

    #[test]
    fn opens_a_file_from_disk() -> Result<()> {
        let tmp = TempGguf::new(&synthetic())?;
        let file = GgufFile::open(&tmp.0)?;
        let header = file.header();
        ensure!(header.version == 3, "version {}", header.version);
        ensure!(
            header.tensor_count == 2,
            "tensor_count {}",
            header.tensor_count
        );
        ensure!(header.alignment == 32, "alignment {}", header.alignment);
        ensure!(
            header.architecture.as_deref() == Some("test"),
            "architecture {:?}",
            header.architecture
        );
        ensure!(
            file.tensor("f32.tensor").is_some_and(|t| t.nbytes == 128),
            "f32.tensor missing or wrong size"
        );
        ensure!(
            file.tensor("q4.tensor").is_some_and(|t| t.nbytes == 36),
            "q4.tensor missing or wrong size"
        );
        ensure!(file.tensors().count() == 2, "tensor iterator count");
        Ok(())
    }

    #[test]
    fn bad_magic_is_rejected() {
        let mut bytes = synthetic();
        bytes[0] = b'X';
        assert!(parse(&bytes).is_err(), "accepted a bad magic");
    }

    #[test]
    fn truncated_file_is_rejected() {
        let bytes = synthetic();
        let cut = bytes.len() / 2;
        assert!(parse(&bytes[..cut]).is_err(), "accepted a truncated file");
    }

    fn collect_gguf(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
        for entry in fs::read_dir(dir)? {
            let path = entry?.path();
            if path.is_dir() {
                collect_gguf(&path, out)?;
            } else if path.extension().is_some_and(|e| e == "gguf") {
                out.push(path);
            }
        }
        Ok(())
    }

    #[test]
    fn parses_published_nvidia_ggufs() -> Result<()> {
        let Some(dir) = std::env::var_os("FF_GGUF_DIR") else {
            return Ok(());
        };
        let mut files = Vec::new();
        collect_gguf(&PathBuf::from(dir), &mut files)?;
        ensure!(!files.is_empty(), "no .gguf files in FF_GGUF_DIR");
        for path in files {
            let file = GgufFile::open(&path)?;
            let header = file.header();
            ensure!(header.version == 3, "{}: version", path.display());
            ensure!(header.alignment == 32, "{}: alignment", path.display());
            ensure!(
                header
                    .architecture
                    .as_deref()
                    .is_some_and(|a| !a.is_empty()),
                "{}: architecture missing",
                path.display()
            );
            let tensors: Vec<_> = file.tensors().collect();
            ensure!(
                header.tensor_count == tensors.len(),
                "{}: header tensor_count {} != {}",
                path.display(),
                header.tensor_count,
                tensors.len()
            );
            for t in tensors {
                let block_size = t
                    .ggml_type
                    .block_size()
                    .context("unsupported tensor type")?;
                let block_bytes = t
                    .ggml_type
                    .block_bytes()
                    .context("unsupported tensor type")?;
                let elements: u128 = t.dims.iter().map(|&d| u128::from(d)).product();
                ensure!(
                    elements.is_multiple_of(block_size as u128),
                    "{}: tensor {:?} has {elements} elements, not a multiple of {block_size}",
                    path.display(),
                    t.name
                );
                let expected = elements / (block_size as u128) * (block_bytes as u128);
                ensure!(
                    t.nbytes as u128 == expected,
                    "{}: tensor {:?} nbytes {} != {expected}",
                    path.display(),
                    t.name,
                    t.nbytes
                );
                ensure!(
                    t.offset.is_multiple_of(u64::from(header.alignment)),
                    "{}: tensor {:?} offset {} misaligned",
                    path.display(),
                    t.name,
                    t.offset
                );
            }
        }
        Ok(())
    }
}

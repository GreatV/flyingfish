use anyhow::{Context, Result, bail, ensure};
use std::{
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

const MUX_TIMEOUT: Duration = Duration::from_secs(600);
const MUX_OUTPUT_LIMIT: usize = 64 * 1024;
static SCRATCH_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(crate) type StdinWriter = Box<dyn FnOnce(&mut dyn Write) -> Result<()> + Send>;

/// Last lines of a bounded stderr tail, each line capped; never the whole stream.
pub(crate) fn stderr_tail(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    let mut lines: Vec<String> = text
        .lines()
        .rev()
        .take(8)
        .map(|line| line.chars().take(240).collect())
        .collect();
    lines.reverse();
    lines.join(" | ")
}

enum ToolExit {
    Status(std::process::ExitStatus),
    TimedOut,
    WaitFailed(String),
}

/// Runs one tool and keeps stdout under `limit` bytes.
pub(crate) fn tool_stdout(
    operation: &str,
    tool: &str,
    args: &[&str],
    timeout: Duration,
    limit: usize,
) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    tool_output(operation, tool, args, timeout, limit, |bytes| {
        output.extend_from_slice(bytes);
        Ok(())
    })?;
    Ok(output)
}

pub(crate) fn tool_output(
    operation: &str,
    tool: &str,
    args: &[&str],
    timeout: Duration,
    limit: usize,
    consume: impl FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
    tool_output_with_stdin(operation, tool, args, timeout, limit, None, consume)
}

fn tool_output_with_stdin(
    operation: &str,
    tool: &str,
    args: &[&str],
    timeout: Duration,
    limit: usize,
    stdin_writer: Option<StdinWriter>,
    mut consume: impl FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
    let mut command = std::process::Command::new(tool);
    command
        .args(args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    if stdin_writer.is_some() {
        command.stdin(std::process::Stdio::piped());
    } else {
        command.stdin(std::process::Stdio::null());
    }
    let mut child = command.spawn().map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            anyhow::anyhow!("{operation} requires {tool} on PATH")
        } else {
            anyhow::Error::new(error).context(format!("run {tool}"))
        }
    })?;
    let mut stdout = child.stdout.take().context("tool stdout pipe missing")?;
    let stderr = child.stderr.take().context("tool stderr pipe missing")?;
    let child_stdin = stdin_writer
        .is_some()
        .then(|| child.stdin.take().context("tool stdin pipe missing"))
        .transpose()?;
    let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
    let (exit_tx, exit_rx) = std::sync::mpsc::channel::<ToolExit>();
    let waiter = std::thread::Builder::new()
        .name(format!("{tool}-wait"))
        .spawn(move || {
            let deadline = std::time::Instant::now() + timeout;
            let exit = loop {
                match child.try_wait() {
                    Ok(Some(status)) => break ToolExit::Status(status),
                    Ok(None) => {}
                    Err(error) => {
                        let _ = child.kill();
                        let _ = child.wait();
                        break ToolExit::WaitFailed(error.to_string());
                    }
                }
                if stop_rx.try_recv().is_ok() {
                    let _ = child.kill();
                    break child.wait().map_or_else(
                        |error| ToolExit::WaitFailed(error.to_string()),
                        ToolExit::Status,
                    );
                }
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    break ToolExit::TimedOut;
                }
                std::thread::sleep(Duration::from_millis(20));
            };
            let _ = exit_tx.send(exit);
        })
        .context("spawn tool waiter")?;
    let stdin = stdin_writer
        .zip(child_stdin)
        .map(|(writer, mut stdin)| {
            let stop = stop_tx.clone();
            std::thread::Builder::new()
                .name(format!("{tool}-stdin"))
                .spawn(move || {
                    let result = writer(&mut stdin);
                    drop(stdin);
                    if result.is_err() {
                        let _ = stop.send(());
                    }
                    result
                })
                .context("spawn tool stdin writer")
        })
        .transpose()?;
    let stderr_reader = std::thread::Builder::new()
        .name(format!("{tool}-stderr"))
        .spawn(move || {
            const TAIL_BYTES: usize = 64 * 1024;
            let mut buffer = Vec::new();
            let mut chunk = [0u8; 4096];
            let mut stderr = stderr;
            loop {
                match stderr.read(&mut chunk) {
                    Ok(0) => break,
                    Ok(read) => {
                        buffer.extend_from_slice(&chunk[..read]);
                        if buffer.len() > TAIL_BYTES {
                            let excess = buffer.len() - TAIL_BYTES;
                            buffer.drain(..excess);
                        }
                    }
                    Err(_) => break,
                }
            }
            buffer
        })
        .context("spawn tool stderr reader")?;
    let mut received = 0usize;
    let mut chunk = vec![0u8; 1 << 20];
    let mut failure = None;
    loop {
        match stdout.read(&mut chunk) {
            Ok(0) => break,
            Ok(read) => {
                if failure.is_none() && received.checked_add(read).is_none_or(|n| n > limit) {
                    failure = Some(anyhow::anyhow!(
                        "{tool} output exceeds the {limit}-byte limit"
                    ));
                    let _ = stop_tx.send(());
                }
                if failure.is_none() {
                    received += read;
                    if let Err(error) = consume(&chunk[..read]) {
                        failure = Some(error);
                        let _ = stop_tx.send(());
                    }
                }
            }
            Err(error) => {
                if failure.is_none() {
                    failure = Some(anyhow::anyhow!("read {tool} stdout: {error}"));
                }
                let _ = stop_tx.send(());
                break;
            }
        }
    }
    drop(stdout);
    let exit = exit_rx.recv().context("tool waiter stopped")?;
    let stderr = stderr_reader
        .join()
        .map_err(|_| anyhow::anyhow!("{tool} stderr reader panicked"))?;
    waiter
        .join()
        .map_err(|_| anyhow::anyhow!("{tool} waiter panicked"))?;
    if let Some(stdin) = stdin {
        match stdin.join() {
            Ok(Ok(())) => {}
            Ok(Err(error)) if failure.is_none() => failure = Some(error),
            Ok(Err(_)) => {}
            Err(_) if failure.is_none() => {
                failure = Some(anyhow::anyhow!("{tool} stdin writer panicked"));
            }
            Err(_) => {}
        }
    }
    if let Some(failure) = failure {
        return Err(failure);
    }
    match exit {
        ToolExit::TimedOut => bail!(
            "{tool} timed out after {} s and was killed",
            timeout.as_secs()
        ),
        ToolExit::WaitFailed(error) => bail!("wait for {tool}: {error}"),
        ToolExit::Status(status) => {
            if !status.success() {
                bail!("{tool} failed ({}): {}", status, stderr_tail(&stderr));
            }
        }
    }
    Ok(())
}

#[derive(Debug)]
pub(crate) struct ScratchFile {
    path: PathBuf,
}

impl ScratchFile {
    pub(crate) fn write(bytes: &[u8], extension: &str) -> Result<Self> {
        Self::write_in(&std::env::temp_dir(), bytes, extension)
    }

    fn write_in(directory: &Path, bytes: &[u8], extension: &str) -> Result<Self> {
        use std::os::unix::fs::OpenOptionsExt;
        ensure!(
            !extension.is_empty() && extension.bytes().all(|b| b.is_ascii_alphanumeric()),
            "invalid scratch extension"
        );
        for _ in 0..8 {
            let counter = SCRATCH_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = directory.join(format!(
                "flyingfish-media-{}-{counter}.{extension}",
                std::process::id()
            ));
            let mut file = match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&path)
            {
                Ok(file) => file,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(error)
                        .with_context(|| format!("create media scratch file {}", path.display()));
                }
            };
            if let Err(error) = file.write_all(bytes) {
                let _ = std::fs::remove_file(&path);
                return Err(error)
                    .with_context(|| format!("write media scratch file {}", path.display()));
            }
            return Ok(Self { path });
        }
        bail!(
            "could not create a media scratch file in {}",
            directory.display()
        )
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ScratchFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

pub fn write_png(writer: impl Write, width: usize, height: usize, rgb: &[u8]) -> Result<()> {
    let width = u32::try_from(width).context("PNG width exceeds u32")?;
    let height = u32::try_from(height).context("PNG height exceeds u32")?;
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(3))
        .context("PNG extent overflow")?;
    ensure!(expected > 0 && rgb.len() == expected, "PNG RGB extent");
    let mut encoder = png::Encoder::new(writer, width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(rgb)?;
    writer.finish()?;
    Ok(())
}

pub fn write_wav(
    mut writer: impl Write,
    sampling_rate: usize,
    channels: usize,
    samples: &[f32],
) -> Result<()> {
    let rate = u32::try_from(sampling_rate).context("WAV sampling rate exceeds u32")?;
    let channels = u16::try_from(channels).context("WAV channel count exceeds u16")?;
    ensure!(rate > 0 && channels > 0, "invalid WAV rate or channels");
    ensure!(
        !samples.is_empty() && samples.len().is_multiple_of(channels as usize),
        "WAV samples are not complete interleaved frames"
    );
    ensure!(
        samples.iter().all(|sample| sample.is_finite()),
        "WAV sample is not finite"
    );
    let data_bytes = samples
        .len()
        .checked_mul(2)
        .and_then(|n| u32::try_from(n).ok())
        .context("WAV data exceeds u32")?;
    let riff_bytes = 36u32
        .checked_add(data_bytes)
        .context("WAV RIFF size overflow")?;
    let block_align = channels
        .checked_mul(2)
        .context("WAV block align overflow")?;
    let byte_rate = rate
        .checked_mul(u32::from(block_align))
        .context("WAV byte rate overflow")?;
    writer.write_all(b"RIFF")?;
    writer.write_all(&riff_bytes.to_le_bytes())?;
    writer.write_all(b"WAVEfmt ")?;
    writer.write_all(&16u32.to_le_bytes())?;
    writer.write_all(&1u16.to_le_bytes())?;
    writer.write_all(&channels.to_le_bytes())?;
    writer.write_all(&rate.to_le_bytes())?;
    writer.write_all(&byte_rate.to_le_bytes())?;
    writer.write_all(&block_align.to_le_bytes())?;
    writer.write_all(&16u16.to_le_bytes())?;
    writer.write_all(b"data")?;
    writer.write_all(&data_bytes.to_le_bytes())?;
    for &sample in samples {
        let quantized = (sample * 32768.0).round().clamp(-32768.0, 32767.0) as i16;
        writer.write_all(&quantized.to_le_bytes())?;
    }
    Ok(())
}

fn y4m_extents(width: usize, height: usize, frames: usize, video: &[u8]) -> Result<(usize, usize)> {
    ensure!(
        width > 0 && height > 0 && width.is_multiple_of(2) && height.is_multiple_of(2),
        "Y4M dimensions must be positive and even"
    );
    let frame_bytes = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(3))
        .context("Y4M frame extent overflow")?;
    let expected = frame_bytes
        .checked_mul(frames)
        .context("Y4M video extent overflow")?;
    ensure!(frames > 0 && video.len() == expected, "Y4M RGB extent");
    Ok((frame_bytes, width * height))
}

fn rgb_to_yuv(rgb: [i64; 3], shift: u32) -> [u8; 3] {
    let [r, g, b] = rgb;
    let rounding = 1i64 << (shift - 1);
    let quantize = |value: i64, offset: i64| {
        ((value + rounding) >> shift)
            .saturating_add(offset)
            .clamp(0, 255) as u8
    };
    [
        quantize(19595 * r + 38470 * g + 7471 * b, 0),
        quantize(-11058 * r - 21710 * g + 32768 * b, 128),
        quantize(32768 * r - 27439 * g - 5329 * b, 128),
    ]
}

pub fn write_y4m(
    mut writer: impl Write,
    width: usize,
    height: usize,
    fps: (u32, u32),
    frames: usize,
    video: &[u8],
) -> Result<()> {
    let (frame_bytes, pixels) = y4m_extents(width, height, frames, video)?;
    ensure!(fps.0 > 0 && fps.1 > 0, "invalid Y4M frame rate");
    writeln!(
        writer,
        "YUV4MPEG2 W{width} H{height} F{}:{} Ip A1:1 C420jpeg XCOLORRANGE=FULL",
        fps.0, fps.1
    )?;
    let chroma_pixels = pixels / 4;
    let mut y = Vec::with_capacity(pixels);
    let mut u = Vec::with_capacity(chroma_pixels);
    let mut v = Vec::with_capacity(chroma_pixels);
    for frame in 0..frames {
        let rgb = &video[frame * frame_bytes..(frame + 1) * frame_bytes];
        y.clear();
        u.clear();
        v.clear();
        for pixel in rgb.as_chunks::<3>().0 {
            y.push(
                rgb_to_yuv(
                    [
                        i64::from(pixel[0]),
                        i64::from(pixel[1]),
                        i64::from(pixel[2]),
                    ],
                    16,
                )[0],
            );
        }
        for row in (0..height).step_by(2) {
            for col in (0..width).step_by(2) {
                let mut sum = [0i64; 3];
                for dy in 0..2 {
                    for dx in 0..2 {
                        let offset = ((row + dy) * width + col + dx) * 3;
                        for channel in 0..3 {
                            sum[channel] += i64::from(rgb[offset + channel]);
                        }
                    }
                }
                let yuv = rgb_to_yuv(sum, 18);
                u.push(yuv[1]);
                v.push(yuv[2]);
            }
        }
        writer.write_all(b"FRAME\n")?;
        writer.write_all(&y)?;
        writer.write_all(&u)?;
        writer.write_all(&v)?;
    }
    Ok(())
}

#[derive(Debug)]
pub struct Y4mVideo {
    pub width: usize,
    pub height: usize,
    pub fps: (u32, u32),
    pub frames: usize,
    pub video: Vec<u8>,
}

pub fn mux_mp4(
    output: &Path,
    video: Y4mVideo,
    audio: Option<(usize, usize, &[f32])>,
) -> Result<()> {
    mux_with_tool("ffmpeg", output, video, audio)
}

fn mux_with_tool(
    tool: &str,
    output: &Path,
    video: Y4mVideo,
    audio: Option<(usize, usize, &[f32])>,
) -> Result<()> {
    let Y4mVideo {
        width,
        height,
        fps,
        frames,
        video,
    } = video;
    ensure!(
        output
            .extension()
            .is_some_and(|extension| extension == "mp4"),
        "MP4 output must end in .mp4"
    );
    ensure!(
        !output.exists(),
        "output already exists: {}",
        output.display()
    );
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    ensure!(parent.is_dir(), "MP4 parent directory is missing");
    let output_name = output.to_str().context("MP4 path is not UTF-8")?;
    y4m_extents(width, height, frames, &video)?;
    ensure!(fps.0 > 0 && fps.1 > 0, "invalid MP4 frame rate");
    let scratch = match audio {
        Some((rate, channels, samples)) => {
            let mut bytes = Vec::new();
            write_wav(&mut bytes, rate, channels, samples)?;
            Some(ScratchFile::write_in(parent, &bytes, "wav")?)
        }
        None => None,
    };
    let mut args = vec![
        "-v".to_string(),
        "warning".to_string(),
        "-y".to_string(),
        "-f".to_string(),
        "yuv4mpegpipe".to_string(),
        "-color_range".to_string(),
        "jpeg".to_string(),
        "-colorspace".to_string(),
        "bt470bg".to_string(),
        "-i".to_string(),
        "pipe:0".to_string(),
    ];
    if let Some(scratch) = &scratch {
        args.push("-i".into());
        args.push(
            scratch
                .path()
                .to_str()
                .context("WAV scratch path is not UTF-8")?
                .to_string(),
        );
    }
    args.extend(["-map".into(), "0:v:0".into()]);
    if scratch.is_some() {
        args.extend(["-map".into(), "1:a:0".into(), "-c:a".into(), "aac".into()]);
    }
    args.extend([
        "-c:v".into(),
        "libx264".into(),
        "-crf".into(),
        "0".into(),
        "-vf".into(),
        "scale=in_range=full:out_range=full:in_color_matrix=bt601:out_color_matrix=bt601".into(),
        "-pix_fmt".into(),
        "yuvj420p".into(),
        "-color_range".into(),
        "jpeg".into(),
        "-colorspace".into(),
        "bt470bg".into(),
        "-color_primaries".into(),
        "bt470bg".into(),
        "-color_trc".into(),
        "bt709".into(),
        output_name.to_string(),
    ]);
    let args: Vec<_> = args.iter().map(String::as_str).collect();
    let input: StdinWriter =
        Box::new(move |writer| write_y4m(writer, width, height, fps, frames, &video));
    let result = tool_output_with_stdin(
        "MP4 mux",
        tool,
        &args,
        MUX_TIMEOUT,
        MUX_OUTPUT_LIMIT,
        Some(input),
        |_| Ok(()),
    )
    .and_then(|_| {
        ensure!(output.is_file(), "ffmpeg produced no MP4 output");
        Ok(())
    });
    drop(scratch);
    if result.is_err() {
        let _ = std::fs::remove_file(output);
    }
    result
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct TextSegment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct Transcription {
    pub text: String,
    pub segments: Vec<TextSegment>,
    pub words: Vec<TextSegment>,
}

impl Transcription {
    pub fn whole_clip(text: String, seconds: f64) -> Result<Self> {
        ensure!(
            seconds.is_finite() && seconds >= 0.0,
            "invalid clip duration"
        );
        Ok(Self {
            text: text.clone(),
            segments: vec![TextSegment {
                start: 0.0,
                end: seconds,
                text,
            }],
            words: Vec::new(),
        })
    }
}

pub fn read_wav(mut reader: impl Read, expected_rate: u32) -> Result<Vec<f32>> {
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    ensure!(bytes.len() >= 12, "WAV header is truncated");
    ensure!(
        &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE",
        "invalid WAV header"
    );
    let mut offset = 12usize;
    let mut format = None;
    let mut data = None;
    while offset + 8 <= bytes.len() {
        let size = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into()?) as usize;
        let body = offset + 8;
        let end = body
            .checked_add(size)
            .context("WAV chunk extent overflow")?;
        ensure!(end <= bytes.len(), "WAV chunk is truncated");
        match &bytes[offset..offset + 4] {
            b"fmt " => {
                ensure!(size >= 16, "short WAV format chunk");
                let encoding = u16::from_le_bytes(bytes[body..body + 2].try_into()?);
                let channels = u16::from_le_bytes(bytes[body + 2..body + 4].try_into()?);
                let rate = u32::from_le_bytes(bytes[body + 4..body + 8].try_into()?);
                let align = u16::from_le_bytes(bytes[body + 12..body + 14].try_into()?);
                let bits = u16::from_le_bytes(bytes[body + 14..body + 16].try_into()?);
                ensure!(channels == 1, "WAV must be mono");
                ensure!(
                    rate == expected_rate,
                    "WAV rate {rate} does not match model rate {expected_rate}; convert with: ffmpeg -i <input> -ar {expected_rate} -ac 1 <output>"
                );
                ensure!(
                    (encoding == 1 && matches!(bits, 16 | 24) || encoding == 3 && bits == 32),
                    "WAV must be PCM16, PCM24 or F32"
                );
                ensure!(align == channels * (bits / 8), "WAV block align mismatch");
                format = Some(bits);
            }
            b"data" => data = Some(body..end),
            _ => {}
        }
        offset = end.checked_add(size & 1).context("WAV padding overflow")?;
    }
    let bits = format.context("missing WAV format chunk")?;
    let data = data.context("missing WAV data chunk")?;
    let width = usize::from(bits / 8);
    ensure!(
        !data.is_empty() && data.len() % width == 0,
        "invalid WAV data extent"
    );
    bytes[data]
        .chunks_exact(width)
        .map(|sample| {
            Ok(match bits {
                16 => f32::from(i16::from_le_bytes([sample[0], sample[1]])) / 32768.0,
                24 => {
                    let value = i32::from(sample[0])
                        | (i32::from(sample[1]) << 8)
                        | (i32::from(sample[2]) << 16);
                    ((value << 8) >> 8) as f32 / 8388608.0
                }
                32 => {
                    let value = f32::from_le_bytes(sample.try_into()?);
                    ensure!(value.is_finite(), "WAV F32 sample is not finite");
                    value
                }
                _ => bail!("unsupported WAV sample width"),
            })
        })
        .collect()
}

fn validate_segment(segment: &TextSegment) -> Result<()> {
    ensure!(
        segment.start.is_finite()
            && segment.end.is_finite()
            && segment.start >= 0.0
            && segment.end >= segment.start,
        "invalid text segment time range"
    );
    Ok(())
}

fn milliseconds(seconds: f64) -> Result<u64> {
    ensure!(seconds.is_finite() && seconds >= 0.0, "invalid timestamp");
    let millis = (seconds * 1000.0).round();
    ensure!(
        millis >= 0.0 && millis <= u64::MAX as f64,
        "timestamp overflow"
    );
    Ok(millis as u64)
}

fn timestamp(seconds: f64, decimal: char) -> Result<String> {
    let millis = milliseconds(seconds)?;
    let hours = millis / 3_600_000;
    let minutes = millis / 60_000 % 60;
    let seconds = millis / 1000 % 60;
    let millis = millis % 1000;
    Ok(format!(
        "{hours:02}:{minutes:02}:{seconds:02}{decimal}{millis:03}"
    ))
}

pub fn write_srt(mut writer: impl Write, segments: &[TextSegment]) -> Result<()> {
    for (index, segment) in segments.iter().enumerate() {
        validate_segment(segment)?;
        writeln!(
            writer,
            "{}\n{} --> {}\n{}\n",
            index + 1,
            timestamp(segment.start, ',')?,
            timestamp(segment.end, ',')?,
            segment.text
        )?;
    }
    Ok(())
}

pub fn write_vtt(mut writer: impl Write, segments: &[TextSegment]) -> Result<()> {
    writer.write_all(b"WEBVTT\n\n")?;
    for segment in segments {
        validate_segment(segment)?;
        writeln!(
            writer,
            "{} --> {}\n{}\n",
            timestamp(segment.start, '.')?,
            timestamp(segment.end, '.')?,
            segment.text
        )?;
    }
    Ok(())
}

pub fn write_rttm(mut writer: impl Write, file_id: &str, segments: &[TextSegment]) -> Result<()> {
    ensure!(
        !file_id.is_empty() && !file_id.chars().any(char::is_whitespace),
        "RTTM file id must be one token"
    );
    for segment in segments {
        validate_segment(segment)?;
        ensure!(
            !segment.text.is_empty() && !segment.text.chars().any(char::is_whitespace),
            "RTTM speaker label must be one token"
        );
        writeln!(
            writer,
            "SPEAKER {file_id} 1 {:.3} {:.3} <NA> <NA> {} <NA> <NA>",
            segment.start,
            segment.end - segment.start,
            segment.text
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn test_dir(name: &str) -> Result<PathBuf> {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "flyingfish-media-test-{}-{}-{}",
            std::process::id(),
            name,
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path)?;
        Ok(path)
    }

    #[test]
    fn png_round_trip_preserves_rgb8() -> Result<()> {
        let pixels = [255, 0, 0, 0, 128, 255];
        let mut bytes = Vec::new();
        write_png(&mut bytes, 2, 1, &pixels)?;
        let mut decoder = png::Decoder::new(Cursor::new(bytes));
        decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
        let mut reader = decoder.read_info()?;
        let mut decoded = vec![0; reader.output_buffer_size().context("PNG buffer size")?];
        let info = reader.next_frame(&mut decoded)?;
        assert_eq!(info.color_type, png::ColorType::Rgb);
        assert_eq!(info.bit_depth, png::BitDepth::Eight);
        assert_eq!(&decoded[..info.buffer_size()], pixels);
        assert!(write_png(Vec::new(), 2, 1, &pixels[..5]).is_err());
        let mut valid = Vec::new();
        write_png(&mut valid, 2, 1, &pixels)?;
        struct FinalWriteFailure {
            bytes: Vec<u8>,
            limit: usize,
        }
        impl Write for FinalWriteFailure {
            fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
                if self.bytes.len() + buffer.len() > self.limit {
                    return Err(std::io::Error::other("injected final PNG write failure"));
                }
                self.bytes.extend_from_slice(buffer);
                Ok(buffer.len())
            }

            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let error = write_png(
            &mut FinalWriteFailure {
                bytes: Vec::new(),
                limit: valid.len() - 4,
            },
            2,
            1,
            &pixels,
        )
        .expect_err("final PNG write failure must propagate")
        .to_string();
        assert!(
            error.contains("injected final PNG write failure"),
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn wav_round_trip_preserves_pcm16_samples() -> Result<()> {
        let samples = [-1.0, -0.5, 0.0, 0.5, 1.0, -1.0];
        let mut bytes = Vec::new();
        write_wav(&mut bytes, 48000, 2, &samples)?;
        assert_eq!(bytes.len(), 44 + samples.len() * 2);
        assert_eq!(&bytes[..4], b"RIFF");
        assert_eq!(&bytes[8..16], b"WAVEfmt ");
        assert_eq!(u32::from_le_bytes(bytes[24..28].try_into()?), 48000);
        assert_eq!(u16::from_le_bytes(bytes[22..24].try_into()?), 2);
        assert_eq!(u32::from_le_bytes(bytes[40..44].try_into()?), 12);
        let decoded: Vec<i16> = bytes[44..]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|bytes| i16::from_le_bytes(*bytes))
            .collect();
        assert_eq!(decoded, [-32768, -16384, 0, 16384, 32767, -32768]);
        for (sample, &quantized) in samples.iter().zip(&decoded) {
            let expected = if *sample == 1.0 {
                32767.0 / 32768.0
            } else {
                *sample
            };
            assert_eq!(f32::from(quantized) / 32768.0, expected);
        }
        assert!(write_wav(Vec::new(), 48000, 2, &[0.0]).is_err());
        assert!(write_wav(Vec::new(), 48000, 1, &[f32::NAN]).is_err());
        Ok(())
    }

    #[test]
    fn y4m_round_trip_preserves_frame_and_chroma_layout() -> Result<()> {
        assert_eq!(rgb_to_yuv([0; 3], 18), [0, 128, 128]);
        assert_eq!(rgb_to_yuv([1, 3, 7], 16), [3, 130, 127]);
        let red = [255, 0, 0];
        let green = [0, 255, 0];
        let mixed = [red, green, [0, 0, 255], [255, 255, 255]].concat();
        let video = [red.repeat(4), mixed].concat();
        let mut bytes = Vec::new();
        write_y4m(&mut bytes, 2, 2, (24, 1), 2, &video)?;
        let header = b"YUV4MPEG2 W2 H2 F24:1 Ip A1:1 C420jpeg XCOLORRANGE=FULL\n";
        assert!(bytes.starts_with(header));
        let frame_bytes = 6 + 4 + 1 + 1;
        assert_eq!(bytes.len(), header.len() + 2 * frame_bytes);
        let first = &bytes[header.len()..header.len() + frame_bytes];
        assert_eq!(&first[..6], b"FRAME\n");
        assert_eq!(&first[6..10], [76; 4]);
        assert_eq!((first[10], first[11]), (85, 255));
        let second = &bytes[header.len() + frame_bytes..];
        assert_eq!(&second[..6], b"FRAME\n");
        let expected_y: Vec<u8> = [red, green, [0, 0, 255], [255, 255, 255]]
            .iter()
            .map(|pixel| {
                (0.299 * f64::from(pixel[0])
                    + 0.587 * f64::from(pixel[1])
                    + 0.114 * f64::from(pixel[2]))
                .round() as u8
            })
            .collect();
        assert_eq!(&second[6..10], expected_y);
        let mean = |channel: usize| {
            [red, green, [0, 0, 255], [255, 255, 255]]
                .iter()
                .map(|pixel| f64::from(pixel[channel]))
                .sum::<f64>()
                / 4.0
        };
        let (r, g, b) = (mean(0), mean(1), mean(2));
        assert_eq!(
            (second[10], second[11]),
            (
                (-0.168736 * r - 0.331264 * g + 0.5 * b + 128.0).round() as u8,
                (0.5 * r - 0.418688 * g - 0.081312 * b + 128.0).round() as u8
            )
        );
        assert!(write_y4m(Vec::new(), 3, 2, (24, 1), 1, &[0; 18]).is_err());
        Ok(())
    }

    #[test]
    fn wav_reader_decodes_pcm16_pcm24_and_f32_with_chunk_padding() -> Result<()> {
        fn container(encoding: u16, bits: u16, rate: u32, data: &[u8]) -> Vec<u8> {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(b"RIFF");
            bytes.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
            bytes.extend_from_slice(b"WAVE");
            bytes.extend_from_slice(b"JUNK");
            bytes.extend_from_slice(&1u32.to_le_bytes());
            bytes.extend_from_slice(&[0x7f, 0]);
            bytes.extend_from_slice(b"fmt ");
            bytes.extend_from_slice(&16u32.to_le_bytes());
            bytes.extend_from_slice(&encoding.to_le_bytes());
            bytes.extend_from_slice(&1u16.to_le_bytes());
            bytes.extend_from_slice(&rate.to_le_bytes());
            let align = bits / 8;
            bytes.extend_from_slice(&(rate * u32::from(align)).to_le_bytes());
            bytes.extend_from_slice(&align.to_le_bytes());
            bytes.extend_from_slice(&bits.to_le_bytes());
            bytes.extend_from_slice(b"data");
            bytes.extend_from_slice(&(data.len() as u32).to_le_bytes());
            bytes.extend_from_slice(data);
            bytes
        }

        let pcm16: Vec<_> = [-32768i16, 0, 32767]
            .into_iter()
            .flat_map(i16::to_le_bytes)
            .collect();
        let decoded = read_wav(Cursor::new(container(1, 16, 16000, &pcm16)), 16000)?;
        assert_eq!(decoded, [-1.0, 0.0, 32767.0 / 32768.0]);
        let pcm24: Vec<_> = [-8388608i32, 0, 8388607]
            .into_iter()
            .flat_map(|value| {
                let bytes = value.to_le_bytes();
                [bytes[0], bytes[1], bytes[2]]
            })
            .collect();
        let decoded = read_wav(Cursor::new(container(1, 24, 16000, &pcm24)), 16000)?;
        assert_eq!(decoded, [-1.0, 0.0, 8388607.0 / 8388608.0]);
        let f32_data: Vec<_> = [-1.0f32, 0.25, 1.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let decoded = read_wav(Cursor::new(container(3, 32, 16000, &f32_data)), 16000)?;
        assert_eq!(decoded, [-1.0, 0.25, 1.0]);
        let nan = f32::NAN.to_le_bytes();
        assert!(read_wav(Cursor::new(container(3, 32, 16000, &nan)), 16000).is_err());
        let error = read_wav(Cursor::new(container(1, 16, 8000, &pcm16)), 16000)
            .expect_err("sample-rate mismatch must fail")
            .to_string();
        assert!(
            error.contains("ffmpeg") && error.contains("-ar 16000"),
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn timed_text_writers_preserve_timestamps_and_overlap() -> Result<()> {
        let segments = vec![
            TextSegment {
                start: 0.0,
                end: 1.25,
                text: "hello".into(),
            },
            TextSegment {
                start: 1.25,
                end: 2.5,
                text: "world".into(),
            },
        ];
        let mut srt = Vec::new();
        write_srt(&mut srt, &segments)?;
        assert_eq!(
            String::from_utf8(srt)?,
            "1\n00:00:00,000 --> 00:00:01,250\nhello\n\n2\n00:00:01,250 --> 00:00:02,500\nworld\n\n"
        );
        let mut vtt = Vec::new();
        write_vtt(&mut vtt, &segments)?;
        assert_eq!(
            String::from_utf8(vtt)?,
            "WEBVTT\n\n00:00:00.000 --> 00:00:01.250\nhello\n\n00:00:01.250 --> 00:00:02.500\nworld\n\n"
        );
        let speakers = vec![
            TextSegment {
                start: 0.0,
                end: 2.0,
                text: "speaker_0".into(),
            },
            TextSegment {
                start: 1.0,
                end: 2.5,
                text: "speaker_1".into(),
            },
        ];
        let mut rttm = Vec::new();
        write_rttm(&mut rttm, "clip", &speakers)?;
        assert_eq!(
            String::from_utf8(rttm)?,
            "SPEAKER clip 1 0.000 2.000 <NA> <NA> speaker_0 <NA> <NA>\nSPEAKER clip 1 1.000 1.500 <NA> <NA> speaker_1 <NA> <NA>\n"
        );
        assert!(
            write_srt(
                Vec::new(),
                &[TextSegment {
                    start: f64::NAN,
                    end: 1.0,
                    text: "bad".into(),
                }]
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires ffmpeg on PATH"]
    fn mp4_mux_round_trip_decodes_video_and_audio() -> Result<()> {
        let directory = test_dir("mp4-round-trip")?;
        let output = directory.join("out.mp4");
        let (width, height, frames) = (4, 2, 2);
        let video: Vec<u8> = (0..frames)
            .flat_map(|frame| {
                let a = [frame as u8, (3 * frame) as u8, (7 * frame) as u8];
                let index = frame * 2 + 1;
                let b = [index as u8, (3 * index) as u8, (7 * index) as u8];
                [a, a, b, b, a, a, b, b].concat()
            })
            .collect();
        let audio: Vec<f32> = (0..480)
            .flat_map(|index| [index as f32 / 4800.0, -(index as f32) / 4800.0])
            .collect();
        let mut encoded = Vec::new();
        write_y4m(&mut encoded, width, height, (24, 1), frames, &video)?;
        let header_len = b"YUV4MPEG2 W4 H2 F24:1 Ip A1:1 C420jpeg XCOLORRANGE=FULL\n".len();
        assert_eq!(
            &encoded[header_len + 6..header_len + 6 + 12],
            &[0, 0, 3, 3, 0, 0, 3, 3, 128, 130, 128, 127]
        );
        let raw_y4m = directory.join("input.y4m");
        std::fs::write(&raw_y4m, encoded)?;
        let raw_decoded = tool_stdout(
            "Y4M test decode",
            "ffmpeg",
            &[
                "-v",
                "error",
                "-i",
                raw_y4m.to_str().context("test Y4M path")?,
                "-map",
                "0:v:0",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ],
            Duration::from_secs(60),
            video.len() + 1,
        )?;
        assert_eq!(raw_decoded.len(), video.len());
        let raw_delta = raw_decoded
            .iter()
            .zip(&video)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .context("decoded Y4M is empty")?;
        assert!(raw_delta <= 2, "raw Y4M RGB max delta {raw_delta}");
        mux_mp4(
            &output,
            Y4mVideo {
                width,
                height,
                fps: (24, 1),
                frames,
                video: video.clone(),
            },
            Some((48000, 2, &audio)),
        )?;
        let output_text = output.to_str().context("test output path")?;
        let probe = tool_stdout(
            "MP4 test probe",
            "ffprobe",
            &[
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,width,height",
                "-of",
                "json",
                output_text,
            ],
            Duration::from_secs(60),
            64 * 1024,
        )?;
        let probe: serde_json::Value = serde_json::from_slice(&probe)?;
        let streams = probe["streams"].as_array().context("probe streams")?;
        assert!(streams.iter().any(|stream| {
            stream["codec_type"] == "video"
                && stream["width"] == width
                && stream["height"] == height
        }));
        assert!(streams.iter().any(|stream| stream["codec_type"] == "audio"));
        let decoded = tool_stdout(
            "MP4 test decode",
            "ffmpeg",
            &[
                "-v",
                "error",
                "-i",
                output_text,
                "-map",
                "0:v:0",
                "-vf",
                "scale=in_range=full:out_range=full:in_color_matrix=bt601:out_color_matrix=bt601",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ],
            Duration::from_secs(60),
            video.len() + 1,
        )?;
        assert_eq!(decoded.len(), video.len());
        let max_delta = decoded
            .iter()
            .zip(&video)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .context("decoded MP4 is empty")?;
        assert!(
            max_delta <= 2,
            "MP4 RGB max delta {max_delta}; decoded={decoded:?}; original={video:?}"
        );
        assert!(!directory.read_dir()?.any(|entry| {
            entry
                .expect("read test directory")
                .path()
                .extension()
                .is_some_and(|extension| extension == "wav")
        }));
        std::fs::remove_dir_all(directory)?;
        Ok(())
    }

    #[test]
    fn failed_mp4_mux_removes_temporary_wav_and_partial_output() -> Result<()> {
        let directory = test_dir("mp4-failure")?;
        let output = directory.join("out.mp4");
        let error = mux_with_tool(
            "ffmpeg-not-on-path-kimi",
            &output,
            Y4mVideo {
                width: 2,
                height: 2,
                fps: (24, 1),
                frames: 1,
                video: vec![255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0],
            },
            Some((48000, 1, &[0.0; 48])),
        )
        .expect_err("missing ffmpeg must fail")
        .to_string();
        assert!(error.contains("MP4 mux requires ffmpeg-not-on-path-kimi on PATH"));
        assert_eq!(directory.read_dir()?.count(), 0);
        std::fs::remove_dir_all(directory)?;
        Ok(())
    }
}

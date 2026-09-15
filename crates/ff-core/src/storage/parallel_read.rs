use std::{
    fs::File,
    io::{self, ErrorKind},
    os::unix::fs::FileExt,
    path::Path,
};

/// Select shared file access or independent per-reader readahead windows.
#[derive(Clone, Copy)]
pub enum ParallelReadSource<'a> {
    Shared(&'a File),
    Reopen(&'a Path),
}

/// Fill disjoint slices with buffered positional reads, joining every reader on failure.
pub fn read_parallel_into(
    source: ParallelReadSource<'_>,
    offset: u64,
    output: &mut [u8],
    readers: usize,
) -> io::Result<()> {
    if readers == 0 || offset.checked_add(output.len() as u64).is_none() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "invalid parallel read range or reader count",
        ));
    }
    if output.is_empty() {
        return Ok(());
    }
    let chunk = output.len().div_ceil(readers);
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for (index, slice) in output.chunks_mut(chunk).enumerate() {
            handles.push(scope.spawn(move || {
                let opened;
                let file = match source {
                    ParallelReadSource::Shared(file) => file,
                    ParallelReadSource::Reopen(path) => {
                        opened = File::open(path)?;
                        &opened
                    }
                };
                read_exact_at(
                    |buffer, position| file.read_at(buffer, position),
                    offset + (index * chunk) as u64,
                    slice,
                )
            }));
        }
        let mut failure = None;
        for handle in handles {
            let result = handle
                .join()
                .unwrap_or_else(|_| Err(io::Error::other("parallel reader panicked")));
            if let Err(error) = result {
                failure.get_or_insert(error);
            }
        }
        failure.map_or(Ok(()), Err)
    })
}

fn read_exact_at(
    mut read: impl FnMut(&mut [u8], u64) -> io::Result<usize>,
    mut offset: u64,
    mut output: &mut [u8],
) -> io::Result<()> {
    while !output.is_empty() {
        let wanted = output.len().min(4 << 20);
        match read(&mut output[..wanted], offset) {
            Ok(0) => {
                return Err(io::Error::new(
                    ErrorKind::UnexpectedEof,
                    "file ended during positional read",
                ));
            }
            Ok(count) => {
                output = &mut output[count..];
                offset += count as u64;
            }
            Err(error) if error.kind() == ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, Write};

    #[test]
    fn parallel_ranges_match_file_without_moving_cursor() -> io::Result<()> {
        let mut file = tempfile::NamedTempFile::new()?;
        let bytes = (0..(5 << 20) + 37)
            .map(|n| (n % 251) as u8)
            .collect::<Vec<_>>();
        file.write_all(&bytes)?;
        let cursor = file.stream_position()?;
        for source in [
            ParallelReadSource::Shared(file.as_file()),
            ParallelReadSource::Reopen(file.path()),
        ] {
            for readers in [1, 3, 8] {
                let mut output = vec![0; bytes.len() - 26];
                read_parallel_into(source, 13, &mut output, readers)?;
                assert_eq!(output, bytes[13..bytes.len() - 13]);
            }
            let mut small = [0; 3];
            read_parallel_into(source, 7, &mut small, 8)?;
            assert_eq!(small, bytes[7..10]);
            read_parallel_into(source, 0, &mut [], 2)?;
        }
        assert_eq!(file.stream_position()?, cursor);
        Ok(())
    }

    #[test]
    fn reports_open_eof_and_invalid_range_errors() -> io::Result<()> {
        let file = tempfile::NamedTempFile::new()?;
        let source = ParallelReadSource::Shared(file.as_file());
        let mut output = [0; 17];
        assert_eq!(
            read_parallel_into(source, 0, &mut output, 4)
                .unwrap_err()
                .kind(),
            ErrorKind::UnexpectedEof
        );
        assert_eq!(
            read_parallel_into(source, u64::MAX, &mut output, 4)
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidInput
        );
        assert_eq!(
            read_parallel_into(source, 0, &mut output, 0)
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidInput
        );
        let missing = file.path().join("missing");
        assert!(
            read_parallel_into(ParallelReadSource::Reopen(&missing), 0, &mut output, 4).is_err()
        );
        Ok(())
    }

    #[test]
    fn short_and_interrupted_reads_preserve_offsets() -> io::Result<()> {
        let mut attempts = 0;
        let mut output = [0; 7];
        read_exact_at(
            |buffer, offset| {
                attempts += 1;
                if attempts % 2 == 1 {
                    return Err(ErrorKind::Interrupted.into());
                }
                let count = buffer.len().min(2);
                for (index, byte) in buffer[..count].iter_mut().enumerate() {
                    *byte = (offset + index as u64) as u8;
                }
                Ok(count)
            },
            11,
            &mut output,
        )?;
        assert_eq!(output, [11, 12, 13, 14, 15, 16, 17]);
        assert_eq!(attempts, 8);
        Ok(())
    }

    #[test]
    fn read_error_keeps_original_cause() {
        let error = read_exact_at(
            |_, _| Err(io::Error::new(ErrorKind::PermissionDenied, "reader denied")),
            0,
            &mut [0; 1],
        )
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::PermissionDenied);
        assert_eq!(error.to_string(), "reader denied");
    }
}

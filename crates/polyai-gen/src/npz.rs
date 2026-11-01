//! A minimal `.npz` writer: a zip (deflate) of `.npy` version 1.0 arrays, little endian, C order.

use std::io::{Seek, Write};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipWriter};

pub trait NpyType: Copy {
    const DESCR: &'static str;
    fn put(self, out: &mut Vec<u8>);
}

macro_rules! npy_type {
    ($t:ty, $d:expr) => {
        impl NpyType for $t {
            const DESCR: &'static str = $d;
            fn put(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }
        }
    };
}
npy_type!(i8, "|i1");
npy_type!(i16, "<i2");
npy_type!(i32, "<i4");
npy_type!(f32, "<f4");

pub struct Npz<W: Write + Seek> {
    zip: ZipWriter<W>,
    opts: SimpleFileOptions,
}

impl<W: Write + Seek> Npz<W> {
    pub fn new(w: W) -> Self {
        let opts = SimpleFileOptions::default()
            .compression_method(CompressionMethod::Deflated)
            .compression_level(Some(3))
            .large_file(true);
        Self {
            zip: ZipWriter::new(w),
            opts,
        }
    }

    /// Add array `name` (stored as `name.npy`) with the given shape.
    pub fn add<T: NpyType>(
        &mut self,
        name: &str,
        shape: &[usize],
        data: &[T],
    ) -> std::io::Result<()> {
        assert_eq!(
            shape.iter().product::<usize>(),
            data.len(),
            "shape of {name}"
        );
        let mut body = Vec::with_capacity(data.len() * std::mem::size_of::<T>());
        for &x in data {
            x.put(&mut body);
        }
        self.add_bytes(name, T::DESCR, shape, &body)
    }

    /// Add a 1-D array of strings as numpy `<U{L}` (UCS-4, `L` the longest string in characters, at least 1).
    pub fn add_strs<S: AsRef<str>>(&mut self, name: &str, data: &[S]) -> std::io::Result<()> {
        let width = data
            .iter()
            .map(|s| s.as_ref().chars().count())
            .max()
            .unwrap_or(0)
            .max(1);
        let mut body = Vec::with_capacity(data.len() * width * 4);
        for s in data {
            let mut chars = 0;
            for c in s.as_ref().chars() {
                body.extend_from_slice(&(c as u32).to_le_bytes());
                chars += 1;
            }
            body.extend(std::iter::repeat(0u8).take((width - chars) * 4));
        }
        self.add_bytes(name, &format!("<U{width}"), &[data.len()], &body)
    }

    fn add_bytes(
        &mut self,
        name: &str,
        descr: &str,
        shape: &[usize],
        body: &[u8],
    ) -> std::io::Result<()> {
        let dims = match shape {
            [n] => format!("({n},)"),
            _ => format!(
                "({})",
                shape
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        };
        let mut header =
            format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {dims}, }}")
                .into_bytes();
        // Magic (6) + version (2) + header length (2) + header + newline is padded to a multiple of 64.
        let total = 10 + header.len() + 1;
        header.extend(std::iter::repeat(b' ').take((64 - total % 64) % 64));
        header.push(b'\n');
        let mut bytes = Vec::with_capacity(10 + header.len() + body.len());
        bytes.extend_from_slice(b"\x93NUMPY\x01\x00");
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(body);
        self.zip
            .start_file(format!("{name}.npy"), self.opts)
            .map_err(std::io::Error::other)?;
        self.zip.write_all(&bytes)
    }

    pub fn finish(self) -> std::io::Result<W> {
        self.zip.finish().map_err(std::io::Error::other)
    }
}

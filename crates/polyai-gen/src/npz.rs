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
        let opts = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated).compression_level(Some(3)).large_file(true);
        Self { zip: ZipWriter::new(w), opts }
    }

    /// Add array `name` (stored as `name.npy`) with the given shape.
    pub fn add<T: NpyType>(&mut self, name: &str, shape: &[usize], data: &[T]) -> std::io::Result<()> {
        assert_eq!(shape.iter().product::<usize>(), data.len(), "shape of {name}");
        let dims = match shape {
            [n] => format!("({n},)"),
            _ => format!("({})", shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")),
        };
        let mut header = format!("{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}", T::DESCR, dims).into_bytes();
        // Magic (6) + version (2) + header length (2) + header + newline is padded to a multiple of 64.
        let total = 10 + header.len() + 1;
        header.extend(std::iter::repeat(b' ').take((64 - total % 64) % 64));
        header.push(b'\n');
        let mut bytes = Vec::with_capacity(10 + header.len() + data.len() * std::mem::size_of::<T>());
        bytes.extend_from_slice(b"\x93NUMPY\x01\x00");
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(&header);
        for &x in data {
            x.put(&mut bytes);
        }
        self.zip.start_file(format!("{name}.npy"), self.opts).map_err(std::io::Error::other)?;
        self.zip.write_all(&bytes)
    }

    pub fn finish(self) -> std::io::Result<W> {
        self.zip.finish().map_err(std::io::Error::other)
    }
}

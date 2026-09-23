use std::fs::File;

use arrow_array::{Array, LargeStringArray, StringArray};
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

/// Rows joined as JSON Lines: every row is followed by `\n`.
pub struct Batch {
    pub bytes: Vec<u8>,
    pub rows: usize,
}

#[derive(Default)]
pub struct Corpus {
    pub batches: Vec<Batch>,
    pub file_rows: usize,
    pub null_rows: usize,
    pub empty_rows: usize,
    pub newline_rows: usize,
    pub max_row_bytes: usize,
}

impl Corpus {
    pub fn rows(&self) -> usize {
        self.batches.iter().map(|b| b.rows).sum()
    }
    pub fn bytes(&self) -> usize {
        self.batches.iter().map(|b| b.bytes.len()).sum()
    }
}

pub fn load(
    path: &str,
    column: &str,
    batch_bytes: usize,
    max_rows: Option<usize>,
) -> Result<Corpus, String> {
    let file = File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| e.to_string())?;
    let idx = builder
        .parquet_schema()
        .columns()
        .iter()
        .position(|c| c.name() == column)
        .ok_or_else(|| format!("no column {column:?} in {path}"))?;
    let mask = ProjectionMask::leaves(builder.parquet_schema(), [idx]);
    let reader = builder
        .with_projection(mask)
        .with_batch_size(4096)
        .build()
        .map_err(|e| e.to_string())?;

    let mut corpus = Corpus::default();
    let mut cur = Batch {
        bytes: Vec::new(),
        rows: 0,
    };
    let push = |value: Option<&str>, corpus: &mut Corpus, cur: &mut Batch| {
        corpus.file_rows += 1;
        let Some(v) = value else {
            corpus.null_rows += 1;
            return;
        };
        if v.is_empty() {
            corpus.empty_rows += 1;
            return;
        }
        if v.contains('\n') {
            corpus.newline_rows += 1;
            return;
        }
        if !cur.bytes.is_empty() && cur.bytes.len() + v.len() + 1 > batch_bytes {
            corpus.batches.push(std::mem::replace(
                cur,
                Batch {
                    bytes: Vec::new(),
                    rows: 0,
                },
            ));
        }
        cur.bytes.extend_from_slice(v.as_bytes());
        cur.bytes.push(b'\n');
        cur.rows += 1;
        corpus.max_row_bytes = corpus.max_row_bytes.max(v.len());
    };

    'outer: for batch in reader {
        let batch = batch.map_err(|e| e.to_string())?;
        let col = batch.column(0);
        if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
            for i in 0..a.len() {
                if max_rows.is_some_and(|m| corpus.rows() + cur.rows >= m) {
                    break 'outer;
                }
                push((!a.is_null(i)).then(|| a.value(i)), &mut corpus, &mut cur);
            }
        } else if let Some(a) = col.as_any().downcast_ref::<LargeStringArray>() {
            for i in 0..a.len() {
                if max_rows.is_some_and(|m| corpus.rows() + cur.rows >= m) {
                    break 'outer;
                }
                push((!a.is_null(i)).then(|| a.value(i)), &mut corpus, &mut cur);
            }
        } else {
            return Err(format!(
                "column {column:?} is {:?}, not a string",
                col.data_type()
            ));
        }
    }
    if cur.rows > 0 {
        corpus.batches.push(cur);
    }
    Ok(corpus)
}

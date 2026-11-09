//! The dispatch-count formula from `native/src/attention_math.cpp` (`validate_qsa_shape`, `qsa_block_budget`, `qsa_dispatches`).
//!
//! Only this production counting rule is ported: the V8 admission gate uses it to check physical dispatches. The CPU attention references in that file stay C++ as independent oracles.

/// `QsaShape`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QsaShape {
    pub channels: u32,
    pub query_heads: u32,
    pub kv_heads: u32,
    pub head_width: u32,
    pub index_heads: u32,
    pub index_width: u32,
    pub rotary_width: u32,
    pub compression: u32,
    pub token_budget: u32,
    pub capacity: u32,
}

/// The C++ `FormatError` messages raised here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QsaError {
    /// `validate_qsa_shape` refused the geometry.
    Shape,
    /// `qsa_dispatches` was asked about a position beyond the cache.
    Capacity,
}

impl std::fmt::Display for QsaError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Shape => "unsupported sparse-attention shape",
            Self::Capacity => "attention cache capacity exceeded",
        })
    }
}

impl std::error::Error for QsaError {}

pub fn validate_qsa_shape(s: QsaShape) -> Result<(), QsaError> {
    let invalid = s.channels == 0
        || s.channels > 65536
        || s.query_heads == 0
        || s.query_heads > 64
        || s.kv_heads == 0
        || s.kv_heads > s.query_heads
        || !s.query_heads.is_multiple_of(s.kv_heads)
        || s.head_width == 0
        || s.head_width > 256
        || !s.head_width.is_multiple_of(2)
        || s.index_heads == 0
        || s.index_heads > 8
        || s.index_width == 0
        || s.index_width > 256
        || !s.index_width.is_multiple_of(2)
        || s.rotary_width == 0
        || !s.rotary_width.is_multiple_of(2)
        || s.rotary_width > s.head_width
        || s.rotary_width > s.index_width
        || s.compression == 0
        || s.compression > 16
        || s.token_budget == 0
        || !s.token_budget.is_multiple_of(s.compression)
        || s.token_budget / s.compression > 512
        || s.capacity == 0
        || s.capacity > 65536;
    if invalid {
        Err(QsaError::Shape)
    } else {
        Ok(())
    }
}

pub fn qsa_block_budget(s: QsaShape) -> Result<u32, QsaError> {
    validate_qsa_shape(s)?;
    Ok(s.token_budget / s.compression)
}

pub fn qsa_dispatches(s: QsaShape, position: u32) -> Result<u64, QsaError> {
    validate_qsa_shape(s)?;
    if position >= s.capacity {
        return Err(QsaError::Capacity);
    }
    let blocks = (position + 1) / s.compression;
    let mut count: u64 = 18
        + if (position + 1).is_multiple_of(s.compression) {
            3
        } else {
            0
        };
    if blocks > qsa_block_budget(s)? {
        count += 3; // Score, local sort and canonical IDs.
        let mut span: u32 = 512;
        while span < blocks {
            count += 1;
            span = span.wrapping_mul(2);
        }
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FLASH_NEXT: QsaShape = QsaShape {
        channels: 2560,
        query_heads: 24,
        kv_heads: 2,
        head_width: 256,
        index_heads: 4,
        index_width: 128,
        rotary_width: 64,
        compression: 4,
        token_budget: 2048,
        capacity: 65536,
    };

    #[test]
    fn dispatches_add_compression_and_selection_work_at_their_boundaries() {
        assert_eq!(qsa_dispatches(FLASH_NEXT, 0), Ok(18));
        assert_eq!(qsa_dispatches(FLASH_NEXT, 3), Ok(21));
        // Selection begins once 513 blocks exist (token 2052).
        assert_eq!(qsa_dispatches(FLASH_NEXT, 2050), Ok(18));
        assert_eq!(qsa_dispatches(FLASH_NEXT, 2051), Ok(25));
        assert_eq!(qsa_dispatches(FLASH_NEXT, 2052), Ok(22));
        assert_eq!(qsa_dispatches(FLASH_NEXT, 65535), Ok(21 + 3 + 5));
    }

    #[test]
    fn dispatches_refuse_positions_beyond_capacity_and_bad_shapes() {
        assert_eq!(qsa_dispatches(FLASH_NEXT, 65536), Err(QsaError::Capacity));
        let bad = QsaShape {
            compression: 3,
            ..FLASH_NEXT
        };
        assert_eq!(qsa_dispatches(bad, 0), Err(QsaError::Shape));
        assert_eq!(
            QsaError::Capacity.to_string(),
            "attention cache capacity exceeded"
        );
    }
}

//! The Foundation `NSRegularExpression` split used by `native/src/text.mm` (`AppleProcessor`), called through libobjc and CoreFoundation directly.
//!
//! Matches are enumerated as the C++ enumerates them, with `-enumerateMatchesInString:options:range:usingBlock:` and a block literal built here in the clang block ABI. One enumeration keeps the matcher's state from match to match, so it is faster than searching again from each match end, and the visitor can stop it early as the C++ piece limit does.

use std::ffi::{CStr, c_int, c_ulong, c_void};

use crate::ffi::{self, BlockDescriptor, Id, NsRange, Sel};

/// `CFStringCreateWithCharacters` failed or `NSRegularExpression` refused the pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegexError {
    /// No string object could be created from the UTF-16 units.
    String,
    /// `initWithPattern:options:error:` returned nil or an error.
    Compile,
}

impl std::fmt::Display for RegexError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::String => "cannot create Foundation UTF-16 string",
            Self::Compile => "cannot compile Foundation regular expression",
        })
    }
}

impl std::error::Error for RegexError {}

/// An immutable string built from UTF-16 code units, kept as given (a leading U+FEFF stays text, as in `ns()`).
pub struct Utf16String {
    object: Id,
    length: usize,
}

impl std::fmt::Debug for Utf16String {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Utf16String({} units)", self.length)
    }
}

impl Utf16String {
    pub fn new(units: &[u16]) -> Result<Self, RegexError> {
        let length = isize::try_from(units.len()).map_err(|_| RegexError::String)?;
        // SAFETY: the pointer and length describe one readable slice; the default allocator is requested with null.
        let object =
            unsafe { ffi::CFStringCreateWithCharacters(std::ptr::null(), units.as_ptr(), length) };
        if object.is_null() {
            return Err(RegexError::String);
        }
        Ok(Self {
            object,
            length: units.len(),
        })
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
}

impl Drop for Utf16String {
    fn drop(&mut self) {
        // SAFETY: `object` is the +1 reference CFStringCreateWithCharacters returned, released exactly once.
        unsafe { ffi::CFRelease(self.object.cast_const()) };
    }
}

/// A retained `NSRegularExpression` compiled with options 0.
pub struct Regex {
    object: Id,
}

impl std::fmt::Debug for Regex {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("Regex(..)")
    }
}

// SAFETY: NSRegularExpression is immutable and documented thread-safe; the wrapper only sends it read-only messages.
unsafe impl Send for Regex {}
// SAFETY: as above, concurrent read-only use is allowed.
unsafe impl Sync for Regex {}

fn selector(name: &CStr) -> Sel {
    // SAFETY: `name` is NUL-terminated; registration is idempotent.
    unsafe { ffi::sel_registerName(name.as_ptr()) }
}

/// The block `-enumerateMatchesInString:options:range:usingBlock:` calls: a stack literal in the clang block ABI with no copy or dispose helpers. Its captures, the `range` selector and the caller's visitor, are plain words, so a heap copy made by `Block_copy` works unchanged.
#[repr(C)]
struct MatchBlock<'a> {
    isa: *const c_void,
    flags: c_int,
    reserved: c_int,
    invoke: unsafe extern "C" fn(*mut MatchBlock<'a>, Id, usize, *mut bool),
    descriptor: *const BlockDescriptor,
    range: Sel,
    visit: *mut (dyn FnMut(usize, usize) -> bool + 'a),
}

static MATCH_BLOCK: BlockDescriptor = BlockDescriptor {
    reserved: 0,
    size: size_of::<MatchBlock<'static>>() as c_ulong,
};

/// The block's invoke function: reads the match range and sets `*stop` when the visitor declines to go on.
unsafe extern "C" fn visit_match(
    block: *mut MatchBlock<'_>,
    result: Id,
    _flags: usize,
    stop: *mut bool,
) {
    if result.is_null() {
        return;
    }
    type RangeOf = unsafe extern "C" fn(Id, Sel) -> NsRange;
    // SAFETY: Foundation passes this block (or a copy with the same captures), a live NSTextCheckingResult and a writable BOOL, which is a C `bool` on arm64. `-[NSTextCheckingResult range]` returns NSRange in registers on arm64. The visitor outlives the `enumerate_matches` call that runs this enumeration.
    unsafe {
        let range_of: RangeOf = std::mem::transmute(ffi::objc_msgSend as unsafe extern "C" fn());
        let found = range_of(result, (*block).range);
        if !(*(*block).visit)(found.location, found.length) {
            *stop = true;
        }
    }
}

/// Runs `body` inside an autorelease pool (`@autoreleasepool`).
fn autoreleased<T>(body: impl FnOnce() -> T) -> T {
    // SAFETY: push/pop are paired on this thread around `body`.
    let pool = unsafe { ffi::objc_autoreleasePoolPush() };
    let result = body();
    // SAFETY: pops the pool pushed above.
    unsafe { ffi::objc_autoreleasePoolPop(pool) };
    result
}

impl Regex {
    /// `initWithPattern:pattern options:0 error:&error`.
    pub fn new(pattern: &Utf16String) -> Result<Self, RegexError> {
        autoreleased(|| {
            // SAFETY: the class name is NUL-terminated; Foundation is linked.
            let class = unsafe { ffi::objc_getClass(c"NSRegularExpression".as_ptr()) };
            if class.is_null() {
                return Err(RegexError::Compile);
            }
            type Alloc = unsafe extern "C" fn(Id, Sel) -> Id;
            type Init = unsafe extern "C" fn(Id, Sel, Id, usize, *mut Id) -> Id;
            // SAFETY: objc_msgSend is called through the exact signatures of `+alloc` and `-initWithPattern:options:error:`.
            let object = unsafe {
                let alloc: Alloc = std::mem::transmute(ffi::objc_msgSend as unsafe extern "C" fn());
                let init: Init = std::mem::transmute(ffi::objc_msgSend as unsafe extern "C" fn());
                let allocated = alloc(class, selector(c"alloc"));
                let mut error: Id = std::ptr::null_mut();
                let object = init(
                    allocated,
                    selector(c"initWithPattern:options:error:"),
                    pattern.object,
                    0,
                    &mut error,
                );
                if !error.is_null() && !object.is_null() {
                    ffi::objc_release(object);
                    return Err(RegexError::Compile);
                }
                object
            };
            if object.is_null() {
                Err(RegexError::Compile)
            } else {
                Ok(Self { object })
            }
        })
    }

    /// Visits every match in `text` in order, as an (offset, length) pair in UTF-16 units, through `-enumerateMatchesInString:options:range:usingBlock:` with options 0. Enumeration stops once `visit` returns false.
    pub fn enumerate_matches(
        &self,
        text: &Utf16String,
        visit: &mut dyn FnMut(usize, usize) -> bool,
    ) {
        let mut block = MatchBlock {
            isa: (&raw const ffi::_NSConcreteStackBlock).cast(),
            flags: 0,
            reserved: 0,
            invoke: visit_match,
            descriptor: &MATCH_BLOCK,
            range: selector(c"range"),
            visit,
        };
        let range = NsRange {
            location: 0,
            length: text.length,
        };
        autoreleased(|| {
            type Enumerate = unsafe extern "C" fn(Id, Sel, Id, usize, NsRange, *mut c_void);
            // SAFETY: objc_msgSend is called through the exact signature of `-enumerateMatchesInString:options:range:usingBlock:`, the range covers `text`, and `block` is a valid block literal that lives across this synchronous call.
            unsafe {
                let enumerate: Enumerate =
                    std::mem::transmute(ffi::objc_msgSend as unsafe extern "C" fn());
                enumerate(
                    self.object,
                    selector(c"enumerateMatchesInString:options:range:usingBlock:"),
                    text.object,
                    0,
                    range,
                    (&raw mut block).cast(),
                );
            }
        });
    }
}

impl Drop for Regex {
    fn drop(&mut self) {
        // SAFETY: `object` is the +1 reference `-init...` returned.
        unsafe { ffi::objc_release(self.object) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn units(text: &str) -> Vec<u16> {
        text.encode_utf16().collect()
    }

    fn matches(regex: &Regex, text: &Utf16String) -> Vec<(usize, usize)> {
        let mut found = Vec::new();
        regex.enumerate_matches(text, &mut |location, length| {
            found.push((location, length));
            true
        });
        found
    }

    /// The earlier split: `-firstMatchInString:options:range:` from `start` to the end, the reference enumeration must reproduce.
    fn first_match(regex: &Regex, text: &Utf16String, start: usize) -> Option<(usize, usize)> {
        let range = NsRange {
            location: start,
            length: text.length - start,
        };
        autoreleased(|| {
            type First = unsafe extern "C" fn(Id, Sel, Id, usize, NsRange) -> Id;
            type RangeOf = unsafe extern "C" fn(Id, Sel) -> NsRange;
            // SAFETY: objc_msgSend is called through the exact signatures of `-firstMatchInString:options:range:` and `-[NSTextCheckingResult range]`, and the range lies within `text`.
            unsafe {
                let first: First = std::mem::transmute(ffi::objc_msgSend as unsafe extern "C" fn());
                let range_of: RangeOf =
                    std::mem::transmute(ffi::objc_msgSend as unsafe extern "C" fn());
                let result = first(
                    regex.object,
                    selector(c"firstMatchInString:options:range:"),
                    text.object,
                    0,
                    range,
                );
                if result.is_null() {
                    return None;
                }
                let found = range_of(result, selector(c"range"));
                Some((found.location, found.length))
            }
        })
    }

    #[test]
    fn enumeration_visits_every_match_in_order() {
        let pattern = Utf16String::new(&units(r"\p{L}+")).unwrap();
        let regex = Regex::new(&pattern).unwrap();
        let text = Utf16String::new(&units("ab 12 cd")).unwrap();
        assert_eq!(matches(&regex, &text), [(0, 2), (6, 2)]);
        let empty = Utf16String::new(&[]).unwrap();
        assert!(matches(&regex, &empty).is_empty());
    }

    #[test]
    fn a_declined_visit_stops_enumeration() {
        let pattern = Utf16String::new(&units(r"\p{L}+")).unwrap();
        let regex = Regex::new(&pattern).unwrap();
        let text = Utf16String::new(&units("ab cd ef")).unwrap();
        let mut visited = Vec::new();
        regex.enumerate_matches(&text, &mut |location, length| {
            visited.push((location, length));
            visited.len() < 2
        });
        assert_eq!(visited, [(0, 2), (3, 2)]);
    }

    #[test]
    fn enumeration_reproduces_the_search_from_each_match_end() {
        // The split pattern `tokspur-server` pins (`SUPPORTED_PATTERN`), over code, contractions, CRLF, combining marks, surrogate pairs, CJK, Arabic-Indic digits and a no-break space.
        let pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        let regex = Regex::new(&Utf16String::new(&units(pattern)).unwrap()).unwrap();
        let corpus = "fn main() {\r\n    let x = 42; // it's   fine\n\n\t'LL've e\u{301}t\u{e9} \u{1f600}\u{1f600} \u{6771}\u{4eac} \u{661}\u{662}3 x\u{a0}y  \n  \n";
        let text_units: Vec<u16> = corpus.repeat(200).encode_utf16().collect();
        let text = Utf16String::new(&text_units).unwrap();
        let mut expected = Vec::new();
        let mut search = 0;
        while let Some((location, length)) = first_match(&regex, &text, search) {
            expected.push((location, length));
            search = if length == 0 {
                location + 1
            } else {
                location + length
            };
            if search > text_units.len() {
                break;
            }
        }
        assert_eq!(expected.len(), 6200);
        assert_eq!(matches(&regex, &text), expected);
    }

    #[test]
    fn invalid_patterns_are_refused() {
        let pattern = Utf16String::new(&units("(")).unwrap();
        assert_eq!(Regex::new(&pattern).unwrap_err(), RegexError::Compile);
    }

    #[test]
    fn a_leading_byte_order_mark_stays_text() {
        let pattern = Utf16String::new(&units("\u{feff}")).unwrap();
        let regex = Regex::new(&pattern).unwrap();
        let text = Utf16String::new(&units("\u{feff}x")).unwrap();
        assert_eq!(text.len(), 2);
        assert_eq!(matches(&regex, &text), [(0, 1)]);
    }
}

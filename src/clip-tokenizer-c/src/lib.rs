use std::collections::{HashMap, HashSet};
use std::ffi::CStr;
use std::io::Read;
use std::os::raw::{c_char, c_int};
use std::cell::RefCell;

use flate2::read::GzDecoder;
use regex::Regex;

// BPE vocab file embedded at compile time
const BPE_VOCAB_GZ: &[u8] = include_bytes!("../bpe_simple_vocab_16e6.txt.gz");

/// Generate bytes_to_unicode mapping (same as Python version)
fn bytes_to_unicode() -> HashMap<u8, char> {
    let mut bs: Vec<u8> = Vec::new();

    // Add printable ASCII
    bs.extend((b'!'..(b'~' + 1)).collect::<Vec<u8>>());
    bs.extend((0xA1..=0xAC).collect::<Vec<u8>>());
    bs.extend((0xAE..=0xFF).collect::<Vec<u8>>());

    // cs holds the Unicode code points (as u32) that each byte maps to
    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
    let mut n = 0u32;

    for b in 0u8..=255 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }

    bs.into_iter()
        .zip(cs.into_iter().map(|c| char::from_u32(c).unwrap()))
        .collect()
}

/// Get pairs of adjacent characters in a word
fn get_pairs(word: &[String]) -> HashSet<(String, String)> {
    let mut pairs = HashSet::new();
    if word.len() < 2 {
        return pairs;
    }

    for i in 0..word.len() - 1 {
        pairs.insert((word[i].clone(), word[i + 1].clone()));
    }
    pairs
}

/// Basic cleaning (matches Python version)
fn basic_clean(text: &str) -> String {
    // Note: Python version uses ftfy.fix_text() and html.unescape()
    // For CLIP's purposes, simple cleaning is sufficient
    text.trim().to_string()
}

/// Whitespace cleaning (matches Python version)
fn whitespace_clean(text: &str) -> String {
    let re = Regex::new(r"\s+").unwrap();
    re.replace_all(text.trim(), " ").to_string()
}

/// CLIP Simple Tokenizer
pub struct SimpleTokenizer {
    byte_encoder: HashMap<u8, char>,
    encoder: HashMap<String, u32>,
    bpe_ranks: HashMap<(String, String), usize>,
    cache: HashMap<String, String>,
    pat: Regex,
}

impl SimpleTokenizer {
    pub fn new() -> Result<Self, String> {
        // Decode the gzipped BPE vocab
        let mut decoder = GzDecoder::new(BPE_VOCAB_GZ);
        let mut vocab_str = String::new();
        decoder.read_to_string(&mut vocab_str)
            .map_err(|e| format!("Failed to decode BPE vocab: {}", e))?;

        let lines: Vec<&str> = vocab_str.lines().collect();
        if lines.is_empty() {
            return Err("Empty BPE vocab".to_string());
        }

        // Parse merges (skip header, take 49152-256-2+1 merges)
        let merges: Vec<(String, String)> = lines[1..(49152 - 256 - 2 + 1)]
            .iter()
            .filter_map(|line| {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() == 2 {
                    Some((parts[0].to_string(), parts[1].to_string()))
                } else {
                    None
                }
            })
            .collect();

        // Build byte encoder/decoder
        let byte_encoder = bytes_to_unicode();

        // Build vocabulary - MUST be sorted by Unicode codepoint to match Python
        let mut vocab_chars: Vec<char> = byte_encoder.values().copied().collect();
        vocab_chars.sort_by_key(|c| *c as u32);

        let mut vocab: Vec<String> = vocab_chars.iter().map(|c| c.to_string()).collect();

        // Add end-of-word variants
        let base_vocab = vocab.clone();
        for v in base_vocab {
            vocab.push(format!("{}</w>", v));
        }

        // Add merged tokens
        for (first, second) in &merges {
            vocab.push(format!("{}{}", first, second));
        }

        // Add special tokens
        vocab.push("<|startoftext|>".to_string());
        vocab.push("<|endoftext|>".to_string());

        // Create encoder/decoder
        let encoder: HashMap<String, u32> = vocab.iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();

        // Create BPE ranks
        let bpe_ranks: HashMap<(String, String), usize> = merges.iter()
            .enumerate()
            .map(|(i, pair)| (pair.clone(), i))
            .collect();

        // Initialize cache with special tokens
        let mut cache = HashMap::new();
        cache.insert("<|startoftext|>".to_string(), "<|startoftext|>".to_string());
        cache.insert("<|endoftext|>".to_string(), "<|endoftext|>".to_string());

        // Compile regex pattern
        let pat = Regex::new(
            r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"
        ).map_err(|e| format!("Regex compilation failed: {}", e))?;

        Ok(SimpleTokenizer {
            byte_encoder,
            encoder,
            bpe_ranks,
            cache,
            pat,
        })
    }

    /// BPE algorithm
    fn bpe(&mut self, token: &str) -> String {
        if let Some(cached) = self.cache.get(token) {
            return cached.clone();
        }

        let mut word: Vec<String> = token.chars()
            .map(|c| c.to_string())
            .collect();

        if !word.is_empty() {
            let last_idx = word.len() - 1;
            word[last_idx] = format!("{}</w>", word[last_idx]);
        }

        let mut pairs = get_pairs(&word);

        if pairs.is_empty() {
            let result = format!("{}</w>", token);
            self.cache.insert(token.to_string(), result.clone());
            return result;
        }

        loop {
            let bigram = pairs.iter()
                .min_by_key(|pair| {
                    self.bpe_ranks.get(pair).copied().unwrap_or(usize::MAX)
                });

            if bigram.is_none() {
                break;
            }

            let bigram = bigram.unwrap().clone();
            if !self.bpe_ranks.contains_key(&bigram) {
                break;
            }

            let (first, second) = bigram;
            let mut new_word = Vec::new();
            let mut i = 0;

            while i < word.len() {
                if let Some(j) = word[i..].iter().position(|w| w == &first) {
                    let j = i + j;
                    new_word.extend_from_slice(&word[i..j]);
                    i = j;

                    if i < word.len() - 1 && word[i] == first && word[i + 1] == second {
                        new_word.push(format!("{}{}", first, second));
                        i += 2;
                    } else {
                        new_word.push(word[i].clone());
                        i += 1;
                    }
                } else {
                    new_word.extend_from_slice(&word[i..]);
                    break;
                }
            }

            word = new_word;
            if word.len() == 1 {
                break;
            }
            pairs = get_pairs(&word);
        }

        let result = word.join(" ");
        self.cache.insert(token.to_string(), result.clone());
        result
    }

    /// Encode text to token IDs
    pub fn encode(&mut self, text: &str) -> Vec<u32> {
        let mut bpe_tokens = Vec::new();

        // Apply preprocessing (basic clean, whitespace clean, lowercase)
        let text = whitespace_clean(&basic_clean(text)).to_lowercase();

        // Collect all regex matches first to avoid borrowing conflicts
        let matches: Vec<String> = self.pat.find_iter(&text)
            .map(|cap| cap.as_str().to_string())
            .collect();

        // Process each match
        for token_str in matches {
            // Byte-level encoding
            let token_bytes: String = token_str.bytes()
                .map(|b| self.byte_encoder.get(&b).copied().unwrap_or('�'))
                .collect();

            // Apply BPE
            let bpe_result = self.bpe(&token_bytes);

            // Convert BPE tokens to IDs
            for bpe_token in bpe_result.split(' ') {
                if let Some(&id) = self.encoder.get(bpe_token) {
                    bpe_tokens.push(id);
                }
            }
        }

        bpe_tokens
    }
}

// Thread-local tokenizer instance - each thread gets its own tokenizer
// This avoids lock contention when called from multiple threads
thread_local! {
    static TOKENIZER: RefCell<Option<SimpleTokenizer>> = RefCell::new(None);
}

/// Initialize thread-local tokenizer if not already initialized
fn with_tokenizer<F, R>(f: F) -> Result<R, String>
where
    F: FnOnce(&mut SimpleTokenizer) -> R,
{
    TOKENIZER.with(|tok| {
        let mut tok_ref = tok.borrow_mut();
        if tok_ref.is_none() {
            *tok_ref = Some(SimpleTokenizer::new()?);
        }
        Ok(f(tok_ref.as_mut().unwrap()))
    })
}

/// C API: Encode text to token IDs with configurable padding
/// Outputs tokens with BOS (49406) and EOS (49407) added, padded to 77 tokens
/// Thread-safe: Each thread gets its own tokenizer instance
#[no_mangle]
pub extern "C" fn clip_tokenizer_encode_with_padding(
    text: *const c_char,
    output: *mut i32,
    pad_token: c_int,
) -> c_int {
    if text.is_null() || output.is_null() {
        return -1;
    }

    let c_str = unsafe { CStr::from_ptr(text) };
    let text_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    // Encode using thread-local tokenizer
    let result = with_tokenizer(|tokenizer| {
        let mut tokens = tokenizer.encode(text_str);

        // Add BOS token (49406) at start
        tokens.insert(0, 49406);

        // Add EOS token (49407) at end
        tokens.push(49407);

        // Pad or truncate to 77 tokens with specified pad token
        // SD-Turbo uses PAD token (0), SD1.5 uses EOS token (49407)
        tokens.resize(77, pad_token as u32);

        tokens
    });

    let tokens = match result {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Tokenizer error: {}", e);
            return -1;
        }
    };

    // Copy to output
    unsafe {
        for (i, &token) in tokens.iter().enumerate() {
            *output.add(i) = token as i32;
        }
    }

    77
}

/// C API: Encode text to token IDs (compatibility wrapper)
/// Uses PAD token (0) for padding to match SD-Turbo behavior
/// For SD1.5 compatibility, use clip_tokenizer_encode_with_padding with pad_token=49407
#[no_mangle]
pub extern "C" fn clip_tokenizer_encode(
    text: *const c_char,
    output: *mut i32,
) -> c_int {
    // Default to PAD token (0) for SD-Turbo
    clip_tokenizer_encode_with_padding(text, output, 0)
}

/// C API: Free tokenizer resources (no-op, handled by thread-local storage)
#[no_mangle]
pub extern "C" fn clip_tokenizer_free() {
    // Nothing to do - thread-local storage is automatically cleaned up
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_to_unicode() {
        let byte_enc = bytes_to_unicode();

        // Test specific known mappings
        assert_eq!(byte_enc[&44], ',', "Byte 44 (comma) should map to ','");
        assert_eq!(byte_enc[&56], '8', "Byte 56 should map to '8'");
        assert_eq!(byte_enc[&107], 'k', "Byte 107 should map to 'k'");

        // Test unmapped bytes (should map to Unicode 256+)
        assert_eq!(byte_enc[&0] as u32, 256, "Byte 0 should map to Unicode 256");
        assert_eq!(byte_enc[&1] as u32, 257, "Byte 1 should map to Unicode 257");

        // Byte 173 (0xAD) should map to Unicode 323 (0x143)
        assert_eq!(byte_enc[&173] as u32, 323, "Byte 173 should map to Unicode 323");

        // Verify we have exactly 256 mappings
        assert_eq!(byte_enc.len(), 256, "Should have 256 byte mappings");

        // Print first 10 for debugging
        println!("First 10 Rust mappings:");
        for b in 0..10u8 {
            println!("  Byte {} -> Unicode {}", b, byte_enc[&b] as u32);
        }
    }

    #[test]
    fn test_bpe() {
        let mut tokenizer = SimpleTokenizer::new().unwrap();

        // Test BPE on comma
        let bpe_result = tokenizer.bpe(",");
        println!("BPE(',') = {:?}", bpe_result);
        assert_eq!(bpe_result, ",</w>", "BPE of ',' should be ',</w>'");

        // Test BPE on '8'
        let bpe_result = tokenizer.bpe("8");
        println!("BPE('8') = {:?}", bpe_result);
        assert_eq!(bpe_result, "8</w>", "BPE of '8' should be '8</w>'");

        // Test BPE on 'k'
        let bpe_result = tokenizer.bpe("k");
        println!("BPE('k') = {:?}", bpe_result);
        assert_eq!(bpe_result, "k</w>", "BPE of 'k' should be 'k</w>'");
    }

    #[test]
    fn test_encoder_vocab() {
        let tokenizer = SimpleTokenizer::new().unwrap();

        // Check specific token mappings
        assert_eq!(tokenizer.encoder.get(",</w>"), Some(&267), "',</w>' should map to token 267");
        assert_eq!(tokenizer.encoder.get("8</w>"), Some(&279), "'8</w>' should map to token 279");
        assert_eq!(tokenizer.encoder.get("k</w>"), Some(&330), "'k</w>' should map to token 330");
        assert_eq!(tokenizer.encoder.get("cat</w>"), Some(&2368), "'cat</w>' should map to token 2368");

        println!("Encoder vocab size: {}", tokenizer.encoder.len());
        println!("',</w>' -> {:?}", tokenizer.encoder.get(",</w>"));
        println!("'8</w>' -> {:?}", tokenizer.encoder.get("8</w>"));
        println!("'k</w>' -> {:?}", tokenizer.encoder.get("k</w>"));
    }

    #[test]
    fn test_tokenizer() {
        let mut tokenizer = SimpleTokenizer::new().unwrap();
        let tokens = tokenizer.encode("cat, 8k, digital art");

        // Expected: [2368, 267, 279, 330, 267, 2794, 794]
        println!("Tokens: {:?}", tokens);
        assert_eq!(tokens, vec![2368, 267, 279, 330, 267, 2794, 794]);
    }
}

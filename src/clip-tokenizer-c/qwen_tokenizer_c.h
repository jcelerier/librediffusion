/**
 * Qwen2/Qwen3 tokenizer C API for FLUX.2-klein-4B (Phase 4).
 *
 * Loads tokenizer.json via the HuggingFace `tokenizers` Rust crate (which provides the Qwen2
 * byte-level pre-tokenizer regex + BPE merges + special-token table) and applies the klein jinja
 * chat template by hand (the crate does not run jinja). Validated to match the golden input_ids.
 */
#ifndef QWEN_TOKENIZER_C_H
#define QWEN_TOKENIZER_C_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Load the Qwen tokenizer from a tokenizer.json path. Returns 0 on success, -1 on failure.
 * Idempotent (re-loading replaces the previous instance). */
int qwen_tokenizer_load(const char* tokenizer_json_path);

/* Apply the klein chat template (<|im_start|>user\n{P}<|im_end|>\n<|im_start|>assistant\n
 * <think>\n\n</think>\n\n) and tokenize to padded max_len ids + attention mask.
 *   out_ids[i]  (int64): token id, padded with <|endoftext|> (151643) to max_len.
 *   out_mask[i] (int64): 1 for real tokens, 0 for padding.
 * Caller pre-allocates both buffers to max_len int64 elements.
 * Returns the number of real (non-pad) tokens, or -1 on error. */
int qwen_tokenizer_encode_chat(
    const char* prompt, int max_len, int64_t* out_ids, int64_t* out_mask);

/* Free the tokenizer (idempotent). */
void qwen_tokenizer_free(void);

#ifdef __cplusplus
}
#endif

#endif /* QWEN_TOKENIZER_C_H */

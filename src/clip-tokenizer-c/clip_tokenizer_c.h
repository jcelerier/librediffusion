/**
 * CLIP Tokenizer C API
 *
 * This header provides a C interface to the HuggingFace tokenizers library
 * for use with CLIP models.
 */

#ifndef CLIP_TOKENIZER_C_H
#define CLIP_TOKENIZER_C_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Encode a text string to token IDs
 *
 * The tokenizer is initialized automatically on first call.
 * Outputs 77 tokens with BOS (49406) and EOS (49407) added and padded.
 *
 * @param text Input text to tokenize
 * @param output Output buffer for token IDs (must be pre-allocated with size 77)
 * @return Number of tokens written (always 77), or -1 on error
 */
int clip_tokenizer_encode(const char* text, int* output);

/**
 * Encode a text string to token IDs with configurable padding
 *
 * The tokenizer is initialized automatically on first call.
 * Outputs 77 tokens with BOS (49406) and EOS (49407) added and padded.
 *
 * @param text Input text to tokenize
 * @param output Output buffer for token IDs (must be pre-allocated with size 77)
 * @param pad_token Token ID to use for padding (0 for SD-Turbo, 49407 for SD1.5)
 * @return Number of tokens written (always 77), or -1 on error
 */
int clip_tokenizer_encode_with_padding(const char* text, int* output, int pad_token);

/**
 * Free tokenizer resources
 *
 * Currently a no-op as the tokenizer uses static lifetime.
 */
void clip_tokenizer_free(void);

#ifdef __cplusplus
}
#endif

#endif // CLIP_TOKENIZER_C_H

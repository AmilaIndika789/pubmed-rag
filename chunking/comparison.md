# Chunking Comparison

## Strategy 1: Fixed-size chunking

I used overlapping word-based chunking to approximate token chunking.
This strategy is simple, robust, and works even when abstracts are inconsistently formatted.

### Pros

- Works on every document
- Predictable chunk size
- Easy to tune

### Cons

- Can split semantically related content across chunks
- May mix unrelated ideas in the same chunk

## Strategy 2: Section-based chunking

I split abstracts by labeled sections such as Background, Methods, Results, and Conclusions when possible.
If no labels existed, I fell back to paragraph-level or full-abstract chunks.

### Pros

- Better semantic coherence
- Better alignment with medical abstract structure
- Often easier for retrieval and citation

### Cons

- Depends on article formatting
- Section labels are not always present
- Inconsistent section labels

## Recommendation

I recommend section-based chunking as the primary strategy because medical abstracts often have meaningful section boundaries that improve retrieval quality. I would keep fixed-size chunking as a fallback for inconsistent or unlabeled abstracts.

/** Prototype pair scorer for the M3.3 semantics benchmark, not a graph export path. */
export interface CoRatedDeviation {
  left: number;
  right: number;
}

export interface ItemSimilarityScore {
  support: number;
  adjustedCosine: number;
  shrunkSimilarity: number;
}

/**
 * Compare per-user mean-centered scores for users who rated both items.
 * Missing overlap and a zero-magnitude side have no defined cosine score.
 */
export function scoreSupportShrunkAdjustedCosine(
  observations: readonly CoRatedDeviation[],
  shrinkage: number,
): ItemSimilarityScore | null {
  if (!Number.isFinite(shrinkage) || shrinkage < 0) {
    throw new Error("Similarity shrinkage must be finite and nonnegative.");
  }
  if (observations.length === 0) return null;

  let dot = 0;
  let leftSquare = 0;
  let rightSquare = 0;
  for (const { left, right } of observations) {
    if (!Number.isFinite(left) || !Number.isFinite(right)) {
      throw new Error("Item similarity requires finite centered scores.");
    }
    dot += left * right;
    leftSquare += left * left;
    rightSquare += right * right;
  }

  const denominator = Math.sqrt(leftSquare * rightSquare);
  if (!Number.isFinite(denominator)) {
    throw new Error("Item similarity accumulated a non-finite magnitude.");
  }
  if (denominator === 0) return null;

  const adjustedCosine = Math.max(-1, Math.min(1, dot / denominator));
  const support = observations.length;
  return {
    support,
    adjustedCosine,
    shrunkSimilarity: adjustedCosine * (support / (support + shrinkage)),
  };
}

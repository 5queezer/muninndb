package engine

import (
	"context"
	"sort"

	"github.com/scrypster/muninndb/internal/engine/activation"
	"github.com/scrypster/muninndb/internal/storage"
)

const (
	// separationSeedTopN is the number of top results whose entity links are
	// used as the "query entity context" for pattern separation scoring.
	// Matches entityBoostTopN so both features derive context from the same seeds.
	separationSeedTopN = 5
)

// applySeparation applies hippocampal pattern separation to post-scored results.
// It collects entities from the top-N seed results (the "query entity context"),
// then penalises candidates whose entity sets have low Jaccard overlap with the
// seeds. This reduces cross-context interference without modifying the HNSW index.
//
// No-op when e.separationScorer is nil (feature disabled).
func (e *Engine) applySeparation(ctx context.Context, ws [8]byte, results []activation.ScoredEngram) []activation.ScoredEngram {
	if e.separationScorer == nil || len(results) == 0 {
		return results
	}

	// Collect entities from the top-N seed results to form the query entity context.
	seedCount := len(results)
	if seedCount > separationSeedTopN {
		seedCount = separationSeedTopN
	}

	queryEntitySet := make(map[string]struct{})
	for _, seed := range results[:seedCount] {
		_ = e.store.ScanEngramEntities(ctx, ws, seed.Engram.ID, func(entityName string) error {
			queryEntitySet[entityName] = struct{}{}
			return nil
		})
	}

	// Flatten the entity set to a slice for the scorer.
	queryEntities := make([]string, 0, len(queryEntitySet))
	for name := range queryEntitySet {
		queryEntities = append(queryEntities, name)
	}

	// No entity context from seeds → no separation signal.
	if len(queryEntities) == 0 {
		return results
	}

	// Build candidate ID slice.
	candidateIDs := make([][16]byte, len(results))
	for i, r := range results {
		candidateIDs[i] = [16]byte(r.Engram.ID)
	}

	multipliers, err := e.separationScorer.ScoreSeparation(ctx, ws, queryEntities, candidateIDs)
	if err != nil {
		// On error, return results unmodified.
		return results
	}

	// Apply multipliers. Skip seed results (first seedCount) — they define
	// the query context and should not be penalised.
	seedIDs := make(map[storage.ULID]struct{}, seedCount)
	for _, seed := range results[:seedCount] {
		seedIDs[seed.Engram.ID] = struct{}{}
	}

	for i := range results {
		if _, isSeed := seedIDs[results[i].Engram.ID]; isSeed {
			continue
		}
		results[i].Score *= multipliers[i]
	}

	// Re-sort descending by score after separation adjustments.
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return results
}

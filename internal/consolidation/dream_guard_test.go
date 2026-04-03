package consolidation

import (
	"context"
	"math"
	"testing"

	"github.com/cockroachdb/pebble"
	"github.com/scrypster/muninndb/internal/storage"
)

func TestDream_SmallVault_SkipsDedup(t *testing.T) {
	store, db, cleanup := testStoreWithDB(t)
	defer cleanup()

	ctx := context.Background()
	vault := "small-vault-guard"
	wsPrefix := store.ResolveVaultPrefix(vault)

	dupAID := writeEngramWithEmbedding(t, ctx, store, db, wsPrefix, &storage.Engram{
		Concept: "dup-a", Content: "France's capital is Paris.", Confidence: 0.9, Relevance: 0.8,
		Stability: 30, Embedding: unitEmbedding(9, 0),
	})
	dupBID := writeEngramWithEmbedding(t, ctx, store, db, wsPrefix, &storage.Engram{
		Concept: "dup-b", Content: "Paris is the capital of France.", Confidence: 0.5, Relevance: 0.5,
		Stability: 20, Embedding: nearEmbedding(9, 0, 1, 0.97),
	})
	writeUniqueEngrams(t, ctx, store, db, wsPrefix, 7, 9, 2)

	w := NewWorker(&mockEngineInterface{store: store})
	report, err := w.DreamOnce(ctx, DreamOpts{Force: true, Scope: vault})
	if err != nil {
		t.Fatal(err)
	}

	if len(report.Reports) != 1 {
		t.Fatalf("expected 1 report, got %d", len(report.Reports))
	}
	if got := report.Reports[0].MergedEngrams; got != 0 {
		t.Fatalf("MergedEngrams = %d, want 0 when vault is below the minimum size", got)
	}

	for _, id := range []storage.ULID{dupAID, dupBID} {
		eng, err := store.GetEngram(ctx, wsPrefix, id)
		if err != nil {
			t.Fatalf("GetEngram(%v): %v", id, err)
		}
		if eng.State == storage.StateArchived {
			t.Fatalf("engram %v was archived even though dream dedup should have been skipped", id)
		}
	}
}

func TestDream_SufficientVault_RunsDedup(t *testing.T) {
	store, db, cleanup := testStoreWithDB(t)
	defer cleanup()

	ctx := context.Background()
	vault := "sufficient-vault-dedup"
	wsPrefix := store.ResolveVaultPrefix(vault)

	repID := writeEngramWithEmbedding(t, ctx, store, db, wsPrefix, &storage.Engram{
		Concept: "rep", Content: "France's capital is Paris.", Confidence: 0.9, Relevance: 0.85,
		Stability: 30, Embedding: unitEmbedding(20, 0),
	})
	memID := writeEngramWithEmbedding(t, ctx, store, db, wsPrefix, &storage.Engram{
		Concept: "member", Content: "Paris is the capital of France.", Confidence: 0.5, Relevance: 0.5,
		Stability: 20, Embedding: nearEmbedding(20, 0, 1, 0.97),
	})
	writeUniqueEngrams(t, ctx, store, db, wsPrefix, 18, 20, 2)

	w := NewWorker(&mockEngineInterface{store: store})
	report, err := w.DreamOnce(ctx, DreamOpts{Force: true, Scope: vault})
	if err != nil {
		t.Fatal(err)
	}

	if len(report.Reports) != 1 {
		t.Fatalf("expected 1 report, got %d", len(report.Reports))
	}
	if got := report.Reports[0].MergedEngrams; got != 1 {
		t.Fatalf("MergedEngrams = %d, want 1 once the vault reaches the minimum size", got)
	}

	rep, err := store.GetEngram(ctx, wsPrefix, repID)
	if err != nil {
		t.Fatal(err)
	}
	if rep.State == storage.StateArchived {
		t.Fatal("representative engram was archived")
	}

	mem, err := store.GetEngram(ctx, wsPrefix, memID)
	if err != nil {
		t.Fatal(err)
	}
	if mem.State != storage.StateArchived {
		t.Fatalf("member state = %v, want archived", mem.State)
	}
}

func TestDream_LegalAdjacent_IsProcessed(t *testing.T) {
	store, db, cleanup := testStoreWithDB(t)
	defer cleanup()

	ctx := context.Background()
	vault := "paralegal-notes"
	wsPrefix := store.ResolveVaultPrefix(vault)

	writeEngramWithEmbedding(t, ctx, store, db, wsPrefix, &storage.Engram{
		Concept: "note-a", Content: "The hearing is on Monday.", Confidence: 0.9, Relevance: 0.85,
		Stability: 30, Embedding: unitEmbedding(20, 0),
	})
	writeEngramWithEmbedding(t, ctx, store, db, wsPrefix, &storage.Engram{
		Concept: "note-b", Content: "Monday is when the hearing takes place.", Confidence: 0.5, Relevance: 0.5,
		Stability: 20, Embedding: nearEmbedding(20, 0, 1, 0.97),
	})
	writeUniqueEngrams(t, ctx, store, db, wsPrefix, 18, 20, 2)

	w := NewWorker(&mockEngineInterface{store: store})
	report, err := w.DreamOnce(ctx, DreamOpts{Force: true, Scope: vault})
	if err != nil {
		t.Fatal(err)
	}

	if len(report.Reports) != 1 {
		t.Fatalf("expected 1 report, got %d", len(report.Reports))
	}
	for _, skipped := range report.Skipped {
		if skipped == vault {
			t.Fatalf("%q was incorrectly treated as a protected legal vault", vault)
		}
	}
	if got := report.Reports[0].MergedEngrams; got == 0 {
		t.Fatalf("MergedEngrams = %d, want > 0 for non-legal vault %q", got, vault)
	}
}

func TestDream_MinDedupVaultSize_Configurable(t *testing.T) {
	t.Run("dedup_runs_when_threshold_below_with_embed", func(t *testing.T) {
		store, db, cleanup := testStoreWithDB(t)
		defer cleanup()

		ctx := context.Background()
		vault := "configurable-low"
		wsPrefix := store.ResolveVaultPrefix(vault)

		writeEngramWithEmbedding(t, ctx, store, db, wsPrefix, &storage.Engram{
			Concept: "dup-a", Content: "content a", Confidence: 0.9, Relevance: 0.8,
			Stability: 25, Embedding: unitEmbedding(15, 0),
		})
		writeEngramWithEmbedding(t, ctx, store, db, wsPrefix, &storage.Engram{
			Concept: "dup-b", Content: "content b", Confidence: 0.5, Relevance: 0.5,
			Stability: 20, Embedding: nearEmbedding(15, 0, 1, 0.97),
		})
		writeUniqueEngrams(t, ctx, store, db, wsPrefix, 13, 15, 2)

		w := NewWorker(&mockEngineInterface{store: store})
		w.MinDedupVaultSize = 10
		report, err := w.DreamOnce(ctx, DreamOpts{DryRun: true, Force: true, Scope: vault})
		if err != nil {
			t.Fatal(err)
		}

		if got := report.Reports[0].DedupClusters; got == 0 {
			t.Fatalf("DedupClusters = %d, want > 0 when custom threshold allows dream dedup", got)
		}
	})

	t.Run("dedup_skips_when_threshold_above_with_embed", func(t *testing.T) {
		store, db, cleanup := testStoreWithDB(t)
		defer cleanup()

		ctx := context.Background()
		vault := "configurable-high"
		wsPrefix := store.ResolveVaultPrefix(vault)

		writeEngramWithEmbedding(t, ctx, store, db, wsPrefix, &storage.Engram{
			Concept: "dup-a", Content: "content a", Confidence: 0.9, Relevance: 0.8,
			Stability: 25, Embedding: unitEmbedding(15, 0),
		})
		writeEngramWithEmbedding(t, ctx, store, db, wsPrefix, &storage.Engram{
			Concept: "dup-b", Content: "content b", Confidence: 0.5, Relevance: 0.5,
			Stability: 20, Embedding: nearEmbedding(15, 0, 1, 0.97),
		})
		writeUniqueEngrams(t, ctx, store, db, wsPrefix, 13, 15, 2)

		w := NewWorker(&mockEngineInterface{store: store})
		w.MinDedupVaultSize = 20
		report, err := w.DreamOnce(ctx, DreamOpts{DryRun: true, Force: true, Scope: vault})
		if err != nil {
			t.Fatal(err)
		}

		if got := report.Reports[0].DedupClusters; got != 0 {
			t.Fatalf("DedupClusters = %d, want 0 when guard should skip dream dedup", got)
		}
		if got := report.Reports[0].MergedEngrams; got != 0 {
			t.Fatalf("MergedEngrams = %d, want 0 when guard should skip dream dedup", got)
		}
	})
}

func TestDream_WithEmbedCount_GuardIgnoresNoEmbedEngrams(t *testing.T) {
	store, db, cleanup := testStoreWithDB(t)
	defer cleanup()

	ctx := context.Background()
	vault := "embed-count-guard"
	wsPrefix := store.ResolveVaultPrefix(vault)

	writeEngramWithEmbedding(t, ctx, store, db, wsPrefix, &storage.Engram{
		Concept: "dup-a", Content: "content a", Confidence: 0.9, Relevance: 0.8,
		Stability: 25, Embedding: unitEmbedding(10, 0),
	})
	writeEngramWithEmbedding(t, ctx, store, db, wsPrefix, &storage.Engram{
		Concept: "dup-b", Content: "content b", Confidence: 0.5, Relevance: 0.5,
		Stability: 20, Embedding: nearEmbedding(10, 0, 1, 0.97),
	})
	writeUniqueEngrams(t, ctx, store, db, wsPrefix, 6, 10, 2)

	for i := 0; i < 17; i++ {
		writeEngramWithEmbedding(t, ctx, store, db, wsPrefix, &storage.Engram{
			Concept: "no-embed", Content: "no embedding engram", Confidence: 0.5, Relevance: 0.5,
			Stability: 10,
		})
	}

	w := NewWorker(&mockEngineInterface{store: store})
	report, err := w.DreamOnce(ctx, DreamOpts{Force: true, Scope: vault})
	if err != nil {
		t.Fatal(err)
	}

	if len(report.Reports) != 1 {
		t.Fatalf("expected 1 report, got %d", len(report.Reports))
	}
	if report.Reports[0].Orient == nil {
		t.Fatal("expected orient summary")
	}
	if got := report.Reports[0].Orient.EngramCount; got != 25 {
		t.Fatalf("EngramCount = %d, want 25", got)
	}
	if got := report.Reports[0].Orient.WithEmbed; got != 8 {
		t.Fatalf("WithEmbed = %d, want 8", got)
	}
	if got := report.Reports[0].DedupClusters; got != 0 {
		t.Fatalf("DedupClusters = %d, want 0 because only 8 engrams participate in dream dedup", got)
	}
	if got := report.Reports[0].MergedEngrams; got != 0 {
		t.Fatalf("MergedEngrams = %d, want 0 because guard should skip when only 8 engrams have embeddings", got)
	}
}

func writeUniqueEngrams(t *testing.T, ctx context.Context, store *storage.PebbleStore, db *pebble.DB, wsPrefix [8]byte, count, dims, startAxis int) {
	t.Helper()

	for i := 0; i < count; i++ {
		writeEngramWithEmbedding(t, ctx, store, db, wsPrefix, &storage.Engram{
			Concept:    "unique",
			Content:    "unique content",
			Confidence: 0.7,
			Relevance:  0.6,
			Stability:  20,
			Embedding:  unitEmbedding(dims, startAxis+i),
		})
	}
}

func unitEmbedding(dims, axis int) []float32 {
	embed := make([]float32, dims)
	embed[axis] = 1
	return embed
}

func nearEmbedding(dims, primaryAxis, secondaryAxis int, cosine float32) []float32 {
	embed := make([]float32, dims)
	embed[primaryAxis] = cosine
	embed[secondaryAxis] = orthogonalWeight(cosine)
	return embed
}

func orthogonalWeight(cosine float32) float32 {
	return float32(math.Sqrt(1 - float64(cosine*cosine)))
}

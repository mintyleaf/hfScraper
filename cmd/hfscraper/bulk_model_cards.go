package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/parquet-go/parquet-go"
)

// modelCardSnapshotRow is intentionally a projection. parquet-go reads only
// the repository id and card text from the wider librarian-bots table.
type modelCardSnapshotRow struct {
	ModelID string `parquet:"modelId"`
	Created string `parquet:"createdAt"`
	Card    string `parquet:"card"`
}

func ensureBulkModelCards(ctx context.Context, client *httpClient, address, path string, logger *catalogLogger) error {
	if info, err := os.Stat(path); err == nil && info.Size() > 0 {
		logger.info("bulk_model_cards_cached", "using cached bulk model-card snapshot", map[string]any{"path": path, "bytes": info.Size()})
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create bulk cache directory: %w", err)
	}
	partial := path + ".part"
	// The catalog client has a short whole-request timeout suitable for JSON API
	// calls. A multi-gigabyte streamed snapshot must not inherit that deadline;
	// cancellation is still controlled by req.Context().
	downloadClient := &http.Client{Transport: client.client.Transport}
	var resp *http.Response
	offset := int64(0)
	for attempt := 0; attempt <= client.retries; attempt++ {
		offset = 0
		if info, err := os.Stat(partial); err == nil {
			offset = info.Size()
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, address, nil)
		if err != nil {
			return err
		}
		req.Header.Set("User-Agent", userAgent)
		if client.token != "" {
			req.Header.Set("Authorization", "Bearer "+client.token)
		}
		if offset > 0 {
			req.Header.Set("Range", fmt.Sprintf("bytes=%d-", offset))
		}
		resp, err = downloadClient.Do(req)
		if err == nil && (resp.StatusCode == http.StatusOK || resp.StatusCode == http.StatusPartialContent) {
			break
		}
		delay := time.Duration(1<<attempt) * time.Second
		status := "network error"
		if resp != nil {
			status = resp.Status
			if serverDelay := rateLimitDelay(resp.Header); serverDelay > delay {
				delay = serverDelay
			}
			_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<20))
			_ = resp.Body.Close()
		}
		if err == nil && resp != nil && !retryableStatus(resp.StatusCode) {
			return fmt.Errorf("download bulk model cards: HTTP %s", resp.Status)
		}
		if attempt == client.retries {
			if err != nil {
				return fmt.Errorf("download bulk model cards after %d attempts: %w", attempt+1, err)
			}
			return fmt.Errorf("download bulk model cards after %d attempts: HTTP %s", attempt+1, status)
		}
		logger.warn("bulk_model_cards_retry", "bulk model-card download request will be retried", map[string]any{"path": partial, "attempt": attempt + 1, "delay_ms": delay.Milliseconds(), "status": status, "offset": offset})
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(delay):
		}
	}
	defer resp.Body.Close()
	flags := os.O_CREATE | os.O_WRONLY
	if resp.StatusCode == http.StatusPartialContent && offset > 0 {
		flags |= os.O_APPEND
	} else {
		flags |= os.O_TRUNC
		offset = 0
	}
	file, err := os.OpenFile(partial, flags, 0o644)
	if err != nil {
		return err
	}
	started := time.Now()
	lastReport := started
	written := offset
	buffer := make([]byte, 1<<20)
	for {
		n, readErr := resp.Body.Read(buffer)
		if n > 0 {
			if _, err := file.Write(buffer[:n]); err != nil {
				file.Close()
				return fmt.Errorf("write bulk model-card cache: %w", err)
			}
			written += int64(n)
		}
		if time.Since(lastReport) >= 30*time.Second {
			logger.info("bulk_model_cards_download_progress", "downloading bulk model-card snapshot", map[string]any{"bytes": written, "path": partial})
			lastReport = time.Now()
		}
		if readErr != nil {
			if errors.Is(readErr, io.EOF) {
				break
			}
			file.Close()
			return fmt.Errorf("read bulk model-card response: %w", readErr)
		}
	}
	if err := file.Sync(); err != nil {
		file.Close()
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	if err := os.Rename(partial, path); err != nil {
		return fmt.Errorf("publish bulk model-card cache: %w", err)
	}
	logger.info("bulk_model_cards_downloaded", "bulk model-card snapshot downloaded", map[string]any{"bytes": written, "duration_seconds": time.Since(started).Seconds(), "path": path})
	return nil
}

func resolveTrainingSignalsFromParquet(ctx context.Context, client *httpClient, addresses []string, path string, models []catalogModel, computeWanted, scratchWanted map[string]bool, logger *catalogLogger) (map[string]*reportedTrainingCompute, map[string]*reportedScratchClaim, error) {
	if path == "" {
		return nil, nil, errors.New("bulk_model_cards_file is required when bulk_model_cards_url is set")
	}
	computeResult := make(map[string]*reportedTrainingCompute)
	scratchResult := make(map[string]*reportedScratchClaim)
	computeEarliest := make(map[string]string)
	scratchEarliest := make(map[string]string)
	type shardResult struct {
		compute         map[string]*reportedTrainingCompute
		scratch         map[string]*reportedScratchClaim
		computeEarliest map[string]string
		scratchEarliest map[string]string
		err             error
	}
	results := make([]shardResult, len(addresses))
	var workers sync.WaitGroup
	for index, address := range addresses {
		workers.Add(1)
		go func() {
			defer workers.Done()
			shardPath := path
			if len(addresses) > 1 {
				extension := filepath.Ext(path)
				shardPath = strings.TrimSuffix(path, extension) + fmt.Sprintf("-%04d", index) + extension
			}
			if err := ensureBulkModelCards(ctx, client, address, shardPath, logger); err != nil {
				results[index].err = fmt.Errorf("bulk model-card shard %d: %w", index, err)
				return
			}
			computeShard, scratchShard, computeEarliestShard, scratchEarliestShard, err := scanTrainingSignalsParquet(ctx, shardPath, models, computeWanted, scratchWanted, logger)
			if err != nil {
				results[index].err = fmt.Errorf("scan bulk model-card shard %d: %w", index, err)
				return
			}
			results[index] = shardResult{compute: computeShard, scratch: scratchShard, computeEarliest: computeEarliestShard, scratchEarliest: scratchEarliestShard}
		}()
	}
	workers.Wait()
	for _, shard := range results {
		if shard.err != nil {
			return nil, nil, shard.err
		}
		for id, compute := range shard.compute {
			computeResult[id] = compute
		}
		for id, claim := range shard.scratch {
			scratchResult[id] = claim
		}
		for fingerprint, created := range shard.computeEarliest {
			if current := computeEarliest[fingerprint]; current == "" || created < current {
				computeEarliest[fingerprint] = created
			}
		}
		for fingerprint, created := range shard.scratchEarliest {
			if current := scratchEarliest[fingerprint]; current == "" || created < current {
				scratchEarliest[fingerprint] = created
			}
		}
	}
	for _, compute := range computeResult {
		if compute == nil {
			continue
		}
		compute.EarliestCardCreatedAt = earliestNonEmpty(computeEarliest["evidence:"+compute.EvidenceHash], computeEarliest["run:"+compute.RunHash])
	}
	for id, claim := range scratchResult {
		if claim == nil {
			continue
		}
		claim.EarliestCardCreatedAt = earliestNonEmpty(scratchEarliest["card:"+claim.CardHash], scratchEarliest["run:"+claim.RunHash], scratchEarliest["family:"+canonicalBaseFamilyName(id)])
	}
	return computeResult, scratchResult, nil
}

func scanReportedComputeParquet(ctx context.Context, path string, models []catalogModel, logger *catalogLogger) (map[string]*reportedTrainingCompute, error) {
	computeWanted := make(map[string]bool, len(models))
	for _, model := range models {
		computeWanted[model.ID] = true
	}
	compute, _, _, _, err := scanTrainingSignalsParquet(ctx, path, models, computeWanted, nil, logger)
	return compute, err
}

func scanTrainingSignalsParquet(ctx context.Context, path string, models []catalogModel, computeWanted, scratchWanted map[string]bool, logger *catalogLogger) (map[string]*reportedTrainingCompute, map[string]*reportedScratchClaim, map[string]string, map[string]string, error) {
	wanted := make(map[string]bool, len(models))
	for _, model := range models {
		wanted[model.ID] = true
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	defer file.Close()
	reader := parquet.NewGenericReader[modelCardSnapshotRow](file)
	defer reader.Close()
	computeResult := make(map[string]*reportedTrainingCompute)
	scratchResult := make(map[string]*reportedScratchClaim)
	computeEarliest := make(map[string]string)
	scratchEarliest := make(map[string]string)
	rows := make([]modelCardSnapshotRow, 512)
	scanned := 0
	matched := 0
	withDescription := 0
	for {
		if err := ctx.Err(); err != nil {
			return nil, nil, nil, nil, err
		}
		n, readErr := reader.Read(rows)
		for _, row := range rows[:n] {
			scanned++
			var compute *reportedTrainingCompute
			if len(computeWanted) > 0 && row.Card != "" && (computeWanted[row.ModelID] || potentialComputeText(row.Card)) {
				compute = reportedComputeFromText(row.Card, "model-card bulk snapshot")
				if compute != nil && row.Created != "" {
					for _, fingerprint := range []string{"evidence:" + compute.EvidenceHash, "run:" + compute.RunHash} {
						if current := computeEarliest[fingerprint]; current == "" || row.Created < current {
							computeEarliest[fingerprint] = row.Created
						}
					}
				}
			}
			var scratch *reportedScratchClaim
			if scratchWanted[row.ModelID] || potentialScratchText(row.Card) || declaredBaseModelNameRE.MatchString(row.ModelID) {
				scratch = scratchClaimFromText(row.Card, "model-card bulk snapshot", row.ModelID)
				if scratch != nil && row.Created != "" {
					for _, fingerprint := range []string{"card:" + scratch.CardHash, "run:" + scratch.RunHash, "family:" + canonicalBaseFamilyName(row.ModelID)} {
						if current := scratchEarliest[fingerprint]; current == "" || row.Created < current {
							scratchEarliest[fingerprint] = row.Created
						}
					}
				}
			}
			if !wanted[row.ModelID] {
				continue
			}
			matched++
			if row.Card == "" {
				continue
			}
			withDescription++
			if computeWanted[row.ModelID] {
				if compute != nil {
					computeResult[row.ModelID] = compute
				}
			}
			if scratchWanted[row.ModelID] {
				// Presence with a nil value is meaningful: the complete card was
				// inspected and rejected the weaker cardData fallback.
				scratchResult[row.ModelID] = scratch
			}
		}
		if errors.Is(readErr, io.EOF) {
			break
		}
		if readErr != nil {
			return nil, nil, nil, nil, fmt.Errorf("scan bulk model-card parquet after %d rows: %w", scanned, readErr)
		}
	}
	if logger != nil {
		scratchResolved := 0
		for _, claim := range scratchResult {
			if claim != nil {
				scratchResolved++
			}
		}
		logger.info("bulk_model_cards_scanned", "bulk model-card snapshot scanned", map[string]any{"rows": scanned, "candidates": len(wanted), "matched": matched, "with_description": withDescription, "compute_resolved": len(computeResult), "scratch_resolved": scratchResolved})
	}
	return computeResult, scratchResult, computeEarliest, scratchEarliest, nil
}

func potentialComputeText(text string) bool {
	lower := strings.ToLower(text)
	for _, needle := range []string{
		"gpu hr", "gpu-hr", "gpu_hr", "gpu hour", "gpu-hour", "gpu_hour",
		"h100 hr", "h100-hr", "h100_hr", "h100 hour", "h100-hour",
		"training time", "training duration", "training took", "train_runtime",
		"trained for", "pretrained for", "pre-trained for", "fine-tuned for", "finetuned for",
		"fine-tuning took", "finetuning took", "hours used", "training cost",
		"pretraining cost", "pre-training cost", "cost of training", "cost to train",
	} {
		if strings.Contains(lower, needle) {
			return true
		}
	}
	return strings.Contains(lower, "training") && strings.Contains(text, "$")
}

func potentialScratchText(text string) bool {
	lower := strings.ToLower(text)
	for _, needle := range []string{"from scratch", "random initialization", "randomly initialized", "from the ground up", "с нуля", "pretrained on", "pre-trained on", "pretraining", "pre-training", "foundation model"} {
		if strings.Contains(lower, needle) {
			return true
		}
	}
	return false
}

func earliestNonEmpty(values ...string) string {
	result := ""
	for _, value := range values {
		if value != "" && (result == "" || value < result) {
			result = value
		}
	}
	return result
}

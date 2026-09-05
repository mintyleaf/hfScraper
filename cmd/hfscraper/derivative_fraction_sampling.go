package main

import (
	"context"
	"crypto/sha256"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

const derivativeFractionPromptVersion = "derivative-lineage-fraction-v1"

type derivativeFractionResult struct {
	Derivative             derivativeManifestEntry `json:"derivative"`
	Review                 localLLMReview          `json:"local_llm_review"`
	ParametersUsed         int64                   `json:"parameters_used,omitempty"`
	ParameterSource        string                  `json:"parameter_source_used,omitempty"`
	BaseFormulaCostUSD     float64                 `json:"base_formula_cost_usd,omitempty"`
	FractionCostUSD        float64                 `json:"fraction_cost_usd,omitempty"`
	WeightedCentralCostUSD float64                 `json:"weighted_central_cost_usd,omitempty"`
	WeightedUpperCostUSD   float64                 `json:"weighted_upper_cost_usd,omitempty"`
	DuplicateOf            string                  `json:"duplicate_of,omitempty"`
	ExclusionReason        string                  `json:"exclusion_reason,omitempty"`
}

type derivativeFractionBreakdown struct {
	Sampled             int     `json:"sampled"`
	HighConfidence      int     `json:"high_confidence"`
	MediumConfidence    int     `json:"medium_confidence"`
	CostedCentral       int     `json:"costed_central"`
	CostedUpper         int     `json:"costed_upper"`
	Duplicates          int     `json:"duplicates"`
	UnknownParameters   int     `json:"unknown_parameters"`
	EstimatedCentralUSD float64 `json:"estimated_central_usd"`
	EstimatedUpperUSD   float64 `json:"estimated_upper_usd"`
}

type derivativeFractionSummary struct {
	GeneratedAt         string                                 `json:"generated_at"`
	Method              string                                 `json:"method"`
	BaseCostFraction    float64                                `json:"base_cost_fraction"`
	Plan                derivativePlanSummary                  `json:"plan"`
	Reviewed            int                                    `json:"reviewed"`
	HighConfidence      int                                    `json:"high_confidence"`
	MediumConfidence    int                                    `json:"medium_confidence"`
	LowConfidence       int                                    `json:"low_confidence"`
	TrainingDerivatives int                                    `json:"training_derivatives"`
	ExcludedNonTraining int                                    `json:"excluded_non_training"`
	Duplicates          int                                    `json:"duplicates"`
	UnknownParameters   int                                    `json:"unknown_parameters"`
	EstimatedCentralUSD float64                                `json:"estimated_central_usd"`
	EstimatedUpperUSD   float64                                `json:"estimated_upper_usd"`
	ByMarket            map[string]derivativeFractionBreakdown `json:"by_market"`
	ByKind              map[string]derivativeFractionBreakdown `json:"by_kind"`
	Warnings            []string                               `json:"warnings"`
}

var parameterBillionsInNameRE = regexp.MustCompile(`(?i)(?:^|[-_/.])([0-9]+(?:\.[0-9]+)?)b(?:$|[-_/.])`)
var parameterCountInEvidenceRE = regexp.MustCompile(`(?i)([0-9]+(?:\.[0-9]+)?)\s*(b|bn|billion|m|mn|million)\b`)

func preflightDerivativeLineageLLM(ctx context.Context, config catalogConfig) error {
	callConfig := config.LocalLLM
	callConfig.Scope = "text_llm_derivative_fraction"
	input := localLLMReviewInput{
		RepoID: "preflight/model-7b-lora", BaseRelation: "adapter", BaseModels: []string{"preflight/model-7b"},
		Parameters: 7_000_000_000, Card: "This LoRA adapter is trained from preflight/model-7b.", CardHash: "preflight", CacheKey: "preflight",
	}
	client := &http.Client{Timeout: time.Duration(callConfig.TimeoutSeconds) * time.Second}
	if _, err := callLocalLLM(ctx, client, callConfig, input); err != nil {
		return fmt.Errorf("derivative lineage LLM preflight failed: %w", err)
	}
	return nil
}

func derivativeLineageCacheKey(entry derivativeManifestEntry, text string) string {
	normalized := strings.Join(strings.Fields(text), " ")
	payload := strings.Join([]string{derivativeFractionPromptVersion, entry.RepoID, entry.Market, entry.Kind, entry.BaseModel, normalized}, "\x00")
	return fmt.Sprintf("%x", sha256.Sum256([]byte(payload)))
}

func reviewDerivativeLineage(ctx context.Context, config catalogConfig, manifest []derivativeManifestEntry, texts map[string]string, logger *catalogLogger) (map[string]localLLMReview, error) {
	cachePath := resolveOutputPath(config.OutputDir, config.LocalLLM.CacheFile)
	cache, err := loadLocalLLMCache(cachePath)
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(filepath.Dir(cachePath), 0o755); err != nil {
		return nil, err
	}
	cacheFile, err := os.OpenFile(cachePath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, err
	}
	defer cacheFile.Close()

	result := make(map[string]localLLMReview, len(manifest))
	jobs := make(chan derivativeManifestEntry, config.LocalLLM.Workers*2)
	client := &http.Client{Timeout: time.Duration(config.LocalLLM.TimeoutSeconds) * time.Second}
	var workers sync.WaitGroup
	var resultMutex sync.Mutex
	var cacheMutex sync.Mutex
	completed, called, cacheHits, failed := 0, 0, 0, 0
	for range config.LocalLLM.Workers {
		workers.Add(1)
		go func() {
			defer workers.Done()
			for entry := range jobs {
				text := strings.TrimSpace(texts[entry.RepoID])
				if text == "" {
					resultMutex.Lock()
					result[entry.RepoID] = localLLMReview{Kind: "unknown", Confidence: "low", Reason: "model card is unavailable", Source: "local LLM prefilter"}
					completed++
					resultMutex.Unlock()
					continue
				}
				key := derivativeLineageCacheKey(entry, text)
				resultMutex.Lock()
				cached, ok := cache[key]
				if ok {
					result[entry.RepoID] = cached
					cacheHits++
					completed++
				}
				resultMutex.Unlock()
				if ok {
					continue
				}
				callConfig := config.LocalLLM
				callConfig.Scope = entry.Market + "_derivative_fraction"
				baseModels := []string{}
				if entry.BaseModel != "" {
					baseModels = append(baseModels, entry.BaseModel)
				}
				cardHash := fmt.Sprintf("%x", sha256.Sum256([]byte(strings.Join(strings.Fields(text), " "))))
				input := localLLMReviewInput{
					RepoID: entry.RepoID, PipelineTag: entry.PipelineTag, LibraryName: entry.LibraryName, Tags: entry.Tags,
					BaseRelation: entry.BaseRelation, BaseModels: baseModels, Parameters: entry.Parameters,
					Card: compactModelCard(text, callConfig.MaxInputChars), CardHash: cardHash, CacheKey: key,
				}
				review, callErr := callLocalLLM(ctx, client, callConfig, input)
				resultMutex.Lock()
				called++
				if callErr != nil {
					failed++
					review = localLLMReview{Kind: "unknown", Confidence: "low", Reason: callErr.Error(), Source: "local OpenAI-compatible LLM " + callConfig.Model, CardHash: cardHash, CacheKey: key}
					logger.warn("derivative_lineage_llm_failed", "derivative lineage review failed", map[string]any{"repo_id": entry.RepoID, "error": callErr.Error()})
				} else {
					if err := appendLocalLLMCache(cacheFile, &cacheMutex, key, review); err != nil {
						logger.warn("derivative_lineage_cache_failed", "could not append derivative lineage cache", map[string]any{"repo_id": entry.RepoID, "error": err.Error()})
					}
					cache[key] = review
				}
				result[entry.RepoID] = review
				completed++
				if completed%100 == 0 || completed == len(manifest) {
					logger.info("derivative_lineage_progress", "derivative lineage review progress", map[string]any{"completed": completed, "sample": len(manifest), "called": called, "cache_hits": cacheHits, "failed": failed})
				}
				resultMutex.Unlock()
			}
		}()
	}
	for _, entry := range manifest {
		select {
		case <-ctx.Done():
			close(jobs)
			workers.Wait()
			return nil, ctx.Err()
		case jobs <- entry:
		}
	}
	close(jobs)
	workers.Wait()
	if called > 0 && failed == called {
		return nil, errors.New("all derivative lineage LLM calls failed")
	}
	return result, nil
}

func trainingDerivativeKind(kind string) bool {
	switch kind {
	case "finetune", "adapter", "continued_pretraining":
		return true
	default:
		return false
	}
}

func parameterBillionsFromName(values ...string) float64 {
	for _, value := range values {
		matches := parameterBillionsInNameRE.FindAllStringSubmatch(value, -1)
		for i := len(matches) - 1; i >= 0; i-- {
			billions, err := strconv.ParseFloat(matches[i][1], 64)
			if err == nil && billions >= 0.01 && billions <= 1000 {
				return billions
			}
		}
	}
	return 0
}

func parameterBillionsFromEvidence(evidence string) float64 {
	match := parameterCountInEvidenceRE.FindStringSubmatch(evidence)
	if len(match) != 3 {
		return 0
	}
	value, err := strconv.ParseFloat(match[1], 64)
	if err != nil || value <= 0 {
		return 0
	}
	switch strings.ToLower(match[2]) {
	case "m", "mn", "million":
		value /= 1000
	}
	if value > 1000 {
		return 0
	}
	return value
}

func derivativeFractionParameters(entry derivativeManifestEntry, review localLLMReview) (int64, string) {
	if entry.Parameters > 0 {
		return entry.Parameters, entry.ParameterSource
	}
	if review.ReportedParametersB > 0 {
		// The evidence is deterministic and authoritative for the unit. A model
		// may copy "900M parameters" correctly but emit 900 (or 600) in a field
		// expressed in billions; never price that unchecked numeric conversion.
		if billions := parameterBillionsFromEvidence(review.ParameterEvidence); billions > 0 {
			return int64(billions * 1e9), "verbatim_model_card_parameter_evidence"
		}
	}
	if billions := parameterBillionsFromName(review.UpstreamModel, entry.BaseModel, entry.RepoID); billions > 0 {
		return int64(billions * 1e9), "model_name_parameter_hint"
	}
	return 0, "unresolved"
}

func derivativeFractionBaseCost(parameters int64, market string, profiles []computeProfile) (float64, string, error) {
	for _, profile := range profiles {
		if market == "diffusion" && hasDiffusionProfile(profile) {
			return estimateDiffusionBaseTrainingCompute(parameters, nil, profile).CostUSD, profile.Name, nil
		}
		if market == "text_llm" && !hasDiffusionProfile(profile) {
			return estimateCompute(parameters, "base", profile).CostUSD, profile.Name, nil
		}
	}
	return 0, "", fmt.Errorf("no compute profile for market %q", market)
}

func buildDerivativeFractionResults(config catalogConfig, manifest []derivativeManifestEntry, reviews map[string]localLLMReview) ([]derivativeFractionResult, error) {
	results := make([]derivativeFractionResult, 0, len(manifest))
	for _, entry := range manifest {
		review := reviews[entry.RepoID]
		result := derivativeFractionResult{Derivative: entry, Review: review}
		if !trainingDerivativeKind(review.Kind) {
			result.ExclusionReason = "not a training derivative"
			results = append(results, result)
			continue
		}
		if review.Confidence != "high" && review.Confidence != "medium" {
			result.ExclusionReason = "lineage confidence is low"
			results = append(results, result)
			continue
		}
		result.ParametersUsed, result.ParameterSource = derivativeFractionParameters(entry, review)
		if result.ParametersUsed <= 0 {
			result.ExclusionReason = "full base/checkpoint parameter count unresolved"
			results = append(results, result)
			continue
		}
		baseCost, _, err := derivativeFractionBaseCost(result.ParametersUsed, entry.Market, config.Compute)
		if err != nil {
			return nil, err
		}
		result.BaseFormulaCostUSD = baseCost
		result.FractionCostUSD = baseCost * config.Derivative.BaseCostFraction
		if review.Confidence == "high" {
			result.WeightedCentralCostUSD = result.FractionCostUSD * entry.SamplingWeight
		}
		result.WeightedUpperCostUSD = result.FractionCostUSD * entry.SamplingWeight
		results = append(results, result)
	}
	deduplicateDerivativeFractionResults(results)
	return results, nil
}

func deduplicateDerivativeFractionResults(results []derivativeFractionResult) {
	seen := map[string]int{}
	for i := range results {
		canonical := strings.ToLower(strings.TrimSpace(results[i].Review.CanonicalTrainingRun))
		if canonical == "" || results[i].FractionCostUSD <= 0 {
			continue
		}
		key := results[i].Derivative.Market + "|" + canonical + "|" + strconv.FormatInt(results[i].ParametersUsed, 10)
		if prior, ok := seen[key]; ok {
			results[i].DuplicateOf = results[prior].Derivative.RepoID
			results[i].WeightedCentralCostUSD = 0
			results[i].WeightedUpperCostUSD = 0
			results[i].ExclusionReason = "duplicate canonical derivative training run"
		} else {
			seen[key] = i
		}
	}
}

func addDerivativeFractionBreakdown(value derivativeFractionBreakdown, result derivativeFractionResult) derivativeFractionBreakdown {
	value.Sampled++
	switch result.Review.Confidence {
	case "high":
		value.HighConfidence++
	case "medium":
		value.MediumConfidence++
	}
	if result.DuplicateOf != "" {
		value.Duplicates++
	}
	if result.ExclusionReason == "full base/checkpoint parameter count unresolved" {
		value.UnknownParameters++
	}
	if result.WeightedCentralCostUSD > 0 {
		value.CostedCentral++
		value.EstimatedCentralUSD += result.WeightedCentralCostUSD
	}
	if result.WeightedUpperCostUSD > 0 {
		value.CostedUpper++
		value.EstimatedUpperUSD += result.WeightedUpperCostUSD
	}
	return value
}

func aggregateDerivativeFractionResults(config catalogConfig, plan derivativePlanSummary, results []derivativeFractionResult) derivativeFractionSummary {
	summary := derivativeFractionSummary{
		GeneratedAt:      time.Now().UTC().Format(time.RFC3339Nano),
		Method:           "stratified Horvitz-Thompson estimate; local-LLM lineage classification and canonical-run deduplication; derivative cost = configured fraction of corresponding base-training formula",
		BaseCostFraction: config.Derivative.BaseCostFraction, Plan: plan,
		ByMarket: map[string]derivativeFractionBreakdown{}, ByKind: map[string]derivativeFractionBreakdown{},
	}
	for _, result := range results {
		summary.Reviewed++
		switch result.Review.Confidence {
		case "high":
			summary.HighConfidence++
		case "medium":
			summary.MediumConfidence++
		default:
			summary.LowConfidence++
		}
		if trainingDerivativeKind(result.Review.Kind) {
			summary.TrainingDerivatives++
		} else {
			summary.ExcludedNonTraining++
		}
		if result.DuplicateOf != "" {
			summary.Duplicates++
		}
		if result.ExclusionReason == "full base/checkpoint parameter count unresolved" {
			summary.UnknownParameters++
		}
		summary.EstimatedCentralUSD += result.WeightedCentralCostUSD
		summary.EstimatedUpperUSD += result.WeightedUpperCostUSD
		summary.ByMarket[result.Derivative.Market] = addDerivativeFractionBreakdown(summary.ByMarket[result.Derivative.Market], result)
		summary.ByKind[result.Review.Kind] = addDerivativeFractionBreakdown(summary.ByKind[result.Review.Kind], result)
	}
	summary.Warnings = append(summary.Warnings,
		"Central includes only high-confidence training derivatives; upper additionally includes medium-confidence training derivatives.",
		"Pure mirrors, forks, quantizations, conversions, and merges receive zero cost.",
		"Sampling uncertainty is not a formal confidence interval; missing cards, unresolved parameters, and unseen duplicates can bias the estimate.",
	)
	return summary
}

func writeDerivativeFractionResultsCSV(path string, results []derivativeFractionResult) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	w := csv.NewWriter(file)
	_ = w.Write([]string{"repo_id", "market", "metadata_kind", "review_kind", "confidence", "upstream_model", "canonical_training_run", "parameters_used", "parameter_source", "sampling_weight", "base_formula_cost_usd", "fraction_cost_usd", "weighted_central_cost_usd", "weighted_upper_cost_usd", "duplicate_of", "exclusion_reason", "evidence"})
	for _, result := range results {
		_ = w.Write([]string{
			result.Derivative.RepoID, result.Derivative.Market, result.Derivative.Kind, result.Review.Kind, result.Review.Confidence,
			result.Review.UpstreamModel, result.Review.CanonicalTrainingRun, strconv.FormatInt(result.ParametersUsed, 10), result.ParameterSource,
			strconv.FormatFloat(result.Derivative.SamplingWeight, 'g', -1, 64), strconv.FormatFloat(result.BaseFormulaCostUSD, 'f', 6, 64),
			strconv.FormatFloat(result.FractionCostUSD, 'f', 6, 64), strconv.FormatFloat(result.WeightedCentralCostUSD, 'f', 6, 64),
			strconv.FormatFloat(result.WeightedUpperCostUSD, 'f', 6, 64), result.DuplicateOf, result.ExclusionReason, result.Review.Evidence,
		})
	}
	w.Flush()
	if err := w.Error(); err != nil {
		file.Close()
		return err
	}
	return file.Close()
}

func writeDerivativeFractionReport(path string, summary derivativeFractionSummary) error {
	var report strings.Builder
	report.WriteString("# Оценка derivative training по правилу 1%\n\n")
	fmt.Fprintf(&report, "- Доля стоимости base training: **%.2f%%**\n", summary.BaseCostFraction*100)
	fmt.Fprintf(&report, "- Population: %d (text: %d; diffusion: %d)\n", summary.Plan.Population, summary.Plan.TextModels, summary.Plan.DiffusionModels)
	fmt.Fprintf(&report, "- Проверено local LLM: %d; high=%d, medium=%d, low/unknown=%d\n", summary.Reviewed, summary.HighConfidence, summary.MediumConfidence, summary.LowConfidence)
	fmt.Fprintf(&report, "- Training derivatives в sample: %d; non-training исключено: %d; canonical duplicates: %d; unknown parameters: %d\n", summary.TrainingDerivatives, summary.ExcludedNonTraining, summary.Duplicates, summary.UnknownParameters)
	fmt.Fprintf(&report, "- **Central (high confidence): $%.2f**\n", summary.EstimatedCentralUSD)
	fmt.Fprintf(&report, "- **Upper (high + medium confidence): $%.2f**\n\n", summary.EstimatedUpperUSD)
	report.WriteString("## По рынкам\n\n| Рынок | Sample | Central costed | Upper costed | Central, USD | Upper, USD |\n|---|---:|---:|---:|---:|---:|\n")
	markets := make([]string, 0, len(summary.ByMarket))
	for market := range summary.ByMarket {
		markets = append(markets, market)
	}
	sort.Strings(markets)
	for _, market := range markets {
		value := summary.ByMarket[market]
		fmt.Fprintf(&report, "| %s | %d | %d | %d | %.2f | %.2f |\n", market, value.Sampled, value.CostedCentral, value.CostedUpper, value.EstimatedCentralUSD, value.EstimatedUpperUSD)
	}
	report.WriteString("\n## Правила\n\n- Local LLM определяет lineage, confidence, upstream model и canonical training run по model card.\n- Fine-tune, continued pretraining и adapters получают configured fraction от полной base-training формулы.\n- Mirrors, forks, quantization/conversion и merges получают ноль.\n- Для adapters используется размер полной base model; размер adapter-файла не заменяет его.\n- Central принимает high-confidence evidence; upper дополнительно принимает medium confidence.\n")
	return os.WriteFile(path, []byte(report.String()), 0o644)
}

func runDerivativeFractionEstimate(ctx context.Context, config catalogConfig, manifest []derivativeManifestEntry, texts map[string]string, logger *catalogLogger) error {
	reviews, err := reviewDerivativeLineage(ctx, config, manifest, texts, logger)
	if err != nil {
		return err
	}
	results, err := buildDerivativeFractionResults(config, manifest, reviews)
	if err != nil {
		return err
	}
	planPath := filepath.Join(config.OutputDir, "derivative-plan-summary.json")
	var plan derivativePlanSummary
	data, err := os.ReadFile(planPath)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(data, &plan); err != nil {
		return err
	}
	summary := aggregateDerivativeFractionResults(config, plan, results)
	if err := writeJSONFile(resolveOutputPath(config.OutputDir, config.Derivative.EvidenceFile), reviews); err != nil {
		return err
	}
	resultsPath := resolveOutputPath(config.OutputDir, config.Derivative.ResultsFile)
	if err := writeJSONFile(resultsPath, results); err != nil {
		return err
	}
	if err := writeDerivativeFractionResultsCSV(strings.TrimSuffix(resultsPath, filepath.Ext(resultsPath))+".csv", results); err != nil {
		return err
	}
	if err := writeJSONFile(resolveOutputPath(config.OutputDir, config.Derivative.SummaryFile), summary); err != nil {
		return err
	}
	reportPath := resolveOutputPath(config.OutputDir, config.Derivative.ReportFile)
	if err := writeDerivativeFractionReport(reportPath, summary); err != nil {
		return err
	}
	fmt.Printf("Derivative 1%% estimate complete: central=$%.2f upper=$%.2f report=%s\n", summary.EstimatedCentralUSD, summary.EstimatedUpperUSD, reportPath)
	return nil
}

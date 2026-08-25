package main

import (
	"context"
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/parquet-go/parquet-go"
)

// marketSamplingConfig controls the bounded, auditable alternative to fetching
// every README on the Hub. Planning is always a separate, network-free step;
// targeted README and local-LLM calls require the explicit --execute flag.
type marketSamplingConfig struct {
	Enabled              bool    `json:"enabled"`
	Seed                 int64   `json:"seed"`
	CensusMinParametersB float64 `json:"census_min_parameters_b"`
	SamplePerStratum     int     `json:"sample_per_stratum"`
	MaxTargetedReadmes   int     `json:"max_targeted_readmes"`
	ReadmeWorkers        int     `json:"readme_workers"`
	RequestIntervalMS    int     `json:"request_interval_ms"`
	ReadmeCacheDir       string  `json:"readme_cache_dir"`
	ManifestFile         string  `json:"manifest_file"`
	ResultsFile          string  `json:"results_file"`
	SummaryFile          string  `json:"summary_file"`
	ReportFile           string  `json:"report_file"`
	BaselineSummaryFile  string  `json:"baseline_summary_file"`
	IncludeCoveredCensus *bool   `json:"include_covered_census"`
	IncludeMissingCensus *bool   `json:"include_missing_census"`
}

type samplingCandidate struct {
	Model     catalogModel
	Selection compiledSelection
	Scope     string
	Params    int64
	Bucket    string
	Covered   bool
}

type samplingManifestEntry struct {
	Selection            string   `json:"selection"`
	Scope                string   `json:"scope"`
	RepoID               string   `json:"repo_id"`
	Owner                string   `json:"owner"`
	CreatedAt            string   `json:"created_at"`
	Parameters           int64    `json:"parameters"`
	ParameterRange       string   `json:"parameter_range"`
	Downloads            int64    `json:"downloads"`
	Likes                int64    `json:"likes"`
	PipelineTag          string   `json:"pipeline_tag"`
	LibraryName          string   `json:"library_name"`
	Tags                 []string `json:"tags"`
	CardSource           string   `json:"card_source"`
	Census               bool     `json:"census"`
	Stratum              string   `json:"stratum"`
	StratumPopulation    int      `json:"stratum_population"`
	StratumSample        int      `json:"stratum_sample"`
	InclusionProbability float64  `json:"inclusion_probability"`
	SamplingWeight       float64  `json:"sampling_weight"`
}

type samplingPlanSummary struct {
	GeneratedAt            string         `json:"generated_at"`
	Seed                   int64          `json:"seed"`
	Candidates             int            `json:"base_candidates"`
	CoveredCandidates      int            `json:"parquet_covered_candidates"`
	MissingCandidates      int            `json:"parquet_missing_candidates"`
	ManifestEntries        int            `json:"manifest_entries"`
	CoveredManifestEntries int            `json:"covered_manifest_entries"`
	TargetedREADMERequests int            `json:"targeted_readme_requests"`
	CensusEntries          int            `json:"census_entries"`
	CandidatesBySelection  map[string]int `json:"base_candidates_by_selection"`
	ManifestBySelection    map[string]int `json:"manifest_entries_by_selection"`
	PopulationByStratum    map[string]int `json:"population_by_stratum"`
	SampleByStratum        map[string]int `json:"sample_by_stratum"`
}

type samplingResult struct {
	Selection              string          `json:"selection"`
	Scope                  string          `json:"scope"`
	RepoID                 string          `json:"repo_id"`
	Owner                  string          `json:"owner"`
	Parameters             int64           `json:"parameters"`
	ParameterSource        string          `json:"parameter_source"`
	ParameterRange         string          `json:"parameter_range"`
	CardSource             string          `json:"card_source"`
	CardAvailable          bool            `json:"card_available"`
	Census                 bool            `json:"census"`
	Stratum                string          `json:"stratum"`
	StratumPopulation      int             `json:"stratum_population"`
	StratumSample          int             `json:"stratum_sample"`
	SamplingWeight         float64         `json:"sampling_weight"`
	Review                 *localLLMReview `json:"local_llm_review,omitempty"`
	DeterministicEvidence  string          `json:"deterministic_evidence_type,omitempty"`
	IndependentLower       bool            `json:"independent_lower"`
	IndependentCentral     bool            `json:"independent_central"`
	IndependentUpper       bool            `json:"independent_upper"`
	DuplicateOf            string          `json:"duplicate_of,omitempty"`
	RawCostUSD             float64         `json:"raw_cost_usd,omitempty"`
	CostMethod             string          `json:"cost_method,omitempty"`
	WeightedLowerCostUSD   float64         `json:"weighted_lower_cost_usd"`
	WeightedCentralCostUSD float64         `json:"weighted_central_cost_usd"`
	WeightedUpperCostUSD   float64         `json:"weighted_upper_cost_usd"`
	Error                  string          `json:"error,omitempty"`
}

type samplingStratumSummary struct {
	Population                 int     `json:"population"`
	Sample                     int     `json:"sample"`
	CardsAvailable             int     `json:"cards_available"`
	IndependentLowerWeighted   float64 `json:"independent_lower_weighted"`
	IndependentCentralWeighted float64 `json:"independent_central_weighted"`
	IndependentUpperWeighted   float64 `json:"independent_upper_weighted"`
	LowerCostUSD               float64 `json:"lower_cost_usd"`
	CentralCostUSD             float64 `json:"central_cost_usd"`
	UpperCostUSD               float64 `json:"upper_cost_usd"`
}

type samplingSelectionSummary struct {
	Candidates                  int                               `json:"base_candidates"`
	Sample                      int                               `json:"sample"`
	CardsAvailable              int                               `json:"cards_available"`
	UncostableUnknownParameters int                               `json:"uncostable_unknown_parameters"`
	EstimatedBaseLowerUSD       float64                           `json:"estimated_base_lower_usd"`
	EstimatedBaseCentralUSD     float64                           `json:"estimated_base_central_usd"`
	EstimatedBaseUpperUSD       float64                           `json:"estimated_base_upper_usd"`
	ConfirmedBaselineUSD        float64                           `json:"confirmed_baseline_usd,omitempty"`
	ReportedDerivativeUSD       float64                           `json:"reported_derivative_usd,omitempty"`
	MarketLowerUSD              float64                           `json:"market_lower_usd"`
	MarketCentralUSD            float64                           `json:"market_central_usd"`
	MarketUpperUSD              float64                           `json:"market_upper_usd"`
	Strata                      map[string]samplingStratumSummary `json:"strata"`
}

type marketSamplingSummary struct {
	GeneratedAt       string                              `json:"generated_at"`
	Method            string                              `json:"method"`
	Plan              samplingPlanSummary                 `json:"plan"`
	Selections        map[string]samplingSelectionSummary `json:"selections"`
	BaselineAvailable bool                                `json:"baseline_available"`
	Warnings          []string                            `json:"warnings"`
}

func runMarketSamplingCLI(ctx context.Context, arguments []string) error {
	flags := flag.NewFlagSet("market-sample", flag.ContinueOnError)
	configPath := flags.String("config", "configs/catalog.2025-stratified-market.local-llm.json", "path to JSON configuration")
	outputOverride := flags.String("output", "", "override output_dir from config")
	execute := flags.Bool("execute", false, "fetch targeted READMEs and call the local LLM; without this flag only the sampling plan is written")
	if err := flags.Parse(arguments); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	config, err := readCatalogConfig(*configPath)
	if err != nil {
		return err
	}
	if *outputOverride != "" {
		config.OutputDir = *outputOverride
	}
	applyCatalogDefaults(&config)
	return runMarketSampling(ctx, config, *execute)
}

func readCatalogConfig(path string) (catalogConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return catalogConfig{}, err
	}
	var config catalogConfig
	decoder := json.NewDecoder(strings.NewReader(string(data)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&config); err != nil {
		return catalogConfig{}, fmt.Errorf("decode config: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		if err == nil {
			return catalogConfig{}, errors.New("decode config: more than one JSON value")
		}
		return catalogConfig{}, fmt.Errorf("decode trailing config data: %w", err)
	}
	return config, nil
}

func runMarketSampling(ctx context.Context, config catalogConfig, execute bool) (resultErr error) {
	if !config.Sampling.Enabled {
		return errors.New("sampling.enabled must be true for market-sample")
	}
	if config.Sampling.CensusMinParametersB <= 0 || config.Sampling.SamplePerStratum <= 0 || config.Sampling.MaxTargetedReadmes < 0 {
		return errors.New("sampling census/sample/readme limits are invalid")
	}
	if config.Sampling.ReadmeWorkers < 1 || config.Sampling.ReadmeWorkers > 8 {
		return errors.New("sampling.readme_workers must be between 1 and 8")
	}
	if execute && !config.LocalLLM.Enabled {
		return errors.New("local_llm.enabled must be true with --execute")
	}
	if execute {
		if strings.TrimSpace(config.LocalLLM.BaseURL) == "" || strings.TrimSpace(config.LocalLLM.Model) == "" {
			return errors.New("HF_LLM_BASE_URL and HF_LLM_MODEL (or local_llm.base_url/model) are required with --execute")
		}
		if config.LocalLLM.Workers < 1 || config.LocalLLM.Workers > 64 || config.LocalLLM.TimeoutSeconds <= 0 || config.LocalLLM.MaxInputChars < 1000 {
			return errors.New("local_llm worker/timeout/input limits are invalid")
		}
	}
	if err := os.MkdirAll(config.OutputDir, 0o755); err != nil {
		return err
	}
	logger, err := newCatalogLogger(config.Logging, config.OutputDir)
	if err != nil {
		return err
	}
	defer func() {
		if resultErr != nil {
			logger.error("sampling_failed", "stratified market sampling failed", resultErr, nil)
		}
		resultErr = errors.Join(resultErr, logger.close())
	}()

	now := time.Now().UTC()
	if config.Scan.Now != "" {
		now, err = parseFlexibleTime(config.Scan.Now, false)
		if err != nil {
			return err
		}
	}
	selections, earliest, err := compileSelections(config, now)
	if err != nil {
		return err
	}
	latest := selections[0].to
	for _, selection := range selections[1:] {
		if selection.to.After(latest) {
			latest = selection.to
		}
	}
	checkpointPath := resolvedCheckpointPath(config.OutputDir, config.Scan.CatalogCheckpoint)
	checkpoint, err := loadCatalogCheckpoint(checkpointPath, config.Endpoint, earliest, latest, boolValue(config.Scan.RequireWeights, true))
	if err != nil || checkpoint == nil || !checkpoint.Complete {
		return fmt.Errorf("completed catalog checkpoint is required at %s: %w", checkpointPath, err)
	}
	logger.info("sampling_checkpoint_loaded", "loaded catalog checkpoint for sampling", map[string]any{"path": checkpointPath, "models": len(checkpoint.Models)})

	candidates := buildSamplingCandidates(checkpoint.Models, selections)
	if len(candidates) == 0 {
		return errors.New("sampling found no preliminary base candidates")
	}
	bulkPaths, err := resolvedBulkShardPaths(config)
	if err != nil {
		return err
	}
	covered, err := scanSamplingCoverage(ctx, bulkPaths, candidates, logger)
	if err != nil {
		return err
	}
	for i := range candidates {
		candidates[i].Covered = covered[candidates[i].Model.ID]
	}
	manifest, plan, err := buildSamplingManifest(candidates, config.Sampling)
	if err != nil {
		return err
	}
	manifestPath := resolveOutputPath(config.OutputDir, config.Sampling.ManifestFile)
	if err := writeJSONFile(manifestPath, manifest); err != nil {
		return err
	}
	if err := writeSamplingManifestCSV(strings.TrimSuffix(manifestPath, filepath.Ext(manifestPath))+".csv", manifest); err != nil {
		return err
	}
	if err := writeJSONFile(filepath.Join(config.OutputDir, "sampling-plan-summary.json"), plan); err != nil {
		return err
	}
	logger.info("sampling_plan_ready", "stratified sampling manifest written", map[string]any{"candidates": plan.Candidates, "manifest": plan.ManifestEntries, "targeted_readmes": plan.TargetedREADMERequests, "path": manifestPath})
	fmt.Printf("Sampling plan: base_candidates=%d covered=%d missing=%d manifest=%d targeted_readmes=%d\n", plan.Candidates, plan.CoveredCandidates, plan.MissingCandidates, plan.ManifestEntries, plan.TargetedREADMERequests)
	if !execute {
		fmt.Printf("Plan only. Review %s and sampling-plan-summary.json; add --execute to fetch only the planned READMEs and call the local LLM.\n", manifestPath)
		return nil
	}
	if err := preflightSamplingLocalLLM(ctx, config); err != nil {
		return err
	}

	client := &httpClient{client: &http.Client{Timeout: time.Duration(config.Scan.TimeoutSeconds) * time.Second}, token: os.Getenv("HF_TOKEN"), retries: *config.Scan.Retries}
	client.log = func(level, event, message string, fields map[string]any) {
		if level != "debug" || config.Logging.HTTPRequests {
			logger.log(level, event, message, fields)
		}
	}
	cards, err := loadSamplingCards(ctx, bulkPaths, manifest, logger)
	if err != nil {
		return err
	}
	if err := fetchTargetedSamplingCards(ctx, client, config, manifest, cards, logger); err != nil {
		return err
	}
	reviews, err := reviewSamplingCards(ctx, config, manifest, checkpoint.Models, cards, logger)
	if err != nil {
		return err
	}
	results := buildSamplingResults(config, selections, manifest, cards, reviews)
	deduplicateSamplingResults(results)
	baseline, baselineAvailable := loadSamplingBaseline(config)
	summary := aggregateSamplingResults(plan, results, baseline, baselineAvailable)
	resultsPath := resolveOutputPath(config.OutputDir, config.Sampling.ResultsFile)
	if err := writeJSONFile(resultsPath, results); err != nil {
		return err
	}
	if err := writeSamplingResultsCSV(strings.TrimSuffix(resultsPath, filepath.Ext(resultsPath))+".csv", results); err != nil {
		return err
	}
	if err := writeJSONFile(resolveOutputPath(config.OutputDir, config.Sampling.SummaryFile), summary); err != nil {
		return err
	}
	if err := writeSamplingReport(resolveOutputPath(config.OutputDir, config.Sampling.ReportFile), summary); err != nil {
		return err
	}
	logger.info("sampling_completed", "stratified market estimate completed", map[string]any{"sample": len(results), "baseline_available": baselineAvailable})
	fmt.Printf("Sampling complete. Report: %s\n", resolveOutputPath(config.OutputDir, config.Sampling.ReportFile))
	return nil
}

func resolveOutputPath(outputDir, path string) string {
	if filepath.IsAbs(path) {
		return path
	}
	return filepath.Join(outputDir, path)
}

func resolvedBulkShardPaths(config catalogConfig) ([]string, error) {
	base := config.Scan.BulkModelCardsFile
	if base == "" {
		return nil, errors.New("scan.bulk_model_cards_file is required")
	}
	if !filepath.IsAbs(base) {
		base = filepath.Join(config.OutputDir, base)
	}
	urls := config.Scan.BulkModelCardsURLs
	if len(urls) == 0 && config.Scan.BulkModelCardsURL != "" {
		urls = []string{config.Scan.BulkModelCardsURL}
	}
	if len(urls) == 0 {
		return nil, errors.New("bulk model-card shard URLs are required to determine shard count")
	}
	paths := make([]string, len(urls))
	for i := range urls {
		paths[i] = bulkShardPath(base, i, len(urls))
		info, err := os.Stat(paths[i])
		if err != nil || info.Size() == 0 {
			return nil, fmt.Errorf("cached Parquet shard is required at %s", paths[i])
		}
	}
	return paths, nil
}

func buildSamplingCandidates(models []catalogModel, selections []compiledSelection) []samplingCandidate {
	var result []samplingCandidate
	for _, selection := range selections {
		scope := "text_llm"
		if selection.config.TargetDiffusionOnly {
			scope = "diffusion"
		}
		for _, model := range models {
			params := model.Safetensors.Total
			if !matchesSelection(model, selection, params) || catalogModelKind(model) != "base" {
				continue
			}
			if selection.config.TargetLLMOnly && isDiffusionModel(model) {
				continue
			}
			result = append(result, samplingCandidate{Model: model, Selection: selection, Scope: scope, Params: params, Bucket: parameterRange(params)})
		}
	}
	sort.Slice(result, func(i, j int) bool {
		if result[i].Selection.config.Name != result[j].Selection.config.Name {
			return result[i].Selection.config.Name < result[j].Selection.config.Name
		}
		return result[i].Model.ID < result[j].Model.ID
	})
	return result
}

func scanSamplingCoverage(ctx context.Context, paths []string, candidates []samplingCandidate, logger *catalogLogger) (map[string]bool, error) {
	wanted := make(map[string]bool, len(candidates))
	for _, candidate := range candidates {
		wanted[candidate.Model.ID] = true
	}
	covered := make(map[string]bool)
	rowsScanned := 0
	for _, path := range paths {
		file, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		reader := parquet.NewGenericReader[modelCardSnapshotRow](file)
		rows := make([]modelCardSnapshotRow, 1024)
		for {
			if err := ctx.Err(); err != nil {
				reader.Close()
				file.Close()
				return nil, err
			}
			n, readErr := reader.Read(rows)
			for _, row := range rows[:n] {
				rowsScanned++
				if wanted[row.ModelID] && strings.TrimSpace(row.Card) != "" {
					covered[row.ModelID] = true
				}
			}
			if errors.Is(readErr, io.EOF) {
				break
			}
			if readErr != nil {
				reader.Close()
				file.Close()
				return nil, readErr
			}
		}
		reader.Close()
		file.Close()
	}
	logger.info("sampling_parquet_coverage", "measured exact base-candidate coverage in bulk model cards", map[string]any{"rows": rowsScanned, "candidates": len(candidates), "covered": len(covered), "missing": len(candidates) - len(covered)})
	return covered, nil
}

func samplingHash(seed int64, repoID string) string {
	sum := sha256.Sum256([]byte(strconv.FormatInt(seed, 10) + "\x00" + repoID))
	return hex.EncodeToString(sum[:])
}

func samplingStratum(candidate samplingCandidate, census bool) string {
	source := "missing"
	if candidate.Covered {
		source = "parquet"
	}
	group := candidate.Bucket
	if census {
		group = "census>=threshold"
	}
	return candidate.Selection.config.Name + "|" + group + "|" + source
}

func buildSamplingManifest(candidates []samplingCandidate, config marketSamplingConfig) ([]samplingManifestEntry, samplingPlanSummary, error) {
	groups := make(map[string][]samplingCandidate)
	var census []samplingCandidate
	population := make(map[string]int)
	plan := samplingPlanSummary{
		GeneratedAt: time.Now().UTC().Format(time.RFC3339Nano), Seed: config.Seed,
		Candidates: len(candidates), CandidatesBySelection: make(map[string]int), ManifestBySelection: make(map[string]int),
		PopulationByStratum: make(map[string]int), SampleByStratum: make(map[string]int),
	}
	includeCovered := boolValue(config.IncludeCoveredCensus, true)
	includeMissing := boolValue(config.IncludeMissingCensus, true)
	for _, candidate := range candidates {
		plan.CandidatesBySelection[candidate.Selection.config.Name]++
		if candidate.Covered {
			plan.CoveredCandidates++
		} else {
			plan.MissingCandidates++
		}
		isCensus := candidate.Params > 0 && float64(candidate.Params)/1e9 >= config.CensusMinParametersB && ((candidate.Covered && includeCovered) || (!candidate.Covered && includeMissing))
		key := samplingStratum(candidate, isCensus)
		population[key]++
		if isCensus {
			census = append(census, candidate)
		} else {
			groups[key] = append(groups[key], candidate)
		}
	}
	selected := append([]samplingCandidate(nil), census...)
	keys := make([]string, 0, len(groups))
	for key := range groups {
		keys = append(keys, key)
		sort.Slice(groups[key], func(i, j int) bool {
			return samplingHash(config.Seed, groups[key][i].Model.ID) < samplingHash(config.Seed, groups[key][j].Model.ID)
		})
	}
	sort.Strings(keys)
	for _, key := range keys {
		count := min(config.SamplePerStratum, len(groups[key]))
		selected = append(selected, groups[key][:count]...)
	}
	censusMissing := 0
	for _, candidate := range census {
		if !candidate.Covered {
			censusMissing++
		}
	}
	if censusMissing > config.MaxTargetedReadmes {
		return nil, plan, fmt.Errorf("large-model census alone needs %d targeted READMEs, above sampling.max_targeted_readmes=%d", censusMissing, config.MaxTargetedReadmes)
	}
	missingSelected := 0
	for _, candidate := range selected {
		if !candidate.Covered {
			missingSelected++
		}
	}
	if missingSelected > config.MaxTargetedReadmes {
		capacity := config.MaxTargetedReadmes - censusMissing
		var coveredSelected, missingNonCensus []samplingCandidate
		for _, candidate := range selected {
			isCensus := candidate.Params > 0 && float64(candidate.Params)/1e9 >= config.CensusMinParametersB
			if candidate.Covered || isCensus {
				coveredSelected = append(coveredSelected, candidate)
			} else {
				missingNonCensus = append(missingNonCensus, candidate)
			}
		}
		sort.Slice(missingNonCensus, func(i, j int) bool {
			return samplingHash(config.Seed+1, missingNonCensus[i].Model.ID) < samplingHash(config.Seed+1, missingNonCensus[j].Model.ID)
		})
		selected = append(coveredSelected, missingNonCensus[:min(capacity, len(missingNonCensus))]...)
	}
	sampleCount := make(map[string]int)
	for _, candidate := range selected {
		isCensus := candidate.Params > 0 && float64(candidate.Params)/1e9 >= config.CensusMinParametersB && ((candidate.Covered && includeCovered) || (!candidate.Covered && includeMissing))
		sampleCount[samplingStratum(candidate, isCensus)]++
	}
	manifest := make([]samplingManifestEntry, 0, len(selected))
	for _, candidate := range selected {
		isCensus := candidate.Params > 0 && float64(candidate.Params)/1e9 >= config.CensusMinParametersB && ((candidate.Covered && includeCovered) || (!candidate.Covered && includeMissing))
		stratum := samplingStratum(candidate, isCensus)
		weight := float64(population[stratum]) / float64(sampleCount[stratum])
		if isCensus {
			weight = 1
			plan.CensusEntries++
		}
		source := "targeted_readme"
		if candidate.Covered {
			source = "parquet"
			plan.CoveredManifestEntries++
		} else {
			plan.TargetedREADMERequests++
		}
		entry := samplingManifestEntry{
			Selection: candidate.Selection.config.Name, Scope: candidate.Scope, RepoID: candidate.Model.ID,
			Owner: modelOwner(candidate.Model), CreatedAt: candidate.Model.CreatedAt, Parameters: candidate.Params,
			ParameterRange: candidate.Bucket, Downloads: candidate.Model.Downloads, Likes: candidate.Model.Likes,
			PipelineTag: candidate.Model.PipelineTag, LibraryName: candidate.Model.LibraryName, Tags: candidate.Model.Tags,
			CardSource: source, Census: isCensus, Stratum: stratum, StratumPopulation: population[stratum], StratumSample: sampleCount[stratum],
			InclusionProbability: 1 / weight, SamplingWeight: weight,
		}
		manifest = append(manifest, entry)
		plan.ManifestBySelection[entry.Selection]++
	}
	sort.Slice(manifest, func(i, j int) bool {
		if manifest[i].Selection != manifest[j].Selection {
			return manifest[i].Selection < manifest[j].Selection
		}
		return manifest[i].RepoID < manifest[j].RepoID
	})
	plan.ManifestEntries = len(manifest)
	for key, count := range population {
		plan.PopulationByStratum[key] = count
	}
	for key, count := range sampleCount {
		plan.SampleByStratum[key] = count
	}
	return manifest, plan, nil
}

func loadSamplingCards(ctx context.Context, paths []string, manifest []samplingManifestEntry, logger *catalogLogger) (map[string]string, error) {
	wanted := make(map[string]bool)
	for _, entry := range manifest {
		if entry.CardSource == "parquet" {
			wanted[entry.RepoID] = true
		}
	}
	cards := make(map[string]string, len(wanted))
	for _, path := range paths {
		file, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		reader := parquet.NewGenericReader[modelCardSnapshotRow](file)
		rows := make([]modelCardSnapshotRow, 1024)
		for {
			if err := ctx.Err(); err != nil {
				reader.Close()
				file.Close()
				return nil, err
			}
			n, readErr := reader.Read(rows)
			for _, row := range rows[:n] {
				if wanted[row.ModelID] && strings.TrimSpace(row.Card) != "" {
					cards[row.ModelID] = row.Card
				}
			}
			if errors.Is(readErr, io.EOF) {
				break
			}
			if readErr != nil {
				reader.Close()
				file.Close()
				return nil, readErr
			}
		}
		reader.Close()
		file.Close()
	}
	logger.info("sampling_parquet_cards_loaded", "loaded sampled cards from local Parquet", map[string]any{"wanted": len(wanted), "loaded": len(cards)})
	return cards, nil
}

func readmeCachePath(config catalogConfig, repoID string) string {
	directory := resolveOutputPath(config.OutputDir, config.Sampling.ReadmeCacheDir)
	sum := sha256.Sum256([]byte(repoID))
	return filepath.Join(directory, hex.EncodeToString(sum[:])+".md")
}

func fetchTargetedSamplingCards(ctx context.Context, client *httpClient, config catalogConfig, manifest []samplingManifestEntry, cards map[string]string, logger *catalogLogger) error {
	if err := os.MkdirAll(resolveOutputPath(config.OutputDir, config.Sampling.ReadmeCacheDir), 0o755); err != nil {
		return err
	}
	var targets []samplingManifestEntry
	for _, entry := range manifest {
		if entry.CardSource != "targeted_readme" {
			continue
		}
		path := readmeCachePath(config, entry.RepoID)
		if body, err := os.ReadFile(path); err == nil {
			cards[entry.RepoID] = string(body)
			continue
		}
		if _, err := os.Stat(path + ".missing"); err == nil {
			continue
		}
		targets = append(targets, entry)
	}
	jobs := make(chan samplingManifestEntry)
	var wg sync.WaitGroup
	var mutex sync.Mutex
	var firstErr error
	completed := 0
	for range config.Sampling.ReadmeWorkers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for entry := range jobs {
				if config.Sampling.RequestIntervalMS > 0 {
					select {
					case <-ctx.Done():
						return
					case <-time.After(time.Duration(config.Sampling.RequestIntervalMS) * time.Millisecond):
					}
				}
				address := strings.TrimRight(config.Endpoint, "/") + "/" + escapeRepoID(entry.RepoID) + "/resolve/main/README.md"
				body, _, err := client.get(ctx, address, true)
				path := readmeCachePath(config, entry.RepoID)
				if err == nil && len(body) > 0 {
					tmp := path + ".tmp"
					writeErr := os.WriteFile(tmp, body, 0o644)
					if writeErr == nil {
						writeErr = os.Rename(tmp, path)
					}
					if writeErr != nil {
						err = writeErr
					} else {
						mutex.Lock()
						cards[entry.RepoID] = string(body)
						mutex.Unlock()
					}
				} else if err == nil {
					err = os.WriteFile(path+".missing", []byte(entry.RepoID+"\n"), 0o644)
				}
				mutex.Lock()
				completed++
				if err != nil && firstErr == nil {
					firstErr = fmt.Errorf("fetch README for %s: %w", entry.RepoID, err)
				}
				if completed%10 == 0 || completed == len(targets) {
					logger.info("sampling_readme_progress", "targeted README fetch progress", map[string]any{"completed": completed, "scheduled": len(targets), "cached_cards": len(cards)})
				}
				mutex.Unlock()
			}
		}()
	}
	for _, entry := range targets {
		select {
		case <-ctx.Done():
			close(jobs)
			wg.Wait()
			return ctx.Err()
		case jobs <- entry:
		}
	}
	close(jobs)
	wg.Wait()
	if firstErr != nil {
		return firstErr
	}
	logger.info("sampling_readmes_completed", "targeted README fetch completed", map[string]any{"scheduled": len(targets), "cards_available": len(cards)})
	return nil
}

func reviewSamplingCards(ctx context.Context, config catalogConfig, manifest []samplingManifestEntry, models []catalogModel, cards map[string]string, logger *catalogLogger) (map[string]*localLLMReview, error) {
	llmClient := &http.Client{Timeout: time.Duration(config.LocalLLM.TimeoutSeconds) * time.Second}
	cachePath := config.LocalLLM.CacheFile
	if !filepath.IsAbs(cachePath) {
		cachePath = filepath.Join(config.OutputDir, cachePath)
	}
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
	modelsByID := make(map[string]catalogModel, len(models))
	for _, model := range models {
		modelsByID[model.ID] = model
	}
	result := make(map[string]*localLLMReview, len(manifest))
	var resultMutex, cacheMutex sync.Mutex
	jobs := make(chan samplingManifestEntry)
	var wg sync.WaitGroup
	completed, failed := 0, 0
	for range config.LocalLLM.Workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for entry := range jobs {
				card := cards[entry.RepoID]
				if strings.TrimSpace(card) == "" {
					review := &localLLMReview{Kind: "unknown", Confidence: "low", Reason: "sampled model card is unavailable", Source: "sampling prefilter"}
					resultMutex.Lock()
					result[entry.RepoID] = review
					completed++
					resultMutex.Unlock()
					continue
				}
				model := modelsByID[entry.RepoID]
				normalized := strings.Join(strings.Fields(card), " ")
				cardHash := fmt.Sprintf("%x", sha256.Sum256([]byte(normalized)))
				key := localLLMCacheKey(model, entry.Parameters, cardHash, entry.Scope)
				if cached, ok := cache[key]; ok {
					copy := cached
					resultMutex.Lock()
					result[entry.RepoID] = &copy
					completed++
					resultMutex.Unlock()
					continue
				}
				baseModels := make([]string, 0, len(model.BaseModels.Models))
				for _, base := range model.BaseModels.Models {
					baseModels = append(baseModels, base.ID)
				}
				input := localLLMReviewInput{RepoID: entry.RepoID, PipelineTag: entry.PipelineTag, LibraryName: entry.LibraryName, Tags: entry.Tags, BaseRelation: model.BaseModels.Relation, BaseModels: baseModels, Parameters: entry.Parameters, Card: compactModelCard(card, config.LocalLLM.MaxInputChars), CardHash: cardHash, CacheKey: key}
				callConfig := config.LocalLLM
				callConfig.Scope = entry.Scope
				review, callErr := callLocalLLM(ctx, llmClient, callConfig, input)
				if callErr != nil {
					review = localLLMReview{Kind: "unknown", Confidence: "low", Reason: callErr.Error(), Source: "local OpenAI-compatible LLM " + config.LocalLLM.Model, CardHash: cardHash, CacheKey: key}
					resultMutex.Lock()
					failed++
					resultMutex.Unlock()
				} else if err := appendLocalLLMCache(cacheFile, &cacheMutex, key, review); err != nil {
					logger.warn("sampling_llm_cache_failed", "could not append sampling LLM cache", map[string]any{"repo_id": entry.RepoID, "error": err.Error()})
				}
				resultMutex.Lock()
				copy := review
				result[entry.RepoID] = &copy
				completed++
				if completed%100 == 0 || completed == len(manifest) {
					logger.info("sampling_llm_progress", "sampling local-LLM progress", map[string]any{"completed": completed, "sample": len(manifest), "failed": failed})
				}
				resultMutex.Unlock()
			}
		}()
	}
	for _, entry := range manifest {
		select {
		case <-ctx.Done():
			close(jobs)
			wg.Wait()
			return nil, ctx.Err()
		case jobs <- entry:
		}
	}
	close(jobs)
	wg.Wait()
	if failed == len(manifest) && failed > 0 {
		return nil, fmt.Errorf("all %d sampling LLM calls failed", failed)
	}
	return result, nil
}

func preflightSamplingLocalLLM(ctx context.Context, config catalogConfig) error {
	client := &http.Client{Timeout: time.Duration(config.LocalLLM.TimeoutSeconds) * time.Second}
	probeConfig := config.LocalLLM
	probeConfig.Scope = "text_llm"
	if err := probeLocalLLM(ctx, client, probeConfig); err != nil {
		return err
	}
	probeInput := localLLMReviewInput{RepoID: "preflight/synthetic-1b", Parameters: 1_000_000_000, Card: "The current model was pretrained from scratch.", CardHash: "preflight", CacheKey: "preflight"}
	if _, err := callLocalLLM(ctx, client, probeConfig, probeInput); err != nil {
		return fmt.Errorf("local LLM completion preflight failed: %w", err)
	}
	return nil
}

func buildSamplingResults(config catalogConfig, selections []compiledSelection, manifest []samplingManifestEntry, cards map[string]string, reviews map[string]*localLLMReview) []samplingResult {
	selectionByName := make(map[string]compiledSelection)
	profileByName := make(map[string]computeProfile)
	for _, selection := range selections {
		selectionByName[selection.config.Name] = selection
	}
	for _, profile := range config.Compute {
		profileByName[profile.Name] = profile
	}
	results := make([]samplingResult, 0, len(manifest))
	for _, entry := range manifest {
		result := samplingResult{Selection: entry.Selection, Scope: entry.Scope, RepoID: entry.RepoID, Owner: entry.Owner, Parameters: entry.Parameters, ParameterSource: "hub_safetensors", ParameterRange: entry.ParameterRange, CardSource: entry.CardSource, CardAvailable: strings.TrimSpace(cards[entry.RepoID]) != "", Census: entry.Census, Stratum: entry.Stratum, StratumPopulation: entry.StratumPopulation, StratumSample: entry.StratumSample, SamplingWeight: entry.SamplingWeight, Review: reviews[entry.RepoID]}
		if result.Parameters <= 0 {
			result.ParameterSource = "unknown"
			if result.Review != nil && result.Review.ReportedParametersB > 0 {
				result.Parameters = int64(result.Review.ReportedParametersB * 1e9)
				result.ParameterRange = parameterRange(result.Parameters)
				result.ParameterSource = "local_llm_verbatim_model_card"
			}
		}
		if !result.CardAvailable {
			result.Error = "model card unavailable after targeted fetch"
			results = append(results, result)
			continue
		}
		card := cards[entry.RepoID]
		deterministicClaim := scratchClaimFromText(card, entry.CardSource, entry.RepoID)
		if deterministicClaim != nil {
			result.DeterministicEvidence = deterministicClaim.EvidenceType
		}
		claim, deterministic := resolveScratchClaimForReview(true, deterministicClaim, result.Review)
		selection := selectionByName[entry.Selection]
		if claim != nil && (!scratchClaimMatchesOwner(catalogModel{ID: entry.RepoID, Author: entry.Owner}, claim) || !scratchClaimFallsInSelection(claim, selection)) {
			claim = nil
		}
		strongDeterministic := deterministic && claim != nil && claim.EvidenceType != "declared_base_model"
		highIndependent := result.Review != nil && result.Review.Kind == "independent_base" && result.Review.IndependentlyPretrained && result.Review.Confidence == "high" && claim != nil
		mediumIndependent := result.Review != nil && result.Review.Kind == "independent_base" && result.Review.IndependentlyPretrained && result.Review.Confidence == "medium" && claim != nil
		result.IndependentLower = strongDeterministic
		result.IndependentCentral = strongDeterministic || highIndependent
		result.IndependentUpper = result.IndependentCentral || mediumIndependent
		if result.Parameters <= 0 {
			if result.IndependentUpper {
				result.Error = "independent base candidate has no parameter count; excluded from cost estimate"
			}
			results = append(results, result)
			continue
		}
		if len(selection.config.ComputeProfiles) == 0 {
			result.Error = "selection has no compute profile"
			results = append(results, result)
			continue
		}
		profile, ok := profileByName[selection.config.ComputeProfiles[0]]
		if !ok {
			result.Error = "compute profile not found: " + selection.config.ComputeProfiles[0]
			results = append(results, result)
			continue
		}
		var estimate computeEstimate
		if entry.Scope == "diffusion" {
			estimate = estimateDiffusionBaseTrainingCompute(result.Parameters, claim, profile)
		} else {
			estimate = estimateBaseTrainingCompute(result.Parameters, claim, profile)
		}
		result.RawCostUSD = estimate.CostUSD
		result.CostMethod = estimate.Method
		if result.IndependentLower {
			result.WeightedLowerCostUSD = result.RawCostUSD * result.SamplingWeight
		}
		if result.IndependentCentral {
			result.WeightedCentralCostUSD = result.RawCostUSD * result.SamplingWeight
		}
		if result.IndependentUpper {
			result.WeightedUpperCostUSD = result.RawCostUSD * result.SamplingWeight
		}
		results = append(results, result)
	}
	return results
}

func deduplicateSamplingResults(results []samplingResult) {
	groups := make(map[string][]int)
	for i := range results {
		if !results[i].IndependentUpper || results[i].Review == nil || results[i].Review.CanonicalTrainingRun == "" || results[i].Parameters <= 0 {
			continue
		}
		key := results[i].Selection + "\x00" + normalizeCanonicalRun(results[i].Review.CanonicalTrainingRun) + "\x00" + strconv.FormatInt(results[i].Parameters, 10)
		groups[key] = append(groups[key], i)
	}
	for _, indices := range groups {
		if len(indices) < 2 {
			continue
		}
		canonical := indices[0]
		for _, index := range indices[1:] {
			if results[index].Census && !results[canonical].Census || results[index].SamplingWeight < results[canonical].SamplingWeight {
				canonical = index
			}
		}
		for _, index := range indices {
			if index == canonical {
				continue
			}
			results[index].DuplicateOf = results[canonical].RepoID
			results[index].IndependentLower = false
			results[index].IndependentCentral = false
			results[index].IndependentUpper = false
			results[index].WeightedLowerCostUSD = 0
			results[index].WeightedCentralCostUSD = 0
			results[index].WeightedUpperCostUSD = 0
		}
	}
}

func loadSamplingBaseline(config catalogConfig) (catalogSummary, bool) {
	if config.Sampling.BaselineSummaryFile == "" {
		return catalogSummary{}, false
	}
	path := resolveOutputPath(config.OutputDir, config.Sampling.BaselineSummaryFile)
	data, err := os.ReadFile(path)
	if err != nil {
		return catalogSummary{}, false
	}
	var summary catalogSummary
	if json.Unmarshal(data, &summary) != nil {
		return catalogSummary{}, false
	}
	return summary, true
}

func aggregateSamplingResults(plan samplingPlanSummary, results []samplingResult, baseline catalogSummary, baselineAvailable bool) marketSamplingSummary {
	summary := marketSamplingSummary{GeneratedAt: time.Now().UTC().Format(time.RFC3339Nano), Method: "stratified base-candidate sample plus complete >= configured threshold census; Horvitz-Thompson weights by selection, parameter range, and Parquet coverage", Plan: plan, BaselineAvailable: baselineAvailable, Selections: make(map[string]samplingSelectionSummary)}
	for selection, count := range plan.CandidatesBySelection {
		value := summary.Selections[selection]
		value.Candidates = count
		value.Strata = make(map[string]samplingStratumSummary)
		summary.Selections[selection] = value
	}
	for _, result := range results {
		selection := summary.Selections[result.Selection]
		selection.Sample++
		if result.CardAvailable {
			selection.CardsAvailable++
		}
		if result.Parameters <= 0 {
			selection.UncostableUnknownParameters++
		}
		selection.EstimatedBaseLowerUSD += result.WeightedLowerCostUSD
		selection.EstimatedBaseCentralUSD += result.WeightedCentralCostUSD
		selection.EstimatedBaseUpperUSD += result.WeightedUpperCostUSD
		stratum := selection.Strata[result.Stratum]
		stratum.Population = result.StratumPopulation
		stratum.Sample++
		if result.CardAvailable {
			stratum.CardsAvailable++
		}
		if result.IndependentLower {
			stratum.IndependentLowerWeighted += result.SamplingWeight
		}
		if result.IndependentCentral {
			stratum.IndependentCentralWeighted += result.SamplingWeight
		}
		if result.IndependentUpper {
			stratum.IndependentUpperWeighted += result.SamplingWeight
		}
		stratum.LowerCostUSD += result.WeightedLowerCostUSD
		stratum.CentralCostUSD += result.WeightedCentralCostUSD
		stratum.UpperCostUSD += result.WeightedUpperCostUSD
		selection.Strata[result.Stratum] = stratum
		summary.Selections[result.Selection] = selection
	}
	for name, selection := range summary.Selections {
		if baselineAvailable {
			if base, ok := baseline.Selections[name]; ok {
				selection.ConfirmedBaselineUSD = base.TotalTrainingCostUSD
				selection.ReportedDerivativeUSD = base.TrainingCostByKind["finetune"] + base.TrainingCostByKind["adapter"]
			}
		}
		selection.MarketLowerUSD = selection.ConfirmedBaselineUSD
		selection.MarketCentralUSD = selection.EstimatedBaseCentralUSD + selection.ReportedDerivativeUSD
		selection.MarketUpperUSD = selection.EstimatedBaseUpperUSD + selection.ReportedDerivativeUSD
		if selection.MarketCentralUSD < selection.MarketLowerUSD {
			selection.MarketCentralUSD = selection.MarketLowerUSD
		}
		if selection.MarketUpperUSD < selection.MarketCentralUSD {
			selection.MarketUpperUSD = selection.MarketCentralUSD
		}
		summary.Selections[name] = selection
	}
	if !baselineAvailable {
		summary.Warnings = append(summary.Warnings, "baseline summary is unavailable: market_lower_usd is zero and reported derivative costs are not added")
	}
	summary.Warnings = append(summary.Warnings,
		"central and upper estimates cover base training plus only publicly reported derivative compute",
		"sampled candidates with unknown parameter counts cannot be priced and are reported separately",
		"scenario bounds reflect evidence confidence; they are not a formal confidence interval and systemic model-card bias remains",
	)
	return summary
}

func writeSamplingManifestCSV(path string, entries []samplingManifestEntry) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	w := csv.NewWriter(file)
	_ = w.Write([]string{"selection", "scope", "repo_id", "owner", "parameters", "parameter_range", "card_source", "census", "stratum", "stratum_population", "stratum_sample", "inclusion_probability", "sampling_weight"})
	for _, entry := range entries {
		_ = w.Write([]string{entry.Selection, entry.Scope, entry.RepoID, entry.Owner, strconv.FormatInt(entry.Parameters, 10), entry.ParameterRange, entry.CardSource, strconv.FormatBool(entry.Census), entry.Stratum, strconv.Itoa(entry.StratumPopulation), strconv.Itoa(entry.StratumSample), strconv.FormatFloat(entry.InclusionProbability, 'g', -1, 64), strconv.FormatFloat(entry.SamplingWeight, 'g', -1, 64)})
	}
	w.Flush()
	if err := w.Error(); err != nil {
		file.Close()
		return err
	}
	return file.Close()
}

func writeSamplingResultsCSV(path string, results []samplingResult) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	w := csv.NewWriter(file)
	_ = w.Write([]string{"selection", "repo_id", "owner", "parameters", "parameter_source", "parameter_range", "card_source", "card_available", "census", "stratum", "sampling_weight", "llm_kind", "llm_confidence", "independent_lower", "independent_central", "independent_upper", "duplicate_of", "raw_cost_usd", "weighted_lower_usd", "weighted_central_usd", "weighted_upper_usd", "error"})
	for _, result := range results {
		kind, confidence := "", ""
		if result.Review != nil {
			kind, confidence = result.Review.Kind, result.Review.Confidence
		}
		_ = w.Write([]string{result.Selection, result.RepoID, result.Owner, strconv.FormatInt(result.Parameters, 10), result.ParameterSource, result.ParameterRange, result.CardSource, strconv.FormatBool(result.CardAvailable), strconv.FormatBool(result.Census), result.Stratum, strconv.FormatFloat(result.SamplingWeight, 'g', -1, 64), kind, confidence, strconv.FormatBool(result.IndependentLower), strconv.FormatBool(result.IndependentCentral), strconv.FormatBool(result.IndependentUpper), result.DuplicateOf, strconv.FormatFloat(result.RawCostUSD, 'f', 6, 64), strconv.FormatFloat(result.WeightedLowerCostUSD, 'f', 6, 64), strconv.FormatFloat(result.WeightedCentralCostUSD, 'f', 6, 64), strconv.FormatFloat(result.WeightedUpperCostUSD, 'f', 6, 64), result.Error})
	}
	w.Flush()
	if err := w.Error(); err != nil {
		file.Close()
		return err
	}
	return file.Close()
}

func writeSamplingReport(path string, summary marketSamplingSummary) error {
	var report strings.Builder
	report.WriteString("# Стратифицированная оценка рынка обучения моделей\n\n")
	fmt.Fprintf(&report, "Сформировано: %s\n\n", summary.GeneratedAt)
	report.WriteString("Метод: полный census дорогих base-кандидатов и детерминированная стратифицированная выборка остальных; веса рассчитаны отдельно по рынку, диапазону параметров и наличию карточки в Parquet.\n\n")
	fmt.Fprintf(&report, "- Base-кандидатов: %d\n- Parquet-covered: %d\n- Parquet-missing: %d\n- Проверено в manifest: %d\n- Точечных README-запросов: %d\n\n", summary.Plan.Candidates, summary.Plan.CoveredCandidates, summary.Plan.MissingCandidates, summary.Plan.ManifestEntries, summary.Plan.TargetedREADMERequests)
	names := make([]string, 0, len(summary.Selections))
	for name := range summary.Selections {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		selection := summary.Selections[name]
		fmt.Fprintf(&report, "## %s\n\n", name)
		fmt.Fprintf(&report, "- Base-кандидатов: %d; sample/census: %d; карточек доступно: %d\n", selection.Candidates, selection.Sample, selection.CardsAvailable)
		fmt.Fprintf(&report, "- Подтверждённый baseline всего: **$%.2f**\n", selection.ConfirmedBaselineUSD)
		fmt.Fprintf(&report, "- Стратифицированная base-оценка: lower **$%.2f**, central **$%.2f**, upper **$%.2f**\n", selection.EstimatedBaseLowerUSD, selection.EstimatedBaseCentralUSD, selection.EstimatedBaseUpperUSD)
		fmt.Fprintf(&report, "- Опубликованный derivative compute: **$%.2f**\n", selection.ReportedDerivativeUSD)
		fmt.Fprintf(&report, "- Рыночный диапазон: confirmed lower **$%.2f**, central **$%.2f**, upper **$%.2f**\n", selection.MarketLowerUSD, selection.MarketCentralUSD, selection.MarketUpperUSD)
		fmt.Fprintf(&report, "- Sample с неизвестными параметрами, которые нельзя оценить: %d\n\n", selection.UncostableUnknownParameters)
		report.WriteString("| Страта | Population | Sample | Cards | Central independent (weighted) | Central cost |\n|---|---:|---:|---:|---:|---:|\n")
		strata := make([]string, 0, len(selection.Strata))
		for stratum := range selection.Strata {
			strata = append(strata, stratum)
		}
		sort.Strings(strata)
		for _, stratum := range strata {
			item := selection.Strata[stratum]
			fmt.Fprintf(&report, "| %s | %d | %d | %d | %.2f | $%.2f |\n", stratum, item.Population, item.Sample, item.CardsAvailable, item.IndependentCentralWeighted, item.CentralCostUSD)
		}
		report.WriteString("\n")
	}
	if len(summary.Warnings) > 0 {
		report.WriteString("## Ограничения\n\n")
		for _, warning := range summary.Warnings {
			report.WriteString("- " + warning + "\n")
		}
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, []byte(report.String()), 0o644)
}

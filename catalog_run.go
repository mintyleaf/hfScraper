package main

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

func boolValue(value *bool, fallback bool) bool {
	if value == nil {
		return fallback
	}
	return *value
}

func runCatalogCLI(arguments []string) error {
	flags := flag.NewFlagSet("catalog", flag.ContinueOnError)
	configPath := flags.String("config", "catalog.example.json", "path to JSON catalog configuration")
	outputOverride := flags.String("output", "", "override output_dir from config")
	if err := flags.Parse(arguments); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	data, err := os.ReadFile(*configPath)
	if err != nil {
		return err
	}
	var config catalogConfig
	decoder := json.NewDecoder(strings.NewReader(string(data)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&config); err != nil {
		return fmt.Errorf("decode config: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("decode config: more than one JSON value")
		}
		return fmt.Errorf("decode trailing config data: %w", err)
	}
	if *outputOverride != "" {
		config.OutputDir = *outputOverride
	}
	applyCatalogDefaults(&config)
	return runCatalog(context.Background(), config)
}

func fetchBaseParameters(ctx context.Context, client *httpClient, endpoint string, ids []string, workers int, logger *catalogLogger) (map[string]int64, map[string]string) {
	result := make(map[string]int64)
	errorsByID := make(map[string]string)
	jobs := make(chan string)
	type response struct {
		id    string
		total int64
		err   error
	}
	responses := make(chan response)
	var wg sync.WaitGroup
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for id := range jobs {
				query := url.Values{}
				query.Add("expand", "safetensors")
				address := strings.TrimRight(endpoint, "/") + "/api/models/" + escapeRepoID(id) + "?" + query.Encode()
				body, _, err := client.get(ctx, address, true)
				if err != nil {
					responses <- response{id: id, err: fmt.Errorf("fetch base model metadata: %w", err)}
					continue
				}
				if len(body) == 0 {
					responses <- response{id: id, err: errors.New("base model metadata is unavailable")}
					continue
				}
				var info struct {
					Safetensors catalogSafetensors `json:"safetensors"`
				}
				if err := json.Unmarshal(body, &info); err != nil {
					responses <- response{id: id, err: fmt.Errorf("decode base model metadata: %w", err)}
					continue
				}
				if info.Safetensors.Total <= 0 {
					responses <- response{id: id, err: errors.New("base model has no safetensors parameter count")}
					continue
				}
				responses <- response{id: id, total: info.Safetensors.Total}
			}
		}()
	}
	go func() {
		for _, id := range ids {
			jobs <- id
		}
		close(jobs)
		wg.Wait()
		close(responses)
	}()
	for response := range responses {
		result[response.id] = response.total
		if response.err != nil {
			errorsByID[response.id] = response.err.Error()
			logger.warn("base_parameters_failed", "could not resolve base model parameters", map[string]any{"base_model": response.id, "error": response.err.Error()})
		}
	}
	return result, errorsByID
}

func escapeRepoID(id string) string {
	parts := strings.Split(id, "/")
	for i := range parts {
		parts[i] = url.PathEscape(parts[i])
	}
	return strings.Join(parts, "/")
}

func fetchOwners(ctx context.Context, client *httpClient, endpoint string, names []string, workers int, logger *catalogLogger) map[string]*ownerOverview {
	result := make(map[string]*ownerOverview)
	jobs := make(chan string)
	type response struct {
		name     string
		overview *ownerOverview
	}
	responses := make(chan response)
	var wg sync.WaitGroup
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for name := range jobs {
				overview := &ownerOverview{Name: name, Type: "unknown"}
				var lookupErrors []string
				orgURL := strings.TrimRight(endpoint, "/") + "/api/organizations/" + url.PathEscape(name) + "/overview"
				body, _, err := client.get(ctx, orgURL, true)
				if err != nil {
					lookupErrors = append(lookupErrors, "organization lookup: "+err.Error())
				} else if len(body) > 0 {
					if decodeErr := json.Unmarshal(body, overview); decodeErr == nil {
						overview.Type = "organization"
						responses <- response{name, overview}
						continue
					} else {
						lookupErrors = append(lookupErrors, "decode organization profile: "+decodeErr.Error())
					}
				}
				overview = &ownerOverview{Name: name, Type: "unknown"}
				userURL := strings.TrimRight(endpoint, "/") + "/api/users/" + url.PathEscape(name) + "/overview"
				body, _, err = client.get(ctx, userURL, true)
				if err != nil {
					lookupErrors = append(lookupErrors, "user lookup: "+err.Error())
				} else if len(body) > 0 {
					if decodeErr := json.Unmarshal(body, overview); decodeErr == nil {
						overview.Type = "user"
						if overview.Name == "" {
							overview.Name = name
						}
						responses <- response{name, overview}
						continue
					} else {
						lookupErrors = append(lookupErrors, "decode user profile: "+decodeErr.Error())
					}
				}
				if len(lookupErrors) > 0 {
					overview.Error = strings.Join(lookupErrors, "; ")
				} else {
					overview.Error = "profile not found"
				}
				logger.warn("owner_profile_failed", "could not enrich repository owner", map[string]any{"owner": name, "error": overview.Error})
				responses <- response{name, overview}
			}
		}()
	}
	go func() {
		for _, name := range names {
			jobs <- name
		}
		close(jobs)
		wg.Wait()
		close(responses)
	}()
	for response := range responses {
		result[response.name] = response.overview
	}
	return result
}

func runCatalog(ctx context.Context, config catalogConfig) (resultErr error) {
	applyCatalogDefaults(&config)
	if err := os.MkdirAll(config.OutputDir, 0o755); err != nil {
		return fmt.Errorf("create output directory %q: %w", config.OutputDir, err)
	}
	logger, err := newCatalogLogger(config.Logging, config.OutputDir)
	if err != nil {
		return err
	}
	defer func() {
		if recovered := recover(); recovered != nil {
			resultErr = fmt.Errorf("catalog panic: %v", recovered)
			logger.error("panic", "unexpected panic was recovered", resultErr, map[string]any{"stack": string(debug.Stack())})
		}
		if resultErr != nil {
			logger.error("run_failed", "catalog run failed", resultErr, nil)
		}
		if closeErr := logger.close(); closeErr != nil {
			resultErr = errors.Join(resultErr, closeErr)
		}
	}()
	logger.info("run_started", "catalog run started", map[string]any{"output_dir": config.OutputDir, "endpoint": config.Endpoint, "selections": len(config.Selections)})
	endpointURL, err := url.Parse(config.Endpoint)
	if err != nil || endpointURL.Host == "" || (endpointURL.Scheme != "http" && endpointURL.Scheme != "https") {
		return fmt.Errorf("endpoint must be an absolute http(s) URL, got %q", config.Endpoint)
	}

	if config.Scan.PageSize < 1 || config.Scan.PageSize > 1000 {
		return fmt.Errorf("scan.page_size must be between 1 and 1000, got %d", config.Scan.PageSize)
	}
	if config.Scan.MaxPages < 0 {
		return fmt.Errorf("scan.max_pages cannot be negative, got %d", config.Scan.MaxPages)
	}
	if config.Scan.TimeoutSeconds <= 0 {
		return fmt.Errorf("scan.timeout_seconds must be positive, got %d", config.Scan.TimeoutSeconds)
	}
	if config.Owners.Workers < 1 || config.Owners.Workers > 64 {
		return fmt.Errorf("owners.workers must be between 1 and 64, got %d", config.Owners.Workers)
	}
	if config.Scan.Retries == nil || *config.Scan.Retries < 0 || *config.Scan.Retries > 20 {
		return errors.New("scan.retries must be between 0 and 20")
	}
	now := time.Now().UTC()
	if config.Scan.Now != "" {
		parsed, err := parseFlexibleTime(config.Scan.Now, false)
		if err != nil {
			return err
		}
		now = parsed
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
	for _, selection := range selections {
		if (selection.config.RequireReportedDerivativeCompute || selection.config.RequireReportedCompute) && !boolValue(config.Scan.ResolveReportedCompute, false) {
			return fmt.Errorf("selection %q requires reported compute but scan.resolve_reported_compute is false", selection.config.Name)
		}
		if selection.config.RequireExplicitScratchForBase && !boolValue(config.Scan.ResolveScratchClaims, false) {
			return fmt.Errorf("selection %q requires explicit scratch claims but scan.resolve_scratch_claims is false", selection.config.Name)
		}
	}
	if (boolValue(config.Scan.ResolveReportedCompute, false) || boolValue(config.Scan.ResolveScratchClaims, false)) && !boolValue(config.Scan.FetchIndividualReadmes, true) {
		if (config.Scan.BulkModelCardsURL == "" && len(config.Scan.BulkModelCardsURLs) == 0) || config.Scan.BulkModelCardsFile == "" {
			return errors.New("a bulk model-card URL and scan.bulk_model_cards_file are required when individual README fetching is disabled")
		}
	}
	client := &httpClient{client: &http.Client{Timeout: time.Duration(config.Scan.TimeoutSeconds) * time.Second}, token: os.Getenv("HF_TOKEN"), retries: *config.Scan.Retries}
	client.log = func(level, event, message string, fields map[string]any) {
		if level != "debug" || config.Logging.HTTPRequests {
			logger.log(level, event, message, fields)
		}
	}
	models := make([]catalogModel, 0)
	pages, scanned := 0, 0
	complete := false
	checkpointPath := resolvedCheckpointPath(config.OutputDir, config.Scan.CatalogCheckpoint)
	checkpoint, checkpointErr := loadCatalogCheckpoint(checkpointPath, config.Endpoint, earliest, latest, boolValue(config.Scan.RequireWeights, true))
	if checkpointErr != nil {
		logger.warn("catalog_checkpoint_failed", "catalog checkpoint could not be loaded; scanning Hub", map[string]any{"path": checkpointPath, "error": checkpointErr.Error()})
	}
	if checkpoint != nil {
		models, pages, scanned, complete = checkpoint.Models, checkpoint.Pages, checkpoint.Scanned, checkpoint.Complete
		logger.info("catalog_checkpoint_loaded", "loaded completed catalog scan from checkpoint", map[string]any{"path": checkpointPath, "pages": pages, "scanned": scanned, "retained": len(models)})
	}
	address := ""
	if checkpoint == nil {
		address = catalogAPIURL(config.Endpoint, config.Scan.PageSize, latest)
		logger.info("scan_positioned", "catalog scan positioned at newest selection boundary", map[string]any{"created_before": latest.Format(time.RFC3339Nano)})
	}
	for address != "" {
		body, headers, err := client.get(ctx, address, false)
		if err != nil {
			return fmt.Errorf("fetch model catalog page %d: %w", pages+1, err)
		}
		var page []catalogModel
		if err := json.Unmarshal(body, &page); err != nil {
			return fmt.Errorf("decode model catalog page %d: %w", pages+1, err)
		}
		if len(page) == 0 {
			complete = true
			break
		}
		pages++
		scanned += len(page)
		reachedOlder := false
		for _, model := range page {
			if strings.TrimSpace(model.ID) == "" {
				logger.warn("model_skipped", "model has an empty repository id", map[string]any{"reason": "empty_repo_id", "page": pages})
				continue
			}
			created, err := time.Parse(time.RFC3339Nano, model.CreatedAt)
			if err != nil {
				logger.warn("model_skipped", "model has an invalid createdAt timestamp", map[string]any{"repo_id": model.ID, "created_at": model.CreatedAt, "reason": "invalid_created_at", "error": err.Error()})
				continue
			}
			if created.Before(earliest) {
				reachedOlder = true
				if config.Logging.SkippedRecords {
					logger.debug("model_skipped", "model is older than the scan window", map[string]any{"repo_id": model.ID, "reason": "older_than_window"})
				}
				continue
			}
			if created.After(now) {
				logger.warn("model_skipped", "model creation time is in the future", map[string]any{"repo_id": model.ID, "created_at": model.CreatedAt, "effective_now": now.Format(time.RFC3339Nano), "reason": "future_created_at"})
				continue
			}
			if created.After(latest) {
				if config.Logging.SkippedRecords {
					logger.debug("model_skipped", "model is newer than every selection window", map[string]any{"repo_id": model.ID, "reason": "newer_than_window"})
				}
				continue
			}
			if boolValue(config.Scan.RequireWeights, true) && model.Safetensors.Total <= 0 && !hasRecognizedWeight(model.Siblings) {
				if config.Logging.SkippedRecords {
					logger.debug("model_skipped", "model has no recognized weight file", map[string]any{"repo_id": model.ID, "reason": "no_supported_weight_file"})
				}
				continue
			}
			models = append(models, model)
		}
		if !config.Scan.Quiet {
			logger.info("scan_progress", "catalog page processed", map[string]any{"pages": pages, "scanned": scanned, "retained": len(models)})
		}
		if reachedOlder {
			complete = true
			break
		}
		if config.Scan.MaxPages > 0 && pages >= config.Scan.MaxPages {
			logger.warn("scan_truncated", "scan stopped at configured max_pages", map[string]any{"max_pages": config.Scan.MaxPages, "scanned": scanned})
			break
		}
		address = nextLink(headers)
		if address == "" {
			complete = true
		}
	}
	if checkpoint == nil && complete && checkpointPath != "" {
		value := catalogCheckpoint{Endpoint: config.Endpoint, Earliest: earliest.Format(time.RFC3339Nano), Latest: latest.Format(time.RFC3339Nano), RequireWeights: boolValue(config.Scan.RequireWeights, true), Complete: complete, Pages: pages, Scanned: scanned, CreatedAt: time.Now().UTC().Format(time.RFC3339Nano), Models: models}
		if err := saveCatalogCheckpoint(checkpointPath, value); err != nil {
			return fmt.Errorf("save catalog checkpoint: %w", err)
		}
		logger.info("catalog_checkpoint_saved", "saved completed catalog scan checkpoint", map[string]any{"path": checkpointPath, "models": len(models)})
	}

	baseSet := make(map[string]bool)
	if boolValue(config.Scan.ResolveBaseParameters, true) {
		for _, model := range models {
			if catalogModelKind(model) != "adapter" {
				continue
			}
			for _, selection := range selections {
				needsParameterRange := selection.config.MinParametersB != nil || selection.config.MaxParametersB != nil
				needsEstimatedCompute := len(selection.config.ComputeProfiles) > 0 && !selection.config.RequireReportedDerivativeCompute
				if !needsParameterRange && !needsEstimatedCompute {
					continue
				}
				withoutRange := selection
				withoutRange.config.MinParametersB = nil
				withoutRange.config.MaxParametersB = nil
				if matchesSelection(model, withoutRange, model.Safetensors.Total) {
					if id := firstBaseModel(model); id != "" {
						baseSet[id] = true
					}
					break
				}
			}
		}
	}
	baseIDs := make([]string, 0, len(baseSet))
	for id := range baseSet {
		baseIDs = append(baseIDs, id)
	}
	sort.Strings(baseIDs)
	baseParams, baseErrors := fetchBaseParameters(ctx, client, config.Endpoint, baseIDs, config.Owners.Workers, logger)

	signalCandidates := make([]catalogModel, 0)
	signalCandidateIDs := make(map[string]bool)
	reportedCandidateIDs := make(map[string]bool)
	scratchCandidateIDs := make(map[string]bool)
	if boolValue(config.Scan.ResolveReportedCompute, false) || boolValue(config.Scan.ResolveScratchClaims, false) {
		for _, model := range models {
			kind := catalogModelKind(model)
			params, _ := effectiveParameters(model, baseParams, baseErrors)
			for _, selection := range selections {
				if !matchesSelection(model, selection, params) {
					continue
				}
				isTrainableKind := kind == "base" || kind == "finetune" || kind == "adapter"
				// Parse compute for residual base candidates too. Missing Hub lineage
				// metadata is common; the card itself may reveal CPT/SFT/adapter work.
				needsCompute := selection.config.RequireReportedCompute || ((selection.config.RequireReportedDerivativeCompute || selection.config.CountReportedDerivativeCosts) && isTrainableKind)
				needsScratch := (selection.config.RequireExplicitScratchForBase || selection.config.FormulaRequiresExplicitScratch) && catalogScratchOverrideEligible(model)
				if needsCompute {
					reportedCandidateIDs[model.ID] = true
				}
				if needsScratch {
					scratchCandidateIDs[model.ID] = true
				}
				if needsCompute || needsScratch {
					signalCandidateIDs[model.ID] = true
				}
			}
			if signalCandidateIDs[model.ID] {
				signalCandidates = append(signalCandidates, model)
			}
		}
	}
	bulkFile := config.Scan.BulkModelCardsFile
	if bulkFile != "" && !filepath.IsAbs(bulkFile) {
		bulkFile = filepath.Join(config.OutputDir, bulkFile)
	}
	bulkURLs := append([]string(nil), config.Scan.BulkModelCardsURLs...)
	if len(bulkURLs) == 0 && config.Scan.BulkModelCardsURL != "" {
		bulkURLs = append(bulkURLs, config.Scan.BulkModelCardsURL)
	}
	reportedCompute, scratchClaims, err := resolveTrainingSignals(ctx, client, config.Endpoint, signalCandidates, reportedCandidateIDs, scratchCandidateIDs, config.Owners.Workers, logger, bulkURLs, bulkFile, boolValue(config.Scan.FetchIndividualReadmes, true))
	if err != nil {
		return fmt.Errorf("resolve training signals: %w", err)
	}
	scratchResolved := 0
	for _, claim := range scratchClaims {
		if claim != nil {
			scratchResolved++
		}
	}
	logger.info("reported_compute_resolved", "reported training compute scan completed", map[string]any{"candidates": len(reportedCandidateIDs), "resolved": len(reportedCompute)})
	logger.info("scratch_claims_resolved", "explicit scratch claim scan completed", map[string]any{"candidates": len(scratchCandidateIDs), "resolved": scratchResolved})

	var records []catalogRecord
	computeProfiles := make(map[string]computeProfile, len(config.Compute))
	for _, profile := range config.Compute {
		computeProfiles[profile.Name] = profile
	}
	for _, selection := range selections {
		var selected []catalogRecord
		for _, model := range models {
			params, parameterError := effectiveParameters(model, baseParams, baseErrors)
			if matchesSelection(model, selection, params) {
				reported := reportedCompute[model.ID]
				scratch := scratchClaims[model.ID]
				scratchUsable := scratch != nil && scratchClaimMatchesOwner(model, scratch) && scratchClaimFallsInSelection(scratch, selection)
				if !scratchUsable {
					scratch = nil
				}
				kind := resolvedCatalogModelKind(model, scratch, reported)
				if selection.config.RequireReportedCompute && reported == nil {
					continue
				}
				if selection.config.RequireReportedDerivativeCompute && (kind == "finetune" || kind == "adapter") && reported == nil {
					continue
				}
				if (selection.config.RequireReportedCompute || kind == "finetune" || kind == "adapter") && !reportedComputeFallsInSelection(reported, selection) {
					continue
				}
				if selection.config.RequireExplicitScratchForBase && kind == "base" && scratch == nil {
					continue
				}
				selected = append(selected, modelToRecord(config.Endpoint, model, selection.config, params, parameterError, computeProfiles, reported, scratch))
			}
		}
		sort.SliceStable(selected, func(i, j int) bool {
			if !sortValueDifferent(selected[i], selected[j], selection.config.SortBy) {
				return selected[i].RepoID < selected[j].RepoID
			}
			return selectionLess(selected[i], selected[j], selection.config)
		})
		if len(selection.config.OwnerTypes) == 0 && selection.config.Limit > 0 && len(selected) > selection.config.Limit {
			selected = selected[:selection.config.Limit]
		}
		records = append(records, selected...)
	}
	deduplicateReportedDerivativeAgainstBase(records)
	deduplicateReportedTrainingRuns(records)
	deduplicateScratchTrainingRuns(records)

	ownerNamesSet := make(map[string]bool)
	for _, record := range records {
		if record.Owner != "" {
			ownerNamesSet[record.Owner] = true
		}
	}
	ownerNames := make([]string, 0, len(ownerNamesSet))
	for name := range ownerNamesSet {
		ownerNames = append(ownerNames, name)
	}
	sort.Strings(ownerNames)
	owners := make(map[string]*ownerOverview)
	if boolValue(config.Owners.Enabled, true) {
		owners = fetchOwners(ctx, client, config.Endpoint, ownerNames, config.Owners.Workers, logger)
	}

	filtered := records[:0]
	selectionByName := make(map[string]selectionConfig)
	for _, item := range config.Selections {
		selectionByName[item.Name] = item
	}
	for _, record := range records {
		overview := owners[record.Owner]
		if overview != nil {
			record.OwnerType = overview.Type
		} else {
			record.OwnerType = "unknown"
		}
		allowed := selectionByName[record.Selection].OwnerTypes
		if len(allowed) > 0 && !containsAny(lowerSet([]string{record.OwnerType}), allowed) {
			continue
		}
		filtered = append(filtered, record)
	}
	records = filtered
	// owner_types is resolved after profile enrichment, so its limit must also be
	// applied afterwards to preserve "filter, then limit" semantics.
	limited := records[:0]
	selectionCounts := make(map[string]int)
	for _, record := range records {
		limit := selectionByName[record.Selection].Limit
		if limit > 0 && selectionCounts[record.Selection] >= limit {
			continue
		}
		selectionCounts[record.Selection]++
		limited = append(limited, record)
	}
	records = limited

	for _, overview := range owners {
		overview.SelectedRepos = 0
		overview.Downloads = 0
		overview.Likes = 0
		overview.KnownTrainingCosts = 0
		overview.TrainingCostUSD = 0
		overview.Selections = nil
	}
	seenRepos := make(map[string]bool)
	ownerSelections := make(map[string]map[string]bool)
	for _, record := range records {
		overview := owners[record.Owner]
		if overview == nil {
			overview = &ownerOverview{Name: record.Owner, Type: record.OwnerType}
			owners[record.Owner] = overview
		}
		key := record.Owner + "\x00" + record.RepoID
		if !seenRepos[key] {
			overview.SelectedRepos++
			overview.Downloads += record.Downloads
			overview.Likes += record.Likes
			if record.TrainingCostUSD != nil {
				overview.KnownTrainingCosts++
				overview.TrainingCostUSD += *record.TrainingCostUSD
			}
			seenRepos[key] = true
		}
		if ownerSelections[record.Owner] == nil {
			ownerSelections[record.Owner] = make(map[string]bool)
		}
		ownerSelections[record.Owner][record.Selection] = true
	}
	for owner, values := range ownerSelections {
		for value := range values {
			owners[owner].Selections = append(owners[owner].Selections, value)
		}
		sort.Strings(owners[owner].Selections)
	}

	if err := writeCatalogModels(config.OutputDir, records); err != nil {
		return fmt.Errorf("write model outputs: %w", err)
	}
	if err := writeCatalogOwners(config.OutputDir, owners); err != nil {
		return fmt.Errorf("write owner outputs: %w", err)
	}
	summary := buildCatalogSummary(now, complete, pages, scanned, len(models), len(reportedCandidateIDs), len(reportedCompute), len(scratchCandidateIDs), scratchResolved, records, config.Selections)
	if err := writeJSONFile(filepath.Join(config.OutputDir, "summary.json"), summary); err != nil {
		return fmt.Errorf("write summary output: %w", err)
	}
	if err := writeMarketReport(config.OutputDir, summary, owners); err != nil {
		return fmt.Errorf("write market report: %w", err)
	}
	logger.info("run_completed", "catalog run completed", map[string]any{"complete": complete, "pages": pages, "scanned": scanned, "selected": len(records), "owners": len(owners)})
	if _, err := fmt.Printf("Catalog complete=%t pages=%d scanned=%d selected=%d owners=%d\n", complete, pages, scanned, len(records), len(owners)); err != nil {
		return fmt.Errorf("write completion message to stdout: %w", err)
	}
	if _, err := fmt.Printf("Output: %s\n", config.OutputDir); err != nil {
		return fmt.Errorf("write output path to stdout: %w", err)
	}
	for name, selection := range summary.Selections {
		if _, err := fmt.Printf("Training cost [%s]: $%.2f (%d models with cost)\n", name, selection.TotalTrainingCostUSD, selection.KnownTrainingCosts); err != nil {
			return fmt.Errorf("write total cost to stdout: %w", err)
		}
	}
	return nil
}

func writeMarketReport(directory string, summary catalogSummary, owners map[string]*ownerOverview) error {
	var report strings.Builder
	report.WriteString("# Оценка стоимости обучения моделей Hugging Face\n\n")
	report.WriteString("Снимок сформирован: " + summary.GeneratedAt + "\n\n")
	names := make([]string, 0, len(summary.Selections))
	for name := range summary.Selections {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		item := summary.Selections[name]
		fmt.Fprintf(&report, "## %s\n\n", name)
		if item.Description != "" {
			report.WriteString(item.Description + "\n\n")
		}
		fmt.Fprintf(&report, "- Итоговая стоимость: **$%.2f**\n", item.TotalTrainingCostUSD)
		fmt.Fprintf(&report, "- Сценарий строго 20 токенов/параметр, MoE по active × total: **$%.2f**\n", item.Formula20HybridUSD)
		fmt.Fprintf(&report, "- Буквальный upper-сценарий 20 токенов/параметр и total² для MoE: **$%.2f**\n", item.Formula20LiteralUSD)
		fmt.Fprintf(&report, "- Репозиториев в целевой выборке: %d\n", item.Models)
		fmt.Fprintf(&report, "- Самостоятельных training run с рассчитанной стоимостью: %d\n", item.KnownTrainingCosts)
		fmt.Fprintf(&report, "- Уникальных владельцев: %d\n", item.UniqueOwners)
		fmt.Fprintf(&report, "- Скачиваний: %d; likes: %d\n\n", item.TotalDownloads, item.TotalLikes)
		report.WriteString("| Категория | Репозиториев | Run с cost | Стоимость, USD |\n|---|---:|---:|---:|\n")
		for _, kind := range []string{"base", "fork", "finetune", "adapter", "quantized", "merge"} {
			if item.ModelKinds[kind] > 0 {
				fmt.Fprintf(&report, "| %s | %d | %d | %.2f |\n", kind, item.ModelKinds[kind], item.CostedModelsByKind[kind], item.TrainingCostByKind[kind])
			}
		}
		report.WriteString("\n### Base по числу параметров\n\n")
		report.WriteString("| Диапазон | Base-репозиториев | Подтверждённых run с cost | Стоимость, USD |\n|---|---:|---:|---:|\n")
		for _, bucket := range []string{"<2B", "2-<7B", "7-<13B", "13-<34B", "34-<70B", "70-<120B", "120-<500B", ">=500B", "unknown"} {
			if item.BaseModelsBySize[bucket] > 0 {
				fmt.Fprintf(&report, "| %s | %d | %d | %.2f |\n", bucket, item.BaseModelsBySize[bucket], item.CostedBaseBySize[bucket], item.BaseCostBySize[bucket])
			}
		}
		report.WriteString("\n")
	}
	fmt.Fprintf(&report, "Scratch-кандидатов: %d; явное обучение с нуля подтверждено: %d; без подтверждения: %d. Fine-tune/adapter-кандидатов на compute: %d; найден compute/cost: %d; без данных: %d. Репозиториев с весами после первичного фильтра: %d.\n\n", summary.ScratchClaimCandidates, summary.ScratchClaimsResolved, summary.ScratchClaimsIgnored, summary.ReportedComputeCandidates, summary.ReportedComputeResolved, summary.ReportedComputeIgnored, summary.RetainedWeightRepositories)
	report.WriteString("Для base-модели используется FLOPs = 6 × N × T из `formula.txt`. Если карточка сообщает фактический объём pretraining-токенов, берётся он; иначе T = 20 × N. Для dense это сводится к cost = 155.881361644759 × P² USD. Для MoE при наличии публичного active count FLOPs/token считаются по active-параметрам, а token fallback — по total. Fork, quantization, conversion и merge не получают стоимость; fine-tune/adapters входят только при опубликованном compute/cost.\n\n")
	report.WriteString("Ограничения интерпретации:\n\n")
	report.WriteString("- Первичный отбор 2025 сделан по `createdAt` репозитория.\n")
	report.WriteString("- Если в training section явно указан другой год и не указан 2025, такой run исключён; без даты используется год `createdAt` как допущение.\n")
	report.WriteString("- Base без достаточного independent-pretraining evidence и derivatives без опубликованного compute в сумму не входят.\n")
	report.WriteString("- Копии с идентичной карточкой дедуплицируются; переписанные зеркала без общего run ID всё ещё невозможно надёжно связать.\n")
	report.WriteString("- Один и тот же текст карточки мог использоваться для разных запусков; консервативная дедупликация может занижать нижнюю границу.\n")
	report.WriteString("- Формула оценивает H100 GPU-rental equivalent; CPU, сеть, storage, данные и работа команды не включены.\n")
	report.WriteString("- Для MoE основной estimate использует опубликованное число active-параметров; `summary.json` отдельно сохраняет буквальный total² upper-сценарий.\n")
	report.WriteString("- Приватные, недоступные и уже удалённые репозитории не входят в текущий публичный снимок.\n\n")
	list := make([]*ownerOverview, 0, len(owners))
	for _, owner := range owners {
		if owner.KnownTrainingCosts > 0 {
			list = append(list, owner)
		}
	}
	sort.Slice(list, func(i, j int) bool {
		if list[i].TrainingCostUSD == list[j].TrainingCostUSD {
			return list[i].Name < list[j].Name
		}
		return list[i].TrainingCostUSD > list[j].TrainingCostUSD
	})
	if len(list) > 0 {
		report.WriteString("## Владельцы с наибольшей оценкой\n\n| Владелец | Тип | Моделей с cost | Стоимость, USD |\n|---|---:|---:|---:|\n")
		limit := min(100, len(list))
		for _, owner := range list[:limit] {
			fmt.Fprintf(&report, "| [%s](https://huggingface.co/%s) | %s | %d | %.2f |\n", strings.ReplaceAll(owner.Name, "|", "\\|"), url.PathEscape(owner.Name), owner.Type, owner.KnownTrainingCosts, owner.TrainingCostUSD)
		}
	}
	return os.WriteFile(filepath.Join(directory, "report.md"), []byte(report.String()), 0o644)
}

func writeJSONFile(path string, value any) error {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return fmt.Errorf("encode JSON for %q: %w", path, err)
	}
	if err := os.WriteFile(path, append(data, '\n'), 0o644); err != nil {
		return fmt.Errorf("write %q: %w", path, err)
	}
	return nil
}

func closeFileAfterError(file *os.File, path string, cause error) error {
	if closeErr := file.Close(); closeErr != nil {
		return errors.Join(cause, fmt.Errorf("close %q after failure: %w", path, closeErr))
	}
	return cause
}

func writeCatalogModels(directory string, records []catalogRecord) error {
	if err := writeJSONFile(filepath.Join(directory, "models.json"), records); err != nil {
		return err
	}
	path := filepath.Join(directory, "models.csv")
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create %q: %w", path, err)
	}
	w := csv.NewWriter(file)
	if err := w.Write([]string{"selection", "repo_id", "repo_url", "owner", "owner_type", "created_at", "last_modified", "pipeline_tag", "library_name", "model_kind", "base_model", "own_parameters", "effective_parameters", "parameters_b", "downloads", "likes", "training_cost_usd", "training_cost_method", "reported_compute_json", "scratch_claim_json", "tags", "compute_estimates_json", "errors"}); err != nil {
		return closeFileAfterError(file, path, fmt.Errorf("write header to %q: %w", path, err))
	}
	for _, r := range records {
		p := ""
		if r.ParametersB != nil {
			p = strconv.FormatFloat(*r.ParametersB, 'f', 6, 64)
		}
		computeJSON, err := json.Marshal(r.Compute)
		if err != nil {
			return closeFileAfterError(file, path, fmt.Errorf("encode compute estimates for %q: %w", r.RepoID, err))
		}
		reportedJSON := ""
		if r.ReportedCompute != nil {
			encoded, err := json.Marshal(r.ReportedCompute)
			if err != nil {
				return closeFileAfterError(file, path, fmt.Errorf("encode reported compute for %q: %w", r.RepoID, err))
			}
			reportedJSON = string(encoded)
		}
		scratchJSON := ""
		if r.ScratchClaim != nil {
			encoded, err := json.Marshal(r.ScratchClaim)
			if err != nil {
				return closeFileAfterError(file, path, fmt.Errorf("encode scratch claim for %q: %w", r.RepoID, err))
			}
			scratchJSON = string(encoded)
		}
		cost := ""
		if r.TrainingCostUSD != nil {
			cost = strconv.FormatFloat(*r.TrainingCostUSD, 'f', 2, 64)
		}
		if err := w.Write([]string{r.Selection, r.RepoID, r.RepoURL, r.Owner, r.OwnerType, r.CreatedAt, r.LastModified, r.PipelineTag, r.LibraryName, r.ModelKind, r.BaseModel, strconv.FormatInt(r.OwnParameters, 10), strconv.FormatInt(r.EffectiveParameters, 10), p, strconv.FormatInt(r.Downloads, 10), strconv.FormatInt(r.Likes, 10), cost, r.TrainingCostMethod, reportedJSON, scratchJSON, strings.Join(r.Tags, "|"), string(computeJSON), strings.Join(r.Errors, " | ")}); err != nil {
			return closeFileAfterError(file, path, fmt.Errorf("write model %q to %q: %w", r.RepoID, path, err))
		}
	}
	w.Flush()
	if err := w.Error(); err != nil {
		return closeFileAfterError(file, path, fmt.Errorf("flush %q: %w", path, err))
	}
	if err := file.Sync(); err != nil {
		return closeFileAfterError(file, path, fmt.Errorf("sync %q: %w", path, err))
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("close %q: %w", path, err)
	}
	return nil
}

func writeCatalogOwners(directory string, owners map[string]*ownerOverview) error {
	list := make([]*ownerOverview, 0, len(owners))
	for _, owner := range owners {
		if owner.SelectedRepos > 0 {
			list = append(list, owner)
		}
	}
	sort.Slice(list, func(i, j int) bool {
		if list[i].Downloads == list[j].Downloads {
			return list[i].Name < list[j].Name
		}
		return list[i].Downloads > list[j].Downloads
	})
	if err := writeJSONFile(filepath.Join(directory, "owners.json"), list); err != nil {
		return err
	}
	path := filepath.Join(directory, "owners.csv")
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create %q: %w", path, err)
	}
	w := csv.NewWriter(file)
	if err := w.Write([]string{"owner", "owner_type", "profile_url", "fullname", "verified", "pro", "plan", "followers", "models", "datasets", "spaces", "papers", "members", "created_at", "details", "selected_repos", "selected_downloads", "selected_likes", "known_training_costs", "training_cost_usd", "selections", "error"}); err != nil {
		return closeFileAfterError(file, path, fmt.Errorf("write header to %q: %w", path, err))
	}
	for _, o := range list {
		if err := w.Write([]string{o.Name, o.Type, "https://huggingface.co/" + o.Name, o.Fullname, strconv.FormatBool(o.IsVerified), strconv.FormatBool(o.IsPro), o.Plan, strconv.FormatInt(o.NumFollowers, 10), strconv.FormatInt(o.NumModels, 10), strconv.FormatInt(o.NumDatasets, 10), strconv.FormatInt(o.NumSpaces, 10), strconv.FormatInt(o.NumPapers, 10), strconv.FormatInt(o.NumUsers, 10), o.CreatedAt, o.Details, strconv.FormatInt(o.SelectedRepos, 10), strconv.FormatInt(o.Downloads, 10), strconv.FormatInt(o.Likes, 10), strconv.FormatInt(o.KnownTrainingCosts, 10), strconv.FormatFloat(o.TrainingCostUSD, 'f', 2, 64), strings.Join(o.Selections, "|"), o.Error}); err != nil {
			return closeFileAfterError(file, path, fmt.Errorf("write owner %q to %q: %w", o.Name, path, err))
		}
	}
	w.Flush()
	if err := w.Error(); err != nil {
		return closeFileAfterError(file, path, fmt.Errorf("flush %q: %w", path, err))
	}
	if err := file.Sync(); err != nil {
		return closeFileAfterError(file, path, fmt.Errorf("sync %q: %w", path, err))
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("close %q: %w", path, err)
	}
	return nil
}

func buildCatalogSummary(now time.Time, complete bool, pages, scanned, retained, reportedCandidates, reportedResolved, scratchCandidates, scratchResolved int, records []catalogRecord, selections []selectionConfig) catalogSummary {
	result := catalogSummary{
		GeneratedAt: time.Now().UTC().Format(time.RFC3339Nano), EffectiveNow: now.Format(time.RFC3339Nano),
		Complete: complete, PagesScanned: pages, ModelsScanned: scanned,
		ReportedComputeCandidates: reportedCandidates, ReportedComputeResolved: reportedResolved,
		ReportedComputeIgnored: reportedCandidates - reportedResolved,
		ScratchClaimCandidates: scratchCandidates, ScratchClaimsResolved: scratchResolved,
		ScratchClaimsIgnored: scratchCandidates - scratchResolved, RetainedWeightRepositories: retained,
		Selections: make(map[string]selectionSummary),
	}
	for _, selection := range selections {
		result.Selections[selection.Name] = selectionSummary{
			Description:          selection.Description,
			OwnerTypes:           make(map[string]int),
			ModelKinds:           make(map[string]int),
			CostedModelsByKind:   make(map[string]int),
			TrainingCostByKind:   make(map[string]float64),
			CostedModelsByMethod: make(map[string]int),
			TrainingCostByMethod: make(map[string]float64),
			BaseModelsBySize:     make(map[string]int),
			CostedBaseBySize:     make(map[string]int),
			BaseCostBySize:       make(map[string]float64),
		}
	}
	owners := make(map[string]map[string]bool)
	for _, record := range records {
		s := result.Selections[record.Selection]
		s.Description = record.Description
		s.Models++
		s.TotalDownloads += record.Downloads
		s.TotalLikes += record.Likes
		if record.TrainingCostUSD != nil {
			s.KnownTrainingCosts++
			s.TotalTrainingCostUSD += *record.TrainingCostUSD
			s.CostedModelsByKind[record.ModelKind]++
			s.TrainingCostByKind[record.ModelKind] += *record.TrainingCostUSD
			s.CostedModelsByMethod[record.TrainingCostMethod]++
			s.TrainingCostByMethod[record.TrainingCostMethod] += *record.TrainingCostUSD
			if record.ModelKind != "base" {
				s.Formula20HybridUSD += *record.TrainingCostUSD
				s.Formula20LiteralUSD += *record.TrainingCostUSD
			} else {
				literalCost := 0.0
				for _, estimate := range record.Compute {
					if estimate.Method == "formula_txt_literal_total_parameters" {
						literalCost = estimate.CostUSD
						break
					}
				}
				if literalCost > 0 {
					s.Formula20LiteralUSD += literalCost
					hybridCost := literalCost
					parametersB := float64(record.EffectiveParameters) / 1e9
					if record.ScratchClaim != nil && record.ScratchClaim.ActiveParametersB > 0 && record.ScratchClaim.ActiveParametersB < parametersB {
						hybridCost *= record.ScratchClaim.ActiveParametersB / parametersB
					}
					s.Formula20HybridUSD += hybridCost
				}
			}
		}
		if s.OwnerTypes == nil {
			s.OwnerTypes = make(map[string]int)
		}
		s.OwnerTypes[record.OwnerType]++
		if s.ModelKinds == nil {
			s.ModelKinds = make(map[string]int)
		}
		s.ModelKinds[record.ModelKind]++
		if record.ModelKind == "base" {
			bucket := parameterRange(record.EffectiveParameters)
			s.BaseModelsBySize[bucket]++
			if record.TrainingCostUSD != nil {
				s.CostedBaseBySize[bucket]++
				s.BaseCostBySize[bucket] += *record.TrainingCostUSD
			}
		}
		if record.EffectiveParameters > 0 {
			s.KnownParams++
		}
		if owners[record.Selection] == nil {
			owners[record.Selection] = make(map[string]bool)
		}
		owners[record.Selection][record.Owner] = true
		result.Selections[record.Selection] = s
	}
	for name, values := range owners {
		s := result.Selections[name]
		s.UniqueOwners = len(values)
		result.Selections[name] = s
	}
	return result
}

func parameterRange(parameters int64) string {
	if parameters <= 0 {
		return "unknown"
	}
	billions := float64(parameters) / 1e9
	switch {
	case billions < 2:
		return "<2B"
	case billions < 7:
		return "2-<7B"
	case billions < 13:
		return "7-<13B"
	case billions < 34:
		return "13-<34B"
	case billions < 70:
		return "34-<70B"
	case billions < 120:
		return "70-<120B"
	case billions < 500:
		return "120-<500B"
	default:
		return ">=500B"
	}
}

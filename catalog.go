package main

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

type catalogConfig struct {
	Endpoint   string            `json:"endpoint"`
	OutputDir  string            `json:"output_dir"`
	Scan       catalogScanConfig `json:"scan"`
	Owners     ownerConfig       `json:"owners"`
	Compute    []computeProfile  `json:"compute_profiles"`
	Selections []selectionConfig `json:"selections"`
}

type catalogScanConfig struct {
	Now                   string `json:"now"`
	PageSize              int    `json:"page_size"`
	MaxPages              int    `json:"max_pages"`
	RequireWeights        *bool  `json:"require_weights"`
	ResolveBaseParameters *bool  `json:"resolve_base_parameters"`
	TimeoutSeconds        int    `json:"timeout_seconds"`
	Retries               *int   `json:"retries"`
	Quiet                 bool   `json:"quiet"`
}

type ownerConfig struct {
	Enabled *bool `json:"enabled"`
	Workers int   `json:"workers"`
}

type selectionConfig struct {
	Name                     string   `json:"name"`
	Description              string   `json:"description"`
	LookbackDays             int      `json:"lookback_days"`
	CreatedFrom              string   `json:"created_from"`
	CreatedTo                string   `json:"created_to"`
	PipelineTags             []string `json:"pipeline_tags"`
	Libraries                []string `json:"libraries"`
	ModelKinds               []string `json:"model_kinds"`
	OwnerTypes               []string `json:"owner_types"`
	TagsAny                  []string `json:"tags_any"`
	TagsAll                  []string `json:"tags_all"`
	ExcludeTags              []string `json:"exclude_tags"`
	ModelIDRegex             string   `json:"model_id_regex"`
	MinParametersB           *float64 `json:"min_parameters_b"`
	MaxParametersB           *float64 `json:"max_parameters_b"`
	IncludeUnknownParameters bool     `json:"include_unknown_parameters"`
	MinDownloads             int64    `json:"min_downloads"`
	MaxDownloads             int64    `json:"max_downloads"`
	MinLikes                 int64    `json:"min_likes"`
	MaxLikes                 int64    `json:"max_likes"`
	SortBy                   string   `json:"sort_by"`
	SortDirection            string   `json:"sort_direction"`
	Limit                    int      `json:"limit"`
	ComputeProfiles          []string `json:"compute_profiles"`
}

type computeProfile struct {
	Name                 string  `json:"name"`
	GPUName              string  `json:"gpu_name"`
	GPUTFLOPS            float64 `json:"gpu_tflops"`
	Efficiency           float64 `json:"efficiency"`
	GPUHourCostUSD       float64 `json:"gpu_hour_cost_usd"`
	TokensPerParameter   float64 `json:"tokens_per_parameter"`
	Machines             int     `json:"machines"`
	GPUsPerMachine       int     `json:"gpus_per_machine"`
	FinetuneCostFraction float64 `json:"finetune_compute_fraction"`
}

type computeEstimate struct {
	Profile   string  `json:"profile"`
	GPUName   string  `json:"gpu_name"`
	Machines  int     `json:"machines"`
	TotalGPUs int     `json:"total_gpus"`
	GPUHours  float64 `json:"gpu_hours"`
	WallDays  float64 `json:"wall_days"`
	CostUSD   float64 `json:"cost_usd"`
	Fraction  float64 `json:"training_fraction"`
}

type catalogSafetensors struct {
	Total int64 `json:"total"`
}

type catalogBaseModel struct {
	ID string `json:"id"`
}

type catalogBaseModels struct {
	Relation string             `json:"relation"`
	Models   []catalogBaseModel `json:"models"`
}

type catalogModel struct {
	ID           string             `json:"id"`
	Author       string             `json:"author"`
	CreatedAt    string             `json:"createdAt"`
	LastModified string             `json:"lastModified"`
	Downloads    int64              `json:"downloads"`
	Likes        int64              `json:"likes"`
	PipelineTag  string             `json:"pipeline_tag"`
	LibraryName  string             `json:"library_name"`
	Tags         []string           `json:"tags"`
	Siblings     []sibling          `json:"siblings"`
	Safetensors  catalogSafetensors `json:"safetensors"`
	BaseModels   catalogBaseModels  `json:"baseModels"`
}

type catalogRecord struct {
	Selection           string            `json:"selection"`
	Description         string            `json:"selection_description,omitempty"`
	RepoID              string            `json:"repo_id"`
	RepoURL             string            `json:"repo_url"`
	Owner               string            `json:"owner"`
	OwnerType           string            `json:"owner_type"`
	CreatedAt           string            `json:"created_at"`
	LastModified        string            `json:"last_modified"`
	PipelineTag         string            `json:"pipeline_tag"`
	LibraryName         string            `json:"library_name"`
	ModelKind           string            `json:"model_kind"`
	BaseModel           string            `json:"base_model,omitempty"`
	OwnParameters       int64             `json:"own_parameters,omitempty"`
	EffectiveParameters int64             `json:"effective_parameters,omitempty"`
	ParametersB         *float64          `json:"parameters_b,omitempty"`
	Downloads           int64             `json:"downloads"`
	Likes               int64             `json:"likes"`
	Tags                []string          `json:"tags"`
	Compute             []computeEstimate `json:"compute_estimates,omitempty"`
}

type ownerOverview struct {
	Name          string   `json:"name"`
	User          string   `json:"user"`
	Type          string   `json:"type"`
	Fullname      string   `json:"fullname"`
	AvatarURL     string   `json:"avatarUrl"`
	Details       string   `json:"details"`
	Plan          string   `json:"plan"`
	IsVerified    bool     `json:"isVerified"`
	IsPro         bool     `json:"isPro"`
	NumUsers      int64    `json:"numUsers"`
	NumModels     int64    `json:"numModels"`
	NumDatasets   int64    `json:"numDatasets"`
	NumSpaces     int64    `json:"numSpaces"`
	NumPapers     int64    `json:"numPapers"`
	NumFollowers  int64    `json:"numFollowers"`
	CreatedAt     string   `json:"createdAt"`
	Error         string   `json:"error,omitempty"`
	SelectedRepos int64    `json:"selected_repos"`
	Downloads     int64    `json:"selected_downloads"`
	Likes         int64    `json:"selected_likes"`
	Selections    []string `json:"selections"`
}

var (
	catalogAdapterNameRE  = regexp.MustCompile(`(?i)(^|[-_./])(lora|qlora|adapter|peft)([-_./]|$)`)
	catalogFinetuneNameRE = regexp.MustCompile(`(?i)(^|[-_./])(finetune|fine[-_]?tune|sft|dpo)([-_./]|$)`)
)

type catalogSummary struct {
	GeneratedAt   string                      `json:"generated_at"`
	EffectiveNow  string                      `json:"effective_now"`
	Complete      bool                        `json:"complete"`
	PagesScanned  int                         `json:"pages_scanned"`
	ModelsScanned int                         `json:"models_scanned"`
	Selections    map[string]selectionSummary `json:"selections"`
}

type selectionSummary struct {
	Description    string         `json:"description,omitempty"`
	Models         int            `json:"models"`
	UniqueOwners   int            `json:"unique_owners"`
	OwnerTypes     map[string]int `json:"owner_types"`
	ModelKinds     map[string]int `json:"model_kinds"`
	KnownParams    int            `json:"known_parameters"`
	TotalDownloads int64          `json:"total_downloads"`
	TotalLikes     int64          `json:"total_likes"`
}

type compiledSelection struct {
	config selectionConfig
	from   time.Time
	to     time.Time
	regex  *regexp.Regexp
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
	if *outputOverride != "" {
		config.OutputDir = *outputOverride
	}
	applyCatalogDefaults(&config)
	return runCatalog(context.Background(), config)
}

func boolValue(value *bool, fallback bool) bool {
	if value == nil {
		return fallback
	}
	return *value
}

func applyCatalogDefaults(config *catalogConfig) {
	if config.Endpoint == "" {
		config.Endpoint = defaultEndpoint
	}
	if config.OutputDir == "" {
		config.OutputDir = "catalog-results"
	}
	if config.Scan.PageSize == 0 {
		config.Scan.PageSize = 1000
	}
	if config.Scan.TimeoutSeconds == 0 {
		config.Scan.TimeoutSeconds = 30
	}
	if config.Scan.Retries == nil {
		value := 5
		config.Scan.Retries = &value
	}
	if config.Owners.Workers == 0 {
		config.Owners.Workers = 8
	}
	for i := range config.Selections {
		if config.Selections[i].SortBy == "" {
			config.Selections[i].SortBy = "downloads"
		}
		if config.Selections[i].SortDirection == "" {
			config.Selections[i].SortDirection = "desc"
		}
	}
}

func parseFlexibleTime(value string, endOfDay bool) (time.Time, error) {
	if value == "" {
		return time.Time{}, nil
	}
	if parsed, err := time.Parse(time.RFC3339, value); err == nil {
		return parsed.UTC(), nil
	}
	parsed, err := time.Parse("2006-01-02", value)
	if err != nil {
		return time.Time{}, fmt.Errorf("invalid date %q: use YYYY-MM-DD or RFC3339", value)
	}
	if endOfDay {
		return parsed.Add(24*time.Hour - time.Nanosecond).UTC(), nil
	}
	return parsed.UTC(), nil
}

func compileSelections(config catalogConfig, now time.Time) ([]compiledSelection, time.Time, error) {
	if len(config.Selections) == 0 {
		return nil, time.Time{}, errors.New("config must contain at least one selection")
	}
	names := make(map[string]struct{})
	earliest := now
	result := make([]compiledSelection, 0, len(config.Selections))
	validKinds := map[string]bool{"base": true, "finetune": true, "adapter": true, "quantized": true, "merge": true}
	validOwnerTypes := map[string]bool{"user": true, "organization": true, "unknown": true}
	validSort := map[string]bool{"downloads": true, "likes": true, "created_at": true, "parameters": true, "repo_id": true}
	profiles := make(map[string]bool)
	for _, profile := range config.Compute {
		if profile.Name == "" || profiles[profile.Name] {
			return nil, time.Time{}, errors.New("compute profile names must be non-empty and unique")
		}
		if profile.GPUTFLOPS <= 0 || profile.Efficiency <= 0 || profile.Efficiency > 1 || profile.GPUHourCostUSD < 0 || profile.TokensPerParameter <= 0 || profile.Machines <= 0 || profile.GPUsPerMachine <= 0 || profile.FinetuneCostFraction < 0 {
			return nil, time.Time{}, fmt.Errorf("compute profile %q has invalid settings", profile.Name)
		}
		profiles[profile.Name] = true
	}
	for _, selection := range config.Selections {
		if selection.Name == "" {
			return nil, time.Time{}, errors.New("every selection needs a name")
		}
		if _, exists := names[selection.Name]; exists {
			return nil, time.Time{}, fmt.Errorf("duplicate selection name %q", selection.Name)
		}
		names[selection.Name] = struct{}{}
		if !validSort[selection.SortBy] {
			return nil, time.Time{}, fmt.Errorf("selection %q has unsupported sort_by %q", selection.Name, selection.SortBy)
		}
		if selection.SortDirection != "asc" && selection.SortDirection != "desc" {
			return nil, time.Time{}, fmt.Errorf("selection %q: sort_direction must be asc or desc", selection.Name)
		}
		if selection.LookbackDays < 0 || selection.Limit < 0 {
			return nil, time.Time{}, fmt.Errorf("selection %q: lookback_days and limit cannot be negative", selection.Name)
		}
		if selection.MinDownloads < 0 || selection.MaxDownloads < 0 || selection.MinLikes < 0 || selection.MaxLikes < 0 {
			return nil, time.Time{}, fmt.Errorf("selection %q: popularity bounds cannot be negative", selection.Name)
		}
		if (selection.MinParametersB != nil && *selection.MinParametersB < 0) || (selection.MaxParametersB != nil && *selection.MaxParametersB < 0) {
			return nil, time.Time{}, fmt.Errorf("selection %q: parameter bounds cannot be negative", selection.Name)
		}
		if selection.MinParametersB != nil && selection.MaxParametersB != nil && *selection.MinParametersB > *selection.MaxParametersB {
			return nil, time.Time{}, fmt.Errorf("selection %q: min_parameters_b exceeds max_parameters_b", selection.Name)
		}
		if selection.MaxDownloads > 0 && selection.MinDownloads > selection.MaxDownloads {
			return nil, time.Time{}, fmt.Errorf("selection %q: min_downloads exceeds max_downloads", selection.Name)
		}
		if selection.MaxLikes > 0 && selection.MinLikes > selection.MaxLikes {
			return nil, time.Time{}, fmt.Errorf("selection %q: min_likes exceeds max_likes", selection.Name)
		}
		for _, kind := range selection.ModelKinds {
			if !validKinds[kind] {
				return nil, time.Time{}, fmt.Errorf("selection %q has unsupported model kind %q", selection.Name, kind)
			}
		}
		for _, ownerType := range selection.OwnerTypes {
			if !validOwnerTypes[ownerType] {
				return nil, time.Time{}, fmt.Errorf("selection %q has unsupported owner type %q", selection.Name, ownerType)
			}
		}
		for _, profile := range selection.ComputeProfiles {
			if !profiles[profile] {
				return nil, time.Time{}, fmt.Errorf("selection %q references unknown compute profile %q", selection.Name, profile)
			}
		}
		from, err := parseFlexibleTime(selection.CreatedFrom, false)
		if err != nil {
			return nil, time.Time{}, fmt.Errorf("selection %q: %w", selection.Name, err)
		}
		to, err := parseFlexibleTime(selection.CreatedTo, true)
		if err != nil {
			return nil, time.Time{}, fmt.Errorf("selection %q: %w", selection.Name, err)
		}
		if to.IsZero() {
			to = now
		}
		if from.IsZero() {
			lookback := selection.LookbackDays
			if lookback == 0 {
				lookback = 365
			}
			from = to.Add(-time.Duration(lookback) * 24 * time.Hour)
		}
		if from.After(to) {
			return nil, time.Time{}, fmt.Errorf("selection %q: created_from is after created_to", selection.Name)
		}
		var expression *regexp.Regexp
		if selection.ModelIDRegex != "" {
			expression, err = regexp.Compile(selection.ModelIDRegex)
			if err != nil {
				return nil, time.Time{}, fmt.Errorf("selection %q model_id_regex: %w", selection.Name, err)
			}
		}
		result = append(result, compiledSelection{config: selection, from: from, to: to, regex: expression})
		if from.Before(earliest) {
			earliest = from
		}
	}
	return result, earliest, nil
}

func catalogAPIURL(endpoint string, pageSize int) string {
	query := url.Values{}
	query.Set("sort", "createdAt")
	query.Set("direction", "-1")
	query.Set("limit", strconv.Itoa(pageSize))
	for _, field := range []string{"author", "createdAt", "lastModified", "downloads", "likes", "pipeline_tag", "library_name", "tags", "siblings", "safetensors", "baseModels"} {
		query.Add("expand", field)
	}
	return strings.TrimRight(endpoint, "/") + "/api/models?" + query.Encode()
}

func catalogModelKind(model catalogModel) string {
	relation := strings.ToLower(model.BaseModels.Relation)
	if relation == "adapter" || relation == "finetune" || relation == "quantized" || relation == "merge" {
		return relation
	}
	tags := lowerSet(model.Tags)
	switch {
	case tags["gguf"] || tags["gptq"] || tags["awq"] || tags["exl2"] || tags["quantized"]:
		return "quantized"
	case tags["model-merge"] || tags["merge"]:
		return "merge"
	case tags["peft"] || tags["lora"] || tags["qlora"] || tagHasPrefix(tags, "base_model:adapter:") || catalogAdapterNameRE.MatchString(model.ID):
		return "adapter"
	case tagHasPrefix(tags, "base_model:finetune:") || catalogFinetuneNameRE.MatchString(model.ID):
		return "finetune"
	default:
		return "base"
	}
}

func tagHasPrefix(tags map[string]bool, prefix string) bool {
	for tag := range tags {
		if strings.HasPrefix(tag, prefix) {
			return true
		}
	}
	return false
}

func firstBaseModel(model catalogModel) string {
	if len(model.BaseModels.Models) == 0 {
		return ""
	}
	return model.BaseModels.Models[0].ID
}

func lowerSet(values []string) map[string]bool {
	result := make(map[string]bool, len(values))
	for _, value := range values {
		result[strings.ToLower(value)] = true
	}
	return result
}

func containsAny(set map[string]bool, values []string) bool {
	if len(values) == 0 {
		return true
	}
	for _, value := range values {
		if set[strings.ToLower(value)] {
			return true
		}
	}
	return false
}

func containsAll(set map[string]bool, values []string) bool {
	for _, value := range values {
		if !set[strings.ToLower(value)] {
			return false
		}
	}
	return true
}

func containsNone(set map[string]bool, values []string) bool {
	for _, value := range values {
		if set[strings.ToLower(value)] {
			return false
		}
	}
	return true
}

func effectiveParameters(model catalogModel, baseParameters map[string]int64) int64 {
	if catalogModelKind(model) == "adapter" {
		if value := baseParameters[firstBaseModel(model)]; value > 0 {
			return value
		}
	}
	return model.Safetensors.Total
}

func matchesSelection(model catalogModel, selection compiledSelection, effectiveParams int64) bool {
	created, err := time.Parse(time.RFC3339Nano, model.CreatedAt)
	if err != nil || created.Before(selection.from) || created.After(selection.to) {
		return false
	}
	config := selection.config
	if len(config.PipelineTags) > 0 && !containsAny(lowerSet([]string{model.PipelineTag}), config.PipelineTags) {
		return false
	}
	if len(config.Libraries) > 0 && !containsAny(lowerSet([]string{model.LibraryName}), config.Libraries) {
		return false
	}
	if len(config.ModelKinds) > 0 && !containsAny(lowerSet([]string{catalogModelKind(model)}), config.ModelKinds) {
		return false
	}
	tags := lowerSet(model.Tags)
	if !containsAny(tags, config.TagsAny) || !containsAll(tags, config.TagsAll) || !containsNone(tags, config.ExcludeTags) {
		return false
	}
	if selection.regex != nil && !selection.regex.MatchString(model.ID) {
		return false
	}
	if config.MinDownloads > 0 && model.Downloads < config.MinDownloads {
		return false
	}
	if config.MaxDownloads > 0 && model.Downloads > config.MaxDownloads {
		return false
	}
	if config.MinLikes > 0 && model.Likes < config.MinLikes {
		return false
	}
	if config.MaxLikes > 0 && model.Likes > config.MaxLikes {
		return false
	}
	if config.MinParametersB != nil || config.MaxParametersB != nil {
		if effectiveParams <= 0 {
			return config.IncludeUnknownParameters
		}
		billions := float64(effectiveParams) / 1e9
		if config.MinParametersB != nil && billions < *config.MinParametersB {
			return false
		}
		if config.MaxParametersB != nil && billions > *config.MaxParametersB {
			return false
		}
	}
	return true
}

func modelOwner(model catalogModel) string {
	if model.Author != "" {
		return model.Author
	}
	if slash := strings.IndexByte(model.ID, '/'); slash > 0 {
		return model.ID[:slash]
	}
	return ""
}

func modelToRecord(endpoint string, model catalogModel, selection selectionConfig, params int64, profiles map[string]computeProfile) catalogRecord {
	var billions *float64
	if params > 0 {
		value := float64(params) / 1e9
		billions = &value
	}
	record := catalogRecord{
		Selection: selection.Name, Description: selection.Description,
		RepoID: model.ID, RepoURL: strings.TrimRight(endpoint, "/") + "/" + model.ID,
		Owner: modelOwner(model), CreatedAt: model.CreatedAt, LastModified: model.LastModified,
		PipelineTag: model.PipelineTag, LibraryName: model.LibraryName,
		ModelKind: catalogModelKind(model), BaseModel: firstBaseModel(model),
		OwnParameters: model.Safetensors.Total, EffectiveParameters: params, ParametersB: billions,
		Downloads: model.Downloads, Likes: model.Likes, Tags: model.Tags,
	}
	for _, profileName := range selection.ComputeProfiles {
		if profile, ok := profiles[profileName]; ok && params > 0 {
			record.Compute = append(record.Compute, estimateCompute(params, catalogModelKind(model), profile))
		}
	}
	return record
}

func estimateCompute(parameters int64, kind string, profile computeProfile) computeEstimate {
	fraction := 1.0
	if kind == "finetune" || kind == "adapter" {
		fraction = profile.FinetuneCostFraction
	}
	n := float64(parameters)
	flops := 6 * n * (profile.TokensPerParameter * n) * fraction
	gpuHours := flops / (profile.GPUTFLOPS * 1e12 * 3600 * profile.Efficiency)
	totalGPUs := profile.Machines * profile.GPUsPerMachine
	return computeEstimate{
		Profile: profile.Name, GPUName: profile.GPUName, Machines: profile.Machines,
		TotalGPUs: totalGPUs, GPUHours: gpuHours, WallDays: gpuHours / float64(totalGPUs) / 24,
		CostUSD: gpuHours * profile.GPUHourCostUSD, Fraction: fraction,
	}
}

func selectionLess(a, b catalogRecord, config selectionConfig) bool {
	var less bool
	switch config.SortBy {
	case "likes":
		less = a.Likes < b.Likes
	case "created_at":
		less = a.CreatedAt < b.CreatedAt
	case "parameters":
		less = a.EffectiveParameters < b.EffectiveParameters
	case "repo_id":
		less = a.RepoID < b.RepoID
	default:
		less = a.Downloads < b.Downloads
	}
	if config.SortDirection == "desc" {
		return !less && sortValueDifferent(a, b, config.SortBy)
	}
	return less
}

func sortValueDifferent(a, b catalogRecord, field string) bool {
	switch field {
	case "likes":
		return a.Likes != b.Likes
	case "created_at":
		return a.CreatedAt != b.CreatedAt
	case "parameters":
		return a.EffectiveParameters != b.EffectiveParameters
	case "repo_id":
		return a.RepoID != b.RepoID
	default:
		return a.Downloads != b.Downloads
	}
}

func fetchBaseParameters(ctx context.Context, client *httpClient, endpoint string, ids []string, workers int) map[string]int64 {
	result := make(map[string]int64)
	jobs := make(chan string)
	type response struct {
		id    string
		total int64
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
				if err != nil || len(body) == 0 {
					responses <- response{id: id}
					continue
				}
				var info struct {
					Safetensors catalogSafetensors `json:"safetensors"`
				}
				_ = json.Unmarshal(body, &info)
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
	}
	return result
}

func escapeRepoID(id string) string {
	parts := strings.Split(id, "/")
	for i := range parts {
		parts[i] = url.PathEscape(parts[i])
	}
	return strings.Join(parts, "/")
}

func fetchOwners(ctx context.Context, client *httpClient, endpoint string, names []string, workers int) map[string]*ownerOverview {
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
				orgURL := strings.TrimRight(endpoint, "/") + "/api/organizations/" + url.PathEscape(name) + "/overview"
				body, _, err := client.get(ctx, orgURL, true)
				if err == nil && len(body) > 0 && json.Unmarshal(body, overview) == nil {
					overview.Type = "organization"
					responses <- response{name, overview}
					continue
				}
				userURL := strings.TrimRight(endpoint, "/") + "/api/users/" + url.PathEscape(name) + "/overview"
				body, _, err = client.get(ctx, userURL, true)
				if err == nil && len(body) > 0 && json.Unmarshal(body, overview) == nil {
					overview.Type = "user"
					if overview.Name == "" {
						overview.Name = name
					}
					responses <- response{name, overview}
					continue
				}
				if err != nil {
					overview.Error = err.Error()
				} else {
					overview.Error = "profile not found"
				}
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

func runCatalog(ctx context.Context, config catalogConfig) error {
	if config.Scan.PageSize < 1 || config.Scan.PageSize > 1000 || config.Scan.MaxPages < 0 || config.Scan.TimeoutSeconds <= 0 || config.Owners.Workers < 1 || config.Owners.Workers > 64 || config.Scan.Retries == nil || *config.Scan.Retries < 0 {
		return errors.New("invalid page_size, max_pages, or owner workers")
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
	client := &httpClient{client: &http.Client{Timeout: time.Duration(config.Scan.TimeoutSeconds) * time.Second}, token: os.Getenv("HF_TOKEN"), retries: *config.Scan.Retries}
	address := catalogAPIURL(config.Endpoint, config.Scan.PageSize)
	models := make([]catalogModel, 0)
	pages, scanned := 0, 0
	complete := false
	for address != "" {
		body, headers, err := client.get(ctx, address, false)
		if err != nil {
			return err
		}
		var page []catalogModel
		if err := json.Unmarshal(body, &page); err != nil {
			return err
		}
		if len(page) == 0 {
			complete = true
			break
		}
		pages++
		scanned += len(page)
		reachedOlder := false
		for _, model := range page {
			created, err := time.Parse(time.RFC3339Nano, model.CreatedAt)
			if err != nil {
				continue
			}
			if created.Before(earliest) {
				reachedOlder = true
				continue
			}
			if created.After(now) {
				continue
			}
			if boolValue(config.Scan.RequireWeights, true) && model.Safetensors.Total <= 0 && !hasRecognizedWeight(model.Siblings) {
				continue
			}
			models = append(models, model)
		}
		if !config.Scan.Quiet {
			fmt.Fprintf(os.Stderr, "catalog pages=%d scanned=%d retained=%d\n", pages, scanned, len(models))
		}
		if reachedOlder {
			complete = true
			break
		}
		if config.Scan.MaxPages > 0 && pages >= config.Scan.MaxPages {
			break
		}
		address = nextLink(headers)
		if address == "" {
			complete = true
		}
	}

	baseSet := make(map[string]bool)
	if boolValue(config.Scan.ResolveBaseParameters, true) {
		for _, model := range models {
			if catalogModelKind(model) != "adapter" {
				continue
			}
			for _, selection := range selections {
				if selection.config.MinParametersB == nil && selection.config.MaxParametersB == nil {
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
	baseParams := fetchBaseParameters(ctx, client, config.Endpoint, baseIDs, config.Owners.Workers)

	var records []catalogRecord
	computeProfiles := make(map[string]computeProfile, len(config.Compute))
	for _, profile := range config.Compute {
		computeProfiles[profile.Name] = profile
	}
	for _, selection := range selections {
		var selected []catalogRecord
		for _, model := range models {
			params := effectiveParameters(model, baseParams)
			if matchesSelection(model, selection, params) {
				selected = append(selected, modelToRecord(config.Endpoint, model, selection.config, params, computeProfiles))
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
		owners = fetchOwners(ctx, client, config.Endpoint, ownerNames, config.Owners.Workers)
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

	if err := os.MkdirAll(config.OutputDir, 0o755); err != nil {
		return err
	}
	if err := writeCatalogModels(config.OutputDir, records); err != nil {
		return err
	}
	if err := writeCatalogOwners(config.OutputDir, owners); err != nil {
		return err
	}
	summary := buildCatalogSummary(now, complete, pages, scanned, records, config.Selections)
	data, _ := json.MarshalIndent(summary, "", "  ")
	if err := os.WriteFile(filepath.Join(config.OutputDir, "summary.json"), append(data, '\n'), 0o644); err != nil {
		return err
	}
	fmt.Printf("Catalog complete=%t pages=%d scanned=%d selected=%d owners=%d\n", complete, pages, scanned, len(records), len(owners))
	fmt.Printf("Output: %s\n", config.OutputDir)
	return nil
}

func writeCatalogModels(directory string, records []catalogRecord) error {
	data, _ := json.MarshalIndent(records, "", "  ")
	if err := os.WriteFile(filepath.Join(directory, "models.json"), append(data, '\n'), 0o644); err != nil {
		return err
	}
	file, err := os.Create(filepath.Join(directory, "models.csv"))
	if err != nil {
		return err
	}
	defer file.Close()
	w := csv.NewWriter(file)
	defer w.Flush()
	_ = w.Write([]string{"selection", "repo_id", "repo_url", "owner", "owner_type", "created_at", "last_modified", "pipeline_tag", "library_name", "model_kind", "base_model", "own_parameters", "effective_parameters", "parameters_b", "downloads", "likes", "tags", "compute_estimates_json"})
	for _, r := range records {
		p := ""
		if r.ParametersB != nil {
			p = strconv.FormatFloat(*r.ParametersB, 'f', 6, 64)
		}
		computeJSON, _ := json.Marshal(r.Compute)
		_ = w.Write([]string{r.Selection, r.RepoID, r.RepoURL, r.Owner, r.OwnerType, r.CreatedAt, r.LastModified, r.PipelineTag, r.LibraryName, r.ModelKind, r.BaseModel, strconv.FormatInt(r.OwnParameters, 10), strconv.FormatInt(r.EffectiveParameters, 10), p, strconv.FormatInt(r.Downloads, 10), strconv.FormatInt(r.Likes, 10), strings.Join(r.Tags, "|"), string(computeJSON)})
	}
	return w.Error()
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
	data, _ := json.MarshalIndent(list, "", "  ")
	if err := os.WriteFile(filepath.Join(directory, "owners.json"), append(data, '\n'), 0o644); err != nil {
		return err
	}
	file, err := os.Create(filepath.Join(directory, "owners.csv"))
	if err != nil {
		return err
	}
	defer file.Close()
	w := csv.NewWriter(file)
	defer w.Flush()
	_ = w.Write([]string{"owner", "owner_type", "profile_url", "fullname", "verified", "pro", "plan", "followers", "models", "datasets", "spaces", "papers", "members", "created_at", "details", "selected_repos", "selected_downloads", "selected_likes", "selections", "error"})
	for _, o := range list {
		_ = w.Write([]string{o.Name, o.Type, "https://huggingface.co/" + o.Name, o.Fullname, strconv.FormatBool(o.IsVerified), strconv.FormatBool(o.IsPro), o.Plan, strconv.FormatInt(o.NumFollowers, 10), strconv.FormatInt(o.NumModels, 10), strconv.FormatInt(o.NumDatasets, 10), strconv.FormatInt(o.NumSpaces, 10), strconv.FormatInt(o.NumPapers, 10), strconv.FormatInt(o.NumUsers, 10), o.CreatedAt, o.Details, strconv.FormatInt(o.SelectedRepos, 10), strconv.FormatInt(o.Downloads, 10), strconv.FormatInt(o.Likes, 10), strings.Join(o.Selections, "|"), o.Error})
	}
	return w.Error()
}

func buildCatalogSummary(now time.Time, complete bool, pages, scanned int, records []catalogRecord, selections []selectionConfig) catalogSummary {
	result := catalogSummary{GeneratedAt: time.Now().UTC().Format(time.RFC3339Nano), EffectiveNow: now.Format(time.RFC3339Nano), Complete: complete, PagesScanned: pages, ModelsScanned: scanned, Selections: make(map[string]selectionSummary)}
	for _, selection := range selections {
		result.Selections[selection.Name] = selectionSummary{
			Description: selection.Description,
			OwnerTypes:  make(map[string]int),
			ModelKinds:  make(map[string]int),
		}
	}
	owners := make(map[string]map[string]bool)
	for _, record := range records {
		s := result.Selections[record.Selection]
		s.Description = record.Description
		s.Models++
		s.TotalDownloads += record.Downloads
		s.TotalLikes += record.Likes
		if s.OwnerTypes == nil {
			s.OwnerTypes = make(map[string]int)
		}
		s.OwnerTypes[record.OwnerType]++
		if s.ModelKinds == nil {
			s.ModelKinds = make(map[string]int)
		}
		s.ModelKinds[record.ModelKind]++
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

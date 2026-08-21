package main

import (
	"context"
	"encoding/base64"
	"encoding/csv"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"
)

type catalogConfig struct {
	Endpoint   string               `json:"endpoint"`
	OutputDir  string               `json:"output_dir"`
	Scan       catalogScanConfig    `json:"scan"`
	Logging    catalogLoggingConfig `json:"logging"`
	Owners     ownerConfig          `json:"owners"`
	Compute    []computeProfile     `json:"compute_profiles"`
	Selections []selectionConfig    `json:"selections"`
}

type catalogScanConfig struct {
	Now                    string   `json:"now"`
	PageSize               int      `json:"page_size"`
	MaxPages               int      `json:"max_pages"`
	RequireWeights         *bool    `json:"require_weights"`
	ResolveBaseParameters  *bool    `json:"resolve_base_parameters"`
	ResolveReportedCompute *bool    `json:"resolve_reported_compute"`
	ResolveScratchClaims   *bool    `json:"resolve_scratch_claims"`
	FetchIndividualReadmes *bool    `json:"fetch_individual_readmes"`
	CatalogCheckpoint      string   `json:"catalog_checkpoint"`
	BulkModelCardsURL      string   `json:"bulk_model_cards_url"`
	BulkModelCardsURLs     []string `json:"bulk_model_cards_urls"`
	BulkModelCardsFile     string   `json:"bulk_model_cards_file"`
	TimeoutSeconds         int      `json:"timeout_seconds"`
	Retries                *int     `json:"retries"`
	Quiet                  bool     `json:"quiet"`
}

type ownerConfig struct {
	Enabled *bool `json:"enabled"`
	Workers int   `json:"workers"`
}

type selectionConfig struct {
	Name                             string   `json:"name"`
	Description                      string   `json:"description"`
	LookbackDays                     int      `json:"lookback_days"`
	CreatedFrom                      string   `json:"created_from"`
	CreatedTo                        string   `json:"created_to"`
	PipelineTags                     []string `json:"pipeline_tags"`
	Libraries                        []string `json:"libraries"`
	ModelKinds                       []string `json:"model_kinds"`
	TargetLLMOnly                    bool     `json:"target_llm_only"`
	OwnerTypes                       []string `json:"owner_types"`
	TagsAny                          []string `json:"tags_any"`
	TagsAll                          []string `json:"tags_all"`
	ExcludeTags                      []string `json:"exclude_tags"`
	ModelIDRegex                     string   `json:"model_id_regex"`
	MinParametersB                   *float64 `json:"min_parameters_b"`
	MaxParametersB                   *float64 `json:"max_parameters_b"`
	IncludeUnknownParameters         bool     `json:"include_unknown_parameters"`
	MinDownloads                     int64    `json:"min_downloads"`
	MaxDownloads                     int64    `json:"max_downloads"`
	MinLikes                         int64    `json:"min_likes"`
	MaxLikes                         int64    `json:"max_likes"`
	SortBy                           string   `json:"sort_by"`
	SortDirection                    string   `json:"sort_direction"`
	Limit                            int      `json:"limit"`
	ComputeProfiles                  []string `json:"compute_profiles"`
	RequireReportedDerivativeCompute bool     `json:"require_reported_derivative_compute"`
	RequireReportedCompute           bool     `json:"require_reported_compute"`
	RequireExplicitScratchForBase    bool     `json:"require_explicit_scratch_for_base"`
	FormulaRequiresExplicitScratch   bool     `json:"formula_requires_explicit_scratch"`
	CountReportedDerivativeCosts     bool     `json:"count_reported_derivative_costs"`
	DeclaredBaseMinLikes             int64    `json:"declared_base_min_likes"`
	DeclaredBaseMinDownloads         int64    `json:"declared_base_min_downloads"`
}

type computeProfile struct {
	Name                 string  `json:"name"`
	Description          string  `json:"description"`
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
	Method    string  `json:"method,omitempty"`
	Source    string  `json:"source,omitempty"`
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
	CardData     map[string]any     `json:"cardData"`
}

type catalogRecord struct {
	Selection           string                   `json:"selection"`
	Description         string                   `json:"selection_description,omitempty"`
	RepoID              string                   `json:"repo_id"`
	RepoURL             string                   `json:"repo_url"`
	Owner               string                   `json:"owner"`
	OwnerType           string                   `json:"owner_type"`
	CreatedAt           string                   `json:"created_at"`
	LastModified        string                   `json:"last_modified"`
	PipelineTag         string                   `json:"pipeline_tag"`
	LibraryName         string                   `json:"library_name"`
	ModelKind           string                   `json:"model_kind"`
	BaseModel           string                   `json:"base_model,omitempty"`
	OwnParameters       int64                    `json:"own_parameters,omitempty"`
	EffectiveParameters int64                    `json:"effective_parameters,omitempty"`
	ParametersB         *float64                 `json:"parameters_b,omitempty"`
	Downloads           int64                    `json:"downloads"`
	Likes               int64                    `json:"likes"`
	Tags                []string                 `json:"tags"`
	Compute             []computeEstimate        `json:"compute_estimates,omitempty"`
	ReportedCompute     *reportedTrainingCompute `json:"reported_training_compute,omitempty"`
	ScratchClaim        *reportedScratchClaim    `json:"scratch_training_claim,omitempty"`
	TrainingCostUSD     *float64                 `json:"training_cost_usd,omitempty"`
	TrainingCostMethod  string                   `json:"training_cost_method,omitempty"`
	Errors              []string                 `json:"errors,omitempty"`
}

type ownerOverview struct {
	Name               string   `json:"name"`
	User               string   `json:"user"`
	Type               string   `json:"type"`
	Fullname           string   `json:"fullname"`
	AvatarURL          string   `json:"avatarUrl"`
	Details            string   `json:"details"`
	Plan               string   `json:"plan"`
	IsVerified         bool     `json:"isVerified"`
	IsPro              bool     `json:"isPro"`
	NumUsers           int64    `json:"numUsers"`
	NumModels          int64    `json:"numModels"`
	NumDatasets        int64    `json:"numDatasets"`
	NumSpaces          int64    `json:"numSpaces"`
	NumPapers          int64    `json:"numPapers"`
	NumFollowers       int64    `json:"numFollowers"`
	CreatedAt          string   `json:"createdAt"`
	Error              string   `json:"error,omitempty"`
	SelectedRepos      int64    `json:"selected_repos"`
	Downloads          int64    `json:"selected_downloads"`
	Likes              int64    `json:"selected_likes"`
	KnownTrainingCosts int64    `json:"known_training_costs"`
	TrainingCostUSD    float64  `json:"training_cost_usd"`
	Selections         []string `json:"selections"`
}

var (
	catalogAdapterNameRE         = regexp.MustCompile(`(?i)(^|[-_./])(lora|qlora|adapter|peft)([-_./]|$)`)
	catalogFinetuneNameRE        = regexp.MustCompile(`(?i)(^|[-_./])(finetuned?[0-9]*|fine[-_]?tuned?[0-9]*|sft|dpo|cpt|distilled?|instruct(?:ed)?|inst|chat|thinking|thinker|reasoning|align(?:ed|ment)?|adapt(?:ed|ation)?|post[-_]?train(?:ed|ing)?|grpo|rl)([-_./]|$)`)
	catalogForkNameRE            = regexp.MustCompile(`(?i)(^|[-_./])(abliterat(?:ed|ion)|ablation|uncensored|unfiltered|ungated|text[-_]?only|unlocked|heretic|blasphemer|repack(?:ed)?|reupload(?:ed)?|mirror|qwenified|adjust(?:ed)?|dequantized?|pruned|sliced|upscaled|patched)([-_./]|$)`)
	catalogMergeNameRE           = regexp.MustCompile(`(?i)(^|[-_./])(merge|merged)([-_./]|$)`)
	catalogQuantNameRE           = regexp.MustCompile(`(?i)(^|[-_./])(gguf|ggml|gptq|awq|exl[2-9]|quantized?|qat|int[248]|[248][-_]?bits?|fp8|bnb)([-_./]|$)`)
	catalogConvertNameRE         = regexp.MustCompile(`(?i)(^|[-_./])(convert|converted|conversion|mlx|onnx|openvino|tensorrt|coreml)([-_./]|$)`)
	catalogUUIDRepoNameRE        = regexp.MustCompile(`(?i)/[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$`)
	catalogEmbeddedSourceNameRE  = regexp.MustCompile(`/[^/]+_-_[^/]+`)
	catalogHFConversionNameRE    = regexp.MustCompile(`(?i)(?:[-_.](?:hf|transformers?))$`)
	catalogAffineArtifactRE      = regexp.MustCompile(`(?i)/affine(?:[-_.]|$)`)
	catalogKnownFamilyRE         = regexp.MustCompile(`(?i)(^|[-_./])(qwen|llama|gemma|mistral|mixtral|deepseek|glm|phi[0-9.-]*|bert|roberta|t5|olmo|falcon|mamba|rwkv|nemotron|granite|exaone|command-r)([-_./0-9]|$)`)
	nonTextPipelineRE            = regexp.MustCompile(`(?i)(image|video|audio|speech|vision|depth|segmentation|object-detection|text-to-3d|feature-extraction|reinforcement-learning|robotics)`)
	nonTextModelTagRE            = regexp.MustCompile(`(?i)(diffusers?|stable[-_ ]?diffusion|flux|sdxl|text-to-image|image-to-image|image-generation|video-generation|text-to-video|computer-vision|vision-language|multimodal|\bvlm\b|(?:^|[-_])vl(?:$|[-_])|qwen[^ ]*[-_]vl|llava|bailingmm|univision|naturelm|emu3(?:\.5)?|chemdfm|audio|speech|whisper|musicgen|bark|vocoder|tts|protein|genomic|\bdna\b|dnagpt|bacformer|molecule|smiles|chemberta)`)
	llmSignalRE                  = regexp.MustCompile(`(?i)(^|[-_./ ])(llm|causal[-_ ]?lm|language[-_ ]?model|text[-_ ]?generation|text2text|conversational|qwen|llama|gemma|mistral|mixtral|deepseek|glm|phi[0-9.-]*|bert|roberta|t5|olmo|falcon|mamba|rwkv|nemotron|granite|exaone|command[-_ ]?r)([-_./ 0-9]|$)`)
	catalogCheckpointRE          = regexp.MustCompile(`(?i)([-_.](?:checkpoint|ckpt|step|stage)[-_.]?[0-9]+(?:\.[0-9]+)?k?)(?:[-_.].*)?$`)
	reportedDerivativeEvidenceRE = regexp.MustCompile(`(?i)\b(fine[- ]?tun(?:e|ed|ing)|instruction[- ]?tun(?:e|ed|ing)|post[- ]?train(?:ed|ing)|continued pretraining|continual pretraining|domain adaptation|distill(?:ed|ation)|\bSFT\b|\bDPO\b|\bRLHF\b|\bGRPO\b)\b`)
	reportedAdapterEvidenceRE    = regexp.MustCompile(`(?i)\b(LoRA|QLoRA|PEFT|adapter)\b`)
	catalogKnownKindOverrides    = map[string]string{
		"aiqtech/LLaDA2.0-flash-preview":                                      "finetune",
		"nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16":                          "finetune",
		"OpenKing/vualtgemma-1b-non-gated":                                    "fork",
		"LSX-UniWue/LLaMmlein_120M":                                           "fork",
		"LSX-UniWue/LLaMmlein_1B":                                             "fork",
		"microsoft/bitnet-b1.58-2B-4T":                                        "quantized",
		"vngrs-ai/Kumru-2B-Base":                                              "finetune",
		"deepseek-ai/DeepSeek-V3.2-Exp-Base":                                  "finetune",
		"ScienceOne-AI/S1-Base-671B":                                          "finetune",
		"ScienceOne-AI/S1-Base-32B":                                           "finetune",
		"keatone/air-base-lacuna":                                             "fork",
		"gradients-io-tournaments/Covenant-Base":                              "fork",
		"supermanaff/Affine-5FC1Dq1kdHAGmrEkSCLwEKeNM7i9YY6rXtZKaLM2q4qaAE6b": "fork",
		"nev8r/SmolLM3-3B-Custom-Base":                                        "fork",
		"mokshahf/CosmuQuantaa":                                               "finetune",
		"ByteDance-Seed/Seed-OSS-36B-Base-woSyn":                              "fork",
		"common-pile/comma-v0.1-1t":                                           "fork",
	}
)

type catalogSummary struct {
	GeneratedAt                string                      `json:"generated_at"`
	EffectiveNow               string                      `json:"effective_now"`
	Complete                   bool                        `json:"complete"`
	PagesScanned               int                         `json:"pages_scanned"`
	ModelsScanned              int                         `json:"models_scanned"`
	ReportedComputeCandidates  int                         `json:"reported_compute_candidates"`
	ReportedComputeResolved    int                         `json:"reported_compute_resolved"`
	ReportedComputeIgnored     int                         `json:"reported_compute_ignored"`
	ScratchClaimCandidates     int                         `json:"scratch_claim_candidates"`
	ScratchClaimsResolved      int                         `json:"scratch_claims_resolved"`
	ScratchClaimsIgnored       int                         `json:"scratch_claims_ignored"`
	RetainedWeightRepositories int                         `json:"retained_weight_repositories"`
	Selections                 map[string]selectionSummary `json:"selections"`
}

type selectionSummary struct {
	Description          string             `json:"description,omitempty"`
	Models               int                `json:"models"`
	UniqueOwners         int                `json:"unique_owners"`
	OwnerTypes           map[string]int     `json:"owner_types"`
	ModelKinds           map[string]int     `json:"model_kinds"`
	KnownParams          int                `json:"known_parameters"`
	TotalDownloads       int64              `json:"total_downloads"`
	TotalLikes           int64              `json:"total_likes"`
	KnownTrainingCosts   int                `json:"known_training_costs"`
	TotalTrainingCostUSD float64            `json:"total_training_cost_usd"`
	Formula20HybridUSD   float64            `json:"formula_20tpp_moe_active_hybrid_total_usd"`
	Formula20LiteralUSD  float64            `json:"formula_20tpp_literal_total_parameters_total_usd"`
	CostedModelsByKind   map[string]int     `json:"costed_models_by_kind"`
	TrainingCostByKind   map[string]float64 `json:"training_cost_by_kind_usd"`
	CostedModelsByMethod map[string]int     `json:"costed_models_by_method"`
	TrainingCostByMethod map[string]float64 `json:"training_cost_by_method_usd"`
	BaseModelsBySize     map[string]int     `json:"base_models_by_parameter_range"`
	CostedBaseBySize     map[string]int     `json:"costed_base_models_by_parameter_range"`
	BaseCostBySize       map[string]float64 `json:"base_training_cost_by_parameter_range_usd"`
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
	if config.Logging.Level == "" {
		config.Logging.Level = "info"
	}
	if config.Logging.File == "" {
		config.Logging.File = "catalog.log.jsonl"
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
	if config.Scan.ResolveReportedCompute == nil {
		value := false
		config.Scan.ResolveReportedCompute = &value
	}
	if config.Scan.FetchIndividualReadmes == nil {
		value := true
		config.Scan.FetchIndividualReadmes = &value
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
	validKinds := map[string]bool{"base": true, "fork": true, "finetune": true, "adapter": true, "quantized": true, "merge": true}
	validOwnerTypes := map[string]bool{"user": true, "organization": true, "unknown": true}
	validSort := map[string]bool{"downloads": true, "likes": true, "created_at": true, "parameters": true, "repo_id": true}
	profiles := make(map[string]bool)
	for _, profile := range config.Compute {
		if profile.Name == "" || profiles[profile.Name] {
			return nil, time.Time{}, errors.New("compute profile names must be non-empty and unique")
		}
		if profile.GPUTFLOPS <= 0 || profile.Efficiency <= 0 || profile.Efficiency > 1 || profile.GPUHourCostUSD < 0 || profile.TokensPerParameter <= 0 || profile.Machines <= 0 || profile.GPUsPerMachine <= 0 || profile.FinetuneCostFraction < 0 || profile.FinetuneCostFraction > 1 {
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

func catalogAPIURL(endpoint string, pageSize int, before time.Time) string {
	query := url.Values{}
	query.Set("sort", "createdAt")
	query.Set("direction", "-1")
	query.Set("limit", strconv.Itoa(pageSize))
	for _, field := range []string{"author", "createdAt", "lastModified", "downloads", "likes", "pipeline_tag", "library_name", "tags", "siblings", "safetensors", "baseModels", "cardData"} {
		query.Add("expand", field)
	}
	if !before.IsZero() {
		// Hub pagination cursors are URL-safe base64 JSON and use the Mongo-style
		// object id, whose first four bytes are the creation Unix timestamp. Jump
		// directly to the selection's exclusive upper bound instead of walking
		// through every newer repository.
		seconds := before.UTC().Unix() + 1
		objectID := fmt.Sprintf("%08x%s", seconds, strings.Repeat("f", 16))
		payload := fmt.Sprintf(`{"_id":{"$lt":"%s"}}`, objectID)
		query.Set("cursor", base64.RawURLEncoding.EncodeToString([]byte(payload)))
	}
	return strings.TrimRight(endpoint, "/") + "/api/models?" + query.Encode()
}

func catalogModelKind(model catalogModel) string {
	if kind := catalogKnownKindOverrides[model.ID]; kind != "" {
		return kind
	}
	relation := strings.ToLower(model.BaseModels.Relation)
	if relation == "quantized" || relation == "merge" {
		return relation
	}
	tags := lowerSet(model.Tags)
	library := strings.ToLower(model.LibraryName)
	switch {
	case tags["gguf"] || tags["ggml"] || tags["gptq"] || tags["awq"] || tags["exl2"] || tags["quantized"] || tags["compressed-tensors"] || tags["bitsandbytes"] || tags["4-bit"] || tags["8-bit"] || library == "mlx" || library == "onnx" || library == "openvino" || catalogQuantNameRE.MatchString(model.ID) || catalogConvertNameRE.MatchString(model.ID) || catalogEmbeddedSourceNameRE.MatchString(model.ID) || catalogHFConversionNameRE.MatchString(model.ID):
		return "quantized"
	case tags["model-merge"] || tags["merge"] || catalogMergeNameRE.MatchString(model.ID):
		return "merge"
	case relation == "adapter" || relation == "finetune":
		return relation
	case tags["generated_from_trainer"] || tags["trl"] || tags["sft"] || tags["dpo"] || tags["grpo"] || tags["rlhf"] || tags["rl-swarm"] || tags["genrl-swarm"] || tags["unsloth"]:
		return "finetune"
	case tags["peft"] || tags["lora"] || tags["qlora"] || tagHasPrefix(tags, "base_model:adapter:") || catalogAdapterNameRE.MatchString(model.ID):
		return "adapter"
	case tagHasPrefix(tags, "base_model:finetune:") || tagHasPrefix(tags, "base_model:") || len(model.BaseModels.Models) > 0 || cardDataDeclaresBaseModel(model.CardData):
		return "finetune"
	case catalogForkNameRE.MatchString(model.ID) || catalogCheckpointRE.MatchString(model.ID) || catalogUUIDRepoNameRE.MatchString(model.ID) || catalogAffineArtifactRE.MatchString(model.ID):
		return "fork"
	case catalogFinetuneNameRE.MatchString(model.ID):
		return "finetune"
	default:
		return "base"
	}
}

func catalogIsTargetLLM(model catalogModel) bool {
	pipeline := strings.ToLower(model.PipelineTag)
	if nonTextPipelineRE.MatchString(pipeline) {
		return false
	}
	joined := strings.ToLower(model.ID + " " + strings.Join(model.Tags, " "))
	// A text-generation pipeline only describes the inference interface. Models
	// over DNA, proteins and other non-language sequences can expose it too, so
	// subject-domain exclusions must run before accepting the pipeline.
	if nonTextModelTagRE.MatchString(joined) {
		return false
	}
	for _, accepted := range []string{"text-generation", "text2text-generation", "conversational", "fill-mask"} {
		if pipeline == accepted {
			return true
		}
	}
	return llmSignalRE.MatchString(joined)
}

// The requested accounting treats instruct/chat/SFT-style repositories as
// derivatives: their full-pretraining formula must not be charged merely
// because the card also describes an underlying scratch stage. Only residual
// base repositories are eligible for the parameter formula.
func catalogScratchOverrideEligible(model catalogModel) bool {
	return catalogModelKind(model) == "base"
}

func resolvedCatalogModelKind(model catalogModel, scratch *reportedScratchClaim, reported *reportedTrainingCompute) string {
	kind := catalogModelKind(model)
	if scratch != nil && catalogScratchOverrideEligible(model) {
		return "base"
	}
	// Hub metadata is frequently missing. A compute disclosure that explicitly
	// describes fine-tuning/CPT/adapter work is stronger than the residual
	// default "base" classification.
	if kind == "base" && reported != nil {
		if reportedAdapterEvidenceRE.MatchString(reported.Evidence) {
			return "adapter"
		}
		if reportedDerivativeEvidenceRE.MatchString(reported.Evidence) {
			return "finetune"
		}
	}
	return kind
}

func cardDataDeclaresBaseModel(card map[string]any) bool {
	for _, key := range []string{"base_model", "baseModel", "base_models"} {
		if value, ok := card[key]; ok && nonEmpty(value) {
			return true
		}
	}
	return false
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

func effectiveParameters(model catalogModel, baseParameters map[string]int64, baseErrors map[string]string) (int64, string) {
	if catalogModelKind(model) == "adapter" {
		baseID := firstBaseModel(model)
		if baseID == "" {
			return 0, "adapter has no declared base model"
		}
		if value := baseParameters[baseID]; value > 0 {
			return value, ""
		}
		if message := baseErrors[baseID]; message != "" {
			return 0, message
		}
		return 0, "base model parameter count was not resolved"
	}
	return model.Safetensors.Total, ""
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
	if config.TargetLLMOnly && !catalogIsTargetLLM(model) {
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

var baseArtifactSuffixRE = regexp.MustCompile(`(?i)(?:[-_.](?:bf16|fp16|fp8|f16|pt|paddle|hf|transformers?|safetensors))+$`)
var reportedTrajectoryDateSuffixRE = regexp.MustCompile(`(?i)[-_.](?:20)?\d{6,8}$`)

func canonicalBaseFamilyName(repoID string) string {
	name := repoID
	if slash := strings.LastIndexByte(name, '/'); slash >= 0 {
		name = name[slash+1:]
	}
	name = strings.ToLower(name)
	name = baseArtifactSuffixRE.ReplaceAllString(name, "")
	name = strings.NewReplacer("_", "-", ".", "-").Replace(name)
	return strings.Trim(name, "-")
}

func modelToRecord(endpoint string, model catalogModel, selection selectionConfig, params int64, parameterError string, profiles map[string]computeProfile, reported *reportedTrainingCompute, scratch *reportedScratchClaim) catalogRecord {
	var billions *float64
	if params > 0 {
		value := float64(params) / 1e9
		billions = &value
	}
	kind := resolvedCatalogModelKind(model, scratch, reported)
	record := catalogRecord{
		Selection: selection.Name, Description: selection.Description,
		RepoID: model.ID, RepoURL: strings.TrimRight(endpoint, "/") + "/" + model.ID,
		Owner: modelOwner(model), CreatedAt: model.CreatedAt, LastModified: model.LastModified,
		PipelineTag: model.PipelineTag, LibraryName: model.LibraryName,
		ModelKind: kind, BaseModel: firstBaseModel(model),
		OwnParameters: model.Safetensors.Total, EffectiveParameters: params, ParametersB: billions,
		Downloads: model.Downloads, Likes: model.Likes, Tags: model.Tags,
		ScratchClaim: scratch,
	}
	if parameterError != "" {
		record.Errors = append(record.Errors, parameterError)
	}
	record.ReportedCompute = reported
	if record.ModelKind != "base" {
		if !selection.CountReportedDerivativeCosts || reported == nil || (record.ModelKind != "finetune" && record.ModelKind != "adapter") {
			return record
		}
		cost := reported.CostUSD
		record.TrainingCostUSD = &cost
		record.TrainingCostMethod = reported.CostMethod
		record.Compute = append(record.Compute, computeEstimate{
			Profile: "reported_training_time", GPUName: reported.GPUName,
			TotalGPUs: reported.GPUCount, GPUHours: reported.GPUHours,
			WallDays: reported.Hours / 24, CostUSD: reported.CostUSD,
			Fraction: 1, Method: reported.CostMethod, Source: reported.Source,
		})
		return record
	}
	formulaRequiresScratch := selection.RequireExplicitScratchForBase || selection.FormulaRequiresExplicitScratch
	if formulaRequiresScratch && scratch == nil {
		return record
	}
	if scratch != nil && scratch.EvidenceType == "declared_base_model" && model.Likes < selection.DeclaredBaseMinLikes && model.Downloads < selection.DeclaredBaseMinDownloads {
		record.Errors = append(record.Errors, "declared base checkpoint lacks independent pretraining evidence and minimum market engagement")
		return record
	}
	for _, profileName := range selection.ComputeProfiles {
		if profile, ok := profiles[profileName]; ok && params > 0 {
			literal := estimateCompute(params, kind, profile)
			literal.Method = "formula_txt_literal_total_parameters"
			literal.Source = "formula.txt"
			estimate := literal
			if formulaRequiresScratch && scratch != nil {
				estimate = estimateBaseTrainingCompute(params, scratch, profile)
			}
			if math.IsNaN(estimate.GPUHours) || math.IsInf(estimate.GPUHours, 0) || math.IsNaN(estimate.WallDays) || math.IsInf(estimate.WallDays, 0) || math.IsNaN(estimate.CostUSD) || math.IsInf(estimate.CostUSD, 0) {
				record.Errors = append(record.Errors, "compute estimate overflow for profile "+profileName)
				continue
			}
			if estimate.Method != literal.Method || math.Abs(estimate.CostUSD-literal.CostUSD) > 0.005 {
				record.Compute = append(record.Compute, literal)
			}
			record.Compute = append(record.Compute, estimate)
			if record.TrainingCostUSD == nil {
				cost := estimate.CostUSD
				record.TrainingCostUSD = &cost
				record.TrainingCostMethod = estimate.Method
				if record.TrainingCostMethod == "" {
					record.TrainingCostMethod = "parameter_scaling_estimate"
				}
			}
		}
	}
	return record
}

func isBaseTrainingCostMethod(method string) bool {
	return strings.HasPrefix(method, "formula_txt_") || strings.HasPrefix(method, "reported_pretraining_tokens_")
}

// deduplicateReportedTrainingRuns prevents copied model cards, conversions and
// mirrors from charging the same disclosed run repeatedly. The oldest repo is
// preferred as the canonical publication; popularity breaks timestamp ties.
// This is deliberately conservative: identical cards may occasionally have
// been reused for separate runs, so the resulting sum is a documented lower
// bound rather than an invented point estimate.
func deduplicateReportedTrainingRuns(records []catalogRecord) {
	groups := make(map[string][]int)
	for index := range records {
		reported := records[index].ReportedCompute
		if reported == nil || records[index].TrainingCostUSD == nil || (records[index].TrainingCostMethod != "gpu_hours_x_hourly_rate" && records[index].TrainingCostMethod != "reported_total_cost") {
			continue
		}
		fingerprint := reported.RunHash
		if fingerprint == "" {
			fingerprint = reported.EvidenceHash
		}
		if fingerprint == "" {
			continue
		}
		key := records[index].Selection + "\x00" + fingerprint
		groups[key] = append(groups[key], index)
	}
	for _, indices := range groups {
		if len(indices) < 2 {
			continue
		}
		canonical := indices[0]
		for _, index := range indices[1:] {
			a, b := records[index], records[canonical]
			if a.CreatedAt < b.CreatedAt || (a.CreatedAt == b.CreatedAt && (a.Likes > b.Likes || (a.Likes == b.Likes && a.Downloads > b.Downloads))) {
				canonical = index
			}
		}
		for _, index := range indices {
			if index == canonical {
				continue
			}
			records[index].TrainingCostUSD = nil
			records[index].TrainingCostMethod = "duplicate_reported_training_run"
			records[index].Compute = nil
			records[index].Errors = append(records[index].Errors, "cost counted once under canonical repository "+records[canonical].RepoID)
		}
	}

	// Some projects publish a dated snapshot beside an undated repository and
	// report cumulative compute in the newer card. Exact evidence hashes cannot
	// join those snapshots because the cumulative total changed. Collapse only
	// a very narrow same-owner/same-kind/same-base/same-parameter family where
	// at least one basename ends in an explicit YYYYMMDD-like date, and retain
	// the largest disclosed cumulative cost.
	type trajectoryGroup struct {
		indices []int
		dated   bool
	}
	trajectories := make(map[string]*trajectoryGroup)
	for index := range records {
		record := records[index]
		if record.ReportedCompute == nil || record.TrainingCostUSD == nil || (record.ModelKind != "finetune" && record.ModelKind != "adapter") {
			continue
		}
		parts := strings.SplitN(record.RepoID, "/", 2)
		if len(parts) != 2 {
			continue
		}
		name := strings.ToLower(parts[1])
		dated := reportedTrajectoryDateSuffixRE.MatchString(name)
		family := reportedTrajectoryDateSuffixRE.ReplaceAllString(name, "")
		key := record.Selection + "\x00" + strings.ToLower(record.Owner) + "\x00" + record.ModelKind + "\x00" + strings.ToLower(record.BaseModel) + "\x00" + family + "\x00" + strconv.FormatInt(record.EffectiveParameters, 10)
		group := trajectories[key]
		if group == nil {
			group = &trajectoryGroup{}
			trajectories[key] = group
		}
		group.indices = append(group.indices, index)
		group.dated = group.dated || dated
	}
	for _, group := range trajectories {
		if !group.dated || len(group.indices) < 2 {
			continue
		}
		canonical := group.indices[0]
		for _, index := range group.indices[1:] {
			a, b := records[index], records[canonical]
			if *a.TrainingCostUSD > *b.TrainingCostUSD || (*a.TrainingCostUSD == *b.TrainingCostUSD && a.CreatedAt > b.CreatedAt) {
				canonical = index
			}
		}
		for _, index := range group.indices {
			if index == canonical || records[index].TrainingCostUSD == nil {
				continue
			}
			records[index].TrainingCostUSD = nil
			records[index].TrainingCostMethod = "duplicate_reported_training_trajectory"
			records[index].Compute = nil
			records[index].Errors = append(records[index].Errors, "reported trajectory cost counted once under cumulative repository "+records[canonical].RepoID)
		}
	}
}

func deduplicateReportedDerivativeAgainstBase(records []catalogRecord) {
	baseRuns := make(map[string]string)
	for index := range records {
		reported := records[index].ReportedCompute
		if records[index].ModelKind != "base" || records[index].TrainingCostUSD == nil || reported == nil || reported.RunHash == "" {
			continue
		}
		baseRuns[records[index].Selection+"\x00"+reported.RunHash] = records[index].RepoID
	}
	for index := range records {
		if records[index].ModelKind != "finetune" && records[index].ModelKind != "adapter" {
			continue
		}
		reported := records[index].ReportedCompute
		if records[index].TrainingCostUSD == nil || reported == nil || reported.RunHash == "" {
			continue
		}
		if canonical := baseRuns[records[index].Selection+"\x00"+reported.RunHash]; canonical != "" {
			records[index].TrainingCostUSD = nil
			records[index].TrainingCostMethod = "duplicate_reported_base_training_run"
			records[index].Compute = nil
			records[index].Errors = append(records[index].Errors, "reported compute belongs to canonical base repository "+canonical)
		}
	}
}

func deduplicateScratchTrainingRuns(records []catalogRecord) {
	groups := make(map[string][]int)
	for index := range records {
		claim := records[index].ScratchClaim
		if claim == nil || records[index].TrainingCostUSD == nil || !isBaseTrainingCostMethod(records[index].TrainingCostMethod) {
			continue
		}
		fingerprint := claim.RunHash
		if fingerprint == "" {
			fingerprint = claim.CardHash
		}
		if fingerprint == "" {
			continue
		}
		key := records[index].Selection + "\x00" + fingerprint + "\x00" + strconv.FormatInt(records[index].EffectiveParameters, 10)
		groups[key] = append(groups[key], index)
	}
	for _, indices := range groups {
		if len(indices) < 2 {
			continue
		}
		canonical := indices[0]
		for _, index := range indices[1:] {
			a, b := records[index], records[canonical]
			if a.CreatedAt < b.CreatedAt || (a.CreatedAt == b.CreatedAt && (a.Likes > b.Likes || (a.Likes == b.Likes && a.Downloads > b.Downloads))) {
				canonical = index
			}
		}
		for _, index := range indices {
			if index == canonical {
				continue
			}
			records[index].TrainingCostUSD = nil
			records[index].TrainingCostMethod = "duplicate_scratch_training_run"
			records[index].Compute = nil
			records[index].Errors = append(records[index].Errors, "scratch cost counted once under canonical repository "+records[canonical].RepoID)
		}
	}

	// Repositories frequently publish every checkpoint of one trajectory with
	// slightly different generated cards. Exact hashes cannot join those, so a
	// second, deliberately narrow key strips only an explicit checkpoint/step
	// suffix and still requires the same owner and exact parameter count.
	trajectories := make(map[string][]int)
	for index := range records {
		if records[index].TrainingCostUSD == nil || !isBaseTrainingCostMethod(records[index].TrainingCostMethod) {
			continue
		}
		parts := strings.SplitN(records[index].RepoID, "/", 2)
		if len(parts) != 2 || !catalogCheckpointRE.MatchString(parts[1]) {
			continue
		}
		family := catalogCheckpointRE.ReplaceAllString(strings.ToLower(parts[1]), "")
		key := records[index].Selection + "\x00" + strings.ToLower(records[index].Owner) + "\x00" + family + "\x00" + strconv.FormatInt(records[index].EffectiveParameters, 10)
		trajectories[key] = append(trajectories[key], index)
	}
	for _, indices := range trajectories {
		if len(indices) < 2 {
			continue
		}
		canonical := indices[0]
		for _, index := range indices[1:] {
			a, b := records[index], records[canonical]
			if a.CreatedAt < b.CreatedAt || (a.CreatedAt == b.CreatedAt && (a.Likes > b.Likes || (a.Likes == b.Likes && a.Downloads > b.Downloads))) {
				canonical = index
			}
		}
		for _, index := range indices {
			if index == canonical {
				continue
			}
			records[index].TrainingCostUSD = nil
			records[index].TrainingCostMethod = "duplicate_scratch_training_trajectory"
			records[index].Compute = nil
			records[index].Errors = append(records[index].Errors, "scratch trajectory cost counted once under canonical repository "+records[canonical].RepoID)
		}
	}

	// Cross-owner mirrors commonly preserve the upstream repository basename
	// while changing only the namespace or serialization suffix. Collapse only
	// exact normalized basename + exact parameter-count matches. Popularity wins
	// because a mirror may have been uploaded slightly before the canonical org.
	families := make(map[string][]int)
	for index := range records {
		if records[index].TrainingCostUSD == nil || !isBaseTrainingCostMethod(records[index].TrainingCostMethod) {
			continue
		}
		family := canonicalBaseFamilyName(records[index].RepoID)
		if len(family) < 6 || family == "model" || family == "base-model" {
			continue
		}
		key := records[index].Selection + "\x00" + family + "\x00" + strconv.FormatInt(records[index].EffectiveParameters, 10)
		families[key] = append(families[key], index)
	}
	for _, indices := range families {
		if len(indices) < 2 {
			continue
		}
		canonical := indices[0]
		for _, index := range indices[1:] {
			a, b := records[index], records[canonical]
			if a.Likes > b.Likes || (a.Likes == b.Likes && (a.Downloads > b.Downloads || (a.Downloads == b.Downloads && a.CreatedAt < b.CreatedAt))) {
				canonical = index
			}
		}
		for _, index := range indices {
			if index == canonical {
				continue
			}
			records[index].TrainingCostUSD = nil
			records[index].TrainingCostMethod = "duplicate_base_family_mirror"
			records[index].Compute = nil
			records[index].Errors = append(records[index].Errors, "base-family cost counted once under canonical repository "+records[canonical].RepoID)
		}
	}
}

func reportedComputeFallsInSelection(compute *reportedTrainingCompute, selection compiledSelection) bool {
	if compute == nil {
		return true
	}
	if len(compute.TrainingYears) > 0 {
		fromYear, toYear := selection.from.Year(), selection.to.Year()
		for _, year := range compute.TrainingYears {
			if year >= fromYear && year <= toYear {
				return true
			}
		}
		return false
	}
	if compute.EarliestCardCreatedAt != "" {
		if earliest, err := time.Parse(time.RFC3339Nano, compute.EarliestCardCreatedAt); err == nil && earliest.Before(selection.from) {
			return false
		}
	}
	return true
}

func scratchClaimFallsInSelection(claim *reportedScratchClaim, selection compiledSelection) bool {
	if claim == nil {
		return true
	}
	if len(claim.TrainingYears) > 0 {
		fromYear, toYear := selection.from.Year(), selection.to.Year()
		for _, year := range claim.TrainingYears {
			if year >= fromYear && year <= toYear {
				return true
			}
		}
		return false
	}
	if claim.EarliestCardCreatedAt != "" {
		if earliest, err := time.Parse(time.RFC3339Nano, claim.EarliestCardCreatedAt); err == nil && earliest.Before(selection.from) {
			return false
		}
	}
	return true
}

func normalizeIdentity(value string) string {
	var result strings.Builder
	for _, r := range strings.ToLower(value) {
		if unicode.IsLetter(r) || unicode.IsNumber(r) {
			result.WriteRune(r)
		}
	}
	return result.String()
}

func scratchClaimMatchesOwner(model catalogModel, claim *reportedScratchClaim) bool {
	if claim == nil || claim.ClaimedTrainer == "" {
		return true
	}
	trainer := normalizeIdentity(claim.ClaimedTrainer)
	owner := normalizeIdentity(modelOwner(model))
	if len(trainer) < 3 || len(owner) < 2 {
		return true
	}
	return strings.Contains(owner, trainer) || strings.Contains(trainer, owner)
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

// estimateBaseTrainingCompute keeps formula.txt's 6*N*T accounting while using
// stronger public evidence when the card supplies it. For sparse/MoE models N
// is the activated parameter count; T is the reported pretraining token count.
// Missing values fall back independently to total parameters and 20 tokens per
// total parameter, respectively.
func estimateBaseTrainingCompute(parameters int64, claim *reportedScratchClaim, profile computeProfile) computeEstimate {
	totalParameters := float64(parameters)
	computeParameters := totalParameters
	method := "formula_txt_parameter_scaling"
	if claim != nil && claim.ActiveParametersB > 0 && claim.ActiveParametersB*1e9 < totalParameters {
		computeParameters = claim.ActiveParametersB * 1e9
		method = "formula_txt_moe_active_parameter_scaling"
	}
	tokens := profile.TokensPerParameter * totalParameters
	if claim != nil && claim.TrainingTokensT > 0 {
		tokens = claim.TrainingTokensT * 1e12
		if computeParameters < totalParameters {
			method = "reported_pretraining_tokens_active_parameters"
		} else {
			method = "reported_pretraining_tokens_total_parameters"
		}
	}
	flops := 6 * computeParameters * tokens
	gpuHours := flops / (profile.GPUTFLOPS * 1e12 * 3600 * profile.Efficiency)
	totalGPUs := profile.Machines * profile.GPUsPerMachine
	return computeEstimate{
		Profile: profile.Name, GPUName: profile.GPUName, Machines: profile.Machines,
		TotalGPUs: totalGPUs, GPUHours: gpuHours, WallDays: gpuHours / float64(totalGPUs) / 24,
		CostUSD: gpuHours * profile.GPUHourCostUSD, Fraction: 1,
		Method: method, Source: "formula.txt plus public model-card training evidence",
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

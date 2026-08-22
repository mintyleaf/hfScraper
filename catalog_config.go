package main

import (
	"fmt"
	"regexp"
	"time"
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
	DiffusionImageBudget float64 `json:"diffusion_image_budget,omitempty"`
	LatentSequenceLength float64 `json:"latent_sequence_length,omitempty"`
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

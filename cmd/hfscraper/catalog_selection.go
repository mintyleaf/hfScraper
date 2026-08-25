package main

import (
	"encoding/base64"
	"errors"
	"fmt"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"time"
)

type compiledSelection struct {
	config selectionConfig
	from   time.Time
	to     time.Time
	regex  *regexp.Regexp
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
		if profile.GPUTFLOPS <= 0 || profile.Efficiency <= 0 || profile.Efficiency > 1 || profile.GPUHourCostUSD < 0 || profile.TokensPerParameter <= 0 || profile.Machines <= 0 || profile.GPUsPerMachine <= 0 || profile.FinetuneCostFraction < 0 || profile.FinetuneCostFraction > 1 || profile.DiffusionImageBudget < 0 || profile.LatentSequenceLength < 0 {
			return nil, time.Time{}, fmt.Errorf("compute profile %q has invalid settings", profile.Name)
		}
		profiles[profile.Name] = true
	}
	for _, selection := range config.Selections {
		if selection.Name == "" {
			return nil, time.Time{}, errors.New("every selection needs a name")
		}
		if selection.TargetLLMOnly && selection.TargetDiffusionOnly {
			return nil, time.Time{}, fmt.Errorf("selection %q cannot enable both target_llm_only and target_diffusion_only", selection.Name)
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
	if config.TargetDiffusionOnly && !catalogIsTargetDiffusion(model) {
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

func tagHasPrefix(tags map[string]bool, prefix string) bool {
	for tag := range tags {
		if strings.HasPrefix(tag, prefix) {
			return true
		}
	}
	return false
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

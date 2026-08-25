package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

type preLLMBaseCandidate struct {
	Selection   string   `json:"selection"`
	RepoID      string   `json:"repo_id"`
	Owner       string   `json:"owner"`
	CreatedAt   string   `json:"created_at"`
	PipelineTag string   `json:"pipeline_tag"`
	LibraryName string   `json:"library_name"`
	Parameters  int64    `json:"parameters"`
	Downloads   int64    `json:"downloads"`
	Likes       int64    `json:"likes"`
	Tags        []string `json:"tags"`
}

type preLLMSelectionSummary struct {
	Selection               string         `json:"selection"`
	Repositories            int            `json:"repositories"`
	PreliminaryKinds        map[string]int `json:"preliminary_kinds"`
	BaseCandidates          int            `json:"base_candidates"`
	BaseWithKnownParameters int            `json:"base_with_known_parameters"`
	BaseWithoutParameters   int            `json:"base_without_parameters"`
	CandidateJSON           string         `json:"base_candidates_json"`
	CandidateCSV            string         `json:"base_candidates_csv"`
}

func preLLMFileComponent(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	var out strings.Builder
	lastDash := false
	for _, r := range value {
		if r >= 'a' && r <= 'z' || r >= '0' && r <= '9' {
			out.WriteRune(r)
			lastDash = false
		} else if !lastDash {
			out.WriteByte('-')
			lastDash = true
		}
	}
	result := strings.Trim(out.String(), "-")
	if result == "" {
		return "selection"
	}
	return result
}

func writePreLLMBaseCandidates(outputDir string, models []catalogModel, selections []compiledSelection, baseParams map[string]int64, baseErrors map[string]string, logger *catalogLogger) error {
	summaries := make([]preLLMSelectionSummary, 0, len(selections))
	for _, selection := range selections {
		summary := preLLMSelectionSummary{Selection: selection.config.Name, PreliminaryKinds: make(map[string]int)}
		var candidates []preLLMBaseCandidate
		for _, model := range models {
			params, _ := effectiveParameters(model, baseParams, baseErrors)
			if !matchesSelection(model, selection, params) {
				continue
			}
			if selection.config.TargetLLMOnly && isDiffusionModel(model) {
				continue
			}
			kind := catalogModelKind(model)
			summary.Repositories++
			summary.PreliminaryKinds[kind]++
			if kind != "base" {
				continue
			}
			summary.BaseCandidates++
			if params > 0 {
				summary.BaseWithKnownParameters++
			} else {
				summary.BaseWithoutParameters++
			}
			candidates = append(candidates, preLLMBaseCandidate{
				Selection: selection.config.Name, RepoID: model.ID, Owner: modelOwner(model),
				CreatedAt: model.CreatedAt, PipelineTag: model.PipelineTag, LibraryName: model.LibraryName,
				Parameters: params, Downloads: model.Downloads, Likes: model.Likes, Tags: model.Tags,
			})
		}
		sort.Slice(candidates, func(i, j int) bool { return candidates[i].RepoID < candidates[j].RepoID })
		component := preLLMFileComponent(selection.config.Name)
		jsonName := "pre-llm-" + component + "-base-candidates.json"
		csvName := "pre-llm-" + component + "-base-candidates.csv"
		summary.CandidateJSON, summary.CandidateCSV = jsonName, csvName
		if err := writeJSONFile(filepath.Join(outputDir, jsonName), candidates); err != nil {
			return err
		}
		if err := writePreLLMBaseCandidateCSV(filepath.Join(outputDir, csvName), candidates); err != nil {
			return err
		}
		logger.info("pre_llm_candidates_ready", "preliminary API-only base candidates saved before model-card/LLM processing", map[string]any{
			"selection": summary.Selection, "repositories": summary.Repositories, "base_candidates": summary.BaseCandidates,
			"base_with_known_parameters": summary.BaseWithKnownParameters, "base_without_parameters": summary.BaseWithoutParameters,
			"json": jsonName, "csv": csvName,
		})
		fmt.Printf("Pre-LLM [%s]: repositories=%d preliminary_base=%d base_with_parameters=%d base_without_parameters=%d\n", summary.Selection, summary.Repositories, summary.BaseCandidates, summary.BaseWithKnownParameters, summary.BaseWithoutParameters)
		summaries = append(summaries, summary)
	}
	return writeJSONFile(filepath.Join(outputDir, "pre-llm-summary.json"), summaries)
}

func writePreLLMBaseCandidateCSV(path string, candidates []preLLMBaseCandidate) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create %q: %w", path, err)
	}
	w := csv.NewWriter(file)
	if err := w.Write([]string{"selection", "repo_id", "owner", "created_at", "pipeline_tag", "library_name", "parameters", "downloads", "likes", "tags"}); err != nil {
		return closeFileAfterError(file, path, err)
	}
	for _, candidate := range candidates {
		if err := w.Write([]string{candidate.Selection, candidate.RepoID, candidate.Owner, candidate.CreatedAt, candidate.PipelineTag, candidate.LibraryName, strconv.FormatInt(candidate.Parameters, 10), strconv.FormatInt(candidate.Downloads, 10), strconv.FormatInt(candidate.Likes, 10), strings.Join(candidate.Tags, "|")}); err != nil {
			return closeFileAfterError(file, path, err)
		}
	}
	w.Flush()
	if err := w.Error(); err != nil {
		return closeFileAfterError(file, path, err)
	}
	if err := file.Sync(); err != nil {
		return closeFileAfterError(file, path, err)
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("close %q: %w", path, err)
	}
	return nil
}

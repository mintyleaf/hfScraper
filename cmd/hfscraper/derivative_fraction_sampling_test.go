package main

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestDerivativeFractionPromptRequestsUpstreamSize(t *testing.T) {
	specialized := localLLMSystemPrompt("text_llm_derivative_fraction")
	if !strings.Contains(specialized, "FULL upstream/base checkpoint") || !strings.Contains(specialized, "never the adapter-file size") {
		t.Fatalf("derivative fraction prompt does not request the full upstream size: %q", specialized)
	}
	ordinary := localLLMSystemPrompt("text_llm")
	if strings.Contains(ordinary, "never the adapter-file size") {
		t.Fatalf("ordinary market prompt unexpectedly changed parameter semantics: %q", ordinary)
	}
}

func TestDerivativeEvidenceCanonicalizesMarkdownPunctuation(t *testing.T) {
	card := "- **Finetuned from model :** TouchNight/Ministral-8B-Instruct-2410-HF\n"
	review := localLLMReview{
		Kind:       "finetune",
		Confidence: "high",
		Evidence:   "Finetuned from model: TouchNight/Ministral-8B-Instruct-2410-HF",
	}
	if err := validateLocalLLMReview(&review, card); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(card, review.Evidence) {
		t.Fatalf("canonical evidence %q is not an exact card substring", review.Evidence)
	}
}

func TestDerivativeEvidenceStillRejectsParaphrase(t *testing.T) {
	card := "This repository contains a serialized checkpoint."
	review := localLLMReview{Kind: "finetune", Confidence: "high", Evidence: "This model was fine-tuned from org/base."}
	if err := validateLocalLLMReview(&review, card); err == nil {
		t.Fatal("paraphrased evidence was accepted")
	}
}

func TestDerivativeReviewCorrectsRejectedEvidence(t *testing.T) {
	card := "This model was fine-tuned from org/base-7b."
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		evidence := "This is a fine-tune of org/base-7b."
		if calls == 2 {
			evidence = card
		}
		content, _ := json.Marshal(localLLMReview{Kind: "finetune", Confidence: "high", Evidence: evidence})
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"message": map[string]any{"content": string(content)}}}})
	}))
	defer server.Close()

	review, err := callLocalLLM(context.Background(), server.Client(), localLLMConfig{
		Scope: "text_llm_derivative_fraction", BaseURL: server.URL + "/v1", Model: "test",
	}, localLLMReviewInput{Card: card})
	if err != nil {
		t.Fatal(err)
	}
	if calls != 2 || review.Evidence != card {
		t.Fatalf("calls=%d review=%#v", calls, review)
	}
}

func TestParameterBillionsFromName(t *testing.T) {
	for _, test := range []struct {
		value string
		want  float64
	}{
		{"Qwen/Qwen2.5-14B-Instruct", 14},
		{"owner/model-1.5b-lora", 1.5},
		{"owner/not-a-size", 0},
	} {
		if got := parameterBillionsFromName(test.value); got != test.want {
			t.Errorf("parameterBillionsFromName(%q) = %v, want %v", test.value, got, test.want)
		}
	}
}

func TestBuildDerivativeFractionResults(t *testing.T) {
	textProfile := computeProfile{Name: "text", GPUTFLOPS: 989, Efficiency: 0.4, GPUHourCostUSD: 1.85, TokensPerParameter: 20, Machines: 1, GPUsPerMachine: 1}
	diffusionProfile := computeProfile{Name: "diffusion", GPUTFLOPS: 989, Efficiency: 0.4, GPUHourCostUSD: 1.85, TokensPerParameter: 20, Machines: 1, GPUsPerMachine: 1, DiffusionImageBudget: 3, LatentSequenceLength: 1024}
	config := catalogConfig{Derivative: derivativeSamplingConfig{BaseCostFraction: 0.01}, Compute: []computeProfile{textProfile, diffusionProfile}}
	manifest := []derivativeManifestEntry{
		{RepoID: "org/text-a", Market: "text_llm", Kind: "finetune", Parameters: 7_000_000_000, ParameterSource: "repository_safetensors", SamplingWeight: 2},
		{RepoID: "org/text-medium", Market: "text_llm", Kind: "adapter", SamplingWeight: 3},
		{RepoID: "org/image-a", Market: "diffusion", Kind: "adapter", Parameters: 1_000_000_000, ParameterSource: "declared_base_model_safetensors", SamplingWeight: 4},
		{RepoID: "org/quant", Market: "text_llm", Kind: "finetune", Parameters: 7_000_000_000, SamplingWeight: 5},
		{RepoID: "org/text-copy", Market: "text_llm", Kind: "finetune", Parameters: 7_000_000_000, SamplingWeight: 6},
	}
	reviews := map[string]localLLMReview{
		"org/text-a":      {Kind: "finetune", Confidence: "high", CanonicalTrainingRun: "org/run-a"},
		"org/text-medium": {Kind: "adapter", Confidence: "medium", ReportedParametersB: 14, CanonicalTrainingRun: "org/run-b"},
		"org/image-a":     {Kind: "adapter", Confidence: "high", CanonicalTrainingRun: "org/image-run"},
		"org/quant":       {Kind: "quantized", Confidence: "high", CanonicalTrainingRun: "org/quant"},
		"org/text-copy":   {Kind: "finetune", Confidence: "high", CanonicalTrainingRun: "org/run-a"},
	}

	results, err := buildDerivativeFractionResults(config, manifest, reviews)
	if err != nil {
		t.Fatal(err)
	}
	text7Base := estimateCompute(7_000_000_000, "base", textProfile).CostUSD
	if !near(results[0].WeightedCentralCostUSD, text7Base*0.01*2) || !near(results[0].WeightedUpperCostUSD, text7Base*0.01*2) {
		t.Fatalf("high-confidence text cost = central %v upper %v", results[0].WeightedCentralCostUSD, results[0].WeightedUpperCostUSD)
	}
	text14Base := estimateCompute(14_000_000_000, "base", textProfile).CostUSD
	if results[1].WeightedCentralCostUSD != 0 || !near(results[1].WeightedUpperCostUSD, text14Base*0.01*3) || results[1].ParametersUsed != 14_000_000_000 {
		t.Fatalf("medium-confidence adapter result = %#v", results[1])
	}
	imageBase := estimateDiffusionBaseTrainingCompute(1_000_000_000, nil, diffusionProfile).CostUSD
	if !near(results[2].WeightedCentralCostUSD, imageBase*0.01*4) {
		t.Fatalf("diffusion cost = %v, want %v", results[2].WeightedCentralCostUSD, imageBase*0.01*4)
	}
	if results[3].WeightedUpperCostUSD != 0 || results[3].ExclusionReason != "not a training derivative" {
		t.Fatalf("quantization was not excluded: %#v", results[3])
	}
	if results[4].WeightedUpperCostUSD != 0 || results[4].DuplicateOf != "org/text-a" {
		t.Fatalf("canonical duplicate was not zeroed: %#v", results[4])
	}
}

func TestBuildDerivativeCandidatesIncludesTextAndDiffusion(t *testing.T) {
	from := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)
	to := time.Date(2025, 12, 31, 23, 59, 59, 0, time.UTC)
	models := []catalogModel{
		{ID: "org/text-lora", CreatedAt: "2025-04-01T00:00:00Z", PipelineTag: "text-generation", Tags: []string{"transformers", "lora"}},
		{ID: "org/sdxl-lora", CreatedAt: "2025-04-01T00:00:00Z", PipelineTag: "text-to-image", Tags: []string{"diffusers", "lora"}},
	}
	selections := []compiledSelection{
		{config: selectionConfig{TargetLLMOnly: true, ModelKinds: []string{"adapter"}}, from: from, to: to},
		{config: selectionConfig{TargetDiffusionOnly: true, ModelKinds: []string{"adapter"}}, from: from, to: to},
	}

	got := buildDerivativeCandidates(models, selections)
	if len(got) != 2 || got[0].Market != "diffusion" || got[1].Market != "text_llm" {
		t.Fatalf("candidates = %#v", got)
	}
}

func near(got, want float64) bool {
	return math.Abs(got-want) <= math.Max(1e-9, math.Abs(want)*1e-12)
}

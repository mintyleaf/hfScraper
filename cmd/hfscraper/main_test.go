package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/parquet-go/parquet-go"
)

func testModel() model {
	return model{
		ID:       "author/original-model",
		Tags:     []string{"transformers"},
		CardData: map[string]any{},
		Siblings: []sibling{{Filename: "model.safetensors"}, {Filename: "README.md"}},
	}
}

func TestPlainWeightRepoNeedsREADME(t *testing.T) {
	if got := metadataFilter(testModel()); got != nil {
		t.Fatalf("metadataFilter() = %#v, want nil", got)
	}
}

func TestBaseModelExcluded(t *testing.T) {
	m := testModel()
	m.CardData["base_model"] = "google/bert-base"
	got := metadataFilter(m)
	if got == nil || got.Reason != "declares_base_model" {
		t.Fatalf("metadataFilter() = %#v", got)
	}
}

func TestAdapterHasNoFullWeights(t *testing.T) {
	m := testModel()
	m.Siblings = []sibling{{Filename: "adapter_model.safetensors"}}
	got := metadataFilter(m)
	if got == nil || got.Reason != "adapter_files" {
		t.Fatalf("metadataFilter() = %#v", got)
	}
}

func TestQuantizedNameExcluded(t *testing.T) {
	m := testModel()
	m.ID = "author/model-GGUF"
	got := metadataFilter(m)
	if got == nil || got.Reason != "derivative_name" {
		t.Fatalf("metadataFilter() = %#v", got)
	}
}

func TestTokenizerBinIsNotWeight(t *testing.T) {
	m := testModel()
	m.Siblings = []sibling{{Filename: "tokenizer.bin"}}
	got := metadataFilter(m)
	if got == nil || got.Reason != "no_recognized_full_weight_file" {
		t.Fatalf("metadataFilter() = %#v", got)
	}
}

func TestExpandedWeightFormats(t *testing.T) {
	files := []string{
		"unet/diffusion_pytorch_model.safetensors",
		"encoder/model.onnx",
		"saved_model.pb",
		"model.tflite",
		"keras_model.h5",
		"flax_weights.msgpack",
		"rust_model.ot",
		"model.pdparams",
		"classifier.joblib",
		"weights.npz",
		"model.ckpt.data-00000-of-00001",
	}
	for _, filename := range files {
		t.Run(filename, func(t *testing.T) {
			if !hasRecognizedWeight([]sibling{{Filename: filename}}) {
				t.Fatalf("%q was not recognized as a weight file", filename)
			}
		})
	}
}

func TestTrainerArtifactsAreNotWeights(t *testing.T) {
	files := []string{"tokenizer.bin", "tokenizer_model.npz", "optimizer.pt", "scheduler.pt", "rng_state_0.pth", "training_args.bin"}
	for _, filename := range files {
		t.Run(filename, func(t *testing.T) {
			if hasRecognizedWeight([]sibling{{Filename: filename}}) {
				t.Fatalf("%q was incorrectly recognized as a weight file", filename)
			}
		})
	}
}

func TestREADMEClassification(t *testing.T) {
	tests := []struct {
		name, text, want string
	}{
		{"positive", "We pretrained this language model from scratch on 50B tokens.", "confirmed"},
		{"random init", "The network was trained from a random initialization.", "confirmed"},
		{"negated", "This model was not trained from scratch; it uses BERT.", "excluded"},
		{"fine tune", "This is a fine-tuned model trained from scratch for a toy task.", "excluded"},
		{"tokenizer", "We trained the tokenizer from scratch, then used BERT weights.", "excluded"},
		{"no claim", "A useful language model.", "candidate"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := classifyREADME(tt.text); got.Status != tt.want {
				t.Fatalf("classifyREADME() status = %q, want %q (%#v)", got.Status, tt.want, got)
			}
		})
	}
}

func TestNextLink(t *testing.T) {
	header := http.Header{"Link": []string{`<https://example/api?cursor=x>; rel="next"`}}
	if got := nextLink(header); got != "https://example/api?cursor=x" {
		t.Fatalf("nextLink() = %q", got)
	}
}

func TestCatalogAPIURLJumpsToSelectionBoundary(t *testing.T) {
	boundary := time.Date(2025, 12, 31, 23, 59, 59, 0, time.UTC)
	address := catalogAPIURL("https://example.test", 1000, boundary)
	parsed, err := url.Parse(address)
	if err != nil {
		t.Fatal(err)
	}
	payload, err := base64.RawURLEncoding.DecodeString(parsed.Query().Get("cursor"))
	if err != nil {
		t.Fatal(err)
	}
	want := `{"_id":{"$lt":"6955b900ffffffffffffffff"}}`
	if string(payload) != want {
		t.Fatalf("cursor payload = %q, want %q", payload, want)
	}
}

func TestParseYears(t *testing.T) {
	got, err := parseYears("2026, 2025,2026")
	if err != nil || len(got) != 2 || got[0] != 2025 || got[1] != 2026 {
		t.Fatalf("parseYears() = %v, %v", got, err)
	}
}

func TestTruncateKeepsUTF8Valid(t *testing.T) {
	if got := truncate("модель", 3); got != "мод" {
		t.Fatalf("truncate() = %q", got)
	}
}

func TestCatalogModelKind(t *testing.T) {
	tests := []struct {
		id, relation, want string
		tags               []string
	}{
		{"org/model", "", "base", nil},
		{"org/model-lora", "", "adapter", nil},
		{"org/model-sft", "", "finetune", nil},
		{"org/model-GGUF", "", "quantized", []string{"gguf"}},
		{"org/model", "adapter", "adapter", nil},
		{"org/model", "", "finetune", []string{"base_model:finetune:org/base"}},
		{"org/model-merged", "", "merge", nil},
		{"org/model-converted", "", "quantized", nil},
		{"org/model-FP8", "", "quantized", nil},
		{"org/model-QAT", "", "quantized", nil},
		{"org/model-finetuned", "", "finetune", nil},
	}
	for _, tt := range tests {
		model := catalogModel{ID: tt.id, Tags: tt.tags, BaseModels: catalogBaseModels{Relation: tt.relation}}
		if got := catalogModelKind(model); got != tt.want {
			t.Errorf("catalogModelKind(%q) = %q, want %q", tt.id, got, tt.want)
		}
	}
}

func TestCatalogModelKindTreatsDeclaredBaseAsDerivative(t *testing.T) {
	model := catalogModel{ID: "org/innocent-name", CardData: map[string]any{"base_model": "upstream/base"}}
	if got := catalogModelKind(model); got != "finetune" {
		t.Fatalf("catalogModelKind() = %q, want finetune", got)
	}
}

func TestGeneratedTrainerCardForKnownUpstreamFamilyIsDerivative(t *testing.T) {
	model := catalogModel{ID: "user/qwen2_5_omni_audio_only", Tags: []string{"generated_from_trainer"}}
	if got := catalogModelKind(model); got != "finetune" {
		t.Fatalf("catalogModelKind() = %q, want finetune", got)
	}
}

func TestComputeEstimate(t *testing.T) {
	profile := computeProfile{Name: "test", GPUName: "GPU", GPUTFLOPS: 989, Efficiency: .4, GPUHourCostUSD: 1.85, TokensPerParameter: 20, Machines: 256, GPUsPerMachine: 4, FinetuneCostFraction: .05}
	base := estimateCompute(70_000_000_000, "base", profile)
	adapter := estimateCompute(70_000_000_000, "adapter", profile)
	if base.TotalGPUs != 1024 || base.WallDays <= 0 || base.CostUSD <= 0 {
		t.Fatalf("invalid base estimate: %#v", base)
	}
	if ratio := adapter.GPUHours / base.GPUHours; ratio < .05-1e-12 || ratio > .05+1e-12 {
		t.Fatalf("adapter fraction = %f", ratio)
	}
}

func TestFormulaTxtCoefficient(t *testing.T) {
	profile := computeProfile{Name: "formula", GPUName: "NVIDIA H100", GPUTFLOPS: 989, Efficiency: .4, GPUHourCostUSD: 1.85, TokensPerParameter: 20, Machines: 1, GPUsPerMachine: 1}
	got := estimateCompute(70_000_000_000, "base", profile)
	want := 155.881361644759 * 70 * 70
	if math.Abs(got.CostUSD-want) > 1e-6 {
		t.Fatalf("formula.txt cost = %.12f, want %.12f", got.CostUSD, want)
	}
}

func TestConfirmedScratchUsesFormulaEvenWhenCardAlsoReportsCompute(t *testing.T) {
	profile := computeProfile{Name: "formula", GPUName: "NVIDIA H100", GPUTFLOPS: 989, Efficiency: .4, GPUHourCostUSD: 1.85, TokensPerParameter: 20, Machines: 1, GPUsPerMachine: 1}
	selection := selectionConfig{Name: "s", ComputeProfiles: []string{"formula"}, RequireExplicitScratchForBase: true}
	reported := reportedComputeFromText("Training took 2 hours on 8 H100 GPUs.", "card")
	claim := scratchClaimFromText("This 7B model was trained from scratch on our corpus.", "card")
	record := modelToRecord("https://huggingface.co", catalogModel{ID: "org/scratch", Safetensors: catalogSafetensors{Total: 7_000_000_000}}, selection, 7_000_000_000, "", map[string]computeProfile{"formula": profile}, reported, claim)
	if record.TrainingCostMethod != "formula_txt_parameter_scaling" || record.TrainingCostUSD == nil {
		t.Fatalf("scratch model did not use formula.txt: %#v", record)
	}
	want := 155.881361644759 * 49
	if math.Abs(*record.TrainingCostUSD-want) > 1e-6 {
		t.Fatalf("scratch cost = %.12f, want %.12f", *record.TrainingCostUSD, want)
	}
}

func TestScratchClaimRejectsComponentAndHypotheticalTraining(t *testing.T) {
	for _, text := range []string{
		"For comparison, training a model of this scale from scratch typically costs $200k.",
		"This bypasses the difficulty of training linear attention from scratch.",
		"Here is an example of training a 340M model from scratch on FineWeb.",
		"The projection layer was trained from scratch while the language model stayed frozen.",
		"Only the decoder component was trained from scratch while the encoder remained frozen.",
		"The projection layer is randomly initialized and requires training.",
		"Токенизатор обучен с нуля, а модель загружена из готового checkpoint.",
		"TODO: write a demo notebook on training from scratch and generation.",
		"This model was pretrained from scratch (or fine-tuned) using masked language modeling.",
		"This architecture was built from scratch in PyTorch.",
	} {
		if got := scratchClaimFromText(text, "card"); got != nil {
			t.Fatalf("false scratch claim for %q: %#v", text, got)
		}
	}
}

func TestScratchClaimRejectsScratchThenPostTraining(t *testing.T) {
	for _, text := range []string{
		"The model was trained from scratch, followed by supervised fine-tuning and RLOO.",
		"The model was trained from scratch with AdEMAMix. Post-training included supervised fine-tuning and QRPO.",
		"Alignment through supervised fine-tuning was performed before the model was trained end-to-end from random initialization.",
		"The model was pre-trained from scratch and supervised fine-tuned on one million examples.",
	} {
		if got := scratchClaimFromText(text, "card"); got != nil {
			t.Fatalf("post-trained checkpoint was accepted as base: %q => %#v", text, got)
		}
	}
}

func TestScratchClaimAcceptsWholeDecoderOnlyAndVisionModels(t *testing.T) {
	for _, text := range []string{
		"This decoder-only autoregressive model was trained from scratch on the complete corpus.",
		"This vision transformer (ViT) was trained from scratch on image-caption pairs.",
	} {
		if got := scratchClaimFromText(text, "card"); got == nil {
			t.Fatalf("whole-model scratch claim was rejected for %q", text)
		}
	}
}

func TestGeneratedTrainerAndDerivativeNamesAreNotBaseModels(t *testing.T) {
	for _, model := range []catalogModel{
		{ID: "org/unknown-output", Tags: []string{"generated_from_trainer"}},
		{ID: "org/research-output", Tags: []string{"trl", "sft"}},
		{ID: "org/swarm-output", Tags: []string{"grpo", "rl-swarm"}},
		{ID: "org/Model-8B-Instruct"},
		{ID: "org/Model-Chat"},
	} {
		if got := catalogModelKind(model); got != "finetune" {
			t.Fatalf("catalogModelKind(%q) = %q, want finetune", model.ID, got)
		}
	}
}

func TestQuantizedConversionsAreNotBaseModels(t *testing.T) {
	for _, model := range []catalogModel{
		{ID: "org/Model-4bits"},
		{ID: "org/Model", Tags: []string{"bitsandbytes", "8-bit"}},
		{ID: "mirror/source_-_Model-8B"},
		{ID: "org/Model-8B-hf"},
	} {
		if got := catalogModelKind(model); got != "quantized" {
			t.Fatalf("catalogModelKind(%q) = %q, want quantized", model.ID, got)
		}
	}
}

func TestOpaqueUUIDUploadsAreForks(t *testing.T) {
	model := catalogModel{ID: "user/000b77fd-c73e-4660-a7b2-71c0b787ed07"}
	if got := catalogModelKind(model); got != "fork" {
		t.Fatalf("catalogModelKind(%q) = %q, want fork", model.ID, got)
	}
}

func TestForkNamesAreSeparatedFromBase(t *testing.T) {
	for _, id := range []string{"user/Qwen-27B-Abliterated", "user/Model-Uncensored", "user/Model-Base-Adjust", "user/Model-Repacked"} {
		if got := catalogModelKind(catalogModel{ID: id}); got != "fork" {
			t.Fatalf("catalogModelKind(%q) = %q, want fork", id, got)
		}
	}
}

func TestTargetLLMExcludesImageVideoAndAcceptsTextFamilies(t *testing.T) {
	positive := []catalogModel{
		{ID: "org/new-language-model", PipelineTag: "text-generation"},
		{ID: "user/Qwen3-8B", Tags: []string{"qwen3", "safetensors"}},
		{ID: "org/encoder", PipelineTag: "fill-mask"},
	}
	for _, model := range positive {
		if !catalogIsTargetLLM(model) {
			t.Fatalf("target LLM was rejected: %#v", model)
		}
	}
	negative := []catalogModel{
		{ID: "org/stable-diffusion-qwen-caption", PipelineTag: "text-to-image", Tags: []string{"diffusers", "qwen"}},
		{ID: "org/Qwen-VL", PipelineTag: "image-text-to-text", Tags: []string{"qwen", "vision"}},
		{ID: "org/video-model", PipelineTag: "text-to-video"},
		{ID: "org/gemma-vlm-custom", Tags: []string{"gemma", "multimodal"}},
		{ID: "org/DNAGPT", Tags: []string{"llama", "genomic"}},
		{ID: "vojtam/DNAGPT2_32", PipelineTag: "text-generation", Tags: []string{"gpt2"}},
		{ID: "macwiatrak/bacformer-causal-MAG", PipelineTag: "text-generation", Tags: []string{"transformers"}},
	}
	for _, model := range negative {
		if catalogIsTargetLLM(model) {
			t.Fatalf("non-text model was accepted as target LLM: %#v", model)
		}
	}
}

func TestKnownLineageOverrides(t *testing.T) {
	wants := map[string]string{
		"aiqtech/LLaDA2.0-flash-preview":             "finetune",
		"nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "finetune",
		"OpenKing/vualtgemma-1b-non-gated":           "fork",
		"LSX-UniWue/LLaMmlein_1B":                    "fork",
		"microsoft/bitnet-b1.58-2B-4T":               "quantized",
		"vngrs-ai/Kumru-2B-Base":                     "finetune",
	}
	for id, want := range wants {
		if got := catalogModelKind(catalogModel{ID: id}); got != want {
			t.Fatalf("catalogModelKind(%q) = %q, want %q", id, got, want)
		}
	}
}

func TestPretrainingEvidenceExtractsTokensAndActiveParameters(t *testing.T) {
	text := "Kimi K2 has 1 trillion total parameters and 32 billion activated parameters. Pre-trained on 15.5T tokens."
	claim := scratchClaimFromText(text, "card", "org/Kimi-K2-Base")
	if claim == nil || claim.EvidenceType != "pretraining_evidence" {
		t.Fatalf("pretraining evidence was not accepted: %#v", claim)
	}
	if math.Abs(claim.ActiveParametersB-32) > 1e-9 || math.Abs(claim.TrainingTokensT-15.5) > 1e-9 {
		t.Fatalf("unexpected parsed training scale: %#v", claim)
	}
}

func TestPretrainingEvidenceAllowsFamilyCardThatAlsoDescribesInstruct(t *testing.T) {
	text := "Apriel-5B-Base is a foundation model pre-trained on 4.5T tokens. Apriel-5B-Instruct is built on top of the base using CPT and SFT."
	claim := scratchClaimFromText(text, "card", "org/Apriel-5B-Base")
	if claim == nil || claim.TrainingTokensT != 4.5 {
		t.Fatalf("base evidence was lost to a derivative described elsewhere in the family card: %#v", claim)
	}
}

func TestActiveParametersFallbackToModelName(t *testing.T) {
	if got := activeParametersFromText("Training Stage: Pretraining", "Qwen/Qwen3-30B-A3.3B-Base"); math.Abs(got-3.3) > 1e-9 {
		t.Fatalf("activeParametersFromText model-name fallback = %v, want 3.3", got)
	}
}

func TestActiveParametersHandleMarkdownAndKnownMoECards(t *testing.T) {
	if got := activeParametersFromText("The model has **355** billion total parameters with **32** billion active parameters.", "org/model"); got != 32 {
		t.Fatalf("markdown active parameters = %v, want 32", got)
	}
	if got := activeParametersFromText("The shared architecture table lists several sizes.", "ibm-granite/granite-4.0-h-small-base"); got != 9 {
		t.Fatalf("known Granite H Small active parameters = %v, want 9", got)
	}
}

func TestActualTrainingTokensPreferRunDisclosureOverDatasetCapacity(t *testing.T) {
	card := "For the pretraining corpus we use FineWeb, which consists of 15T tokens. The released model used the 350BT subset."
	if got := actualTrainingTokensFromText(card, ""); got != 0 {
		t.Fatalf("dataset capacity was treated as tokens processed: %v", got)
	}
	if claim := scratchClaimFromText(card, "card", "org/NoPE-GPT-400M-Base"); claim == nil || claim.TrainingTokensT != 0 {
		t.Fatalf("dataset capacity leaked through scratch evidence: %#v", claim)
	}
	card = "The model was trained for approximately 25 trillion tokens. The source corpus contains 100T tokens."
	if got := actualTrainingTokensFromText(card, ""); got != 25 {
		t.Fatalf("actual training tokens = %v, want 25", got)
	}
	card = "* **Pre-training Tokens:** 19.7 Trillion"
	if got := actualTrainingTokensFromText(card, ""); got != 19.7 {
		t.Fatalf("labeled pretraining tokens = %v, want 19.7", got)
	}
}

func TestActualTrainingTokensHandleDescriptiveTrainingSentences(t *testing.T) {
	for text, want := range map[string]float64{
		"Pre-trained a 1T parameter MoE model on 15.5T tokens.":                                       15.5,
		"The model was trained from scratch on a multilingual corpus comprising 2.1 trillion tokens.": 2.1,
		"The model is trained from scratch on approximately 23 trillion tokens.":                      23,
	} {
		if got := actualTrainingTokensFromText(text, ""); got != want {
			t.Fatalf("actualTrainingTokensFromText(%q) = %v, want %v", text, got, want)
		}
	}
}

func TestDeclaredBaseAndDerivativeCardClassification(t *testing.T) {
	if got := scratchClaimFromText("This is the released foundation checkpoint.", "card", "org/Model-8B-Base"); got == nil || got.EvidenceType != "declared_base_model" {
		t.Fatalf("declared base checkpoint was rejected: %#v", got)
	}
	if got := scratchClaimFromText("We introduce Shuttle, a fine-tuned version of Qwen3.", "card", "org/Shuttle-Base"); got != nil {
		t.Fatalf("fine-tuned checkpoint was accepted as base: %#v", got)
	}
	if got := scratchClaimFromText("This base is built upon the original V3 checkpoint through context extension.", "card", "org/V3.1-Base"); got != nil {
		t.Fatalf("continued checkpoint was accepted as independent base: %#v", got)
	}
	if got := scratchClaimFromText("This model is continue pre-trained from DeepSeek-LLM 7B on 2T tokens.", "card", "org/Custom-Base"); got != nil {
		t.Fatalf("continued pretraining from an upstream model was accepted as independent base: %#v", got)
	}
}

func TestAuxiliaryONNXTagDoesNotTurnCanonicalWeightsIntoAConversion(t *testing.T) {
	model := catalogModel{ID: "org/Model-Base", LibraryName: "transformers", Tags: []string{"safetensors", "onnx"}}
	if got := catalogModelKind(model); got != "base" {
		t.Fatalf("canonical repo with auxiliary ONNX files classified as %q", got)
	}
	model.LibraryName = "onnx"
	if got := catalogModelKind(model); got != "quantized" {
		t.Fatalf("ONNX-native conversion classified as %q", got)
	}
}

func TestStageCheckpointIsAForkArtifact(t *testing.T) {
	if got := catalogModelKind(catalogModel{ID: "org/Model-40B-Base-Stage1"}); got != "fork" {
		t.Fatalf("stage checkpoint classified as %q", got)
	}
}

func TestAffineSwarmUploadIsAForkArtifact(t *testing.T) {
	if got := catalogModelKind(catalogModel{ID: "user/Affine-5CXQgGhdq1mJ1aYz"}); got != "fork" {
		t.Fatalf("Affine swarm upload classified as %q", got)
	}
}

func TestBaseComputeUsesReportedTokensAndActiveParameters(t *testing.T) {
	profile := computeProfile{Name: "h100", GPUName: "H100", GPUTFLOPS: 989, Efficiency: 0.4, GPUHourCostUSD: 1.85, TokensPerParameter: 20, Machines: 1, GPUsPerMachine: 1}
	claim := &reportedScratchClaim{ActiveParametersB: 32, TrainingTokensT: 15.5}
	got := estimateBaseTrainingCompute(1_000_000_000_000, claim, profile)
	want := 6 * 32e9 * 15.5e12 / (989e12 * 0.4 * 3600) * 1.85
	if math.Abs(got.CostUSD-want) > 1e-6 || got.Method != "reported_pretraining_tokens_active_parameters" {
		t.Fatalf("estimateBaseTrainingCompute = %#v, want cost %.9f", got, want)
	}
}

func TestDerivativeNeverReceivesFullPretrainingFormula(t *testing.T) {
	profile := computeProfile{Name: "formula", GPUName: "NVIDIA H100", GPUTFLOPS: 989, Efficiency: .4, GPUHourCostUSD: 1.85, TokensPerParameter: 20, Machines: 1, GPUsPerMachine: 1}
	selection := selectionConfig{Name: "s", ComputeProfiles: []string{"formula"}, FormulaRequiresExplicitScratch: true}
	model := catalogModel{ID: "org/Model-8B-Instruct", Safetensors: catalogSafetensors{Total: 8_000_000_000}}
	claim := scratchClaimFromText("The underlying model was trained from scratch.", "card")
	record := modelToRecord("https://huggingface.co", model, selection, 8_000_000_000, "", map[string]computeProfile{"formula": profile}, nil, claim)
	if record.ModelKind != "finetune" || record.TrainingCostUSD != nil {
		t.Fatalf("derivative received a full-pretraining cost: %#v", record)
	}
}

func TestConfiguredDerivativeFractionPricesNonBaseKinds(t *testing.T) {
	profile := computeProfile{Name: "formula", GPUName: "H100", GPUTFLOPS: 989, Efficiency: .4, GPUHourCostUSD: 1.85, TokensPerParameter: 20, Machines: 1, GPUsPerMachine: 1, FinetuneCostFraction: .01}
	selection := selectionConfig{Name: "s", ComputeProfiles: []string{"formula"}}
	models := []catalogModel{
		{ID: "org/model-finetuned"},
		{ID: "org/model-lora"},
		{ID: "org/model-uncensored"},
		{ID: "org/model-gguf"},
		{ID: "org/model-merge"},
	}
	baseCost := estimateCompute(7_000_000_000, "base", profile).CostUSD
	for _, model := range models {
		record := modelToRecordWithReview("https://huggingface.co", model, selection, 7_000_000_000, "", map[string]computeProfile{"formula": profile}, nil, nil, nil, false)
		if record.ModelKind == "base" || record.TrainingCostUSD == nil {
			t.Fatalf("%s was not priced as a derivative: %#v", model.ID, record)
		}
		if math.Abs(*record.TrainingCostUSD-baseCost*.01) > 1e-6 {
			t.Fatalf("%s cost = %.12f, want %.12f", model.ID, *record.TrainingCostUSD, baseCost*.01)
		}
		if record.TrainingCostMethod != "formula_txt_text_derivative_fraction" || record.TrainingCostTier != "configured_derivative_fraction" {
			t.Fatalf("%s has unexpected provenance: %#v", model.ID, record)
		}
	}
}

func TestConfiguredDiffusionDerivativeFractionUsesDiffusionFormula(t *testing.T) {
	profile := computeProfile{Name: "diffusion", GPUName: "H100", GPUTFLOPS: 989, Efficiency: .4, GPUHourCostUSD: 1.85, TokensPerParameter: 20, Machines: 1, GPUsPerMachine: 1, FinetuneCostFraction: .01, DiffusionImageBudget: 3, LatentSequenceLength: 1024}
	selection := selectionConfig{Name: "s", ComputeProfiles: []string{"diffusion"}}
	model := catalogModel{ID: "org/image-finetuned", PipelineTag: "text-to-image"}
	record := modelToRecordWithReview("https://huggingface.co", model, selection, 2_000_000_000, "", map[string]computeProfile{"diffusion": profile}, nil, nil, nil, false)
	base := estimateDiffusionBaseTrainingCompute(2_000_000_000, nil, profile)
	if record.TrainingCostUSD == nil || math.Abs(*record.TrainingCostUSD-base.CostUSD*.01) > 1e-6 {
		t.Fatalf("diffusion derivative cost = %#v, want %.12f", record.TrainingCostUSD, base.CostUSD*.01)
	}
	if record.TrainingCostMethod != "formula_txt_diffusion_derivative_fraction" {
		t.Fatalf("unexpected diffusion derivative method: %q", record.TrainingCostMethod)
	}
}

func TestParameterRanges(t *testing.T) {
	for parameters, want := range map[int64]string{
		0: "unknown", 1_999_999_999: "<2B", 2_000_000_000: "2-<7B",
		7_000_000_000: "7-<13B", 13_000_000_000: "13-<34B",
		34_000_000_000: "34-<70B", 70_000_000_000: "70-<120B",
		120_000_000_000: "120-<500B", 500_000_000_000: ">=500B",
	} {
		if got := parameterRange(parameters); got != want {
			t.Fatalf("parameterRange(%d) = %q, want %q", parameters, got, want)
		}
	}
}

func TestExplicitScratchDoesNotOverrideDerivativeClassification(t *testing.T) {
	claim := scratchClaimFromText("This model was trained from scratch on our own corpus.", "card")
	nameOnly := catalogModel{ID: "org/Stockmark-100B-Instruct"}
	if got := resolvedCatalogModelKind(nameOnly, claim, nil); got != "finetune" {
		t.Fatalf("scratch text overrode the derivative-name policy: %q", got)
	}
	strongLineage := catalogModel{
		ID:         "org/Model-Instruct",
		BaseModels: catalogBaseModels{Relation: "finetune", Models: []catalogBaseModel{{ID: "org/Model-Base"}}},
	}
	if got := resolvedCatalogModelKind(strongLineage, claim, nil); got != "finetune" {
		t.Fatalf("scratch text overrode strong Hub lineage metadata: %q", got)
	}
}

func TestReportedDerivativeEvidenceRepairsResidualBaseKind(t *testing.T) {
	model := catalogModel{ID: "org/metadata-missing"}
	reported := reportedComputeFromText("The current SFT fine-tuning took 2 hours on 8 H100 GPUs.", "card")
	if got := resolvedCatalogModelKind(model, nil, reported); got != "finetune" {
		t.Fatalf("card-aware derivative classification = %q, want finetune", got)
	}
}

func TestScratchClaimRejectsCurrentModelPostTraining(t *testing.T) {
	for _, text := range []string{
		"This model was trained from scratch on 20T tokens. The model was further fine-tuned on instruction and RL data.",
		"This is a fine-tuned model; its base model was trained from scratch on 20T tokens.",
	} {
		if got := scratchClaimFromText(text, "card"); got != nil {
			t.Fatalf("post-trained current model was accepted as base scratch: %#v", got)
		}
	}
}

func TestScratchClaimAllowsFutureFineTuningRecommendation(t *testing.T) {
	text := "This base model was trained from scratch and is a good starting point for instruction fine-tuning."
	if got := scratchClaimFromText(text, "card"); got == nil {
		t.Fatal("a future fine-tuning recommendation invalidated a real scratch claim")
	}
}

func TestScratchClaimDoesNotConfuseFineTunedBaseline(t *testing.T) {
	text := "Our model was trained from scratch. In evaluation, the model is compared with a fine-tuned BERT baseline."
	if got := scratchClaimFromText(text, "card"); got == nil {
		t.Fatal("a fine-tuned evaluation baseline invalidated the current model's scratch claim")
	}
}

func TestScratchClaimAllowsRetrospectiveWhenClause(t *testing.T) {
	text := "When we trained this model from scratch, we used two trillion tokens."
	if got := scratchClaimFromText(text, "card"); got == nil {
		t.Fatal("an actual retrospective when-clause was rejected")
	}
}

func TestScratchClaimRejectsCardPublishedBeforeSelection(t *testing.T) {
	claim := scratchClaimFromText("This model was trained from scratch on our corpus.", "card")
	claim.EarliestCardCreatedAt = "2024-03-27T00:00:00Z"
	selection := compiledSelection{from: time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC), to: time.Date(2025, 12, 31, 23, 59, 59, 0, time.UTC)}
	if scratchClaimFallsInSelection(claim, selection) {
		t.Fatal("a copied pre-2025 scratch card must not be charged to 2025")
	}
}

func TestScratchClaimTrainerMustMatchRepositoryOwner(t *testing.T) {
	claim := scratchClaimFromText("DBRX is a model trained from scratch by Databricks.", "card")
	if claim == nil || claim.ClaimedTrainer != "Databricks" {
		t.Fatalf("trainer was not extracted: %#v", claim)
	}
	if scratchClaimMatchesOwner(catalogModel{ID: "nicoboss/dbrx-base", Author: "nicoboss"}, claim) {
		t.Fatal("third-party mirror must not own Databricks training cost")
	}
	if !scratchClaimMatchesOwner(catalogModel{ID: "databricks/dbrx-base", Author: "databricks"}, claim) {
		t.Fatal("official Databricks repository must retain its training cost")
	}
}

func TestReportedComputeUsesGPUHoursAndDefaultPrice(t *testing.T) {
	got := reportedComputeFromText("Fine-tuning took 90 minutes on 8x H100 GPUs.", "README.md")
	if got == nil {
		t.Fatal("reported compute was not parsed")
	}
	if got.Hours != 1.5 || got.GPUCount != 8 || got.GPUHours != 12 || math.Abs(got.CostUSD-22.2) > 1e-9 {
		t.Fatalf("unexpected reported compute: %#v", got)
	}
	if got.AssumedGPU || got.AssumedGPUCount {
		t.Fatalf("hardware should not be assumed: %#v", got)
	}
}

func TestReportedComputeDoesNotTreatRTXModelNumberAsGPUHours(t *testing.T) {
	text := "Pre-Training: | Model | RTX 4090 GPU hours |\n| Doge-160M | 120 |"
	if got := reportedComputeFromText(text, "README.md"); got != nil {
		t.Fatalf("GPU model number became compute: %#v", got)
	}
}

func TestReportedComputeParsesTPURuntimeAndCoreCount(t *testing.T) {
	got := reportedComputeFromText("Training Time: 4.82 TPUv3-8 Hours", "README.md")
	if got == nil || math.Abs(got.Hours-4.82) > 1e-9 || got.GPUCount != 8 || math.Abs(got.GPUHours-38.56) > 1e-9 {
		t.Fatalf("unexpected TPU runtime: %#v", got)
	}
}

func TestReportedComputeAssumesOneH100AndHonorsReportedCost(t *testing.T) {
	got := reportedComputeFromText("Training time: 2 days. Total training cost was $100.", "README.md")
	if got == nil {
		t.Fatal("reported compute was not parsed")
	}
	if got.Hours != 48 || got.GPUCount != 1 || got.GPUName != "NVIDIA H100" || got.CostUSD != 100 || got.CostMethod != "reported_total_cost" {
		t.Fatalf("unexpected reported compute: %#v", got)
	}
	if !got.AssumedGPU || !got.AssumedGPUCount {
		t.Fatalf("default H100 should be marked as assumed: %#v", got)
	}
}

func TestReportedComputeFromStructuredRuntime(t *testing.T) {
	got := reportedComputeFromCardData(map[string]any{"training_results": map[string]any{"train_runtime": 7200.0}})
	if got == nil || got.Hours != 2 || got.CostUSD != 3.7 {
		t.Fatalf("unexpected structured reported compute: %#v", got)
	}
}

func TestReportedGPUHoursAreNotMultipliedTwice(t *testing.T) {
	got := reportedComputeFromText("Fine-tuning used 120 GPU hours on H100.", "README.md")
	if got == nil || got.GPUHours != 120 || math.Abs(got.CostUSD-222) > 1e-9 {
		t.Fatalf("unexpected GPU-hour compute: %#v", got)
	}
}

func TestReportedTotalGPUHoursInHoursUsedFieldAreNotMultiplied(t *testing.T) {
	got := reportedComputeFromText("Hardware: 4 x A100. **Hours used (total GPU hours):** 16005.12h", "README.md")
	if got == nil || got.GPUHours != 16005.12 || math.Abs(got.CostUSD-29609.472) > 1e-9 {
		t.Fatalf("unexpected total GPU-hour compute: %#v", got)
	}
}

func TestReportedCostWithoutDurationIsAccepted(t *testing.T) {
	got := reportedComputeFromText("The total training cost was $1,234.50.", "README.md")
	if got == nil || got.CostUSD != 1234.5 || got.CostMethod != "reported_total_cost" || got.Hours != 0 {
		t.Fatalf("unexpected directly reported cost: %#v", got)
	}
}

func TestReportedCostSupportsMagnitudeSuffix(t *testing.T) {
	got := reportedComputeFromText("The total training cost was $2.5M.", "README.md")
	if got == nil || got.CostUSD != 2_500_000 || got.CostMethod != "reported_total_cost" {
		t.Fatalf("unexpected suffixed cost: %#v", got)
	}
}

func TestReportedHardwareAllowsNVIDIAPrefix(t *testing.T) {
	got := reportedComputeFromText("Training took 2 hours on 8 x NVIDIA H100 GPUs.", "README.md")
	if got == nil || got.GPUCount != 8 || got.GPUHours != 16 {
		t.Fatalf("unexpected NVIDIA hardware parse: %#v", got)
	}
}

func TestReportedHardwareUsesGenericCountBesideNamedGPU(t *testing.T) {
	got := reportedComputeFromText("Training took 10 hours on 8 GPUs on H100.", "README.md")
	if got == nil || got.GPUCount != 8 || got.GPUHours != 80 {
		t.Fatalf("generic count next to named hardware was lost: %#v", got)
	}
	got = reportedComputeFromText("Inference used 16 GPUs. Training took 10 hours on H100.", "README.md")
	if got == nil || got.GPUCount != 1 || got.GPUHours != 10 {
		t.Fatalf("GPU count leaked across sentences: %#v", got)
	}
}

func TestReportedGPUHoursSupportsThousandsAndMagnitudeSuffix(t *testing.T) {
	for _, test := range []struct {
		text string
		want float64
	}{
		{"Hours used: 61,440 GPU hours", 61_440},
		{"We used approximately 150k GPU hours for model training.", 150_000},
	} {
		got := reportedComputeFromText(test.text, "README.md")
		if got == nil || got.GPUHours != test.want {
			t.Fatalf("reportedComputeFromText(%q) = %#v, want %.0f GPU-hours", test.text, got, test.want)
		}
	}
}

func TestReportedGPUHoursSumsDistinctStagesButNotRepeatedTotals(t *testing.T) {
	got := reportedComputeFromText("Stage 1 used 2,880 H100 GPU hours. Stage 2 used 11,520 H100 GPU hours.", "README.md")
	if got == nil || got.GPUHours != 14_400 {
		t.Fatalf("staged GPU-hours were not summed: %#v", got)
	}
	got = reportedComputeFromText("Stage 1 used 100 H100 GPU hours. Stage 2 used 100 H100 GPU hours.", "README.md")
	if got == nil || got.GPUHours != 200 {
		t.Fatalf("equal-valued stages were collapsed: %#v", got)
	}
	got = reportedComputeFromText("Total GPU hours: 100. Components used 40 GPU hours and 60 GPU hours.", "README.md")
	if got == nil || got.GPUHours != 100 {
		t.Fatalf("explicit total did not supersede components: %#v", got)
	}
}

func TestReportedGPUHoursRejectsInferenceOnlyCompute(t *testing.T) {
	got := reportedComputeFromText("Inference used 1,000 H100 GPU hours. Fine-tuning used 100 H100 GPU hours.", "README.md")
	if got == nil || got.GPUHours != 100 {
		t.Fatalf("inference GPU-hours leaked into training compute: %#v", got)
	}
	if got := reportedComputeFromText("Evaluation used 250 H100 GPU hours.", "README.md"); got != nil {
		t.Fatalf("evaluation-only compute was accepted: %#v", got)
	}
	if got := reportedComputeFromText("We report aggregated statistics for models fine-tuned and inferences. A cumulative 6,500 GPU hours were used.", "README.md"); got != nil {
		t.Fatalf("mixed training/inference aggregate was accepted: %#v", got)
	}
}

func TestReportedDurationPrefersCurrentFineTuningAndNearestHardware(t *testing.T) {
	text := "Base pre-training took 45 days on 512 H100 GPUs. Current fine-tuning took 2 hours on 8 H100 GPUs."
	got := reportedComputeFromText(text, "README.md")
	if got == nil || got.Hours != 2 || got.GPUCount != 8 || got.GPUHours != 16 {
		t.Fatalf("upstream pre-training overrode current fine-tuning: %#v", got)
	}
}

func TestFlattenCardDataUsesStableKeyOrder(t *testing.T) {
	card := map[string]any{
		"z": "last",
		"a": map[string]any{"d": "fourth", "b": "second"},
	}
	var lines []string
	flattenCardData(card, "cardData", &lines)
	want := []string{"cardData.a.b: second", "cardData.a.d: fourth", "cardData.z: last"}
	if strings.Join(lines, "\n") != strings.Join(want, "\n") {
		t.Fatalf("flattenCardData order = %#v, want %#v", lines, want)
	}
	if number, ok := findPositiveNumber(cardWithNumericSignals(), map[string]bool{"hours_used": true, "training_hours": true}); !ok || number != 10 {
		t.Fatalf("structured numeric field selection = %v, %t; want deterministic lexical first value 10", number, ok)
	}
}

func cardWithNumericSignals() map[string]any {
	return map[string]any{"training_hours": 20.0, "hours_used": 10.0}
}

func TestReportedDurationSupportsCompoundTimeAndSuffixGPUCount(t *testing.T) {
	got := reportedComputeFromText("Training duration: 5h50m. Hardware: GPU T4 x2.", "README.md")
	if got == nil || math.Abs(got.Hours-(5+50.0/60)) > 1e-9 || got.GPUCount != 2 || math.Abs(got.GPUHours-(11+2.0/3.0)) > 1e-9 {
		t.Fatalf("unexpected compound duration: %#v", got)
	}
	got = reportedComputeFromText("Training took 3 months and 23 days on 1 H100.", "README.md")
	if got == nil || got.Hours != (3*30+23)*24 {
		t.Fatalf("unexpected month/day duration: %#v", got)
	}
	got = reportedComputeFromText("Training runtime: 44,426 seconds on GPUs: 8.", "README.md")
	if got == nil || math.Abs(got.Hours-44_426.0/3600) > 1e-9 || got.GPUCount != 8 {
		t.Fatalf("unexpected comma runtime or generic GPU count: %#v", got)
	}
}

func TestReportedHardwareDoesNotTreat4090AsGPUCount(t *testing.T) {
	got := reportedComputeFromText("Training time: 216 hours. Hardware: NVIDIA 4090 GPU.", "README.md")
	if got == nil || got.GPUCount != 1 || got.GPUName != "NVIDIA RTX 4090" || got.GPUHours != 216 {
		t.Fatalf("consumer GPU model became a GPU count: %#v", got)
	}
}

func TestReportedComputeIgnoresWidgetSourcePassages(t *testing.T) {
	text := "cardData.widget.sentences: Llama trained for 30,840,000 GPU hours.\ncardData.model_name: legal-ft"
	if got := reportedComputeFromText(text, "cardData"); got != nil {
		t.Fatalf("widget source passage became reported compute: %#v", got)
	}
}

func TestReportedBareLessThanCostIsAnUpperBound(t *testing.T) {
	got := reportedComputeFromText("Fine-tuning took 2 hours on one RTX 4090. Cost: <$2 on RunPod.", "README.md")
	if got == nil || got.CostMethod != "reported_total_cost" || got.CostUSD != 2 {
		t.Fatalf("less-than training cost was not parsed: %#v", got)
	}
}

func TestReportedDurationRejectsModelSizeDatasetHoursAndCPU(t *testing.T) {
	for _, text := range []string{
		"It was trained with the same recipe as SmolLM2-135M, but starts from another checkpoint.",
		"Fine-tuned for European Portuguese on around 425h of speech recordings.",
		"Training time: 40 hours (CPU).",
	} {
		if got := reportedComputeFromText(text, "README.md"); got != nil {
			t.Fatalf("false duration for %q: %#v", text, got)
		}
	}
}

func TestReportedCostRejectsGenericWidgetCost(t *testing.T) {
	text := "Each image costs tokens. That's a total cost of $1.68 to process 68,000 images."
	if got := reportedComputeFromText(text, "cardData.widget"); got != nil {
		t.Fatalf("generic non-training cost was accepted: %#v", got)
	}
}

func TestReportedTrainingRunsAreDeduplicatedByFullCardHash(t *testing.T) {
	computeA := reportedComputeFromText("Training took 10 hours on 2x H100 GPUs.", "card")
	computeB := reportedComputeFromText("Training took 10 hours on 2x H100 GPUs.", "card")
	costA, costB := computeA.CostUSD, computeB.CostUSD
	records := []catalogRecord{
		{Selection: "s", RepoID: "mirror/model", CreatedAt: "2025-02-01T00:00:00Z", ReportedCompute: computeA, TrainingCostUSD: &costA, TrainingCostMethod: computeA.CostMethod},
		{Selection: "s", RepoID: "original/model", CreatedAt: "2025-01-01T00:00:00Z", ReportedCompute: computeB, TrainingCostUSD: &costB, TrainingCostMethod: computeB.CostMethod},
	}
	deduplicateReportedTrainingRuns(records)
	if records[0].TrainingCostUSD != nil || records[0].TrainingCostMethod != "duplicate_reported_training_run" {
		t.Fatalf("mirror was not deduplicated: %#v", records[0])
	}
	if records[1].TrainingCostUSD == nil || *records[1].TrainingCostUSD != 37 {
		t.Fatalf("canonical run lost its cost: %#v", records[1])
	}
}

func TestDerivativeReportedComputeAlreadyAttributedToBaseIsNotCountedTwice(t *testing.T) {
	baseCost, derivativeCost := 100.0, 100.0
	reportedBase := &reportedTrainingCompute{RunHash: "same-run"}
	reportedDerivative := &reportedTrainingCompute{RunHash: "same-run"}
	records := []catalogRecord{
		{Selection: "s", RepoID: "org/model-base", ModelKind: "base", ReportedCompute: reportedBase, TrainingCostUSD: &baseCost, TrainingCostMethod: "formula_txt_parameter_scaling"},
		{Selection: "s", RepoID: "org/model-instruct", ModelKind: "finetune", ReportedCompute: reportedDerivative, TrainingCostUSD: &derivativeCost, TrainingCostMethod: "gpu_hours_x_hourly_rate"},
	}
	deduplicateReportedDerivativeAgainstBase(records)
	if records[0].TrainingCostUSD == nil || records[1].TrainingCostUSD != nil || records[1].TrainingCostMethod != "duplicate_reported_base_training_run" {
		t.Fatalf("base/derivative reported run was not attributed once: %#v", records)
	}
}

func TestDatedReportedTrajectoryKeepsLargestCumulativeRun(t *testing.T) {
	oldCost, newCost := 334850.0, 434750.0
	oldCompute := &reportedTrainingCompute{GPUHours: 181000, CostUSD: oldCost, CostMethod: "gpu_hours_x_hourly_rate", RunHash: "old"}
	newCompute := &reportedTrainingCompute{GPUHours: 235000, CostUSD: newCost, CostMethod: "gpu_hours_x_hourly_rate", RunHash: "new"}
	records := []catalogRecord{
		{Selection: "s", Owner: "AstroMLab", RepoID: "AstroMLab/AstroSage-70B", BaseModel: "meta/Llama-70B", ModelKind: "finetune", EffectiveParameters: 70, CreatedAt: "2025-05-18", ReportedCompute: oldCompute, TrainingCostUSD: &oldCost, TrainingCostMethod: oldCompute.CostMethod},
		{Selection: "s", Owner: "AstroMLab", RepoID: "AstroMLab/AstroSage-70B-20251009", BaseModel: "meta/Llama-70B", ModelKind: "finetune", EffectiveParameters: 70, CreatedAt: "2025-10-09", ReportedCompute: newCompute, TrainingCostUSD: &newCost, TrainingCostMethod: newCompute.CostMethod},
	}
	deduplicateReportedTrainingRuns(records)
	if records[0].TrainingCostUSD != nil || records[0].TrainingCostMethod != "duplicate_reported_training_trajectory" || records[1].TrainingCostUSD == nil {
		t.Fatalf("unexpected dated trajectory deduplication: %#v", records)
	}
}

func TestSummaryIncludesFormula20SensitivityScenarios(t *testing.T) {
	baseCost, derivativeCost := 100.0, 10.0
	literal := computeEstimate{Method: "formula_txt_literal_total_parameters", CostUSD: 400.0}
	records := []catalogRecord{
		{Selection: "s", ModelKind: "base", EffectiveParameters: 10_000_000_000, ScratchClaim: &reportedScratchClaim{ActiveParametersB: 2}, TrainingCostUSD: &baseCost, TrainingCostMethod: "reported_pretraining_tokens_active_parameters", Compute: []computeEstimate{literal}},
		{Selection: "s", ModelKind: "finetune", TrainingCostUSD: &derivativeCost, TrainingCostMethod: "gpu_hours_x_hourly_rate"},
	}
	summary := buildCatalogSummary(time.Now(), true, 1, 2, 2, 0, 0, 0, 0, records, []selectionConfig{{Name: "s"}})
	got := summary.Selections["s"]
	if math.Abs(got.Formula20HybridUSD-90) > 1e-9 || math.Abs(got.Formula20LiteralUSD-410) > 1e-9 {
		t.Fatalf("unexpected sensitivity totals: %#v", got)
	}
}

func TestScratchTrainingRunsAreDeduplicatedByCardAndParameters(t *testing.T) {
	claimA := scratchClaimFromText("The model was trained from scratch on corpus A.", "card")
	claimB := scratchClaimFromText("The model was trained from scratch on corpus A.", "card")
	costA, costB := 100.0, 100.0
	records := []catalogRecord{
		{Selection: "s", RepoID: "mirror/model", CreatedAt: "2025-02-01T00:00:00Z", EffectiveParameters: 1_000_000_000, ScratchClaim: claimA, TrainingCostUSD: &costA, TrainingCostMethod: "formula_txt_parameter_scaling"},
		{Selection: "s", RepoID: "original/model", CreatedAt: "2025-01-01T00:00:00Z", EffectiveParameters: 1_000_000_000, ScratchClaim: claimB, TrainingCostUSD: &costB, TrainingCostMethod: "formula_txt_parameter_scaling"},
	}
	deduplicateScratchTrainingRuns(records)
	if records[0].TrainingCostUSD != nil || records[0].TrainingCostMethod != "duplicate_scratch_training_run" {
		t.Fatalf("scratch mirror was not deduplicated: %#v", records[0])
	}
	if records[1].TrainingCostUSD == nil {
		t.Fatalf("canonical scratch run lost its cost: %#v", records[1])
	}
}

func TestScratchCheckpointsInOneTrajectoryAreDeduplicated(t *testing.T) {
	costA, costB := 100.0, 100.0
	records := []catalogRecord{
		{Selection: "s", RepoID: "org/model-step-100", Owner: "org", CreatedAt: "2025-01-01T00:00:00Z", EffectiveParameters: 1_000_000_000, ScratchClaim: &reportedScratchClaim{RunHash: "different-a"}, TrainingCostUSD: &costA, TrainingCostMethod: "formula_txt_parameter_scaling"},
		{Selection: "s", RepoID: "org/model-step-1.5k", Owner: "org", CreatedAt: "2025-02-01T00:00:00Z", EffectiveParameters: 1_000_000_000, ScratchClaim: &reportedScratchClaim{RunHash: "different-b"}, TrainingCostUSD: &costB, TrainingCostMethod: "formula_txt_parameter_scaling"},
	}
	deduplicateScratchTrainingRuns(records)
	if records[0].TrainingCostUSD == nil || records[1].TrainingCostUSD != nil || records[1].TrainingCostMethod != "duplicate_scratch_training_trajectory" {
		t.Fatalf("checkpoint trajectory was not deduplicated: %#v", records)
	}
}

func TestReportedTrainingYearExcludesAnOlderRunUploadedIn2025(t *testing.T) {
	compute := reportedComputeFromText("**GPUs** | 1920 H100 | **Training time** | 21 days | **Dates** | October 2024 - November 2024 |", "card")
	selection := compiledSelection{from: time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC), to: time.Date(2025, 12, 31, 23, 59, 59, 0, time.UTC)}
	if compute == nil || len(compute.TrainingYears) != 1 || compute.TrainingYears[0] != 2024 {
		t.Fatalf("training year was not extracted: %#v", compute)
	}
	if reportedComputeFallsInSelection(compute, selection) {
		t.Fatal("explicit 2024 run must not be charged to 2025")
	}
}

func TestReportedTrainingYearUsesTheDatesFieldOnly(t *testing.T) {
	compute := reportedComputeFromText("**GPUs:** 128 H100<br>**Training time:** 2 days<br>**Dates:** Trained in February 2024<br>Dataset cutoff dates: 2025", "card")
	if compute == nil || len(compute.TrainingYears) != 1 || compute.TrainingYears[0] != 2024 {
		t.Fatalf("training dates leaked into adjacent fields: %#v", compute)
	}
}

func TestExplicitTrainingYearOverridesRepositoryCreationProxy(t *testing.T) {
	selection := compiledSelection{from: time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC), to: time.Date(2025, 12, 31, 23, 59, 59, 0, time.UTC)}
	compute := &reportedTrainingCompute{TrainingYears: []int{2025}, EarliestCardCreatedAt: "2024-03-01T00:00:00Z"}
	if !reportedComputeFallsInSelection(compute, selection) {
		t.Fatal("explicit 2025 compute date lost to an older repository creation proxy")
	}
	claim := &reportedScratchClaim{TrainingYears: []int{2025}, EarliestCardCreatedAt: "2024-03-01T00:00:00Z"}
	if !scratchClaimFallsInSelection(claim, selection) {
		t.Fatal("explicit 2025 scratch date lost to an older repository creation proxy")
	}
}

func TestBulkParquetProjectsAndParsesDescriptions(t *testing.T) {
	type wideRow struct {
		ModelID string `parquet:"modelId"`
		Card    string `parquet:"card"`
		Unused  string `parquet:"unused"`
	}
	description := "Fine-tuning took 2 hours on 4x H100 GPUs."
	path := filepath.Join(t.TempDir(), "cards.parquet")
	if err := parquet.WriteFile(path, []wideRow{{ModelID: "org/wanted", Card: description, Unused: "ignored"}, {ModelID: "org/other", Card: description}}); err != nil {
		t.Fatal(err)
	}
	got, err := scanReportedComputeParquet(context.Background(), path, []catalogModel{{ID: "org/wanted"}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got["org/wanted"] == nil || got["org/wanted"].GPUHours != 8 || got["org/other"] != nil {
		t.Fatalf("unexpected bulk result: %#v", got)
	}
}

func TestBulkParquetFindsEarlierScratchCardOutsideWantedSet(t *testing.T) {
	type row struct {
		ModelID string `parquet:"modelId"`
		Created string `parquet:"createdAt"`
		Card    string `parquet:"card"`
	}
	card := "This model was trained from scratch on our corpus."
	path := filepath.Join(t.TempDir(), "cards.parquet")
	if err := parquet.WriteFile(path, []row{
		{ModelID: "original/model", Created: "2024-03-01T00:00:00Z", Card: card},
		{ModelID: "mirror/model", Created: "2025-03-01T00:00:00Z", Card: card},
	}); err != nil {
		t.Fatal(err)
	}
	compute, claims, computeEarliest, earliest, err := scanTrainingSignalsParquet(context.Background(), path, []catalogModel{{ID: "mirror/model"}}, nil, map[string]bool{"mirror/model": true}, nil)
	if err != nil {
		t.Fatal(err)
	}
	claim := claims["mirror/model"]
	if claim == nil || earliest["run:"+claim.RunHash] != "2024-03-01T00:00:00Z" {
		t.Fatalf("older card was not detected: claim=%#v earliest=%#v", claim, earliest)
	}
	if len(compute) != 0 || len(computeEarliest) != 0 {
		t.Fatalf("compute parsing ran despite an empty compute candidate set: compute=%#v earliest=%#v", compute, computeEarliest)
	}
}

func TestBulkParquetFindsEarlierReportedRunOutsideWantedSet(t *testing.T) {
	type row struct {
		ModelID string `parquet:"modelId"`
		Created string `parquet:"createdAt"`
		Card    string `parquet:"card"`
	}
	card := "Fine-tuned for 2 days on 4x H100 GPUs."
	path := filepath.Join(t.TempDir(), "cards.parquet")
	if err := parquet.WriteFile(path, []row{
		{ModelID: "original/model", Created: "2024-03-01T00:00:00Z", Card: card},
		{ModelID: "mirror/model", Created: "2025-03-01T00:00:00Z", Card: card},
	}); err != nil {
		t.Fatal(err)
	}
	compute, _, earliest, _, err := scanTrainingSignalsParquet(context.Background(), path, []catalogModel{{ID: "mirror/model"}}, map[string]bool{"mirror/model": true}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	resolved := compute["mirror/model"]
	if resolved == nil || earliest["run:"+resolved.RunHash] != "2024-03-01T00:00:00Z" {
		t.Fatalf("older compute card was not detected: compute=%#v earliest=%#v", resolved, earliest)
	}
}

func TestPotentialScratchTextIsCaseInsensitive(t *testing.T) {
	if !potentialScratchText("This model was TrAiNeD FrOm ScRaTcH.") {
		t.Fatal("mixed-case scratch claim was not recognized")
	}
}

func TestRateLimitDelay(t *testing.T) {
	header := http.Header{}
	header.Set("RateLimit", `"api";r=0;t=42`)
	if got := rateLimitDelay(header); got != 43*time.Second {
		t.Fatalf("rateLimitDelay() = %s", got)
	}
}

func TestCatalogScenarioConfigIsValid(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("..", "..", "configs", "catalog.scenarios.json"))
	if err != nil {
		t.Fatal(err)
	}
	var config catalogConfig
	decoder := json.NewDecoder(strings.NewReader(string(data)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&config); err != nil {
		t.Fatalf("decode catalog.scenarios.json: %v", err)
	}
	applyCatalogDefaults(&config)
	if _, _, err := compileSelections(config, time.Date(2026, 8, 17, 0, 0, 0, 0, time.UTC)); err != nil {
		t.Fatalf("compile catalog.scenarios.json: %v", err)
	}
	if len(config.Selections) < 10 {
		t.Fatalf("scenario config has only %d selections", len(config.Selections))
	}
}

func TestProduction2025ConfigIsValid(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("..", "..", "configs", "catalog.2025-production.json"))
	if err != nil {
		t.Fatal(err)
	}
	var config catalogConfig
	decoder := json.NewDecoder(strings.NewReader(string(data)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&config); err != nil {
		t.Fatalf("decode catalog.2025-production.json: %v", err)
	}
	applyCatalogDefaults(&config)
	selections, _, err := compileSelections(config, time.Date(2026, 8, 19, 0, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	if len(selections) != 1 || selections[0].from.Year() != 2025 || selections[0].to.Year() != 2025 || selections[0].config.Limit != 0 || selections[0].config.RequireReportedCompute || !selections[0].config.RequireReportedDerivativeCompute || !selections[0].config.RequireExplicitScratchForBase || !selections[0].config.CountReportedDerivativeCosts || len(selections[0].config.ComputeProfiles) != 1 {
		t.Fatalf("production selection does not enforce the requested scope: %#v", selections)
	}
	if !boolValue(config.Scan.ResolveReportedCompute, false) {
		t.Fatal("production config must resolve reported compute")
	}
	if !boolValue(config.Scan.ResolveScratchClaims, false) {
		t.Fatal("production config must resolve explicit scratch claims")
	}
	if boolValue(config.Scan.FetchIndividualReadmes, true) || len(config.Scan.BulkModelCardsURLs) == 0 || config.Scan.CatalogCheckpoint == "" {
		t.Fatal("production config must use bulk cards and a catalog checkpoint without individual README requests")
	}
	if boolValue(config.Owners.Enabled, true) {
		t.Fatal("production config must not issue per-owner profile requests")
	}
}

func TestLLMMarket2025ConfigIsValid(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("..", "..", "configs", "catalog.2025-llm-market.json"))
	if err != nil {
		t.Fatal(err)
	}
	var config catalogConfig
	decoder := json.NewDecoder(strings.NewReader(string(data)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&config); err != nil {
		t.Fatalf("decode catalog.2025-llm-market.json: %v", err)
	}
	applyCatalogDefaults(&config)
	selections, _, err := compileSelections(config, time.Date(2026, 8, 20, 0, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	if len(selections) != 1 || !selections[0].config.TargetLLMOnly || !selections[0].config.FormulaRequiresExplicitScratch || selections[0].config.RequireExplicitScratchForBase || !selections[0].config.CountReportedDerivativeCosts || len(selections[0].config.ModelKinds) != 6 {
		t.Fatalf("LLM market selection does not enforce the requested taxonomy: %#v", selections)
	}
	if !boolValue(config.Scan.ResolveReportedCompute, false) || !boolValue(config.Scan.ResolveScratchClaims, false) || boolValue(config.Scan.FetchIndividualReadmes, true) {
		t.Fatal("LLM market config must use bulk scratch and derivative-compute analysis without per-README requests")
	}
}

func TestCatalogLoggerWritesJSONLAndFiltersLevels(t *testing.T) {
	stderr := false
	directory := t.TempDir()
	logger, err := newCatalogLogger(catalogLoggingConfig{Level: "info", File: "run.jsonl", Stderr: &stderr}, directory)
	if err != nil {
		t.Fatal(err)
	}
	logger.debug("hidden", "must not be written", nil)
	logger.warn("visible", "warning text", map[string]any{"repo_id": "org/model"})
	if err := logger.close(); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(directory, "run.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	text := string(data)
	if strings.Contains(text, "hidden") || !strings.Contains(text, `"event":"visible"`) || !strings.Contains(text, `"repo_id":"org/model"`) {
		t.Fatalf("unexpected log content: %s", text)
	}
}

func TestAdapterParameterResolutionErrorIsPreserved(t *testing.T) {
	model := catalogModel{ID: "org/adapter", BaseModels: catalogBaseModels{Relation: "adapter", Models: []catalogBaseModel{{ID: "org/base"}}}}
	parameters, message := effectiveParameters(model, nil, map[string]string{"org/base": "metadata unavailable"})
	if parameters != 0 || message != "metadata unavailable" {
		t.Fatalf("effectiveParameters() = %d, %q", parameters, message)
	}
}

func TestLocalLLMReviewRequiresVerbatimEvidence(t *testing.T) {
	review := localLLMReview{Kind: "independent_base", Confidence: "high", IndependentlyPretrained: true, Evidence: "invented evidence"}
	if err := validateLocalLLMReview(&review, "This model was pretrained on 2T tokens."); err == nil {
		t.Fatal("invented evidence was accepted")
	}
	review.Evidence = "pretrained on 2T tokens"
	if err := validateLocalLLMReview(&review, "This model was pretrained on 2T tokens."); err != nil {
		t.Fatalf("verbatim evidence rejected: %v", err)
	}
}

func TestLocalLLMTrainingYearMustAppearInEvidence(t *testing.T) {
	review := localLLMReview{
		Kind: "independent_base", Confidence: "high", IndependentlyPretrained: true,
		TrainingYears: []int{2025}, Evidence: "Pretraining finished in 2024.",
	}
	if err := validateLocalLLMReview(&review, "Pretraining finished in 2024."); err != nil {
		t.Fatal(err)
	}
	if len(review.TrainingYears) != 1 || review.TrainingYears[0] != 2024 {
		t.Fatalf("expected evidence-derived year, got %v", review.TrainingYears)
	}
}

func TestLocalLLMJSONCanFollowReasoningText(t *testing.T) {
	card := "The current model was pretrained from scratch in 2025."
	content := `<think>First consider {broken JSON}.</think>
{"kind":"independent_base","confidence":"high","independently_pretrained":true,"canonical_training_run":"org/model-1b","training_years":[2025],"evidence":"pretrained from scratch in 2025","reason":"direct evidence"}`
	review, err := parseLocalLLMContent(content, card)
	if err != nil {
		t.Fatal(err)
	}
	if review.Kind != "independent_base" || review.CanonicalTrainingRun != "org/model-1b" || len(review.TrainingYears) != 1 || review.TrainingYears[0] != 2025 {
		t.Fatalf("unexpected parsed review: %#v", review)
	}
}

func TestDeclaredBaseNeedsLocalEvidenceWhenPopularityFiltersAreDisabled(t *testing.T) {
	declared := &reportedScratchClaim{EvidenceType: "declared_base_model", Evidence: "Model-7B-Base", ActiveParametersB: 2, TrainingTokensT: 3, TrainingYears: []int{2025}}
	if got, deterministic := resolveScratchClaimForReview(true, declared, &localLLMReview{Kind: "unknown", Confidence: "low"}); got != nil || deterministic {
		t.Fatalf("weak declared-base claim must not become costed: got=%#v deterministic=%v", got, deterministic)
	}
	review := &localLLMReview{
		Kind: "independent_base", Confidence: "high", IndependentlyPretrained: true,
		Evidence: "pretrained from scratch in 2025", TrainingYears: []int{2025},
	}
	got, deterministic := resolveScratchClaimForReview(true, declared, review)
	if got == nil || deterministic || got.EvidenceType != "local_llm_high" || got.ActiveParametersB != 2 || got.TrainingTokensT != 3 || len(got.TrainingYears) != 1 || got.TrainingYears[0] != 2025 {
		t.Fatalf("expected reviewed scratch claim: got=%#v deterministic=%v", got, deterministic)
	}
	explicit := &reportedScratchClaim{EvidenceType: "explicit_scratch", Evidence: "trained from scratch"}
	if got, deterministic := resolveScratchClaimForReview(true, explicit, nil); got != explicit || !deterministic {
		t.Fatalf("explicit deterministic evidence should survive: got=%#v deterministic=%v", got, deterministic)
	}
}

func TestLocalLLMOpenAICompatibleClient(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"data":[{"id":"local-model"}]}`))
		case "/v1/chat/completions":
			var request openAIChatRequest
			if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
				t.Errorf("decode chat request: %v", err)
			}
			if request.ResponseFormat["type"] != "json_object" {
				t.Errorf("response format type = %v, want json_object", request.ResponseFormat["type"])
			}
			schema, ok := request.ResponseFormat["schema"].(map[string]any)
			if !ok {
				t.Errorf("response format schema = %#v, want object", request.ResponseFormat["schema"])
			} else if schema["additionalProperties"] != false {
				t.Errorf("schema additionalProperties = %v, want false", schema["additionalProperties"])
			}
			w.Header().Set("Content-Type", "application/json")
			content := `{"kind":"independent_base","confidence":"high","independently_pretrained":true,"upstream_model":"","canonical_training_run":"org/model-7b","training_years":[2025],"evidence":"pretrained from scratch on 2T tokens","reason":"direct evidence"}`
			_ = json.NewEncoder(w).Encode(map[string]any{"choices": []any{map[string]any{"message": map[string]any{"content": content}}}})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	config := localLLMConfig{BaseURL: server.URL + "/v1", Model: "local-model", TimeoutSeconds: 5}
	client := &http.Client{Timeout: 5 * time.Second}
	if err := probeLocalLLM(context.Background(), client, config); err != nil {
		t.Fatal(err)
	}
	input := localLLMReviewInput{RepoID: "org/model-7b", Card: "The current model was pretrained from scratch on 2T tokens.", CardHash: "hash", CacheKey: "key"}
	review, err := callLocalLLM(context.Background(), client, config, input)
	if err != nil {
		t.Fatal(err)
	}
	if review.Kind != "independent_base" || review.Confidence != "high" || review.CanonicalTrainingRun != "org/model-7b" || review.Source == "" {
		t.Fatalf("unexpected review: %#v", review)
	}
}

func TestLocalLLMHighAndMediumCostTiers(t *testing.T) {
	profile := computeProfile{Name: "formula", GPUName: "H100", GPUTFLOPS: 989, Efficiency: 0.4, GPUHourCostUSD: 1.85, TokensPerParameter: 20, Machines: 1, GPUsPerMachine: 1}
	selection := selectionConfig{Name: "s", FormulaRequiresExplicitScratch: true, ComputeProfiles: []string{"formula"}}
	model := catalogModel{ID: "org/new-7b", Safetensors: catalogSafetensors{Total: 7_000_000_000}}
	highReview := &localLLMReview{Kind: "independent_base", Confidence: "high", IndependentlyPretrained: true, Evidence: "pretrained from scratch", CanonicalTrainingRun: "org/new-7b", Source: "local", CardHash: "card"}
	highClaim := scratchClaimFromLocalLLM(highReview)
	high := modelToRecordWithReview("https://huggingface.co", model, selection, model.Safetensors.Total, "", map[string]computeProfile{"formula": profile}, nil, highClaim, highReview, false)
	if high.TrainingCostUSD == nil || high.LowerTrainingCostUSD != nil || high.UpperTrainingCostUSD == nil || high.TrainingCostTier != "llm_high" {
		t.Fatalf("unexpected high-confidence tier: %#v", high)
	}
	mediumReview := *highReview
	mediumReview.Confidence = "medium"
	mediumClaim := scratchClaimFromLocalLLM(&mediumReview)
	medium := modelToRecordWithReview("https://huggingface.co", model, selection, model.Safetensors.Total, "", map[string]computeProfile{"formula": profile}, nil, mediumClaim, &mediumReview, false)
	if medium.TrainingCostUSD != nil || medium.LowerTrainingCostUSD != nil || medium.UpperTrainingCostUSD == nil || medium.TrainingCostTier != "llm_medium_upper_only" {
		t.Fatalf("unexpected medium-confidence tier: %#v", medium)
	}
}

func TestLocalLLMHighDerivativeOverridesResidualBase(t *testing.T) {
	model := catalogModel{ID: "org/ambiguous-model", Safetensors: catalogSafetensors{Total: 7_000_000_000}}
	review := &localLLMReview{Kind: "continued_pretraining", Confidence: "high", Evidence: "continued pretraining"}
	if got := resolvedCatalogModelKindWithReview(model, &reportedScratchClaim{Evidence: "trained from scratch"}, nil, review); got != "finetune" {
		t.Fatalf("reviewed kind = %q, want finetune", got)
	}
}

func TestLocalLLMCanonicalRunDeduplication(t *testing.T) {
	aCost, bCost := 100.0, 100.0
	aUpper, bUpper := 100.0, 100.0
	records := []catalogRecord{
		{Selection: "s", RepoID: "mirror/model", EffectiveParameters: 7, Likes: 1, TrainingCostUSD: &aCost, UpperTrainingCostUSD: &aUpper, LocalLLMReview: &localLLMReview{Kind: "independent_base", Confidence: "high", CanonicalTrainingRun: "org/model"}},
		{Selection: "s", RepoID: "org/model", EffectiveParameters: 7, Likes: 10, TrainingCostUSD: &bCost, UpperTrainingCostUSD: &bUpper, LocalLLMReview: &localLLMReview{Kind: "independent_base", Confidence: "high", CanonicalTrainingRun: "org/model"}},
	}
	deduplicateLocalLLMTrainingRuns(records)
	if records[0].TrainingCostUSD != nil || records[0].UpperTrainingCostUSD != nil || records[1].TrainingCostUSD == nil {
		t.Fatalf("unexpected local LLM deduplication: %#v", records)
	}
}

func TestLocalLLMMarketConfigIsValid(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("..", "..", "configs", "catalog.2025-llm-market.local-llm.json"))
	if err != nil {
		t.Fatal(err)
	}
	var config catalogConfig
	decoder := json.NewDecoder(strings.NewReader(string(data)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&config); err != nil {
		t.Fatal(err)
	}
	applyCatalogDefaults(&config)
	if !config.LocalLLM.Enabled || config.LocalLLM.Workers != 4 || config.LocalLLM.MaxInputChars != 12000 || config.Selections[0].DeclaredBaseMinLikes != 0 || config.Selections[0].DeclaredBaseMinDownloads != 0 {
		t.Fatalf("unexpected local LLM config: %#v", config.LocalLLM)
	}
}

func TestSamplingManifestIsDeterministicAndWeightedByCoverageStratum(t *testing.T) {
	selection := compiledSelection{config: selectionConfig{Name: "text", TargetLLMOnly: true}}
	makeCandidate := func(id string, parameters int64, covered bool) samplingCandidate {
		return samplingCandidate{Model: catalogModel{ID: id, Author: "org", Safetensors: catalogSafetensors{Total: parameters}}, Selection: selection, Scope: "text_llm", Params: parameters, Bucket: parameterRange(parameters), Covered: covered}
	}
	candidates := []samplingCandidate{
		makeCandidate("org/large-covered", 70_000_000_000, true),
		makeCandidate("org/large-missing", 70_000_000_000, false),
		makeCandidate("org/small-covered-a", 3_000_000_000, true),
		makeCandidate("org/small-covered-b", 3_000_000_000, true),
		makeCandidate("org/small-missing-a", 3_000_000_000, false),
		makeCandidate("org/small-missing-b", 3_000_000_000, false),
	}
	include := true
	config := marketSamplingConfig{Seed: 2025, CensusMinParametersB: 34, SamplePerStratum: 1, MaxTargetedReadmes: 2, IncludeCoveredCensus: &include, IncludeMissingCensus: &include}
	first, plan, err := buildSamplingManifest(candidates, config)
	if err != nil {
		t.Fatal(err)
	}
	second, _, err := buildSamplingManifest(candidates, config)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatal("sampling manifest is not deterministic for a fixed seed")
	}
	if len(first) != 4 || plan.CensusEntries != 2 || plan.TargetedREADMERequests != 2 {
		t.Fatalf("manifest=%d census=%d targeted=%d, want 4/2/2", len(first), plan.CensusEntries, plan.TargetedREADMERequests)
	}
	for _, entry := range first {
		if entry.Census && entry.SamplingWeight != 1 {
			t.Fatalf("census entry %s has weight %v", entry.RepoID, entry.SamplingWeight)
		}
		if !entry.Census && entry.SamplingWeight != 2 {
			t.Fatalf("sample entry %s has weight %v, want 2", entry.RepoID, entry.SamplingWeight)
		}
	}
}

func TestLocalLLMParameterCountRequiresVerbatimEvidence(t *testing.T) {
	card := "The current model contains 7 billion total parameters."
	review := localLLMReview{Kind: "unknown", Confidence: "low", ReportedParametersB: 7, ParameterEvidence: "7 billion total parameters"}
	if err := validateLocalLLMReview(&review, card); err != nil {
		t.Fatalf("valid parameter evidence rejected: %v", err)
	}
	review.ParameterEvidence = "8 billion total parameters"
	if err := validateLocalLLMReview(&review, card); err != nil {
		t.Fatalf("bad optional parameter evidence discarded the lineage review: %v", err)
	}
	if review.ReportedParametersB != 0 || review.ParameterEvidence != "" {
		t.Fatal("invented parameter evidence was not cleared")
	}
}

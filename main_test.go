package main

import (
	"net/http"
	"testing"
	"time"
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
	}
	for _, tt := range tests {
		model := catalogModel{ID: tt.id, Tags: tt.tags, BaseModels: catalogBaseModels{Relation: tt.relation}}
		if got := catalogModelKind(model); got != tt.want {
			t.Errorf("catalogModelKind(%q) = %q, want %q", tt.id, got, tt.want)
		}
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

func TestRateLimitDelay(t *testing.T) {
	header := http.Header{}
	header.Set("RateLimit", `"api";r=0;t=42`)
	if got := rateLimitDelay(header); got != 43*time.Second {
		t.Fatalf("rateLimitDelay() = %s", got)
	}
}

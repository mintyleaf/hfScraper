package main

import (
	"regexp"
	"strings"
)

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

func firstBaseModel(model catalogModel) string {
	if len(model.BaseModels.Models) == 0 {
		return ""
	}
	return model.BaseModels.Models[0].ID
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

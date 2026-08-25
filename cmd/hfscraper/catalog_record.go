package main

import (
	"math"
	"strings"
)

// isDiffusionModel checks if a model is a diffusion model based on pipeline tag
func isDiffusionModel(model catalogModel) bool {
	return catalogIsTargetDiffusion(model)
}

// hasDiffusionProfile checks if a compute profile is enabled for diffusion models.
// A diffusion-enabled profile must have both DiffusionImageBudget and LatentSequenceLength > 0.
func hasDiffusionProfile(profile computeProfile) bool {
	return profile.DiffusionImageBudget > 0 && profile.LatentSequenceLength > 0
}

func modelToRecord(endpoint string, model catalogModel, selection selectionConfig, params int64, parameterError string, profiles map[string]computeProfile, reported *reportedTrainingCompute, scratch *reportedScratchClaim) catalogRecord {
	return modelToRecordWithReview(endpoint, model, selection, params, parameterError, profiles, reported, scratch, nil, scratch != nil)
}

func modelToRecordWithReview(endpoint string, model catalogModel, selection selectionConfig, params int64, parameterError string, profiles map[string]computeProfile, reported *reportedTrainingCompute, scratch *reportedScratchClaim, review *localLLMReview, deterministicScratch bool) catalogRecord {
	var billions *float64
	if params > 0 {
		value := float64(params) / 1e9
		billions = &value
	}
	kind := resolvedCatalogModelKindWithReview(model, scratch, reported, review)
	record := catalogRecord{
		Selection: selection.Name, Description: selection.Description,
		RepoID: model.ID, RepoURL: strings.TrimRight(endpoint, "/") + "/" + model.ID,
		Owner: modelOwner(model), CreatedAt: model.CreatedAt, LastModified: model.LastModified,
		PipelineTag: model.PipelineTag, LibraryName: model.LibraryName,
		ModelKind: kind, BaseModel: firstBaseModel(model),
		OwnParameters: model.Safetensors.Total, EffectiveParameters: params, ParametersB: billions,
		Downloads: model.Downloads, Likes: model.Likes, Tags: model.Tags,
		ScratchClaim: scratch, LocalLLMReview: review,
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
		record.LowerTrainingCostUSD = floatPointer(cost)
		record.UpperTrainingCostUSD = floatPointer(cost)
		record.TrainingCostMethod = reported.CostMethod
		record.TrainingCostTier = "reported"
		record.Compute = append(record.Compute, computeEstimate{
			Profile: "reported_training_time", GPUName: reported.GPUName,
			TotalGPUs: reported.GPUCount, GPUHours: reported.GPUHours,
			WallDays: reported.Hours / 24, CostUSD: reported.CostUSD,
			Fraction: 1, Method: reported.CostMethod, Source: reported.Source,
		})
		return record
	}

	// Check if this is a diffusion model
	isDiffusion := isDiffusionModel(model)

	formulaRequiresScratch := selection.RequireExplicitScratchForBase || selection.FormulaRequiresExplicitScratch

	if formulaRequiresScratch && scratch == nil {
		return record
	}

	if review == nil && scratch != nil && scratch.EvidenceType == "declared_base_model" {
		record.Errors = append(record.Errors, "repository name says base but model card does not prove independent pretraining")
		return record
	}

	for _, profileName := range selection.ComputeProfiles {
		if profile, ok := profiles[profileName]; ok && params > 0 {
			literal := estimateCompute(params, kind, profile)
			literal.Method = "formula_txt_literal_total_parameters"
			literal.Source = "formula.txt"

			// For diffusion models with valid diffusion profiles, use special compute estimation
			var estimate computeEstimate
			if isDiffusion && hasDiffusionProfile(profile) {
				estimate = estimateDiffusionBaseTrainingCompute(params, scratch, profile)
			} else if formulaRequiresScratch && scratch != nil {
				estimate = estimateBaseTrainingCompute(params, scratch, profile)
			} else {
				estimate = literal
			}

			if math.IsNaN(estimate.GPUHours) || math.IsInf(estimate.GPUHours, 0) || math.IsNaN(estimate.WallDays) || math.IsInf(estimate.WallDays, 0) || math.IsNaN(estimate.CostUSD) || math.IsInf(estimate.CostUSD, 0) {
				record.Errors = append(record.Errors, "compute estimate overflow for profile "+profileName)
				continue
			}

			if estimate.Method != literal.Method || math.Abs(estimate.CostUSD-literal.CostUSD) > 0.005 {
				record.Compute = append(record.Compute, literal)
			}
			record.Compute = append(record.Compute, estimate)

			if record.TrainingCostUSD == nil && record.UpperTrainingCostUSD == nil {
				cost := estimate.CostUSD
				record.UpperTrainingCostUSD = floatPointer(cost)
				if scratch != nil && scratch.EvidenceType == "local_llm_medium" && !deterministicScratch {
					record.TrainingCostTier = "llm_medium_upper_only"
					record.TrainingCostMethod = estimate.Method
				} else {
					record.TrainingCostUSD = floatPointer(cost)
					record.TrainingCostMethod = estimate.Method
					if record.TrainingCostMethod == "" {
						record.TrainingCostMethod = "parameter_scaling_estimate"
					}
					if scratch != nil && scratch.EvidenceType == "local_llm_high" && !deterministicScratch {
						record.TrainingCostTier = "llm_high"
					} else {
						record.TrainingCostTier = "deterministic"
						record.LowerTrainingCostUSD = floatPointer(cost)
					}
				}
			}
		}
	}

	return record
}

func floatPointer(value float64) *float64 {
	copy := value
	return &copy
}

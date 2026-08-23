package main

import (
	"math"
	"strings"
)

// isDiffusionModel checks if a model is a diffusion model based on pipeline tag
func isDiffusionModel(model catalogModel) bool {
	pipeline := strings.ToLower(model.PipelineTag)
	switch pipeline {
	case "text-to-image", "image-to-image", "text-to-video":
		return true
	default:
		// Fallback: check for diffusion-related tags in non-text models 
		joined := strings.ToLower(model.ID + " " + strings.Join(model.Tags, " "))
		if nonTextModelTagRE.MatchString(joined) {
			return true
		}
		return false
	}
}

// hasDiffusionProfile checks if a compute profile is enabled for diffusion models.
// A diffusion-enabled profile must have both DiffusionImageBudget and LatentSequenceLength > 0.
func hasDiffusionProfile(profile computeProfile) bool {
	return profile.DiffusionImageBudget > 0 && profile.LatentSequenceLength > 0
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
	
	// Check if this is a diffusion model
	isDiffusion := isDiffusionModel(model)
	
	formulaRequiresScratch := selection.RequireExplicitScratchForBase || selection.FormulaRequiresExplicitScratch
	
	// For diffusion models, use estimateDiffusionBaseTrainingCompute even without scratch-claim
	if isDiffusion {
		// Check if we should proceed with compute estimation for this model
		if formulaRequiresScratch && scratch == nil {
			return record
		}
	} else {
		// Non-diffusion models follow original logic
		if formulaRequiresScratch && scratch == nil {
			return record
		}
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
			
			// For diffusion models with valid diffusion profiles, use special compute estimation
			var estimate computeEstimate
			if isDiffusion && hasDiffusionProfile(profile) {
				estimate = estimateDiffusionBaseTrainingCompute(params, scratch, profile)
				// Use the diffusion prefix for method name to enable automatic deduplication
				estimate.Method = "formula_txt_diffusion_" + strings.TrimPrefix(estimate.Method, "formula_txt_")
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
			
			// Priority for reported compute is maintained for non-diffusion models
			if record.TrainingCostUSD == nil && !isDiffusion {
				cost := estimate.CostUSD
				record.TrainingCostUSD = &cost
				record.TrainingCostMethod = estimate.Method
				if record.TrainingCostMethod == "" {
					record.TrainingCostMethod = "parameter_scaling_estimate"
				}
			}
			
			// For diffusion models, we also set the training cost if it's not already set by reported compute
			if record.TrainingCostUSD == nil && isDiffusion {
				cost := estimate.CostUSD
				record.TrainingCostUSD = &cost
				record.TrainingCostMethod = estimate.Method
			}
		}
	}
	
	return record
}
package main

import (
	"math"
	"strings"
)

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

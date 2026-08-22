package main

func estimateCompute(parameters int64, kind string, profile computeProfile) computeEstimate {
	fraction := 1.0
	if kind == "finetune" || kind == "adapter" {
		fraction = profile.FinetuneCostFraction
	}
	n := float64(parameters)
	flops := 6 * n * (profile.TokensPerParameter * n) * fraction
	gpuHours := flops / (profile.GPUTFLOPS * 1e12 * 3600 * profile.Efficiency)
	totalGPUs := profile.Machines * profile.GPUsPerMachine
	return computeEstimate{
		Profile: profile.Name, GPUName: profile.GPUName, Machines: profile.Machines,
		TotalGPUs: totalGPUs, GPUHours: gpuHours, WallDays: gpuHours / float64(totalGPUs) / 24,
		CostUSD: gpuHours * profile.GPUHourCostUSD, Fraction: fraction,
	}
}

// estimateBaseTrainingCompute keeps formula.txt's 6*N*T accounting while using
// stronger public evidence when the card supplies it. For sparse/MoE models N
// is the activated parameter count; T is the reported pretraining token count.
// Missing values fall back independently to total parameters and 20 tokens per
// total parameter, respectively.
func estimateBaseTrainingCompute(parameters int64, claim *reportedScratchClaim, profile computeProfile) computeEstimate {
	totalParameters := float64(parameters)
	computeParameters := totalParameters
	method := "formula_txt_parameter_scaling"
	if claim != nil && claim.ActiveParametersB > 0 && claim.ActiveParametersB*1e9 < totalParameters {
		computeParameters = claim.ActiveParametersB * 1e9
		method = "formula_txt_moe_active_parameter_scaling"
	}
	tokens := profile.TokensPerParameter * totalParameters
	if claim != nil && claim.TrainingTokensT > 0 {
		tokens = claim.TrainingTokensT * 1e12
		if computeParameters < totalParameters {
			method = "reported_pretraining_tokens_active_parameters"
		} else {
			method = "reported_pretraining_tokens_total_parameters"
		}
	}
	flops := 6 * computeParameters * tokens
	gpuHours := flops / (profile.GPUTFLOPS * 1e12 * 3600 * profile.Efficiency)
	totalGPUs := profile.Machines * profile.GPUsPerMachine
	return computeEstimate{
		Profile: profile.Name, GPUName: profile.GPUName, Machines: profile.Machines,
		TotalGPUs: totalGPUs, GPUHours: gpuHours, WallDays: gpuHours / float64(totalGPUs) / 24,
		CostUSD: gpuHours * profile.GPUHourCostUSD, Fraction: 1,
		Method: method, Source: "formula.txt plus public model-card training evidence",
	}
}

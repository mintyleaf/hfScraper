package main

import (
	"strconv"
	"strings"
	"time"
	"unicode"
)

func isBaseTrainingCostMethod(method string) bool {
	return strings.HasPrefix(method, "formula_txt_") || strings.HasPrefix(method, "reported_pretraining_tokens_")
}

// deduplicateReportedTrainingRuns prevents copied model cards, conversions and
// mirrors from charging the same disclosed run repeatedly. The oldest repo is
// preferred as the canonical publication; popularity breaks timestamp ties.
// This is deliberately conservative: identical cards may occasionally have
// been reused for separate runs, so the resulting sum is a documented lower
// bound rather than an invented point estimate.
func deduplicateReportedTrainingRuns(records []catalogRecord) {
	groups := make(map[string][]int)
	for index := range records {
		reported := records[index].ReportedCompute
		if reported == nil || records[index].TrainingCostUSD == nil || (records[index].TrainingCostMethod != "gpu_hours_x_hourly_rate" && records[index].TrainingCostMethod != "reported_total_cost") {
			continue
		}
		fingerprint := reported.RunHash
		if fingerprint == "" {
			fingerprint = reported.EvidenceHash
		}
		if fingerprint == "" {
			continue
		}
		key := records[index].Selection + "\x00" + fingerprint
		groups[key] = append(groups[key], index)
	}
	for _, indices := range groups {
		if len(indices) < 2 {
			continue
		}
		canonical := indices[0]
		for _, index := range indices[1:] {
			a, b := records[index], records[canonical]
			if a.CreatedAt < b.CreatedAt || (a.CreatedAt == b.CreatedAt && (a.Likes > b.Likes || (a.Likes == b.Likes && a.Downloads > b.Downloads))) {
				canonical = index
			}
		}
		for _, index := range indices {
			if index == canonical {
				continue
			}
			records[index].TrainingCostUSD = nil
			records[index].LowerTrainingCostUSD = nil
			records[index].UpperTrainingCostUSD = nil
			records[index].TrainingCostMethod = "duplicate_reported_training_run"
			records[index].Compute = nil
			records[index].Errors = append(records[index].Errors, "cost counted once under canonical repository "+records[canonical].RepoID)
		}
	}

	// Some projects publish a dated snapshot beside an undated repository and
	// report cumulative compute in the newer card. Exact evidence hashes cannot
	// join those snapshots because the cumulative total changed. Collapse only
	// a very narrow same-owner/same-kind/same-base/same-parameter family where
	// at least one basename ends in an explicit YYYYMMDD-like date, and retain
	// the largest disclosed cumulative cost.
	type trajectoryGroup struct {
		indices []int
		dated   bool
	}
	trajectories := make(map[string]*trajectoryGroup)
	for index := range records {
		record := records[index]
		if record.ReportedCompute == nil || record.TrainingCostUSD == nil || (record.ModelKind != "finetune" && record.ModelKind != "adapter") {
			continue
		}
		parts := strings.SplitN(record.RepoID, "/", 2)
		if len(parts) != 2 {
			continue
		}
		name := strings.ToLower(parts[1])
		dated := reportedTrajectoryDateSuffixRE.MatchString(name)
		family := reportedTrajectoryDateSuffixRE.ReplaceAllString(name, "")
		key := record.Selection + "\x00" + strings.ToLower(record.Owner) + "\x00" + record.ModelKind + "\x00" + strings.ToLower(record.BaseModel) + "\x00" + family + "\x00" + strconv.FormatInt(record.EffectiveParameters, 10)
		group := trajectories[key]
		if group == nil {
			group = &trajectoryGroup{}
			trajectories[key] = group
		}
		group.indices = append(group.indices, index)
		group.dated = group.dated || dated
	}
	for _, group := range trajectories {
		if !group.dated || len(group.indices) < 2 {
			continue
		}
		canonical := group.indices[0]
		for _, index := range group.indices[1:] {
			a, b := records[index], records[canonical]
			if *a.TrainingCostUSD > *b.TrainingCostUSD || (*a.TrainingCostUSD == *b.TrainingCostUSD && a.CreatedAt > b.CreatedAt) {
				canonical = index
			}
		}
		for _, index := range group.indices {
			if index == canonical || records[index].TrainingCostUSD == nil {
				continue
			}
			records[index].TrainingCostUSD = nil
			records[index].LowerTrainingCostUSD = nil
			records[index].UpperTrainingCostUSD = nil
			records[index].TrainingCostMethod = "duplicate_reported_training_trajectory"
			records[index].Compute = nil
			records[index].Errors = append(records[index].Errors, "reported trajectory cost counted once under cumulative repository "+records[canonical].RepoID)
		}
	}
}

func deduplicateReportedDerivativeAgainstBase(records []catalogRecord) {
	baseRuns := make(map[string]string)
	for index := range records {
		reported := records[index].ReportedCompute
		if records[index].ModelKind != "base" || records[index].TrainingCostUSD == nil || reported == nil || reported.RunHash == "" {
			continue
		}
		baseRuns[records[index].Selection+"\x00"+reported.RunHash] = records[index].RepoID
	}
	for index := range records {
		if records[index].ModelKind != "finetune" && records[index].ModelKind != "adapter" {
			continue
		}
		reported := records[index].ReportedCompute
		if records[index].TrainingCostUSD == nil || reported == nil || reported.RunHash == "" {
			continue
		}
		if canonical := baseRuns[records[index].Selection+"\x00"+reported.RunHash]; canonical != "" {
			records[index].TrainingCostUSD = nil
			records[index].LowerTrainingCostUSD = nil
			records[index].UpperTrainingCostUSD = nil
			records[index].TrainingCostMethod = "duplicate_reported_base_training_run"
			records[index].Compute = nil
			records[index].Errors = append(records[index].Errors, "reported compute belongs to canonical base repository "+canonical)
		}
	}
}

func deduplicateScratchTrainingRuns(records []catalogRecord) {
	groups := make(map[string][]int)
	for index := range records {
		claim := records[index].ScratchClaim
		if claim == nil || records[index].TrainingCostUSD == nil || !isBaseTrainingCostMethod(records[index].TrainingCostMethod) {
			continue
		}
		fingerprint := claim.RunHash
		if fingerprint == "" {
			fingerprint = claim.CardHash
		}
		if fingerprint == "" {
			continue
		}
		key := records[index].Selection + "\x00" + fingerprint + "\x00" + strconv.FormatInt(records[index].EffectiveParameters, 10)
		groups[key] = append(groups[key], index)
	}
	for _, indices := range groups {
		if len(indices) < 2 {
			continue
		}
		canonical := indices[0]
		for _, index := range indices[1:] {
			a, b := records[index], records[canonical]
			if a.CreatedAt < b.CreatedAt || (a.CreatedAt == b.CreatedAt && (a.Likes > b.Likes || (a.Likes == b.Likes && a.Downloads > b.Downloads))) {
				canonical = index
			}
		}
		for _, index := range indices {
			if index == canonical {
				continue
			}
			records[index].TrainingCostUSD = nil
			records[index].LowerTrainingCostUSD = nil
			records[index].UpperTrainingCostUSD = nil
			records[index].TrainingCostMethod = "duplicate_scratch_training_run"
			records[index].Compute = nil
			records[index].Errors = append(records[index].Errors, "scratch cost counted once under canonical repository "+records[canonical].RepoID)
		}
	}

	// Repositories frequently publish every checkpoint of one trajectory with
	// slightly different generated cards. Exact hashes cannot join those, so a
	// second, deliberately narrow key strips only an explicit checkpoint/step
	// suffix and still requires the same owner and exact parameter count.
	trajectories := make(map[string][]int)
	for index := range records {
		if records[index].TrainingCostUSD == nil || !isBaseTrainingCostMethod(records[index].TrainingCostMethod) {
			continue
		}
		parts := strings.SplitN(records[index].RepoID, "/", 2)
		if len(parts) != 2 || !catalogCheckpointRE.MatchString(parts[1]) {
			continue
		}
		family := catalogCheckpointRE.ReplaceAllString(strings.ToLower(parts[1]), "")
		key := records[index].Selection + "\x00" + strings.ToLower(records[index].Owner) + "\x00" + family + "\x00" + strconv.FormatInt(records[index].EffectiveParameters, 10)
		trajectories[key] = append(trajectories[key], index)
	}
	for _, indices := range trajectories {
		if len(indices) < 2 {
			continue
		}
		canonical := indices[0]
		for _, index := range indices[1:] {
			a, b := records[index], records[canonical]
			if a.CreatedAt < b.CreatedAt || (a.CreatedAt == b.CreatedAt && (a.Likes > b.Likes || (a.Likes == b.Likes && a.Downloads > b.Downloads))) {
				canonical = index
			}
		}
		for _, index := range indices {
			if index == canonical {
				continue
			}
			records[index].TrainingCostUSD = nil
			records[index].LowerTrainingCostUSD = nil
			records[index].UpperTrainingCostUSD = nil
			records[index].TrainingCostMethod = "duplicate_scratch_training_trajectory"
			records[index].Compute = nil
			records[index].Errors = append(records[index].Errors, "scratch trajectory cost counted once under canonical repository "+records[canonical].RepoID)
		}
	}

	// Cross-owner mirrors commonly preserve the upstream repository basename
	// while changing only the namespace or serialization suffix. Collapse only
	// exact normalized basename + exact parameter-count matches. Popularity wins
	// because a mirror may have been uploaded slightly before the canonical org.
	families := make(map[string][]int)
	for index := range records {
		if records[index].TrainingCostUSD == nil || !isBaseTrainingCostMethod(records[index].TrainingCostMethod) {
			continue
		}
		family := canonicalBaseFamilyName(records[index].RepoID)
		if len(family) < 6 || family == "model" || family == "base-model" {
			continue
		}
		key := records[index].Selection + "\x00" + family + "\x00" + strconv.FormatInt(records[index].EffectiveParameters, 10)
		families[key] = append(families[key], index)
	}
	for _, indices := range families {
		if len(indices) < 2 {
			continue
		}
		canonical := indices[0]
		for _, index := range indices[1:] {
			a, b := records[index], records[canonical]
			if a.Likes > b.Likes || (a.Likes == b.Likes && (a.Downloads > b.Downloads || (a.Downloads == b.Downloads && a.CreatedAt < b.CreatedAt))) {
				canonical = index
			}
		}
		for _, index := range indices {
			if index == canonical {
				continue
			}
			records[index].TrainingCostUSD = nil
			records[index].LowerTrainingCostUSD = nil
			records[index].UpperTrainingCostUSD = nil
			records[index].TrainingCostMethod = "duplicate_base_family_mirror"
			records[index].Compute = nil
			records[index].Errors = append(records[index].Errors, "base-family cost counted once under canonical repository "+records[canonical].RepoID)
		}
	}
}

// deduplicateLocalLLMTrainingRuns joins rewritten mirrors which do not share an
// exact card/evidence hash. The local reviewer must supply a stable canonical
// run identity; exact parameter count remains mandatory so independently
// trained sizes in one family are never collapsed together.
func deduplicateLocalLLMTrainingRuns(records []catalogRecord) {
	groups := make(map[string][]int)
	for index := range records {
		review := records[index].LocalLLMReview
		if review == nil || review.Kind != "independent_base" || (review.Confidence != "high" && review.Confidence != "medium") || records[index].UpperTrainingCostUSD == nil {
			continue
		}
		canonicalRun := normalizeCanonicalRun(review.CanonicalTrainingRun)
		if len(canonicalRun) < 6 || canonicalRun == "unknown" {
			continue
		}
		key := records[index].Selection + "\x00" + canonicalRun + "\x00" + strconv.FormatInt(records[index].EffectiveParameters, 10)
		groups[key] = append(groups[key], index)
	}
	for _, indices := range groups {
		if len(indices) < 2 {
			continue
		}
		canonical := indices[0]
		for _, index := range indices[1:] {
			a, b := records[index], records[canonical]
			aHigh := a.LocalLLMReview != nil && a.LocalLLMReview.Confidence == "high"
			bHigh := b.LocalLLMReview != nil && b.LocalLLMReview.Confidence == "high"
			if aHigh && !bHigh || (aHigh == bHigh && (a.Likes > b.Likes || (a.Likes == b.Likes && (a.Downloads > b.Downloads || (a.Downloads == b.Downloads && a.CreatedAt < b.CreatedAt))))) {
				canonical = index
			}
		}
		for _, index := range indices {
			if index == canonical {
				continue
			}
			records[index].TrainingCostUSD = nil
			records[index].LowerTrainingCostUSD = nil
			records[index].UpperTrainingCostUSD = nil
			records[index].TrainingCostMethod = "duplicate_local_llm_training_run"
			records[index].TrainingCostTier = "duplicate"
			records[index].Compute = nil
			records[index].Errors = append(records[index].Errors, "local-LLM canonical run counted once under "+records[canonical].RepoID)
		}
	}
}

func reportedComputeFallsInSelection(compute *reportedTrainingCompute, selection compiledSelection) bool {
	if compute == nil {
		return true
	}
	if len(compute.TrainingYears) > 0 {
		fromYear, toYear := selection.from.Year(), selection.to.Year()
		for _, year := range compute.TrainingYears {
			if year >= fromYear && year <= toYear {
				return true
			}
		}
		return false
	}
	if compute.EarliestCardCreatedAt != "" {
		if earliest, err := time.Parse(time.RFC3339Nano, compute.EarliestCardCreatedAt); err == nil && earliest.Before(selection.from) {
			return false
		}
	}
	return true
}

func scratchClaimFallsInSelection(claim *reportedScratchClaim, selection compiledSelection) bool {
	if claim == nil {
		return true
	}
	if len(claim.TrainingYears) > 0 {
		fromYear, toYear := selection.from.Year(), selection.to.Year()
		for _, year := range claim.TrainingYears {
			if year >= fromYear && year <= toYear {
				return true
			}
		}
		return false
	}
	if claim.EarliestCardCreatedAt != "" {
		if earliest, err := time.Parse(time.RFC3339Nano, claim.EarliestCardCreatedAt); err == nil && earliest.Before(selection.from) {
			return false
		}
	}
	return true
}

func normalizeIdentity(value string) string {
	var result strings.Builder
	for _, r := range strings.ToLower(value) {
		if unicode.IsLetter(r) || unicode.IsNumber(r) {
			result.WriteRune(r)
		}
	}
	return result.String()
}

func scratchClaimMatchesOwner(model catalogModel, claim *reportedScratchClaim) bool {
	if claim == nil || claim.ClaimedTrainer == "" {
		return true
	}
	trainer := normalizeIdentity(claim.ClaimedTrainer)
	owner := normalizeIdentity(modelOwner(model))
	if len(trainer) < 3 || len(owner) < 2 {
		return true
	}
	return strings.Contains(owner, trainer) || strings.Contains(trainer, owner)
}

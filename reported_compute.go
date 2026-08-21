package main

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
)

// reportedTrainingCompute is evidence reported by the repository owner. Duration
// is wall-clock training time; GPUHours is duration multiplied by the reported
// GPU count. If the count is omitted, one H100 is assumed as requested by the
// production scenario and AssumedGPUCount is set.
type reportedTrainingCompute struct {
	Hours                 float64 `json:"hours"`
	GPUCount              int     `json:"gpu_count"`
	GPUName               string  `json:"gpu_name"`
	GPUHours              float64 `json:"gpu_hours"`
	HourlyRateUSD         float64 `json:"hourly_rate_usd"`
	CostUSD               float64 `json:"cost_usd"`
	CostMethod            string  `json:"cost_method"`
	ReportedCostUSD       float64 `json:"reported_cost_usd,omitempty"`
	Source                string  `json:"source"`
	Evidence              string  `json:"evidence"`
	EvidenceHash          string  `json:"evidence_hash,omitempty"`
	RunHash               string  `json:"run_hash,omitempty"`
	TrainingYears         []int   `json:"training_years,omitempty"`
	EarliestCardCreatedAt string  `json:"earliest_card_created_at,omitempty"`
	AssumedGPU            bool    `json:"assumed_gpu,omitempty"`
	AssumedGPUCount       bool    `json:"assumed_gpu_count,omitempty"`
	AssumedRate           bool    `json:"assumed_hourly_rate,omitempty"`
}

type reportedScratchClaim struct {
	Source                string  `json:"source"`
	EvidenceType          string  `json:"evidence_type,omitempty"`
	Evidence              string  `json:"evidence"`
	CardHash              string  `json:"card_hash"`
	RunHash               string  `json:"run_hash"`
	EarliestCardCreatedAt string  `json:"earliest_card_created_at,omitempty"`
	ClaimedTrainer        string  `json:"claimed_trainer,omitempty"`
	TrainingYears         []int   `json:"training_years,omitempty"`
	ActiveParametersB     float64 `json:"active_parameters_b,omitempty"`
	TrainingTokensT       float64 `json:"training_tokens_trillions,omitempty"`
}

var (
	reportedGPUHoursRE           = regexp.MustCompile(`(?i)(?:(?:gpu|h\s*100(?:\s+gpu)?)[-_ ]?(?:hours?|hrs?|h)(?:\s+(?:used|required))?|total\s+(?:h\s*100\s+)?gpu[-_ ]?(?:hours?|hrs?|h)|hours?[_ ]used\s*\(\s*total\s+gpu[-_ ]?(?:hours?|hrs?|h)\s*\))[*_ ]*[:=]?[*_ ]*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)(?:\s*(k|m|thousand|million))?\s*h?\b`)
	reportedGPUHoursValueFirstRE = regexp.MustCompile(`(?i)\b([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)(?:\s*(k|m|thousand|million))?\s*(?:gpu[-_ ]?(?:hours?|hrs?|h)|h\s*100(?:[-_ ]?gpu)?[-_ ]?(?:hours?|hrs?|h))\b`)
	reportedDurationLabelRE      = regexp.MustCompile(`(?i)(?:(?:total\s+)?(?:pre[- ]?training|training|fine[- ]?tuning)\s+(?:time|duration)|train(?:ing)?[_ -]?runtime|hours?[_ ]used|(?:pre[- ]?training|training|fine[- ]?tuning)\s+took)[^\n]{0,60}?((?:[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?\s*(?:months?|weeks?|days?|hours?|hrs?|minutes?|mins?|seconds?|secs?|[hms])\s*(?:(?:,|and)\s*)?){1,4})`)
	reportedTrainedForRE         = regexp.MustCompile(`(?i)(?:pre[- ]?trained|trained|fine[- ]?tun(?:e|ed))\s+for\s*(?:approximately|about|around|roughly|~)?\s*((?:[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?\s*(?:months?|weeks?|days?|hours?|hrs?|minutes?|mins?|seconds?|secs?|[hms])\s*(?:(?:,|and)\s*)?){1,4})`)
	tpuRuntimeRE                 = regexp.MustCompile(`(?i)\b([0-9]+(?:\.[0-9]+)?)\s*(TPU(?:[- ]?V?\d+)?)[-_ ](\d+)\s*(?:hours?|hrs?|h)\b`)
	durationComponentRE          = regexp.MustCompile(`(?i)([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*(months?|weeks?|days?|hours?|hrs?|minutes?|mins?|seconds?|secs?|[hms])`)
	hardwareRE                   = regexp.MustCompile(`(?i)(?:(\d+)\s*(?:x|×)?\s*)?(?:GPUs?\s*[:=-]?\s*)?(?:(?:NVIDIA|AMD)\s+)?(H\s*(?:20|100|200)|B\s*200|A\s*(?:10|40|100|6000)|V\s*100|T\s*4|L\s*4(?:0S?)?|RTX\s*[0-9]{4}|MI\s*[0-9][0-9A-Z]*|TPU(?:[- ]?V?\d+)?)\b[^\n,;]{0,35}?(?:\s+(?:nodes?\s*)?(?:x|×)\s*(\d+))?`)
	gpuCountPrefixRE             = regexp.MustCompile(`(?i)\b(\d+)\s+GPUs\b`)
	gpuCountXRE                  = regexp.MustCompile(`(?i)\b(\d+)\s*(?:x|×)\s*GPUs?\b`)
	gpuCountSuffixRE             = regexp.MustCompile(`(?i)\bGPUs?\s*[:=]\s*(\d+)\b`)
	consumerGPUModelRE           = regexp.MustCompile(`(?i)\b(?:NVIDIA\s+)?((?:20|30|40|50)[0-9]{2})\s*GPU\b`)
	gpuModelBeforeHoursRE        = regexp.MustCompile(`(?i)(?:RTX|GeForce|Radeon)\s*$`)
	explicitNonGPUHardwareRE     = regexp.MustCompile(`(?i)\b(?:CPU(?:-only)?|Apple\s+Silicon|M[1-4]\s+(?:Pro|Max|Ultra))\b`)
	reportedCostRE               = regexp.MustCompile(`(?i)(?:(?:total\s+|end[- ]to[- ]end\s+)?(?:pre[- ]?training|training|fine[- ]?tuning)\s+cost|cost\s+of\s+(?:pre[- ]?training|training|fine[- ]?tuning)|cost\s+to\s+(?:pre[- ]?train|train|fine[- ]?tune))[^\n$]{0,50}?\$\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)(?:\s*(k|m|thousand|million))?\b`)
	reportedBareCostRE           = regexp.MustCompile(`(?i)\bcost\s*[:=]\s*(?:~|<)?\s*\$\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)(?:\s*(k|m|thousand|million))?\b`)
	hourlyRateRE                 = regexp.MustCompile(`(?i)\$\s*([0-9]+(?:\.[0-9]+)?)\s*(?:/|per\s+)(?:gpu[- ]?)?(?:hour|hr|h)\b`)
	mixedTrainingInferenceRE     = regexp.MustCompile(`(?is)\b(?:train(?:ed|ing)?|pre[- ]?train(?:ed|ing)?|fine[- ]?tun(?:e|ed|ing))\b.{0,45}\b(?:and|or|plus|including)\b.{0,25}\b(?:inferences?|evaluation|serving)\b|\b(?:inferences?|evaluation|serving)\b.{0,45}\b(?:and|or|plus|including)\b.{0,25}\b(?:train(?:ed|ing)?|pre[- ]?train(?:ed|ing)?|fine[- ]?tun(?:e|ed|ing))\b`)
	aggregateComputeScopeRE      = regexp.MustCompile(`(?i)\b(?:aggregate[ds]?|cumulative|combined|total|all)\b`)
	trainingDatesRE              = regexp.MustCompile(`(?im)(?:training\s+dates?)[^\n<]{0,160}|(?:^|[\n|>+])\s*\*{0,2}dates\*{0,2}\s*(?::\*{0,2}|\|)\s*[^\n|<]{0,120}`)
	yearRE                       = regexp.MustCompile(`\b(20[0-9]{2})\b`)
	scratchTrainerRE             = regexp.MustCompile(`(?i)(?:pre[- ]?trained|trained)[^\n]{0,40}?from scratch by\s+([^,.;\n*]{2,60})`)
	pretrainingEvidenceRE        = regexp.MustCompile(`(?is)\bpre[- ]?train(?:ed|ing)\b.{0,220}\b(?:[0-9]+(?:\.[0-9]+)?\s*(?:trillion|billion|[tb])\s+tokens?|trillions?\s+(?:of\s+)?tokens?|training (?:corpus|corpora|dataset|data)|corpus|corpora|dataset)\b|\b(?:[0-9]+(?:\.[0-9]+)?\s*(?:trillion|billion|[tb])\s+tokens?)\b.{0,180}\b(?:used\s+for\s+)?pre[- ]?train(?:ed|ing)\b|\b(?:the\s+)?model\s+was\s+trained\s+(?:on|with)\b.{0,120}\b[0-9]+(?:\.[0-9]+)?\s*(?:trillion|billion|[tb])\s+tokens?\b|\bfoundation models?\b.{0,260}\b[0-9]+(?:\.[0-9]+)?\s*(?:trillion|billion|[tb])\s+(?:total\s+|active\s+|activated\s+)?parameters?\b`)
	continuedPretrainingRE       = regexp.MustCompile(`(?is)\b(?:built|based)\s+(?:up)?on\b.{0,160}\b(?:checkpoint|model|weights?)\b|\b(?:continued|continual|additional|further|continue)\s+pre[- ]?train(?:ed|ing)?\b|\b(?:initialized|initialised|pre[- ]?trained)\s+from\b.{0,100}\b(?:checkpoint|model|weights?|[0-9]+b)\b`)
	genericDerivativeCardRE      = regexp.MustCompile(`(?is)\b(?:fine[- ]?tuned|instruction[- ]?tuned|post[- ]?trained|distilled)\s+version\s+of\b|\b(?:we|this work)\s+(?:fine[- ]?tune|post[- ]?train|distill)\b|\btraining stage\s*[:|]\s*pre[- ]?training\s*(?:&|and|\+)\s*post[- ]?training\b|\bpost[- ]?training\s+(?:utilizes?|adopts?|uses?|stage|pipeline)\b|\b(?:trained on|built upon)\s+(?:the\s+)?[^.\n]{0,100}\bbase\s+(?:foundation\s+)?model\b|\bundergone\b.{0,100}\b(?:RLHF|reinforcement learning|post[- ]?training)\b`)
	declaredBaseModelNameRE      = regexp.MustCompile(`(?i)(^|[-_./])base($|[-_./0-9])`)
	activeParametersRE           = regexp.MustCompile(`(?i)(?:about|approximately|~|≈)?\s*[*_]*([0-9]+(?:\.[0-9]+)?)[*_]*\s*(?:billion|b)\s+(?:activated|active)(?:\s+parameters?)?|(?:activates?|activated parameters(?: per token)?\s*[:=]?)\s*(?:about|approximately|~|≈)?\s*[*_]*([0-9]+(?:\.[0-9]+)?)[*_]*\s*(?:billion|b)\b`)
	activeModelNameRE            = regexp.MustCompile(`(?i)[-_]A([0-9]+(?:\.[0-9]+)?)B(?:[-_.]|$)`)
	trainingTokensRE             = regexp.MustCompile(`(?i)([0-9]+(?:\.[0-9]+)?)\s*(trillion|billion|[tb])\+?\s+(?:high[- ]quality\s+|reasoning[- ]dense\s+|multilingual\s+|text\s+|pre[- ]?training\s+)?tokens?\b`)
	actualTrainingTokensRE       = regexp.MustCompile(`(?is)(?:\b(?:the\s+)?(?:model|it)\s+(?:was\s+|is\s+)?(?:pre[- ]?trained|trained)(?:\s+(?:(?:entirely|fully)\s+)?from\s+scratch)?\s+(?:on|with|for)\s+[^.\n]{0,140}?[*_]*([0-9]+(?:\.[0-9]+)?)[*_]*\s*(trillion|billion|[tb])\+?\s+tokens?\b|\bpre[- ]?train(?:ing)?\s+tokens?[*_ ]*[:=]\s*[*_]*\s*([0-9]+(?:\.[0-9]+)?)[*_]*\s*(trillion|billion|[tb])\b|\bpre[- ]?training\s+stage\s*[0-9]*[^\n]{0,80}?([0-9]+(?:\.[0-9]+)?)\s*(trillion|billion|[tb])\s+tokens?\b|\b([0-9]+(?:\.[0-9]+)?)\s*(trillion|billion|[tb])\s+tokens?[^\n]{0,80}?stage\s*[0-9]+\s*[:|-]\s*pre[- ]?training\b)`)
	directPretrainedTokensRE     = regexp.MustCompile(`(?i)\bpre[- ]?trained\b[^.\n]{0,120}?\b(?:on|with|for)\s+[^.\n]{0,100}?[*_]*([0-9]+(?:\.[0-9]+)?)[*_]*\s*(trillion|billion|[tb])\+?\s+tokens?\b`)
)

const defaultH100HourlyRateUSD = 1.85

func scratchClaimFromText(text, source string, modelIDs ...string) *reportedScratchClaim {
	classification := classifyREADME(text)
	evidenceType := "explicit_scratch"
	evidence := classification.Evidence
	if classification.Status != "confirmed" {
		declaredBase := len(modelIDs) > 0 && declaredBaseModelNameRE.MatchString(modelIDs[0])
		match := pretrainingEvidenceRE.FindString(text)
		localIndependentPretraining := match != "" && !continuedPretrainingRE.MatchString(match) && !currentModelDerivativeRE.MatchString(match) && !genericDerivativeCardRE.MatchString(match)
		globalDerivative := continuedPretrainingRE.MatchString(text) || currentModelDerivativeRE.MatchString(text) || genericDerivativeCardRE.MatchString(text)
		if localIndependentPretraining && (!globalDerivative || declaredBase) {
			evidenceType = "pretraining_evidence"
			evidence = truncate(strings.Join(strings.Fields(match), " "), 500)
		} else if declaredBase && !globalDerivative {
			evidenceType = "declared_base_model"
			evidence = "repository name declares an independent base checkpoint: " + modelIDs[0]
		} else {
			return nil
		}
	}
	normalizedCard := strings.Join(strings.Fields(text), " ")
	normalizedEvidence := strings.Join(strings.Fields(evidence), " ")
	claimedTrainer := ""
	if match := scratchTrainerRE.FindStringSubmatch(text); match != nil {
		claimedTrainer = strings.TrimSpace(match[1])
	}
	return &reportedScratchClaim{
		Source:            source,
		EvidenceType:      evidenceType,
		Evidence:          evidence,
		CardHash:          fmt.Sprintf("%x", sha256.Sum256([]byte(normalizedCard))),
		RunHash:           fmt.Sprintf("%x", sha256.Sum256([]byte(normalizedEvidence))),
		ClaimedTrainer:    claimedTrainer,
		TrainingYears:     reportedTrainingYears(text),
		ActiveParametersB: activeParametersFromText(text, modelIDs...),
		TrainingTokensT:   scratchTrainingTokens(text, evidence, evidenceType, modelIDs...),
	}
}

func scratchTrainingTokens(text, evidence, evidenceType string, modelIDs ...string) float64 {
	known := map[string]float64{
		"allenai/olmo-3-1125-32b": 5.5,
		"allenai/olmo-3-1025-7b":  5.93,
	}
	for _, id := range modelIDs {
		if value := known[strings.ToLower(id)]; value > 0 {
			return value
		}
	}
	if value := actualTrainingTokensFromText(text, evidence); value > 0 {
		return value
	}
	if evidenceType == "explicit_scratch" {
		return trainingTokensFromText(evidence)
	}
	return 0
}

func activeParametersFromText(text string, modelIDs ...string) float64 {
	known := map[string]float64{
		"zai-org/glm-4.5-base":                      32,
		"zai-org/glm-4.5-air-base":                  12,
		"ibm-granite/granite-4.0-h-small-base":      9,
		"ibm-granite/granite-4.0-h-tiny-base":       1,
		"ibm-granite/granite-4.0-tiny-base-preview": 1,
	}
	for _, id := range modelIDs {
		if value := known[strings.ToLower(id)]; value > 0 {
			return value
		}
	}
	match := activeParametersRE.FindStringSubmatch(text)
	if match != nil {
		for _, value := range match[1:] {
			if value == "" {
				continue
			}
			parsed, err := strconv.ParseFloat(value, 64)
			if err == nil && parsed > 0 {
				return parsed
			}
		}
	}
	for _, id := range modelIDs {
		if named := activeModelNameRE.FindStringSubmatch(id); named != nil {
			parsed, err := strconv.ParseFloat(named[1], 64)
			if err == nil && parsed > 0 {
				return parsed
			}
		}
	}
	return 0
}

func actualTrainingTokensFromText(text, evidence string) float64 {
	best := 0.0
	for _, match := range actualTrainingTokensRE.FindAllStringSubmatch(text, -1) {
		valueText, unit := match[1], match[2]
		if valueText == "" {
			valueText, unit = match[3], match[4]
		}
		if valueText == "" {
			valueText, unit = match[5], match[6]
		}
		if valueText == "" {
			valueText, unit = match[7], match[8]
		}
		value, err := strconv.ParseFloat(valueText, 64)
		if err != nil || value <= 0 {
			continue
		}
		if strings.EqualFold(unit, "b") || strings.EqualFold(unit, "billion") {
			value /= 1000
		}
		if value > best {
			best = value
		}
	}
	for _, match := range directPretrainedTokensRE.FindAllStringSubmatch(text, -1) {
		value, err := strconv.ParseFloat(match[1], 64)
		if err != nil || value <= 0 {
			continue
		}
		if strings.EqualFold(match[2], "b") || strings.EqualFold(match[2], "billion") {
			value /= 1000
		}
		if value > best {
			best = value
		}
	}
	// Evidence from the strict scratch classifier may be truncated but can still
	// carry a compact, unambiguous token disclosure. Re-run only the strong
	// patterns; never promote a generic dataset-capacity number here.
	if best == 0 && evidence != "" && evidence != text {
		return actualTrainingTokensFromText(evidence, "")
	}
	return best
}

func trainingTokensFromText(text string) float64 {
	best := 0.0
	for _, match := range trainingTokensRE.FindAllStringSubmatch(text, -1) {
		value, err := strconv.ParseFloat(match[1], 64)
		if err != nil || value <= 0 {
			continue
		}
		switch strings.ToLower(match[2]) {
		case "b", "billion":
			value /= 1000
		}
		if value > best {
			best = value
		}
	}
	return best
}

func numberValue(value any) (float64, bool) {
	switch typed := value.(type) {
	case float64:
		return typed, typed > 0
	case json.Number:
		parsed, err := typed.Float64()
		return parsed, err == nil && parsed > 0
	case string:
		parsed, err := strconv.ParseFloat(strings.TrimSpace(typed), 64)
		return parsed, err == nil && parsed > 0
	default:
		return 0, false
	}
}

func normalizeDuration(value float64, unit string) float64 {
	switch strings.ToLower(unit) {
	case "second", "seconds", "sec", "secs", "s":
		return value / 3600
	case "minute", "minutes", "min", "mins", "m":
		return value / 60
	case "day", "days":
		return value * 24
	case "week", "weeks":
		return value * 7 * 24
	case "month", "months":
		// Model cards report calendar-like wall time without start/end dates.
		// Thirty days is explicit and reproducible; the assumption is kept in
		// the evidence rather than pretending month length is exact.
		return value * 30 * 24
	default:
		return value
	}
}

func parseReportedCost(match []string) (float64, bool) {
	if len(match) < 2 {
		return 0, false
	}
	clean := strings.ReplaceAll(match[1], ",", "")
	value, err := strconv.ParseFloat(clean, 64)
	if err != nil || value <= 0 {
		return 0, false
	}
	if len(match) > 2 {
		switch strings.ToLower(match[2]) {
		case "k", "thousand":
			value *= 1_000
		case "m", "million":
			value *= 1_000_000
		}
	}
	return value, true
}

type reportedNumericSignal struct {
	Start int
	End   int
	Value float64
}

func firstScaledSignal(text string, expression *regexp.Regexp) (reportedNumericSignal, bool) {
	for _, indices := range expression.FindAllStringSubmatchIndex(text, -1) {
		if len(indices) < 4 || indices[2] < 0 {
			continue
		}
		parts := []string{"", text[indices[2]:indices[3]], ""}
		if len(indices) >= 6 && indices[4] >= 0 {
			parts[2] = text[indices[4]:indices[5]]
		}
		if value, ok := parseReportedCost(parts); ok {
			return reportedNumericSignal{Start: indices[0], End: indices[1], Value: value}, true
		}
	}
	return reportedNumericSignal{}, false
}

func aggregateGPUHoursSignal(text string) (reportedNumericSignal, bool) {
	type candidate struct {
		reportedNumericSignal
		total bool
	}
	var candidates []candidate
	for _, expression := range []*regexp.Regexp{reportedGPUHoursRE, reportedGPUHoursValueFirstRE} {
		for _, indices := range expression.FindAllStringSubmatchIndex(text, -1) {
			if len(indices) < 4 || indices[2] < 0 {
				continue
			}
			parts := []string{"", text[indices[2]:indices[3]], ""}
			// A table header such as "RTX 4090 GPU hours" names the GPU; it
			// does not report 4,090 GPU-hours.  The actual values, if present,
			// are parsed from the table rows independently.
			if gpuModelBeforeHoursRE.MatchString(text[max(0, indices[0]-24):indices[0]]) {
				continue
			}
			if len(indices) >= 6 && indices[4] >= 0 {
				parts[2] = text[indices[4]:indices[5]]
			}
			value, ok := parseReportedCost(parts)
			if !ok {
				continue
			}
			if gpuHoursAreNonTraining(text, indices[0], indices[1]) {
				continue
			}
			matched := strings.ToLower(text[indices[0]:indices[1]])
			item := candidate{
				reportedNumericSignal: reportedNumericSignal{Start: indices[0], End: indices[1], Value: value},
				total:                 strings.Contains(matched, "total"),
			}
			// The two supported syntaxes can occasionally recognize the same textual
			// occurrence. De-duplicate that overlap, but never use the numeric value
			// alone: two separate 100 GPU-hour stages are 200 GPU-hours.
			duplicateOccurrence := false
			for _, existing := range candidates {
				if existing.Value == item.Value && existing.Start < item.End && item.Start < existing.End {
					duplicateOccurrence = true
					break
				}
			}
			if !duplicateOccurrence {
				candidates = append(candidates, item)
			}
		}
	}
	if len(candidates) == 0 {
		return reportedNumericSignal{}, false
	}
	// An explicitly labelled total supersedes component stages. Repeated total
	// labels with the same value are commonly bilingual copies and are counted
	// once; equal-valued component stages remain distinct occurrences.
	hasTotal := false
	for _, item := range candidates {
		hasTotal = hasTotal || item.total
	}
	seenTotals := make(map[string]bool)
	result := reportedNumericSignal{Start: len(text)}
	for _, item := range candidates {
		if hasTotal && !item.total {
			continue
		}
		if item.total {
			key := strconv.FormatFloat(item.Value, 'g', -1, 64)
			if seenTotals[key] {
				continue
			}
			seenTotals[key] = true
		}
		result.Value += item.Value
		result.Start = min(result.Start, item.Start)
		result.End = max(result.End, item.End)
	}
	return result, result.Value > 0
}

func gpuHoursAreNonTraining(text string, start, end int) bool {
	sentenceStart, sentenceEnd := sentenceBounds(text, start, end)
	sentence := strings.ToLower(text[sentenceStart:sentenceEnd])
	hasTraining := strings.Contains(sentence, "train") || strings.Contains(sentence, "fine-tun")
	hasNonTraining := strings.Contains(sentence, "inference") || strings.Contains(sentence, "evaluation") || strings.Contains(sentence, "serving")
	if hasNonTraining && !hasTraining {
		return true
	}
	// Reject explicitly mixed aggregates: there is no defensible way to recover
	// the training-only share from a combined training/inference number.
	if mixedTrainingInferenceRE.MatchString(sentence) {
		return true
	}
	context := text[max(0, start-180):min(len(text), end+180)]
	return mixedTrainingInferenceRE.MatchString(context) && aggregateComputeScopeRE.MatchString(context)
}

func sentenceBounds(text string, start, end int) (int, int) {
	left := start
	for left > 0 {
		if strings.ContainsRune("\n\r.!?", rune(text[left-1])) {
			break
		}
		left--
	}
	right := end
	for right < len(text) {
		if strings.ContainsRune("\n\r.!?", rune(text[right])) {
			right++
			break
		}
		right++
	}
	return left, right
}

func durationExpressionHours(expression string) (float64, bool) {
	total := 0.0
	found := false
	for _, match := range durationComponentRE.FindAllStringSubmatch(expression, -1) {
		value, err := strconv.ParseFloat(strings.ReplaceAll(match[1], ",", ""), 64)
		if err != nil || value <= 0 {
			continue
		}
		unit := match[2]
		// Case-insensitive matching otherwise turns the parameter suffix in
		// names such as SmolLM2-135M into 135 minutes.
		if len(unit) == 1 && unit != strings.ToLower(unit) {
			continue
		}
		total += normalizeDuration(value, unit)
		found = true
	}
	return total, found && total > 0
}

func firstDurationSignal(text string) (reportedNumericSignal, bool) {
	best := reportedNumericSignal{}
	bestPriority := -1
	// TPU cards commonly write "4.82 TPUv3-8 Hours": 4.82 is wall time
	// and 8 is the accelerator count. The generic duration expression would
	// otherwise grab only the trailing "8 Hours".
	for _, indices := range tpuRuntimeRE.FindAllStringSubmatchIndex(text, -1) {
		if len(indices) < 8 || indices[2] < 0 {
			continue
		}
		value, err := strconv.ParseFloat(text[indices[2]:indices[3]], 64)
		if err == nil && value > 0 {
			best = reportedNumericSignal{Start: indices[0], End: indices[1], Value: value}
			bestPriority = 4
			break
		}
	}
	for _, expression := range []*regexp.Regexp{reportedDurationLabelRE, reportedTrainedForRE} {
		for _, indices := range expression.FindAllStringSubmatchIndex(text, -1) {
			if len(indices) < 4 || indices[2] < 0 {
				continue
			}
			if hours, ok := durationExpressionHours(text[indices[2]:indices[3]]); ok {
				matched := strings.ToLower(text[indices[0]:indices[1]])
				priority := 1
				if strings.Contains(matched, "fine-tun") {
					priority = 3
				} else if !strings.Contains(matched, "pretrain") && !strings.Contains(matched, "pre-train") && !strings.Contains(matched, "pre training") {
					priority = 2
				}
				if priority > bestPriority {
					best = reportedNumericSignal{Start: indices[0], End: indices[1], Value: hours}
					bestPriority = priority
				}
			}
		}
	}
	return best, bestPriority >= 0
}

func firstDirectTrainingCost(text string) (reportedNumericSignal, bool) {
	if signal, ok := firstScaledSignal(text, reportedCostRE); ok {
		return signal, true
	}
	for _, indices := range reportedBareCostRE.FindAllStringSubmatchIndex(text, -1) {
		if len(indices) < 4 || indices[2] < 0 {
			continue
		}
		context := strings.ToLower(text[max(0, indices[0]-350):min(len(text), indices[1]+150)])
		if !strings.Contains(context, "training") && !strings.Contains(context, "fine-tun") && !strings.Contains(context, "pretrain") && !strings.Contains(context, "gpu hour") {
			continue
		}
		parts := []string{"", text[indices[2]:indices[3]], ""}
		if len(indices) >= 6 && indices[4] >= 0 {
			parts[2] = text[indices[4]:indices[5]]
		}
		if value, ok := parseReportedCost(parts); ok {
			return reportedNumericSignal{Start: indices[0], End: indices[1], Value: value}, true
		}
	}
	return reportedNumericSignal{}, false
}

func reportedTrainingYears(text string) []int {
	seen := make(map[int]bool)
	for _, line := range trainingDatesRE.FindAllString(text, -1) {
		for _, match := range yearRE.FindAllStringSubmatch(line, -1) {
			year, _ := strconv.Atoi(match[1])
			seen[year] = true
		}
	}
	result := make([]int, 0, len(seen))
	for year := range seen {
		result = append(result, year)
	}
	sort.Ints(result)
	return result
}

func parseHardware(text string, anchorStart, anchorEnd int) (int, string, bool, bool) {
	if special := nearestMatch(tpuRuntimeRE, text, anchorStart, anchorEnd); special != nil && len(special) >= 8 && special[4] >= 0 && special[6] >= 0 {
		count, countErr := strconv.Atoi(text[special[6]:special[7]])
		if countErr == nil && count > 0 {
			name := strings.ToUpper(strings.ReplaceAll(text[special[4]:special[5]], " ", ""))
			return count, name, false, false
		}
	}
	matches := hardwareRE.FindAllStringSubmatchIndex(text, -1)
	if len(matches) == 0 {
		if consumer := nearestMatch(consumerGPUModelRE, text, anchorStart, anchorEnd); consumer != nil {
			return 1, "NVIDIA RTX " + text[consumer[2]:consumer[3]], false, true
		}
		for _, expression := range []*regexp.Regexp{gpuCountPrefixRE, gpuCountXRE, gpuCountSuffixRE} {
			if countMatch := nearestMatch(expression, text, anchorStart, anchorEnd); countMatch != nil {
				if count, err := strconv.Atoi(text[countMatch[2]:countMatch[3]]); err == nil && count > 0 {
					return count, "NVIDIA H100", true, false
				}
			}
		}
		return 1, "NVIDIA H100", true, true
	}
	match := matches[0]
	bestDistance := intervalDistance(match[0], match[1], anchorStart, anchorEnd)
	for _, candidate := range matches[1:] {
		if distance := intervalDistance(candidate[0], candidate[1], anchorStart, anchorEnd); distance < bestDistance {
			match = candidate
			bestDistance = distance
		}
	}
	count := 1
	assumedCount := true
	if match[2] >= 0 {
		if parsed, err := strconv.Atoi(text[match[2]:match[3]]); err == nil && parsed > 0 {
			count = parsed
			assumedCount = false
		}
	}
	if match[6] >= 0 {
		if parsed, err := strconv.Atoi(text[match[6]:match[7]]); err == nil && parsed > 0 {
			count *= parsed
			assumedCount = false
		}
	}
	if assumedCount {
		sentenceStart, sentenceEnd := sentenceBounds(text, match[0], match[1])
		for _, expression := range []*regexp.Regexp{gpuCountPrefixRE, gpuCountXRE, gpuCountSuffixRE} {
			if countMatch := nearestMatch(expression, text[sentenceStart:sentenceEnd], match[0]-sentenceStart, match[1]-sentenceStart); countMatch != nil {
				if parsed, err := strconv.Atoi(text[sentenceStart+countMatch[2] : sentenceStart+countMatch[3]]); err == nil && parsed > 0 {
					count = parsed
					assumedCount = false
					break
				}
			}
		}
	}
	name := strings.ToUpper(strings.ReplaceAll(text[match[4]:match[5]], " ", ""))
	if name == "H100" {
		name = "NVIDIA H100"
	}
	return count, name, false, assumedCount
}

func nearestMatch(expression *regexp.Regexp, text string, anchorStart, anchorEnd int) []int {
	var best []int
	bestDistance := 0
	for _, candidate := range expression.FindAllStringSubmatchIndex(text, -1) {
		if len(candidate) < 4 || candidate[2] < 0 {
			continue
		}
		distance := intervalDistance(candidate[0], candidate[1], anchorStart, anchorEnd)
		if best == nil || distance < bestDistance {
			best = candidate
			bestDistance = distance
		}
	}
	return best
}

func intervalDistance(start, end, anchorStart, anchorEnd int) int {
	if end < anchorStart {
		return anchorStart - end
	}
	if anchorEnd < start {
		return start - anchorEnd
	}
	return 0
}

func sanitizeReportedText(text string) string {
	lines := strings.Split(text, "\n")
	kept := lines[:0]
	for _, line := range lines {
		lower := strings.ToLower(strings.TrimSpace(line))
		// Widgets contain prompts, source passages and expected answers. Numbers
		// inside them describe the subject matter, never this repository's run.
		if strings.Contains(lower, "carddata.widget") || strings.HasPrefix(lower, "widget:") || strings.HasPrefix(lower, "widget.") {
			continue
		}
		kept = append(kept, line)
	}
	return strings.Join(kept, "\n")
}

func reportedComputeFromText(text, source string) *reportedTrainingCompute {
	text = sanitizeReportedText(text)
	if strings.TrimSpace(text) == "" {
		return nil
	}
	evidenceHash := fmt.Sprintf("%x", sha256.Sum256([]byte(strings.Join(strings.Fields(text), " "))))
	// A directly reported training total is the strongest signal and remains
	// usable when duration or hardware is omitted.
	directCost, hasDirectCost := firstDirectTrainingCost(text)
	directEvidence := ""
	if hasDirectCost {
		contextStart := max(0, directCost.Start-500)
		contextEnd := min(len(text), directCost.End+700)
		directEvidence = strings.TrimSpace(strings.Join(strings.Fields(text[contextStart:contextEnd]), " "))
	}

	signal, aggregateGPUHours := aggregateGPUHoursSignal(text)
	if !aggregateGPUHours {
		var ok bool
		signal, ok = firstDurationSignal(text)
		if !ok {
			if !hasDirectCost {
				return nil
			}
			return &reportedTrainingCompute{
				CostUSD: directCost.Value, CostMethod: "reported_total_cost", ReportedCostUSD: directCost.Value,
				Source: source, Evidence: truncate(directEvidence, 700), EvidenceHash: evidenceHash,
				RunHash: fmt.Sprintf("%x", sha256.Sum256([]byte(directEvidence))), TrainingYears: reportedTrainingYears(directEvidence),
			}
		}
	}
	hours := signal.Value
	contextStart := max(0, signal.Start-500)
	contextEnd := min(len(text), signal.End+700)
	rawEvidence := text[contextStart:contextEnd]
	evidence := strings.TrimSpace(strings.Join(strings.Fields(rawEvidence), " "))
	if explicitNonGPUHardwareRE.MatchString(rawEvidence) && !hardwareRE.MatchString(rawEvidence) {
		return nil
	}
	count, gpu, assumedGPU, assumedCount := parseHardware(rawEvidence, signal.Start-contextStart, signal.End-contextStart)
	if aggregateGPUHours {
		// GPU-hours are already aggregated. Do not multiply them by a hardware
		// count that may merely be part of an H100-hours phrase.
		count = 1
		assumedCount = true
	}
	result := &reportedTrainingCompute{
		Hours: hours, GPUCount: count, GPUName: gpu, GPUHours: hours * float64(count),
		HourlyRateUSD: defaultH100HourlyRateUSD, Source: source, Evidence: truncate(evidence, 700),
		AssumedGPU: assumedGPU, AssumedGPUCount: assumedCount, AssumedRate: true,
		EvidenceHash:  evidenceHash,
		RunHash:       fmt.Sprintf("%x", sha256.Sum256([]byte(evidence))),
		TrainingYears: reportedTrainingYears(evidence),
	}
	if rate := hourlyRateRE.FindStringSubmatch(evidence); rate != nil {
		result.HourlyRateUSD, _ = strconv.ParseFloat(rate[1], 64)
		result.AssumedRate = false
	}
	result.CostUSD = result.GPUHours * result.HourlyRateUSD
	result.CostMethod = "gpu_hours_x_hourly_rate"
	if hasDirectCost {
		result.ReportedCostUSD = directCost.Value
		result.CostUSD = directCost.Value
		result.CostMethod = "reported_total_cost"
		result.AssumedRate = false
		result.Evidence = truncate(directEvidence, 700)
		result.RunHash = fmt.Sprintf("%x", sha256.Sum256([]byte(directEvidence)))
		result.TrainingYears = reportedTrainingYears(directEvidence)
	}
	return result
}

func flattenCardData(value any, path string, output *[]string) {
	switch typed := value.(type) {
	case map[string]any:
		for _, key := range sortedMapKeys(typed) {
			if strings.EqualFold(key, "widget") || strings.EqualFold(key, "model-index") || strings.EqualFold(key, "model_index") {
				continue
			}
			child := typed[key]
			flattenCardData(child, path+"."+key, output)
		}
	case []any:
		for _, child := range typed {
			flattenCardData(child, path, output)
		}
	case string:
		*output = append(*output, path+": "+typed)
	case float64, json.Number:
		*output = append(*output, fmt.Sprintf("%s: %v", path, typed))
	}
}

func findPositiveNumber(value any, wanted map[string]bool) (float64, bool) {
	switch typed := value.(type) {
	case map[string]any:
		for _, key := range sortedMapKeys(typed) {
			child := typed[key]
			if wanted[strings.ToLower(key)] {
				if number, ok := numberValue(child); ok {
					return number, true
				}
			}
			if number, ok := findPositiveNumber(child, wanted); ok {
				return number, true
			}
		}
	case []any:
		for _, child := range typed {
			if number, ok := findPositiveNumber(child, wanted); ok {
				return number, true
			}
		}
	}
	return 0, false
}

func sortedMapKeys(value map[string]any) []string {
	keys := make([]string, 0, len(value))
	for key := range value {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func reportedComputeFromCardData(card map[string]any) *reportedTrainingCompute {
	if len(card) == 0 {
		return nil
	}
	var lines []string
	flattenCardData(card, "cardData", &lines)
	text := strings.Join(lines, "\n")
	// Normalize common structured fields into the same conservative parser.
	if hours, ok := findPositiveNumber(card, map[string]bool{"hours_used": true, "training_hours": true, "training_time_hours": true}); ok {
		text = "training time: " + strconv.FormatFloat(hours, 'f', -1, 64) + " hours\n" + text
	} else if seconds, ok := findPositiveNumber(card, map[string]bool{"train_runtime": true, "training_time_seconds": true, "training_seconds": true}); ok {
		text = "training time: " + strconv.FormatFloat(seconds, 'f', -1, 64) + " seconds\n" + text
	}
	result := reportedComputeFromText(text, "cardData")
	if result != nil {
		// Map iteration above is intentionally presentation-only. Hash canonical
		// JSON so identical structured metadata deduplicates deterministically.
		if encoded, err := json.Marshal(card); err == nil {
			result.EvidenceHash = fmt.Sprintf("%x", sha256.Sum256(encoded))
		}
	}
	return result
}

func scratchClaimFromCardData(card map[string]any, modelIDs ...string) *reportedScratchClaim {
	if len(card) == 0 {
		return nil
	}
	var lines []string
	flattenCardData(card, "cardData", &lines)
	result := scratchClaimFromText(strings.Join(lines, "\n"), "cardData", modelIDs...)
	if result != nil {
		if encoded, err := json.Marshal(card); err == nil {
			result.CardHash = fmt.Sprintf("%x", sha256.Sum256(encoded))
		}
	}
	return result
}

func resolveTrainingSignals(ctx context.Context, client *httpClient, endpoint string, models []catalogModel, computeWanted, scratchWanted map[string]bool, workers int, logger *catalogLogger, bulkURLs []string, bulkFile string, fetchReadmes bool) (map[string]*reportedTrainingCompute, map[string]*reportedScratchClaim, error) {
	computeResult := make(map[string]*reportedTrainingCompute)
	scratchResult := make(map[string]*reportedScratchClaim)
	unresolved := make([]catalogModel, 0, len(models))
	for _, model := range models {
		if computeWanted[model.ID] {
			if compute := reportedComputeFromCardData(model.CardData); compute != nil {
				computeResult[model.ID] = compute
			}
		}
		if scratchWanted[model.ID] {
			if claim := scratchClaimFromCardData(model.CardData, model.ID); claim != nil {
				scratchResult[model.ID] = claim
			}
		}
		if (computeWanted[model.ID] && computeResult[model.ID] == nil) || (scratchWanted[model.ID] && scratchResult[model.ID] == nil) {
			unresolved = append(unresolved, model)
		}
	}
	if len(bulkURLs) > 0 && len(models) > 0 {
		// Scan every wanted card even when cardData already contains a runtime.
		// The full card often adds the GPU count or a direct total cost, so it is
		// stronger evidence and intentionally replaces the structured fallback.
		bulkCompute, bulkScratch, err := resolveTrainingSignalsFromParquet(ctx, client, bulkURLs, bulkFile, models, computeWanted, scratchWanted, logger)
		if err != nil {
			if !fetchReadmes {
				return nil, nil, fmt.Errorf("bulk model-card source is required because individual README fetching is disabled: %w", err)
			}
			logger.warn("bulk_model_cards_failed", "bulk model-card scan failed; continuing with configured fallback", map[string]any{"error": err.Error()})
		} else {
			for id, compute := range bulkCompute {
				computeResult[id] = compute
			}
			for id, claim := range bulkScratch {
				scratchResult[id] = claim
			}
		}
	}
	if !fetchReadmes {
		return computeResult, scratchResult, nil
	}
	remaining := unresolved[:0]
	for _, model := range unresolved {
		if (computeWanted[model.ID] && computeResult[model.ID] == nil) || (scratchWanted[model.ID] && scratchResult[model.ID] == nil) {
			remaining = append(remaining, model)
		}
	}
	jobs := make(chan catalogModel)
	type response struct {
		id      string
		compute *reportedTrainingCompute
		scratch *reportedScratchClaim
	}
	responses := make(chan response)
	var wg sync.WaitGroup
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for model := range jobs {
				body, _, err := client.get(ctx, readmeURL(endpoint, model.ID), true)
				if err != nil {
					logger.warn("reported_compute_fetch_failed", "could not fetch model card", map[string]any{"repo_id": model.ID, "error": err.Error()})
					responses <- response{id: model.ID}
					continue
				}
				text := string(body)
				item := response{id: model.ID}
				if computeWanted[model.ID] {
					item.compute = reportedComputeFromText(text, "README.md")
				}
				if scratchWanted[model.ID] {
					item.scratch = scratchClaimFromText(text, "README.md", model.ID)
				}
				responses <- item
			}
		}()
	}
	go func() {
		for _, model := range remaining {
			jobs <- model
		}
		close(jobs)
		wg.Wait()
		close(responses)
	}()
	for response := range responses {
		if response.compute != nil {
			computeResult[response.id] = response.compute
		}
		if response.scratch != nil {
			scratchResult[response.id] = response.scratch
		}
	}
	return computeResult, scratchResult, nil
}

package main

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	defaultEndpoint = "https://huggingface.co"
	userAgent       = "hf-from-scratch-counter/1.0"
)

var (
	// Weight filenames are intentionally broader than the Transformers defaults:
	// Diffusers and many non-Transformers libraries use component-specific names.
	weightRE = regexp.MustCompile(`(?i)\.(safetensors|bin|pt|pth|ckpt|h5|hdf5|msgpack|onnx|tflite|pb|ot|params|pdparams|joblib|pkl|pickle|npz|mlmodel|nemo|mar|uff|engine|gguf|ggml)$|(?i)\.data-[0-9]+-of-[0-9]+$`)
	// These files share weight-like extensions but contain tokenizers or trainer state.
	nonWeightArtifactRE = regexp.MustCompile(`(?i)(^|/)(tokenizer|tokenizer_model|spiece|sentencepiece|vocab)([-_.].*)?\.(bin|pt|pth|pkl|pickle|npz)$|(?i)(^|/)(training_args\.bin|optimizer\.pt|scheduler\.pt|scaler\.pt|rng_state[^/]*\.pth)$`)

	positiveScratchRE = regexp.MustCompile(`(?is)\b(pre[- ]?trained|trained)\b.{0,80}\bfrom scratch\b|\bfrom scratch\b.{0,80}\b(pre[- ]?trained|trained)\b|\btrain(ed|ing)?\b.{0,80}\bfrom (a )?random(ly)? init(ialization|ialized)?\b|\brandomly initialized\b.{0,80}\btrain(ed|ing)?\b|\btrained\b.{0,80}\bfrom the ground up\b|обуч(ена|ен|ено|ены|али|алась|ался).{0,80}с нуля|\bentraîn(é|ée)\b.{0,80}\b(à partir de zéro|from scratch)\b`)
	negatedScratchRE  = regexp.MustCompile(`(?is)\b(not|wasn['’]t|isn['’]t|never|bypass(?:es|ed|ing)?|avoid(?:s|ed|ing)?)\b.{0,80}\b(trained|training|pretrained|pretraining)\b.{0,50}\bfrom scratch\b|\b(without|no)\b.{0,25}\btraining from scratch\b|не.{0,35}обуч(ена|ен|ено|ены|али|алась|ался).{0,35}с нуля`)
	// Reject scratch claims about an auxiliary component, not about the whole
	// model.  Bare architecture words such as "decoder" or "ViT" are
	// deliberately absent: decoder-only LMs and whole vision transformers can
	// genuinely be trained from scratch.
	nonModelScratchRE     = regexp.MustCompile(`(?is)\b(tokenizer|tokeniser|токенизатор|vocab(ulary)?|sentencepiece|bpe|vision encoder|visual encoder|visual layers?|projector|projection layer|multimodal projector|embedding layer|channel embeddings?|classification head|decoder (?:layer|module|component|head))\b.{0,100}\b(from scratch|с нуля)\b|\b(from scratch|с нуля)\b.{0,100}\b(tokenizer|tokeniser|токенизатор|vocab(ulary)?|sentencepiece|bpe|vision encoder|visual encoder|visual layers?|projector|projection layer|multimodal projector|embedding layer|channel embeddings?|classification head|decoder (?:layer|module|component|head))\b|\b(?:only\s+)?(?:the\s+)?decoder\b.{0,80}\b(from scratch|с нуля)\b.{0,100}\b(?:encoder|backbone)\b.{0,50}\b(?:frozen|pre[- ]?trained)\b|(?:токенизатор|проектор|слой|эмбеддинг).{0,100}обуч(ен|ена).{0,30}с нуля`)
	randomComponentInitRE = regexp.MustCompile(`(?is)\b(tokenizer|tokeniser|vocab(?:ulary)?|vision encoder|visual encoder|projector|projection layer|multimodal projector|embedding(?: matrix| layer)?|channel embeddings?|classification head|decoder (?:layer|module|component|head))\b.{0,100}\b(randomly initialized|random initialization|initialized from scratch)\b|\b(randomly initialized|random initialization|initialized from scratch)\b.{0,100}\b(tokenizer|tokeniser|vocab(?:ulary)?|vision encoder|visual encoder|projector|projection layer|multimodal projector|embedding(?: matrix| layer)?|channel embeddings?|classification head|decoder (?:layer|module|component|head))\b`)
	nonActualScratchRE    = regexp.MustCompile(`(?is)\b(for comparison|compared (?:with|to)|typically|would|could|estimated?|hypothetical|instead of|rather than|versus|vs\.?|example of|recipe for|guide to|how to|difficulty of|allow(?:s|ed|ing)?|demo|notebook|to[- ]?do|ongoing|implement)\b.{0,160}\b(?:train(?:ed|ing)?|pretrain(?:ed|ing)?)?\b.{0,70}\bfrom scratch\b|\b(?:fewer|less)\s+(?:tokens?|flops?|compute|steps?|time)\b.{0,80}\bthan\b.{0,80}\b(?:training|pretraining)\s+from scratch\b|\bwhen\s+(?:training|pretraining)\s+(?:a|an|the|such)\s+(?:model|network|system)\b.{0,100}\bfrom scratch\b|\b(?:pre[- ]?trained|trained)\s+from scratch\s*\(\s*or\s+fine[- ]?tuned\s*\)`)
	// A generic "the model" elsewhere in a long card may describe a baseline.
	// Accept it only next to the scratch claim; globally require an unambiguous
	// reference to this/current/released model.
	localCurrentModelDerivativeRE = regexp.MustCompile(`(?is)\b(?:this|the|our)\s+(?:model|checkpoint)\s+(?:is|was|has been)\s+(?:(?:further|subsequently|then)\s+)?(?:fine[- ]?tuned|instruction[- ]?tuned|post[- ]?trained|converted|distilled)\b|\bthis\s+(?:model\s+)?(?:is|was)\s+(?:an?\s+)?(?:fine[- ]?tuned|instruction[- ]?tuned|post[- ]?trained|converted|distilled)\s+(?:model|checkpoint|iteration|version)\b|\bas\s+an?\s+(?:fine[- ]?tuned|instruction[- ]?tuned|post[- ]?trained)\s+(?:model|iteration|version)\b`)
	currentModelDerivativeRE      = regexp.MustCompile(`(?is)\bthis\s+(?:model|checkpoint|iteration|version)\s+(?:is|was|has been)\s+(?:(?:further|subsequently|then)\s+)?(?:fine[- ]?tuned|instruction[- ]?tuned|post[- ]?trained|converted|distilled)\b|\bthis\s+(?:model\s+)?(?:is|was)\s+(?:an?\s+)?(?:fine[- ]?tuned|instruction[- ]?tuned|post[- ]?trained|converted|distilled)\s+(?:model|checkpoint|iteration|version)\b|\b(?:current|released|resulting|final)\s+(?:model|checkpoint|iteration|version)\s+(?:is|was|has been)\s+(?:(?:further|subsequently|then)\s+)?(?:fine[- ]?tuned|instruction[- ]?tuned|post[- ]?trained|converted|distilled)\b`)
	derivativeSequenceRE          = regexp.MustCompile(`(?is)\b(?:trained|pre[- ]?trained)\b.{0,90}\bfrom scratch\b.{0,220}\b(?:(?:followed by|post[- ]?training included).{0,80}(?:supervised fine[- ]?tuning|instruction fine[- ]?tuning|RLOO|RLHF|QRPO)|(?:then underwent|and (?:was )?)(?:supervised |instruction )?fine[- ]?tuned)\b|\bfrom scratch\b.{0,180}\band (?:was )?(?:supervised |instruction )?fine[- ]?tuned\b|\b(?:supervised fine[- ]?tuning|instruction fine[- ]?tuning|alignment)\b.{0,180}\b(?:trained end[- ]to[- ]end from random initialization|trained from scratch)\b|\b(?:enhanced[,]?\s*)?instruction[- ]?tuned\s+(?:iteration|version)\b`)
	upstreamScratchRE             = regexp.MustCompile(`(?is)\b(?:based on|initialized from|uses?)\b.{0,120}\bpre[- ]?trained\b.{0,120}\b(?:trained|built)\b.{0,70}\bfrom scratch\b`)
	derivativeNameRE              = regexp.MustCompile(`(?i)(^|[-_.])(lora|qlora|adapter|merged?|gguf|gptq|awq|exl2|bnb|int[248]|fp8|finetuned?|fine[-_]?tuned?|sft|dpo)($|[-_.])`)
	adapterFileRE                 = regexp.MustCompile(`(?i)(^|/)(adapter_(model|config)|.*lora.*)\.`)
	conversionFileRE              = regexp.MustCompile(`(?i)\.(gguf|ggml)$`)

	derivativeTags = map[string]struct{}{
		"adapter-transformers": {}, "peft": {}, "lora": {}, "qlora": {},
		"merge": {}, "model-merge": {}, "gguf": {}, "gptq": {}, "awq": {},
		"exl2": {}, "quantized": {},
	}
)

type classification struct {
	Status   string
	Reason   string
	Evidence string
}

type sibling struct {
	Filename string `json:"rfilename"`
}

type model struct {
	ID         string         `json:"id"`
	CreatedAt  string         `json:"createdAt"`
	Tags       []string       `json:"tags"`
	CardData   map[string]any `json:"cardData"`
	BaseModels any            `json:"baseModels"`
	Siblings   []sibling      `json:"siblings"`
}

type options struct {
	Years    []int
	Output   string
	Endpoint string
	Token    string
	PageSize int
	Workers  int
	Timeout  time.Duration
	Retries  int
	MaxPages int
	Quiet    bool
}

type httpClient struct {
	client  *http.Client
	token   string
	retries int
	log     func(level, event, message string, fields map[string]any)
}

type pageItem struct {
	Model   model
	Year    int
	Created time.Time
}

type classifiedItem struct {
	Item           pageItem
	Classification classification
}

func (c *httpClient) get(ctx context.Context, address string, allowMissing bool) ([]byte, http.Header, error) {
	for attempt := 0; attempt <= c.retries; attempt++ {
		started := time.Now()
		if c.log != nil {
			c.log("debug", "http_request", "sending HTTP request", map[string]any{"method": http.MethodGet, "url": address, "attempt": attempt + 1})
		}
		serverDelay := time.Duration(0)
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, address, nil)
		if err != nil {
			return nil, nil, fmt.Errorf("create GET request for %q: %w", address, err)
		}
		req.Header.Set("User-Agent", userAgent)
		req.Header.Set("Accept", "application/json, text/plain;q=0.9")
		if c.token != "" {
			req.Header.Set("Authorization", "Bearer "+c.token)
		}
		resp, err := c.client.Do(req)
		if err == nil {
			body, readErr := io.ReadAll(resp.Body)
			closeErr := resp.Body.Close()
			if readErr != nil {
				err = fmt.Errorf("read response body from %q: %w", address, readErr)
				if closeErr != nil {
					err = errors.Join(err, fmt.Errorf("close response body from %q: %w", address, closeErr))
				}
			} else if closeErr != nil {
				err = fmt.Errorf("close response body: %w", closeErr)
			} else if resp.StatusCode >= 200 && resp.StatusCode < 300 {
				if c.log != nil {
					c.log("debug", "http_response", "HTTP request succeeded", map[string]any{"method": http.MethodGet, "url": address, "attempt": attempt + 1, "status": resp.StatusCode, "bytes": len(body), "duration_ms": time.Since(started).Milliseconds()})
				}
				return body, resp.Header, nil
			} else if allowMissing && (resp.StatusCode == 401 || resp.StatusCode == 403 || resp.StatusCode == 404) {
				if c.log != nil {
					c.log("debug", "http_missing", "optional resource is unavailable", map[string]any{"method": http.MethodGet, "url": address, "status": resp.StatusCode})
				}
				return nil, resp.Header, nil
			} else if !retryableStatus(resp.StatusCode) {
				err = fmt.Errorf("GET %s: HTTP %s", address, resp.Status)
				if c.log != nil {
					c.log("error", "http_error", "HTTP request failed without retry", map[string]any{"url": address, "status": resp.StatusCode, "attempt": attempt + 1, "error": err.Error()})
				}
				return nil, resp.Header, err
			} else {
				err = fmt.Errorf("GET %s: HTTP %s", address, resp.Status)
				serverDelay = rateLimitDelay(resp.Header)
			}
		}
		if attempt == c.retries {
			if c.log != nil {
				c.log("error", "http_retries_exhausted", "HTTP request failed after all attempts", map[string]any{"url": address, "attempts": attempt + 1, "error": fmt.Sprint(err)})
			}
			return nil, nil, err
		}
		delay := time.Duration(1<<attempt) * time.Second
		if serverDelay > delay {
			delay = serverDelay
		}
		delay += time.Duration(rand.Intn(250)) * time.Millisecond
		if c.log != nil {
			c.log("warn", "http_retry", "HTTP request will be retried", map[string]any{"url": address, "attempt": attempt + 1, "delay_ms": delay.Milliseconds(), "error": fmt.Sprint(err)})
		}
		select {
		case <-ctx.Done():
			if c.log != nil {
				c.log("error", "http_canceled", "HTTP retry canceled by context", map[string]any{"url": address, "error": ctx.Err().Error()})
			}
			return nil, nil, ctx.Err()
		case <-time.After(delay):
		}
	}
	return nil, nil, errors.New("unreachable")
}

func rateLimitDelay(header http.Header) time.Duration {
	if value := header.Get("Retry-After"); value != "" {
		if seconds, err := strconv.Atoi(value); err == nil && seconds >= 0 {
			return time.Duration(seconds+1) * time.Second
		}
		if when, err := http.ParseTime(value); err == nil {
			if delay := time.Until(when) + time.Second; delay > 0 {
				return delay
			}
		}
	}
	for _, part := range strings.Split(header.Get("RateLimit"), ";") {
		part = strings.TrimSpace(part)
		if strings.HasPrefix(part, "t=") {
			if seconds, err := strconv.Atoi(strings.TrimPrefix(part, "t=")); err == nil && seconds >= 0 {
				return time.Duration(seconds+1) * time.Second
			}
		}
	}
	return 0
}

func retryableStatus(code int) bool {
	return code == 429 || code == 500 || code == 502 || code == 503 || code == 504
}

func nonEmpty(value any) bool {
	switch v := value.(type) {
	case nil:
		return false
	case string:
		return strings.TrimSpace(v) != ""
	case []any:
		return len(v) > 0
	case map[string]any:
		return len(v) > 0
	default:
		return true
	}
}

func metadataFilter(m model) *classification {
	if !hasRecognizedWeight(m.Siblings) {
		return &classification{Status: "excluded", Reason: "no_recognized_full_weight_file"}
	}

	for _, key := range []string{"base_model", "baseModel", "base_models"} {
		if value, ok := m.CardData[key]; ok && nonEmpty(value) {
			return &classification{Status: "excluded", Reason: "declares_base_model", Evidence: truncate(fmt.Sprint(value), 300)}
		}
	}
	if nonEmpty(m.BaseModels) {
		return &classification{Status: "excluded", Reason: "declares_base_model", Evidence: truncate(fmt.Sprint(m.BaseModels), 300)}
	}

	tags := make(map[string]struct{})
	for _, tag := range m.Tags {
		tags[strings.ToLower(tag)] = struct{}{}
	}
	if cardTags, ok := m.CardData["tags"]; ok {
		switch values := cardTags.(type) {
		case []any:
			for _, tag := range values {
				tags[strings.ToLower(fmt.Sprint(tag))] = struct{}{}
			}
		case string:
			tags[strings.ToLower(values)] = struct{}{}
		}
	}
	var foundTags []string
	for tag := range tags {
		if _, found := derivativeTags[tag]; found {
			foundTags = append(foundTags, tag)
		}
	}
	if len(foundTags) > 0 {
		sort.Strings(foundTags)
		return &classification{Status: "excluded", Reason: "derivative_tag", Evidence: strings.Join(foundTags, ", ")}
	}

	name := m.ID
	if slash := strings.LastIndexByte(name, '/'); slash >= 0 {
		name = name[slash+1:]
	}
	if derivativeNameRE.MatchString(name) {
		return &classification{Status: "excluded", Reason: "derivative_name", Evidence: m.ID}
	}
	for _, file := range m.Siblings {
		if adapterFileRE.MatchString(file.Filename) {
			return &classification{Status: "excluded", Reason: "adapter_files"}
		}
		if conversionFileRE.MatchString(file.Filename) {
			return &classification{Status: "excluded", Reason: "conversion_files"}
		}
	}
	return nil
}

func hasRecognizedWeight(files []sibling) bool {
	for _, file := range files {
		if weightRE.MatchString(file.Filename) && !nonWeightArtifactRE.MatchString(file.Filename) {
			return true
		}
	}
	return false
}

func classifyREADME(text string) classification {
	matches := positiveScratchRE.FindAllStringIndex(text, -1)
	if len(matches) == 0 {
		return classification{Status: "candidate", Reason: "no_explicit_scratch_claim"}
	}
	for _, match := range matches {
		start := max(0, match[0]-100)
		end := min(len(text), match[1]+100)
		context := safeSlice(text, start, end)
		wideContext := safeSlice(text, max(0, match[0]-300), min(len(text), match[1]+300))
		if negatedScratchRE.MatchString(context) || nonModelScratchRE.MatchString(context) || randomComponentInitRE.MatchString(context) || nonActualScratchRE.MatchString(context) || upstreamScratchRE.MatchString(context) || localCurrentModelDerivativeRE.MatchString(context) || derivativeSequenceRE.MatchString(wideContext) {
			continue
		}
		if currentModelDerivativeRE.MatchString(text) {
			return classification{Status: "excluded", Reason: "current_model_is_derivative"}
		}
		evidenceStart := max(0, match[0]-100)
		evidenceEnd := min(len(text), match[1]+100)
		evidence := strings.Join(strings.Fields(safeSlice(text, evidenceStart, evidenceEnd)), " ")
		return classification{Status: "confirmed", Reason: "explicit_scratch_claim", Evidence: truncate(evidence, 500)}
	}
	return classification{Status: "excluded", Reason: "scratch_claim_not_usable"}
}

func truncate(value string, limit int) string {
	runes := []rune(value)
	if len(runes) <= limit {
		return value
	}
	return string(runes[:limit])
}

func safeSlice(value string, start, end int) string {
	for start < end && start < len(value) && value[start]&0xc0 == 0x80 {
		start++
	}
	for end > start && end < len(value) && value[end]&0xc0 == 0x80 {
		end--
	}
	return value[start:end]
}

func initialAPIURL(endpoint string, pageSize int) string {
	query := url.Values{}
	query.Set("sort", "createdAt")
	query.Set("direction", "-1")
	query.Set("limit", strconv.Itoa(pageSize))
	for _, field := range []string{"createdAt", "tags", "cardData", "siblings", "baseModels"} {
		query.Add("expand", field)
	}
	return strings.TrimRight(endpoint, "/") + "/api/models?" + query.Encode()
}

func readmeURL(endpoint, repoID string) string {
	parts := strings.Split(repoID, "/")
	for i := range parts {
		parts[i] = url.PathEscape(parts[i])
	}
	return strings.TrimRight(endpoint, "/") + "/" + strings.Join(parts, "/") + "/resolve/main/README.md"
}

func nextLink(header http.Header) string {
	for _, part := range strings.Split(header.Get("Link"), ",") {
		sections := strings.Split(strings.TrimSpace(part), ";")
		if len(sections) < 2 || !strings.Contains(strings.Join(sections[1:], ";"), `rel="next"`) {
			continue
		}
		return strings.Trim(strings.TrimSpace(sections[0]), "<>")
	}
	return ""
}

func classifyPending(ctx context.Context, client *httpClient, endpoint string, pending []pageItem, workers int) <-chan classifiedItem {
	jobs := make(chan pageItem)
	results := make(chan classifiedItem)
	var wg sync.WaitGroup
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for item := range jobs {
				body, _, err := client.get(ctx, readmeURL(endpoint, item.Model.ID), true)
				result := classification{}
				if err != nil {
					result = classification{Status: "error", Reason: "readme_fetch_error", Evidence: truncate(err.Error(), 300)}
				} else {
					result = classifyREADME(string(body))
				}
				results <- classifiedItem{Item: item, Classification: result}
			}
		}()
	}
	go func() {
		for _, item := range pending {
			jobs <- item
		}
		close(jobs)
		wg.Wait()
		close(results)
	}()
	return results
}

func writeCSVRow(writer *csv.Writer, item pageItem, result classification) error {
	return writer.Write([]string{
		strconv.Itoa(item.Year), item.Model.ID, item.Created.Format(time.RFC3339Nano),
		result.Status, result.Reason, result.Evidence,
	})
}

func writeSummary(path string, years []int, counts, reasons map[int]map[string]int, complete bool) error {
	yearResults := make(map[string]any, len(years))
	for _, year := range years {
		item := make(map[string]any)
		for status, count := range counts[year] {
			item[status] = count
		}
		item["broad_upper_bound"] = counts[year]["confirmed"] + counts[year]["candidate"]
		item["reasons"] = reasons[year]
		yearResults[strconv.Itoa(year)] = item
	}
	summary := map[string]any{
		"generated_at": time.Now().UTC().Format(time.RFC3339Nano),
		"complete":     complete,
		"definition": map[string]string{
			"confirmed": "weights present, no derivative metadata, explicit non-negated README claim of training from scratch",
			"candidate": "weights present and no derivative evidence, but README has no explicit scratch claim",
			"year":      "UTC year of the Hugging Face repository createdAt timestamp",
		},
		"years": yearResults,
	}
	data, err := json.MarshalIndent(summary, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o644)
}

func increment(target map[int]map[string]int, year int, key string) {
	if target[year] == nil {
		target[year] = make(map[string]int)
	}
	target[year][key]++
}

func run(ctx context.Context, opts options) (resultErr error) {
	if err := os.MkdirAll(opts.Output, 0o755); err != nil {
		return err
	}
	detailsPath := filepath.Join(opts.Output, "models.csv")
	summaryPath := filepath.Join(opts.Output, "summary.json")
	file, err := os.Create(detailsPath)
	if err != nil {
		return err
	}
	writer := csv.NewWriter(file)
	defer func() {
		writer.Flush()
		if err := writer.Error(); err != nil {
			resultErr = errors.Join(resultErr, fmt.Errorf("flush %q: %w", detailsPath, err))
		}
		if err := file.Sync(); err != nil {
			resultErr = errors.Join(resultErr, fmt.Errorf("sync %q: %w", detailsPath, err))
		}
		if err := file.Close(); err != nil {
			resultErr = errors.Join(resultErr, fmt.Errorf("close %q: %w", detailsPath, err))
		}
	}()
	if err := writer.Write([]string{"year", "repo_id", "created_at", "status", "reason", "evidence"}); err != nil {
		return err
	}

	client := &httpClient{client: &http.Client{Timeout: opts.Timeout}, token: opts.Token, retries: opts.Retries}
	counts := make(map[int]map[string]int)
	reasons := make(map[int]map[string]int)
	yearSet := make(map[int]struct{}, len(opts.Years))
	minYear, maxYear := opts.Years[0], opts.Years[len(opts.Years)-1]
	for _, year := range opts.Years {
		yearSet[year] = struct{}{}
		counts[year] = make(map[string]int)
		reasons[year] = make(map[string]int)
	}

	address := initialAPIURL(opts.Endpoint, opts.PageSize)
	complete := false
	pages := 0
	for address != "" {
		body, headers, err := client.get(ctx, address, false)
		if err != nil {
			return err
		}
		var page []model
		if err := json.Unmarshal(body, &page); err != nil {
			return fmt.Errorf("decode Hub response: %w", err)
		}
		if len(page) == 0 {
			complete = true
			break
		}
		pages++
		reachedOlder := false
		var pending []pageItem
		for _, m := range page {
			created, err := time.Parse(time.RFC3339Nano, m.CreatedAt)
			if err != nil {
				continue
			}
			year := created.UTC().Year()
			if year < minYear {
				reachedOlder = true
				continue
			}
			if year > maxYear {
				continue
			}
			if _, wanted := yearSet[year]; !wanted {
				continue
			}
			item := pageItem{Model: m, Year: year, Created: created.UTC()}
			if preliminary := metadataFilter(m); preliminary != nil {
				increment(counts, year, preliminary.Status)
				increment(reasons, year, preliminary.Reason)
				if err := writeCSVRow(writer, item, *preliminary); err != nil {
					return err
				}
			} else {
				pending = append(pending, item)
			}
		}
		for result := range classifyPending(ctx, client, opts.Endpoint, pending, opts.Workers) {
			increment(counts, result.Item.Year, result.Classification.Status)
			increment(reasons, result.Item.Year, result.Classification.Reason)
			if err := writeCSVRow(writer, result.Item, result.Classification); err != nil {
				return err
			}
		}
		writer.Flush()
		if err := writer.Error(); err != nil {
			return err
		}
		if err := writeSummary(summaryPath, opts.Years, counts, reasons, false); err != nil {
			return err
		}
		if !opts.Quiet {
			parts := make([]string, 0, len(opts.Years))
			for _, year := range opts.Years {
				parts = append(parts, fmt.Sprintf("%d: confirmed=%d, candidate=%d", year, counts[year]["confirmed"], counts[year]["candidate"]))
			}
			if _, err := fmt.Fprintf(os.Stderr, "pages=%d; %s\n", pages, strings.Join(parts, ", ")); err != nil {
				return fmt.Errorf("write progress to stderr: %w", err)
			}
		}
		if opts.MaxPages > 0 && pages >= opts.MaxPages {
			break
		}
		if reachedOlder {
			complete = true
			break
		}
		address = nextLink(headers)
		if address == "" {
			complete = true
		}
	}
	if err := writeSummary(summaryPath, opts.Years, counts, reasons, complete); err != nil {
		return err
	}
	for _, year := range opts.Years {
		if _, err := fmt.Printf("%d: confirmed=%d; candidates=%d; broad_upper_bound=%d\n", year, counts[year]["confirmed"], counts[year]["candidate"], counts[year]["confirmed"]+counts[year]["candidate"]); err != nil {
			return fmt.Errorf("write yearly result to stdout: %w", err)
		}
	}
	if _, err := fmt.Printf("Details: %s\nSummary: %s\n", detailsPath, summaryPath); err != nil {
		return fmt.Errorf("write output paths to stdout: %w", err)
	}
	return nil
}

func parseYears(value string) ([]int, error) {
	parts := strings.FieldsFunc(value, func(r rune) bool { return r == ',' || r == ' ' })
	seen := make(map[int]struct{})
	currentYear := time.Now().UTC().Year()
	for _, part := range parts {
		year, err := strconv.Atoi(part)
		if err != nil || year < 2000 || year > currentYear {
			return nil, fmt.Errorf("years must be integers between 2000 and %d", currentYear)
		}
		seen[year] = struct{}{}
	}
	if len(seen) == 0 {
		return nil, errors.New("at least one year is required")
	}
	years := make([]int, 0, len(seen))
	for year := range seen {
		years = append(years, year)
	}
	sort.Ints(years)
	return years, nil
}

package main

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"flag"
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

	positiveScratchRE = regexp.MustCompile(`(?is)\b(pre[- ]?trained|trained|training|built)\b.{0,80}\bfrom scratch\b|\bfrom scratch\b.{0,80}\b(pre[- ]?trained|trained|training|model)\b|\btrain(ed|ing)?\b.{0,80}\bfrom (a )?random(ly)? init(ialization|ialized)?\b|\brandomly initialized\b.{0,80}\btrain(ed|ing)?\b|\btrained\b.{0,80}\bfrom the ground up\b|обуч(ена|ен|ено|ены|али|алась|ался).{0,80}с нуля|\bentraîn(é|ée|ement)\b.{0,80}\b(à partir de zéro|from scratch)\b`)
	negatedScratchRE  = regexp.MustCompile(`(?is)\b(not|wasn['’]t|isn['’]t|never)\b.{0,35}\b(trained|training|pretrained)\b.{0,35}\bfrom scratch\b|\b(without|no)\b.{0,25}\btraining from scratch\b|не.{0,35}обуч(ена|ен|ено|ены|али|алась|ался).{0,35}с нуля`)
	derivativeTextRE  = regexp.MustCompile(`(?i)\b(fine[- ]?tun(e|ed|ing)|instruction[- ]?tun(e|ed|ing)|continued pretraining|continual pretraining|domain adaptation|distill(ed|ation)|knowledge distillation|merged? (model|checkpoint)|quantiz(ed|ation)|converted? (from|to)|(lora|qlora|peft) adapter)\b`)
	nonModelScratchRE = regexp.MustCompile(`(?is)\b(tokenizer|tokeniser|vocab(ulary)?|sentencepiece|bpe)\b.{0,80}\bfrom scratch\b|\bfrom scratch\b.{0,80}\b(tokenizer|tokeniser|vocab(ulary)?|sentencepiece|bpe)\b`)
	derivativeNameRE  = regexp.MustCompile(`(?i)(^|[-_.])(lora|qlora|adapter|merged?|gguf|gptq|awq|exl2|bnb|int[248]|fp8|finetuned?|fine[-_]?tuned?|sft|dpo)($|[-_.])`)
	adapterFileRE     = regexp.MustCompile(`(?i)(^|/)(adapter_(model|config)|.*lora.*)\.`)
	conversionFileRE  = regexp.MustCompile(`(?i)\.(gguf|ggml)$`)

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
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, address, nil)
		if err != nil {
			return nil, nil, err
		}
		req.Header.Set("User-Agent", userAgent)
		req.Header.Set("Accept", "application/json, text/plain;q=0.9")
		if c.token != "" {
			req.Header.Set("Authorization", "Bearer "+c.token)
		}
		resp, err := c.client.Do(req)
		if err == nil {
			body, readErr := io.ReadAll(resp.Body)
			resp.Body.Close()
			if readErr != nil {
				err = readErr
			} else if resp.StatusCode >= 200 && resp.StatusCode < 300 {
				return body, resp.Header, nil
			} else if allowMissing && (resp.StatusCode == 401 || resp.StatusCode == 403 || resp.StatusCode == 404) {
				return nil, resp.Header, nil
			} else if !retryableStatus(resp.StatusCode) {
				return nil, resp.Header, fmt.Errorf("GET %s: HTTP %s", address, resp.Status)
			} else {
				err = fmt.Errorf("GET %s: HTTP %s", address, resp.Status)
			}
		}
		if attempt == c.retries {
			return nil, nil, err
		}
		delay := time.Duration(1<<attempt) * time.Second
		delay += time.Duration(rand.Intn(250)) * time.Millisecond
		select {
		case <-ctx.Done():
			return nil, nil, ctx.Err()
		case <-time.After(delay):
		}
	}
	return nil, nil, errors.New("unreachable")
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
		if negatedScratchRE.MatchString(context) || nonModelScratchRE.MatchString(context) || derivativeTextRE.MatchString(context) {
			continue
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

func run(ctx context.Context, opts options) error {
	if err := os.MkdirAll(opts.Output, 0o755); err != nil {
		return err
	}
	detailsPath := filepath.Join(opts.Output, "models.csv")
	summaryPath := filepath.Join(opts.Output, "summary.json")
	file, err := os.Create(detailsPath)
	if err != nil {
		return err
	}
	defer file.Close()
	writer := csv.NewWriter(file)
	defer writer.Flush()
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
			fmt.Fprintf(os.Stderr, "pages=%d; %s\n", pages, strings.Join(parts, ", "))
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
		fmt.Printf("%d: confirmed=%d; candidates=%d; broad_upper_bound=%d\n", year, counts[year]["confirmed"], counts[year]["candidate"], counts[year]["confirmed"]+counts[year]["candidate"])
	}
	fmt.Printf("Details: %s\nSummary: %s\n", detailsPath, summaryPath)
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

func main() {
	var yearsValue string
	opts := options{}
	flag.StringVar(&yearsValue, "years", "2025,2026", "comma-separated UTC publication years")
	flag.StringVar(&opts.Output, "output", "results", "output directory")
	flag.StringVar(&opts.Endpoint, "endpoint", defaultEndpoint, "Hugging Face endpoint")
	flag.StringVar(&opts.Token, "token", "", "HF token (defaults to HF_TOKEN)")
	flag.IntVar(&opts.PageSize, "page-size", 1000, "models per API page (1..1000)")
	flag.IntVar(&opts.Workers, "workers", 8, "parallel README downloads (1..64)")
	flag.DurationVar(&opts.Timeout, "timeout", 30*time.Second, "HTTP request timeout")
	flag.IntVar(&opts.Retries, "retries", 5, "retries for transient HTTP errors")
	flag.IntVar(&opts.MaxPages, "max-pages", 0, "stop after N API pages; 0 means unlimited")
	flag.BoolVar(&opts.Quiet, "quiet", false, "hide per-page progress")
	flag.Parse()

	var err error
	opts.Years, err = parseYears(yearsValue)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(2)
	}
	if opts.PageSize < 1 || opts.PageSize > 1000 || opts.Workers < 1 || opts.Workers > 64 || opts.Timeout <= 0 || opts.Retries < 0 || opts.MaxPages < 0 {
		fmt.Fprintln(os.Stderr, "error: invalid page-size, workers, timeout, retries, or max-pages")
		os.Exit(2)
	}
	if opts.Token == "" {
		opts.Token = os.Getenv("HF_TOKEN")
	}
	if err := run(context.Background(), opts); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

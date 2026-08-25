package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

type catalogLoggingConfig struct {
	Level          string `json:"level"`
	File           string `json:"file"`
	Stderr         *bool  `json:"stderr"`
	HTTPRequests   bool   `json:"http_requests"`
	SkippedRecords bool   `json:"skipped_records"`
}

type catalogLogger struct {
	mu        sync.Mutex
	file      *os.File
	filePath  string
	stderr    bool
	threshold int
	writeErr  error
}

var catalogLogLevels = map[string]int{
	"debug": 10,
	"info":  20,
	"warn":  30,
	"error": 40,
}

func newCatalogLogger(config catalogLoggingConfig, outputDir string) (*catalogLogger, error) {
	level, ok := catalogLogLevels[strings.ToLower(config.Level)]
	if !ok {
		return nil, fmt.Errorf("logging.level %q is invalid: use debug, info, warn, or error", config.Level)
	}
	path := config.File
	if !filepath.IsAbs(path) {
		path = filepath.Join(outputDir, path)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, fmt.Errorf("create log directory %q: %w", filepath.Dir(path), err)
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil, fmt.Errorf("open log file %q: %w", path, err)
	}
	return &catalogLogger{
		file: file, filePath: path, stderr: boolValue(config.Stderr, true), threshold: level,
	}, nil
}

func (l *catalogLogger) log(level, event, message string, fields map[string]any) {
	if l == nil || catalogLogLevels[level] < l.threshold {
		return
	}
	entry := make(map[string]any, len(fields)+4)
	entry["time"] = time.Now().UTC().Format(time.RFC3339Nano)
	entry["level"] = level
	entry["event"] = event
	entry["message"] = message
	for key, value := range fields {
		entry[key] = value
	}
	data, err := json.Marshal(entry)
	if err != nil {
		// This fallback deliberately avoids the logger itself to prevent recursion.
		l.mu.Lock()
		l.writeErr = errors.Join(l.writeErr, fmt.Errorf("encode catalog log event %q: %w", event, err))
		if _, stderrErr := fmt.Fprintf(os.Stderr, "ERROR catalog_log_encode event=%s error=%q\n", event, err); stderrErr != nil {
			l.writeErr = errors.Join(l.writeErr, fmt.Errorf("write catalog logging failure to stderr: %w", stderrErr))
		}
		l.mu.Unlock()
		return
	}

	l.mu.Lock()
	defer l.mu.Unlock()
	if _, err := l.file.Write(append(data, '\n')); err != nil {
		l.writeErr = errors.Join(l.writeErr, fmt.Errorf("write log file %q: %w", l.filePath, err))
		if _, stderrErr := fmt.Fprintf(os.Stderr, "ERROR catalog_log_write path=%q error=%q\n", l.filePath, err); stderrErr != nil {
			l.writeErr = errors.Join(l.writeErr, fmt.Errorf("write catalog logging failure to stderr: %w", stderrErr))
		}
	}
	if l.stderr {
		line := fmt.Sprintf("%s %-5s %s: %s", entry["time"], strings.ToUpper(level), event, message)
		if len(fields) > 0 {
			encoded, marshalErr := json.Marshal(fields)
			if marshalErr == nil {
				line += " " + string(encoded)
			} else {
				l.writeErr = errors.Join(l.writeErr, fmt.Errorf("encode stderr log fields: %w", marshalErr))
			}
		}
		if _, err := fmt.Fprintln(os.Stderr, line); err != nil {
			l.writeErr = errors.Join(l.writeErr, fmt.Errorf("write catalog log to stderr: %w", err))
		}
	}
}

func (l *catalogLogger) debug(event, message string, fields map[string]any) {
	l.log("debug", event, message, fields)
}

func (l *catalogLogger) info(event, message string, fields map[string]any) {
	l.log("info", event, message, fields)
}

func (l *catalogLogger) warn(event, message string, fields map[string]any) {
	l.log("warn", event, message, fields)
}

func (l *catalogLogger) error(event, message string, err error, fields map[string]any) {
	cloned := make(map[string]any, len(fields)+1)
	for key, value := range fields {
		cloned[key] = value
	}
	if err != nil {
		cloned["error"] = err.Error()
	}
	l.log("error", event, message, cloned)
}

func (l *catalogLogger) close() error {
	if l == nil || l.file == nil {
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	resultErr := l.writeErr
	if err := l.file.Sync(); err != nil {
		syncErr := fmt.Errorf("sync log file %q: %w", l.filePath, err)
		if closeErr := l.file.Close(); closeErr != nil {
			return errors.Join(resultErr, syncErr, fmt.Errorf("close log file after sync failure: %w", closeErr))
		}
		return errors.Join(resultErr, syncErr)
	}
	if err := l.file.Close(); err != nil {
		return errors.Join(resultErr, fmt.Errorf("close log file %q: %w", l.filePath, err))
	}
	return resultErr
}

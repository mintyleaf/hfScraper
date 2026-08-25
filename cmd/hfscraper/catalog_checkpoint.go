package main

import (
	"compress/gzip"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

type catalogCheckpoint struct {
	Endpoint       string         `json:"endpoint"`
	Earliest       string         `json:"earliest"`
	Latest         string         `json:"latest"`
	RequireWeights bool           `json:"require_weights"`
	Complete       bool           `json:"complete"`
	Pages          int            `json:"pages"`
	Scanned        int            `json:"scanned"`
	CreatedAt      string         `json:"created_at"`
	Models         []catalogModel `json:"models"`
}

func resolvedCheckpointPath(outputDir, configured string) string {
	if configured == "" || filepath.IsAbs(configured) {
		return configured
	}
	return filepath.Join(outputDir, configured)
}

func loadCatalogCheckpoint(path, endpoint string, earliest, latest time.Time, requireWeights bool) (*catalogCheckpoint, error) {
	if path == "" {
		return nil, nil
	}
	file, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	defer file.Close()
	gz, err := gzip.NewReader(file)
	if err != nil {
		return nil, err
	}
	defer gz.Close()
	var checkpoint catalogCheckpoint
	if err := json.NewDecoder(gz).Decode(&checkpoint); err != nil {
		return nil, err
	}
	if !checkpoint.Complete || checkpoint.Endpoint != endpoint || checkpoint.Earliest != earliest.Format(time.RFC3339Nano) || checkpoint.Latest != latest.Format(time.RFC3339Nano) || checkpoint.RequireWeights != requireWeights {
		return nil, nil
	}
	return &checkpoint, nil
}

func saveCatalogCheckpoint(path string, checkpoint catalogCheckpoint) error {
	if path == "" || !checkpoint.Complete {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	partial := path + ".part"
	file, err := os.Create(partial)
	if err != nil {
		return err
	}
	defer file.Close()
	gz := gzip.NewWriter(file)
	if err := json.NewEncoder(gz).Encode(checkpoint); err != nil {
		return fmt.Errorf("encode catalog checkpoint: %w", err)
	}
	if err := gz.Close(); err != nil {
		return err
	}
	if err := file.Sync(); err != nil {
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	if err := os.Rename(partial, path); err != nil {
		return err
	}
	return nil
}

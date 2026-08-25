package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"time"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if len(os.Args) > 1 && os.Args[1] == "catalog" {
		if err := runCatalogCLI(ctx, os.Args[2:]); err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "market-sample" {
		if err := runMarketSamplingCLI(ctx, os.Args[2:]); err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
			os.Exit(1)
		}
		return
	}

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
	if err := run(ctx, opts); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

type AnalyzeRequest struct {
	BBox          []float64 `json:"bbox"`
	T1Range       string    `json:"t1_range"`
	T2Range       string    `json:"t2_range"`
	NDVIThreshold *float64  `json:"ndvi_threshold,omitempty"`
	MinPixels     *int      `json:"min_pixels,omitempty"`
}

type MLStats struct {
	ChangedPixels           int     `json:"changed_pixels"`
	ChangedAreaHectares     float64 `json:"changed_area_hectares"`
	MeanDeltaNDVI           float64 `json:"mean_delta_ndvi"`
	MeanConfidence          float64 `json:"mean_confidence"`
	IsDeforestationDetected bool    `json:"is_deforestation_detected"`
}

type MLResponse struct {
	Status     string                 `json:"status"`
	BBox       []float64              `json:"bbox"`
	DataSource map[string]interface{} `json:"data_source"`
	Stats      MLStats                `json:"stats"`
	Images     struct {
		T1RGB       string `json:"t1_rgb"`
		T2RGB       string `json:"t2_rgb"`
		MaskOverlay string `json:"mask_overlay"`
	} `json:"images"`
}

type AuditReportResponse struct {
	MLResponse
	AuditVerdict    string    `json:"audit_verdict"`
	AuditStandard   string    `json:"audit_standard"`
	Timestamp       time.Time `json:"timestamp"`
	ExecutionTimeMs int64     `json:"execution_time_ms"`
}

const (
	MLServiceURL = "http://127.0.0.1:8000/analyze"
	ServerPort   = ":8080"
)

func enableCORS(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		next(w, r)
	}
}

func handleAnalyze(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	startTime := time.Now()

	var req AnalyzeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid JSON body: %v", err), http.StatusBadRequest)
		return
	}

	if len(req.BBox) != 4 {
		http.Error(w, "BBox must contain exactly 4 coordinates [min_lon, min_lat, max_lon, max_lat]", http.StatusBadRequest)
		return
	}

	payloadBytes, err := json.Marshal(req)
	if err != nil {
		http.Error(w, "Failed to serialize ML request", http.StatusInternalServerError)
		return
	}

	log.Printf("[Go Gateway] Dispatching ML analysis task for BBox: %v", req.BBox)
	mlReq, err := http.NewRequestWithContext(r.Context(), http.MethodPost, MLServiceURL, bytes.NewBuffer(payloadBytes))
	if err != nil {
		http.Error(w, fmt.Sprintf("Error building ML request: %v", err), http.StatusInternalServerError)
		return
	}
	mlReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 90 * time.Second}
	resp, err := client.Do(mlReq)
	if err != nil {
		log.Printf("[Go Gateway] ML Service error: %v", err)
		http.Error(w, fmt.Sprintf("ML Engine is unavailable on %s: %v", MLServiceURL, err), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		http.Error(w, fmt.Sprintf("ML Engine error (%d): %s", resp.StatusCode, string(body)), resp.StatusCode)
		return
	}

	var mlResult MLResponse
	if err := json.NewDecoder(resp.Body).Decode(&mlResult); err != nil {
		http.Error(w, "Failed to parse ML response", http.StatusInternalServerError)
		return
	}

	// EUDR Аудиторское заключение (дерево решений по стандарту ЕС 2023/1115)
	verdict := "СООТВЕТСТВУЕТ ТРЕБОВАНИЯМ (EUDR COMPLIANT): Значимого снижения лесного полога не зафиксировано."
	if mlResult.Stats.IsDeforestationDetected {
		verdict = fmt.Sprintf("ТРЕВОГА: НЕСООТВЕТСТВИЕ EUDR (NON-COMPLIANT). Зафиксировано обезлесение %.2f га (порог > 0.5 га) с уверенностью %.1f%% после 31.12.2020.",
			mlResult.Stats.ChangedAreaHectares,
			mlResult.Stats.MeanConfidence*100,
		)
	}

	finalReport := AuditReportResponse{
		MLResponse:      mlResult,
		AuditVerdict:    verdict,
		AuditStandard:   "EUDR Regulation (EU) 2023/1115 Due Diligence Statement",
		Timestamp:       time.Now().UTC(),
		ExecutionTimeMs: time.Since(startTime).Milliseconds(),
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(finalReport); err != nil {
		log.Printf("Failed to encode response: %v", err)
	}
}

func main() {
	mux := http.NewServeMux()

	// REST API
	mux.HandleFunc("/api/v1/analyze", enableCORS(handleAnalyze))
	mux.HandleFunc("/health", enableCORS(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"status":"ok","server":"earth-change-go-gateway","time":"%s"}`, time.Now().Format(time.RFC3339))
	}))

	// Поиск директории web-frontend
	frontendPath := filepath.Join("..", "web-frontend")
	if _, err := os.Stat(frontendPath); os.IsNotExist(err) {
		frontendPath = "./web-frontend"
	}

	fs := http.FileServer(http.Dir(frontendPath))
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		// Не кэшировать index.html во время разработки
		if strings.HasSuffix(r.URL.Path, ".html") || r.URL.Path == "/" {
			w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")
		}
		fs.ServeHTTP(w, r)
	})

	log.Printf("====================================================================")
	log.Printf("  EARTH CHANGE DETECTION PLATFORM (Go + PyTorch + Web UI)")
	log.Printf("  Веб-интерфейс:   http://localhost%s", ServerPort)
	log.Printf("  API Gateway:     http://localhost%s/api/v1/analyze", ServerPort)
	log.Printf("  ML Engine Proxy: %s", MLServiceURL)
	log.Printf("====================================================================")

	if err := http.ListenAndServe(ServerPort, mux); err != nil {
		log.Fatalf("Server failed to start: %v", err)
	}
}

package main

import (
	"bytes"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// =============================================================================
// Data Types & Contracts
// =============================================================================

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
	Category                string  `json:"category,omitempty"`
	Severity                string  `json:"severity,omitempty"`
	DescriptionRU           string  `json:"description_ru,omitempty"`
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
	RequestID       string    `json:"request_id"`
}

type IncidentAlert struct {
	IncidentID     string    `json:"incident_id"`
	Timestamp      time.Time `json:"timestamp"`
	Category       string    `json:"category"`
	Severity       string    `json:"severity"`
	AreaHectares   float64   `json:"area_hectares"`
	DescriptionRU  string    `json:"description_ru"`
	Verdict        string    `json:"verdict"`
	Standard       string    `json:"standard"`
	CoordinatesBBox []float64 `json:"coordinates_bbox"`
}

// =============================================================================
// In-Memory High-Performance LRU/TTL Cache
// =============================================================================

type CacheItem struct {
	Data        []byte
	ContentType string
	ExpiresAt   time.Time
}

type MemoryCache struct {
	mu     sync.RWMutex
	items  map[string]CacheItem
	hits   uint64
	misses uint64
}

func NewMemoryCache() *MemoryCache {
	mc := &MemoryCache{
		items: make(map[string]CacheItem),
	}
	// Background garbage collector for expired entries
	go func() {
		ticker := time.NewTicker(2 * time.Minute)
		for range ticker.C {
			mc.cleanup()
		}
	}()
	return mc
}

func (mc *MemoryCache) Get(key string) ([]byte, string, bool) {
	mc.mu.RLock()
	defer mc.mu.RUnlock()

	item, found := mc.items[key]
	if !found || time.Now().After(item.ExpiresAt) {
		atomic.AddUint64(&mc.misses, 1)
		return nil, "", false
	}
	atomic.AddUint64(&mc.hits, 1)
	return item.Data, item.ContentType, true
}

func (mc *MemoryCache) Set(key string, data []byte, contentType string, ttl time.Duration) {
	mc.mu.Lock()
	defer mc.mu.Unlock()
	mc.items[key] = CacheItem{
		Data:        data,
		ContentType: contentType,
		ExpiresAt:   time.Now().Add(ttl),
	}
}

func (mc *MemoryCache) cleanup() {
	mc.mu.Lock()
	defer mc.mu.Unlock()
	now := time.Now()
	for k, v := range mc.items {
		if now.After(v.ExpiresAt) {
			delete(mc.items, k)
		}
	}
}

// =============================================================================
// Metrics Telemetry
// =============================================================================

type ServerMetrics struct {
	mu            sync.RWMutex
	TotalRequests uint64            `json:"total_requests"`
	TotalErrors   uint64            `json:"total_errors"`
	StartTime     time.Time         `json:"start_time"`
	Endpoints     map[string]uint64 `json:"endpoints"`
}

func NewServerMetrics() *ServerMetrics {
	return &ServerMetrics{
		StartTime: time.Now(),
		Endpoints: make(map[string]uint64),
	}
}

func (sm *ServerMetrics) RecordRequest(path string, isError bool) {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	sm.TotalRequests++
	sm.Endpoints[path]++
	if isError {
		sm.TotalErrors++
	}
}

// =============================================================================
// Webhook Alert Dispatcher
// =============================================================================

type WebhookManager struct {
	mu   sync.RWMutex
	URLs []string
}

func NewWebhookManager() *WebhookManager {
	return &WebhookManager{
		URLs: []string{},
	}
}

func (wm *WebhookManager) Register(url string) {
	wm.mu.Lock()
	defer wm.mu.Unlock()
	for _, u := range wm.URLs {
		if u == url {
			return
		}
	}
	wm.URLs = append(wm.URLs, url)
	log.Printf("[Webhooks] Зарегистрирован новый вебхук получатель: %s", url)
}

func (wm *WebhookManager) List() []string {
	wm.mu.RLock()
	defer wm.mu.RUnlock()
	res := make([]string, len(wm.URLs))
	copy(res, wm.URLs)
	return res
}

func (wm *WebhookManager) Unregister(url string) bool {
	wm.mu.Lock()
	defer wm.mu.Unlock()
	for i, u := range wm.URLs {
		if u == url {
			wm.URLs = append(wm.URLs[:i], wm.URLs[i+1:]...)
			log.Printf("[Webhooks] Вебхук разрегистрирован: %s", url)
			return true
		}
	}
	return false
}

func (wm *WebhookManager) DispatchAlert(alert IncidentAlert) {
	wm.mu.RLock()
	urls := make([]string, len(wm.URLs))
	copy(urls, wm.URLs)
	wm.mu.RUnlock()

	if len(urls) == 0 {
		return
	}

	payload, err := json.Marshal(alert)
	if err != nil {
		return
	}

	// Dispatch asynchronously
	go func() {
		client := &http.Client{Timeout: 8 * time.Second}
		for _, targetURL := range urls {
			req, err := http.NewRequest(http.MethodPost, targetURL, bytes.NewBuffer(payload))
			if err != nil {
				continue
			}
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("User-Agent", "TerraVision-Alert-Dispatcher/2.5")
			resp, err := client.Do(req)
			if err != nil {
				log.Printf("[Webhooks] Ошибка отправки на %s: %v", targetURL, err)
				continue
			}
			resp.Body.Close()
			log.Printf("[Webhooks] Оповещение ЧС успешно доставлено на %s (HTTP %d)", targetURL, resp.StatusCode)
		}
	}()
}

// =============================================================================
// Global Services & Constants
// =============================================================================

const (
	MLServiceURL     = "http://127.0.0.1:8000/analyze"
	MLServicePairURL = "http://127.0.0.1:8000/analyze_pair"
	MLHealthURL      = "http://127.0.0.1:8000/health"
	ServerPort       = ":8080"
)

var (
	cache   = NewMemoryCache()
	metrics = NewServerMetrics()
	alerts  = NewWebhookManager()
)

func generateRequestID() string {
	b := make([]byte, 8)
	rand.Read(b)
	return hex.EncodeToString(b)
}

func hashKey(prefix string, data []byte) string {
	h := sha256.Sum256(data)
	return fmt.Sprintf("%s:%s", prefix, hex.EncodeToString(h[:8]))
}

// =============================================================================
// Middlewares: CORS, Tracing, Metrics
// =============================================================================

func apiMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// CORS Headers
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, DELETE")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Request-ID")
		w.Header().Set("Access-Control-Expose-Headers", "X-Request-ID, X-Cache, X-Response-Time-Ms")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}

		// Request ID
		reqID := r.Header.Get("X-Request-ID")
		if reqID == "" {
			reqID = generateRequestID()
		}
		w.Header().Set("X-Request-ID", reqID)

		startTime := time.Now()
		isError := false

		// Wrap writer to detect status
		next(w, r)

		duration := time.Since(startTime)
		w.Header().Set("X-Response-Time-Ms", fmt.Sprintf("%d", duration.Milliseconds()))
		metrics.RecordRequest(r.URL.Path, isError)

		if !strings.HasPrefix(r.URL.Path, "/styles.css") && !strings.HasPrefix(r.URL.Path, "/app.js") {
			log.Printf("[GATEWAY] [%s] %s %s -> %v", reqID, r.Method, r.URL.Path, duration)
		}
	}
}

// =============================================================================
// Handlers
// =============================================================================

func handleAnalyze(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	reqID := w.Header().Get("X-Request-ID")
	startTime := time.Now()

	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Failed to read request body", http.StatusBadRequest)
		return
	}

	// Check Cache
	cacheKey := hashKey("analyze", bodyBytes)
	if cachedData, cType, found := cache.Get(cacheKey); found {
		w.Header().Set("Content-Type", cType)
		w.Header().Set("X-Cache", "HIT")
		w.Write(cachedData)
		return
	}

	var req AnalyzeRequest
	if err := json.Unmarshal(bodyBytes, &req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid JSON body: %v", err), http.StatusBadRequest)
		return
	}

	if len(req.BBox) != 4 {
		http.Error(w, "BBox must contain exactly 4 coordinates [min_lon, min_lat, max_lon, max_lat]", http.StatusBadRequest)
		return
	}

	mlReq, err := http.NewRequestWithContext(r.Context(), http.MethodPost, MLServiceURL, bytes.NewBuffer(bodyBytes))
	if err != nil {
		http.Error(w, fmt.Sprintf("Error building ML request: %v", err), http.StatusInternalServerError)
		return
	}
	mlReq.Header.Set("Content-Type", "application/json")
	mlReq.Header.Set("X-Request-ID", reqID)

	client := &http.Client{Timeout: 90 * time.Second}
	resp, err := client.Do(mlReq)
	if err != nil {
		log.Printf("[Gateway Error] ML Service: %v", err)
		http.Error(w, fmt.Sprintf("ML Engine is unavailable on %s: %v", MLServiceURL, err), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		http.Error(w, fmt.Sprintf("ML Engine error (%d): %s", resp.StatusCode, string(respBody)), resp.StatusCode)
		return
	}

	var mlResult MLResponse
	if err := json.NewDecoder(resp.Body).Decode(&mlResult); err != nil {
		http.Error(w, "Failed to parse ML response", http.StatusInternalServerError)
		return
	}

	// Formulate Legal & Emergency Verdict
	var verdict string
	switch mlResult.Stats.Category {
	case "FLOOD":
		verdict = fmt.Sprintf("ТРЕВОГА (ПАВОДОК/НАВОДНЕНИЕ): Зафиксировано затопление на площади %.2f га. %s",
			mlResult.Stats.ChangedAreaHectares, mlResult.Stats.DescriptionRU)
	case "WILDFIRE":
		verdict = fmt.Sprintf("ТРЕВОГА (ЛЕСНОЙ ПОЖАР): Обнаружена свежая гарь / термопоражение на площади %.2f га. %s",
			mlResult.Stats.ChangedAreaHectares, mlResult.Stats.DescriptionRU)
	case "DEFORESTATION":
		verdict = fmt.Sprintf("ТРЕВОГА: НЕСООТВЕТСТВИЕ EUDR (ВЫРУБКА). Зафиксировано обезлесение %.2f га (порог > 0.5 га) с уверенностью %.1f%% после 31.12.2020. %s",
			mlResult.Stats.ChangedAreaHectares, mlResult.Stats.MeanConfidence*100, mlResult.Stats.DescriptionRU)
	default:
		if mlResult.Stats.IsDeforestationDetected {
			verdict = fmt.Sprintf("ТРЕВОГА: НЕСООТВЕТСТВИЕ EUDR (ОБЕЗЛЕСЕНИЕ). Зафиксирована потеря полога %.2f га.",
				mlResult.Stats.ChangedAreaHectares)
		} else {
			verdict = "СООТВЕТСТВУЕТ ТРЕБОВАНИЯМ: Критических стихийных бедствий и нарушений лесного фонда не зафиксировано."
		}
	}

	finalReport := AuditReportResponse{
		MLResponse:      mlResult,
		AuditVerdict:    verdict,
		AuditStandard:   "EUDR Regulation (EU) 2023/1115 Due Diligence Statement",
		Timestamp:       time.Now().UTC(),
		ExecutionTimeMs: time.Since(startTime).Milliseconds(),
		RequestID:       reqID,
	}

	// Dispatch Webhook if critical event detected
	if mlResult.Stats.ChangedAreaHectares > 0.5 && mlResult.Stats.Category != "" {
		alerts.DispatchAlert(IncidentAlert{
			IncidentID:      reqID,
			Timestamp:       time.Now().UTC(),
			Category:        mlResult.Stats.Category,
			Severity:        mlResult.Stats.Severity,
			AreaHectares:    mlResult.Stats.ChangedAreaHectares,
			DescriptionRU:   mlResult.Stats.DescriptionRU,
			Verdict:         verdict,
			Standard:        finalReport.AuditStandard,
			CoordinatesBBox: req.BBox,
		})
	}

	resBytes, err := json.Marshal(finalReport)
	if err != nil {
		http.Error(w, "Failed to encode response", http.StatusInternalServerError)
		return
	}

	// Store in Cache (15 min TTL)
	cache.Set(cacheKey, resBytes, "application/json", 15*time.Minute)

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Cache", "MISS")
	w.Write(resBytes)
}

func handleAnalyzePair(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Failed to read body", http.StatusBadRequest)
		return
	}
	mlReq, err := http.NewRequestWithContext(r.Context(), http.MethodPost, MLServicePairURL, bytes.NewBuffer(body))
	if err != nil {
		http.Error(w, "Failed to create ML request", http.StatusInternalServerError)
		return
	}
	mlReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 90 * time.Second}
	resp, err := client.Do(mlReq)
	if err != nil {
		http.Error(w, fmt.Sprintf("ML Service error: %v", err), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	io.Copy(w, resp.Body)
}

func handleScenarios(w http.ResponseWriter, r *http.Request) {
	cacheKey := "scenarios:list"
	if data, cType, found := cache.Get(cacheKey); found {
		w.Header().Set("Content-Type", cType)
		w.Header().Set("X-Cache", "HIT")
		w.Write(data)
		return
	}

	resp, err := http.Get("http://127.0.0.1:8000/preset_scenarios")
	if err != nil {
		http.Error(w, fmt.Sprintf("ML Service error: %v", err), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	cache.Set(cacheKey, body, "application/json", 1*time.Hour)

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Cache", "MISS")
	w.WriteHeader(resp.StatusCode)
	w.Write(body)
}

func handleScenarioDetail(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(strings.Trim(r.URL.Path, "/"), "/")
	if len(parts) < 4 {
		http.Error(w, "Scenario ID required in path /api/v1/scenario/{id}", http.StatusBadRequest)
		return
	}
	scID := parts[3]
	cacheKey := fmt.Sprintf("scenario:detail:%s", scID)

	if data, cType, found := cache.Get(cacheKey); found {
		w.Header().Set("Content-Type", cType)
		w.Header().Set("X-Cache", "HIT")
		w.Write(data)
		return
	}

	resp, err := http.Get(fmt.Sprintf("http://127.0.0.1:8000/preset_scenario/%s", scID))
	if err != nil {
		http.Error(w, fmt.Sprintf("ML Service error: %v", err), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	cache.Set(cacheKey, body, "application/json", 1*time.Hour)

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Cache", "MISS")
	w.WriteHeader(resp.StatusCode)
	w.Write(body)
}

func handleGeoJSON(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(strings.Trim(r.URL.Path, "/"), "/")
	if len(parts) < 4 {
		http.Error(w, "Scenario ID required in path /api/v1/geojson/{id}", http.StatusBadRequest)
		return
	}
	scID := parts[3]
	cacheKey := fmt.Sprintf("geojson:%s", scID)

	if data, cType, found := cache.Get(cacheKey); found {
		w.Header().Set("Content-Type", cType)
		w.Header().Set("X-Cache", "HIT")
		w.Write(data)
		return
	}

	resp, err := http.Get(fmt.Sprintf("http://127.0.0.1:8000/api/v1/geojson/%s", scID))
	if err != nil {
		http.Error(w, fmt.Sprintf("ML Service error: %v", err), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	cache.Set(cacheKey, body, "application/geo+json", 1*time.Hour)

	w.Header().Set("Content-Type", "application/geo+json")
	w.Header().Set("X-Cache", "MISS")
	w.WriteHeader(resp.StatusCode)
	w.Write(body)
}

func handleMetrics(w http.ResponseWriter, r *http.Request) {
	metrics.mu.RLock()
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	hitCount := atomic.LoadUint64(&cache.hits)
	missCount := atomic.LoadUint64(&cache.misses)
	var hitRatio float64
	if hitCount+missCount > 0 {
		hitRatio = float64(hitCount) / float64(hitCount+missCount) * 100
	}

	// Support Prometheus exposition format (/metrics or format=prometheus or Accept: text/plain)
	if r.URL.Query().Get("format") == "prometheus" || strings.Contains(r.Header.Get("Accept"), "text/plain") || r.URL.Path == "/metrics" {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
		fmt.Fprintf(w, "# HELP terravision_requests_total Total HTTP requests handled by Go gateway\n")
		fmt.Fprintf(w, "# TYPE terravision_requests_total counter\n")
		fmt.Fprintf(w, "terravision_requests_total %d\n", metrics.TotalRequests)
		fmt.Fprintf(w, "# HELP terravision_errors_total Total errors encountered\n")
		fmt.Fprintf(w, "# TYPE terravision_errors_total counter\n")
		fmt.Fprintf(w, "terravision_errors_total %d\n", metrics.TotalErrors)
		fmt.Fprintf(w, "# HELP terravision_cache_hits_total In-memory cache hits\n")
		fmt.Fprintf(w, "# TYPE terravision_cache_hits_total counter\n")
		fmt.Fprintf(w, "terravision_cache_hits_total %d\n", hitCount)
		fmt.Fprintf(w, "# HELP terravision_cache_misses_total In-memory cache misses\n")
		fmt.Fprintf(w, "# TYPE terravision_cache_misses_total counter\n")
		fmt.Fprintf(w, "terravision_cache_misses_total %d\n", missCount)
		fmt.Fprintf(w, "# HELP terravision_cache_hit_ratio_percent Percentage of cache hits\n")
		fmt.Fprintf(w, "# TYPE terravision_cache_hit_ratio_percent gauge\n")
		fmt.Fprintf(w, "terravision_cache_hit_ratio_percent %.2f\n", hitRatio)
		fmt.Fprintf(w, "# HELP terravision_goroutines Active goroutines count\n")
		fmt.Fprintf(w, "# TYPE terravision_goroutines gauge\n")
		fmt.Fprintf(w, "terravision_goroutines %d\n", runtime.NumGoroutine())
		fmt.Fprintf(w, "# HELP terravision_memory_alloc_bytes Current allocated heap memory bytes\n")
		fmt.Fprintf(w, "# TYPE terravision_memory_alloc_bytes gauge\n")
		fmt.Fprintf(w, "terravision_memory_alloc_bytes %d\n", m.Alloc)
		metrics.mu.RUnlock()
		return
	}

	data := map[string]interface{}{
		"gateway_uptime_seconds": time.Since(metrics.StartTime).Seconds(),
		"total_requests":         metrics.TotalRequests,
		"total_errors":           metrics.TotalErrors,
		"goroutines_count":       runtime.NumGoroutine(),
		"memory_alloc_mb":        float64(m.Alloc) / (1024 * 1024),
		"memory_sys_mb":          float64(m.Sys) / (1024 * 1024),
		"cache": map[string]interface{}{
			"hits":      hitCount,
			"misses":    missCount,
			"hit_ratio": fmt.Sprintf("%.1f%%", hitRatio),
		},
		"endpoints": metrics.Endpoints,
	}
	metrics.mu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(data)
}

func handleWebhooks(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodPost {
		var req struct {
			URL string `json:"url"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.URL == "" {
			http.Error(w, "Invalid body: 'url' field required", http.StatusBadRequest)
			return
		}
		alerts.Register(req.URL)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"status":"registered","url":"%s"}`, req.URL)
		return
	}

	if r.Method == http.MethodDelete {
		var req struct {
			URL string `json:"url"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.URL == "" {
			http.Error(w, "Invalid body: 'url' field required", http.StatusBadRequest)
			return
		}
		deleted := alerts.Unregister(req.URL)
		w.Header().Set("Content-Type", "application/json")
		if deleted {
			fmt.Fprintf(w, `{"status":"unregistered","url":"%s"}`, req.URL)
		} else {
			http.Error(w, "Webhook URL not found", http.StatusNotFound)
		}
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"registered_webhooks": alerts.List(),
	})
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	var pyHealth map[string]interface{}
	resp, err := http.Get(MLHealthURL)
	if err == nil {
		json.NewDecoder(resp.Body).Decode(&pyHealth)
		resp.Body.Close()
	}

	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	healthStatus := map[string]interface{}{
		"status":    "healthy",
		"gateway":   "earth-change-go-gateway",
		"version":   "2.5.0",
		"uptime":    time.Since(metrics.StartTime).String(),
		"num_cpu":   runtime.NumCPU(),
		"alloc_mb":  float64(m.Alloc) / (1024 * 1024),
		"ml_engine": pyHealth,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(healthStatus)
}

// =============================================================================
// Main Entrypoint
// =============================================================================

func main() {
	mux := http.NewServeMux()

	// REST API Routes
	mux.HandleFunc("/api/v1/analyze", apiMiddleware(handleAnalyze))
	mux.HandleFunc("/api/v1/analyze_pair", apiMiddleware(handleAnalyzePair))
	mux.HandleFunc("/api/v1/scenarios", apiMiddleware(handleScenarios))
	mux.HandleFunc("/api/v1/scenario/", apiMiddleware(handleScenarioDetail))
	mux.HandleFunc("/api/v1/geojson/", apiMiddleware(handleGeoJSON))
	mux.HandleFunc("/api/v1/metrics", apiMiddleware(handleMetrics))
	mux.HandleFunc("/metrics", apiMiddleware(handleMetrics))
	mux.HandleFunc("/api/v1/webhooks", apiMiddleware(handleWebhooks))
	mux.HandleFunc("/health", apiMiddleware(handleHealth))

	// Static Web Frontend Server
	frontendPath := filepath.Join("..", "web-frontend")
	if _, err := os.Stat(frontendPath); os.IsNotExist(err) {
		frontendPath = "./web-frontend"
	}

	fs := http.FileServer(http.Dir(frontendPath))
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, ".html") || r.URL.Path == "/" {
			w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")
		}
		fs.ServeHTTP(w, r)
	})

	log.Printf("====================================================================")
	log.Printf("  TERRAVISION AI ENTERPRISE GATEWAY v2.5 (Go + PyTorch)")
	log.Printf("  Веб-интерфейс:    http://localhost%s", ServerPort)
	log.Printf("  API Анализа:      http://localhost%s/api/v1/analyze", ServerPort)
	log.Printf("  Вектор GeoJSON:   http://localhost%s/api/v1/geojson/{scenario_id}", ServerPort)
	log.Printf("  Метрики сервиса:  http://localhost%s/api/v1/metrics", ServerPort)
	log.Printf("  Телеметрия GPU:   http://localhost%s/health", ServerPort)
	log.Printf("  Вебхуки тревог:   http://localhost%s/api/v1/webhooks", ServerPort)
	log.Printf("====================================================================")

	server := &http.Server{
		Addr:         ServerPort,
		Handler:      mux,
		ReadTimeout:  120 * time.Second,
		WriteTimeout: 120 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	if err := server.ListenAndServe(); err != nil {
		log.Fatalf("Server failed to start: %v", err)
	}
}

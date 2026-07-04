let cropProfiles = [];
let diseaseProfiles = [];
let dataset = null;
let analyticsData = null;
let chartInstances = {};
let recommendationMode = "recommend";

const featureWeights = {
  nitrogen: 0.2,
  phosphorous: 0.16,
  potassium: 0.16,
  temperature: 0.14,
  humidity: 0.13,
  ph: 0.1,
  rainfall: 0.11,
};

const cropColors = {
  Rice: "#3ccf7a",
  Wheat: "#d4a574",
  Maize: "#f4d03f",
  Cotton: "#e8f4ff",
  Sugarcane: "#7ddc5f",
  Tomato: "#ff6b6b",
  Potato: "#c9a66b",
  Soybean: "#8bc34a",
  Chickpea: "#ffb347",
  Groundnut: "#c68642",
  Millet: "#daa520",
  Mustard: "#ffd700",
  Sunflower: "#ffc107",
  Banana: "#ffe066",
};

async function loadDataset() {
  try {
    const response = await fetch("/data/opticrop-data.json");
    if (!response.ok)
      throw new Error(`Dataset request failed with ${response.status}`);
    dataset = await response.json();
    const seen = new Set();
    cropProfiles = [];
    dataset.crops.forEach((entry) => {
      if (seen.has(entry.crop)) return;
      seen.add(entry.crop);
      cropProfiles.push({
        name: entry.crop,
        description: entry.description,
        yield: entry.yield,
        strategy: entry.strategy,
      });
    });
    diseaseProfiles = dataset.diseases;
  } catch (error) {
    console.error("Failed to load dataset", error);
    cropProfiles = [];
    diseaseProfiles = [];
  }
}

function getCropProfiles() {
  return cropProfiles.length ? cropProfiles : [];
}

function getDiseaseProfiles() {
  return diseaseProfiles.length ? diseaseProfiles : [];
}

function filterCropRecords(location = "", season = "") {
  if (!dataset?.crops) return [];
  return dataset.crops.filter((entry) => {
    if (location && entry.location !== location) return false;
    if (season && entry.season !== season) return false;
    return true;
  });
}

function getRecordsForCrop(cropName, location = "", season = "") {
  return filterCropRecords(location, season).filter(
    (entry) => entry.crop === cropName,
  );
}

function scoreMetric(value, range) {
  const [min, max] = range;
  const idealMid = (min + max) / 2;
  const halfRange = (max - min) / 2 || 1;
  if (value >= min && value <= max) return 1;
  const deviation = Math.abs(value - idealMid) - halfRange;
  return Math.max(0.15, 1 - deviation / (halfRange * 1.5));
}

function buildRecommendation(values, location = "", season = "") {
  const recordsByCrop = {};
  filterCropRecords(location, season).forEach((entry) => {
    recordsByCrop[entry.crop] = recordsByCrop[entry.crop] || [];
    recordsByCrop[entry.crop].push(entry);
  });

  const crops = Object.keys(recordsByCrop).map((name) => {
    const profile = getCropProfiles().find((crop) => crop.name === name) || {
      name,
      description: "Crop profile from regional dataset.",
      yield: "Moderate",
      strategy: "Monitor soil nutrients and seasonal rainfall.",
    };
    return profile;
  });

  const scored = crops.map((crop) => {
    const records = recordsByCrop[crop.name] || [];
    const metrics = [
      [
        "Nitrogen",
        records.length
          ? scoreMetric(values.nitrogen, [
              Math.min(...records.map((e) => e.nitrogen)) * 0.9,
              Math.max(...records.map((e) => e.nitrogen)) * 1.1,
            ])
          : 0.5,
      ],
      [
        "Phosphorous",
        records.length
          ? scoreMetric(values.phosphorous, [
              Math.min(...records.map((e) => e.phosphorous)) * 0.9,
              Math.max(...records.map((e) => e.phosphorous)) * 1.1,
            ])
          : 0.5,
      ],
      [
        "Potassium",
        records.length
          ? scoreMetric(values.potassium, [
              Math.min(...records.map((e) => e.potassium)) * 0.9,
              Math.max(...records.map((e) => e.potassium)) * 1.1,
            ])
          : 0.5,
      ],
      [
        "Temperature",
        records.length
          ? scoreMetric(values.temperature, [
              Math.min(...records.map((e) => e.temperature)) * 0.9,
              Math.max(...records.map((e) => e.temperature)) * 1.1,
            ])
          : 0.5,
      ],
      [
        "Humidity",
        records.length
          ? scoreMetric(values.humidity, [
              Math.min(...records.map((e) => e.humidity)) * 0.9,
              Math.max(...records.map((e) => e.humidity)) * 1.1,
            ])
          : 0.5,
      ],
      [
        "pH",
        records.length
          ? scoreMetric(values.ph, [
              Math.min(...records.map((e) => e.ph)) * 0.95,
              Math.max(...records.map((e) => e.ph)) * 1.05,
            ])
          : 0.5,
      ],
      [
        "Rainfall",
        records.length
          ? scoreMetric(values.rainfall, [
              Math.min(...records.map((e) => e.rainfall)) * 0.9,
              Math.max(...records.map((e) => e.rainfall)) * 1.1,
            ])
          : 0.5,
      ],
    ];

    const heuristicAverage =
      metrics.reduce((sum, [, score]) => sum + score, 0) / metrics.length;
    const trainingSimilarity = records.length
      ? records.reduce((sum, sample) => {
          const similarity = Object.entries(featureWeights).reduce(
            (acc, [key, weight]) => {
              const scale =
                key === "ph"
                  ? 1.2
                  : key === "temperature"
                    ? 7
                    : key === "humidity"
                      ? 20
                      : key === "rainfall"
                        ? 60
                        : 50;
              acc +=
                weight *
                Math.max(0, 1 - Math.abs(values[key] - sample[key]) / scale);
              return acc;
            },
            0,
          );
          return sum + similarity;
        }, 0) / records.length
      : heuristicAverage;

    const average = Number(
      (0.6 * heuristicAverage + 0.4 * trainingSimilarity).toFixed(3),
    );
    return { ...crop, average, metrics, trainingSimilarity };
  });

  scored.sort((a, b) => b.average - a.average);
  const best = scored[0] || null;
  const suitability =
    best?.average >= 0.85
      ? "Excellent"
      : best?.average >= 0.7
        ? "Strong"
        : best?.average >= 0.55
          ? "Moderate"
          : "Needs adjustment";
  return { best, scored, suitability };
}

async function getMlRecommendation(values) {
  try {
    const response = await fetch("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(values),
    });
    if (!response.ok) throw new Error(`Request failed with ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn(
      "ML recommendation unavailable, using fallback scoring.",
      error,
    );
    return null;
  }
}

async function getSuitabilityAssessment(values) {
  try {
    const response = await fetch("/api/suitability", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(values),
    });
    if (!response.ok) throw new Error(`Request failed with ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn("Suitability API unavailable, using client fallback.", error);
    return null;
  }
}

function buildClientSuitability(values, cropName) {
  const records = getRecordsForCrop(
    cropName,
    values.location || "",
    values.season || "",
  );
  if (!records.length) return null;

  const metrics = [
    ["Nitrogen", "nitrogen"],
    ["Phosphorous", "phosphorous"],
    ["Potassium", "potassium"],
    ["Temperature", "temperature"],
    ["Humidity", "humidity"],
    ["pH", "ph"],
    ["Rainfall", "rainfall"],
  ].map(([label, key]) => {
    const fieldValues = records.map((entry) => entry[key]);
    const min = Math.min(...fieldValues) * (key === "ph" ? 0.95 : 0.9);
    const max = Math.max(...fieldValues) * (key === "ph" ? 1.05 : 1.1);
    const score = scoreMetric(values[key], [min, max]);
    const status =
      score >= 0.85
        ? "Optimal"
        : score >= 0.65
          ? "Acceptable"
          : score >= 0.45
            ? "Suboptimal"
            : "Poor";
    return {
      label,
      key,
      score,
      status,
      ideal_range: [Number(min.toFixed(1)), Number(max.toFixed(1))],
      current_value: values[key],
    };
  });

  const overall =
    metrics.reduce((sum, metric) => sum + metric.score, 0) / metrics.length;
  const profile = records[0];
  return {
    crop: cropName,
    compatible: overall >= 0.55,
    suitability:
      overall >= 0.85
        ? "Excellent"
        : overall >= 0.7
          ? "Strong"
          : overall >= 0.55
            ? "Moderate"
            : "Poor",
    productivity_potential:
      overall >= 0.85
        ? "Very High"
        : overall >= 0.7
          ? "High"
          : overall >= 0.55
            ? "Moderate"
            : "Low",
    overall_score: Number((overall * 100).toFixed(1)),
    description: profile.description,
    yield_outlook: profile.yield,
    strategy: profile.strategy,
    metrics,
    remediation: metrics
      .filter((metric) => metric.score < 0.65)
      .map((metric) => ({
        factor: metric.label,
        action: `Adjust ${metric.label.toLowerCase()} toward ${metric.ideal_range[0]}–${metric.ideal_range[1]}`,
        detail: `Current value is ${metric.current_value}.`,
        priority: metric.score < 0.45 ? "High" : "Medium",
      })),
    summary: `${cropName} compatibility is ${Math.round(overall * 100)}% based on regional crop profiles.`,
  };
}

function statusClass(status) {
  return (status || "").toLowerCase();
}

async function renderRecommendation(values) {
  const resultCard = document.getElementById("result-card");
  if (!resultCard) return;

  if (recommendationMode === "evaluate") {
    resultCard.innerHTML = `<h3>Crop suitability</h3><div class="loading-overlay"><div class="spinner"></div>Evaluating crop suitability...</div>`;
    await renderCropSuitability(values);
    return;
  }

  resultCard.innerHTML = `<h3>Recommendation output</h3><div class="loading-overlay"><div class="spinner"></div>Analyzing field conditions...</div>`;
  const { best, scored } = buildRecommendation(
    values,
    values.location || "",
    values.season || "",
  );
  const mlRecommendation = await getMlRecommendation(values);
  const modelCandidates = (mlRecommendation?.candidates || []).map(
    (candidate) => ({
      name: candidate.crop,
      probability: Number(candidate.probability) || 0,
    }),
  );
  const rankedCandidates = scored
    .map((crop) => {
      const modelCandidate = modelCandidates.find(
        (candidate) => candidate.name === crop.name,
      );
      const modelProbability = modelCandidate ? modelCandidate.probability : 0;
      const ensembleScore = Number(
        (0.55 * modelProbability + 0.45 * crop.average).toFixed(3),
      );
      return { ...crop, modelProbability, ensembleScore };
    })
    .sort((a, b) => b.ensembleScore - a.ensembleScore);
  const primary = rankedCandidates[0] || best;
  if (!primary) {
    resultCard.innerHTML = `<h3>Recommendation output</h3><div class="placeholder">No crop profiles match the selected region and season filters.</div>`;
    return;
  }

  const scorePercent = Math.round(primary.ensembleScore * 100);
  const confidence = mlRecommendation?.confidence
    ? Math.min(96, Math.round(mlRecommendation.confidence))
    : Math.min(96, Math.round(65 + primary.average * 25));
  const suitability =
    primary.average >= 0.85
      ? "Excellent"
      : primary.average >= 0.7
        ? "Strong"
        : primary.average >= 0.55
          ? "Moderate"
          : "Needs adjustment";
  const fertilizer = mlRecommendation?.fertilizer || {
    summary: `${primary.name} benefits from a balanced nutrient plan that responds to current field conditions.`,
    recommendation: [
      {
        nutrient: "Nitrogen",
        target:
          "Use split doses to support early growth without excess stress.",
        priority: "High",
      },
      {
        nutrient: "Phosphorus",
        target: "Maintain phosphorus for root strength and establishment.",
        priority: "Medium",
      },
      {
        nutrient: "Potassium",
        target:
          "Support resilience and yield quality with steady potash supply.",
        priority: "High",
      },
    ],
    soil_note: "Soil pH appears suitable for the selected crop.",
  };

  const contextNote = [values.locationLabel, values.seasonLabel]
    .filter(Boolean)
    .join(" • ");

  resultCard.innerHTML = `
    <h3>Recommendation output</h3>
    ${contextNote ? `<p class="muted">${contextNote}</p>` : ""}
    <div class="result-banner">
      <h4>${primary.name}</h4>
      <p>${primary.description}</p>
      <div class="score-pill">Suitability: ${suitability} • ${scorePercent}% fit</div>
      <div class="score-pill" style="margin-left:8px;">ML confidence: ${confidence}%</div>
    </div>
    <div class="metric-list">
      ${primary.metrics
        .map(([label, score]) => {
          const percent = Math.round(score * 100);
          return `
        <div class="metric-row">
          <div class="metric-meta"><span>${label}</span><strong>${percent}%</strong></div>
          <div class="bar"><div style="width:${percent}%;"></div></div>
        </div>`;
        })
        .join("")}
    </div>
    <div class="mini-panel" style="margin-top:16px;">
      <h3>Action plan</h3>
      <p><strong>Yield outlook:</strong> ${primary.yield}</p>
      <p>${primary.strategy}</p>
      <p><strong>Field note:</strong> ${fertilizer.summary}</p>
    </div>
    <div class="mini-panel" style="margin-top:16px;">
      <h3>Fertilizer guidance</h3>
      <p>${fertilizer.soil_note}</p>
      <div class="matrix">
        ${fertilizer.recommendation.map((item) => `<div class="matrix-item"><strong>${item.nutrient}</strong><span>${item.priority} • ${item.target}</span></div>`).join("")}
      </div>
    </div>
    <div class="mini-panel" style="margin-top:16px;">
      <h3>Top crop candidates</h3>
      <div class="matrix">
        ${rankedCandidates
          .slice(0, 8)
          .map(
            (crop) =>
              `<div class="matrix-item"><strong>${crop.name}</strong><span>${Math.round(crop.ensembleScore * 100)}% fit</span></div>`,
          )
          .join("")}
      </div>
    </div>
  `;
}

async function renderCropSuitability(values) {
  const resultCard = document.getElementById("result-card");
  const cropName = values.crop;
  if (!cropName) {
    resultCard.innerHTML = `<h3>Crop suitability</h3><div class="placeholder">Select a crop to evaluate compatibility.</div>`;
    return;
  }

  const assessment =
    (await getSuitabilityAssessment(values)) ||
    buildClientSuitability(values, cropName);
  if (!assessment) {
    resultCard.innerHTML = `<h3>Crop suitability</h3><div class="placeholder">No profile found for ${cropName} in the selected region/season.</div>`;
    return;
  }

  const fertilizer = assessment.fertilizer || {};
  const contextNote = [values.locationLabel, values.seasonLabel]
    .filter(Boolean)
    .join(" • ");

  resultCard.innerHTML = `
    <h3>Crop suitability assessment</h3>
    ${contextNote ? `<p class="muted">${contextNote}</p>` : ""}
    <div class="result-banner">
      <h4>${assessment.crop}</h4>
      <p>${assessment.description || ""}</p>
      <div class="score-pill">${assessment.compatible ? "Compatible" : "Not recommended"} • ${assessment.suitability}</div>
      <div class="score-pill" style="margin-left:8px;">${assessment.overall_score}% fit • ${assessment.productivity_potential} productivity</div>
    </div>
    <div class="mini-panel" style="margin-top:16px;">
      <h3>Assessment summary</h3>
      <p>${assessment.summary}</p>
      <p><strong>Yield outlook:</strong> ${assessment.yield_outlook}</p>
      <p>${assessment.strategy || ""}</p>
    </div>
    <div class="metric-list" style="margin-top:16px;">
      ${(assessment.metrics || [])
        .map((metric) => {
          const percent = Math.round((metric.score || 0) * 100);
          return `
          <div class="metric-row">
            <div class="metric-meta">
              <span>${metric.label} <span class="status-badge ${statusClass(metric.status)}">${metric.status}</span></span>
              <strong>${percent}%</strong>
            </div>
            <div class="bar"><div style="width:${percent}%;"></div></div>
            <small class="muted">Current: ${metric.current_value} • Ideal: ${metric.ideal_range?.[0]}–${metric.ideal_range?.[1]}</small>
          </div>`;
        })
        .join("")}
    </div>
    ${
      (assessment.remediation || []).length
        ? `
      <div class="mini-panel" style="margin-top:16px;">
        <h3>Remediation suggestions</h3>
        <div class="matrix">
          ${assessment.remediation.map((item) => `<div class="matrix-item"><strong>${item.factor} (${item.priority})</strong><span>${item.action}. ${item.detail}</span></div>`).join("")}
        </div>
      </div>`
        : `
      <div class="mini-panel" style="margin-top:16px;">
        <h3>Remediation suggestions</h3>
        <p>All measured factors are within acceptable ranges for this crop in the selected context.</p>
      </div>`
    }
    ${
      fertilizer.recommendation
        ? `
      <div class="mini-panel" style="margin-top:16px;">
        <h3>Fertilizer guidance</h3>
        <p>${fertilizer.soil_note || ""}</p>
        <div class="matrix">
          ${fertilizer.recommendation.map((item) => `<div class="matrix-item"><strong>${item.nutrient}</strong><span>${item.priority} • ${item.target}</span></div>`).join("")}
        </div>
      </div>`
        : ""
    }
  `;
}

function getRecommendationValues(form) {
  const data = new FormData(form);
  const location = data.get("location") || "";
  const season = data.get("season") || "";
  const locationSelect = form.elements.location;
  const seasonSelect = form.elements.season;
  return {
    nitrogen: Number(data.get("nitrogen")),
    phosphorous: Number(data.get("phosphorous")),
    potassium: Number(data.get("potassium")),
    temperature: Number(data.get("temperature")),
    humidity: Number(data.get("humidity")),
    ph: Number(data.get("ph")),
    rainfall: Number(data.get("rainfall")),
    location,
    season,
    crop: data.get("target_crop") || "",
    locationLabel: locationSelect?.selectedOptions?.[0]?.textContent || "",
    seasonLabel: seasonSelect?.selectedOptions?.[0]?.textContent || "",
  };
}

function populateContextSelectors() {
  const locationSelect = document.getElementById("location-select");
  const seasonSelect = document.getElementById("season-select");
  const targetCropSelect = document.getElementById("target-crop-select");
  const filterCrop = document.getElementById("filter-crop");
  const filterLocation = document.getElementById("filter-location");
  const filterSeason = document.getElementById("filter-season");

  if (locationSelect)
    locationSelect.innerHTML = '<option value="">Any region</option>';
  if (seasonSelect)
    seasonSelect.innerHTML = '<option value="">Any season</option>';
  if (targetCropSelect) targetCropSelect.innerHTML = "";
  if (filterCrop) filterCrop.innerHTML = '<option value="">All crops</option>';
  if (filterLocation)
    filterLocation.innerHTML = '<option value="">All regions</option>';
  if (filterSeason)
    filterSeason.innerHTML = '<option value="">All seasons</option>';

  if (dataset?.locations) {
    dataset.locations.forEach((location) => {
      locationSelect?.insertAdjacentHTML(
        "beforeend",
        `<option value="${location.id}">${location.name}</option>`,
      );
      filterLocation?.insertAdjacentHTML(
        "beforeend",
        `<option value="${location.id}">${location.name}</option>`,
      );
    });
  }

  if (dataset?.seasons) {
    dataset.seasons.forEach((season) => {
      seasonSelect?.insertAdjacentHTML(
        "beforeend",
        `<option value="${season.id}">${season.name} (${season.months})</option>`,
      );
      filterSeason?.insertAdjacentHTML(
        "beforeend",
        `<option value="${season.id}">${season.name}</option>`,
      );
    });
  }

  const cropProfiles = getCropProfiles();
  if (cropProfiles.length) {
    cropProfiles.forEach((crop) => {
      targetCropSelect?.insertAdjacentHTML(
        "beforeend",
        `<option value="${crop.name}">${crop.name}</option>`,
      );
      filterCrop?.insertAdjacentHTML(
        "beforeend",
        `<option value="${crop.name}">${crop.name}</option>`,
      );
    });
  }
}

function initModeTabs() {
  const tabs = document.querySelectorAll(".mode-tab");
  const evaluateFields = document.querySelectorAll(".evaluate-only");
  const submitBtn = document.getElementById("submit-btn");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      recommendationMode = tab.dataset.mode;
      tabs.forEach((item) => item.classList.toggle("active", item === tab));
      evaluateFields.forEach((field) =>
        field.classList.toggle("hidden", recommendationMode !== "evaluate"),
      );
      if (submitBtn) {
        submitBtn.textContent =
          recommendationMode === "evaluate"
            ? "Evaluate crop suitability"
            : "Analyze field conditions";
      }
      const form = document.getElementById("crop-form");
      if (form) void renderRecommendation(getRecommendationValues(form));
    });
  });
}

function initRecommendationForm() {
  const form = document.getElementById("crop-form");
  const resultCard = document.getElementById("result-card");
  if (!form || !resultCard) return;

  populateContextSelectors();
  initModeTabs();
  initWeatherControls();

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const values = getRecommendationValues(form);
    await renderRecommendation(values);
  });

  const sampleBtn = document.getElementById("sample-btn");
  sampleBtn?.addEventListener("click", () => {
    form.elements.nitrogen.value = 110;
    form.elements.phosphorous.value = 55;
    form.elements.potassium.value = 70;
    form.elements.temperature.value = 26;
    form.elements.humidity.value = 72;
    form.elements.ph.value = 6.4;
    form.elements.rainfall.value = 95;
    form.elements.location.value = "";
    form.elements.season.value = "";
    form.elements.target_crop.value = "Rice";
  });
}

async function fetchLiveWeather(city, lat, lon) {
  const status = document.getElementById("weather-status");
  if (status) {
    status.textContent = "Fetching live weather...";
    status.className = "weather-status muted";
  }

  const params = new URLSearchParams();
  if (city) params.set("city", city);
  if (lat != null && lon != null) {
    params.set("lat", String(lat));
    params.set("lon", String(lon));
  }

  try {
    const response = await fetch(`/api/weather?${params.toString()}`);
    const payload = await response.json();
    if (!response.ok || payload.error)
      throw new Error(payload.error || "Weather request failed");

    const form = document.getElementById("crop-form");
    if (form) {
      form.elements.temperature.value = payload.temperature;
      form.elements.humidity.value = payload.humidity;
      form.elements.rainfall.value = payload.rainfall;
    }
    if (status) {
      status.textContent = `Loaded ${payload.location}: ${payload.temperature}°C, ${payload.humidity}% humidity, ${payload.rainfall} mm (${payload.rainfall_period}).`;
      status.className = "weather-status success";
    }
    return payload;
  } catch (error) {
    if (status) {
      status.textContent = `Weather unavailable: ${error.message}`;
      status.className = "weather-status error";
    }
    return null;
  }
}

function initWeatherControls() {
  const weatherBtn = document.getElementById("weather-btn");
  const gpsBtn = document.getElementById("gps-btn");
  const cityInput = document.getElementById("city-input");

  weatherBtn?.addEventListener("click", () => {
    const city = cityInput?.value?.trim();
    if (!city) {
      const status = document.getElementById("weather-status");
      if (status) {
        status.textContent =
          "Enter a city name or use GPS before fetching weather.";
        status.className = "weather-status error";
      }
      return;
    }
    void fetchLiveWeather(city);
  });

  gpsBtn?.addEventListener("click", () => {
    if (!navigator.geolocation) {
      const status = document.getElementById("weather-status");
      if (status) {
        status.textContent = "Geolocation is not supported in this browser.";
        status.className = "weather-status error";
      }
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (position) =>
        void fetchLiveWeather(
          null,
          position.coords.latitude,
          position.coords.longitude,
        ),
      (error) => {
        const status = document.getElementById("weather-status");
        if (status) {
          status.textContent = `GPS unavailable: ${error.message}`;
          status.className = "weather-status error";
        }
      },
      { enableHighAccuracy: false, timeout: 10000 },
    );
  });
}

async function getDiseasePrediction(payload) {
  try {
    const response = await fetch("/api/disease", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error(`Request failed with ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn("Disease model unavailable, using fallback scoring.", error);
    return null;
  }
}

function renderDiseasePrediction() {
  const form = document.getElementById("disease-form");
  const output = document.getElementById("disease-output");
  const symptomList = document.getElementById("symptom-list");
  const imageInput = document.getElementById("image-upload");
  const imageButton = document.getElementById("image-submit");
  if (!form || !output || !symptomList) return;

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    output.innerHTML = `<h3>Detection output</h3><div class="loading-overlay"><div class="spinner"></div>Analyzing symptoms and environment...</div>`;
    const data = new FormData(form);
    const crop = data.get("crop");
    const humidity = Number(data.get("humidity"));
    const temperature = Number(data.get("temperature"));
    const ph = Number(data.get("ph"));
    const selectedSymptoms = [];
    symptomList
      .querySelectorAll(".check-chip.active")
      .forEach((chip) => selectedSymptoms.push(chip.dataset.symptom));

    const fallback = () => {
      const normalizedCrop = (crop || "").toString().trim();
      const filtered = getDiseaseProfiles().filter(
        (profile) => profile.crop === normalizedCrop,
      );
      const fallbackProfiles = filtered.length
        ? filtered
        : getDiseaseProfiles();
      const scored = fallbackProfiles.map((profile) => {
        let score = 0;
        profile.symptoms.forEach((symptom) => {
          if (selectedSymptoms.includes(symptom)) score += 1.2;
        });
        const humidityMatch = Math.max(
          0,
          1 - Math.abs(humidity - profile.humidity) / 30,
        );
        const temperatureMatch = Math.max(
          0,
          1 - Math.abs(temperature - profile.temperature) / 20,
        );
        const phMatch = Math.max(0, 1 - Math.abs(ph - profile.ph) / 2.5);
        score += humidityMatch + temperatureMatch + phMatch;
        return { ...profile, score };
      });

      scored.sort((a, b) => b.score - a.score);
      const best = scored[0];
      const confidence = Math.min(96, Math.round(55 + best.score * 10));
      return { best, confidence };
    };

    const payload = {
      crop,
      temperature,
      humidity,
      ph,
      symptoms: selectedSymptoms,
    };
    const result = await getDiseasePrediction(payload);
    const fallbackResult = fallback();
    const shouldUseFallback =
      Boolean(result) &&
      crop &&
      crop !== "Rice" &&
      result.predicted_disease === "Rice Blast";
    const best =
      !shouldUseFallback && result
        ? {
            disease: result.predicted_disease,
            description: `${result.risk_level} risk under current conditions`,
            treatment:
              "Prioritize early intervention and monitor field spread.",
            severity: result.risk_level,
          }
        : fallbackResult.best;
    const confidence =
      !shouldUseFallback && result
        ? result.confidence
        : fallbackResult.confidence;
    output.innerHTML = `
      <div class="result-banner">
        <h4>${best.disease}</h4>
        <p>${best.description}</p>
        <div class="score-pill">Confidence ${confidence}% • ${best.severity || "Medium"} severity</div>
      </div>
      <div class="mini-panel" style="margin-top:16px;">
        <h3>Recommended response</h3>
        <p>${best.treatment}</p>
      </div>
      <div class="mini-panel" style="margin-top:16px;">
        <h3>Decision support</h3>
        <p>The disease output combines environmental context and symptom matches to rank likely stresses for field action.</p>
      </div>
    `;
  });

  imageButton?.addEventListener("click", async () => {
    if (!imageInput?.files?.length) {
      output.innerHTML = `<div class="placeholder">Please upload a leaf image first.</div>`;
      return;
    }
    const file = imageInput.files[0];
    const reader = new FileReader();
    reader.onload = async () => {
      const base64 = reader.result.split(",")[1];
      const selectedCrop =
        document.querySelector('#disease-form select[name="crop"]')?.value ||
        "Rice";
      output.innerHTML = `<div class="placeholder">Analyzing image...</div>`;
      try {
        const response = await fetch(
          `/api/image-disease?crop=${encodeURIComponent(selectedCrop)}`,
          {
            method: "POST",
            headers: { "Content-Type": "application/octet-stream" },
            body: Uint8Array.from(atob(base64), (c) => c.charCodeAt(0)),
          },
        );
        if (!response.ok)
          throw new Error(`Request failed with ${response.status}`);
        const result = await response.json();
        output.innerHTML = `
          <div class="result-banner">
            <h4>${result.predicted_disease}</h4>
            <p>Image-based disease inference from leaf appearance.</p>
            <div class="score-pill">Confidence ${result.confidence}%</div>
          </div>
          <div class="mini-panel" style="margin-top:16px;">
            <h3>Next step</h3>
            <p>Use this result alongside field symptoms and soil conditions for intervention planning.</p>
          </div>
        `;
      } catch (error) {
        output.innerHTML = `<div class="placeholder">Image analysis is not available yet. Please use symptom-based detection.</div>`;
      }
    };
    reader.readAsDataURL(file);
  });
}

function initSymptoms() {
  const symptomList = document.getElementById("symptom-list");
  if (!symptomList) return;
  const symptoms = [
    "leaf spots",
    "neck blast",
    "gray centers",
    "high humidity",
    "brown lesions",
    "yellowing",
    "orange pustules",
    "leaf discoloration",
    "dark patches",
    "water-soaked spots",
    "concentric rings",
    "leaf collapse",
    "leaf browning",
    "small spots",
    "black streaks",
    "leaf necrosis",
  ];
  symptomList.innerHTML = symptoms
    .map(
      (symptom) =>
        `<button type="button" class="check-chip" data-symptom="${symptom}">${symptom}</button>`,
    )
    .join("");
  symptomList.querySelectorAll(".check-chip").forEach((chip) => {
    chip.addEventListener("click", () => chip.classList.toggle("active"));
  });
}

function destroyChart(id) {
  if (chartInstances[id]) {
    chartInstances[id].destroy();
    delete chartInstances[id];
  }
}

function getFilteredScatterPoints() {
  const crop = document.getElementById("filter-crop")?.value || "";
  const location = document.getElementById("filter-location")?.value || "";
  const season = document.getElementById("filter-season")?.value || "";
  return (analyticsData?.scatter_points || []).filter((point) => {
    if (crop && point.crop !== crop) return false;
    if (location && point.location !== location) return false;
    if (season && point.season !== season) return false;
    return true;
  });
}

function getAnalyticsFilterValues() {
  return {
    crop: document.getElementById("filter-crop")?.value || "",
    location: document.getElementById("filter-location")?.value || "",
    season: document.getElementById("filter-season")?.value || "",
  };
}

function renderAnalyticsStats(points) {
  const statsPanel = document.getElementById("analytics-stats");
  if (!statsPanel) return;
  const cropCount = new Set(points.map((point) => point.crop)).size;
  const avgTemp = points.length
    ? (
        points.reduce((sum, point) => sum + point.temperature, 0) /
        points.length
      ).toFixed(1)
    : "—";
  const avgRain = points.length
    ? (
        points.reduce((sum, point) => sum + point.rainfall, 0) / points.length
      ).toFixed(1)
    : "—";
  statsPanel.innerHTML = `
    <div class="card"><h3>Filtered records</h3><strong style="font-size:1.8rem;">${points.length}</strong><p class="muted">Matching current dashboard filters</p></div>
    <div class="card"><h3>Crops represented</h3><strong style="font-size:1.8rem;">${cropCount}</strong><p class="muted">Unique crop types in view</p></div>
    <div class="card"><h3>Avg climate</h3><strong style="font-size:1.8rem;">${avgTemp}°C / ${avgRain} mm</strong><p class="muted">Mean temperature and rainfall</p></div>
  `;
}

function renderOverviewTables(points) {
  const locationTable = document.getElementById("location-table");
  const seasonTable = document.getElementById("season-table");
  if (!locationTable || !seasonTable || !analyticsData) return;

  const locationMap = {};
  points.forEach((point) => {
    locationMap[point.location] = locationMap[point.location] || {
      count: 0,
      rainfall: 0,
      temp: 0,
    };
    locationMap[point.location].count += 1;
    locationMap[point.location].rainfall += point.rainfall;
    locationMap[point.location].temp += point.temperature;
  });

  const seasonMap = {};
  points.forEach((point) => {
    seasonMap[point.season] = seasonMap[point.season] || {
      count: 0,
      rainfall: 0,
      temp: 0,
    };
    seasonMap[point.season].count += 1;
    seasonMap[point.season].rainfall += point.rainfall;
    seasonMap[point.season].temp += point.temperature;
  });

  const locationNames = Object.fromEntries(
    (analyticsData.locations || []).map((entry) => [entry.id, entry.name]),
  );
  const seasonNames = Object.fromEntries(
    (analyticsData.seasons || []).map((entry) => [entry.id, entry.name]),
  );

  locationTable.innerHTML =
    Object.entries(locationMap)
      .map(
        ([id, stats]) => `
    <div class="matrix-item">
      <strong>${locationNames[id] || id}</strong>
      <span>${stats.count} records • ${(stats.temp / stats.count).toFixed(1)}°C • ${(stats.rainfall / stats.count).toFixed(0)} mm rain</span>
    </div>`,
      )
      .join("") ||
    `<div class="placeholder">No regional records for current filters.</div>`;

  seasonTable.innerHTML =
    Object.entries(seasonMap)
      .map(
        ([id, stats]) => `
    <div class="matrix-item">
      <strong>${seasonNames[id] || id}</strong>
      <span>${stats.count} records • ${(stats.temp / stats.count).toFixed(1)}°C • ${(stats.rainfall / stats.count).toFixed(0)} mm rain</span>
    </div>`,
      )
      .join("") ||
    `<div class="placeholder">No seasonal records for current filters.</div>`;
}

function renderAnalyticsCharts(points) {
  const cropGroups = {};
  points.forEach((point) => {
    cropGroups[point.crop] = cropGroups[point.crop] || {
      n: 0,
      p: 0,
      k: 0,
      count: 0,
    };
    cropGroups[point.crop].n += point.nitrogen;
    cropGroups[point.crop].p += point.phosphorous;
    cropGroups[point.crop].k += point.potassium;
    cropGroups[point.crop].count += 1;
  });

  const cropLabels = Object.keys(cropGroups).sort();
  const avgN = cropLabels.map((crop) =>
    Number((cropGroups[crop].n / cropGroups[crop].count).toFixed(1)),
  );
  const avgP = cropLabels.map((crop) =>
    Number((cropGroups[crop].p / cropGroups[crop].count).toFixed(1)),
  );
  const avgK = cropLabels.map((crop) =>
    Number((cropGroups[crop].k / cropGroups[crop].count).toFixed(1)),
  );
  const cropCounts = cropLabels.map((crop) => cropGroups[crop].count);

  destroyChart("npk");
  destroyChart("cropDist");
  destroyChart("climate");
  destroyChart("rainfall");

  const npkCtx = document.getElementById("npk-chart");
  if (npkCtx) {
    chartInstances.npk = new Chart(npkCtx, {
      type: "bar",
      data: {
        labels: cropLabels,
        datasets: [
          {
            label: "N",
            data: avgN,
            backgroundColor: "rgba(60, 207, 122, 0.75)",
          },
          {
            label: "P",
            data: avgP,
            backgroundColor: "rgba(125, 220, 95, 0.65)",
          },
          {
            label: "K",
            data: avgK,
            backgroundColor: "rgba(212, 165, 116, 0.75)",
          },
        ],
      },
      options: {
        responsive: true,
        plugins: { legend: { labels: { color: "#d7e8da" } } },
        scales: {
          x: {
            ticks: { color: "#9db4a2" },
            grid: { color: "rgba(255,255,255,0.05)" },
          },
          y: {
            ticks: { color: "#9db4a2" },
            grid: { color: "rgba(255,255,255,0.05)" },
          },
        },
      },
    });
  }

  const distCtx = document.getElementById("crop-dist-chart");
  if (distCtx) {
    chartInstances.cropDist = new Chart(distCtx, {
      type: "doughnut",
      data: {
        labels: cropLabels,
        datasets: [
          {
            data: cropCounts,
            backgroundColor: cropLabels.map(
              (crop) => cropColors[crop] || "#7ddc5f",
            ),
          },
        ],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { position: "bottom", labels: { color: "#d7e8da" } },
        },
      },
    });
  }

  const climateCtx = document.getElementById("climate-chart");
  if (climateCtx) {
    const cropsInView = [...new Set(points.map((point) => point.crop))];
    chartInstances.climate = new Chart(climateCtx, {
      type: "scatter",
      data: {
        datasets: cropsInView.map((crop) => ({
          label: crop,
          data: points
            .filter((point) => point.crop === crop)
            .map((point) => ({ x: point.temperature, y: point.humidity })),
          backgroundColor: cropColors[crop] || "#7ddc5f",
        })),
      },
      options: {
        responsive: true,
        plugins: { legend: { labels: { color: "#d7e8da" } } },
        scales: {
          x: {
            title: {
              display: true,
              text: "Temperature (°C)",
              color: "#9db4a2",
            },
            ticks: { color: "#9db4a2" },
            grid: { color: "rgba(255,255,255,0.05)" },
          },
          y: {
            title: { display: true, text: "Humidity (%)", color: "#9db4a2" },
            ticks: { color: "#9db4a2" },
            grid: { color: "rgba(255,255,255,0.05)" },
          },
        },
      },
    });
  }

  const rainfallCtx = document.getElementById("rainfall-chart");
  if (rainfallCtx && analyticsData?.by_location) {
    const locationNames = Object.fromEntries(
      (analyticsData.locations || []).map((entry) => [entry.id, entry.name]),
    );
    const locationIds = [...new Set(points.map((point) => point.location))];
    const rainfallByLocation = locationIds.map((id) => {
      const rows = points.filter((point) => point.location === id);
      return rows.length
        ? Number(
            (
              rows.reduce((sum, row) => sum + row.rainfall, 0) / rows.length
            ).toFixed(1),
          )
        : 0;
    });
    chartInstances.rainfall = new Chart(rainfallCtx, {
      type: "bar",
      data: {
        labels: locationIds.map((id) => locationNames[id] || id),
        datasets: [
          {
            label: "Avg rainfall (mm)",
            data: rainfallByLocation,
            backgroundColor: "rgba(60, 207, 122, 0.75)",
          },
        ],
      },
      options: {
        responsive: true,
        plugins: { legend: { labels: { color: "#d7e8da" } } },
        scales: {
          x: {
            ticks: { color: "#9db4a2" },
            grid: { color: "rgba(255,255,255,0.05)" },
          },
          y: {
            ticks: { color: "#9db4a2" },
            grid: { color: "rgba(255,255,255,0.05)" },
          },
        },
      },
    });
  }
}

function exportFilteredCsv() {
  const points = getFilteredScatterPoints();
  if (!points.length) return;
  const headers = [
    "crop",
    "location",
    "season",
    "nitrogen",
    "phosphorous",
    "potassium",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
    "yield",
  ];
  const rows = points.map((point) =>
    headers.map((key) => point[key] ?? "").join(","),
  );
  const blob = new Blob([[headers.join(","), ...rows].join("\n")], {
    type: "text/csv;charset=utf-8;",
  });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "opticrop-analytics.csv";
  link.click();
  URL.revokeObjectURL(link.href);
}

function initInsightCards(points) {
  const insightPanel = document.getElementById("insight-panel");
  if (!insightPanel) return;
  const highYield = points.filter(
    (point) => point.yield === "Very High" || point.yield === "High",
  );
  const avgN = points.length
    ? (
        points.reduce((sum, point) => sum + point.nitrogen, 0) / points.length
      ).toFixed(0)
    : "—";
  insightPanel.innerHTML = [
    {
      title: "Climate resilience",
      detail: `${highYield.length} high-yield records appear in the current filtered view, useful for regional planning.`,
    },
    {
      title: "Input efficiency",
      detail: `Average nitrogen in view is ${avgN} units; compare this against crop-specific NPK charts above.`,
    },
    {
      title: "Regional context",
      detail:
        "Use region and season filters to study where specific crops cluster in temperature and rainfall space.",
    },
    {
      title: "Policy value",
      detail:
        "Export filtered CSV snapshots for extension services, research papers, or policy briefs.",
    },
  ]
    .map(
      (item) =>
        `<div class="tag-item"><strong>${item.title}</strong><span>${item.detail}</span></div>`,
    )
    .join("");
}

function refreshAnalyticsDashboard() {
  const points = getFilteredScatterPoints();
  renderAnalyticsStats(points);
  renderAnalyticsCharts(points);
  renderOverviewTables(points);
  initInsightCards(points);
}

async function initInsightsPage() {
  await loadDataset();
  populateContextSelectors();

  try {
    const response = await fetch("/api/analytics");
    analyticsData = await response.json();
  } catch (error) {
    analyticsData = {
      locations: dataset?.locations || [],
      seasons: dataset?.seasons || [],
      scatter_points: (dataset?.crops || []).map((entry) => ({
        crop: entry.crop,
        location: entry.location,
        season: entry.season,
        temperature: entry.temperature,
        humidity: entry.humidity,
        rainfall: entry.rainfall,
        nitrogen: entry.nitrogen,
        phosphorous: entry.phosphorous,
        potassium: entry.potassium,
        ph: entry.ph,
        yield: entry.yield,
      })),
    };
  }

  ["filter-crop", "filter-location", "filter-season"].forEach((id) => {
    document
      .getElementById(id)
      ?.addEventListener("change", refreshAnalyticsDashboard);
  });

  document.getElementById("reset-filters")?.addEventListener("click", () => {
    ["filter-crop", "filter-location", "filter-season"].forEach((id) => {
      const element = document.getElementById(id);
      if (element) element.value = "";
    });
    refreshAnalyticsDashboard();
  });

  document
    .getElementById("export-csv")
    ?.addEventListener("click", exportFilteredCsv);
  refreshAnalyticsDashboard();
}

async function initRecommendationPage() {
  const form = document.getElementById("crop-form");
  const sampleBtn = document.getElementById("sample-btn");
  if (!form) return;
  await loadDataset();
  populateContextSelectors();
  initModeTabs();
  initWeatherControls();
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    void renderRecommendation(getRecommendationValues(form));
  });
  sampleBtn?.addEventListener("click", () => {
    form.elements.nitrogen.value = 110;
    form.elements.phosphorous.value = 55;
    form.elements.potassium.value = 70;
    form.elements.temperature.value = 26;
    form.elements.humidity.value = 72;
    form.elements.ph.value = 6.4;
    form.elements.rainfall.value = 95;
    form.elements.location.value = "";
    form.elements.season.value = "";
    form.elements.target_crop.value = "Rice";
    void renderRecommendation(getRecommendationValues(form));
  });
  void renderRecommendation(getRecommendationValues(form));
}

async function initDiseasePage() {
  await loadDataset();
  initSymptoms();
  renderDiseasePrediction();
}

function initGlobalLayout() {
  const menuToggle = document.getElementById("menu-toggle");
  const navLinks = document.getElementById("nav-links");
  if (menuToggle && navLinks) {
    menuToggle.addEventListener("click", () => {
      menuToggle.classList.toggle("open");
      navLinks.classList.toggle("show");
    });
  }

  // Active link highlighting
  const currentPath = window.location.pathname.split("/").pop() || "index.html";
  const links = document.querySelectorAll(".nav-links a");
  links.forEach((link) => {
    const linkPath = link.getAttribute("href");
    if (linkPath === currentPath) {
      link.classList.add("active");
    } else {
      link.classList.remove("active");
    }
  });
}

document.addEventListener("DOMContentLoaded", initGlobalLayout);

if (document.body.dataset.page === "recommend") {
  initRecommendationPage();
}

if (document.body.dataset.page === "disease") {
  initDiseasePage();
}

if (document.body.dataset.page === "insights") {
  initInsightsPage();
}

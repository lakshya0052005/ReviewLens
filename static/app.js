/**
 * ReviewLens Frontend Application Logic
 * Pure Vanilla JavaScript (Zero build step, zero dependencies)
 */

document.addEventListener("DOMContentLoaded", () => {
  // Application State
  let currentJobId = null;
  let allReviews = [];
  let insightsData = null;

  // DOM Elements
  const dropArea = document.getElementById("dropArea");
  const fileInput = document.getElementById("fileInput");
  const fileNameDisplay = document.getElementById("fileNameDisplay");
  const uploadBtn = document.getElementById("uploadBtn");
  const uploadForm = document.getElementById("uploadForm");

  // Progress Elements
  const progressContainer = document.getElementById("progressContainer");
  const progressBarFill = document.getElementById("progressBarFill");
  const progressPercent = document.getElementById("progressPercent");
  const progressStatusText = document.getElementById("progressStatusText");
  const progressCounts = document.getElementById("progressCounts");
  const statCalls = document.getElementById("statCalls");
  const statHits = document.getElementById("statHits");

  // Dashboard Sections
  const dashboardContent = document.getElementById("dashboardContent");

  // Metrics Banner
  const metricLlmCalls = document.getElementById("metricLlmCalls");
  const metricCacheHits = document.getElementById("metricCacheHits");
  const metricDuration = document.getElementById("metricDuration");

  // KPI Elements
  const kpiTotalReviews = document.getElementById("kpiTotalReviews");
  const kpiAnalyzedSub = document.getElementById("kpiAnalyzedSub");
  const kpiPositivePct = document.getElementById("kpiPositivePct");
  const kpiPositiveCount = document.getElementById("kpiPositiveCount");
  const kpiNeutralPct = document.getElementById("kpiNeutralPct");
  const kpiNeutralCount = document.getElementById("kpiNeutralCount");
  const kpiNegativePct = document.getElementById("kpiNegativePct");
  const kpiNegativeCount = document.getElementById("kpiNegativeCount");
  const kpiNeedsReply = document.getElementById("kpiNeedsReply");
  const kpiRatingAgreement = document.getElementById("kpiRatingAgreement");
  const kpiMismatches = document.getElementById("kpiMismatches");

  // Top Lists
  const topPraisesList = document.getElementById("topPraisesList");
  const topComplaintsList = document.getElementById("topComplaintsList");
  const productBreakdownBody = document.getElementById("productBreakdownBody");

  // Filter Controls
  const filterProduct = document.getElementById("filterProduct");
  const filterSentiment = document.getElementById("filterSentiment");
  const filterTopic = document.getElementById("filterTopic");
  const filterSpecial = document.getElementById("filterSpecial");
  const resetFiltersBtn = document.getElementById("resetFiltersBtn");
  const reviewsTableBody = document.getElementById("reviewsTableBody");
  const tableRowCount = document.getElementById("tableRowCount");

  // -------------------------------------------------------------------------
  // 1. File Selection & Drag-and-Drop Handling
  // -------------------------------------------------------------------------
  function handleFileSelected(file) {
    if (file && file.name.endsWith(".csv")) {
      fileNameDisplay.textContent = `Selected: ${file.name} (${Math.round(file.size / 1024)} KB)`;
      uploadBtn.disabled = false;
    } else {
      fileNameDisplay.textContent = "Please select a valid .csv file";
      uploadBtn.disabled = true;
    }
  }

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    dropArea.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropArea.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropArea.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropArea.classList.remove("dragover");
    });
  });

  dropArea.addEventListener("drop", (e) => {
    if (e.dataTransfer.files.length > 0) {
      fileInput.files = e.dataTransfer.files;
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  // -------------------------------------------------------------------------
  // 2. Form Submission & Polling
  // -------------------------------------------------------------------------
  uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!fileInput.files.length) return;

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("file", file);

    uploadBtn.disabled = true;
    uploadBtn.textContent = "Uploading...";
    progressContainer.classList.remove("hidden");
    dashboardContent.classList.add("hidden");

    try {
      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Upload failed");
      }

      const data = await response.json();
      currentJobId = data.job_id;
      startPollingProgress(currentJobId);
    } catch (err) {
      alert("Error: " + err.message);
      uploadBtn.disabled = false;
      uploadBtn.textContent = "Analyze Reviews";
    }
  });

  function startPollingProgress(jobId) {
    progressStatusText.textContent = "Analyzing reviews with Gemini in batches...";

    const pollInterval = setInterval(async () => {
      try {
        const res = await fetch(`/api/jobs/${jobId}`);
        if (!res.ok) return;

        const job = await res.json();
        const pct = job.progress_percentage || 0;

        progressBarFill.style.width = `${pct}%`;
        progressPercent.textContent = `${pct}%`;
        progressCounts.textContent = `${job.processed_reviews} / ${job.total_reviews}`;
        statCalls.textContent = job.llm_calls;
        statHits.textContent = job.cache_hits;

        if (job.status === "completed") {
          clearInterval(pollInterval);
          progressStatusText.textContent = "Analysis complete!";
          uploadBtn.textContent = "Analyze Reviews";
          uploadBtn.disabled = false;
          await loadDashboardData(jobId);
        } else if (job.status === "failed") {
          clearInterval(pollInterval);
          progressStatusText.textContent = "Analysis failed: " + (job.error_message || "Unknown error");
          uploadBtn.textContent = "Analyze Reviews";
          uploadBtn.disabled = false;
        }
      } catch (err) {
        console.error("Polling error:", err);
      }
    }, 800);
  }

  // -------------------------------------------------------------------------
  // 3. Load & Render Dashboard Data
  // -------------------------------------------------------------------------
  async function loadDashboardData(jobId) {
    try {
      const [resultsRes, insightsRes] = await Promise.all([
        fetch(`/api/jobs/${jobId}/results`),
        fetch(`/api/jobs/${jobId}/insights`),
      ]);

      const resultsData = await resultsRes.json();
      insightsData = await insightsRes.json();
      allReviews = resultsData.reviews || [];

      renderMetricsBanner(insightsData.metrics);
      renderKPIs(insightsData.summary);
      renderTopInsights(insightsData.top_praises, insightsData.top_complaints);
      renderProductBreakdown(insightsData.per_product);
      populateProductFilter(allReviews);
      renderReviewsTable(allReviews);

      dashboardContent.classList.remove("hidden");
    } catch (err) {
      console.error("Error loading dashboard data:", err);
      alert("Could not load dashboard data: " + err.message);
    }
  }

  function renderMetricsBanner(metrics) {
    metricLlmCalls.textContent = metrics.llm_calls ?? 0;
    metricCacheHits.textContent = metrics.cache_hits ?? 0;
    metricDuration.textContent = metrics.duration_seconds ? `${metrics.duration_seconds}s` : "N/A";
  }

  function renderKPIs(summary) {
    kpiTotalReviews.textContent = summary.total_reviews;
    kpiAnalyzedSub.textContent = `${summary.analyzed_count} analyzed (${summary.skipped_empty_count} skipped, ${summary.duplicates_count} duplicates)`;

    kpiPositivePct.textContent = `${summary.positive_pct}%`;
    kpiPositiveCount.textContent = `${summary.positive_count} reviews`;

    kpiNeutralPct.textContent = `${summary.neutral_pct}%`;
    kpiNeutralCount.textContent = `${summary.neutral_count} reviews`;

    kpiNegativePct.textContent = `${summary.negative_pct}%`;
    kpiNegativeCount.textContent = `${summary.negative_count} reviews`;

    kpiNeedsReply.textContent = summary.needs_reply_count;
    kpiRatingAgreement.textContent = `${summary.rating_agreement_pct}%`;
    kpiMismatches.textContent = `${summary.mismatch_count} star mismatches`;
  }

  function renderTopInsights(praises, complaints) {
    topPraisesList.innerHTML = (praises || [])
      .map((p) => `<li>${escapeHtml(p)}</li>`)
      .join("");

    topComplaintsList.innerHTML = (complaints || [])
      .map((c) => `<li>${escapeHtml(c)}</li>`)
      .join("");
  }

  function renderProductBreakdown(products) {
    if (!products || !products.length) {
      productBreakdownBody.innerHTML = `<tr><td colspan="6" style="text-align:center;">No product data</td></tr>`;
      return;
    }

    productBreakdownBody.innerHTML = products
      .map(
        (p) => `
        <tr>
          <td><strong>${escapeHtml(p.product)}</strong></td>
          <td>${p.total_reviews}</td>
          <td><span class="badge ${p.positive_pct >= 60 ? "badge-positive" : "badge-neutral"}">${p.positive_pct}%</span></td>
          <td><span class="badge badge-topic">${escapeHtml(p.top_praise_topic)}</span></td>
          <td><span class="badge ${p.top_complaint_topic !== 'None' ? 'badge-negative' : 'badge-neutral'}">${escapeHtml(p.top_complaint_topic)}</span></td>
          <td>${p.needs_reply_count > 0 ? `<span class="badge-flag flag-reply">${p.needs_reply_count} reply</span>` : "0"}</td>
        </tr>
      `
      )
      .join("");
  }

  function populateProductFilter(reviews) {
    const products = Array.from(new Set(reviews.map((r) => r.product).filter(Boolean))).sort();
    filterProduct.innerHTML = `<option value="ALL">All Products (${products.length})</option>`;
    products.forEach((prod) => {
      const opt = document.createElement("option");
      opt.value = prod;
      opt.textContent = prod;
      filterProduct.appendChild(opt);
    });
  }

  // -------------------------------------------------------------------------
  // 4. Reviews Table Rendering & Filtering
  // -------------------------------------------------------------------------
  function renderReviewsTable(reviews) {
    tableRowCount.textContent = `Showing ${reviews.length} reviews`;

    if (!reviews.length) {
      reviewsTableBody.innerHTML = `<tr><td colspan="8" style="text-align:center; padding: 24px; color: #64748b;">No reviews match the current filters.</td></tr>`;
      return;
    }

    reviewsTableBody.innerHTML = reviews
      .map((r) => {
        // Star rating formatting
        let ratingHtml = `<span style="color:#94a3b8;">Unknown</span>`;
        if (r.rating !== null && r.rating !== undefined) {
          ratingHtml = `<span style="color:#f59e0b; font-weight:600;">${"★".repeat(r.rating)}</span> <small>(${r.rating})</small>`;
        }

        // Sentiment badge
        let sentimentHtml = `<span class="badge badge-neutral">—</span>`;
        if (r.sentiment === "positive") {
          sentimentHtml = `<span class="badge badge-positive">Positive</span>`;
        } else if (r.sentiment === "neutral") {
          sentimentHtml = `<span class="badge badge-neutral">Neutral</span>`;
        } else if (r.sentiment === "negative") {
          sentimentHtml = `<span class="badge badge-negative">Negative</span>`;
        }

        // Topics pills
        const topicsHtml = (r.topics || [])
          .map((t) => `<span class="badge badge-topic">${escapeHtml(t)}</span>`)
          .join("");

        // Status flags
        const flags = [];
        if (r.clean_status.startsWith("duplicate_of:")) {
          flags.push(`<span class="badge-flag flag-duplicate">Duplicate</span>`);
        } else if (r.clean_status === "skipped_empty") {
          flags.push(`<span class="badge-flag flag-skipped">Skipped Empty</span>`);
        } else if (r.clean_status === "analysis_failed") {
          flags.push(`<span class="badge-flag flag-failed">Analysis Failed</span>`);
        }

        if (r.needs_reply) {
          flags.push(`<span class="badge-flag flag-reply">Needs Reply</span>`);
        }
        if (r.mismatch_flag) {
          flags.push(`<span class="badge-flag flag-mismatch" title="Rating disagrees with sentiment">★ Mismatch</span>`);
        }
        if (r.injection_suspected) {
          flags.push(`<span class="badge-flag flag-injection" title="Adversarial text detected">🛡️ Injection</span>`);
        }

        // Action button for Suggested Reply
        let actionHtml = "—";
        if (r.needs_reply) {
          if (r.suggested_reply) {
            actionHtml = `<span style="font-size:11px; color:#10b981; font-weight:600;">✓ Replied</span>`;
          } else {
            actionHtml = `<button class="btn btn-primary btn-sm suggest-btn" data-review-id="${r.review_id}">Suggest reply</button>`;
          }
        }

        // Suggested reply block if exists
        let replyBlockHtml = "";
        if (r.suggested_reply) {
          replyBlockHtml = `
            <div class="suggested-reply-container" id="reply-container-${r.review_id}">
              <div class="reply-header">💬 Suggested Merchant Reply:</div>
              <div class="reply-text">"${escapeHtml(r.suggested_reply)}"</div>
            </div>
          `;
        } else {
          replyBlockHtml = `<div class="suggested-reply-container hidden" id="reply-container-${r.review_id}"></div>`;
        }

        return `
          <tr id="row-${r.review_id}">
            <td><strong>${escapeHtml(r.review_id)}</strong></td>
            <td>${escapeHtml(r.product)}</td>
            <td>${ratingHtml}</td>
            <td>${sentimentHtml}</td>
            <td>${topicsHtml || "—"}</td>
            <td>
              <div class="review-summary-text">${escapeHtml(r.summary || "No summary")}</div>
              <div class="review-raw-excerpt">${escapeHtml(r.cleaned_text || r.raw_text || "(empty)")}</div>
              ${replyBlockHtml}
            </td>
            <td>${flags.join("") || "—"}</td>
            <td id="action-cell-${r.review_id}">${actionHtml}</td>
          </tr>
        `;
      })
      .join("");

    attachReplyButtonListeners();
  }

  // -------------------------------------------------------------------------
  // 5. On-Click "Suggest Reply" Handler
  // -------------------------------------------------------------------------
  function attachReplyButtonListeners() {
    document.querySelectorAll(".suggest-btn").forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        const reviewId = e.target.getAttribute("data-review-id");
        if (!reviewId || !currentJobId) return;

        e.target.disabled = true;
        e.target.textContent = "Drafting...";

        try {
          const res = await fetch(`/api/reply/${currentJobId}/${reviewId}`, {
            method: "POST",
          });
          if (!res.ok) {
            throw new Error("Failed to generate suggested reply");
          }
          const data = await res.json();

          // Update local state
          const targetReview = allReviews.find((r) => r.review_id === reviewId);
          if (targetReview) {
            targetReview.suggested_reply = data.suggested_reply;
          }

          // Render into reply container under the summary
          const replyBox = document.getElementById(`reply-container-${reviewId}`);
          if (replyBox) {
            replyBox.innerHTML = `
              <div class="reply-header">💬 Suggested Merchant Reply:</div>
              <div class="reply-text">"${escapeHtml(data.suggested_reply)}"</div>
            `;
            replyBox.classList.remove("hidden");
          }

          // Update action cell
          const actionCell = document.getElementById(`action-cell-${reviewId}`);
          if (actionCell) {
            actionCell.innerHTML = `<span style="font-size:11px; color:#10b981; font-weight:600;">✓ Replied</span>`;
          }
        } catch (err) {
          alert("Error: " + err.message);
          e.target.disabled = false;
          e.target.textContent = "Suggest reply";
        }
      });
    });
  }

  // -------------------------------------------------------------------------
  // 6. Interactive Filtering
  // -------------------------------------------------------------------------
  function applyFilters() {
    const prodVal = filterProduct.value;
    const sentVal = filterSentiment.value;
    const topicVal = filterTopic.value;
    const specVal = filterSpecial.value;

    const filtered = allReviews.filter((r) => {
      if (prodVal !== "ALL" && r.product !== prodVal) return false;
      if (sentVal !== "ALL" && r.sentiment !== sentVal) return false;
      if (topicVal !== "ALL" && !(r.topics || []).includes(topicVal)) return false;

      if (specVal === "needs_reply" && !r.needs_reply) return false;
      if (specVal === "mismatch" && !r.mismatch_flag) return false;
      if (specVal === "injection" && !r.injection_suspected) return false;
      if (specVal === "skipped" && r.clean_status === "ok") return false;

      return true;
    });

    renderReviewsTable(filtered);
  }

  [filterProduct, filterSentiment, filterTopic, filterSpecial].forEach((select) => {
    select.addEventListener("change", applyFilters);
  });

  resetFiltersBtn.addEventListener("click", () => {
    filterProduct.value = "ALL";
    filterSentiment.value = "ALL";
    filterTopic.value = "ALL";
    filterSpecial.value = "ALL";
    applyFilters();
  });

  // Utility to escape HTML to prevent XSS
  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});

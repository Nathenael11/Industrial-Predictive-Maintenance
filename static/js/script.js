document.addEventListener('DOMContentLoaded', () => {
    // --- SINGLE PREDICTION FORM LOGIC ---
    const predForm = document.getElementById('prediction-form');
    if (predForm) {
        predForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const btnSubmit = document.getElementById('btn-submit');
            const btnText = btnSubmit.querySelector('.btn-text');
            const spinner = btnSubmit.querySelector('.spinner');

            if (btnText) btnText.textContent = 'Analyzing...';
            btnSubmit.disabled = true;

            const formData = new FormData(predForm);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (response.ok) {
                    updateDiagnosticUI(data);
                    addHistoryRow(data);
                } else {
                    alert(data.error || 'Prediction failed');
                }
            } catch (err) {
                console.error('Error:', err);
                alert('Connection error during prediction submission');
            } finally {
                if (btnText) btnText.textContent = 'Predict Failure Risk';
                btnSubmit.disabled = false;
            }
        });

        const btnReset = document.getElementById('btn-reset');
        if (btnReset) {
            btnReset.addEventListener('click', () => {
                predForm.reset();
                resetGaugeUI();
            });
        }
    }

    // --- UI GAUGE & RESULT UPDATER ---
    function updateDiagnosticUI(data) {
        const riskValEl = document.getElementById('risk-score-val');
        const badgeEl = document.getElementById('prediction-badge');
        const confValEl = document.getElementById('confidence-val');
        const actionEl = document.getElementById('action-text');
        const gaugeFill = document.getElementById('gauge-fill');

        const risk = data.risk_score || 0;
        const isFailure = data.is_failure;

        if (riskValEl) riskValEl.textContent = `${risk.toFixed(1)}%`;
        if (confValEl) confValEl.textContent = `${data.confidence}%`;
        if (actionEl) actionEl.textContent = data.recommended_action;

        if (badgeEl) {
            if (isFailure) {
                badgeEl.textContent = 'HIGH RISK / FAILURE RISK';
                badgeEl.className = 'status-badge status-danger';
            } else {
                badgeEl.textContent = 'NORMAL OPERATION';
                badgeEl.className = 'status-badge status-normal';
            }
        }

        if (gaugeFill) {
            const circumference = 314; // 2 * PI * 50
            const offset = circumference - (risk / 100) * circumference;
            gaugeFill.style.strokeDashoffset = offset;
            gaugeFill.style.stroke = isFailure ? '#ff4b4b' : '#00f2fe';
        }
    }

    function resetGaugeUI() {
        const riskValEl = document.getElementById('risk-score-val');
        const badgeEl = document.getElementById('prediction-badge');
        const confValEl = document.getElementById('confidence-val');
        const actionEl = document.getElementById('action-text');
        const gaugeFill = document.getElementById('gauge-fill');

        if (riskValEl) riskValEl.textContent = '0.0%';
        if (confValEl) confValEl.textContent = '--%';
        if (actionEl) actionEl.textContent = 'Submit machine telemetry data to compute diagnostic recommendations.';
        if (badgeEl) {
            badgeEl.textContent = 'NORMAL OPERATION';
            badgeEl.className = 'status-badge status-normal';
        }
        if (gaugeFill) {
            gaugeFill.style.strokeDashoffset = 314;
            gaugeFill.style.stroke = '#00f2fe';
        }
    }

    function addHistoryRow(data) {
        const tbody = document.getElementById('history-tbody');
        if (!tbody) return;

        const noRow = document.getElementById('no-history-row');
        if (noRow) noRow.remove();

        const tr = document.createElement('tr');
        const inp = data.inputs;
        const isFail = data.is_failure;

        tr.innerHTML = `
            <td><span class="type-pill">${inp.Type}</span></td>
            <td>${inp.Air_temperature_K}</td>
            <td>${inp.Process_temperature_K}</td>
            <td>${inp.Rotational_speed_rpm}</td>
            <td>${inp.Torque_Nm}</td>
            <td>${inp.Tool_wear_min} min</td>
            <td><strong>${data.risk_score}%</strong></td>
            <td>
                <span class="badge ${isFail ? 'badge-danger' : 'badge-success'}">
                    ${data.prediction}
                </span>
            </td>
            <td class="action-cell">${data.recommended_action}</td>
        `;

        tbody.insertBefore(tr, tbody.firstChild);

        // Keep maximum 10 rows
        while (tbody.children.length > 10) {
            tbody.removeChild(tbody.lastChild);
        }
    }

    // Clear history button
    const btnClear = document.getElementById('btn-clear-history');
    if (btnClear) {
        btnClear.addEventListener('click', () => {
            const tbody = document.getElementById('history-tbody');
            if (tbody) {
                tbody.innerHTML = '<tr id="no-history-row"><td colspan="9" class="text-center text-muted">History cleared.</td></tr>';
            }
        });
    }

    // --- BATCH UPLOAD LOGIC ---
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('batch-file-input');
    const btnBrowse = document.getElementById('btn-browse');
    const selectedFilename = document.getElementById('selected-filename');

    if (dropzone && fileInput) {
        btnBrowse.addEventListener('click', () => fileInput.click());

        fileInput.addEventListener('change', (e) => {
            if (fileInput.files.length > 0) {
                handleBatchFile(fileInput.files[0]);
            }
        });

        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });

        dropzone.addEventListener('dragleave', () => {
            dropzone.classList.remove('dragover');
        });

        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                handleBatchFile(e.dataTransfer.files[0]);
            }
        });
    }

    let currentBatchResults = [];

    async function handleBatchFile(file) {
        if (!file.name.endsWith('.csv')) {
            alert('Please select a valid CSV file');
            return;
        }

        if (selectedFilename) selectedFilename.textContent = `Selected: ${file.name}`;

        const progressContainer = document.getElementById('progress-container');
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');

        if (progressContainer) progressContainer.classList.remove('hidden');
        if (progressFill) progressFill.style.width = '30%';
        if (progressText) progressText.textContent = 'Uploading and processing CSV...';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/batch', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (progressFill) progressFill.style.width = '100%';

            if (response.ok) {
                if (progressText) progressText.textContent = 'Processing Complete!';
                setTimeout(() => {
                    if (progressContainer) progressContainer.classList.add('hidden');
                }, 1000);

                currentBatchResults = data.predictions || [];
                renderBatchSummary(data);
                renderBatchTable(currentBatchResults);
            } else {
                alert(data.error || 'Batch processing failed');
                if (progressContainer) progressContainer.classList.add('hidden');
            }
        } catch (err) {
            console.error('Batch error:', err);
            alert('Failed to connect to batch processing endpoint');
            if (progressContainer) progressContainer.classList.add('hidden');
        }
    }

    function renderBatchSummary(data) {
        const kpiGrid = document.getElementById('batch-summary-cards');
        const batchSection = document.getElementById('batch-results-section');

        if (kpiGrid) kpiGrid.classList.remove('hidden');
        if (batchSection) batchSection.classList.remove('hidden');

        document.getElementById('kpi-total').textContent = data.total_records;
        document.getElementById('kpi-failures').textContent = data.failure_count;
        document.getElementById('kpi-rate').textContent = `${data.failure_rate}%`;
        document.getElementById('kpi-risk').textContent = `${data.avg_risk_score}%`;
    }

    function renderBatchTable(items) {
        const tbody = document.getElementById('batch-tbody');
        if (!tbody) return;

        tbody.innerHTML = '';

        items.forEach((item) => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${item.row_id}</td>
                <td><span class="type-pill">${item.type}</span></td>
                <td>${item.air_temp}</td>
                <td>${item.proc_temp}</td>
                <td>${item.speed}</td>
                <td>${item.torque}</td>
                <td>${item.tool_wear} min</td>
                <td><strong>${item.risk_score}%</strong></td>
                <td>
                    <span class="badge ${item.is_failure ? 'badge-danger' : 'badge-success'}">
                        ${item.prediction}
                    </span>
                </td>
                <td class="action-cell">${item.action}</td>
            `;
            tbody.appendChild(tr);
        });
    }

    // Search filter for batch results
    const searchInput = document.getElementById('table-search');
    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase();
            const filtered = currentBatchResults.filter(r => 
                r.type.toLowerCase().includes(query) ||
                r.prediction.toLowerCase().includes(query) ||
                r.action.toLowerCase().includes(query)
            );
            renderBatchTable(filtered);
        });
    }

    // Export batch results to CSV
    const btnExport = document.getElementById('btn-export-csv');
    if (btnExport) {
        btnExport.addEventListener('click', () => {
            if (!currentBatchResults || currentBatchResults.length === 0) {
                alert('No batch predictions available to export.');
                return;
            }

            const headers = ['Row_ID', 'Product_Type', 'Air_Temp_K', 'Proc_Temp_K', 'Speed_RPM', 'Torque_Nm', 'Tool_Wear_Min', 'Risk_Score_Pct', 'Prediction', 'Action_Recommendation'];
            const rows = currentBatchResults.map(r => [
                r.row_id,
                r.type,
                r.air_temp,
                r.proc_temp,
                r.speed,
                r.torque,
                r.tool_wear,
                r.risk_score,
                r.prediction,
                `"${r.action.replace(/"/g, '""')}"`
            ]);

            const csvContent = [headers.join(','), ...rows.map(row => row.join(','))].join('\n');
            const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.setAttribute('href', url);
            link.setAttribute('download', 'predictive_maintenance_batch_results.csv');
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        });
    }
});

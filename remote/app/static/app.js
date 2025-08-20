// Function to update the browser tab title with the current site name
function updateTabTitleWithSiteName() {
    const siteElement = document.querySelector('.site-id');
    let siteName = '';
    if (siteElement) {
        const text = siteElement.textContent || '';
        const parts = text.split(':');
        if (parts.length > 1) {
            siteName = parts[1].trim();
        } else {
            siteName = text.trim();
        }
    }
    if (siteName) {
        document.title = `remote site - ${siteName}`;
    } else {
        document.title = 'remote site';
    }
}

// Make updateTabTitleWithSiteName available globally
window.updateTabTitleWithSiteName = updateTabTitleWithSiteName;
// This file contains JavaScript code for the remote site

// Helper function to get current site_index from URL
function getCurrentSiteIndex() {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get('site_index') || '0';
}

// Helper function to add site_index to URL
function addSiteIndexToUrl(url) {
    const siteIndex = getCurrentSiteIndex();
    if (url.includes('?')) {
        return url + '&site_index=' + siteIndex;
    } else {
        return url + '?site_index=' + siteIndex;
    }
}

document.addEventListener('DOMContentLoaded', function() {
    // Initialize global job running state
    window.jobIsRunning = false;

    // Set browser tab title to include site name
    // Add a small delay to ensure DOM is fully loaded
    setTimeout(function() {
        updateTabTitleWithSiteName();
        console.log("Updated browser tab title to:", document.title);
    }, 100);
    
    // Set up an additional timer to ensure title is set correctly
    setTimeout(function() {
        updateTabTitleWithSiteName();
        console.log("Re-checked browser tab title:", document.title);
    }, 500);
    
    // Check if job status container is visible, which indicates a running job
    const jobStatusContainer = document.getElementById('job-status-container');
    if (jobStatusContainer && !jobStatusContainer.classList.contains('hidden')) {
        console.log("Found visible job status container on page load - marking job as running");
        window.jobIsRunning = true;
        disableAllJobSubmitButtons();
    }
    
    // Set up the refresh jobs button
    const refreshJobsBtn = document.getElementById('refresh-jobs-btn');
    if (refreshJobsBtn) {
        refreshJobsBtn.addEventListener('click', function() {
            refreshJobData();
        });
    }
    
    // Initialize job data
    const jobData = {};
    
    // Parse job data from hidden JSON field
    try {
        const jobDataElement = document.getElementById('job-data');
        if (jobDataElement) {
            const jobsJSON = jobDataElement.textContent;
            const jobs = JSON.parse(jobsJSON);
            jobs.forEach(job => {
                jobData[job.id] = job;
            });
            console.log("Job data loaded:", jobData);
        }
    } catch (error) {
        console.error("Error parsing job data:", error);
    }
    
    // Set up event listener for job dropdown
    const jobSelect = document.getElementById('job_id');
    if (jobSelect) {
        jobSelect.addEventListener('change', function() {
            updateJobForm(jobData);
        });
        
        // If a job is already selected, update the form
        if (jobSelect.value) {
            updateJobForm(jobData);
        }
    }
    
    // Set up job form submission
    const jobForm = document.getElementById('job-form');
    if (jobForm) {
        jobForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Check first if we already know a job is running (from the window flag)
            if (window.jobIsRunning === true) {
                console.log("Job submission prevented: a job is already running (from window flag)");
                // There's a running job, disable all submit buttons
                disableAllJobSubmitButtons();
                
                // Show error in the job status container
                const jobStatusContainer = document.getElementById('job-status-container');
                const statusMessagesContainer = document.getElementById('status-messages');
                const statusIndicator = document.getElementById('status-indicator');
                
                if (statusMessagesContainer) {
                    statusMessagesContainer.innerHTML = '';
                    const messageEl = document.createElement('div');
                    messageEl.className = 'job-status-line';
                    messageEl.textContent = 'Error: A job is already running. You must stop it before starting a new job.';
                    statusMessagesContainer.appendChild(messageEl);
                    jobStatusContainer.classList.remove('hidden');
                    statusIndicator.textContent = 'Failed';
                    statusIndicator.className = 'status-indicator failed';
                }
                return;
            }
            
            // Immediately disable the submit button to prevent multiple clicks
            const submitButton = jobForm.querySelector('input[type="submit"]');
            if (submitButton) {
                submitButton.disabled = true;
                submitButton.setAttribute('disabled', 'disabled');
                submitButton.value = 'Checking...';
            }
            
            // Get site ID to check for running jobs
            const siteElement = document.querySelector('.site-id');
            const currentSiteId = siteElement?.textContent?.trim()?.split(':')[1]?.trim();
            
            if (currentSiteId) {
                // Double-check for running jobs right before submission
                const checkUrl = addSiteIndexToUrl(`/check_running_jobs?site_id=${currentSiteId}`);
                fetch(checkUrl)
                    .then(response => response.json())
                    .then(data => {
                        if (data.running_job_id) {
                            // There's a running job, disable all submit buttons
                            disableAllJobSubmitButtons();
                            
                            // Show error in the job status container
                            const jobStatusContainer = document.getElementById('job-status-container');
                            const statusMessagesContainer = document.getElementById('status-messages');
                            const statusIndicator = document.getElementById('status-indicator');
                            
                            if (statusMessagesContainer) {
                                statusMessagesContainer.innerHTML = '';
                                const messageEl = document.createElement('div');
                                messageEl.className = 'job-status-line';
                                messageEl.textContent = 'Error: A job is already running. You must stop it before starting a new job.';
                                statusMessagesContainer.appendChild(messageEl);
                                jobStatusContainer.classList.remove('hidden');
                                statusIndicator.textContent = 'Failed';
                                statusIndicator.className = 'status-indicator failed';
                            }
                        } else {
                            // No running job, re-enable the submit button and proceed with submission
                            const submitButton = jobForm.querySelector('input[type="submit"]');
                            if (submitButton) {
                                submitButton.value = 'Start Job';
                                // Don't re-enable here, it will be submitted immediately anyway
                            }
                            submitJobForm(jobForm);
                        }
                    })
                    .catch(error => {
                        console.error('Error checking for running jobs:', error);
                        // If error checking, re-enable the button and proceed with submission anyway
                        const submitButton = jobForm.querySelector('input[type="submit"]');
                        if (submitButton) {
                            submitButton.value = 'Start Job';
                            // Don't re-enable here, it will be submitted immediately anyway
                        }
                        submitJobForm(jobForm);
                    });
            } else {
                // No site ID, proceed with submission
                submitJobForm(jobForm);
            }
        });
    }
    
    // Set up stop job button
    const stopJobButton = document.getElementById('stop-job-button');
    if (stopJobButton) {
        stopJobButton.addEventListener('click', function() {
            stopCurrentJob();
        });
    }
    
    // Get the current site ID from the page
    const siteElement = document.querySelector('.site-id');
    const currentSiteId = siteElement?.textContent?.trim()?.split(':')[1]?.trim();
    
    console.log('Page load: Checking for running jobs directly from server');
    console.log(`- Current site element found: ${siteElement ? 'yes' : 'no'}`);
    console.log(`- Current site ID from DOM: ${currentSiteId || 'not found'}`);
    
    // Check for running jobs directly from the server - no localStorage needed
    if (currentSiteId) {
        console.log(`Checking for running jobs on site ${currentSiteId}...`);
        
        // Query for any running jobs on this site
        fetch(`/check_running_jobs?site_id=${currentSiteId}`)
            .then(response => response.json())
            .then(data => {
                if (data.running_job_id) {
                    console.log(`Found running job: ${data.running_job_id}`);
                    
                    // Start monitoring the running job
                    startJobMonitoring(data.running_job_id);
                    
                    // Show the job status container (startJobMonitoring already disables buttons)
                    const jobStatusContainer = document.getElementById('job-status-container');
                    if (jobStatusContainer) {
                        jobStatusContainer.classList.remove('hidden');
                    }
                } else {
                    console.log('No running jobs found');
                    // Ensure submit buttons are enabled if no job is running
                    enableAllJobSubmitButtons();
                }
            })
            .catch(error => {
                console.error('Error checking for running jobs:', error);
            });
    } else {
        console.log('No site ID available, cannot check for running jobs');
    }
});

// Function to update form fields based on selected job
function updateJobForm(jobData) {
    const jobId = document.getElementById('job_id').value;
    const formContainer = document.getElementById('dynamic-form-fields');
    
    if (!jobId) {
        formContainer.innerHTML = '';
        return;
    }
    
    const job = jobData[jobId];
    if (!job) {
        console.error("Selected job not found in job data");
        return;
    }
    
    console.log("Updating form for job:", job);
    
    let formFields = '';
    
    // All algorithms will have a consistent display format
    formFields = `
        <input type="hidden" name="job_type" value="${job.algorithm}">
        <input type="hidden" id="mvar" name="mvar" value="1">
    `;
    
    // Display variables information based on algorithm type
    if (job.algorithm === 'SIMI') {
        // For SIMI algorithm
        const targetColumnIndex = job.parameters?.target_column_index || '';
        
        formFields += `
            <div class="variables-info">
                <p class="form-info"><strong>Available variables in this job:</strong></p>
                <ul class="variables-list">
                    <li>Column ${targetColumnIndex} (${job.parameters?.is_binary ? 'Binary' : 'Continuous'})</li>
                </ul>
            </div>
            <p class="form-info">This job uses the SIMI algorithm for imputation of a single variable.</p>
        `;
        
        // Display all parameters in a read-only format
        formFields += `<div class="parameters-box">
            <h3>All Job Parameters:</h3>
            <table class="parameters-table">`;
        
        // Loop through all parameters
        for (const [key, value] of Object.entries(job.parameters || {})) {
            formFields += `
                <tr>
                    <td><strong>${key}:</strong></td>
                    <td>${JSON.stringify(value)}</td>
                </tr>
            `;
        }
        
        formFields += `</table></div>`;
        
        // Pre-set the missing variable index to the target column
        setTimeout(() => {
            const mvarInput = document.getElementById('mvar');
            if (mvarInput && targetColumnIndex) {
                mvarInput.value = targetColumnIndex;
            }
        }, 100);
        
    } else if (job.algorithm === 'SIMICE') {
        // For SIMICE algorithm
        const targetColumns = job.parameters?.target_column_indexes || [];
        
        // Show available columns information
        if (Array.isArray(targetColumns) && targetColumns.length > 0) {
            formFields += `
                <div class="variables-info">
                    <p class="form-info"><strong>Available variables in this job:</strong></p>
                    <ul class="variables-list">
            `;
            
            targetColumns.forEach((colIndex, i) => {
                const isBinary = Array.isArray(job.parameters?.is_binary) && job.parameters.is_binary[i] 
                    ? 'Binary' : 'Continuous';
                formFields += `<li>Column ${colIndex} (${isBinary})</li>`;
            });
            
            formFields += `
                    </ul>
                </div>
            `;
        }
        
        formFields += `<p class="form-info">This job uses the SIMICE algorithm for multiple imputation.</p>`;
        
        // Display all parameters in a read-only format
        formFields += `<div class="parameters-box">
            <h3 id="simice-job-params-heading">All Job Parameters:</h3>
            <table class="parameters-table" aria-labelledby="simice-job-params-heading">
            <tbody>`;
        
        // Loop through all parameters
        for (const [key, value] of Object.entries(job.parameters || {})) {
            const paramId = `simice-param-${key.replace(/\s+/g, '-').toLowerCase()}`;
            formFields += `
                <tr>
                    <th scope="row" id="${paramId}">${key}:</th>
                    <td aria-labelledby="${paramId}">${JSON.stringify(value)}</td>
                </tr>
            `;
        }
        
        formFields += `</tbody></table></div>`;
        
        // Add other SIMICE parameters if needed
        if (job.parameters?.iteration_before_first_imputation !== undefined) {
            formFields += `
                <input type="hidden" name="iteration_before_first_imputation" 
                    value="${job.parameters.iteration_before_first_imputation}">
            `;
        }
        
        if (job.parameters?.iteration_between_imputations !== undefined) {
            formFields += `
                <input type="hidden" name="iteration_between_imputations" 
                    value="${job.parameters.iteration_between_imputations}">
            `;
        }
    } else {
        // Default case for other algorithms
        formFields = `
            <label for="mvar"><strong>Missing Variable Index (1-based):</strong></label><br>
            <input type="number" id="mvar" name="mvar" min="1" required><br>
            <p class="form-info">Please enter the index of the variable you want to impute.</p>
        `;
    }
    
    // The CSV file input is already in the HTML, so we don't need to add it dynamically
    formFields += `
        <div class="form-group" style="margin-top: 20px;">
            <p class="form-info">Please upload a CSV file with your data. Missing values should be represented as empty cells or 'NA'.</p>
        </div>
    `;
    
    formContainer.innerHTML = formFields;
}

let pollingInterval;

// Function to submit the job form via AJAX
function submitJobForm(form) {
    const formData = new FormData(form);
    const jobStatusContainer = document.getElementById('job-status-container');
    const statusMessagesContainer = document.getElementById('status-messages');
    const statusIndicator = document.getElementById('status-indicator');
    
    // Always disable the submit button at the start of submission
    const submitButton = form.querySelector('input[type="submit"]');
    if (submitButton) {
        submitButton.disabled = true;
        submitButton.value = 'Starting Job...';
    }
    
    // Check if there's already a job warning notification on the page
    const existingWarning = document.querySelector('.alert.alert-warning');
    if (existingWarning && existingWarning.textContent.includes('A job is already running')) {
        // If there's already a warning, keep the submit button disabled and show an error
        if (submitButton) {
            submitButton.disabled = true;
            submitButton.value = 'Job Already Running...';
        }
        
        // Show error in the status container
        statusMessagesContainer.innerHTML = '';
        const messageEl = document.createElement('div');
        messageEl.className = 'job-status-line';
        messageEl.textContent = 'Error: A job is already running. You must stop it before starting a new job.';
        statusMessagesContainer.appendChild(messageEl);
        jobStatusContainer.classList.remove('hidden');
        statusIndicator.textContent = 'Failed';
        statusIndicator.className = 'status-indicator failed';
        return;
    }
    
    // Get the current site index from URL parameters instead of dropdown
    const currentSiteIndex = getCurrentSiteIndex();
    formData.append('site_index', currentSiteIndex);
    console.log("Added site_index to form submission:", currentSiteIndex);
    
    // Check if CSV file is included
    if (!formData.get('csv_file') || formData.get('csv_file').size === 0) {
        statusMessagesContainer.innerHTML = '';
        const messageEl = document.createElement('div');
        messageEl.className = 'job-status-line';
        messageEl.textContent = 'Error: No CSV file selected';
        statusMessagesContainer.appendChild(messageEl);
        jobStatusContainer.classList.remove('hidden');
        statusIndicator.textContent = 'Failed';
        statusIndicator.className = 'status-indicator failed';
        return;
    }
    
    // Clear previous messages and reset UI
    if (pollingInterval) clearInterval(pollingInterval);
    statusMessagesContainer.innerHTML = '';
    const messageEl = document.createElement('div');
    messageEl.className = 'job-status-line';
    messageEl.textContent = 'Starting job...';
    statusMessagesContainer.appendChild(messageEl);
    
    jobStatusContainer.classList.remove('hidden');
    statusIndicator.textContent = 'Running';
    statusIndicator.className = 'status-indicator';
    
    form.setAttribute('enctype', 'multipart/form-data');
    
    // Add the current site ID to the form data for tracking
    const currentSiteId = document.querySelector('.site-id')?.textContent?.trim()?.split(':')[1]?.trim();
    if (currentSiteId) {
        formData.append('current_site_id', currentSiteId);
    }
    
    fetch('/start_job', {
        method: 'POST',
        headers: {
            'Accept': 'application/json',
            'X-Requested-With': 'XMLHttpRequest'
        },
        body: formData
    })
    .then(response => {
        const contentType = response.headers.get("content-type");
        if (contentType && contentType.indexOf("application/json") !== -1) {
            return response.json();
        }
        return response.text().then(text => {
            console.error("Received HTML instead of JSON", text.substring(0, 200) + "...");
            let errorMsg = "Server returned an HTML error page instead of JSON.";
            if (text.includes("CSRF")) {
                errorMsg = "CSRF verification failed. Please refresh the page and try again.";
            }
            throw new Error(errorMsg);
        });
    })
    .then(data => {
        if (data.success) {
            const messageEl = document.createElement('div');
            // Create a regular status message line
            messageEl.className = 'job-status-line';
            messageEl.textContent = `Job started successfully! Job ID: ${data.job_id}`;
            statusMessagesContainer.appendChild(messageEl);
            
            // Store job ID in a hidden span with job-id class
            const jobIdSpan = document.createElement('span');
            jobIdSpan.className = 'job-id';
            jobIdSpan.style.display = 'none';
            jobIdSpan.dataset.jobId = data.job_id;
            statusMessagesContainer.appendChild(jobIdSpan);
            
            // Disable all submit buttons after successful job submission
            disableAllJobSubmitButtons();
            
            const currentSiteId = document.querySelector('.site-id')?.textContent?.trim()?.split(':')[1]?.trim();
            
            console.log(`Job started successfully! Job ID: ${data.job_id}, Site ID: ${currentSiteId || 'not set'}`);
            console.log(`Site ID from DOM element:`, document.querySelector('.site-id')?.textContent);
            
            // Add a small delay before starting job monitoring to ensure backend has registered the job
            console.log(`Waiting 500ms before starting job monitoring...`);
            setTimeout(() => {
                startJobMonitoring(data.job_id);
            }, 500);
        } else {
            throw new Error(data.error || 'Unknown error occurred.');
        }
    })
    .catch(error => {
        console.error("Job submission error:", error);
        const errorMsg = `Error submitting job: ${error.message || error}`;
        
        if (!Array.from(statusMessagesContainer.children).some(el => el.textContent.includes('Error'))) {
            const messageEl = document.createElement('div');
            messageEl.className = 'job-status-line';
            messageEl.textContent = errorMsg;
            statusMessagesContainer.appendChild(messageEl);
        }
        
        statusIndicator.textContent = 'Failed';
        statusIndicator.className = 'status-indicator failed';
    });
}

// Function to start monitoring a job
function startJobMonitoring(jobId) {
    const statusContainer = document.getElementById('status-messages');
    const statusIndicator = document.getElementById('status-indicator');
    const stopButton = document.getElementById('stop-job-button');
    const jobStatusContainer = document.getElementById('job-status-container');
    
    // Get the current site ID from the page
    const currentSiteId = document.querySelector('.site-id')?.textContent?.trim()?.split(':')[1]?.trim();
    
    console.log(`Starting job monitoring for job ID: ${jobId}, site ID: ${currentSiteId || 'not set'}`);

    // Explicitly set the global job running flag
    window.jobIsRunning = true;
    
    // Make sure containers and buttons are visible
    jobStatusContainer.classList.remove('hidden');
    stopButton.classList.remove('hidden');
    stopButton.disabled = false;
    
    // Disable all job submission buttons when a job is running
    disableAllJobSubmitButtons();
    
    // Set the job ID as a data attribute on the stop button for easy access
    stopButton.dataset.jobId = jobId;
    // Also set it on the job status container as a backup
    jobStatusContainer.dataset.jobId = jobId;

    // Track message count to only append new messages
    let lastMessageCount = 0;
    let isCompleted = false;

    const updateJobStatus = async () => {
        try {
            // Include site_id in the status request if available
            const url = currentSiteId ? 
                `/job_status?job_id=${jobId}&site_id=${currentSiteId}` : 
                `/job_status?job_id=${jobId}`;
            
            console.log(`Fetching job status from: ${url}`);
                
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`Failed to fetch job status: ${response.status}`);
            }
            const data = await response.json();
            console.log(`Job status response:`, data);
            
            // If job ID exists in the response, make sure it's saved in the UI elements
            if (data.job_id) {
                stopButton.dataset.jobId = data.job_id;
                jobStatusContainer.dataset.jobId = data.job_id;
            }
            
            // If job not found or has error flag, check if it's due to service restart
            if ((data.status === "Job not found" || data.error) && !isCompleted) {
                console.log(`Job ${jobId} lost due to service restart for site ${currentSiteId || 'unknown'}`);
                
                // Show service restart error immediately
                isCompleted = true;
                
                // Clear existing messages and show restart error
                statusContainer.innerHTML = '';
                const messageEl = document.createElement('div');
                messageEl.className = 'job-status-line error';
                messageEl.textContent = data.status || "Job failed: Service was restarted or crashed. Please start a new job.";
                statusContainer.appendChild(messageEl);
                
                // Set status to failed
                statusIndicator.textContent = 'Failed';
                statusIndicator.className = 'status-indicator failed';
                
                // Clean up and re-enable controls
                stopButton.disabled = true;
                stopButton.classList.add('hidden');
                clearInterval(pollingInterval);
                enableAllJobSubmitButtons();
                
                return;
            }

            // If job not found but no error flag, retry a few times before showing error
            if (data.status === "Job not found" && !data.error && !isCompleted) {
                console.log(`Job ${jobId} not found for site ${currentSiteId || 'unknown'}, will retry...`);
                return; // Don't update UI, just wait for next polling cycle
            }

            // Process status messages
            if (data.status) {
                const messages = data.status.split('\n').filter(msg => msg.trim() !== '');
                
                // Add only new messages to the display
                if (messages.length > lastMessageCount) {
                    for (let i = lastMessageCount; i < messages.length; i++) {
                        const messageEl = document.createElement('div');
                        messageEl.className = 'job-status-line';
                        messageEl.textContent = messages[i];
                        statusContainer.appendChild(messageEl);
                    }
                    lastMessageCount = messages.length;
                    
                    // Auto-scroll to the latest messages
                    const messagesContainer = document.getElementById('status-messages-container');
                    if(messagesContainer) {
                        messagesContainer.scrollTop = messagesContainer.scrollHeight;
                    }
                }
            }

            // Update status indicator based on job completion
            if (data.completed) {
                isCompleted = true;
                
                // Check if this is a real failure or just a retry situation
                const isRealFailure = data.status && (
                    (data.status.toLowerCase().includes('error') || data.status.toLowerCase().includes('failed')) &&
                    !data.status.toLowerCase().includes('waiting for jobs') &&
                    !data.status.toLowerCase().includes('no jobs are currently running') &&
                    !data.status.toLowerCase().includes('no jobs available')
                );
                
                if (isRealFailure) {
                    statusIndicator.textContent = 'Failed';
                    statusIndicator.className = 'status-indicator failed';
                } else {
                    statusIndicator.textContent = 'Completed';
                    statusIndicator.className = 'status-indicator completed';
                }
                
                // Clean up after job completion
                stopButton.disabled = true;
                stopButton.classList.add('hidden');
                clearInterval(pollingInterval);
                
                // Re-enable job submit buttons after completion
                enableAllJobSubmitButtons();
            } else if (data.round_completed) {
                // A round of data exchange is complete but job continues running
                statusIndicator.textContent = 'Running (Round Complete)';
                statusIndicator.className = 'status-indicator round-complete';
                console.log('Round completed, but job continues running');
            } else {
                // Check if we're in a waiting/retry state
                const isWaitingState = data.status && (
                    data.status.toLowerCase().includes('waiting for jobs') ||
                    data.status.toLowerCase().includes('no jobs are currently running') ||
                    data.status.toLowerCase().includes('no jobs available') ||
                    data.status.toLowerCase().includes('waiting') && data.status.toLowerCase().includes('retry')
                );
                
                if (isWaitingState) {
                    statusIndicator.textContent = 'Waiting for Jobs';
                    statusIndicator.className = 'status-indicator waiting';
                } else {
                    statusIndicator.textContent = 'Running';
                    statusIndicator.className = 'status-indicator';
                }
            }
        } catch (error) {
            console.error('Error fetching job status:', error);
            // Don't stop polling on fetch error, just log it
        }
    };

    // Start polling for updates
    if (pollingInterval) clearInterval(pollingInterval);
    updateJobStatus();
    pollingInterval = setInterval(updateJobStatus, 2000); // Match the central site's 2-second polling interval
}

// Function to stop the currently running job
function stopCurrentJob() {
    // Get the job ID directly from the stop button or job status container
    const stopButton = document.getElementById('stop-job-button');
    const jobStatusContainer = document.getElementById('job-status-container');
    
    // Try to get job ID from stop button first, fall back to job status container
    const jobId = stopButton.dataset.jobId || jobStatusContainer.dataset.jobId;
    
    if (!jobId) {
        alert('No running job found.');
        return;
    }

    // Get the current site ID from the page
    const currentSiteId = document.querySelector('.site-id')?.textContent?.trim()?.split(':')[1]?.trim();

    if (confirm('Are you sure you want to stop this job?')) {
        // Add a status message about the stop request
        const statusContainer = document.getElementById('status-messages');
        const messageEl = document.createElement('div');
        messageEl.className = 'job-status-line';
        messageEl.textContent = 'Stop request sent';
        statusContainer.appendChild(messageEl);

        // Disable the button to prevent multiple clicks
        const stopButton = document.getElementById('stop-job-button');
        if (stopButton) {
            stopButton.disabled = true;
        }

        // Include site_id in the request if available
        const url = currentSiteId ? 
            `/stop_job?job_id=${jobId}&site_id=${currentSiteId}` : 
            `/stop_job?job_id=${jobId}`;

        // Send the stop request
        fetch(url, {
            method: 'POST',
            headers: {
                'Accept': 'application/json',
                'X-Requested-With': 'XMLHttpRequest'
            }
        })
        .then(response => response.json())
        .then(data => {
            const resultMessageEl = document.createElement('div');
            resultMessageEl.className = 'job-status-line';
            if (data.success) {
                resultMessageEl.textContent = 'Job stopped successfully.';
                if (pollingInterval) clearInterval(pollingInterval);
                
                if (stopButton) {
                    stopButton.classList.add('hidden');
                }

                const statusIndicator = document.getElementById('status-indicator');
                statusIndicator.textContent = 'Stopped';
                statusIndicator.className = 'status-indicator completed';
                
                // Re-enable all job forms
                enableAllJobSubmitButtons();
            } else {
                resultMessageEl.textContent = `Error stopping job: ${data.error}`;
                // Re-enable the button if the stop failed
                if (stopButton) {
                    stopButton.disabled = false;
                }
            }
            statusContainer.appendChild(resultMessageEl);
            
            // Auto-scroll to the latest messages
            const messagesContainer = document.getElementById('status-messages-container');
            if(messagesContainer) {
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
        })
        .catch(error => {
            console.error('Error stopping job:', error);
            
            const errorMsgEl = document.createElement('div');
            errorMsgEl.className = 'job-status-line';
            errorMsgEl.textContent = `Error stopping job: ${error.message}`;
            statusContainer.appendChild(errorMsgEl);
            
            // Re-enable the button if the stop failed
            if (stopButton) {
                stopButton.disabled = false;
            }
        });
    }
}

// Helper function to disable all job submit buttons
function disableAllJobSubmitButtons() {
    console.log("Disabling all job submit buttons");
    console.log("Current window.jobIsRunning state:", window.jobIsRunning);
    
    // Get the specific job form
    const jobForm = document.getElementById('job-form');
    if (!jobForm) {
        console.log("No job form found");
        return;
    }

    const submitButton = jobForm.querySelector('input[type="submit"]');
    if (submitButton) {
        console.log("Submit button before disable:", {
            disabled: submitButton.disabled,
            value: submitButton.value,
            hasDisabledAttribute: submitButton.hasAttribute('disabled')
        });
        
        // Force disable it - some browsers might restore the previous state on refresh
        submitButton.disabled = true;
        submitButton.setAttribute('disabled', 'disabled');
        submitButton.value = 'Job Already Running...';
        
        console.log("Submit button after disable:", {
            disabled: submitButton.disabled,
            value: submitButton.value,
            hasDisabledAttribute: submitButton.hasAttribute('disabled')
        });
        
        // Add notification at the top of the form if it doesn't exist already
        if (!jobForm.querySelector('.alert-warning')) {
            const notification = document.createElement('div');
            notification.className = 'alert alert-warning';
            notification.style.padding = '10px 15px';
            notification.style.marginBottom = '15px';
            notification.style.backgroundColor = '#fff3cd';
            notification.style.color = '#664d03';
            notification.style.borderLeft = '4px solid #ffca2c';
            notification.innerHTML = `<strong>Note:</strong> A job is already running. You must stop it before starting a new job.`;
            jobForm.prepend(notification);
        }
    } else {
        console.log("No submit button found in job form");
    }
    
    // Also add a global flag to indicate a job is running
    window.jobIsRunning = true;
    console.log("Set window.jobIsRunning to true");
}

// Helper function to enable all job submit buttons
function enableAllJobSubmitButtons() {
    console.log("Enabling all job submit buttons");
    console.log("Current window.jobIsRunning state before enable:", window.jobIsRunning);
    
    // Get the specific job form
    const jobForm = document.getElementById('job-form');
    if (!jobForm) {
        console.log("No job form found");
        return;
    }

    const submitButton = jobForm.querySelector('input[type="submit"]');
    if (submitButton) {
        console.log("Submit button before enable:", {
            disabled: submitButton.disabled,
            value: submitButton.value,
            hasDisabledAttribute: submitButton.hasAttribute('disabled')
        });
        
        submitButton.disabled = false;
        submitButton.removeAttribute('disabled');
        submitButton.value = 'Start Job';
        
        console.log("Submit button after enable:", {
            disabled: submitButton.disabled,
            value: submitButton.value,
            hasDisabledAttribute: submitButton.hasAttribute('disabled')
        });
        
        // Remove any warning notifications
        const warnings = jobForm.querySelectorAll('.alert-warning');
        warnings.forEach(warning => warning.remove());
    } else {
        console.log("No submit button found in job form");
    }
    
    // Clear the global flag
    window.jobIsRunning = false;
    console.log("Set window.jobIsRunning to false");
}

// Function to refresh job data without reloading the page
function refreshJobData() {
    console.log("Refreshing job data...");
    
    // Get the current site ID from the page
    const siteElement = document.querySelector('.site-id');
    const currentSiteId = siteElement?.textContent?.trim()?.split(':')[1]?.trim();
    
    // Show loading indicator on the button
    const refreshBtn = document.getElementById('refresh-jobs-btn');
    const originalBtnText = refreshBtn.textContent;
    refreshBtn.textContent = '⟳ Loading...';
    refreshBtn.disabled = true;
    
    // Construct the URL for fetching updated job data, preserving site_index
    const baseUrl = currentSiteId ? 
        `/get_jobs?site_id=${currentSiteId}` : 
        '/get_jobs';
    const url = addSiteIndexToUrl(baseUrl);
    
    // Fetch updated job data
    fetch(url)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Failed to refresh data: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("Refreshed job data:", data);
            
            // Update the job dropdown
            const jobSelect = document.getElementById('job_id');
            if (jobSelect) {
                // Save the currently selected value
                const currentlySelected = jobSelect.value;
                
                // Clear existing options
                jobSelect.innerHTML = '<option value="">-- Select a job --</option>';
                
                // Add new options
                data.forEach(job => {
                    const option = document.createElement('option');
                    option.value = job.id;
                    option.textContent = `#${job.id} - ${job.name} (${job.algorithm})`;
                    jobSelect.appendChild(option);
                });
                
                // Try to restore the previously selected value if it still exists
                if (currentlySelected) {
                    const exists = Array.from(jobSelect.options).some(opt => opt.value === currentlySelected);
                    if (exists) {
                        jobSelect.value = currentlySelected;
                    }
                }
                
                // Update the hidden job data element
                const jobDataElement = document.getElementById('job-data');
                if (jobDataElement) {
                    jobDataElement.textContent = JSON.stringify(data);
                    
                    // Parse job data for the form updates
                    const jobData = {};
                    data.forEach(job => {
                        jobData[job.id] = job;
                    });
                    
                    // Update form if a job is selected
                    if (jobSelect.value) {
                        updateJobForm(jobData);
                    }
                }
            }
            
            // Reset the refresh button
            refreshBtn.textContent = originalBtnText;
            refreshBtn.disabled = false;
        })
        .catch(error => {
            console.error("Error refreshing job data:", error);
            
            // Show error on the button temporarily
            refreshBtn.textContent = '✗ Error';
            
            // Reset button after a delay
            setTimeout(() => {
                refreshBtn.textContent = originalBtnText;
                refreshBtn.disabled = false;
            }, 2000);
        });
}

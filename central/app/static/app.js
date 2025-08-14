document.addEventListener('DOMContentLoaded', () => {
  // Initialize global job running state
  window.jobIsRunning = false;
  
  // Attach CSRF token from meta to all forms as hidden input if missing
  const meta = document.querySelector('meta[name="csrf-token"]');
  if (meta) {
    const token = meta.getAttribute('content');
    document.querySelectorAll('form').forEach(f => {
      if (!f.querySelector('input[name="csrf_token"]')) {
        const i = document.createElement('input');
        i.type = 'hidden';
        i.name = 'csrf_token';
        i.value = token;
        f.appendChild(i);
      }
    });
  }
  
  // Check if job status container is visible, which indicates a running job
  const jobStatusContainer = document.getElementById('job-status-container');
  if (jobStatusContainer && !jobStatusContainer.classList.contains('hidden')) {
    console.log("Found visible job status container on page load - marking job as running");
    window.jobIsRunning = true;
    disableJobStartButton();
  }
  
  // Add accessible table sorting functionality
  document.querySelectorAll('table.table th').forEach(th => {
    if (th.textContent.trim() !== 'Actions' && th.textContent.trim() !== 'Action') {
      th.setAttribute('role', 'button');
      th.setAttribute('tabindex', '0');
      th.style.cursor = 'pointer';
      
      // Add sort functionality
      th.addEventListener('click', () => {
        sortTable(th);
      });
      
      // Add keyboard support
      th.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          sortTable(th);
        }
      });
    }
  });
  
  // Function to sort tables
  function sortTable(th) {
    const table = th.closest('table');
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const index = Array.from(th.parentNode.children).indexOf(th);
    const sortDirection = th.getAttribute('aria-sort') === 'ascending' ? 'descending' : 'ascending';
    
    // Remove sort attributes from all headers
    table.querySelectorAll('th').forEach(header => {
      header.removeAttribute('aria-sort');
    });
    
    // Set new sort direction
    th.setAttribute('aria-sort', sortDirection);
    
    // Announce to screen readers
    const announcement = document.createElement('div');
    announcement.setAttribute('aria-live', 'polite');
    announcement.setAttribute('class', 'sr-only');
    announcement.textContent = `Table sorted by ${th.textContent} in ${sortDirection} order`;
    document.body.appendChild(announcement);
    setTimeout(() => announcement.remove(), 3000);
    
    // Sort rows
    rows.sort((a, b) => {
      const aValue = a.cells[index].textContent.trim();
      const bValue = b.cells[index].textContent.trim();
      
      // Check if values are dates
      const aDate = new Date(aValue);
      const bDate = new Date(bValue);
      if (!isNaN(aDate) && !isNaN(bDate)) {
        return sortDirection === 'ascending' ? aDate - bDate : bDate - aDate;
      }
      
      // Check if values are numbers
      const aNum = parseFloat(aValue);
      const bNum = parseFloat(bValue);
      if (!isNaN(aNum) && !isNaN(bNum)) {
        return sortDirection === 'ascending' ? aNum - bNum : bNum - aNum;
      }
      
      // Sort as strings
      return sortDirection === 'ascending' 
        ? aValue.localeCompare(bValue) 
        : bValue.localeCompare(aValue);
    });
    
    // Reattach sorted rows
    rows.forEach(row => tbody.appendChild(row));
  }
  
  // Check for running jobs on page load
  console.log('Page load: Checking for running jobs directly from server');
  // Query for any running jobs
  fetch('/test/job-check')
    .then(response => {
      console.log('Response status:', response.status);
      return response.json();
    })
    .then(data => {
      console.log('Job check data received:', data);
      if (data.running_job_id) {
        console.log(`Found running job: ${data.running_job_id}`);
        
        // Start monitoring the running job
        startJobMonitoring(data.running_job_id);
        
        // Show the job status container
        const jobStatusContainer = document.getElementById('job-status-container');
        if (jobStatusContainer) {
          jobStatusContainer.classList.remove('hidden');
        }
      } else {
        console.log('No running jobs found');
        // Ensure submit button is enabled if no job is running
        enableJobStartButton();
      }
    })
    .catch(error => {
      console.error('Error checking for running jobs:', error);
    });
    
  // Function to start job monitoring
  function startJobMonitoring(jobId) {
    // Explicitly set the global job running flag
    window.jobIsRunning = true;
    window.activeJobId = jobId;
    
    // Disable the start job button
    disableJobStartButton();
    
    // Set up job status display
    const statusContainer = document.getElementById('job-status-messages');
    const statusIndicator = document.getElementById('job-status-indicator');
    const stopButton = document.getElementById('stop-job-btn');
    
    // Make sure containers and buttons are visible
    const jobStatusContainer = document.getElementById('job-status-container');
    if (jobStatusContainer) {
      jobStatusContainer.classList.remove('hidden');
    }
    
    if (stopButton) {
      stopButton.classList.remove('hidden');
      stopButton.disabled = false;
    }
    
    // Display job ID in the status area
    if (jobStatusContainer && jobStatusContainer.querySelector('h2')) {
      jobStatusContainer.querySelector('h2').textContent = `Job #${jobId} Status`;
    }
    
    monitorJobStatus(jobId);
  }
  
  // Function to monitor job status
  function monitorJobStatus(jobId) {
    const statusContainer = document.getElementById('job-status-messages');
    const statusIndicator = document.getElementById('job-status-indicator');
    const stopButton = document.getElementById('stop-job-btn');
    
    let lastMessageCount = 0;
    let isCompleted = false;
    let pollingInterval;
    
    // Function to update job status display
    const updateJobStatus = async () => {
      try {
        const response = await fetch(`/api/jobs/status/${jobId}`);
        if (!response.ok) {
          throw new Error(`Failed to fetch job status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Update status messages
        if (data.messages && data.messages.length > lastMessageCount) {
          // Add only new messages
          for (let i = lastMessageCount; i < data.messages.length; i++) {
            const messageEl = document.createElement('div');
            messageEl.className = 'job-status-line';
            messageEl.textContent = data.messages[i];
            statusContainer.appendChild(messageEl);
          }
          lastMessageCount = data.messages.length;
          
          // Auto-scroll to bottom
          statusContainer.scrollTop = statusContainer.scrollHeight;
        }
        
        // Update status indicator
        if (data.completed) {
          isCompleted = true;
          if (data.error) {
            statusIndicator.textContent = 'Failed';
            statusIndicator.className = 'status-running status-failed';
          } else {
            statusIndicator.textContent = 'Completed';
            statusIndicator.className = 'status-running status-completed';
            
            console.log("Job completed, checking for imputed dataset:", data);
            
            // Only add the download link if there's an imputed dataset path available
            if (data.imputed_dataset_path && !document.getElementById('download-results-btn')) {
              console.log("Adding download button for imputed dataset:", data.imputed_dataset_path);
              
              const downloadButton = document.createElement('a');
              downloadButton.id = 'download-results-btn';
              downloadButton.href = `/api/jobs/${jobId}/download`;
              downloadButton.className = 'btn-primary';
              downloadButton.textContent = 'Download Results';
              downloadButton.style.marginRight = '10px';
              
              // Insert button before the stop button
              if (stopButton && stopButton.parentNode) {
                stopButton.parentNode.insertBefore(downloadButton, stopButton);
              }
              
              // Also add a message about the available download
              const messageEl = document.createElement('div');
              messageEl.className = 'job-status-line';
              messageEl.innerHTML = '<strong>✓ Results ready:</strong> Click the "Download Results" button to download the imputed dataset.';
              statusContainer.appendChild(messageEl);
              statusContainer.scrollTop = statusContainer.scrollHeight;
            } else if (!data.imputed_dataset_path) {
              console.log("No imputed dataset available for this job");
              
              // Add a message explaining that no dataset is available
              const messageEl = document.createElement('div');
              messageEl.className = 'job-status-line';
              messageEl.innerHTML = '<strong>ℹ️ Note:</strong> Job completed successfully, but no downloadable results are available yet.';
              statusContainer.appendChild(messageEl);
              statusContainer.scrollTop = statusContainer.scrollHeight;
            }
            
            // Update job status in the job list if it exists on the page
            updateJobStatusInTable(jobId, 'Completed');
          }
          
          // Disable stop button
          if (stopButton) {
            stopButton.disabled = true;
          }
          
          // Stop polling
          clearInterval(pollingInterval);
          
          // Re-enable job submit button after completion
          enableJobStartButton();
          
          // Clear global running flag
          window.jobIsRunning = false;
        }
      } catch (error) {
        console.error('Error fetching job status:', error);
      }
    };
    
    // Set up stop button handler
    if (stopButton) {
      stopButton.addEventListener('click', async () => {
        if (confirm('Are you sure you want to stop this job?')) {
          try {
            // Get the CSRF token from the meta tag
            const csrfMeta = document.querySelector('meta[name="csrf-token"]');
            const csrfToken = csrfMeta ? csrfMeta.getAttribute('content') : '';
            
            const response = await fetch(`/api/jobs/stop/${jobId}`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                'X-CSRF-Token': csrfToken
              }
            });
            
            if (response.ok) {
              // Add a message about stopping
              const messageEl = document.createElement('div');
              messageEl.className = 'job-status-line';
              messageEl.textContent = 'Stop request sent';
              statusContainer.appendChild(messageEl);
              
              // Disable the button to prevent multiple clicks
              stopButton.disabled = true;
            } else {
              alert('Failed to stop job');
            }
          } catch (error) {
            console.error('Error stopping job:', error);
            alert('Error stopping job: ' + error.message);
          }
        }
      });
    }
    
    // Initial status update
    updateJobStatus();
    
    // Poll for updates every 2 seconds
    pollingInterval = setInterval(updateJobStatus, 2000);
  }
  
  // Helper function to disable job start button
  function disableJobStartButton() {
    console.log("Disabling job start button");
    console.log("Current window.jobIsRunning state:", window.jobIsRunning);
    
    // Get the start job form
    const jobForm = document.getElementById('start-job-form');
    if (!jobForm) {
      console.log("No job form found");
      return;
    }

    const submitButton = jobForm.querySelector('button[type="submit"]');
    if (submitButton) {
      console.log("Submit button before disable:", {
        disabled: submitButton.disabled,
        textContent: submitButton.textContent,
        hasDisabledAttribute: submitButton.hasAttribute('disabled')
      });
      
      // Force disable it - some browsers might restore the previous state on refresh
      submitButton.disabled = true;
      submitButton.setAttribute('disabled', 'disabled');
      submitButton.textContent = 'Job Already Running...';
      
      console.log("Submit button after disable:", {
        disabled: submitButton.disabled,
        textContent: submitButton.textContent,
        hasDisabledAttribute: submitButton.hasAttribute('disabled')
      });
      
      // Also disable the job select dropdown and file input
      const jobSelect = jobForm.querySelector('select[name="job_id"]');
      const fileInput = jobForm.querySelector('input[name="central_data_file"]');
      
      if (jobSelect) {
        jobSelect.disabled = true;
        jobSelect.setAttribute('disabled', 'disabled');
      }
      
      if (fileInput) {
        fileInput.disabled = true;
        fileInput.setAttribute('disabled', 'disabled');
      }
      
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
    
    // Set the global flag
    window.jobIsRunning = true;
    console.log("Set window.jobIsRunning to true");
  }
  
  // Helper function to enable job start button
  function enableJobStartButton() {
    console.log("Enabling job start button");
    console.log("Current window.jobIsRunning state before enable:", window.jobIsRunning);
    
    // Get the start job form
    const jobForm = document.getElementById('start-job-form');
    if (!jobForm) {
      console.log("No job form found");
      return;
    }

    const submitButton = jobForm.querySelector('button[type="submit"]');
    if (submitButton) {
      console.log("Submit button before enable:", {
        disabled: submitButton.disabled,
        textContent: submitButton.textContent,
        hasDisabledAttribute: submitButton.hasAttribute('disabled')
      });
      
      submitButton.disabled = false;
      submitButton.removeAttribute('disabled');
      submitButton.textContent = 'Start';
      
      console.log("Submit button after enable:", {
        disabled: submitButton.disabled,
        textContent: submitButton.textContent,
        hasDisabledAttribute: submitButton.hasAttribute('disabled')
      });
      
      // Also enable the job select dropdown and file input
      const jobSelect = jobForm.querySelector('select[name="job_id"]');
      const fileInput = jobForm.querySelector('input[name="central_data_file"]');
      
      if (jobSelect) {
        jobSelect.disabled = false;
        jobSelect.removeAttribute('disabled');
      }
      
      if (fileInput) {
        fileInput.disabled = false;
        fileInput.removeAttribute('disabled');
      }
      
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
  
  // Function to update job status in the jobs table
  function updateJobStatusInTable(jobId, status) {
    // Check if we're on the jobs page with a table
    const jobsTable = document.querySelector('table.table');
    if (!jobsTable) return;
    
    console.log(`Updating job ${jobId} status in table to ${status}`);
    
    // Find the row with the matching job ID
    const rows = jobsTable.querySelectorAll('tbody tr');
    for (const row of rows) {
      const idCell = row.cells[0]; // Assuming ID is in the first column
      if (idCell && idCell.textContent.trim() == jobId) {
        console.log(`Found row for job ${jobId}`);
        
        // Find the status cell (5th column based on the template)
        const statusCell = row.cells[4]; // Status is in the 5th column (0-based index is 4)
        if (statusCell) {
          statusCell.textContent = status;
        }
        
        // Check for imputed dataset path
        fetch(`/api/jobs/${jobId}`)
          .then(response => {
            if (!response.ok) {
              throw new Error(`Failed to fetch job details: ${response.status}`);
            }
            return response.json();
          })
          .then(jobData => {
            console.log(`Job ${jobId} details:`, jobData);
            
            // Check if there's a download button in the actions cell
            const actionsCell = row.cells[6]; // Actions column (0-based index is 6)
            if (actionsCell && jobData.imputed_dataset_path) {
              console.log(`Job ${jobId} has imputed dataset path:`, jobData.imputed_dataset_path);
              
              // If there's no download button, add one
              if (!actionsCell.querySelector('a[href*="/download"]')) {
                const downloadLink = document.createElement('a');
                downloadLink.href = `/api/jobs/${jobId}/download`;
                downloadLink.className = 'button-link';
                downloadLink.setAttribute('aria-label', `Download imputed dataset for job ${jobId}`);
                
                const downloadButton = document.createElement('button');
                downloadButton.textContent = 'Download';
                
                downloadLink.appendChild(downloadButton);
                
                // Insert after the edit button
                const editLink = actionsCell.querySelector('a[href*="/edit"]');
                if (editLink) {
                  editLink.insertAdjacentElement('afterend', downloadLink);
                  // Add a space for better formatting
                  editLink.insertAdjacentHTML('afterend', ' ');
                } else {
                  actionsCell.prepend(downloadLink);
                }
                
                console.log(`Added download button for job ${jobId}`);
              }
            }
          })
          .catch(error => {
            console.error(`Error fetching job ${jobId} details:`, error);
          });
          
        break;
      }
    }
  }
  
  // Job form submission handling
  const startJobForm = document.getElementById('start-job-form');
  if (startJobForm) {
    startJobForm.addEventListener('submit', function(e) {
      e.preventDefault();
      
      // Check first if we already know a job is running (from the window flag)
      if (window.jobIsRunning === true) {
        console.log("Job submission prevented: a job is already running");
        // There's a running job, disable the submit button
        disableJobStartButton();
        
        // Show error in the job status container
        const jobStatusContainer = document.getElementById('job-status-container');
        const statusMessagesContainer = document.getElementById('job-status-messages');
        const statusIndicator = document.getElementById('job-status-indicator');
        
        if (jobStatusContainer && statusMessagesContainer) {
          jobStatusContainer.classList.remove('hidden');
          statusMessagesContainer.innerHTML = '';
          const messageEl = document.createElement('div');
          messageEl.className = 'job-status-line';
          messageEl.textContent = 'Error: A job is already running. You must stop it before starting a new job.';
          statusMessagesContainer.appendChild(messageEl);
          if (statusIndicator) {
            statusIndicator.textContent = 'Failed';
            statusIndicator.className = 'status-running status-failed';
          }
        }
        return;
      }
      
      // Immediately disable the submit button to prevent multiple clicks
      const submitButton = startJobForm.querySelector('button[type="submit"]');
      if (submitButton) {
        submitButton.disabled = true;
        submitButton.setAttribute('disabled', 'disabled');
        submitButton.textContent = 'Checking...';
      }
      
      // Double-check for running jobs right before submission
      fetch(`/test/job-check`)
        .then(response => {
          console.log('Check before submit - response status:', response.status);
          return response.json();
        })
        .then(data => {
          console.log('Check before submit - data received:', data);
          if (data.running_job_id) {
            // There's a running job, disable the submit button
            disableJobStartButton();
            
            // Show error in the job status container
            const jobStatusContainer = document.getElementById('job-status-container');
            const statusMessagesContainer = document.getElementById('job-status-messages');
            const statusIndicator = document.getElementById('job-status-indicator');
            
            if (jobStatusContainer && statusMessagesContainer) {
              jobStatusContainer.classList.remove('hidden');
              statusMessagesContainer.innerHTML = '';
              const messageEl = document.createElement('div');
              messageEl.className = 'job-status-line';
              messageEl.textContent = 'Error: A job is already running. You must stop it before starting a new job.';
              statusMessagesContainer.appendChild(messageEl);
              if (statusIndicator) {
                statusIndicator.textContent = 'Failed';
                statusIndicator.className = 'status-running status-failed';
              }
            }
          } else {
            // No running job, submit the form
            startJobForm.removeEventListener('submit', arguments.callee);
            startJobForm.submit();
          }
        })
        .catch(error => {
          console.error('Error checking for running jobs:', error);
          // If error checking, re-enable the button
          if (submitButton) {
            submitButton.disabled = false;
            submitButton.removeAttribute('disabled');
            submitButton.textContent = 'Start';
          }
          alert('Error checking for running jobs: ' + error.message);
        });
    });
  }
  
  // Job selection functionality - for displaying parameters on job selection
  const jobSelect = document.getElementById('job-select');
  const jobParamsContainer = document.getElementById('job-parameters');
  
  if (jobSelect && jobParamsContainer) {
    // Initialize job data from the hidden JSON element
    const jobData = {};
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
    
    // Update parameters display when job is selected
    jobSelect.addEventListener('change', function() {
      const jobId = this.value;
      if (!jobId) {
        jobParamsContainer.innerHTML = '';
        return;
      }
      
      const job = jobData[jobId];
      if (!job) {
        console.error("Selected job not found in job data");
        return;
      }
      
      // Function to format values nicely
      const formatValue = (value) => {
        if (value === null || value === undefined) return 'Not specified';
        if (Array.isArray(value)) {
          if (value.length === 0) return 'Empty list';
          return value.join(', ');
        }
        if (typeof value === 'object') {
          return JSON.stringify(value, null, 2)
            .replace(/[{}"]/g, '')
            .replace(/,/g, '')
            .trim();
        }
        return value.toString();
      };
      
      // Display job parameters
      let html = `
        <div class="parameters-box">
          <h3 id="job-parameters-heading">Job Parameters:</h3>
          <table class="parameters-table" aria-labelledby="job-parameters-heading">
            <tbody>
              <tr>
                <th scope="row">Algorithm:</th>
                <td>${job.algorithm}</td>
              </tr>
              <tr>
                <th scope="row">Description:</th>
                <td>${job.description || 'No description'}</td>
              </tr>
      `;
      
      // First, add algorithm-specific parameters in a logical order
      if (job.parameters && Object.keys(job.parameters).length > 0) {
        // Add header for algorithm parameters section
        html += `<tr><td colspan="2"><h4 id="algorithm-params-heading">${job.algorithm} Parameters:</h4></td></tr>`;
        
        // Add parameters in the order we specified in the server code
        for (const [key, value] of Object.entries(job.parameters)) {
          const formattedKey = key
            .replace(/_/g, ' ')
            .replace(/\b\w/g, c => c.toUpperCase());
          const paramId = `param-${key.replace(/\s+/g, '-').toLowerCase()}`;
            
          html += `
            <tr>
              <th scope="row" id="${paramId}">${formattedKey}:</th>
              <td aria-labelledby="${paramId}">${formatValue(value)}</td>
            </tr>
          `;
        }
      }
      
      // Add additional job settings
      html += `<tr><td colspan="2"><h4 id="additional-settings-heading">Additional Settings:</h4></td></tr>`;
      
      if (job.iteration_before_first_imputation !== null && job.iteration_before_first_imputation !== undefined) {
        html += `
          <tr>
            <th scope="row" id="iterations-first">Iterations Before First Imputation:</th>
            <td aria-labelledby="iterations-first">${job.iteration_before_first_imputation}</td>
          </tr>
        `;
      }
      
      if (job.iteration_between_imputations !== null && job.iteration_between_imputations !== undefined) {
        html += `
          <tr>
            <th scope="row" id="iterations-between">Iterations Between Imputations:</th>
            <td aria-labelledby="iterations-between">${job.iteration_between_imputations}</td>
          </tr>
        `;
      }
      
      if (job.imputation_trials !== null && job.imputation_trials !== undefined) {
        html += `
          <tr>
            <th scope="row" id="imputation-trials">Imputation Trials:</th>
            <td aria-labelledby="imputation-trials">${job.imputation_trials}</td>
          </tr>
        `;
      }
      
      // Add participating sites
      if (job.participants && job.participants.length > 0) {
        html += `
          <tr>
            <th scope="row" id="participating-sites">Participating Sites:</th>
            <td aria-labelledby="participating-sites">${job.participants.join(', ')}</td>
          </tr>
        `;
      }
      
      html += `</tbody></table></div>`;
      
      jobParamsContainer.innerHTML = html;
    });
    
    // Trigger change event if a job is already selected
    if (jobSelect.value) {
      jobSelect.dispatchEvent(new Event('change'));
    }
  }

  const algo = document.getElementById('algo-select');
  const simiSection = document.querySelectorAll('.section-simi');
  const simiceSection = document.querySelectorAll('.section-simice');
  const toggleSections = () => {
    const v = (algo?.value || '').toUpperCase();
    simiSection.forEach(el => el.style.display = (v === 'SIMICE') ? 'none' : 'block');
    simiceSection.forEach(el => el.style.display = (v === 'SIMICE') ? 'block' : 'none');
  };
  if (algo) { algo.addEventListener('change', toggleSections); toggleSections(); }

  // CSV header picker
  const fileInput = document.getElementById('header-file');
  const simiHeader = document.getElementById('simi-header-select');
  const simiIndex = document.getElementById('target_column_index');
  const simiceHeaders = document.getElementById('simice-headers-select');
  const simiceIndexes = document.getElementById('target_column_indexes');
  const openBinaryBtn = document.getElementById('open-binary-modal');
  const binaryModal = document.getElementById('binary-modal');
  const binaryClose = document.getElementById('binary-modal-close');
  const binaryCancel = document.getElementById('binary-modal-cancel');
  const binarySave = document.getElementById('binary-modal-save');
  const binaryList = document.getElementById('binary-list');
  const isBinaryListInput = document.getElementById('is_binary_list');

  function parseCSVHeaders(text) {
    // simple split by first newline, handle quotes minimally
    const firstLine = text.split(/\r?\n/)[0] || '';
    // split by comma not inside quotes
    const headers = [];
    let current = '';
    let inQuotes = false;
    for (let i = 0; i < firstLine.length; i++) {
      const c = firstLine[i];
      if (c === '"') { inQuotes = !inQuotes; continue; }
      if (c === ',' && !inQuotes) { headers.push(current.trim()); current = ''; }
      else current += c;
    }
    headers.push(current.trim());
    return headers.filter(h => h.length > 0);
  }

  function populateHeaders(headers) {
    // Clear existing
    if (simiHeader) simiHeader.innerHTML = '';
    if (simiceHeaders) simiceHeaders.innerHTML = '';
    headers.forEach((h, idx) => {
      const val = String(idx + 1); // 1-based
      if (simiHeader) {
        const opt1 = document.createElement('option');
        opt1.value = val;
        opt1.textContent = `${idx + 1}: ${h}`;
        simiHeader.appendChild(opt1);
      }
      if (simiceHeaders) {
        const opt2 = document.createElement('option');
        opt2.value = val;
        opt2.textContent = `${idx + 1}: ${h}`;
        simiceHeaders.appendChild(opt2);
      }
    });
    // Set defaults into inputs once populated
    if (simiHeader && simiIndex && simiHeader.options.length > 0) {
      if (simiHeader.selectedIndex < 0) simiHeader.selectedIndex = 0;
      simiIndex.value = simiHeader.value || '';
    }
    if (simiceHeaders && simiceIndexes) {
      const sel = Array.from(simiceHeaders.selectedOptions).map(o => o.value);
      simiceIndexes.value = sel.join(',');
    }
  }

  if (fileInput) {
    fileInput.addEventListener('change', (e) => {
      const f = e.target.files?.[0];
      if (!f) return;
      const reader = new FileReader();
      reader.onload = () => {
        const headers = parseCSVHeaders(String(reader.result || ''));
        if (headers.length) { populateHeaders(headers); }
      };
      reader.readAsText(f);
    });
  }

  // keep inputs synced when selection changes
  if (simiHeader && simiIndex) {
    const syncSimi = () => { simiIndex.value = simiHeader.value || ''; };
    simiHeader.addEventListener('change', syncSimi);
    simiHeader.addEventListener('input', syncSimi);
    simiHeader.addEventListener('click', syncSimi);
  }
  if (simiceHeaders && simiceIndexes) {
    const syncSimice = () => {
      const vals = Array.from(simiceHeaders.selectedOptions).map(o => o.value);
      simiceIndexes.value = vals.join(',');
    };
    simiceHeaders.addEventListener('change', syncSimice);
    simiceHeaders.addEventListener('input', syncSimice);
    simiceHeaders.addEventListener('click', syncSimice);
  }

  const syncSimiceIndexes = () => {
    const vals = Array.from(simiceHeaders.selectedOptions).map(o => o.value);
    simiceIndexes.value = vals.join(',');
  };

  const openBinaryModal = () => {
    // Build checkboxes for each selected header
    binaryList.innerHTML = '';
    const selected = Array.from(simiceHeaders.selectedOptions);
    selected.forEach(opt => {
      const wrap = document.createElement('div');
      wrap.className = 'binary-item';
      const id = `bin-${opt.value}`;
      wrap.innerHTML = `
        <label>
          <input type="checkbox" id="${id}">
          <span>${opt.textContent}</span>
        </label>
      `;
      binaryList.appendChild(wrap);
    });
    // preload existing
    const prev = (isBinaryListInput.value || '').split(',').map(s => s.trim().toLowerCase());
    const boxes = binaryList.querySelectorAll('input[type="checkbox"]');
    if (prev.length === boxes.length) {
      boxes.forEach((b, i) => { b.checked = ['true', '1', 'yes'].includes(prev[i]); });
    }
    binaryModal.style.display = 'flex';
  };

  const closeBinaryModal = () => { binaryModal.style.display = 'none'; };

  const saveBinary = () => {
    const boxes = Array.from(binaryList.querySelectorAll('input[type="checkbox"]'));
    const vals = boxes.map(b => b.checked ? 'true' : 'false');
    isBinaryListInput.value = vals.join(',');
    closeBinaryModal();
  };

  if (simiceHeaders) {
    simiceHeaders.addEventListener('change', () => { syncSimiceIndexes(); });
  }
  if (openBinaryBtn) { openBinaryBtn.addEventListener('click', openBinaryModal); }
  if (binaryClose) { binaryClose.addEventListener('click', closeBinaryModal); }
  if (binaryCancel) { binaryCancel.addEventListener('click', closeBinaryModal); }
  if (binarySave) { binarySave.addEventListener('click', saveBinary); }
});

// This file contains JavaScript code for the remote site with accessibility improvements

document.addEventListener('DOMContentLoaded', function() {
    // Add accessible table sorting functionality
    document.querySelectorAll('table.table th').forEach(th => {
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
    
    // Make form controls accessible with keyboard navigation
    const toggleForms = document.querySelectorAll('.toggle-form');
    toggleForms.forEach(toggle => {
        // Ensure toggle buttons have proper role and keyboard support
        toggle.setAttribute('role', 'tab');
        toggle.setAttribute('tabindex', '0');
        
        toggle.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                this.click();
            }
        });
    });

    // Add ARIA labels to form fields
    document.querySelectorAll('input, select, button').forEach(element => {
        const id = element.id;
        if (id && !element.hasAttribute('aria-label')) {
            const label = document.querySelector(`label[for="${id}"]`);
            if (label) {
                const labelText = label.textContent.trim();
                if (labelText) {
                    element.setAttribute('aria-label', labelText);
                }
            }
        }
    });
});

// Update job form based on selected algorithm
function updateJobForm(jobData) {
    const jobSelect = document.getElementById('job_id');
    const jobId = jobSelect.value;
    const formContainer = document.getElementById('job-form-fields');
    
    if (!formContainer) return;
    
    let formFields = '';
    
    if (!jobId) {
        formContainer.innerHTML = '<p>Please select a job.</p>';
        return;
    }
    
    const job = jobData[jobId];
    
    if (!job) {
        formContainer.innerHTML = '<p class="error">Selected job data not found.</p>';
        return;
    }
    
    // Common fields for all algorithms
    formFields = `
        <input type="hidden" name="algorithm" value="${job.algorithm}">
        <input type="hidden" name="job_id" value="${job.id}">
    `;
    
    if (job.algorithm === 'SIMI') {
        // For SIMI algorithm
        const targetColumnIndex = job.parameters?.target_column_index;
        
        formFields += `
            <div class="variables-info">
                <p class="form-info"><strong>Target variable in this job:</strong></p>
                <ul class="variables-list">
                    <li>Column ${targetColumnIndex} (${job.parameters?.is_binary ? 'Binary' : 'Continuous'})</li>
                </ul>
            </div>
            <p class="form-info">This job uses the SIMI algorithm for imputation of a single variable.</p>
        `;
        
        // Display all parameters in a read-only format
        formFields += `<div class="parameters-box">
            <h3 id="simi-job-params-heading">All Job Parameters:</h3>
            <table class="parameters-table" aria-labelledby="simi-job-params-heading">
            <tbody>`;
        
        // Loop through all parameters
        for (const [key, value] of Object.entries(job.parameters || {})) {
            const paramId = `simi-param-${key.replace(/\s+/g, '-').toLowerCase()}`;
            formFields += `
                <tr>
                    <th scope="row" id="${paramId}">${key}:</th>
                    <td aria-labelledby="${paramId}">${JSON.stringify(value)}</td>
                </tr>
            `;
        }
        
        formFields += `</tbody></table></div>`;
        
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
    }
    
    formContainer.innerHTML = formFields;
}

// Submit job form to the server
function submitJobForm(form) {
    const formData = new FormData(form);
    const submitBtn = form.querySelector('button[type="submit"]');
    const originalText = submitBtn.textContent;
    
    // Disable button and show loading state
    submitBtn.disabled = true;
    submitBtn.textContent = 'Submitting...';
    
    // Create status message area for screen readers
    let statusArea = document.getElementById('form-status');
    if (!statusArea) {
        statusArea = document.createElement('div');
        statusArea.id = 'form-status';
        statusArea.setAttribute('aria-live', 'polite');
        statusArea.setAttribute('class', 'sr-only');
        form.appendChild(statusArea);
    }
    
    statusArea.textContent = 'Submitting form...';
    
    fetch(form.action, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Re-enable button
        submitBtn.disabled = false;
        submitBtn.textContent = originalText;
        
        if (data.success) {
            statusArea.textContent = 'Form submitted successfully!';
            form.reset();
            
            // Show success message
            const messageDiv = document.createElement('div');
            messageDiv.className = 'alert alert-success';
            messageDiv.setAttribute('role', 'alert');
            messageDiv.textContent = data.message || 'Job submitted successfully!';
            form.prepend(messageDiv);
            
            // Remove message after delay
            setTimeout(() => messageDiv.remove(), 5000);
        } else {
            statusArea.textContent = 'Error: ' + (data.message || 'Something went wrong');
            
            // Show error message
            const errorDiv = document.createElement('div');
            errorDiv.className = 'alert alert-error';
            errorDiv.setAttribute('role', 'alert');
            errorDiv.textContent = data.message || 'Failed to submit job.';
            form.prepend(errorDiv);
            
            // Remove message after delay
            setTimeout(() => errorDiv.remove(), 5000);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        statusArea.textContent = 'Error: ' + error.message;
        
        // Re-enable button
        submitBtn.disabled = false;
        submitBtn.textContent = originalText;
        
        // Show error message
        const errorDiv = document.createElement('div');
        errorDiv.className = 'alert alert-error';
        errorDiv.setAttribute('role', 'alert');
        errorDiv.textContent = 'Network error: ' + error.message;
        form.prepend(errorDiv);
        
        // Remove message after delay
        setTimeout(() => errorDiv.remove(), 5000);
    });
}

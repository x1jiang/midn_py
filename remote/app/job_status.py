import asyncio

class JobStatusCallback:
    """A callback handler for job status updates"""
    
    def __init__(self, app, job_id, site_id):
        self.app = app
        self.job_id = job_id
        self.site_id = site_id
        
    async def on_message(self, message):
        """Handle a status message from the job"""
        job_state = self.get_job_state()
        if job_state:
            job_state['messages'].append(message)
            job_state['status'] = message
            
    async def on_complete(self):
        """Handle job completion"""
        job_state = self.get_job_state()
        if job_state:
            # Explicitly ensure completed flag is False to allow reconnection
            job_state['completed'] = False
            job_state['status'] = "Job iteration completed, will reconnect shortly"
            job_state['messages'].append("Job iteration completed, will reconnect shortly")
            
            # Log status update for debugging
            print(f"JobStatusCallback: Job {self.job_id} for site {self.site_id} status updated to reconnecting")
            print(f"JobStatusCallback: Explicitly set completed=False to ensure reconnection")
            
            # Print full job state for debugging
            print(f"JobStatusCallback: Current job state = {job_state}")
            
    async def on_error(self, error):
        """Handle job error"""
        job_state = self.get_job_state()
        if job_state:
            job_state['completed'] = True
            job_state['status'] = f"Error: {error}"
            job_state['messages'].append(f"Error: {error}")
            
    def get_job_state(self):
        """Get the job state object"""
        # Get site-specific jobs dictionary
        site_jobs = getattr(self.app.state, f'running_jobs_{self.site_id}', None)
        
        # If site dictionary doesn't exist, create it
        if site_jobs is None:
            setattr(self.app.state, f'running_jobs_{self.site_id}', {})
            site_jobs = getattr(self.app.state, f'running_jobs_{self.site_id}')
            
        return site_jobs.get(self.job_id)
